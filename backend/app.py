# === Standard Library Imports ===
import json
import logging
import time
import traceback
from datetime import datetime
from typing import Dict, List, Any, Optional, Union, Tuple
import warnings
import io
import os

# === Flask and Web Framework Imports ===
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
from werkzeug.utils import secure_filename
from flask_sqlalchemy import SQLAlchemy

# === Database Models Import ===
from models import db, Dataset, AnalysisResult, UserThreshold

# === Data Analysis and Statistical Imports ===
import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import binomtest, shapiro
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm
from statsmodels.stats.diagnostic import het_breuschpagan
from statsmodels.stats.stattools import durbin_watson
from statsmodels.tsa.stattools import adfuller, kpss, zivot_andrews, pacf
from arch.unitroot import PhillipsPerron

# === Configuration ===
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Database configuration
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///' + os.path.join(os.path.dirname(__file__), 'mmt_database.db')
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

# Initialize extensions
CORS(app)
db.init_app(app)

USER_THRESHOLDS = {
    'pdCriteria': [
        {'metric': 'Gini Coefficient', 'threshold': 0.2, 'description': 'Minimum acceptable Gini value'},
        {'metric': 'KS Statistic', 'threshold': 0.3, 'description': 'Minimum acceptable KS value'},
        {'metric': 'PSI', 'threshold': 0.1, 'description': 'Maximum acceptable PSI value'}
    ],
    'macroThresholds': [
        {'metric': 'R-squared', 'threshold': 0.7, 'description': 'Minimum R-squared value'},
        {'metric': 'P-value', 'threshold': 0.05, 'description': 'Maximum p-value for significance'}
    ]
}

# === Utility Functions ===
def convert_numpy_types(obj: Any) -> Any:
    """Convert NumPy types to native Python types for JSON serialization."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, dict):
        return {k: convert_numpy_types(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(v) for v in obj]
    elif isinstance(obj, tuple):
        return tuple(convert_numpy_types(v) for v in obj)
    return obj

def normalize_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize DataFrame column names for consistency across data sources."""
    df = df.copy()
    column_mappings = {
        # PD-specific columns
        'DEF_FLAG': ['def flag', 'default_flag', 'default flag', 'def_flg', 'default', 'is_default', 'def_flag', 'default_indicator', 'default_status', 'default_event', 'default_binary'],
        'PD_1_YR': ['pd', 'pd_1yr', 'probability_of_default', 'prob_default', 'pd_1_yr', 'pd_one_year', 'probability_of_default_1yr', 'default_probability', 'pd_estimate', 'one_year_pd'],
        # Macro-specific columns
        'Defaultrate': ['default_rate', 'defaultrate', 'default rate', 'dr', 'actual_dr', 'actual_default_rate'],
        'pred_dr': ['predicted_dr', 'pred_dr', 'predicted_default_rate', 'pred_default_rate', 'pd_predicted'],
        'snapshot_ccyymm': ['snapshot', 'date', 'snapshot_ccyymm', 'period', 'time', 'snapshot_date']
    }

    def find_matching_column(col: str, variations: List[str]) -> bool:
        norm_col = col.lower().replace(' ', '_').replace('-', '_')
        for var in variations:
            norm_var = var.lower().replace(' ', '_').replace('-', '_')
            if norm_col == norm_var or norm_var in norm_col or norm_col in norm_var:
                return True
        return False

    column_rename = {col: standard_name for col in df.columns for standard_name, variations in column_mappings.items() if find_matching_column(col, variations)}
    df.rename(columns=column_rename, inplace=True)
    return df

def validate_pd_data(df: pd.DataFrame) -> Tuple[Optional[pd.DataFrame], List[str]]:
    """Validate PD data structure and prepare it for analysis."""
    df = normalize_column_names(df)
    required_columns = {'PD_1_YR': ['pd', 'pd_1yr', 'probability_of_default', 'prob_default', 'pd1yr'], 
                       'DEF_FLAG': ['def flag', 'default_flag', 'default flag', 'def_flg', 'default', 'is_default', 'def']}
    errors = []

    column_mapping = {}
    for standard_col, variations in required_columns.items():
        for col in df.columns:
            col_norm = col.lower().replace(' ', '_').replace('-', '_')
            for var in variations:
                var_norm = var.lower().replace(' ', '_').replace('-', '_')
                if col_norm == var_norm or var_norm in col_norm or col_norm in var_norm:
                    column_mapping[col] = standard_col
                    break
    
    missing_columns = [col for col in required_columns.keys() if col not in column_mapping.values()]
    
    if missing_columns:
        error_details = [f"Missing required column: {col}. Acceptable variations: {', '.join(required_columns[col])}" for col in missing_columns]
        logger.error(f"Validation Errors: {error_details}")
        return None, error_details
    
    df = df.rename(columns=column_mapping)
    
    for col in required_columns.keys():
        if col not in df.columns:
            errors.append(f"Required column {col} not found after column mapping")
            return None, errors

    try:
        df['PD_1_YR'] = pd.to_numeric(df['PD_1_YR'], errors='coerce')
        df['DEF_FLAG'] = pd.to_numeric(df['DEF_FLAG'], errors='coerce').fillna(0).clip(0, 1).astype(int)

        if df['PD_1_YR'].isna().all():
            errors.append("PD column contains no valid numeric values")
        if len(df[df['DEF_FLAG'].isin([0, 1])]) / len(df) < 0.8:
            errors.append("Default flag column does not contain valid binary values")
    except Exception as e:
        errors.append(f"Data type conversion error: {str(e)}")
        logger.error(f"Data conversion error: {str(e)}\n{traceback.format_exc()}")

    return (df, errors) if not errors else (None, errors)

# === Model Monitor Class ===
class ModelMonitor:
    def load_data(self, file_type: str, dataset_id: Optional[int] = None) -> Tuple[Optional[pd.DataFrame], Optional[str]]:
        """Load data from the database based on file type or specific dataset ID."""
        try:
            if dataset_id:
                dataset = Dataset.query.get(dataset_id)
                if not dataset:
                    return None, f"Dataset with ID {dataset_id} not found"
            else:
                dataset = Dataset.query.filter_by(file_type=file_type).order_by(Dataset.upload_date.desc()).first()
                if not dataset:
                    return None, f"No {file_type} data available in database"

            file_content = dataset.file_content
            if file_type == 'pd' and dataset.name.startswith('pdBaseline_'):
                file_type = 'pd_baseline'
            file_io = io.BytesIO(file_content)
            df = pd.read_csv(file_io) if dataset.name.endswith('.csv') else pd.read_excel(file_io, engine='openpyxl')
            df = normalize_column_names(df)
            return df, None
        except Exception as e:
            logger.error(f"Error loading data for {file_type}: {str(e)}")
            return None, f"Error loading data: {str(e)}"

    def analyze_pd_data(self, df: pd.DataFrame, sorting_method: str = 'PD_1_YR', baseline_datasets: Optional[List[Dataset]] = None) -> Dict[str, Any]:
        """Perform comprehensive PD data analysis."""
        try:
            df['PD_1_YR'] = pd.to_numeric(df['PD_1_YR'], errors='coerce').fillna(df['PD_1_YR'].mean())
            df['DEF_FLAG'] = pd.to_numeric(df['DEF_FLAG'], errors='coerce').fillna(0)

            results = {
                'discriminatory_power': {
                    'gini': self.calculate_gini_coefficient(df['DEF_FLAG'].values.astype('float64'), df['PD_1_YR'].values.astype('float64'), sorting_method, df)
                }
            }
            return results
        except Exception as e:
            logger.error(f"Error analyzing PD data: {str(e)}")
            raise Exception(f"Error analyzing PD data: {str(e)}")

    def calculate_gini_coefficient(self, actual_values: np.ndarray, predicted_scores: np.ndarray, sort_column: str = 'PD_1_YR', df: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
        """Calculate Gini coefficient using CAP curve analysis."""
        actual_values = np.asarray(actual_values).flatten()
        predicted_scores = np.asarray(predicted_scores).flatten()

        if sort_column == 'PD_1_YR':
            sorted_indices = np.argsort(predicted_scores)
            sorted_actual = actual_values[sorted_indices]
            total_samples = len(sorted_actual)
            total_bad = np.sum(sorted_actual)
            total_good = total_samples - total_bad

            cumulative_population = np.linspace(0, 1, total_samples)
            cumulative_bad_rate = np.cumsum(sorted_actual) / total_bad
            cumulative_good_rate = np.cumsum(1 - sorted_actual) / total_good
        elif sort_column == 'TTCReportingRating' and df is not None:
            grouped = df.groupby('TTCReportingRating').agg({'DEF_FLAG': ['sum', 'count']})
            grouped.columns = ['bads', 'total']
            grouped['goods'] = grouped['total'] - grouped['bads']
            grouped['rank'] = grouped.index.map(self.map_credit_rating_to_rank)
            grouped = grouped.sort_values('rank', ascending=False)

            total_good = grouped['goods'].sum()
            total_bad = grouped['bads'].sum()
            cumulative_population = np.linspace(0, 1, len(grouped))
            cumulative_bad_rate = np.cumsum(grouped['bads']) / total_bad
            cumulative_good_rate = np.cumsum(grouped['goods']) / total_good

        random_model_x = cumulative_population
        random_model_y = cumulative_population
        cap_area = np.trapz(cumulative_bad_rate, cumulative_good_rate)
        random_area = np.trapz(random_model_y, random_model_x)
        gini_coefficient = 2 * cap_area - 1

        return {
            'gini_coefficient': float(gini_coefficient),
            'cap_curve': {
                'x': cumulative_population.tolist(),
                'y': cumulative_bad_rate.tolist(),
                'random_model_x': random_model_x.tolist(),
                'random_model_y': random_model_y.tolist()
            }
        }

    def calculate_ks_test(self, actual_values: np.ndarray, predicted_scores: np.ndarray, sort_column: str = 'PD_1_YR', df: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
        """Calculate Kolmogorov-Smirnov (KS) statistic."""
        actual_values = np.asarray(actual_values, dtype=float).flatten()
        predicted_scores = np.asarray(predicted_scores, dtype=float).flatten()

        if sort_column == 'PD_1_YR':
            sorted_indices = np.argsort(predicted_scores)
            sorted_actual = actual_values[sorted_indices]
            total_good = np.sum(sorted_actual == 0)
            total_bad = np.sum(sorted_actual == 1)
            good_cdf = np.cumsum(sorted_actual == 0) / total_good
            bad_cdf = np.cumsum(sorted_actual == 1) / total_bad
        elif sort_column == 'TTCReportingRating' and df is not None:
            grouped = df.groupby('TTCReportingRating').agg({'DEF_FLAG': ['sum', 'count']})
            grouped.columns = ['bads', 'total']
            grouped['goods'] = grouped['total'] - grouped['bads']
            grouped['rank'] = grouped.index.map(self.map_credit_rating_to_rank)
            grouped = grouped.sort_values('rank', ascending=False)
            total_good = grouped['goods'].sum()
            total_bad = grouped['bads'].sum()
            good_cdf = np.cumsum(grouped['goods']) / total_good
            bad_cdf = np.cumsum(grouped['bads']) / total_bad

        ks_curve_x = np.linspace(0, 1, len(good_cdf))
        ks_statistic = np.max(np.abs(good_cdf - bad_cdf))
        ks_point_index = np.argmax(np.abs(good_cdf - bad_cdf))
        ks_point = {'x': ks_curve_x[ks_point_index], 'y_good': good_cdf[ks_point_index], 'y_bad': bad_cdf[ks_point_index]}

        return {
            'ks_statistic': float(ks_statistic),
            'ks_curve': {'x': ks_curve_x.tolist(), 'good_cdf': good_cdf.tolist(), 'bad_cdf': bad_cdf.tolist(), 'ks_point': ks_point}
        }

    def calculate_information_value(self, feature_values: np.ndarray, target_values: np.ndarray, bins: int = 10, categorical: bool = False) -> Dict[str, Any]:
        """Calculate Information Value (IV) for a feature."""
        if len(feature_values) != len(target_values):
            raise ValueError("Feature values and target values must have the same length")

        if categorical:
            unique_values = np.unique(feature_values)
            binned_feature = pd.Series(feature_values).astype('category').cat.codes
        else:
            if not np.issubdtype(feature_values.dtype, np.number):
                raise ValueError("Feature values must be numeric for binning")
            target_values = np.asarray(target_values, dtype=float).flatten()
            feature_values = np.asarray(feature_values, dtype=float).flatten()
            binned_feature, bin_edges = pd.qcut(feature_values, q=bins, retbins=True, duplicates='drop')
            binned_feature = binned_feature.codes

        total_good = np.sum(1 - target_values)
        total_bad = np.sum(target_values)
        bin_details = {'bins': [], 'good_distribution': [], 'bad_distribution': [], 'woe': [], 'iv_per_bin': [], 'bin_range': []}
        iv_total = 0
        bin_stats = []

        for bin_num in range(np.min(binned_feature), np.max(binned_feature) + 1):
            bin_mask = (binned_feature == bin_num)
            good_count = np.sum((1 - target_values)[bin_mask]) + 1e-6
            bad_count = np.sum(target_values[bin_mask]) + 1e-6
            good_dist = good_count / total_good
            bad_dist = bad_count / total_bad
            woe = np.log(bad_dist / good_dist) if good_dist > 0 and bad_dist > 0 else 0
            iv_bin = (bad_dist - good_dist) * woe
            iv_total += iv_bin

            if categorical:
                bin_range = feature_values[bin_mask][0]
                bin_label = bin_range
            else:
                bin_label = f'Bin {bin_num}'
                bin_min = bin_edges[bin_num]
                bin_max = bin_edges[bin_num + 1]
                bin_range = f'[{bin_min:.2f}, {bin_max:.2f}]'

            bin_details['bins'].append({'bin_number': bin_num, 'bin_range': bin_range, 'good_count': float(good_count), 'bad_count': float(bad_count), 'woe': float(woe), 'iv_bin': float(iv_bin)})
            bin_details['good_distribution'].append(float(good_dist))
            bin_details['bad_distribution'].append(float(bad_dist))
            bin_details['woe'].append(float(woe))
            bin_details['iv_per_bin'].append(float(iv_bin))
            bin_details['bin_range'].append(bin_range)

            bin_mean = np.mean(feature_values[bin_mask]) if not categorical and len(feature_values[bin_mask]) > 0 else None
            bin_stats.append({'bin_number': bin_num, 'bin_mean': round(bin_mean, 4) if bin_mean else None, 'bin_label': bin_label, 'total_count': float(good_count + bad_count), 'default_count': float(bad_count), 'non_default_count': float(good_count), 'woe': float(woe)})

        woe_plot_data = [{'x': bin_details['bins'][i]['bin_range'] if categorical else np.mean(feature_values[binned_feature == i]) if len(feature_values[binned_feature == i]) > 0 else bin_edges[i], 'y': round(bin_details['woe'][i], 4), 'label': str(bin_details['bins'][i]['bin_range'] if categorical else np.mean(feature_values[binned_feature == i]) if len(feature_values[binned_feature == i]) > 0 else bin_edges[i])} for i in range(len(bin_details['bins']))]

        return {
            'iv_total': float(abs(iv_total)),
            'iv_by_bin': bin_details['iv_per_bin'],
            'details': {'bins': bin_stats, 'woe': bin_details['woe'], 'bin_details': bin_details},
            'woe_plot_data': woe_plot_data,
            'is_categorical': categorical,
            'unique_values': feature_values if categorical else None
        }
    def calculate_information_value(self, feature_values: np.ndarray, target_values: np.ndarray, bins: int = 10, categorical: bool = False) -> Dict[str, Any]:
        """Calculate Information Value (IV) for a feature."""
        if len(feature_values) != len(target_values):
            raise ValueError("Feature values and target values must have the same length")

        if categorical:
            unique_values = np.unique(feature_values)
            binned_feature = pd.Series(feature_values).astype('category').cat.codes
        else:
            if not np.issubdtype(feature_values.dtype, np.number):
                raise ValueError("Feature values must be numeric for binning")
            target_values = np.asarray(target_values, dtype=float).flatten()
            feature_values = np.asarray(feature_values, dtype=float).flatten()
            binned_feature, bin_edges = pd.qcut(feature_values, q=bins, retbins=True, duplicates='drop')
            binned_feature = binned_feature.codes

        total_good = np.sum(1 - target_values)
        total_bad = np.sum(target_values)
        bin_details = {'bins': [], 'good_distribution': [], 'bad_distribution': [], 'woe': [], 'iv_per_bin': [], 'bin_range': []}
        iv_total = 0
        bin_stats = []

        for bin_num in range(np.min(binned_feature), np.max(binned_feature) + 1):
            bin_mask = (binned_feature == bin_num)
            good_count = np.sum((1 - target_values)[bin_mask]) + 1e-6
            bad_count = np.sum(target_values[bin_mask]) + 1e-6
            good_dist = good_count / total_good
            bad_dist = bad_count / total_bad
            woe = np.log(bad_dist / good_dist) if good_dist > 0 and bad_dist > 0 else 0
            iv_bin = (bad_dist - good_dist) * woe
            iv_total += iv_bin

            if categorical:
                bin_range = feature_values[bin_mask][0] if len(feature_values[bin_mask]) > 0 else 'N/A'
                bin_label = bin_range
            else:
                bin_label = f'Bin {bin_num}'
                bin_min = bin_edges[bin_num]
                bin_max = bin_edges[bin_num + 1]
                bin_range = f'[{bin_min:.2f}, {bin_max:.2f}]'

            bin_details['bins'].append({'bin_number': bin_num, 'bin_range': bin_range, 'good_count': float(good_count), 'bad_count': float(bad_count), 'woe': float(woe), 'iv_bin': float(iv_bin)})
            bin_details['good_distribution'].append(float(good_dist))
            bin_details['bad_distribution'].append(float(bad_dist))
            bin_details['woe'].append(float(woe))
            bin_details['iv_per_bin'].append(float(iv_bin))
            bin_details['bin_range'].append(bin_range)

            bin_mean = np.mean(feature_values[bin_mask]) if not categorical and len(feature_values[bin_mask]) > 0 else None
            bin_stats.append({'bin_number': bin_num, 'bin_mean': round(bin_mean, 4) if bin_mean else None, 'bin_label': bin_label, 'total_count': float(good_count + bad_count), 'default_count': float(bad_count), 'non_default_count': float(good_count), 'woe': float(woe)})

        woe_plot_data = [{'x': bin_details['bins'][i]['bin_range'] if categorical else np.mean(feature_values[binned_feature == i]) if len(feature_values[binned_feature == i]) > 0 else bin_edges[i], 'y': round(bin_details['woe'][i], 4), 'label': str(bin_details['bins'][i]['bin_range'] if categorical else np.mean(feature_values[binned_feature == i]) if len(feature_values[binned_feature == i]) > 0 else bin_edges[i])} for i in range(len(bin_details['bins']))]

        return {
            'iv_total': float(abs(iv_total)),
            'iv_by_bin': bin_details['iv_per_bin'],
            'details': {'bins': bin_stats, 'woe': bin_details['woe'], 'bin_details': bin_details},
            'woe_plot_data': woe_plot_data,
            'is_categorical': categorical,
            'unique_values': list(feature_values) if categorical else None
        }

    @staticmethod
    def test_stationarity(data: pd.Series, tests: List[str] = ['adf', 'kpss', 'pp']) -> Dict[str, Dict[str, Any]]:
        """Perform comprehensive stationarity testing with multiple methods."""
        results = {}
        data = pd.Series(data).dropna()

        if 'adf' in tests:
            try:
                adf_result = adfuller(data, autolag='AIC')
                results['adf'] = {
                    'test_statistic': adf_result[0], 'p_value': adf_result[1], 'used_lag': adf_result[2], 'number_of_observations': adf_result[3], 'critical_values': adf_result[4],
                    'interpretation': 'Reject H0 (stationary)' if adf_result[1] <= 0.05 else 'Fail to reject H0 (non-stationary)'
                }
            except Exception as e:
                logger.error(f"ADF Test failed: {e}")
                results['adf'] = {'error': str(e)}

        if 'kpss' in tests:
            try:
                kpss_result = kpss(data, regression='c')
                results['kpss'] = {
                    'test_statistic': kpss_result[0], 'p_value': kpss_result[1], 'lags': kpss_result[2], 'critical_values': kpss_result[3],
                    'interpretation': 'Reject H0 (non-stationary)' if kpss_result[1] <= 0.05 else 'Fail to reject H0 (stationary)'
                }
            except Exception as e:
                logger.error(f"KPSS Test failed: {e}")
                results['kpss'] = {'error': str(e)}

        if 'pp' in tests:
            try:
                pp_result = PhillipsPerron(data)
                results['phillips_perron'] = {
                    'test_statistic': pp_result.stat, 'p_value': pp_result.pvalue, 'critical_values': pp_result.critical_values,
                    'interpretation': 'Reject H0 (stationary)' if pp_result.pvalue <= 0.05 else 'Fail to reject H0 (non-stationary)'
                }
            except Exception as e:
                logger.error(f"Phillips-Perron Test failed: {e}")
                results['phillips_perron'] = {'error': str(e)}

        return results

    @staticmethod
    def rating_transition_analysis(prev_ratings: pd.Series, curr_ratings: pd.Series) -> Dict[str, Any]:
        """Analyze rating transitions between two periods."""
        transition_data = pd.DataFrame({'Previous_Rating': prev_ratings, 'Current_Rating': curr_ratings}).dropna()
        transition_matrix = pd.crosstab(transition_data['Previous_Rating'], transition_data['Current_Rating'], normalize='index')
        metrics = {
            'unique_ratings': len(set(prev_ratings) | set(curr_ratings)),
            'total_transitions': len(transition_data),
            'stability_rate': np.mean(prev_ratings == curr_ratings),
            'transition_matrix': transition_matrix.to_dict()
        }
        significant_migrations = transition_matrix[(transition_matrix.index != transition_matrix.columns) & (transition_matrix > 0.1)]
        metrics['significant_migrations'] = significant_migrations.to_dict()
        return metrics

    @staticmethod
    def macro_model_analysis(macro_data: pd.DataFrame) -> Dict[str, Any]:
        """Perform enhanced macro model analysis."""
        results = {'stationarity_tests': {}, 'correlation_matrix': {}, 'trend_analysis': {}}
        for column in macro_data.select_dtypes(include=[np.number]).columns:
            results['stationarity_tests'][column] = ModelMonitor.test_stationarity(macro_data[column])
        results['correlation_matrix'] = macro_data.corr().to_dict()
        results['trend_analysis'] = {column: {'mean': macro_data[column].mean(), 'std': macro_data[column].std(), 'min': macro_data[column].min(), 'max': macro_data[column].max(), 'trend_slope': np.polyfit(range(len(macro_data)), macro_data[column], 1)[0]} for column in macro_data.select_dtypes(include=[np.number]).columns}
        return results

    @staticmethod
    def calculate_rmse(actual: np.ndarray, predicted: np.ndarray) -> float:
        """Calculate Root Mean Square Error."""
        return float(np.sqrt(mean_squared_error(actual, predicted)))

    @staticmethod
    def calculate_adjusted_r2(actual: np.ndarray, predicted: np.ndarray, n_predictors: int) -> float:
        """Calculate Adjusted R-squared."""
        r2 = r2_score(actual, predicted)
        n = len(actual)
        return float(1 - (1 - r2) * (n - 1) / (n - n_predictors - 1))

    @staticmethod
    def calculate_mape(actual: np.ndarray, predicted: np.ndarray) -> float:
        """Calculate Mean Absolute Percentage Error."""
        return float(np.mean(np.abs((actual - predicted) / actual)) * 100)

    @staticmethod
    def calculate_decile_crosstab(actual: np.ndarray, predicted: np.ndarray) -> Dict[str, Any]:
        """Calculate decile-based crosstab analysis."""
        df = pd.DataFrame({'actual': actual, 'predicted': predicted})
        df['actual_decile'] = pd.qcut(df['actual'], q=10, labels=False)
        df['predicted_decile'] = pd.qcut(df['predicted'], q=10, labels=False)
        crosstab = pd.crosstab(df['actual_decile'], df['predicted_decile'], normalize='all') * 100
        decile_data = [{'actual_decile': f'D{i+1}', 'values': [round(crosstab.iloc[i, j], 2) for j in range(10)]} for i in range(10)]
        return {'data': decile_data, 'total_count': len(df)}

    @staticmethod
    def calculate_recovery_rate(exposure: np.ndarray, recovery: np.ndarray) -> np.ndarray:
        """Calculate Recovery Rate."""
        return np.where(exposure > 0, recovery / exposure, np.nan)

    @staticmethod
    def calculate_lgd(exposure: np.ndarray, recovery: np.ndarray) -> np.ndarray:
        """Calculate Loss Given Default."""
        return np.where(exposure > 0, 1 - (recovery / exposure), np.nan)

    def perform_hosmer_lemeshow_test(self, actual_values: np.ndarray, predicted_probs: np.ndarray, n_bins: int = 10) -> Dict[str, Any]:
        """Perform Hosmer-Lemeshow test for calibration."""
        actual_values = np.asarray(actual_values, dtype=float).flatten()
        predicted_probs = np.asarray(predicted_probs, dtype=float).flatten()

        if len(actual_values) != len(predicted_probs):
            raise ValueError("Actual values and predicted probabilities must have the same length")

        sorted_indices = np.argsort(predicted_probs)
        actual_values = actual_values[sorted_indices]
        predicted_probs = predicted_probs[sorted_indices]

        bin_edges = np.linspace(0, len(actual_values), n_bins + 1).astype(int)
        bin_details = {'total': [], 'observed': [], 'expected': [], 'edges': []}

        for i in range(n_bins):
            start, end = bin_edges[i], bin_edges[i + 1]
            bin_actual = actual_values[start:end]
            bin_predicted = predicted_probs[start:end]
            bin_details['total'].append(int(len(bin_actual)))
            bin_details['observed'].append(int(np.sum(bin_actual)))
            bin_details['expected'].append(float(np.mean(bin_predicted) * len(bin_actual)))
            bin_details['edges'].append(float(np.mean(bin_predicted)))

        observed = np.array(bin_details['observed'], dtype=float)
        expected = np.array(bin_details['expected'], dtype=float)
        expected = np.where(expected == 0, 1e-10, expected)
        chi_square = float(np.sum((observed - expected) ** 2 / expected))
        df = n_bins - 2
        p_value = float(1 - stats.chi2.cdf(chi_square, df))

        return {
            'chi_square': chi_square, 'p_value': p_value, 'degrees_of_freedom': df, 'bin_details': bin_details,
            'interpretation': 'Good calibration' if p_value > 0.05 else 'Poor calibration'
        }

    def map_credit_rating_to_rank(self, rating: str) -> int:
        """Map credit ratings to numeric ranks for consistent sorting."""
        rating_rank_map = {
            'A1': 1, 'A2': 2, 'A3': 3, 'Aa2': 4, 'Aa3': 5, 'B1': 6, 'B2': 7, 'B3': 8, 'Ba1': 9, 'Ba2': 10,
            'Ba3': 11, 'Baa1': 12, 'Baa2': 13, 'Baa3': 14, 'Caa1': 15, 'Caa2': 16
        }
        return rating_rank_map.get(rating, 99)

    def perform_binomial_test_by_rating(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """Perform binomial test for each credit rating category."""
        if 'TTCReportingRating' not in df.columns or 'DEF_FLAG' not in df.columns:
            logger.error("Missing required columns for binomial test")
            return []

        rating_results = []
        ratings = df['TTCReportingRating'].unique()

        for rating in ratings:
            rating_data = df[df['TTCReportingRating'] == rating]
            observed_defaults = rating_data['DEF_FLAG'].sum()
            total_samples = len(rating_data)

            if total_samples == 0:
                continue

            rating_pd = rating_data['PD_1_YR'].mean() if 'PD_1_YR' in rating_data.columns else rating_data['DEF_FLAG'].mean()

            try:
                p_value = binomtest(k=int(observed_defaults), n=total_samples, p=rating_pd).pvalue
                test_result = 'PASS' if p_value > 0.05 else 'FAIL'
                rating_results.append({
                    'rating': rating, 'observed_defaults': int(observed_defaults), 'total_samples': int(total_samples),
                    'expected_default_prob': round(rating_pd, 4), 'p_value': round(p_value, 4), 'test_result': test_result
                })
            except Exception as e:
                logger.error(f"Error performing binomial test for rating {rating}: {str(e)}")

        rating_results.sort(key=lambda x: x['rating'])
        return rating_results

    def calculate_psi(self, baseline: pd.DataFrame, current: pd.DataFrame, column: str = 'DEF_FLAG') -> Dict[str, Any]:
        """Calculate Population Stability Index (PSI) by binning DEF_FLAG based on TTCReportingRating."""
        if 'TTCReportingRating' not in baseline.columns or 'TTCReportingRating' not in current.columns:
            return {'error': 'TTCReportingRating column is missing', 'psi_total': None}

        baseline_counts = baseline.groupby('TTCReportingRating')[column].count()
        current_counts = current.groupby('TTCReportingRating')[column].count()

        all_ratings = sorted(set(baseline_counts.index) | set(current_counts.index))
        baseline_counts = baseline_counts.reindex(all_ratings, fill_value=0)
        current_counts = current_counts.reindex(all_ratings, fill_value=0)

        psi_values = []
        bin_details = {}

        for rating in all_ratings:
            baseline_count = baseline_counts[rating]
            current_count = current_counts[rating]
            
            if baseline_count > 0 and current_count > 0:
                baseline_prop = baseline_count / baseline_counts.sum()
                current_prop = current_count / current_counts.sum()
                psi_value = (current_prop - baseline_prop) * np.log(current_prop / baseline_prop)
                psi_values.append(psi_value)
            else:
                psi_value = 0

            bin_details[rating] = {
                'baseline_count': int(baseline_count),
                'current_count': int(current_count),
                'baseline_prop': float(baseline_count / baseline_counts.sum()),
                'current_prop': float(current_count / current_counts.sum()),
                'psi': float(psi_value)
            }

        total_psi = float(np.sum(psi_values))
        return {'psi_total': total_psi, 'bin_details': bin_details}

    def calculate_csi(self, baseline: pd.DataFrame, current: pd.DataFrame, column: str = 'DEF_FLAG') -> Dict[str, Any]:
        """Calculate Characteristic Stability Index (CSI) for numeric and categorical variables."""
        exclude_columns = ['DEF_FLAG', 'PD_1_YR']
        numeric_columns = [col for col in baseline.select_dtypes(include=['int64', 'float64']).columns if col not in exclude_columns]
        categorical_columns = [col for col in baseline.select_dtypes(include=['object', 'category']).columns if col not in exclude_columns]
        csi_results = {}

        for num_col in numeric_columns:
            baseline_bins_labels, baseline_bins_edges = pd.qcut(baseline[num_col], q=10, labels=False, retbins=True, duplicates='drop')
            current_bins_labels, current_bins_edges = pd.qcut(current[num_col], q=10, labels=False, retbins=True, duplicates='drop')
            csi_values = []
            bin_details = {}

            unique_bin_indices = sorted(baseline_bins_labels.unique())
            
            for i, bin_idx in enumerate(unique_bin_indices):
                if bin_idx >= len(current_bins_edges) - 1:
                    continue
                    
                baseline_subset = baseline[baseline_bins_labels == bin_idx]
                current_subset = current[current_bins_labels == bin_idx]
                total_baseline_count = len(baseline[column])
                total_current_count = len(current[column])
                baseline_proportion = len(baseline_subset[column]) / total_baseline_count + 1e-10
                current_proportion = len(current_subset[column]) / total_current_count + 1e-10
                
                bin_min = current_bins_edges[bin_idx]
                bin_max = current_bins_edges[bin_idx + 1]
                bin_range = f'[{bin_min:.2f}, {bin_max:.2f}]'
                
                if total_baseline_count > 0 and total_current_count > 0:
                    csi_value = (current_proportion - baseline_proportion) * np.log(current_proportion / baseline_proportion)
                    csi_values.append(csi_value)
                    
                bin_details[str(i)] = {
                    'baseline_count': len(baseline_subset[column]), 
                    'current_count': len(current_subset[column]), 
                    'baseline_proportion': baseline_proportion, 
                    'current_proportion': current_proportion,
                    'bin_range': bin_range
                }

            csi_results[num_col] = {'csi_total': float(np.sum(csi_values)), 'bin_details': bin_details}

        for cat_col in categorical_columns:
            unique_values = sorted(set(baseline[cat_col].unique()) | set(current[cat_col].unique()))
            csi_values = []
            bin_details = {}

            for value in unique_values:
                baseline_subset = baseline[baseline[cat_col] == value]
                current_subset = current[current[cat_col] == value]
                total_baseline_count = len(baseline[column])
                total_current_count = len(current[column])
                baseline_proportion = len(baseline_subset[column]) / total_baseline_count + 1e-10
                current_proportion = len(current_subset[column]) / total_current_count + 1e-10
                if total_baseline_count > 0 and total_current_count > 0:
                    csi_value = (current_proportion - baseline_proportion) * np.log(current_proportion / baseline_proportion)
                    csi_values.append(csi_value)
                bin_details[str(value)] = {'baseline_count': len(baseline_subset[column]), 'current_count': len(current_subset[column]), 'baseline_proportion': baseline_proportion, 'current_proportion': current_proportion}

            csi_results[cat_col] = {'csi_total': float(np.sum(csi_values)), 'bin_details': bin_details}

        return csi_results

    def analyze_ead(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze EAD data and return results."""
        df['EAD'] = pd.to_numeric(df['EAD'], errors='coerce').fillna(0)
        return {}  # Placeholder for actual EAD analysis logic

# === API Endpoints ===
# --- PD Model Endpoints ---
@app.route('/api/upload/pd', methods=['POST'])
def upload_pd_file() -> Tuple[Dict[str, Any], int]:
    """Upload PD data file and store in database."""
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded', 'details': 'No file found in the request'}), 400

        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No selected file', 'details': 'Filename cannot be empty'}), 400

        if not file.filename.lower().endswith(('.csv', '.xlsx')):
            return jsonify({'error': 'Invalid file type', 'details': 'Only CSV and XLSX files are allowed'}), 400

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        file_extension = os.path.splitext(file.filename)[1].lower()
        filename = f"pd_{timestamp}{file_extension}"
        file_content = file.read()

        df = pd.read_csv(io.BytesIO(file_content)) if file_extension == '.csv' else pd.read_excel(io.BytesIO(file_content), engine='openpyxl')
        df, validation_errors = validate_pd_data(df)

        if validation_errors:
            return jsonify({'error': 'Invalid PD data', 'details': validation_errors}), 400

        df = normalize_column_names(df)
        new_dataset = Dataset(
            name=filename,
            description=request.form.get('description', ''),
            file_content=file_content,
            file_type='pd',
            file_size=len(file_content),
            is_baseline=request.form.get('is_baseline', 'false').lower() == 'true',
            column_names=json.dumps(df.columns.tolist()),
            row_count=len(df)
        )
        
        db.session.add(new_dataset)
        db.session.commit()

        return jsonify({
            'message': 'PD file uploaded successfully', 
            'filename': filename, 
            'dataset_id': new_dataset.id,
            'details': {'rows': len(df), 'columns': list(df.columns)}
        }), 200
    except Exception as e:
        logger.error(f"PD file upload error: {str(e)}\n{traceback.format_exc()}")
        return jsonify({'error': 'Upload failed', 'details': str(e)}), 500

@app.route('/api/upload/pd_baseline', methods=['POST'])
def upload_pd_baseline_file() -> Tuple[Dict[str, Any], int]:
    """Upload PD baseline data file for stability analysis and store in database."""
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400

        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No selected file'}), 400

        filename = secure_filename(file.filename)
        unique_filename = f'pdBaseline_{int(time.time())}_{filename}'
        file_content = file.read()

        df = pd.read_csv(io.BytesIO(file_content)) if filename.endswith('.csv') else pd.read_excel(io.BytesIO(file_content), engine='openpyxl')
        df = normalize_column_names(df)
        required_columns = ['PD_1_YR', 'DEF_FLAG']
        missing_columns = [col for col in required_columns if col not in df.columns]

        if missing_columns:
            return jsonify({'error': f'Missing required columns: {", ".join(missing_columns)}'}), 400
            
        new_dataset = Dataset(
            name=unique_filename,
            description=request.form.get('description', 'PD Baseline Dataset'),
            file_content=file_content,
            file_type='pd',
            file_size=len(file_content),
            is_baseline=True,
            column_names=json.dumps(df.columns.tolist()),
            row_count=len(df)
        )
        
        db.session.add(new_dataset)
        db.session.commit()

        return jsonify({
            'message': 'PD Baseline data uploaded successfully', 
            'filename': unique_filename, 
            'dataset_id': new_dataset.id,
            'rows': len(df), 
            'columns': list(df.columns)
        }), 200
    except Exception as e:
        logger.error(f'Error uploading PD baseline file: {str(e)}')
        return jsonify({'error': f'Upload failed: {str(e)}'}), 500

@app.route('/api/analyze/stability', methods=['POST'])
def calculate_stability_metrics() -> Tuple[Dict[str, Any], int]:
    """Calculate PSI and CSI stability metrics."""
    try:
        monitor = ModelMonitor()
        baseline_df, baseline_error = monitor.load_data('pd_baseline')
        current_df, current_error = monitor.load_data('pd')

        if baseline_error or current_error:
            return jsonify({'error': f'Data loading error: {baseline_error or current_error}'}), 400

        required_columns = ['PD_1_YR', 'DEF_FLAG']
        for col in required_columns:
            if col not in baseline_df.columns or col not in current_df.columns:
                return jsonify({'error': f'Missing required column: {col}'}), 400

        psi_results = monitor.calculate_psi(baseline_df, current_df)
        return jsonify({
            'psi': psi_results,
            'interpretation': {'psi': {'low_risk': 'PSI < 0.1', 'medium_risk': '0.1 <= PSI < 0.2', 'high_risk': 'PSI >= 0.2'}}
        }), 200
    except Exception as e:
        logger.error(f'Error calculating stability metrics: {str(e)}')
        return jsonify({'error': f'Stability analysis failed: {str(e)}'}), 500

@app.route('/api/analyze/pd', methods=['POST'])
def analyze_pd() -> Tuple[Dict[str, Any], int]:
    """Analyze PD data with comprehensive metrics and store results in database if not already present."""
    try:
        sorting_method = request.json.get('sortingMethod', 'PD_1_YR') if request.json else 'PD_1_YR'
        dataset_id = request.json.get('datasetId') if request.json else None
        
        if sorting_method not in ['PD_1_YR', 'TTCReportingRating']:
            sorting_method = 'PD_1_YR'

        monitor = ModelMonitor()
        if dataset_id:
            df, error = monitor.load_data('pd', dataset_id)
            if error:
                return jsonify({'error': 'Dataset not found', 'details': error}), 404
            dataset = Dataset.query.get(dataset_id)
        else:
            df, error = monitor.load_data('pd')
            if error:
                return jsonify({'error': 'No PD data available', 'details': 'Please upload PD data first'}), 400
            dataset = Dataset.query.filter_by(file_type='pd').order_by(Dataset.upload_date.desc()).first()

        required_columns = ['PD_1_YR', 'DEF_FLAG']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            return jsonify({'error': 'Invalid data structure', 'details': f'Missing columns: {", ".join(missing_columns)}'}), 400

        df['PD_1_YR'] = pd.to_numeric(df['PD_1_YR'], errors='coerce').fillna(df['PD_1_YR'].mean())
        df['DEF_FLAG'] = pd.to_numeric(df['DEF_FLAG'], errors='coerce').fillna(0)

        gini_result = monitor.calculate_gini_coefficient(df['DEF_FLAG'].values.astype('float64'), df['PD_1_YR'].values.astype('float64'), sorting_method, df)
        ks_result = monitor.calculate_ks_test(df['DEF_FLAG'].values.astype('float64'), df['PD_1_YR'].values.astype('float64'), sorting_method, df)

        results = {
            'discriminatory_power': {'gini': gini_result, 'ks_test': ks_result},
            'gini': gini_result, 'gini_coefficient': gini_result['gini_coefficient'], 'cap_curve': gini_result['cap_curve'],
            'ks_test': ks_result, 'ks_statistic': ks_result['ks_statistic'], 'ks_curve': ks_result['ks_curve'],
            'stability': {'psi': None},
            'calibration': {
                'binomial_test_by_rating': monitor.perform_binomial_test_by_rating(df),
                'hosmer_lemeshow': monitor.perform_hosmer_lemeshow_test(df['DEF_FLAG'].values.astype('float64'), df['PD_1_YR'].values.astype('float64'))
            },
            'variable_assessment': {'categorical_variables': {}, 'numeric_variables': {}}
        }

        baseline_datasets = Dataset.query.filter_by(file_type='pd', is_baseline=True).all()
        if baseline_datasets:
            latest_baseline = max(baseline_datasets, key=lambda x: x.upload_date)
            baseline_df, _ = monitor.load_data('pd', latest_baseline.id)
            if 'PD_1_YR' in baseline_df.columns:
                results['stability']['psi'] = monitor.calculate_psi(baseline_df, df)

        exclude_cols = ['PD_1_YR', 'DEF_FLAG']
        feature_columns = [col for col in df.columns if col not in exclude_cols]
        numeric_columns = [col for col in feature_columns if pd.to_numeric(df[col], errors='coerce').isna().mean() < 0.8]
        categorical_columns = [col for col in feature_columns if col not in numeric_columns]

        for var in numeric_columns:
            try:
                feature_values = pd.to_numeric(df[var], errors='coerce').fillna(0)
                iv_result = monitor.calculate_information_value(feature_values.values.astype('float64'), df['DEF_FLAG'].values.astype('float64'), bins=10)
                results['variable_assessment']['numeric_variables'][var] = {'iv': {'iv_total': float(iv_result['iv_total']), 'details': iv_result['details'], 'woe_plot_data': iv_result['woe_plot_data']}}
                if baseline_datasets:
                    csi_result = monitor.calculate_csi(baseline_df, df, 'DEF_FLAG')
                    if var in csi_result:
                        results['variable_assessment']['numeric_variables'][var]['csi_total'] = csi_result[var]['csi_total']
                        results['variable_assessment']['numeric_variables'][var]['csi_bin_details'] = csi_result[var]['bin_details']
            except Exception as e:
                logger.error(f"Could not process numeric variable {var}: {str(e)}")

        for var in categorical_columns:
            try:
                cat_iv_result = monitor.calculate_information_value(df[var].values, df['DEF_FLAG'].values.astype('float64'), categorical=True)
                results['variable_assessment']['categorical_variables'][var] = {'iv': {'iv_total': float(cat_iv_result['iv_total']), 'details': cat_iv_result['details'], 'woe_plot_data': cat_iv_result['woe_plot_data']}}
                if baseline_datasets:
                    csi_result = monitor.calculate_csi(baseline_df, df, 'DEF_FLAG')
                    if var in csi_result:
                        results['variable_assessment']['categorical_variables'][var]['csi_total'] = csi_result[var]['csi_total']
            except Exception as e:
                logger.error(f"Could not process categorical variable {var}: {str(e)}")
        
        if dataset:
            # Check for existing analysis result
            existing_result = AnalysisResult.query.filter_by(dataset_id=dataset.id, analysis_type='pd').first()
            if existing_result:
                logger.info(f"Analysis result for dataset_id {dataset.id} and type 'pd' already exists. Skipping save.")
                results['analysis_result_id'] = existing_result.id
            else:
                results_json = json.dumps(convert_numpy_types(results))
                parameters_json = json.dumps({
                    'sorting_method': sorting_method,
                    'baseline_dataset_ids': [d.id for d in baseline_datasets]
                })
                
                new_result = AnalysisResult(
                    dataset_id=dataset.id,
                    analysis_type='pd',
                    result_data=results_json,
                    parameters=parameters_json
                )
                db.session.add(new_result)
                db.session.commit()
                results['analysis_result_id'] = new_result.id

        return jsonify(convert_numpy_types(results)), 200
    except Exception as e:
        logger.error(f"PD Analysis Error: {str(e)}\n{traceback.format_exc()}")
        return jsonify({'error': 'PD Analysis Failed', 'details': str(e), 'traceback': traceback.format_exc()}), 500

# --- LGD Model Endpoints ---
@app.route('/api/upload/lgd', methods=['POST'])
def upload_lgd_file() -> Tuple[Dict[str, Any], int]:
    """Upload LGD data file and store in database."""
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded', 'details': 'Please upload a valid CSV or Excel file'}), 400

        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected', 'details': 'Please choose a file to upload'}), 400

        filename = f"lgd_{int(time.time())}_{secure_filename(file.filename)}"
        file_content = file.read()

        df = pd.read_csv(io.BytesIO(file_content)) if filename.endswith('.csv') else pd.read_excel(io.BytesIO(file_content))
        column_mapping = {
            'actual_lgd': ['actual_lgd', 'actual lgd', 'actual', 'lgd_actual', 'lgd actual'],
            'predicted_lgd': ['predicted_lgd', 'predicted lgd', 'predicted', 'lgd_predicted', 'lgd predicted'],
            'Quarter': ['quarter', 'Quarter', 'Qtr', 'qtr'],
            'Portfolio': ['portfolio', 'Portfolio', 'port', 'Port'],
            'ModelName': ['model_name', 'modelname', 'model', 'Model', 'ModelName']
        }

        matched_columns = {required_col: next((col for col in possible_cols if col in df.columns), None) for required_col, possible_cols in column_mapping.items()}
        missing_columns = [col for col, match in matched_columns.items() if match is None]

        if missing_columns:
            return jsonify({'error': 'Invalid file format', 'details': f'Missing required columns: {", ".join(missing_columns)}', 'column_hints': column_mapping}), 400

        rename_dict = {matched_columns[col]: col for col in matched_columns}
        df = df.rename(columns=rename_dict)

        new_dataset = Dataset(
            name=filename,
            description=request.form.get('description', ''),
            file_content=file_content,
            file_type='lgd',
            file_size=len(file_content),
            is_baseline=False,
            column_names=json.dumps(df.columns.tolist()),
            row_count=len(df)
        )
        
        db.session.add(new_dataset)
        db.session.commit()

        logger.info(f'LGD file uploaded successfully: {filename}')
        return jsonify({
            'message': 'LGD file uploaded successfully', 
            'filename': filename, 
            'dataset_id': new_dataset.id,
            'data_preview': df.head(10).to_dict(orient='records'), 
            'column_mapping': rename_dict
        }), 200
    except Exception as e:
        logger.error(f"LGD file upload error: {str(e)}\n{traceback.format_exc()}")
        return jsonify({'error': 'Unexpected error during file upload', 'details': str(e)}), 500

@app.route('/api/lgd/options', methods=['GET'])
def get_lgd_options() -> Tuple[Dict[str, Any], int]:
    """Retrieve LGD filter options from the latest dataset."""
    try:
        # Fetch all LGD datasets, ordered by upload date
        datasets = Dataset.query.filter_by(file_type='lgd').order_by(Dataset.upload_date.desc()).all()
        if not datasets:
            logger.info("No LGD datasets found in the database.")
            return jsonify({
                'quarters': [],
                'portfolios': [],
                'modelNames': [],
                'message': 'No LGD datasets available. Please upload an LGD dataset.'
            }), 200

        # Use the latest dataset
        latest_dataset = datasets[0]
        logger.info(f"Fetching LGD options from dataset: {latest_dataset.name}, dataset_id: {latest_dataset.id}")

        # Load the dataset from file_content
        df = pd.read_csv(io.BytesIO(latest_dataset.file_content)) if latest_dataset.name.endswith('.csv') else pd.read_excel(io.BytesIO(latest_dataset.file_content))

        # Normalize column names
        df = normalize_column_names(df)

        # Define expected columns with fallback handling
        column_mapping = {
            'Quarter': ['quarter', 'Quarter', 'Qtr', 'qtr'],
            'Portfolio': ['portfolio', 'Portfolio', 'port', 'Port'],
            'ModelName': ['model_name', 'modelname', 'model', 'Model', 'ModelName']
        }

        # Find actual column names in the dataset
        quarter_col = next((col for col in column_mapping['Quarter'] if col in df.columns), None)
        portfolio_col = next((col for col in column_mapping['Portfolio'] if col in df.columns), None)
        model_name_col = next((col for col in column_mapping['ModelName'] if col in df.columns), None)

        # Check if required columns are present
        missing_cols = [key for key, val in {'Quarter': quarter_col, 'Portfolio': portfolio_col, 'ModelName': model_name_col}.items() if val is None]
        if missing_cols:
            logger.warning(f"Missing expected columns in LGD dataset {latest_dataset.name}: {', '.join(missing_cols)}")
            return jsonify({
                'quarters': [],
                'portfolios': [],
                'modelNames': [],
                'message': f"Missing columns in dataset {latest_dataset.name}: {', '.join(missing_cols)}. Expected variations: {json.dumps(column_mapping)}"
            }), 200

        # Extract unique values for dropdown options
        quarters = sorted(df[quarter_col].dropna().unique().tolist())
        portfolios = sorted(df[portfolio_col].dropna().unique().tolist())
        model_names = sorted(df[model_name_col].dropna().unique().tolist())

        logger.info(f"LGD options retrieved: {len(quarters)} quarters, {len(portfolios)} portfolios, {len(model_names)} model names")
        return jsonify({
            'quarters': quarters,
            'portfolios': portfolios,
            'modelNames': model_names
        }), 200
    except Exception as e:
        logger.error(f"Error getting LGD options: {str(e)}\n{traceback.format_exc()}")
        return jsonify({
            'error': 'Failed to get LGD options',
            'details': str(e),
            'traceback': traceback.format_exc()
        }), 500

@app.route('/api/analyze/lgd', methods=['POST'])
def analyze_lgd() -> Tuple[Dict[str, Any], int]:
    """Analyze LGD data with specified filters and store results in database if not already present."""
    try:
        quarter = request.json.get('quarter')
        portfolio = request.json.get('portfolio')
        model_name = request.json.get('modelName')
        dataset_id = request.json.get('datasetId')

        monitor = ModelMonitor()
        if dataset_id:
            lgd_df, error = monitor.load_data('lgd', dataset_id)
            if error:
                return jsonify({'error': 'Dataset not found', 'details': error}), 404
            dataset = Dataset.query.get(dataset_id)
        else:
            lgd_df, error = monitor.load_data('lgd')
            if error:
                return jsonify({'error': 'No LGD data uploaded. Please upload LGD data first.'}), 400
            dataset = Dataset.query.filter_by(file_type='lgd').order_by(Dataset.upload_date.desc()).first()

        column_mapping = {
            'quarter': ['quarter', 'Quarter', 'Qtr', 'qtr'],
            'portfolio': ['portfolio', 'Portfolio', 'port', 'Port'],
            'model_name': ['model_name', 'modelname', 'model', 'Model', 'ModelName'],
            'actual_lgd': ['actual_lgd', 'actual lgd', 'actual', 'lgd_actual', 'lgd actual'],
            'predicted_lgd': ['predicted_lgd', 'predicted lgd', 'predicted', 'lgd_predicted', 'lgd predicted']
        }

        quarter_col = next((col for col in column_mapping['quarter'] if col in lgd_df.columns), None)
        portfolio_col = next((col for col in column_mapping['portfolio'] if col in lgd_df.columns), None)
        model_name_col = next((col for col in column_mapping['model_name'] if col in lgd_df.columns), None)
        actual_lgd_col = next((col for col in column_mapping['actual_lgd'] if col in lgd_df.columns), None)
        predicted_lgd_col = next((col for col in column_mapping['predicted_lgd'] if col in lgd_df.columns), None)

        if not all([quarter_col, portfolio_col, model_name_col, actual_lgd_col, predicted_lgd_col]):
            missing_cols = [k for k, v in {'Quarter': quarter_col, 'Portfolio': portfolio_col, 'ModelName': model_name_col, 'Actual LGD': actual_lgd_col, 'Predicted LGD': predicted_lgd_col}.items() if not v]
            return jsonify({'error': 'Missing required columns', 'details': f'Columns missing: {", ".join(missing_cols)}'}), 400

        filtered_df = lgd_df.copy()
        if quarter:
            filtered_df = filtered_df[filtered_df[quarter_col] == quarter]
        if portfolio:
            filtered_df = filtered_df[filtered_df[portfolio_col] == portfolio]
        if model_name:
            model_names = [name.strip() for name in model_name.split(',')] if isinstance(model_name, str) else model_name
            filtered_df = filtered_df[filtered_df[model_name_col].isin(model_names)]

        if len(filtered_df) == 0:
            return jsonify({'error': 'No data found matching the specified filters'}), 400

        metrics = {
            'MAPE': monitor.calculate_mape(filtered_df[actual_lgd_col].values, filtered_df[predicted_lgd_col].values),
            'R-squared': r2_score(filtered_df[actual_lgd_col].values, filtered_df[predicted_lgd_col].values),
            'MAE': mean_absolute_error(filtered_df[actual_lgd_col].values, filtered_df[predicted_lgd_col].values),
            'Bias': np.mean(filtered_df[predicted_lgd_col].values - filtered_df[actual_lgd_col].values)
        }
        plot_data = {'actual_lgd': filtered_df[actual_lgd_col].tolist(), 'predicted_lgd': filtered_df[predicted_lgd_col].tolist()}
        decile_data = monitor.calculate_decile_crosstab(filtered_df[actual_lgd_col].values, filtered_df[predicted_lgd_col].values)
        additional_data = filtered_df[[quarter_col, portfolio_col, model_name_col]].drop_duplicates().to_dict('records')

        results = {
            'metrics': metrics, 
            'plot_data': plot_data, 
            'decile_data': decile_data, 
            'additional_data': additional_data
        }
        
        if dataset:
            # Check for existing analysis result
            existing_result = AnalysisResult.query.filter_by(dataset_id=dataset.id, analysis_type='lgd').first()
            if existing_result:
                logger.info(f"Analysis result for dataset_id {dataset.id} and type 'lgd' already exists. Skipping save.")
                results['analysis_result_id'] = existing_result.id
            else:
                results_json = json.dumps(convert_numpy_types(results))
                parameters_json = json.dumps({
                    'quarter': quarter,
                    'portfolio': portfolio,
                    'model_name': model_name
                })
                
                new_result = AnalysisResult(
                    dataset_id=dataset.id,
                    analysis_type='lgd',
                    result_data=results_json,
                    parameters=parameters_json
                )
                db.session.add(new_result)
                db.session.commit()
                results['analysis_result_id'] = new_result.id

        return jsonify(convert_numpy_types(results)), 200
    except Exception as e:
        logger.error(f"Error in LGD analysis: {str(e)}")
        return jsonify({'error': f'LGD Analysis Failed: {str(e)}', 'details': traceback.format_exc()}), 500

# --- EAD Model Endpoints ---
@app.route('/api/upload/ead', methods=['POST'])
def upload_ead_file() -> Tuple[Dict[str, Any], int]:
    """Upload EAD data file and store in database."""
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded', 'details': 'Please upload a valid CSV or Excel file'}), 400

        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected', 'details': 'Please choose a file to upload'}), 400

        filename = f"ead_{int(time.time())}_{secure_filename(file.filename)}"
        file_content = file.read()

        df = pd.read_csv(io.BytesIO(file_content)) if filename.endswith('.csv') else pd.read_excel(io.BytesIO(file_content))
        required_columns = ['Exposure', 'Quarter', 'Portfolio', 'ModelName']
        missing_columns = [col for col in required_columns if col not in df.columns]

        if missing_columns:
            return jsonify({'error': 'Invalid file format', 'details': f'Missing required columns: {", ".join(missing_columns)}'}), 400

        new_dataset = Dataset(
            name=filename,
            description=request.form.get('description', ''),
            file_content=file_content,
            file_type='ead',
            file_size=len(file_content),
            is_baseline=False,
            column_names=json.dumps(df.columns.tolist()),
            row_count=len(df)
        )
        
        db.session.add(new_dataset)
        db.session.commit()

        logger.info(f'EAD file uploaded successfully: {filename}')
        return jsonify({
            'message': 'EAD file uploaded successfully', 
            'filename': filename, 
            'dataset_id': new_dataset.id,
            'data_preview': df.head(10).to_dict(orient='records')
        }), 200
    except Exception as e:
        logger.error(f"EAD file upload error: {str(e)}\n{traceback.format_exc()}")
        return jsonify({'error': 'Unexpected error during file upload', 'details': str(e)}), 500

@app.route('/api/analyze/ead', methods=['POST'])
def analyze_ead() -> Tuple[Dict[str, Any], int]:
    """Analyze EAD data with specified filters and store results in database if not already present."""
    try:
        quarter = request.json.get('quarter')
        portfolio = request.json.get('portfolio')
        model_name = request.json.get('modelName')
        dataset_id = request.json.get('datasetId')

        monitor = ModelMonitor()
        if dataset_id:
            df, error = monitor.load_data('ead', dataset_id)
            if error:
                return jsonify({'error': 'Dataset not found', 'details': error}), 404
            dataset = Dataset.query.get(dataset_id)
        else:
            df, error = monitor.load_data('ead')
            if error:
                return jsonify({'error': 'No EAD data file found. Please upload a file first.'}), 400
            dataset = Dataset.query.filter_by(file_type='ead').order_by(Dataset.upload_date.desc()).first()

        original_df = df.copy()
        if quarter:
            df = df[df['Quarter'] == quarter]
        if portfolio:
            df = df[df['Portfolio'] == portfolio]
        if model_name:
            model_names = [name.strip() for name in model_name.split(',')] if isinstance(model_name, str) else model_name
            df = df[df['ModelName'].isin(model_names)]

        if df.empty:
            results = {
                'metrics': {'MAPE': 0, 'R-squared': 0}, 
                'plot_data': {'actual_ead': [], 'predicted_ead': []}, 
                'additional_data': original_df[['Quarter', 'Portfolio', 'ModelName']].drop_duplicates().to_dict('records')
            }
            return jsonify(results), 200

        actual_ead = df['Exposure'].values
        predicted_ead = df.get('predicted_exposure', actual_ead).values
        metrics = {'MAPE': monitor.calculate_mape(actual_ead, predicted_ead), 'R-squared': r2_score(actual_ead, predicted_ead)}
        plot_data = {'actual_ead': actual_ead.tolist(), 'predicted_ead': predicted_ead.tolist()}
        decile_data = monitor.calculate_decile_crosstab(actual_ead, predicted_ead)

        results = {
            'metrics': metrics, 
            'plot_data': plot_data, 
            'decile_data': decile_data, 
            'additional_data': original_df[['Quarter', 'Portfolio', 'ModelName']].drop_duplicates().to_dict('records')
        }
        
        if dataset:
            # Check for existing analysis result
            existing_result = AnalysisResult.query.filter_by(dataset_id=dataset.id, analysis_type='ead').first()
            if existing_result:
                logger.info(f"Analysis result for dataset_id {dataset.id} and type 'ead' already exists. Skipping save.")
                results['analysis_result_id'] = existing_result.id
            else:
                parameters = {
                    'quarter': quarter,
                    'portfolio': portfolio,
                    'model_name': model_name
                }
                parameters_json = json.dumps(parameters)
                results_json = json.dumps(convert_numpy_types(results))
                
                new_result = AnalysisResult(
                    dataset_id=dataset.id,
                    analysis_type='ead',
                    result_data=results_json,
                    parameters=parameters_json
                )
                db.session.add(new_result)
                db.session.commit()
                results['analysis_result_id'] = new_result.id

        return jsonify(convert_numpy_types(results)), 200
    except Exception as e:
        logger.error(f"Error in EAD analysis: {str(e)}")
        return jsonify({'error': f'EAD Analysis Failed: {str(e)}', 'details': traceback.format_exc()}), 500

# --- Summary and Threshold Endpoints ---
@app.route('/api/summary', methods=['GET'])
def get_summary() -> Tuple[Dict[str, Any], int]:
    """Retrieve summary data."""
    try:
        summary_data = {
            'metrics': {'PD Gini': 0.82, 'LGD R²': 0.78, 'EAD MAPE': 12.5, 'Macro R²': 0.85},
            'timeSeries': [{'date': '2023 Q1', 'value': 0.82}, {'date': '2023 Q2', 'value': 0.83}, {'date': '2023 Q3', 'value': 0.81}, {'date': '2023 Q4', 'value': 0.84}]
        }
        return jsonify(convert_numpy_types(summary_data)), 200
    except Exception as e:
        logger.error(f"Error in summary: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/user-thresholds', methods=['GET'])
def get_user_thresholds() -> Tuple[Dict[str, Any], int]:
    """Retrieve user-defined thresholds."""
    try:
        return jsonify(convert_numpy_types({'pdCriteria': USER_THRESHOLDS['pdCriteria'], 'macroThresholds': USER_THRESHOLDS['macroThresholds']})), 200
    except Exception as e:
        logger.error(f"Error retrieving thresholds: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/user-thresholds', methods=['POST'])
def save_user_thresholds() -> Tuple[Dict[str, Any], int]:
    """Save user-defined thresholds."""
    try:
        data = request.get_json()
        if 'pdCriteria' in data:
            USER_THRESHOLDS['pdCriteria'] = data['pdCriteria']
        if 'macroThresholds' in data:
            USER_THRESHOLDS['macroThresholds'] = data['macroThresholds']
        return jsonify(convert_numpy_types({'message': 'Thresholds saved successfully', 'pdCriteria': USER_THRESHOLDS['pdCriteria'], 'macroThresholds': USER_THRESHOLDS['macroThresholds']})), 200
    except Exception as e:
        logger.error(f"Error saving thresholds: {str(e)}")
        return jsonify({'error': str(e)}), 500

# --- Rating Transition Endpoint ---
@app.route('/api/analyze/rating_transition', methods=['POST'])
def analyze_rating_transition() -> Tuple[Dict[str, Any], int]:
    """Analyze rating transitions between two periods."""
    try:
        if not request.is_json:
            logger.error('Rating transition analysis - Invalid request format')
            return jsonify({'error': 'Invalid request', 'details': 'Request must be in JSON format'}), 400

        data = request.get_json()
        prev_ratings = data.get('previous_ratings')
        curr_ratings = data.get('current_ratings')

        if not prev_ratings or not curr_ratings:
            logger.error('Rating transition analysis - Missing ratings data')
            return jsonify({'error': 'Invalid input', 'details': 'Both previous and current ratings are required'}), 400

        try:
            prev_ratings = pd.Series([str(r).strip() for r in prev_ratings])
            curr_ratings = pd.Series([str(r).strip() for r in curr_ratings])
        except Exception as e:
            logger.error(f'Rating conversion error: {e}')
            return jsonify({'error': 'Data conversion failed', 'details': 'Unable to convert ratings to valid format'}), 400

        if len(prev_ratings) != len(curr_ratings):
            logger.error('Rating transition analysis - Mismatched ratings length')
            return jsonify({'error': 'Length mismatch', 'details': 'Previous and current ratings must have equal length'}), 400

        valid_mask = ~(prev_ratings.isna() | curr_ratings.isna() | (prev_ratings == '') | (curr_ratings == ''))
        prev_ratings = prev_ratings[valid_mask]
        curr_ratings = curr_ratings[valid_mask]

        if len(prev_ratings) == 0:
            logger.error('Rating transition analysis - No valid ratings')
            return jsonify({'error': 'No valid ratings', 'details': 'All provided ratings are invalid or empty'}), 400

        results = ModelMonitor.rating_transition_analysis(prev_ratings, curr_ratings)
        logger.info(f'Rating Transition Analysis: Total Transitions: {len(prev_ratings)}, Previous Unique Ratings: {set(prev_ratings)}, Current Unique Ratings: {set(curr_ratings)}')
        return jsonify(convert_numpy_types(results)), 200
    except Exception as e:
        logger.error(f"Unexpected error in rating transition analysis: {str(e)}")
        return jsonify({'error': 'Unexpected error', 'details': str(e)}), 500

# --- Macro Model Endpoints ---
@app.route('/api/upload/macro', methods=['POST'])
def upload_macro_file() -> Tuple[Dict[str, Any], int]:
    """Upload macro data file and store in database."""
    try:
        if 'file' not in request.files:
            logger.error("No file part in request")
            return jsonify({'error': 'No file uploaded', 'details': 'Please upload a valid CSV or Excel file'}), 400

        file = request.files['file']
        if file.filename == '':
            logger.error("No file selected")
            return jsonify({'error': 'No file selected', 'details': 'Please choose a file to upload'}), 400

        filename = f'macro_{int(time.time())}_{secure_filename(file.filename)}'
        file_content = file.read()

        if not file_content:
            logger.error(f"Empty file content for {filename}")
            return jsonify({'error': 'Empty file', 'details': 'Uploaded file is empty'}), 400

        if filename.lower().endswith('.csv'):
            try:
                macro_data = pd.read_csv(io.BytesIO(file_content))
            except Exception as e:
                logger.error(f"Failed to read CSV: {str(e)}")
                return jsonify({'error': 'Invalid CSV', 'details': str(e)}), 400
        elif filename.lower().endswith(('.xls', '.xlsx')):
            try:
                macro_data = pd.read_excel(io.BytesIO(file_content))
            except Exception as e:
                logger.error(f"Failed to read Excel: {str(e)}")
                return jsonify({'error': 'Invalid Excel', 'details': str(e)}), 400
        else:
            logger.error(f'Unsupported file format: {filename}')
            return jsonify({'error': 'Unsupported file format', 'details': 'Only CSV and Excel files are supported'}), 400

        if macro_data is None or macro_data.empty:
            logger.error("Macro Model Analysis - Input data is None or empty")
            return jsonify({'error': 'Invalid input data', 'details': 'Macro data is None or empty'}), 400

        new_dataset = Dataset(
            name=filename,
            description=request.form.get('description', 'Macro economic data file'),
            file_content=file_content,
            file_type='macro',
            file_size=len(file_content),
            is_baseline=False,
            column_names=json.dumps(macro_data.columns.tolist()),
            row_count=len(macro_data)
        )
        
        db.session.add(new_dataset)
        db.session.commit()

        logger.info(f"Macro file uploaded successfully: {filename}, dataset_id: {new_dataset.id}")
        return jsonify({
            'message': 'File uploaded successfully', 
            'filename': filename, 
            'dataset_id': new_dataset.id
        }), 200
    except Exception as e:
        logger.error(f'Macro file upload error: {str(e)}\n{traceback.format_exc()}')
        return jsonify({'error': 'Unexpected error during file upload', 'details': str(e), 'traceback': traceback.format_exc()}), 500

@app.route('/api/analyze/macro', methods=['POST'])
def analyze_macro_model() -> Tuple[Dict[str, Any], int]:
    """Analyze macro model data with comprehensive statistical tests and store results in database if not already present."""
    try:
        monitor = ModelMonitor()
        dataset_id = request.json.get('datasetId') if request.json else None
        
        if dataset_id:
            df, error = monitor.load_data('macro', dataset_id)
            if error:
                return jsonify({'error': 'Dataset not found', 'details': error}), 404
            dataset = Dataset.query.get(dataset_id)
        else:
            df, error = monitor.load_data('macro')
            if error:
                return jsonify({'error': 'Macro data file not found'}), 404
            dataset = Dataset.query.filter_by(file_type='macro').order_by(Dataset.upload_date.desc()).first()

        df = normalize_column_names(df)
        required_columns = ['Defaultrate', 'pred_dr', 'snapshot_ccyymm']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            error_msg = f"Missing required columns: {', '.join(missing_columns)}. Please ensure your macro data includes columns for actual default rate, predicted default rate, and time period."
            logger.error(error_msg)
            return jsonify({'error': 'Invalid macro data', 'details': error_msg}), 400

        df.sort_values(by='snapshot_ccyymm', ascending=True, inplace=True)
        actual_default_rate = df['Defaultrate']
        predicted_pd = df['pred_dr']
        macro_variables = df.drop(columns=['Defaultrate', 'pred_dr', 'snapshot_ccyymm'])
        model_error = predicted_pd - actual_default_rate

        model_error_AD = stats.anderson(model_error)
        n = len(model_error)
        probs = (np.arange(1, n + 1) - 0.5) / n
        theoretical_quantiles = stats.norm.ppf(probs)
        standardized_errors = (model_error - np.mean(model_error)) / np.std(model_error)
        sample_quantiles = np.sort(standardized_errors)
        min_val = min(min(theoretical_quantiles), min(sample_quantiles))
        max_val = max(max(theoretical_quantiles), max(sample_quantiles))
        line_x = np.array([min_val, max_val])
        line_y = line_x

        normality_results = {
            'model_error': {
                'AD Statistic': float(model_error_AD.statistic), 'Critical Values': model_error_AD.critical_values.tolist(),
                'Significance Level': model_error_AD.significance_level.tolist(),
                'qq_plot': {'theoretical_quantiles': theoretical_quantiles.tolist(), 'sample_quantiles': sample_quantiles.tolist(), 'line': {'x': line_x.tolist(), 'y': line_y.tolist()}}
            }
        }

        model_error_DW = durbin_watson(model_error)
        model_error_BP = het_breuschpagan(model_error, sm.add_constant(actual_default_rate))
        heteroscedasticity_results = {
            'LM Statistic': model_error_BP[0], 'p-value': model_error_BP[1], 'predicted_pd': predicted_pd.tolist(), 'model_error': model_error.tolist()
        }

        rmse = ModelMonitor.calculate_rmse(actual_default_rate, predicted_pd)
        adjusted_r2 = 1 - (1 - (np.corrcoef(actual_default_rate, predicted_pd)[0, 1]) ** 2) * (len(actual_default_rate) - 1) / (len(actual_default_rate) - 2)
        y = np.array(predicted_pd).reshape(-1, 1)
        X = np.array(actual_default_rate).reshape(-1, 1)
        model = LinearRegression().fit(X, y)
        slope = model.coef_[0]
        intercept = model.intercept_
        trend_x = np.array([float(actual_default_rate.min()), float(actual_default_rate.max())])
        trend_y = model.predict(trend_x.reshape(-1, 1))

        comparison_results = {
            'Adjusted R-squared': float(adjusted_r2), 'RMSE': float(rmse), 'actual_default_rate': actual_default_rate.tolist(), 'predicted_pd': predicted_pd.tolist(),
            'trend_line': {'x': trend_x.tolist(), 'y': trend_y.tolist(), 'equation': {'slope': float(slope), 'intercept': float(intercept)}}
        }

        stationarity_results = {}
        for column in macro_variables.columns:
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter('always')
                adf_result = adfuller(macro_variables[column])
                kpss_result = kpss(macro_variables[column], regression='c')
                kpss_p_value = float(kpss_result[1])
                kpss_note = 'Exact p-value'
                if w and 'interpolation' in str(w[-1].message).lower():
                    kpss_note = f'p > {kpss_p_value:.3f}' if 'greater than' in str(w[-1].message) else f'p < {kpss_p_value:.3f}'
                za_result = zivot_andrews(macro_variables[column])
                pp_test = PhillipsPerron(macro_variables[column])

                stationarity_results[column] = {
                    'ADF Statistic': float(adf_result[0]), 'ADF p-value': float(adf_result[1]), 'KPSS Statistic': float(kpss_result[0]), 'KPSS p-value': kpss_p_value, 'KPSS Note': kpss_note,
                    'Zivot-Andrews Statistic': float(za_result[0]), 'ZA p-value': float(za_result[1]), 'Phillips-Perron Statistic': float(pp_test.stat), 'PP p-value': float(pp_test.pvalue)
                }

        def calculate_pacf(residuals: np.ndarray, alpha: float = 0.05) -> Dict[str, Any]:
            pacf_values = pacf(residuals, nlags=20, method='ols')
            n = len(residuals)
            z_score = stats.norm.ppf(1 - alpha / 2)
            confidence_interval = z_score / np.sqrt(n)
            return {
                'lags': list(range(len(pacf_values))), 'pacf_values': list(pacf_values),
                'confidence_interval': {'upper': [confidence_interval] * len(pacf_values), 'lower': [-confidence_interval] * len(pacf_values)}
            }

        def calculate_acf(residuals: np.ndarray, alpha: float = 0.05) -> Dict[str, Any]:
            acf_values, confint = sm.tsa.acf(residuals, nlags=20, fft=False, alpha=alpha, bartlett_confint=True)
            return {
                'lags': list(range(len(acf_values))), 'acf_values': list(acf_values),
                'confidence_interval': {'lower': list(confint[:, 0]), 'upper': list(confint[:, 1])}
            }

        pacf_results = calculate_pacf(model_error)
        acf_results = calculate_acf(model_error)
        autocorrelation_results = {
            'model_error': {
                'DW Statistic': float(model_error_DW),
                'Interpretation': 'Evidence of positive autocorrelation.' if model_error_DW < 1.5 else 'Little to no autocorrelation.' if 1.5 < model_error_DW < 2.5 else 'Evidence of negative autocorrelation.',
                'acf': acf_results, 'pacf': pacf_results
            }
        }

        def generate_normality_visualization(model_errors: np.ndarray) -> Dict[str, Any]:
            hist, bin_edges = np.histogram(model_errors, bins='auto', density=True)
            bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
            mu, std = stats.norm.fit(model_errors)
            x = np.linspace(min(bin_edges), max(bin_edges), 100)
            pdf = stats.norm.pdf(x, mu, std)
            return {
                'histogram': {'x': bin_centers.tolist(), 'y': hist.tolist()},
                'normal_curve': {'x': x.tolist(), 'y': pdf.tolist()},
                'distribution_params': {'mean': float(mu), 'std': float(std)}
            }

        results = {
            'normality_results': normality_results, 'autocorrelation_results': autocorrelation_results, 'heteroscedasticity_results': heteroscedasticity_results,
            'comparison_results': comparison_results, 'stationarity_results': stationarity_results, 'pacf_results': pacf_results, 'acf_results': acf_results,
            'time_series_data': df.to_dict(orient='records')
        }
        results['normality_results']['model_error']['normality_visualization'] = generate_normality_visualization(model_error)
        
        if dataset:
            # Check for existing analysis result
            existing_result = AnalysisResult.query.filter_by(dataset_id=dataset.id, analysis_type='macro').first()
            if existing_result:
                logger.info(f"Analysis result for dataset_id {dataset.id} and type 'macro' already exists. Skipping save.")
                results['analysis_result_id'] = existing_result.id
            else:
                parameters = request.json or {}
                parameters_json = json.dumps(parameters)
                results_json = json.dumps(convert_numpy_types(results))
                
                new_result = AnalysisResult(
                    dataset_id=dataset.id,
                    analysis_type='macro',
                    result_data=results_json,
                    parameters=parameters_json
                )
                db.session.add(new_result)
                db.session.commit()
                results['analysis_result_id'] = new_result.id

        return jsonify(convert_numpy_types(results)), 200
    except Exception as e:
        logger.error(f'Error during macro model analysis: {str(e)}\n{traceback.format_exc()}')
        return jsonify({'error': 'An error occurred during analysis', 'details': str(e), 'traceback': traceback.format_exc()}), 500

# === Database Management Endpoints ===
@app.route('/api/database/datasets', methods=['GET'])
def get_all_datasets():
    """Get all datasets from the database"""
    try:
        datasets = Dataset.query.all()
        return jsonify({
            'success': True,
            'datasets': [dataset.to_dict() for dataset in datasets]
        })
    except Exception as e:
        logger.error(f"Error retrieving datasets: {str(e)}")
        return jsonify({
            'success': False,
            'error': f"Error retrieving datasets: {str(e)}"
        }), 500

@app.route('/api/database/datasets/<int:dataset_id>', methods=['GET'])
def get_dataset(dataset_id):
    """Get a specific dataset by ID"""
    try:
        dataset = Dataset.query.get(dataset_id)
        if not dataset:
            return jsonify({
                'success': False,
                'error': f"Dataset with ID {dataset_id} not found"
            }), 404
        
        return jsonify({
            'success': True,
            'dataset': dataset.to_dict()
        })
    except Exception as e:
        logger.error(f"Error retrieving dataset: {str(e)}")
        return jsonify({
            'success': False,
            'error': f"Error retrieving dataset: {str(e)}"
        }), 500

@app.route('/api/database/datasets/<int:dataset_id>', methods=['DELETE'])
def delete_dataset(dataset_id):
    """Delete a dataset by ID"""
    try:
        dataset = Dataset.query.get(dataset_id)
        if not dataset:
            return jsonify({
                'success': False,
                'error': f"Dataset with ID {dataset_id} not found"
            }), 404
        
        AnalysisResult.query.filter_by(dataset_id=dataset_id).delete()
        db.session.delete(dataset)
        db.session.commit()
        
        return jsonify({
            'success': True,
            'message': f"Dataset with ID {dataset_id} successfully deleted"
        })
    except Exception as e:
        db.session.rollback()
        logger.error(f"Error deleting dataset: {str(e)}")
        return jsonify({
            'success': False,
            'error': f"Error deleting dataset: {str(e)}"
        }), 500

@app.route('/api/database/datasets/<int:dataset_id>', methods=['PUT'])
def update_dataset(dataset_id):
    """Update dataset information"""
    try:
        dataset = Dataset.query.get(dataset_id)
        if not dataset:
            return jsonify({
                'success': False,
                'error': f"Dataset with ID {dataset_id} not found"
            }), 404
        
        data = request.json
        if 'name' in data:
            dataset.name = data['name']
        if 'description' in data:
            dataset.description = data['description']
        if 'is_baseline' in data:
            dataset.is_baseline = data['is_baseline']
        
        db.session.commit()
        
        return jsonify({
            'success': True,
            'dataset': dataset.to_dict()
        })
    except Exception as e:
        db.session.rollback()
        logger.error(f"Error updating dataset: {str(e)}")
        return jsonify({
            'success': False,
            'error': f"Error updating dataset: {str(e)}"
        }), 500

@app.route('/api/database/analysis-results', methods=['GET'])
def get_all_analysis_results():
    """Get all analysis results or filter by dataset_id"""
    try:
        dataset_id = request.args.get('dataset_id', type=int)
        analysis_type = request.args.get('analysis_type')
        
        query = AnalysisResult.query
        if dataset_id:
            query = query.filter_by(dataset_id=dataset_id)
        if analysis_type:
            query = query.filter_by(analysis_type=analysis_type)
        
        results = query.all()
        
        return jsonify({
            'success': True,
            'analysis_results': [result.to_dict() for result in results]
        })
    except Exception as e:
        logger.error(f"Error retrieving analysis results: {str(e)}")
        return jsonify({
            'success': False,
            'error': f"Error retrieving analysis results: {str(e)}"
        }), 500

@app.route('/api/database/analysis-results/<int:result_id>', methods=['GET'])
def get_analysis_result(result_id):
    """Get a specific analysis result by ID"""
    try:
        result = AnalysisResult.query.get(result_id)
        if not result:
            return jsonify({
                'success': False,
                'error': f"Analysis result with ID {result_id} not found"
            }), 404
        
        return jsonify({
            'success': True,
            'analysis_result': result.to_dict()
        })
    except Exception as e:
        logger.error(f"Error retrieving analysis result: {str(e)}")
        return jsonify({
            'success': False,
            'error': f"Error retrieving analysis result: {str(e)}"
        }), 500

@app.route('/api/database/analysis-results/<int:result_id>', methods=['DELETE'])
def delete_analysis_result(result_id):
    """Delete an analysis result by ID"""
    try:
        result = AnalysisResult.query.get(result_id)
        if not result:
            return jsonify({
                'success': False,
                'error': f"Analysis result with ID {result_id} not found"
            }), 404
        
        db.session.delete(result)
        db.session.commit()
        
        return jsonify({
            'success': True,
            'message': f"Analysis result with ID {result_id} successfully deleted"
        })
    except Exception as e:
        db.session.rollback()
        logger.error(f"Error deleting analysis result: {str(e)}")
        return jsonify({
            'success': False,
            'error': f"Error deleting analysis result: {str(e)}"
        }), 500

@app.route('/api/database/download/<int:dataset_id>', methods=['GET'])
def download_dataset_file(dataset_id):
    """Download the file associated with a dataset."""
    try:
        dataset = Dataset.query.get(dataset_id)
        if not dataset:
            logger.error(f"Dataset with ID {dataset_id} not found")
            return jsonify({
                'success': False,
                'error': f"Dataset with ID {dataset_id} not found"
            }), 404
        
        if not dataset.file_content or len(dataset.file_content) == 0:
            logger.error(f"Dataset {dataset_id} ({dataset.name}) has no file content or content is empty")
            return jsonify({
                'success': False,
                'error': f"No file content available for dataset ID {dataset_id}"
            }), 400

        logger.info(f"Preparing download for dataset: {dataset.name}, dataset_id: {dataset_id}, size: {len(dataset.file_content)} bytes")
        mime_type = 'text/csv' if dataset.name.lower().endswith('.csv') else 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
        
        response = send_file(
            io.BytesIO(dataset.file_content),
            as_attachment=True,
            download_name=dataset.name,
            mimetype=mime_type,
            max_age=0
        )
        logger.info(f"Download response prepared for {dataset.name} with Content-Type: {mime_type}")
        return response
    except Exception as e:
        logger.error(f"Error downloading dataset file {dataset_id}: {str(e)}\n{traceback.format_exc()}")
        return jsonify({
            'success': False,
            'error': f"Error downloading dataset file: {str(e)}",
            'traceback': traceback.format_exc()
        }), 500

# Create database tables if they don't exist
with app.app_context():
    db.create_all()

# === Main Execution ===
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)