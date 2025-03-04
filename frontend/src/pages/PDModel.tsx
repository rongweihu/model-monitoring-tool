import React, { useState, useEffect } from 'react';
import {
  Paper, Typography, Box, Tabs, Tab, Grid, Table, TableBody, TableCell, TableContainer, TableHead, TableRow,
  CircularProgress, Select, MenuItem, SelectChangeEvent, Checkbox, Button, Menu, Divider, Chip, Fade,
} from '@mui/material';
import {
  ResponsiveContainer, LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip as RechartsTooltip, Scatter, Legend,
} from 'recharts';
import InfoIcon from '@mui/icons-material/Info';
import { Tooltip as MUITooltip } from '@mui/material';
import { useTheme } from '@mui/material/styles';
import { api } from '../utils/api';

// === Interfaces ===
interface TabPanelProps {
  children?: React.ReactNode;
  value: number;
  index: number;
}

interface PerformanceData {
  x: number;
  y?: number;
  good_cdf?: number;
  bad_cdf?: number;
  y_good?: number;
  y_bad?: number;
}

interface HosmerLemeshowResult {
  statistic: number;
  p_value: number;
  degrees_of_freedom: number;
  interpretation: string;
}

interface PDAnalysisData {
  gini_coefficient: number;
  ks_statistic: number;
  discriminatory_power: {
    gini: { gini_coefficient: number; cap_curve?: { x: number[]; y: number[]; random_model_x: number[]; random_model_y: number[] } };
    ks_test: { ks_statistic: number; ks_curve: { x: number[]; good_cdf: number[]; bad_cdf: number[]; ks_point: { x: number; y_good: number; y_bad: number } } };
  };
  calibration: { binomial_test_by_rating: any[]; hosmer_lemeshow: HosmerLemeshowResult };
  stability: { psi: { psi_total: number; bin_details: { [rating: string]: { baseline_prop?: number; current_prop?: number; psi: number } } } | null };
  variable_assessment?: { categorical_variables: { [key: string]: any }; numeric_variables: { [key: string]: any } };
}

interface WOEPlotData {
  x?: number | string;
  y?: number;
  label?: string;
  bin_number?: number;
  bin_range?: string;
  woe?: number;
  total_count?: number;
  default_count?: number;
}

// === Utility Components ===
const TabPanel: React.FC<TabPanelProps> = ({ children, value, index }) => (
  <div role="tabpanel" hidden={value !== index} id={`tabpanel-${index}`} aria-labelledby={`tab-${index}`} style={{ minHeight: '600px' }}>
    {value === index && <Box sx={{ p: 3, height: '100%', overflow: 'auto' }}>{children}</Box>}
  </div>
);

const MetricBox: React.FC<{ value: number; threshold: number; label: string }> = ({ value, threshold, label }) => {
  const isPassing = value >= threshold;
  return (
    <Grid item xs={12} md={6}>
      <Paper sx={{ p: 2, backgroundColor: isPassing ? 'success.light' : 'error.light', color: 'white', textAlign: 'center' }}>
        <Typography variant="h5" sx={{ fontWeight: 'bold' }}>
          {label}
          <Typography component="span" variant="body1" sx={{ display: 'block', mt: 1, fontWeight: 'normal' }}>
            {value.toFixed(4)} (Threshold: {threshold.toFixed(4)})
          </Typography>
          <Typography component="span" variant="body1" sx={{ display: 'block', mt: 0, fontWeight: 'bold' }}>
            {isPassing ? 'PASS' : 'FAIL'}
          </Typography>
        </Typography>
      </Paper>
    </Grid>
  );
};

// === Main Component ===
const PDModel: React.FC = () => {
  const theme = useTheme();
  const isDarkMode = theme.palette.mode === 'dark';

  // State Management
  const [tabValue, setTabValue] = useState(0);
  const [variableSubTab, setVariableSubTab] = useState(0);
  const [performanceData, setPerformanceData] = useState<PerformanceData[]>([]);
  const [randomModelData, setRandomModelData] = useState<PerformanceData[]>([]);
  const [ksCurveData, setKsCurveData] = useState<PerformanceData[]>([]);
  const [ksPoint, setKsPoint] = useState<PerformanceData>({ x: 0, y_good: 0, y_bad: 0 });
  const [pdAnalysisResults, setPDAnalysisResults] = useState<PDAnalysisData | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [giniThreshold, setGiniThreshold] = useState(0.3);
  const [ksThreshold, setKsThreshold] = useState(0.45);
  const [sortingMethod, setSortingMethod] = useState<'PD_1_YR' | 'TTCReportingRating'>('PD_1_YR');
  const [ratingBinomialTests, setRatingBinomialTests] = useState<any[]>([]);
  const [numericFilter, setNumericFilter] = useState<string[]>([]);
  const [categoricalFilter, setCategoricalFilter] = useState<string[]>([]);
  const [numericMenuAnchor, setNumericMenuAnchor] = useState<null | HTMLElement>(null);
  const [categoricalMenuAnchor, setCategoricalMenuAnchor] = useState<null | HTMLElement>(null);

  // Fetch Thresholds
  const fetchPDThresholds = () => {
    try {
      const pdCriteriaStr = localStorage.getItem('pdCriteria');
      if (pdCriteriaStr) {
        const pdCriteria = JSON.parse(pdCriteriaStr);
        const gini = pdCriteria.find((c: any) => c.metric === 'Gini Coefficient');
        const ks = pdCriteria.find((c: any) => c.metric === 'KS Statistic');
        if (gini) setGiniThreshold(gini.threshold);
        if (ks) setKsThreshold(ks.threshold);
      }
    } catch (err) {
      console.error('Error fetching PD thresholds:', err);
    }
  };

  // Fetch PD Analysis Data
  const fetchPDAnalysis = async () => {
    setIsLoading(true);
    setError(null);
    try {
      const response = (await api.analyzePD({ sortingMethod })) as PDAnalysisData;
      if (!response) throw new Error('No PD analysis results received');

      setPDAnalysisResults(response);
      setPerformanceData(response.discriminatory_power.gini.cap_curve?.x.map((x, i) => ({ x, y: response.discriminatory_power.gini.cap_curve?.y[i] || 0 })) || []);
      setRandomModelData(response.discriminatory_power.gini.cap_curve?.random_model_x.map((x, i) => ({ x, y: response.discriminatory_power.gini.cap_curve?.random_model_y[i] || 0 })) || []);
      setKsCurveData(response.discriminatory_power.ks_test.ks_curve.x.map((x, i) => ({ x, good_cdf: response.discriminatory_power.ks_test.ks_curve.good_cdf[i], bad_cdf: response.discriminatory_power.ks_test.ks_curve.bad_cdf[i] })));
      setKsPoint(response.discriminatory_power.ks_test.ks_curve.ks_point);
      setRatingBinomialTests(response.calibration?.binomial_test_by_rating || []);

      if (response.variable_assessment) {
        setNumericFilter(Object.keys(response.variable_assessment.numeric_variables || {}));
        setCategoricalFilter(Object.keys(response.variable_assessment.categorical_variables || {}));
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Unknown error');
      setPDAnalysisResults(null);
    } finally {
      setIsLoading(false);
    }
  };

  // Effects
  useEffect(() => {
    fetchPDThresholds();
    const handleStorageChange = (event: StorageEvent) => event.key === 'pdCriteria' && fetchPDThresholds();
    window.addEventListener('storage', handleStorageChange);
    return () => window.removeEventListener('storage', handleStorageChange);
  }, []);

  useEffect(() => {
    fetchPDAnalysis();
  }, [sortingMethod]);

  // Handlers
  const handleTabChange = (_: React.SyntheticEvent, newValue: number) => setTabValue(newValue);
  const handleSubTabChange = (_: React.SyntheticEvent, newValue: number) => setVariableSubTab(newValue);
  const handleSortingChange = (event: SelectChangeEvent<'PD_1_YR' | 'TTCReportingRating'>) => setSortingMethod(event.target.value as 'PD_1_YR' | 'TTCReportingRating');

  // Sub-Components
  const DiscriminatoryPowerTab: React.FC = () => {
    if (!pdAnalysisResults) return <Typography color="error">No data available</Typography>;

    const { discriminatory_power } = pdAnalysisResults;
    const giniCoefficient = discriminatory_power?.gini?.gini_coefficient ?? 0; // Fallback to 0 if undefined
    const ksStatistic = discriminatory_power?.ks_test?.ks_statistic ?? 0; // Fallback to 0 if undefined

    return (
      <Box>
        <Box sx={{ display: 'flex', alignItems: 'center', mb: 2 }}>
          <Typography variant="h6" sx={{ mr: 2 }}>Binning by:</Typography>
          <Select value={sortingMethod} onChange={handleSortingChange} sx={{ minWidth: 200 }}>
            <MenuItem value="PD_1_YR">PD 1 Year</MenuItem>
            <MenuItem value="TTCReportingRating">Credit Rating</MenuItem>
          </Select>
        </Box>
        <Grid container spacing={2}>
          <MetricBox value={giniCoefficient} threshold={giniThreshold} label="Gini Coefficient" />
          <MetricBox value={ksStatistic} threshold={ksThreshold} label="KS Statistic" />
          <Grid item xs={12} md={6}>
            <Typography variant="h6">Gini Coefficient</Typography>
            <ResponsiveContainer width="100%" height={300}>
              <LineChart data={performanceData} margin={{ top: 5, right: 30, left: 20, bottom: 5 }}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis type="number" dataKey="x" label={{ value: 'Proportion of Goods', position: 'insideBottom', offset: -5 }} />
                <YAxis label={{ value: 'Proportion of Bads', angle: -90, position: 'insideLeft' }} />
                <Legend />
                <Line type="monotone" dataKey="y" stroke="#8884d8" dot={false} name="Model Performance" />
                <Line data={randomModelData} type="monotone" dataKey="y" stroke="#82ca9d" strokeDasharray="5 5" dot={false} name="Random Model" />
                <RechartsTooltip contentStyle={{ backgroundColor: isDarkMode ? theme.palette.background.paper : 'white', color: isDarkMode ? theme.palette.text.primary : 'black' }} />
              </LineChart>
            </ResponsiveContainer>
            <Typography variant="body2">Gini Coefficient: {giniCoefficient.toFixed(4)}</Typography>
          </Grid>
          <Grid item xs={12} md={6}>
            <Typography variant="h6">KS Test</Typography>
            <ResponsiveContainer width="100%" height={300}>
              <LineChart data={ksCurveData} margin={{ top: 5, right: 30, left: 20, bottom: 5 }}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis type="number" dataKey="x" label={{ value: 'Proportion of Population', position: 'insideBottom', offset: -5 }} />
                <YAxis label={{ value: 'Cumulative Distribution', angle: -90, position: 'insideLeft' }} />
                <Legend verticalAlign="top" height={36} align="right" />
                <Line type="monotone" dataKey="good_cdf" stroke="#8884d8" name="Goods CDF" dot={false} />
                <Line type="monotone" dataKey="bad_cdf" stroke="#82ca9d" name="Bads CDF" dot={false} />
                <Scatter data={[ksPoint]} fill="red" shape="cross" strokeWidth={3} name="KS Point" />
                <RechartsTooltip contentStyle={{ backgroundColor: isDarkMode ? theme.palette.background.paper : 'white', color: isDarkMode ? theme.palette.text.primary : 'black' }} />
              </LineChart>
            </ResponsiveContainer>
            <Typography variant="body2">KS Statistic: {ksStatistic.toFixed(4)}</Typography>
          </Grid>
        </Grid>
      </Box>
    );
  };

  const StabilityTab: React.FC = () => {
    if (!pdAnalysisResults?.stability?.psi) return <Typography>No Stability Index (PSI) available</Typography>;

    const { psi_total, bin_details } = pdAnalysisResults.stability.psi;
    return (
      <Paper sx={{ p: 2 }}>
        <Typography variant="h6">Population Stability Index (PSI)</Typography>
        <Table size="small">
          <TableBody>
            <TableRow>
              <TableCell>Total PSI</TableCell>
              <TableCell>{psi_total !== undefined ? `${(psi_total * 100).toFixed(2)}%` : 'N/A'}</TableCell>
            </TableRow>
            <TableRow>
              <TableCell>PSI by Rating</TableCell>
              <TableCell>
                {bin_details ? (
                  <Table size="small">
                    <TableHead>
                      <TableRow>
                        <TableCell>Rating</TableCell>
                        <TableCell>Baseline %</TableCell>
                        <TableCell>Current %</TableCell>
                        <TableCell>PSI</TableCell>
                      </TableRow>
                    </TableHead>
                    <TableBody>
                      {Object.entries(bin_details).map(([rating, details]) => (
                        <TableRow key={rating}>
                          <TableCell>{rating}</TableCell>
                          <TableCell>{details.baseline_prop ? `${(details.baseline_prop * 100).toFixed(2)}%` : 'N/A'}</TableCell>
                          <TableCell>{details.current_prop ? `${(details.current_prop * 100).toFixed(2)}%` : 'N/A'}</TableCell>
                          <TableCell>{details.psi !== undefined ? (details.psi === 0 ? '0.00%' : `${(details.psi * 100).toFixed(2)}%`) : 'N/A'}</TableCell>
                        </TableRow>
                      ))}
                    </TableBody>
                  </Table>
                ) : 'N/A'}
              </TableCell>
            </TableRow>
            <TableRow>
              <TableCell>Interpretation</TableCell>
              <TableCell>
                <Box sx={{ display: 'flex', alignItems: 'center' }}>
                  {psi_total !== undefined ? (psi_total < 0.1 ? 'No significant population change' : psi_total < 0.2 ? 'Moderate population change' : 'Significant population change') : 'N/A'}
                  <MUITooltip title="Population Stability Index (PSI) measures the stability of a model's population over time. PSI < 0.1: No significant change, 0.1 - 0.2: Moderate change, > 0.2: Significant change">
                    <InfoIcon fontSize="small" sx={{ ml: 1 }} />
                  </MUITooltip>
                </Box>
              </TableCell>
            </TableRow>
          </TableBody>
        </Table>
      </Paper>
    );
  };

  const CalibrationTab: React.FC = () => {
    if (!pdAnalysisResults?.calibration) return <Typography color="error">No Calibration Data Available</Typography>;

    const { hosmer_lemeshow } = pdAnalysisResults.calibration;
    return (
      <Grid container spacing={2}>
        <Grid item xs={12}>
          <Typography variant="h6" sx={{ mt: 3, mb: 2 }}>Binomial Test by Credit Rating (Two-sided test, alpha = 0.05)</Typography>
          {ratingBinomialTests.length > 0 ? (
            <TableContainer component={Paper}>
              <Table>
                <TableHead>
                  <TableRow>
                    <TableCell>Rating Category</TableCell>
                    <TableCell align="right">Total Samples</TableCell>
                    <TableCell align="right">Observed Defaults</TableCell>
                    <TableCell align="right">Observed Default Prob (%)</TableCell>
                    <TableCell align="right">Expected Default Prob (%)</TableCell>
                    <TableCell align="right">P-Value</TableCell>
                    <TableCell align="right">Test Result</TableCell>
                  </TableRow>
                </TableHead>
                <TableBody>
                  {ratingBinomialTests.map((test, index) => (
                    <TableRow key={test.rating || index} sx={{ backgroundColor: test.test_result === 'PASS' ? 'rgba(0, 255, 0, 0.1)' : 'rgba(255, 0, 0, 0.1)' }}>
                      <TableCell>{test.rating || 'N/A'}</TableCell>
                      <TableCell align="right">{test.total_samples ?? 'N/A'}</TableCell>
                      <TableCell align="right">{test.observed_defaults ?? 'N/A'}</TableCell>
                      <TableCell align="right">{typeof test.observed_defaults === 'number' && typeof test.total_samples === 'number' ? `${((test.observed_defaults / test.total_samples) * 100).toFixed(2)}%` : 'N/A'}</TableCell>
                      <TableCell align="right">{typeof test.expected_default_prob === 'number' ? `${(test.expected_default_prob * 100).toFixed(2)}%` : 'N/A'}</TableCell>
                      <TableCell align="right">{typeof test.p_value === 'number' ? test.p_value.toFixed(4) : 'N/A'}</TableCell>
                      <TableCell align="right">{test.test_result || 'N/A'}</TableCell>
                    </TableRow>
                  ))}
                </TableBody>
              </Table>
            </TableContainer>
          ) : (
            <Typography color="textSecondary" variant="body2">No binomial test results available for individual ratings.</Typography>
          )}
        </Grid>
        <Grid item xs={12}>
          <Paper sx={{ p: 2 }}>
            <Typography variant="h6" gutterBottom>Hosmer-Lemeshow Test</Typography>
            <TableContainer>
              <Table size="small">
                <TableBody>
                  <TableRow>
                    <TableCell>P-Value</TableCell>
                    <TableCell>{typeof hosmer_lemeshow.p_value === 'number' ? hosmer_lemeshow.p_value.toFixed(4) : 'N/A'}</TableCell>
                  </TableRow>
                  <TableRow>
                    <TableCell>Degrees of Freedom</TableCell>
                    <TableCell>{hosmer_lemeshow.degrees_of_freedom ?? 'N/A'}</TableCell>
                  </TableRow>
                  {typeof hosmer_lemeshow.p_value === 'number' && (
                    <TableRow>
                      <TableCell>Test Result (alpha = 0.05)</TableCell>
                      <TableCell sx={{ color: hosmer_lemeshow.p_value >= 0.05 ? '#2e7d32' : '#d32f2f' }}>{hosmer_lemeshow.p_value >= 0.05 ? 'PASS' : 'FAIL'}</TableCell>
                    </TableRow>
                  )}
                </TableBody>
              </Table>
            </TableContainer>
          </Paper>
        </Grid>
      </Grid>
    );
  };

  const VariableFilterDropdown: React.FC<{
    variables: [string, any][];
    selected: string[];
    setSelected: React.Dispatch<React.SetStateAction<string[]>>;
    filterType: 'Numeric' | 'Categorical';
    anchorEl: null | HTMLElement;
    setAnchorEl: React.Dispatch<React.SetStateAction<null | HTMLElement>>;
  }> = ({ variables, selected, setSelected, anchorEl, setAnchorEl }) => {
    const allVariables = variables.map(([v]) => v);
    const open = Boolean(anchorEl);

    const handleClick = (e: React.MouseEvent<HTMLButtonElement>) => setAnchorEl(e.currentTarget);
    const handleClose = () => setAnchorEl(null);
    const toggleVariable = (variable: string) => setSelected(prev => prev.includes(variable) ? prev.filter(v => v !== variable) : [...prev, variable]);
    const selectAll = () => { setSelected(allVariables); handleClose(); };
    const clearAll = () => { setSelected([]); handleClose(); };
    const removeVariable = (variable: string) => setSelected(prev => prev.filter(v => v !== variable));

    const isAllSelected = selected.length === allVariables.length;
    return (
      <Box>
        <Grid container spacing={2} alignItems="center">
          <Grid item xs={12} sm={6} md={4} lg={3}>
            <Button variant="outlined" onClick={handleClick} sx={{ width: '100%', backgroundColor: 'rgba(0, 0, 0, 0.04)', borderColor: 'transparent', '&:hover': { backgroundColor: 'rgba(0, 0, 0, 0.08)', borderColor: 'transparent' } }} endIcon={<Chip label={`${selected.length}/${allVariables.length}`} size="small" sx={{ backgroundColor: 'rgba(0, 0, 0, 0.12)' }} />}>
              Variable Filter
            </Button>
          </Grid>
          {selected.length > 0 && (
            <Grid item xs={12} sm>
              <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 1 }}>
                {selected.map(v => <Chip key={v} label={v} onDelete={() => removeVariable(v)} variant="outlined" />)}
              </Box>
            </Grid>
          )}
        </Grid>
        <Menu anchorEl={anchorEl} open={open} onClose={handleClose} TransitionComponent={Fade} PaperProps={{ sx: { maxHeight: 400, width: anchorEl?.clientWidth } }}>
          <MenuItem onClick={isAllSelected ? clearAll : selectAll}>
            <Checkbox checked={isAllSelected} indeterminate={selected.length > 0 && !isAllSelected} /> {isAllSelected ? 'Clear All' : 'Select All'}
          </MenuItem>
          <Divider />
          {allVariables.map(v => (
            <MenuItem key={v}>
              <Checkbox checked={selected.includes(v)} onChange={() => toggleVariable(v)} /> {v}
            </MenuItem>
          ))}
        </Menu>
      </Box>
    );
  };

  const WOEPlot: React.FC<{ woeData: WOEPlotData[]; isCategorical: boolean }> = ({ woeData, isCategorical }) => {
    // Format numeric x values to have at most 2 decimal places
    const normalizedData = woeData.map((item, i) => {
      const xValue = item.x !== undefined ? item.x : (item.bin_number !== undefined ? item.bin_number : i);
      // Format numeric x values to have at most 2 decimal places
      const formattedX = typeof xValue === 'number' && !isCategorical ? Number(xValue.toFixed(2)) : xValue;
      
      return {
        x: formattedX,
        y: item.y !== undefined ? item.y : (item.woe !== undefined ? item.woe : 0),
        // For numeric variables, ensure the label also has at most 2 decimal places
        label: isCategorical 
          ? (item.label || item.bin_range || `Category ${i}`) 
          : (item.label 
              ? (typeof item.label === 'number' ? Number(item.label).toFixed(2) : item.label) 
              : (item.bin_range || (typeof formattedX === 'number' ? formattedX.toFixed(2) : `Bin ${i}`))),
      };
    });
    
    return (
      <ResponsiveContainer width="100%" height={300}>
        <LineChart data={normalizedData}>
          <CartesianGrid strokeDasharray="3 3" />
          <XAxis 
            dataKey="label" 
            type={isCategorical ? "category" : "number"} 
            domain={isCategorical ? ['auto', 'auto'] : [Math.min(...normalizedData.map(d => Number(d.x))), Math.max(...normalizedData.map(d => Number(d.x)))]} 
            interval={0} 
            angle={isCategorical ? -45 : 0} 
            height={isCategorical ? 80 : 40} 
            tick={{ fontSize: 10, transform: isCategorical ? 'translate(0, 10)' : undefined }}
            label={{ value: isCategorical ? 'Categories' : 'Mean Value', position: 'insideBottom', offset: isCategorical ? -15 : -5 }}
            // Format the ticks to show at most 2 decimal places for numeric variables
            tickFormatter={isCategorical ? undefined : (value) => typeof value === 'number' ? value.toFixed(2) : value}
          />
          <YAxis label={{ value: 'Weight of Evidence (WOE)', angle: -90, position: 'insideLeft' }} />
          <RechartsTooltip 
            contentStyle={{ backgroundColor: isDarkMode ? theme.palette.background.paper : 'white', color: isDarkMode ? theme.palette.text.primary : 'black' }}
            // Format tooltip values to show at most 2 decimal places for x-axis
            formatter={(value, name) => [value, name]}
            labelFormatter={(label) => typeof label === 'number' ? label.toFixed(2) : label}
          />
          <Line type="monotone" dataKey="y" stroke="#8884d8" strokeWidth={2} dot={{ r: 5 }} activeDot={{ r: 8 }} />
        </LineChart>
      </ResponsiveContainer>
    );
  };

  const VariableAssessmentTab: React.FC = () => {
    if (!pdAnalysisResults?.variable_assessment) return <Typography color="error">No Variable Assessment Data Available</Typography>;

    const categoricalVars = Object.entries(pdAnalysisResults.variable_assessment.categorical_variables || {});
    const numericVars = Object.entries(pdAnalysisResults.variable_assessment.numeric_variables || {});
    const filteredNumeric = numericFilter.length > 0 ? numericVars.filter(([v]) => numericFilter.includes(v)) : numericVars;
    const filteredCategorical = categoricalFilter.length > 0 ? categoricalVars.filter(([v]) => categoricalFilter.includes(v)) : categoricalVars;

    const renderTable = (vars: [string, any][]) => (
      <TableBody>
        {vars.map(([name, data]) => (
          <TableRow key={name}>
            <TableCell>{name}</TableCell>
            <TableCell>{data.iv?.iv_total?.toFixed(4) || 'N/A'}</TableCell>
            <TableCell>{data.csi_total !== undefined ? `${(data.csi_total * 100).toFixed(2)}%` : 'N/A'}</TableCell>
          </TableRow>
        ))}
      </TableBody>
    );

    const renderDetails = (type: 'numeric_variables' | 'categorical_variables', vars: [string, any][]) =>
      vars.map(([name, data]) => {
        if (!data.iv) return null;
        const woeData = data.iv?.woe_plot_data || [];
        const isCategorical = type === 'categorical_variables';

        return (
          <Grid item xs={12} key={name}>
            <Paper elevation={3} sx={{ p: 2, mb: 2 }}>
              <Typography variant="h6">{name}</Typography>
              <Grid container spacing={2}>
                <Grid item xs={12} md={6}>
                  <Typography>
                    Information Value (IV): {data.iv?.iv_total?.toFixed(4) || 'N/A'}
                    <MUITooltip title="Information Value indicates predictive power: < 0.02: Unpredictive, 0.02-0.1: Weak, 0.1-0.3: Medium, > 0.3: Strong">
                      <InfoIcon fontSize="small" sx={{ ml: 1 }} />
                    </MUITooltip>
                  </Typography>
                </Grid>
                <Grid item xs={12}>
                  <WOEPlot woeData={woeData} isCategorical={isCategorical} />
                </Grid>
                <Grid item xs={12}>
                  <TableContainer>
                    <Table size="small">
                      <TableHead>
                        <TableRow>
                          <TableCell>Bin</TableCell>
                          <TableCell>Total Count</TableCell>
                          <TableCell>Default Count</TableCell>
                          <TableCell>Non-Default Count</TableCell>
                          <TableCell>WOE</TableCell>
                          <TableCell>IV</TableCell>
                        </TableRow>
                      </TableHead>
                      <TableBody>
                        {data.iv?.details?.bins?.map((bin: any, i: number) => {
                          const woeValue = bin.woe || data.iv?.details?.woe?.[i] || 0;
                          const ivBinValue = data.iv?.details?.bin_details?.iv_per_bin?.[i] || bin.iv_bin || 0;
                          // Use bin_range from CSI data for numeric, woeData.label for categorical
                          let binLabel;
                          if (isCategorical) {
                            binLabel = woeData[i]?.label || bin.bin_range || `Category ${i}`;
                          } else {
                            // For numeric variables, prioritize bin_range from bin object, then CSI data
                            if (bin.bin_range) {
                              binLabel = bin.bin_range;
                            } else if (data.iv.details.bin_details.bin_range[String(i)]) {
                              binLabel = data.iv.details.bin_details.bin_range[String(i)]
                            } else if (data.csi_bin_details && data.csi_bin_details[String(i)] && data.csi_bin_details[String(i)].bin_range) {
                              binLabel = data.csi_bin_details[String(i)].bin_range;
                            } else {
                              binLabel = `Bin ${i}`;
                            }
                          }
                          const totalCount = bin.total_count || 0;
                          const defaultCount = bin.default_count || 0;

                          return (
                            <TableRow key={i}>
                              <TableCell>{binLabel}</TableCell>
                              <TableCell>{Math.round(totalCount)}</TableCell>
                              <TableCell>{Math.round(defaultCount)}</TableCell>
                              <TableCell>{Math.round(totalCount - defaultCount)}</TableCell>
                              <TableCell>{Number(woeValue).toFixed(4)}</TableCell>
                              <TableCell>{Number(ivBinValue).toFixed(4)}</TableCell>
                            </TableRow>
                          );
                        })}
                      </TableBody>
                    </Table>
                  </TableContainer>
                </Grid>
              </Grid>
            </Paper>
          </Grid>
        );
      });

    return (
      <Grid container spacing={2}>
        <Grid item xs={12}>
          <Paper>
            <Tabs value={variableSubTab} onChange={handleSubTabChange} variant="fullWidth">
              <Tab label="Numeric Variables" />
              <Tab label="Categorical Variables" />
            </Tabs>
          </Paper>
        </Grid>
        {variableSubTab === 0 && (
          <Grid item xs={12}>
            <Paper sx={{ p: 2 }}>
              <Typography variant="h6">Numeric Variables</Typography>
              <VariableFilterDropdown variables={numericVars} selected={numericFilter} setSelected={setNumericFilter} filterType="Numeric" anchorEl={numericMenuAnchor} setAnchorEl={setNumericMenuAnchor} />
              <TableContainer>
                <Table size="small">
                  <TableHead>
                    <TableRow>
                      <TableCell>Variable</TableCell>
                      <TableCell>Information Value (IV)</TableCell>
                      <TableCell>Characteristic Stability Index (CSI)</TableCell>
                    </TableRow>
                  </TableHead>
                  {renderTable(filteredNumeric)}
                </Table>
              </TableContainer>
              {renderDetails('numeric_variables', filteredNumeric)}
            </Paper>
          </Grid>
        )}
        {variableSubTab === 1 && (
          <Grid item xs={12}>
            <Paper sx={{ p: 2 }}>
              <Typography variant="h6">Categorical Variables</Typography>
              <VariableFilterDropdown variables={categoricalVars} selected={categoricalFilter} setSelected={setCategoricalFilter} filterType="Categorical" anchorEl={categoricalMenuAnchor} setAnchorEl={setCategoricalMenuAnchor} />
              <TableContainer>
                <Table size="small">
                  <TableHead>
                    <TableRow>
                      <TableCell>Variable</TableCell>
                      <TableCell>Information Value (IV)</TableCell>
                      <TableCell>Characteristic Stability Index (CSI)</TableCell>
                    </TableRow>
                  </TableHead>
                  {renderTable(filteredCategorical)}
                </Table>
              </TableContainer>
              {renderDetails('categorical_variables', filteredCategorical)}
            </Paper>
          </Grid>
        )}
      </Grid>
    );
  };

  // Main Render
  if (isLoading) return <Paper sx={{ p: 2, m: 2, display: 'flex', justifyContent: 'center', alignItems: 'center', height: '80vh' }}><CircularProgress /></Paper>;
  if (error) return <Paper sx={{ p: 2, m: 2 }}><Typography color="error" variant="h6">Error Loading PD Analysis</Typography><Typography>{error}</Typography></Paper>;

  return (
    <Paper elevation={3} sx={{ p: 2, bboxShadow: 3, borderRadius: 4  }}>
      <Typography variant="h4" gutterBottom>PD Model Analysis</Typography>
      <Tabs value={tabValue} onChange={handleTabChange} sx={{ borderBottom: 1, borderColor: 'divider', mb: 2 }} variant="scrollable" scrollButtons="auto">
        <Tab label="Discriminatory Power" />
        <Tab label="Stability" />
        <Tab label="Calibration" />
        <Tab label="Variable Assessment" />
      </Tabs>
      <Box sx={{ flex: 1 }}>
        <TabPanel value={tabValue} index={0}><DiscriminatoryPowerTab /></TabPanel>
        <TabPanel value={tabValue} index={1}><StabilityTab /></TabPanel>
        <TabPanel value={tabValue} index={2}><CalibrationTab /></TabPanel>
        <TabPanel value={tabValue} index={3}><VariableAssessmentTab /></TabPanel>
      </Box>
    </Paper>
  );
};

export default PDModel;