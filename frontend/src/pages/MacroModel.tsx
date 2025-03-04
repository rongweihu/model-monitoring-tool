import React, { useState, useEffect } from 'react';
import {
  Paper, Typography, Box, Table, TableBody, TableCell, TableContainer, TableHead, TableRow,
  Grid, Chip, CircularProgress, FormControl, InputLabel, Select, MenuItem, Checkbox, ListItemText,
  Tabs, Tab, Tooltip as MUITooltip, SelectChangeEvent
} from '@mui/material';
import SummarizeIcon from '@mui/icons-material/Summarize';
import AssessmentIcon from '@mui/icons-material/Assessment';
import TimelineIcon from '@mui/icons-material/Timeline';
import HelpOutlineIcon from '@mui/icons-material/HelpOutline';
import InfoOutlinedIcon from '@mui/icons-material/InfoOutlined';
import IconButton from '@mui/material/IconButton';
import { useTheme } from '@mui/material/styles';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip as RechartsTooltip, ResponsiveContainer, Legend, ScatterChart, Scatter, Bar, ComposedChart } from 'recharts';
import { api } from '../utils/api';

// === Types and Interfaces ===
interface MacroModelThresholds {
  normality_threshold: number;
  autocorrelation_threshold: number;
  heteroscedasticity_threshold: number;
  rmse_threshold: number;
  r_squared_threshold: number;
  stationarity_threshold: number;
}

interface NormalityResult {
  'AD Statistic': number;
  'Critical Values': number[];
  'Significance Level': number[];
  qq_plot?: { theoretical_quantiles: number[]; sample_quantiles: number[]; line: { x: number[]; y: number[] } };
  normality_visualization?: { normal_curve: { x: number[]; y: number[] }; histogram: { x: number[]; y: number[] }; distribution_params: { mean: number; std: number } };
}

interface AutocorrelationResult {
  'DW Statistic': number;
  'Interpretation': string;
  acf?: { lags: number[]; acf_values: number[]; confidence_interval: { upper: number[]; lower: number[] } };
  pacf?: { lags: number[]; pacf_values: number[]; confidence_interval: { upper: number[]; lower: number[] } };
}

interface HeteroscedasticityResult {
  'LM Statistic': number;
  'p-value': number;
  predicted_pd: number[];
  model_error: number[];
}

interface StationarityResult {
  'ADF Statistic': number;
  'ADF p-value': number;
  'KPSS Statistic': number;
  'KPSS p-value': number;
  'KPSS Note': string;
  'Zivot-Andrews Statistic': number;
  'ZA p-value': number;
  'Phillips-Perron Statistic': number;
  'PP p-value': number;
}

interface MacroModelResults {
  normality_results: { model_error: NormalityResult };
  autocorrelation_results: { model_error: AutocorrelationResult };
  heteroscedasticity_results: HeteroscedasticityResult;
  comparison_results: {
    'Adjusted R-squared': number;
    'RMSE': number;
    actual_default_rate: number[];
    predicted_pd: number[];
    trend_line?: { x: number[]; y: number[]; equation: { slope: number; intercept: number } };
  };
  stationarity_results: { [key: string]: StationarityResult };
  time_series_data: Array<{ snapshot_ccyymm: string; [key: string]: number | string }>;
}

// === Constants ===
const DEFAULT_THRESHOLDS: MacroModelThresholds = {
  normality_threshold: 0.05,
  autocorrelation_threshold: 1.5,
  heteroscedasticity_threshold: 0.05,
  rmse_threshold: 0.1,
  r_squared_threshold: 0.7,
  stationarity_threshold: 0.05,
};

const CHART_COLORS = ['#8884d8', '#82ca9d', '#ffc658'];

// === Utility Functions ===
const safeToFixed = (value: number | undefined | null, decimals: number = 4): string =>
  value === undefined || value === null ? 'N/A' : value.toFixed(decimals);

const getNormalityInterpretation = (result: NormalityResult): string => {
  const stat = result['AD Statistic'];
  const critValues = result['Critical Values'];
  const sigLevels = result['Significance Level'];
  const firstGreater = critValues.findIndex(cv => cv > stat);
  if (firstGreater === -1) return `Model error is not normally distributed (reject null hypothesis) at significance level of ${sigLevels[sigLevels.length - 1]}%`;
  if (firstGreater === 0) return `Model error is normally distributed (fail to reject null hypothesis) at all tested significance levels`;
  return `Model error is normally distributed (fail to reject null hypothesis) at significance level of ${sigLevels[firstGreater]}%`;
};

const getHeteroscedasticityInterpretation = (pValue: number): string => {
  if (pValue > 0.05) return 'No significant evidence of heteroscedasticity';
  if (pValue > 0.01) return 'Reject null hypothesis at 5% significance level. Evidence of heteroscedasticity.';
  return 'Reject null hypothesis of homoscedasticity (p < 0.01). Strong evidence of heteroscedasticity.';
};

// === Reusable Components ===
const TestCard: React.FC<{ title: string | JSX.Element; children: React.ReactNode; sx?: any }> = ({ title, children, sx }) => (
  <Paper elevation={3} sx={{ border: '2px solid rgba(0, 0, 0, 0.2)', borderRadius: 2, p: 2, mb: 2, ...sx }}>
    <Typography variant="h6" sx={{ mb: 2, pb: 1, borderBottom: '1px solid rgba(0, 0, 0, 0.1)', fontWeight: 'bold', color: 'primary.main' }}>
      {title}
    </Typography>
    {children}
  </Paper>
);

const TestResultChip: React.FC<{ test: string; result: any }> = ({ test, result }) => {
  let status: 'success' | 'error' = 'error';
  let label = 'Fail';

  switch (test) {
    case 'Normality Test':
      status = result.is_normal ? 'success' : 'error';
      label = result.is_normal ? 'Pass' : 'Fail';
      break;
    case 'Autocorrelation Test':
      status = result.status.toLowerCase().includes('no autocorrelation') ? 'success' : 'error';
      label = status === 'success' ? 'Pass' : 'Fail';
      break;
    case 'Heteroscedasticity Test':
      status = result.is_homoscedastic ? 'success' : 'error';
      label = result.is_homoscedastic ? 'Pass' : 'Fail';
      break;
    case 'Stationarity Test':
      status = result.is_stationary ? 'success' : 'error';
      label = result.is_stationary ? 'Pass' : 'Fail';
      break;
  }

  return <Chip label={label} color={status} variant="outlined" />;
};

// === Sub-Components ===
const NormalityTestSection: React.FC<{ results: MacroModelResults; theme: any }> = ({ results, theme }) => {
  const { model_error } = results.normality_results;
  const normalityViz = model_error.normality_visualization;
  const interpretation = getNormalityInterpretation(model_error);

  return (
    <TestCard title="Normality Test (Anderson-Darling Test)">
      <TableContainer component={Paper} variant="outlined" style={{ padding: '16px', width: '100%', margin: '0 auto' }}>
        <Table>
          <TableBody>
            <TableRow>
              <TableCell component="th" scope="row">AD Statistic</TableCell>
              <TableCell>{safeToFixed(model_error['AD Statistic'])}</TableCell>
            </TableRow>
            <TableRow>
              <TableCell component="th" scope="row">Critical Values</TableCell>
              <TableCell>[{model_error['Critical Values'].map(v => safeToFixed(v)).join(', ')}]</TableCell>
            </TableRow>
            <TableRow>
              <TableCell component="th" scope="row">Significance Levels</TableCell>
              <TableCell>[{model_error['Significance Level'].map(v => `${v}%`).join(', ')}]</TableCell>
            </TableRow>
            <TableRow>
              <TableCell component="th" scope="row">
                <Box sx={{ display: 'flex', alignItems: 'center' }}>
                  Interpretation
                  <MUITooltip title={<Typography variant="body2">
                    <strong>Anderson-Darling Normality Test:</strong>
                    <br /><br />
                    <ul>
                      <li>Statistics &lt; critical value at certain significance level: Fail to reject null hypothesis</li>
                      <li>Suggests the data is likely normally distributed</li>
                      <li>Indicates residuals follow a normal distribution</li>
                    </ul>
                    <br />
                    A normal distribution of residuals is crucial for reliable statistical inference and model validity.
                  </Typography>} placement="right">
                    <IconButton size="small" sx={{ ml: 1 }}><InfoOutlinedIcon fontSize="small" /></IconButton>
                  </MUITooltip>
                </Box>
              </TableCell>
              <TableCell sx={{ color: interpretation.includes('normally distributed') ? 'green' : 'red' }}>{interpretation}</TableCell>
            </TableRow>
            {normalityViz && (
              <>
                <TableRow>
                  <TableCell colSpan={2} rowSpan={3}>
                    <TestCard title="Model Error Distribution Analysis">
                      <Typography variant="subtitle1" sx={{ textAlign: 'center', fontWeight: 'bold', marginBottom: 2 }}>
                        Model Error Histogram with Normal Distribution Curve
                      </Typography>
                      <Box sx={{ width: '100%', height: 450 }}>
                        <ResponsiveContainer>
                          <ComposedChart width={500} height={450} data={normalityViz.normal_curve.x.map((x: number, i: number) => {
                            const histogramIndex = normalityViz.histogram.x.findIndex((histX, index) => x >= histX && x <= normalityViz.histogram.x[index + 1]);
                            return {
                              x,
                              histogram: histogramIndex !== -1 ? normalityViz.histogram.y[histogramIndex] : 0,
                              normalCurve: normalityViz.normal_curve.y[i],
                            };
                          })}>
                            <CartesianGrid strokeDasharray="3 3" />
                            <XAxis type="number" dataKey="x" domain={['auto', 'auto']} label={{ value: 'Model Error', position: 'insideBottom', offset: -5 }} />
                            <YAxis label={{ value: 'Density', angle: -90, position: 'insideLeft' }} />
                            <RechartsTooltip contentStyle={{
                              backgroundColor: theme.palette.background.paper,
                              color: theme.palette.text.primary,
                              border: `1px solid ${theme.palette.divider}`,
                              borderRadius: '4px',
                              boxShadow: '0 2px 10px rgba(0,0,0,0.1)',
                            }} labelStyle={{ color: theme.palette.text.primary, fontWeight: 'bold' }} itemStyle={{ color: theme.palette.text.secondary }} />
                            <Bar dataKey="histogram" fill="#8884d8" name="Histogram" barSize={80} data={normalityViz.histogram.x.map((x, i) => ({ x, histogram: normalityViz.histogram.y[i] }))} />
                            <Line type="monotone" dataKey="normalCurve" stroke="#ff7300" dot={false} name="Normal Distribution" />
                            <Legend verticalAlign="bottom" height={36} layout="horizontal" align="center" wrapperStyle={{ bottom: -20, left: 0, right: 0 }} />
                          </ComposedChart>
                        </ResponsiveContainer>
                      </Box>
                      <Typography variant="subtitle1" sx={{ textAlign: 'center', fontWeight: 'bold', marginTop: 4, marginBottom: 2 }}>
                        Q-Q Plot of Model Error
                      </Typography>
                      <Box sx={{ width: '100%', height: 400 }}>
                        <ResponsiveContainer>
                          <ComposedChart margin={{ top: 20, right: 20, bottom: 20, left: 20 }}>
                            <CartesianGrid strokeDasharray="3 3" />
                            <XAxis type="number" dataKey="x" name="Theoretical Quantiles" label={{ value: 'Theoretical Quantiles', position: 'bottom', offset: 0 }} />
                            <YAxis type="number" dataKey="y" name="Sample Quantiles" label={{ value: 'Sample Quantiles', angle: -90, position: 'insideLeft' }} />
                            <RechartsTooltip contentStyle={{
                              backgroundColor: theme.palette.background.paper,
                              color: theme.palette.text.primary,
                              border: `1px solid ${theme.palette.divider}`,
                              borderRadius: '4px',
                              boxShadow: theme.palette.mode === 'dark' ? '0 2px 10px rgba(255,255,255,0.1)' : '0 2px 10px rgba(0,0,0,0.1)',
                            }} labelStyle={{ color: theme.palette.text.primary, fontWeight: 'bold' }} itemStyle={{ color: theme.palette.text.secondary }} />
                            <Line name="Reference Line" type="linear" dataKey="y" stroke="#ff7300" strokeWidth={2} dot={false} data={[
                              { x: model_error.qq_plot!.line.x[0], y: model_error.qq_plot!.line.y[0] },
                              { x: model_error.qq_plot!.line.x[1], y: model_error.qq_plot!.line.y[1] },
                            ]} />
                            <Scatter name="Q-Q Plot Points" data={model_error.qq_plot!.theoretical_quantiles.map((q, i) => ({
                              x: q,
                              y: model_error.qq_plot!.sample_quantiles[i],
                            }))} fill="#8884d8" />
                            <Legend verticalAlign="bottom" height={36} layout="horizontal" align="center" wrapperStyle={{ bottom: -20, left: 0, right: 0 }} />
                          </ComposedChart>
                        </ResponsiveContainer>
                      </Box>
                    </TestCard>
                  </TableCell>
                </TableRow>
                <TableRow></TableRow>
                <TableRow></TableRow>
                <TableRow>
                  <TableCell component="th" scope="row">Distribution Parameters</TableCell>
                  <TableCell>{normalityViz ? `Mean: ${normalityViz.distribution_params.mean.toFixed(4)}, Std Dev: ${normalityViz.distribution_params.std.toFixed(4)}` : 'N/A'}</TableCell>
                </TableRow>
              </>
            )}
          </TableBody>
        </Table>
      </TableContainer>
    </TestCard>
  );
};

const AutocorrelationTestSection: React.FC<{ results: MacroModelResults; theme: any }> = ({ results, theme }) => {
  const { model_error } = results.autocorrelation_results;
  const { acf, pacf } = model_error;

  return (
    <TestCard title="Autocorrelation Test (Durbin-Watson Test)">
      <TableContainer component={Paper} variant="outlined" style={{ padding: '16px', width: '100%', margin: '0 auto' }}>
        <Table>
          <TableBody>
            <TableRow>
              <TableCell component="th" scope="row">
                <Box sx={{ display: 'flex', alignItems: 'center' }}>
                  DW Statistic
                  <MUITooltip title={<Typography variant="body2">
                    <strong>Durbin-Watson Statistic Interpretation:</strong>
                    <br /><br />
                    <ul>
                      <li>DW &lt; 1.5: Strong evidence of positive autocorrelation in model error</li>
                      <li>1.5 ≤ DW ≤ 2.5: Little to no autocorrelation in model error</li>
                      <li>DW &gt; 2.5: Strong evidence of negative autocorrelation in model error</li>
                    </ul>
                    <br />
                    Positive autocorrelation suggests that consecutive residuals are similar, indicating potential model specification issues or time-dependent patterns.
                  </Typography>} placement="right">
                    <IconButton size="small" sx={{ ml: 1 }}><InfoOutlinedIcon fontSize="small" /></IconButton>
                  </MUITooltip>
                </Box>
              </TableCell>
              <TableCell>{safeToFixed(model_error['DW Statistic'])}</TableCell>
            </TableRow>
            <TableRow>
              <TableCell component="th" scope="row">
                <Box sx={{ display: 'flex', alignItems: 'center' }}>
                  Interpretation
                  <MUITooltip title={<Typography variant="body2">
                    <strong>Autocorrelation Assessment:</strong>
                    <br /><br />
                    Indicates the presence and nature of serial correlation in model residuals. Ideal models should have residuals that are independent of each other.
                  </Typography>} placement="right">
                    <IconButton size="small" sx={{ ml: 1 }}><InfoOutlinedIcon fontSize="small" /></IconButton>
                  </MUITooltip>
                </Box>
              </TableCell>
              <TableCell>
                <Typography color={model_error.Interpretation.toLowerCase().includes('no autocorrelation') ? 'success.main' : 'error.main'}>
                  {model_error.Interpretation}
                </Typography>
              </TableCell>
            </TableRow>
            {acf && pacf && (
              <TableRow>
                <TableCell colSpan={2}>
                  <TestCard title="Autocorrelation and Partial Autocorrelation Analysis">
                    <Grid container spacing={2}>
                      <Grid item xs={12} md={6}>
                        <Typography variant="subtitle1" sx={{ textAlign: 'center', fontWeight: 'bold', marginBottom: 2 }}>
                          Autocorrelation Function (ACF)
                        </Typography>
                        <Box sx={{ width: '100%', height: 400 }}>
                          <ResponsiveContainer>
                            <ComposedChart width={500} height={400} data={acf.lags.map((lag, i) => ({
                              lag,
                              acf: acf.acf_values[i],
                              upperCI: i > 0 ? acf.confidence_interval.upper[i] : null,
                              lowerCI: i > 0 ? acf.confidence_interval.lower[i] : null,
                            }))}>
                              <CartesianGrid strokeDasharray="3 3" />
                              <XAxis dataKey="lag" type="number" label={{ value: 'Lag', position: 'insideBottom', offset: -5 }} />
                              <YAxis label={{ value: 'Autocorrelation', angle: -90, position: 'insideLeft' }} />
                              <RechartsTooltip contentStyle={{
                                backgroundColor: theme.palette.background.paper,
                                color: theme.palette.text.primary,
                                border: `1px solid ${theme.palette.divider}`,
                                borderRadius: '4px',
                                boxShadow: '0 2px 10px rgba(0,0,0,0.1)',
                              }} labelStyle={{ color: theme.palette.text.primary, fontWeight: 'bold' }} itemStyle={{ color: theme.palette.text.secondary }} />
                              <Bar dataKey="acf" fill="#8884d8" name="ACF" barSize={20} />
                              <Line dataKey="upperCI" stroke="red" dot={false} name="Upper CI" strokeDasharray="5 5" />
                              <Line dataKey="lowerCI" stroke="red" dot={false} name="Lower CI" strokeDasharray="5 5" />
                              <Legend verticalAlign="bottom" height={36} layout="horizontal" align="center" wrapperStyle={{ bottom: -20, left: 0, right: 0 }} />
                            </ComposedChart>
                          </ResponsiveContainer>
                        </Box>
                      </Grid>
                      <Grid item xs={12} md={6}>
                        <Typography variant="subtitle1" sx={{ textAlign: 'center', fontWeight: 'bold', marginBottom: 2 }}>
                          Partial Autocorrelation Function (PACF)
                        </Typography>
                        <Box sx={{ width: '100%', height: 400 }}>
                          <ResponsiveContainer>
                            <ComposedChart width={500} height={400} data={pacf.lags.map((lag, i) => ({
                              lag,
                              pacf: pacf.pacf_values[i],
                              upperCI: pacf.confidence_interval.upper[i],
                              lowerCI: pacf.confidence_interval.lower[i],
                            }))}>
                              <CartesianGrid strokeDasharray="3 3" />
                              <XAxis dataKey="lag" type="number" label={{ value: 'Lag', position: 'insideBottom', offset: -5 }} />
                              <YAxis label={{ value: 'Partial Autocorrelation', angle: -90, position: 'insideLeft' }} />
                              <RechartsTooltip contentStyle={{
                                backgroundColor: theme.palette.background.paper,
                                color: theme.palette.text.primary,
                                border: `1px solid ${theme.palette.divider}`,
                                borderRadius: '4px',
                                boxShadow: '0 2px 10px rgba(0,0,0,0.1)',
                              }} labelStyle={{ color: theme.palette.text.primary, fontWeight: 'bold' }} itemStyle={{ color: theme.palette.text.secondary }} />
                              <Bar dataKey="pacf" fill="#8884d8" name="PACF" barSize={20} />
                              <Line dataKey="upperCI" stroke="red" dot={false} name="Upper CI" strokeDasharray="5 5" />
                              <Line dataKey="lowerCI" stroke="red" dot={false} name="Lower CI" strokeDasharray="5 5" />
                              <Legend verticalAlign="bottom" height={36} layout="horizontal" align="center" wrapperStyle={{ bottom: -20, left: 0, right: 0 }} />
                            </ComposedChart>
                          </ResponsiveContainer>
                        </Box>
                      </Grid>
                    </Grid>
                  </TestCard>
                </TableCell>
              </TableRow>
            )}
          </TableBody>
        </Table>
      </TableContainer>
    </TestCard>
  );
};

const HeteroscedasticityTestSection: React.FC<{ results: MacroModelResults; theme: any }> = ({ results, theme }) => {
  const { heteroscedasticity_results } = results;
  const interpretation = getHeteroscedasticityInterpretation(heteroscedasticity_results['p-value']);

  // Debug data
  console.log('Heteroscedasticity Data:', {
    predicted_pd: heteroscedasticity_results.predicted_pd,
    model_error: heteroscedasticity_results.model_error,
  });

  const scatterData = heteroscedasticity_results.predicted_pd.map((x, i) => ({
    x,
    y: heteroscedasticity_results.model_error[i],
  }));

  return (
    <TestCard title="Heteroscedasticity Test (Breusch-Pagan Test)">
      <TableContainer component={Paper} variant="outlined" style={{ padding: '16px', width: '100%', margin: '0 auto' }}>
        <Table size="small">
          <TableBody>
            <TableRow>
              <TableCell component="th" scope="row"><strong>LM Statistic</strong></TableCell>
              <TableCell>{safeToFixed(heteroscedasticity_results['LM Statistic'])}</TableCell>
            </TableRow>
            <TableRow>
              <TableCell component="th" scope="row"><strong>P-value</strong></TableCell>
              <TableCell>{safeToFixed(heteroscedasticity_results['p-value'])}</TableCell>
            </TableRow>
            <TableRow>
              <TableCell component="th" scope="row"><strong>Interpretation</strong></TableCell>
              <TableCell>
                <Typography color={interpretation.includes('No significant evidence') ? 'success.main' : 'error.main'}>
                  {interpretation}
                </Typography>
              </TableCell>
            </TableRow>
          </TableBody>
        </Table>
      </TableContainer>
      <Box sx={{ height: 20 }} />
      {heteroscedasticity_results.predicted_pd.length > 0 && (
        <Grid item xs={12}>
          <TestCard title="Heteroscedasticity Analysis">
            <Typography variant="subtitle1" sx={{ textAlign: 'center', fontWeight: 'bold', marginBottom: 2 }}>
              Model Error vs Predicted Probability of Default
            </Typography>
            <Box sx={{ width: '100%', height: 400 }}>
              <ResponsiveContainer>
                <ScatterChart margin={{ top: 20, right: 40, bottom: 40, left: 60 }}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis 
                    type="number" 
                    dataKey="x" 
                    name="Predicted PD" 
                    tickFormatter={(value: number) => value.toFixed(4)} 
                    label={{ value: 'Predicted Probability of Default', position: 'bottom', offset: 20 }} 
                  />
                  <YAxis 
                    type="number" 
                    dataKey="y" 
                    name="Model Error" 
                    tickFormatter={(value: number) => value.toFixed(4)} 
                    label={{ value: 'Model Error', angle: -90, position: 'insideLeft', offset: -30, style: { textAnchor: 'middle' } }} 
                  />
                  <RechartsTooltip 
                    cursor={{ strokeDasharray: '3 3' }}
                    formatter={(value: number, name: string) => {
                      // Use entry.payload to access x and y directly
                      if (name === 'x') return [value.toFixed(4), 'Predicted PD'];
                      if (name === 'y') return [value.toFixed(4), 'Model Error'];
                      return [value.toFixed(4), name]; // Fallback
                    }}
                    content={({ active, payload }) => {
                      if (active && payload && payload.length) {
                        const data = payload[0].payload; // Access the full data point
                        return (
                          <div style={{
                            backgroundColor: theme.palette.background.paper,
                            color: theme.palette.text.primary,
                            border: `1px solid ${theme.palette.divider}`,
                            borderRadius: '4px',
                            padding: '8px',
                            boxShadow: '0 2px 10px rgba(0,0,0,0.1)',
                          }}>
                            <p style={{ fontWeight: 'bold' }}>{`Predicted PD: ${data.x.toFixed(4)}`}</p>
                            <p>{`Model Error: ${data.y.toFixed(4)}`}</p>
                          </div>
                        );
                      }
                      return null;
                    }}
                  />
                  <Scatter 
                    name="Model Error vs Predicted PD" 
                    data={scatterData} 
                    fill="#8884d8" 
                    fillOpacity={0.7} 
                  />
                  <Legend verticalAlign="bottom" height={36} layout="horizontal" align="center" wrapperStyle={{ bottom: -20, left: 0, right: 0 }} />
                </ScatterChart>
              </ResponsiveContainer>
            </Box>
          </TestCard>
        </Grid>
      )}
    </TestCard>
  );
};

const ModelPerformanceSection: React.FC<{ results: MacroModelResults; thresholds: MacroModelThresholds; theme: any }> = ({ results, thresholds, theme }) => {
  const { comparison_results } = results;
  const rSquaredPass = comparison_results['Adjusted R-squared'] >= thresholds.r_squared_threshold;
  const rmsePass = comparison_results['RMSE'] < thresholds.rmse_threshold;

  return (
    <TestCard title="Model Performance">
      <TableContainer component={Paper} variant="outlined" style={{ padding: '16px', width: '100%', margin: '0 auto' }}>
        <Table size="small">
          <TableBody>
            <TableRow>
              <TableCell component="th" scope="row"><strong>Adjusted R-squared</strong></TableCell>
              <TableCell>
                {safeToFixed(comparison_results['Adjusted R-squared'])}
                <br />
                <Typography color={rSquaredPass ? 'success.main' : 'error.main'}>
                  {rSquaredPass ? `Passes threshold (≥ ${thresholds.r_squared_threshold}). Model explains variation well.` : `Below threshold (< ${thresholds.r_squared_threshold}). Model may not explain variation adequately.`}
                </Typography>
              </TableCell>
            </TableRow>
            <TableRow>
              <TableCell component="th" scope="row"><strong>RMSE</strong></TableCell>
              <TableCell>
                {safeToFixed(comparison_results['RMSE'])}
                <br />
                <Typography color={rmsePass ? 'success.main' : 'error.main'}>
                  {rmsePass ? `Good prediction accuracy (RMSE < ${thresholds.rmse_threshold}).` : `High prediction error (RMSE ≥ ${thresholds.rmse_threshold}). Consider model improvements.`}
                </Typography>
              </TableCell>
            </TableRow>
            <TableRow>
              <TableCell component="th" scope="row"><strong>Overall Assessment</strong></TableCell>
              <TableCell>
                <Typography color={rSquaredPass && rmsePass ? 'success.main' : 'warning.main'}>
                  {(() => {
                    if (rSquaredPass && rmsePass) return 'Model performance meets all criteria.';
                    if (!rSquaredPass && !rmsePass) return 'Model performance needs improvement in both fit and accuracy.';
                    if (!rSquaredPass) return 'Model fit needs improvement, but accuracy is acceptable.';
                    return 'Model has good fit, but prediction accuracy needs improvement.';
                  })()}
                </Typography>
              </TableCell>
            </TableRow>
          </TableBody>
        </Table>
      </TableContainer>
      <Box sx={{ height: 20 }} />
      {comparison_results.actual_default_rate.length > 0 && (
        <Grid item xs={12}>
          <TestCard title="Model Prediction Analysis">
            <Typography variant="subtitle1" sx={{ textAlign: 'center', fontWeight: 'bold', marginBottom: 2 }}>
              Predicted Probability of Default vs Actual Default Rate
            </Typography>
            <Box sx={{ width: '100%', height: 400 }}>
              <ResponsiveContainer>
                <ComposedChart width={500} height={400}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis type="number" dataKey="x" name="Actual Default Rate" label={{ value: 'Actual Default Rate', position: 'insideBottom', offset: -5 }} />
                  <YAxis type="number" dataKey="y" name="Predicted PD" label={{ value: 'Predicted Probability of Default', angle: -90, position: 'insideLeft' }} />
                  <RechartsTooltip formatter={(value, name, props) => {
                    const axisId = props.dataKey === 'Predicted PD' ? 'left' : 'right';
                    return [`${value} (${axisId} axis)`, name];
                  }} contentStyle={{
                    backgroundColor: theme.palette.background.paper,
                    color: theme.palette.text.primary,
                    border: `1px solid ${theme.palette.divider}`,
                    borderRadius: '4px',
                    boxShadow: '0 2px 10px rgba(0,0,0,0.1)',
                  }} labelStyle={{ color: theme.palette.text.primary, fontWeight: 'bold' }} itemStyle={{ color: theme.palette.text.secondary }} />
                  <Scatter name="Predicted PD vs Actual Default Rate" data={comparison_results.actual_default_rate.map((x, i) => ({
                    x,
                    y: comparison_results.predicted_pd[i],
                  }))} fill="#8884d8" fillOpacity={0.7} />
                  {comparison_results.trend_line && (
                    <>
                      <Line name="Trend Line" type="linear" dataKey="y" stroke="#ff7300" strokeWidth={2} dot={false} data={[
                        { x: comparison_results.trend_line.x[0], y: comparison_results.trend_line.y[0] },
                        { x: comparison_results.trend_line.x[1], y: comparison_results.trend_line.y[1] },
                      ]} />
                      <text x={120} y={40} fill={theme.palette.text.primary} fontFamily="Arial, sans-serif" fontSize={14} fontWeight="bold" textAnchor="start" style={{
                        backgroundColor: theme.palette.background.paper,
                        padding: '4px 8px',
                        borderRadius: '4px',
                        boxShadow: '0 2px 4px rgba(0,0,0,0.1)',
                      }}>{`Trend Line: y = ${comparison_results.trend_line.equation.slope.toFixed(4)}x + ${comparison_results.trend_line.equation.intercept.toFixed(4)}`}</text>
                    </>
                  )}
                  <Legend verticalAlign="bottom" height={36} layout="horizontal" align="center" wrapperStyle={{ bottom: -20, left: 0, right: 0 }} />
                </ComposedChart>
              </ResponsiveContainer>
            </Box>
          </TestCard>
        </Grid>
      )}
    </TestCard>
  );
};

const SummaryTab: React.FC<{ results: MacroModelResults; thresholds: MacroModelThresholds; theme: any }> = ({ results, thresholds, theme }) => {
  const normalityInterpretation = getNormalityInterpretation(results.normality_results.model_error);
  const autocorrelationInterpretation = results.autocorrelation_results.model_error.Interpretation;

  const stationarityResults = Object.entries(results.stationarity_results).reduce((acc, [variable, stats]) => {
    const nonStationaryCount = ['ADF p-value', 'KPSS p-value', 'ZA p-value', 'PP p-value'].filter(key => {
      const value = stats[key as keyof StationarityResult];
      return typeof value === 'number' && value >= thresholds.stationarity_threshold;
    }).length;
    acc[variable] = { status: nonStationaryCount >= 2 ? 'Fail' : 'Pass', interpretation: nonStationaryCount >= 2 ? 'Non-stationary' : 'Stationary' };
    return acc;
  }, {} as Record<string, { status: string; interpretation: string }>);

  return (
    <Box>
      <TableContainer component={Paper} variant="outlined" sx={{ p: 2, mb: 2 }}>
        <Typography variant="h6" sx={{ p: 2, pb: 0 }}>Model Performance Tests</Typography>
        <Table>
          <TableHead>
            <TableRow>
              <TableCell><strong>Test</strong></TableCell>
              <TableCell><strong>Result</strong></TableCell>
              <TableCell><strong>Interpretation</strong></TableCell>
            </TableRow>
          </TableHead>
          <TableBody>
            <TableRow>
              <TableCell>Normality Test</TableCell>
              <TableCell><TestResultChip test="Normality Test" result={{ is_normal: !normalityInterpretation.includes('not normally distributed') }} /></TableCell>
              <TableCell>{normalityInterpretation}</TableCell>
            </TableRow>
            <TableRow>
              <TableCell>Autocorrelation Test</TableCell>
              <TableCell><TestResultChip test="Autocorrelation Test" result={{ status: autocorrelationInterpretation }} /></TableCell>
              <TableCell>{autocorrelationInterpretation}</TableCell>
            </TableRow>
            <TableRow>
              <TableCell>Heteroscedasticity Test</TableCell>
              <TableCell><TestResultChip test="Heteroscedasticity Test" result={{ is_homoscedastic: results.heteroscedasticity_results['p-value'] > 0.05 }} /></TableCell>
              <TableCell>{results.heteroscedasticity_results['p-value'] > 0.05 ? 'No significant evidence of heteroscedasticity in model error' : 'Evidence of heteroscedasticity in model error'}</TableCell>
            </TableRow>
            <TableRow>
              <TableCell>Model Performance</TableCell>
              <TableCell>
                <Chip label={results.comparison_results['Adjusted R-squared'] >= thresholds.r_squared_threshold && results.comparison_results['RMSE'] < thresholds.rmse_threshold ? 'Pass' : 'Fail'}
                  color={results.comparison_results['Adjusted R-squared'] >= thresholds.r_squared_threshold && results.comparison_results['RMSE'] < thresholds.rmse_threshold ? 'success' : 'error'} variant="outlined" />
              </TableCell>
              <TableCell>{`R-squared: ${safeToFixed(results.comparison_results['Adjusted R-squared'])} | RMSE: ${safeToFixed(results.comparison_results['RMSE'])}`}</TableCell>
            </TableRow>
          </TableBody>
        </Table>
      </TableContainer>
      <Box sx={{ mt: 4, mb: 4 }}>
        <Paper variant="outlined" sx={{ p: 2 }}>
          <Typography variant="h6" sx={{ mb: 2 }}>Model Performance Time Series</Typography>
          <ResponsiveContainer width="100%" height={400}>
            <LineChart data={results.time_series_data.map(item => ({
              snapshot_ccyymm: item.snapshot_ccyymm,
              'Predicted PD': item.pred_dr,
              'Actual Default Rate': item.Defaultrate,
            }))} margin={{ top: 5, right: 30, left: 20, bottom: 5 }}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="snapshot_ccyymm" angle={-45} textAnchor="end" height={60} tick={{ fontSize: 12 }} />
              <YAxis yAxisId="left" label={{ value: 'Rate', angle: -90, position: 'insideLeft', style: { textAnchor: 'middle' } }} />
              <RechartsTooltip contentStyle={{
                backgroundColor: theme.palette.background.paper,
                color: theme.palette.text.primary,
                border: `1px solid ${theme.palette.divider}`,
                borderRadius: '4px',
                boxShadow: theme.palette.mode === 'dark' ? '0 2px 10px rgba(255,255,255,0.1)' : '0 2px 10px rgba(0,0,0,0.1)',
              }} labelStyle={{ color: theme.palette.text.primary, fontWeight: 'bold' }} itemStyle={{ color: theme.palette.text.secondary }} />
              <Legend verticalAlign="top" height={36} />
              <Line yAxisId="left" type="monotone" dataKey="Predicted PD" stroke="#8884d8" dot={{ r: 3 }} activeDot={{ r: 5 }} strokeWidth={2} />
              <Line yAxisId="left" type="monotone" dataKey="Actual Default Rate" stroke="#82ca9d" dot={{ r: 3 }} activeDot={{ r: 5 }} strokeWidth={2} />
            </LineChart>
          </ResponsiveContainer>
        </Paper>
      </Box>
      {Object.keys(stationarityResults).length > 0 && (
        <TableContainer component={Paper} variant="outlined" sx={{ p: 2 }}>
          <Typography variant="h6" sx={{ p: 2, pb: 0 }}>Macro Variable Stationarity Tests</Typography>
          <Table>
            <TableHead>
              <TableRow>
                <TableCell><strong>Macro Variable</strong></TableCell>
                <TableCell><strong>Result</strong></TableCell>
                <TableCell><strong>Interpretation</strong></TableCell>
              </TableRow>
            </TableHead>
            <TableBody>
              {Object.entries(stationarityResults).map(([variable, { status, interpretation }]) => (
                <TableRow key={variable}>
                  <TableCell>{variable}</TableCell>
                  <TableCell><TestResultChip test="Stationarity Test" result={{ is_stationary: status === 'Pass' }} /></TableCell>
                  <TableCell>{interpretation}</TableCell>
                </TableRow>
              ))}
            </TableBody>
          </Table>
        </TableContainer>
      )}
    </Box>
  );
};

const StationarityTab: React.FC<{ results: MacroModelResults; thresholds: MacroModelThresholds; selectedSeries: string[]; setSelectedSeries: (series: string[]) => void; seriesOptions: string[] }> = ({ results, thresholds, selectedSeries, setSelectedSeries, seriesOptions }) => {
  const theme = useTheme();
  const handleChange = (event: SelectChangeEvent<string[]>) => {
    const value = event.target.value as string[];
    setSelectedSeries(value.includes('clear-all') ? [] : value.filter(s => s !== 'snapshot_ccyymm'));
  };

  return (
    <Grid container spacing={3}>
      {Object.entries(results.stationarity_results).map(([variable, stats]) => (
        <Grid item xs={12} key={variable}>
          <TestCard title={`Stationarity Test - ${variable} (α = ${thresholds.stationarity_threshold})`}>
            <TableContainer component={Paper} variant="outlined" style={{ padding: '16px', width: '100%', margin: '0 auto' }}>
              <Table size="small">
                <TableHead>
                  <TableRow>
                    <TableCell><strong>Test</strong></TableCell>
                    <TableCell><strong>Statistic</strong></TableCell>
                    <TableCell><strong>P-value</strong></TableCell>
                    <TableCell><strong>Result</strong></TableCell>
                  </TableRow>
                </TableHead>
                <TableBody>
                  {[
                    { name: 'ADF', stat: 'ADF Statistic', p: 'ADF p-value' },
                    { name: 'KPSS', stat: 'KPSS Statistic', p: 'KPSS p-value' },
                    { name: 'Zivot-Andrews', stat: 'Zivot-Andrews Statistic', p: 'ZA p-value' },
                    { name: 'Phillips-Perron', stat: 'Phillips-Perron Statistic', p: 'PP p-value' },
                  ].map(({ name, stat, p }) => (
                    <TableRow key={name}>
                      <TableCell><strong>{name} Test</strong></TableCell>
                      <TableCell>{safeToFixed(Number(stats[stat as keyof StationarityResult]))}</TableCell>
                      <TableCell>{safeToFixed(Number(stats[p as keyof StationarityResult]))}</TableCell>
                      <TableCell>
                        <Chip label={Number(stats[p as keyof StationarityResult]) < thresholds.stationarity_threshold ? 'Stationary' : 'Non-Stationary'}
                          color={Number(stats[p as keyof StationarityResult]) < thresholds.stationarity_threshold ? 'success' : 'error'} variant="outlined" />
                      </TableCell>
                    </TableRow>
                  ))}
                </TableBody>
              </Table>
            </TableContainer>
          </TestCard>
        </Grid>
      ))}
      <Grid item xs={12}>
        <TestCard title={<Box display="flex" alignItems="center">
          Time Series Visualization
          <MUITooltip title="You can select up to two variables to plot. When two variables are selected, they will be displayed on separate Y-axes for better comparison." placement="right">
            <HelpOutlineIcon color="action" sx={{ marginLeft: 1, fontSize: 20, cursor: 'help' }} />
          </MUITooltip>
        </Box>}>
          <FormControl fullWidth>
            <InputLabel id="series-select-label">Select Series</InputLabel>
            <Select multiple value={selectedSeries} onChange={handleChange} renderValue={selected => selected.join(', ')} displayEmpty MenuProps={{
              PaperProps: { style: { maxHeight: 48 * 4.5 + 8, width: 250 } },
            }}>
              <MenuItem key="clear-all" value="clear-all" onClick={() => setSelectedSeries([])}>
                <ListItemText primary="Clear All" />
              </MenuItem>
              {seriesOptions.map(series => (
                <MenuItem key={series} value={series} disabled={selectedSeries.length >= 2 && !selectedSeries.includes(series)}>
                  <Checkbox checked={selectedSeries.includes(series)} />
                  <ListItemText primary={series} />
                </MenuItem>
              ))}
            </Select>
          </FormControl>
          <Box sx={{ width: '100%', height: 400 }}>
            <ResponsiveContainer>
              <LineChart data={results.time_series_data}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="snapshot_ccyymm" name="Time" />
                <YAxis yAxisId="left" label={{ value: selectedSeries[0] || 'Left Axis', angle: -90, position: 'insideLeft' }} />
                {selectedSeries.length > 1 && <YAxis yAxisId="right" orientation="right" label={{ value: selectedSeries[1] || 'Right Axis', angle: 90, position: 'insideRight' }} />}
                <RechartsTooltip formatter={(value, name, props) => {
                  const axisId = props.dataKey === selectedSeries[0] ? 'left' : 'right';
                  return [`${value} (${axisId} axis)`, name];
                }} contentStyle={{
                  backgroundColor: theme.palette.background.paper,
                  color: theme.palette.text.primary,
                  border: `1px solid ${theme.palette.divider}`,
                  borderRadius: '4px',
                  boxShadow: '0 2px 10px rgba(0,0,0,0.1)',
                }} labelStyle={{ color: theme.palette.text.primary, fontWeight: 'bold' }} itemStyle={{ color: theme.palette.text.secondary }} />
                <Legend verticalAlign="bottom" height={36} layout="horizontal" align="center" wrapperStyle={{ bottom: -20, left: 0, right: 0 }} />
                {selectedSeries.map((series, index) => (
                  <Line key={series} type="monotone" dataKey={series} yAxisId={index === 0 ? 'left' : 'right'} stroke={CHART_COLORS[index % CHART_COLORS.length]} dot={{ strokeWidth: 2, r: 5 }} />
                ))}
              </LineChart>
            </ResponsiveContainer>
          </Box>
        </TestCard>
      </Grid>
    </Grid>
  );
};

// === Main Component ===
const MacroModel: React.FC = () => {
  const theme = useTheme();

  const [thresholds, setThresholds] = useState<MacroModelThresholds>(() => {
    try {
      const stored = localStorage.getItem('macroModelThresholds');
      return stored ? { ...DEFAULT_THRESHOLDS, ...JSON.parse(stored) } : DEFAULT_THRESHOLDS;
    } catch {
      return DEFAULT_THRESHOLDS;
    }
  });
  const [results, setResults] = useState<MacroModelResults | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [activeTab, setActiveTab] = useState<'summary' | 'model_performance' | 'stationarity'>('summary');
  const [selectedSeries, setSelectedSeries] = useState<string[]>([]);
  const [seriesOptions, setSeriesOptions] = useState<string[]>([]);

  useEffect(() => {
    const handleStorageChange = (e: StorageEvent) => {
      if (e.key === 'macroModelThresholds' && e.newValue) setThresholds(JSON.parse(e.newValue));
    };
    window.addEventListener('storage', handleStorageChange);
    return () => window.removeEventListener('storage', handleStorageChange);
  }, []);

  useEffect(() => {
    const loadData = async () => {
      setIsLoading(true);
      try {
        const response = await api.analyzeMacroModel() as MacroModelResults;
        setResults(response);
        const seriesNames = Object.keys(response.time_series_data[0] || {}).filter(key => key !== 'snapshot_ccyymm');
        setSeriesOptions(seriesNames);
        setSelectedSeries([seriesNames[0]]);
      } catch (err) {
        setError(err instanceof Error && err.message.includes('Network Error') ? 'Network Error' : 'Error Loading Macro Analysis');
      } finally {
        setIsLoading(false);
      }
    };
    loadData();
  }, []);

  return (
      <Paper elevation={1} sx={{ width: '100%', minHeight: '80vh', padding: '16px', border: '1px solid rgba(0, 0, 0, 0.12)', boxShadow: 3, borderRadius: 4 }}>
        <Typography variant="h5" gutterBottom sx={{ mb: 2, pb: 1, borderBottom: '1px solid rgba(0, 0, 0, 0.12)' }}>
          Macro Model Analysis Results
        </Typography>
        <Tabs value={activeTab} onChange={(_, val) => setActiveTab(val)} variant="fullWidth" sx={{ borderBottom: 1, borderColor: 'divider', mb: 2 }}>
          <Tab value="summary" label="Summary" icon={<SummarizeIcon />} iconPosition="start" />
          <Tab value="model_performance" label="Model Performance" icon={<AssessmentIcon />} iconPosition="start" />
          <Tab value="stationarity" label="Stationarity Tests" icon={<TimelineIcon />} iconPosition="start" />
        </Tabs>
        {isLoading ? (
          <Grid container justifyContent="center"><CircularProgress /></Grid>
        ) : error ? (
          <Typography color="error">{error}</Typography>
        ) : results ? (
          <Grid item xs={12}>
            {activeTab === 'summary' && <SummaryTab results={results} thresholds={thresholds} theme={theme} />}
            {activeTab === 'model_performance' && (
              <Grid container spacing={3}>
                <Grid item xs={13}><NormalityTestSection results={results} theme={theme} /></Grid>
                <Grid item xs={13}><AutocorrelationTestSection results={results} theme={theme} /></Grid>
                <Grid item xs={12}><HeteroscedasticityTestSection results={results} theme={theme} /></Grid>
                <Grid item xs={12}><ModelPerformanceSection results={results} thresholds={thresholds} theme={theme} /></Grid>
              </Grid>
            )}
            {activeTab === 'stationarity' && <StationarityTab results={results} thresholds={thresholds} selectedSeries={selectedSeries} setSelectedSeries={setSelectedSeries} seriesOptions={seriesOptions} />}
          </Grid>
        ) : null}
      </Paper>
  );
};

export default MacroModel;