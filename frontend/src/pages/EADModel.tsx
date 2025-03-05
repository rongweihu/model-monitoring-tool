import React, { useState, useEffect } from 'react';
import {
  Box, FormControl, Grid, InputLabel, MenuItem, Paper, Select, SelectChangeEvent, Table, TableBody, TableCell,
  TableContainer, TableHead, TableRow, Typography,
} from '@mui/material';
import {
  ScatterChart, Scatter, XAxis, YAxis, CartesianGrid, Tooltip as RechartsTooltip, ResponsiveContainer, ReferenceLine,
} from 'recharts';
import { useTheme } from '@mui/material/styles';
import { debounce } from 'lodash';
import { api } from '../utils/api';

// === Interfaces ===
interface DecileData {
  actual_decile: string;
  values: number[];
}

interface EADComparisonData {
  actual: number;
  predicted: number;
  modelName: string;
}

interface PerformanceMetric {
  metric: string;
  value: number;
  threshold: number;
  status: string;
  quarter?: string;
  portfolio?: string;
  modelName?: string;
}

interface EADAnalysisResults {
  metrics: Record<string, number>;
  plot_data: { actual_ead: number[]; predicted_ead: number[] };
  decile_data: { data: DecileData[]; total_count: number };
  additional_data: any[];
}

interface Thresholds {
  MAPE: number;
  'R-squared': number;
}

// === Utility Components ===
const MetricBox: React.FC<{ metric: PerformanceMetric }> = ({ metric }) => {
  const isPassing = metric.status === 'Pass';
  return (
    <Grid item xs={12} md={6}>
      <Paper sx={{ p: 2, backgroundColor: isPassing ? 'success.light' : 'error.light', color: 'white', textAlign: 'center' }}>
        <Typography variant="h5" sx={{ fontWeight: 'bold' }}>
          {metric.metric}
          <Typography component="span" variant="body1" sx={{ display: 'block', mt: 1, fontWeight: 'normal' }}>
            {metric.value.toFixed(4)} (Threshold: {metric.threshold.toFixed(4)})
          </Typography>
          <Typography component="span" variant="body1" sx={{ display: 'block', mt: 0, fontWeight: 'bold' }}>
            {isPassing ? 'PASS' : 'FAIL'}
          </Typography>
        </Typography>
      </Paper>
    </Grid>
  );
};

const DecileTable: React.FC<{ data: DecileData[]; totalCount: number; isDarkMode: boolean }> = ({ data, totalCount, isDarkMode }) => (
  <TableContainer component={Paper} sx={{ backgroundColor: isDarkMode ? 'background.default' : undefined, boxShadow: isDarkMode ? 'none' : undefined }}>
    <Table size="small">
      <TableHead>
        <TableRow>
          <TableCell sx={{ backgroundColor: isDarkMode ? 'background.paper' : undefined, color: isDarkMode ? 'text.primary' : undefined }}>Actual/Predicted</TableCell>
          {Array.from({ length: 10 }, (_, i) => (
            <TableCell key={i} align="right" sx={{ backgroundColor: isDarkMode ? 'background.paper' : undefined, color: isDarkMode ? 'text.primary' : undefined }}>D{i + 1}</TableCell>
          ))}
        </TableRow>
      </TableHead>
      <TableBody>
        {data.map((row, index) => (
          <TableRow key={index}>
            <TableCell sx={{ backgroundColor: isDarkMode ? 'background.paper' : undefined, color: isDarkMode ? 'text.primary' : undefined }}>{row.actual_decile}</TableCell>
            {row.values.map((value, cellIndex) => (
              <TableCell
                key={cellIndex}
                align="right"
                sx={{
                  backgroundColor: isDarkMode
                    ? value === 0 ? 'background.paper' : value < 2 ? '#2e7d32' : value < 5 ? '#1b5e20' : value < 10 ? '#0d47a1' : value < 15 ? '#1565c0' : '#0d47a1'
                    : value === 0 ? '#ffffff' : value < 2 ? '#e8f5e9' : value < 5 ? '#c8e6c9' : value < 10 ? '#81c784' : value < 15 ? '#66bb6a' : '#4caf50',
                  color: isDarkMode ? (value === 0 ? 'text.primary' : 'white') : (value === 0 ? 'inherit' : 'black'),
                  fontWeight: value > 0 ? 'bold' : 'normal',
                }}
              >
                {value.toFixed(2)}%
              </TableCell>
            ))}
          </TableRow>
        ))}
        <TableRow>
          <TableCell colSpan={11} align="left" sx={{ borderTop: '2px solid', borderColor: isDarkMode ? 'divider' : 'rgba(224, 224, 224, 1)', backgroundColor: isDarkMode ? 'background.paper' : '#fafafa', fontWeight: 500, color: isDarkMode ? 'text.primary' : undefined }}>
            Total Count: {totalCount.toLocaleString()}
          </TableCell>
        </TableRow>
      </TableBody>
    </Table>
  </TableContainer>
);

// === Main Component ===
const EADModel: React.FC = () => {
  const theme = useTheme();
  const isDarkMode = theme.palette.mode === 'dark';

  // State Management
  const [selectedQuarter, setSelectedQuarter] = useState<string>('');
  const [selectedPortfolio, setSelectedPortfolio] = useState<string>('');
  const [selectedModelName, setSelectedModelName] = useState<string>('');
  const [modelNames, setModelNames] = useState<string[]>([]);
  const [comparisonData, setComparisonData] = useState<EADComparisonData[]>([]);
  const [error, setError] = useState<{ message: string; details?: string } | null>(null);
  const [performanceMetrics, setPerformanceMetrics] = useState<PerformanceMetric[]>([]);
  const [results, setResults] = useState<EADAnalysisResults | null>(null);
  const [isInitialFetch, setIsInitialFetch] = useState(true);
  const [thresholds, setThresholds] = useState<Thresholds>(() => {
    const stored = localStorage.getItem('eadThresholds');
    const defaults = { MAPE: 15.0, 'R-squared': 0.8 };
    if (!stored) return defaults;
    try {
      const parsed = JSON.parse(stored);
      return {
        MAPE: parsed.find((t: any) => t.metric === 'MAPE')?.threshold ?? defaults.MAPE,
        'R-squared': parsed.find((t: any) => t.metric === 'R-squared')?.threshold ?? defaults['R-squared'],
      };
    } catch {
      return defaults;
    }
  });

  // Threshold Handling
  const updateThresholds = () => {
    const stored = localStorage.getItem('eadThresholds');
    if (stored) {
      try {
        const parsed = JSON.parse(stored);
        setThresholds({
          MAPE: parsed.find((t: any) => t.metric === 'MAPE')?.threshold ?? 15.0,
          'R-squared': parsed.find((t: any) => t.metric === 'R-squared')?.threshold ?? 0.8,
        });
      } catch {}
    }
    if (results) calculatePerformanceMetrics(results);
  };

  useEffect(() => {
    const handleStorageChange = (event: StorageEvent) => event.key === 'eadThresholds' && updateThresholds();
    window.addEventListener('storage', handleStorageChange);
    window.addEventListener('eadThresholdUpdate', updateThresholds);
    return () => {
      window.removeEventListener('storage', handleStorageChange);
      window.removeEventListener('eadThresholdUpdate', updateThresholds);
    };
  }, [results]);

  // Data Fetching with Debouncing
  const fetchEADAnalysis = debounce(async (quarter?: string, portfolio?: string, modelName?: string) => {
    const requestId = `${Date.now()}-${Math.random().toString(36).substring(2, 9)}`;
    try {
      console.log('Fetching EAD analysis with:', { quarter, portfolio, modelName, requestId });
      const response = await api.analyzeEAD({
        quarter: quarter || undefined,
        portfolio: portfolio || undefined,
        modelName: modelName || undefined,
        request_id: requestId,
      }) as EADAnalysisResults;

      if (!response) {
        throw new Error('No response data received');
      }

      setResults(response);
      setComparisonData(response.plot_data.actual_ead.map((actual, i) => ({
        actual,
        predicted: response.plot_data.predicted_ead[i],
        modelName: modelName || 'Default Model',
      })));
      calculatePerformanceMetrics(response);

      const modelNamesFromData = [...new Set(response.additional_data.map((item: any) => item.ModelName))].sort();
      setModelNames(modelNamesFromData);
      if (!selectedModelName && modelNamesFromData.length > 0 && isInitialFetch) {
        setSelectedModelName(modelNamesFromData[0]); // Set without immediate re-fetch
      }
    } catch (err) {
      const errorMsg = err instanceof Error ? err.message : 'Unknown error';
      const errorDetails = err instanceof Error && err.stack ? err.stack : JSON.stringify(err);
      console.error('Error fetching EAD analysis:', {
        message: errorMsg,
        details: errorDetails,
        quarter,
        portfolio,
        modelName,
      });
      setError({
        message: 'Error Loading EAD Analysis',
        details: err instanceof Error && err.message.includes('Network Error') ? 'Network Error' : errorMsg,
      });
    } finally {
      if (isInitialFetch) setIsInitialFetch(false);
    }
  }, 300); // 300ms debounce delay

  const calculatePerformanceMetrics = (data: EADAnalysisResults) => {
    const metrics = [
      { metric: 'MAPE', value: data.metrics.MAPE || 0, threshold: thresholds.MAPE, status: (data.metrics.MAPE || 0) <= thresholds.MAPE ? 'Pass' : 'Fail', quarter: selectedQuarter || 'All', portfolio: selectedPortfolio || 'All', modelName: selectedModelName || 'All' },
      { metric: 'R-squared', value: data.metrics['R-squared'] || 0, threshold: thresholds['R-squared'], status: (data.metrics['R-squared'] || 0) >= thresholds['R-squared'] ? 'Pass' : 'Fail', quarter: selectedQuarter || 'All', portfolio: selectedPortfolio || 'All', modelName: selectedModelName || 'All' },
    ];
    setPerformanceMetrics(metrics.filter(m => (!selectedQuarter || m.quarter === selectedQuarter) && (!selectedPortfolio || m.portfolio === selectedPortfolio) && (!selectedModelName || m.modelName === selectedModelName)));
  };

  useEffect(() => {
    fetchEADAnalysis(selectedQuarter, selectedPortfolio, selectedModelName);
    return () => fetchEADAnalysis.cancel(); // Cleanup debounce on unmount
  }, [selectedQuarter, selectedPortfolio, selectedModelName]);

  // Handlers
  const handleChange = (setter: React.Dispatch<React.SetStateAction<string>>) => (event: SelectChangeEvent<string>) => setter(event.target.value);

  // Render
  if (error) return (
    <Paper elevation={3} sx={{ p: 2, bboxShadow: 3, borderRadius: 4 }}>
      <Typography color="error" variant="h6">{error.message}</Typography>
      {error.details && <Typography variant="body1" sx={{ mt: 1, color: 'black' }}>{error.details}</Typography>}
      <Typography variant="body2" sx={{ mt: 1 }}>Check the browser console for more details.</Typography>
    </Paper>
  );

  return (
    <Paper elevation={3} sx={{ p: 2, bboxShadow: 3, borderRadius: 4 }}>
      <Typography variant="h6" gutterBottom>EAD Model Analysis</Typography>
      <Grid container spacing={2}>
        <Grid item xs={12} sm={4}>
          <FormControl fullWidth variant="outlined" size="small">
            <InputLabel>Quarter</InputLabel>
            <Select value={selectedQuarter} label="Quarter" onChange={handleChange(setSelectedQuarter)}>
              <MenuItem value="">All Quarters</MenuItem>
              <MenuItem value="2024Q1">2024 Q1</MenuItem>
              <MenuItem value="2023Q4">2023 Q4</MenuItem>
            </Select>
          </FormControl>
        </Grid>
        <Grid item xs={12} sm={4}>
          <FormControl fullWidth variant="outlined" size="small">
            <InputLabel>Portfolio</InputLabel>
            <Select value={selectedPortfolio} label="Portfolio" onChange={handleChange(setSelectedPortfolio)}>
              <MenuItem value="">All Portfolios</MenuItem>
              <MenuItem value="retail">Retail</MenuItem>
              <MenuItem value="corporate">Corporate</MenuItem>
            </Select>
          </FormControl>
        </Grid>
        <Grid item xs={12} sm={4}>
          <FormControl fullWidth variant="outlined" size="small">
            <InputLabel>Model Name</InputLabel>
            <Select value={selectedModelName} label="Model Name" onChange={handleChange(setSelectedModelName)}>
              <MenuItem value="">All Models</MenuItem>
              {modelNames.map(m => <MenuItem key={m} value={m}>{m}</MenuItem>)}
            </Select>
          </FormControl>
        </Grid>
      </Grid>
      {results && (
        <Box sx={{ flex: 1, overflow: 'auto', width: '100%', mt: 2 }}>
          <Grid container spacing={2} sx={{ mb: 2 }}>
            {performanceMetrics.map((metric, i) => <MetricBox key={i} metric={metric} />)}
          </Grid>
          <Box sx={{ width: '100%', height: '50vh', minHeight: 400, mb: 2, px: 0 }}>
            <ResponsiveContainer width="100%" height="100%">
              <ScatterChart margin={{ top: 20, right: 40, bottom: 40, left: 60 }}>
                <CartesianGrid />
                <XAxis type="number" dataKey="actual" name="Actual EAD" tickFormatter={value => value.toFixed(0)} label={{ value: 'Actual EAD', position: 'bottom', offset: 20 }} />
                <YAxis type="number" dataKey="predicted" name="Predicted EAD" tickFormatter={value => value.toFixed(0)} label={{ value: 'Predicted EAD', angle: -90, position: 'insideLeft', offset: -30, style: { textAnchor: 'middle' } }} />
                <RechartsTooltip contentStyle={{ backgroundColor: isDarkMode ? theme.palette.background.paper : 'white', color: isDarkMode ? theme.palette.text.primary : 'black' }} />
                <Scatter name="EAD Comparison" data={comparisonData} fill="#8884d8" />
                <ReferenceLine x={0} y={0} stroke="red" strokeDasharray="3 3" />
              </ScatterChart>
            </ResponsiveContainer>
          </Box>
          {results.decile_data && (
            <Box sx={{ mt: 4 }}>
              <Typography variant="h6" gutterBottom>Decile Analysis</Typography>
              <DecileTable data={results.decile_data.data} totalCount={results.decile_data.total_count} isDarkMode={isDarkMode} />
            </Box>
          )}
        </Box>
      )}
    </Paper>
  );
};

export default EADModel;