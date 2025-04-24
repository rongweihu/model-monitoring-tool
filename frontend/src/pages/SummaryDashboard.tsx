import React, { useState, useEffect } from 'react';
import { Box, Typography, Paper, Grid, CircularProgress, Alert } from '@mui/material';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts';
import api from '../utils/api';
import { useSnackbar } from 'notistack';
import { useTheme } from '@mui/material/styles';

interface MetricData {
  date: string;
  PD_Gini?: number;
  PD_KS?: number;
  LGD_MAPE?: number;
  EAD_MAPE?: number;
  Macro_R2?: number;
}

interface AnalysisResult {
  id: number;
  dataset_id: number;
  analysis_type: string;
  result_data: string;
  parameters: string;
  created_at: string;
  as_of_date: string | null;
}

const SummaryDashboard: React.FC = () => {
  const [metrics, setMetrics] = useState<MetricData[]>([]);
  const [loading, setLoading] = useState<boolean>(true);
  const [error, setError] = useState<string | null>(null);
  const { enqueueSnackbar } = useSnackbar();

  useEffect(() => {
    const fetchMetrics = async () => {
      setLoading(true);
      try {
        const response = await api.getAllAnalysisResults();
        const results: AnalysisResult[] = response.analysis_results || [];

        const metricData: MetricData[] = results
          .filter((r) => r.as_of_date)
          .map((r) => {
            let resultData;
            try {
              resultData = JSON.parse(r.result_data);
            } catch (e) {
              console.warn(`Invalid result_data for analysis result ${r.id}:`, r.result_data);
              return null;
            }

            const date = r.as_of_date!;
            const dataPoint: MetricData = { date };

            try {
              if (r.analysis_type === 'pd') {
                dataPoint.PD_Gini = resultData.gini_coefficient;
                dataPoint.PD_KS = resultData.ks_statistic;
              } else if (r.analysis_type === 'lgd') {
                dataPoint.LGD_MAPE = resultData.metrics?.MAPE;
              } else if (r.analysis_type === 'ead') {
                dataPoint.EAD_MAPE = resultData.metrics?.MAPE;
              } else if (r.analysis_type === 'macro') {
                dataPoint.Macro_R2 = resultData.comparison_results?.['Adjusted R-squared'];
              }
            } catch (e) {
              console.warn(`Error processing result_data for analysis result ${r.id}:`, e);
              return null;
            }

            return dataPoint;
          })
          .filter((data): data is MetricData => data !== null)
          .reduce((acc: MetricData[], curr) => {
            const existing = acc.find((d) => d.date === curr.date);
            if (existing) {
              Object.assign(existing, curr);
            } else {
              acc.push(curr);
            }
            return acc;
          }, [])
          .sort((a, b) => new Date(a.date).getTime() - new Date(b.date).getTime());

        setMetrics(metricData);
        if (metricData.length === 0) {
          setError('No valid metrics available. Please upload and analyze data with as-of-dates.');
        }
      } catch (err: any) {
        console.error('Failed to load summary metrics:', err);
        setError(`Failed to load summary metrics: ${err.message || 'Unknown error'}`);
        enqueueSnackbar('Failed to load summary metrics', { variant: 'error' });
      } finally {
        setLoading(false);
      }
    };

    fetchMetrics();
  }, [enqueueSnackbar]);

  if (loading) {
    return (
      <Box display="flex" justifyContent="center" mt={4}>
        <CircularProgress />
      </Box>
    );
  }

  if (error) {
    return (
      <Box mx={4} mt={4}>
        <Alert severity="error">{error}</Alert>
      </Box>
    );
  }
  const theme = useTheme();
  const isDarkMode = theme.palette.mode === 'dark';
  // Define the metrics to display with their properties
  const metricPlots = [
    {
      dataKey: 'PD_Gini',
      name: 'PD Gini Coefficient',
      stroke: '#8884d8',
    },
    {
      dataKey: 'PD_KS',
      name: 'PD KS Statistics',
      stroke: '#82ca9d',
    },
    {
      dataKey: 'LGD_MAPE',
      name: 'LGD MAPE',
      stroke: '#ffc658',
    },
    {
      dataKey: 'EAD_MAPE',
      name: 'EAD MAPE',
      stroke: '#ff7300',
    },
    {
      dataKey: 'Macro_R2',
      name: 'Macro R²',
      stroke: '#a4de6c',
    },
  ];

  return (
    <Box mx={4} mt={4}>
      <Typography variant="h4" gutterBottom>
        Model Performance Summary
      </Typography>
      <Paper elevation={3} sx={{ p: 4 }}>
        <Grid container spacing={4}>
          <Grid item xs={12}>
            <Typography variant="h6" gutterBottom>
              Performance Metrics Over Time
            </Typography>
            {metrics.length === 0 ? (
              <Alert severity="info">No metrics available. Please upload and analyze data with as-of-dates.</Alert>
            ) : (
              <Grid container spacing={4}>
                {metricPlots.map((plot) => {
                  // Filter data to include only points where the metric is defined
                  const filteredData = metrics.filter((data) => data[plot.dataKey as keyof MetricData] !== undefined);
                  return (
                    <Grid item xs={12} md={6} key={plot.dataKey}>
                      <Typography variant="subtitle1" gutterBottom>
                        {plot.name}
                      </Typography>
                      {filteredData.length === 0 ? (
                        <Alert severity="info">No data available for {plot.name}.</Alert>
                      ) : (
                        <Box sx={{ height: 300 }}>
                          <ResponsiveContainer width="100%" height="100%">
                            <LineChart
                              data={filteredData}
                              margin={{ top: 20, right: 30, left: 10, bottom: 20 }}
                            >
                              <CartesianGrid strokeDasharray="3 3" />
                              <XAxis
                                dataKey="date"
                                tickFormatter={(date) => new Date(date).toLocaleDateString()}
                                stroke={isDarkMode ? "#E0E0E0" : "#000000"}
                                tick={{ fill: isDarkMode ? "#E0E0E0" : "#000000"}}
                              />
                              <YAxis 
                                stroke={isDarkMode ? "#E0E0E0" : "#000000"}
                                tick={{ fill: isDarkMode ? "#E0E0E0" : "#000000"}}
                              />
                              <Tooltip
                                formatter={(value: number) => [value.toFixed(3), plot.name.replace('_', ' ')]}
                                labelFormatter={(date) => new Date(date).toLocaleDateString()}
                                
                              />
                              <Line
                                type="monotone"
                                dataKey={plot.dataKey}
                                name={plot.name}
                                stroke={plot.stroke}
                                strokeWidth={2}
                                dot={{ r: 4, strokeWidth: 2 }}
                                isAnimationActive={false}
                              />
                            </LineChart>
                          </ResponsiveContainer>
                        </Box>
                      )}
                    </Grid>
                  );
                })}
              </Grid>
            )}
          </Grid>
        </Grid>
      </Paper>
    </Box>
  );
};

export default SummaryDashboard;