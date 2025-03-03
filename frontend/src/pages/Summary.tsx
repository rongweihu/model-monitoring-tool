import * as React from 'react';
import { Box, Typography, Paper, Table, TableBody, TableCell, TableContainer, TableHead, TableRow } from '@mui/material';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';
import { api } from '../utils/api';

interface MetricsData {
  [key: string]: number;
}

interface TimeSeriesData {
  date: string;
  value: number;
}

interface SummaryData {
  metrics: MetricsData;
  timeSeries: TimeSeriesData[];
}

const Summary: React.FC = () => {
  const [summaryData, setSummaryData] = React.useState<SummaryData>({
    metrics: {
      'PD Gini': 0,
      'LGD R²': 0,
      'EAD MAPE': 0,
      'Macro R²': 0
    },
    timeSeries: []
  });

  const fetchSummary = async () => {
    try {
      const results = await api.getSummary() as SummaryData;
      setSummaryData({
        metrics: results.metrics || {
          'PD Gini': 0,
          'LGD R²': 0,
          'EAD MAPE': 0,
          'Macro R²': 0
        },
        timeSeries: results.timeSeries || []
      });
    } catch (error) {
      console.error('Summary Fetch Error:', error);
    }
  };

  React.useEffect(() => {
    fetchSummary();
  }, []);

  const metricsData = Object.entries(summaryData.metrics).map(([key, value]) => ({
    name: key,
    value: value
  }));

  return (
    <Paper sx={{ p: 3 }}>
      <Typography variant="h4" gutterBottom>
        Model Performance Summary
      </Typography>

      <TableContainer component={Paper}>
        <Table>
          <TableHead>
            <TableRow>
              <TableCell>Metric</TableCell>
              <TableCell align="right">Value</TableCell>
            </TableRow>
          </TableHead>
          <TableBody>
            {metricsData.map(({ name, value }) => (
              <TableRow key={name}>
                <TableCell>{name}</TableCell>
                <TableCell align="right">{value.toFixed(2)}</TableCell>
              </TableRow>
            ))}
          </TableBody>
        </Table>
      </TableContainer>

      <Typography variant="h6" gutterBottom sx={{ mt: 4 }}>
        Historical Performance
      </Typography>

      <Box sx={{ height: 400 }}>
        <ResponsiveContainer width="100%" height="100%">
          <LineChart data={summaryData.timeSeries}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="date" />
            <YAxis domain={[0, 1]} />
            <Tooltip />
            <Legend />
            <Line type="monotone" dataKey="value" stroke="#8884d8" name="Performance" />
          </LineChart>
        </ResponsiveContainer>
      </Box>
    </Paper>
  );
};

export default Summary;
