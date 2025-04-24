import React, { useEffect, useState } from 'react';
import {
  Paper,
  Box,
  Typography,
  Button,
  Tabs,
  Tab,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  IconButton,
  Tooltip,
  Checkbox,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogContentText,
  DialogActions,
  TextField,
  CircularProgress,
  Alert,
  Snackbar,
  Chip,
} from '@mui/material';
import RefreshIcon from '@mui/icons-material/Refresh';
import EditIcon from '@mui/icons-material/Edit';
import DeleteIcon from '@mui/icons-material/Delete';
import DownloadIcon from '@mui/icons-material/Download';
import ViewIcon from '@mui/icons-material/Visibility';
import api from "../utils/api";
import { formatFileSize, formatDate } from "../utils/utils";
import { Dataset, AnalysisResult } from "../utils/api";

// Color mapping for model types
const modelTypeColors: { [key: string]: string } = {
  macro: 'success',
  pd: 'primary',
  lgd: 'warning',
  ead: 'error',
  pd_baseline: 'secondary',
};

const getModelTypeColor = (type: string): string => {
  return modelTypeColors[type.toLowerCase()] || 'default';
};

const DatabaseManager: React.FC = () => {
  const [datasets, setDatasets] = useState<Dataset[]>([]);
  const [analysisResults, setAnalysisResults] = useState<AnalysisResult[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [tabValue, setTabValue] = useState(0);
  const [selectedDataset, setSelectedDataset] = useState<Dataset | null>(null);
  const [editDialogOpen, setEditDialogOpen] = useState(false);
  const [deleteDialogOpen, setDeleteDialogOpen] = useState(false);
  const [viewResultDialogOpen, setViewResultDialogOpen] = useState(false);
  const [selectedResult, setSelectedResult] = useState<AnalysisResult | null>(null);
  const [editName, setEditName] = useState('');
  const [editDescription, setEditDescription] = useState('');
  const [editIsBaseline, setEditIsBaseline] = useState(false);
  const [editAsOfDate, setEditAsOfDate] = useState('');
  const [snackbarOpen, setSnackbarOpen] = useState(false);
  const [snackbarMessage, setSnackbarMessage] = useState('');
  const [snackbarSeverity, setSnackbarSeverity] = useState<'success' | 'error'>('success');
  const [selectedDatasets, setSelectedDatasets] = useState<number[]>([]);
  const [selectedAnalysisResults, setSelectedAnalysisResults] = useState<number[]>([]);
  const [isAllDatasetsSelected, setIsAllDatasetsSelected] = useState(false);
  const [isAllAnalysisResultsSelected, setIsAllAnalysisResultsSelected] = useState(false);

  const fetchData = async () => {
    setLoading(true);
    setError(null);
    try {
      const datasetsData = await api.getAllDatasets() as { datasets: Dataset[] };
      setDatasets(datasetsData.datasets || []);

      const resultsData = await api.getAllAnalysisResults() as { analysis_results: AnalysisResult[] };
      setAnalysisResults(resultsData.analysis_results || []);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'An unknown error occurred');
      setSnackbarMessage('Failed to load data');
      setSnackbarSeverity('error');
      setSnackbarOpen(true);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchData();
  }, []);

  const handleTabChange = (_event: React.SyntheticEvent, newValue: number) => {
    setTabValue(newValue);
  };

  const handleEditClick = (dataset: Dataset) => {
    setSelectedDataset(dataset);
    setEditName(dataset.name);
    setEditDescription(dataset.description || '');
    setEditIsBaseline(dataset.is_baseline);
    setEditAsOfDate(dataset.as_of_date || '');
    setEditDialogOpen(true);
  };

  const handleDeleteClick = (dataset: Dataset) => {
    setSelectedDataset(dataset);
    setDeleteDialogOpen(true);
  };

  const handleViewResultClick = (result: AnalysisResult) => {
    setSelectedResult(result);
    setViewResultDialogOpen(true);
  };

  const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:5000';

  const handleDownloadClick = async (datasetId: number, fileName: string) => {
    try {
      console.log(`Initiating download for dataset ID: ${datasetId}, provided filename: ${fileName}`);

      const response = await fetch(`${API_URL}/api/database/download/${datasetId}`, {
        method: 'GET',
      });

      console.log('Response Headers:');
      response.headers.forEach((value, key) => console.log(`${key}: ${value}`));

      const contentType = response.headers.get('Content-Type') || 'unknown';
      console.log(`Response Content-Type: ${contentType}`);

      if (!response.ok) {
        const responseText = await response.text();
        console.error(`Download failed (status ${response.status}): ${responseText}`);

        if (contentType.includes('text/html')) {
          console.error('Received HTML instead of binary data:', responseText);
          setSnackbarMessage('Failed to download dataset: Received HTML response instead of file');
          setSnackbarSeverity('error');
          setSnackbarOpen(true);
        } else if (contentType.includes('application/json')) {
          const errorData = JSON.parse(responseText);
          console.error('Backend error:', errorData);
          setSnackbarMessage(`Failed to download dataset: ${errorData.error || 'Unknown error'}`);
          setSnackbarSeverity('error');
          setSnackbarOpen(true);
        } else {
          setSnackbarMessage('Failed to download dataset: Unknown error');
          setSnackbarSeverity('error');
          setSnackbarOpen(true);
        }
        return;
      }

      if (contentType.includes('text/html')) {
        const htmlContent = await response.text();
        console.error('Received HTML response:', htmlContent);
        setSnackbarMessage('Failed to download dataset: Received HTML response instead of file');
        setSnackbarSeverity('error');
        setSnackbarOpen(true);
        return;
      }

      const disposition = response.headers.get('Content-Disposition');
      let downloadFileName = fileName;
      if (disposition && disposition.includes('attachment')) {
        const matches = disposition.match(/filename="(.+)"/);
        if (matches && matches[1]) {
          downloadFileName = matches[1];
          console.log(`Using filename from header: ${downloadFileName}`);
        } else {
          console.log(`No filename in header, fallback to: ${downloadFileName}`);
        }
      } else {
        console.log(`No Content-Disposition header, using: ${downloadFileName}`);
      }

      if (!downloadFileName || downloadFileName === 'undefined') {
        downloadFileName = `dataset_${datasetId}.${contentType.includes('csv') ? 'csv' : 'xlsx'}`;
        console.warn(`Filename invalid, using fallback: ${downloadFileName}`);
      }

      const blob = await response.blob();
      console.log(`Blob received: size=${blob.size}, type=${blob.type}`);

      if (blob.size === 0) {
        console.error('Received empty Blob');
        setSnackbarMessage('Downloaded file is empty');
        setSnackbarSeverity('error');
        setSnackbarOpen(true);
        return;
      }

      const url = window.URL.createObjectURL(blob);
      const link = document.createElement('a');
      link.href = url;
      link.setAttribute('download', downloadFileName);
      document.body.appendChild(link);
      link.click();
      document.body.removeChild(link);
      window.URL.revokeObjectURL(url);

      setSnackbarMessage(`Dataset ${downloadFileName} downloaded successfully`);
      setSnackbarSeverity('success');
      setSnackbarOpen(true);
    } catch (error) {
      console.error('Error downloading dataset:', error);
      setSnackbarMessage('Failed to download dataset');
      setSnackbarSeverity('error');
      setSnackbarOpen(true);
    }
  };

  const handleEditSave = async () => {
    if (!selectedDataset) return;

    try {
      await api.updateDataset(selectedDataset.id, {
        name: editName,
        description: editDescription,
        is_baseline: editIsBaseline,
        as_of_date: editAsOfDate,
      });

      setSnackbarMessage('Dataset updated successfully');
      setSnackbarSeverity('success');
      setSnackbarOpen(true);

      fetchData();
    } catch (err) {
      setSnackbarMessage(err instanceof Error ? err.message : 'An unknown error occurred');
      setSnackbarSeverity('error');
      setSnackbarOpen(true);
    } finally {
      setEditDialogOpen(false);
    }
  };

  const handleDeleteConfirm = async () => {
    if (!selectedDataset) return;

    try {
      await api.deleteDataset(selectedDataset.id);

      setSnackbarMessage('Dataset deleted successfully');
      setSnackbarSeverity('success');
      setSnackbarOpen(true);

      fetchData();
    } catch (err) {
      setSnackbarMessage(err instanceof Error ? err.message : 'An unknown error occurred');
      setSnackbarSeverity('error');
      setSnackbarOpen(true);
    } finally {
      setDeleteDialogOpen(false);
    }
  };

  const handleDeleteResult = async (resultId: number) => {
    try {
      await api.deleteAnalysisResult(resultId);

      setSnackbarMessage('Analysis result deleted successfully');
      setSnackbarSeverity('success');
      setSnackbarOpen(true);

      fetchData();
    } catch (err) {
      setSnackbarMessage(err instanceof Error ? err.message : 'An unknown error occurred');
      setSnackbarSeverity('error');
      setSnackbarOpen(true);
    }
  };

  const handleSnackbarClose = (_event?: React.SyntheticEvent | Event, reason?: string) => {
    if (reason === 'clickaway') {
      return;
    }
    setSnackbarOpen(false);
  };

  const handleDatasetSelect = (datasetId: number) => {
    const isSelected = selectedDatasets.includes(datasetId);
    if (isSelected) {
      setSelectedDatasets(selectedDatasets.filter(id => id !== datasetId));
    } else {
      setSelectedDatasets([...selectedDatasets, datasetId]);
    }
  };

  const handleAnalysisResultSelect = (resultId: number) => {
    const isSelected = selectedAnalysisResults.includes(resultId);
    if (isSelected) {
      setSelectedAnalysisResults(selectedAnalysisResults.filter(id => id !== resultId));
    } else {
      setSelectedAnalysisResults([...selectedAnalysisResults, resultId]);
    }
  };

  const handleSelectAllDatasets = () => {
    if (isAllDatasetsSelected) {
      setSelectedDatasets([]);
      setIsAllDatasetsSelected(false);
    } else {
      setSelectedDatasets(datasets.map(dataset => dataset.id));
      setIsAllDatasetsSelected(true);
    }
  };

  const handleSelectAllAnalysisResults = () => {
    if (isAllAnalysisResultsSelected) {
      setSelectedAnalysisResults([]);
      setIsAllAnalysisResultsSelected(false);
    } else {
      setSelectedAnalysisResults(analysisResults.map(result => result.id));
      setIsAllAnalysisResultsSelected(true);
    }
  };

  const handleBulkDeleteDatasets = async () => {
    try {
      for (const datasetId of selectedDatasets) {
        await api.deleteDataset(datasetId);
      }
      setSnackbarMessage(`${selectedDatasets.length} dataset(s) deleted successfully`);
      setSnackbarSeverity('success');
      setSnackbarOpen(true);
      setSelectedDatasets([]);
      setIsAllDatasetsSelected(false);
      fetchData();
    } catch (err) {
      setSnackbarMessage(err instanceof Error ? err.message : 'An unknown error occurred');
      setSnackbarSeverity('error');
      setSnackbarOpen(true);
    }
  };

  const handleBulkDeleteAnalysisResults = async () => {
    try {
      for (const resultId of selectedAnalysisResults) {
        await api.deleteAnalysisResult(resultId);
      }
      setSnackbarMessage(`${selectedAnalysisResults.length} analysis result(s) deleted successfully`);
      setSnackbarSeverity('success');
      setSnackbarOpen(true);
      setSelectedAnalysisResults([]);
      setIsAllAnalysisResultsSelected(false);
      fetchData();
    } catch (err) {
      setSnackbarMessage(err instanceof Error ? err.message : 'An unknown error occurred');
      setSnackbarSeverity('error');
      setSnackbarOpen(true);
    }
  };

  return (
    <Paper sx={{ p: 2, m: 2 }}>
      <Box display="flex" justifyContent="space-between" alignItems="center" mb={2}>
        <Typography variant="h4">Database Manager</Typography>
        <Button
          variant="contained"
          startIcon={<RefreshIcon />}
          onClick={fetchData}
          disabled={loading}
        >
          Refresh
        </Button>
      </Box>

      {error && (
        <Alert severity="error" sx={{ mb: 2 }}>
          {error}
        </Alert>
      )}

      <Tabs value={tabValue} onChange={handleTabChange} sx={{ mb: 2 }}>
        <Tab label="Datasets" />
        <Tab label="Analysis Results" />
      </Tabs>

      {loading ? (
        <Box display="flex" justifyContent="center" my={4}>
          <CircularProgress />
        </Box>
      ) : (
        <>
          {tabValue === 0 && (
            <Box>
              {selectedDatasets.length > 0 && (
                <Box mb={2}>
                  <Button
                    variant="contained"
                    color="error"
                    onClick={handleBulkDeleteDatasets}
                  >
                    Delete Selected ({selectedDatasets.length})
                  </Button>
                </Box>
              )}
              <TableContainer>
                <Table>
                  <TableHead>
                    <TableRow>
                      <TableCell>
                        <Checkbox
                          checked={isAllDatasetsSelected}
                          onChange={handleSelectAllDatasets}
                        />
                      </TableCell>
                      <TableCell>ID</TableCell>
                      <TableCell>Name</TableCell>
                      <TableCell>Description</TableCell>
                      <TableCell>Type</TableCell>
                      <TableCell>Size</TableCell>
                      <TableCell>Rows</TableCell>
                      <TableCell>Upload Date</TableCell>
                      <TableCell>As-of Date</TableCell>
                      <TableCell>Baseline</TableCell>
                      <TableCell>Actions</TableCell>
                    </TableRow>
                  </TableHead>
                  <TableBody>
                    {datasets.map((dataset) => (
                      <TableRow key={dataset.id}>
                        <TableCell>
                          <Checkbox
                            checked={selectedDatasets.includes(dataset.id)}
                            onChange={() => handleDatasetSelect(dataset.id)}
                          />
                        </TableCell>
                        <TableCell>{dataset.id}</TableCell>
                        <TableCell>{dataset.name}</TableCell>
                        <TableCell>{dataset.description || '-'}</TableCell>
                        <TableCell>
                          <Chip
                            label={dataset.file_type.toUpperCase()}
                            color={getModelTypeColor(dataset.file_type)}
                          />
                        </TableCell>
                        <TableCell>{formatFileSize(dataset.file_size)}</TableCell>
                        <TableCell>{dataset.row_count}</TableCell>
                        <TableCell>{formatDate(dataset.upload_date)}</TableCell>
                        <TableCell>{dataset.as_of_date ? formatDate(dataset.as_of_date) : '-'}</TableCell>
                        <TableCell>
                          <Chip
                            label={dataset.is_baseline ? 'Yes' : 'No'}
                            color={dataset.is_baseline ? 'primary' : 'default'}
                          />
                        </TableCell>
                        <TableCell>
                          <Tooltip title="Edit">
                            <IconButton onClick={() => handleEditClick(dataset)}>
                              <EditIcon />
                            </IconButton>
                          </Tooltip>
                          <Tooltip title="Delete">
                            <IconButton onClick={() => handleDeleteClick(dataset)}>
                              <DeleteIcon />
                            </IconButton>
                          </Tooltip>
                          <Tooltip title="Download">
                            <IconButton
                              onClick={() => handleDownloadClick(dataset.id, dataset.name)}
                            >
                              <DownloadIcon />
                            </IconButton>
                          </Tooltip>
                        </TableCell>
                      </TableRow>
                    ))}
                  </TableBody>
                </Table>
              </TableContainer>
            </Box>
          )}

          {tabValue === 1 && (
            <Box>
              {selectedAnalysisResults.length > 0 && (
                <Box mb={2}>
                  <Button
                    variant="contained"
                    color="error"
                    onClick={handleBulkDeleteAnalysisResults}
                  >
                    Delete Selected ({selectedAnalysisResults.length})
                  </Button>
                </Box>
              )}
              <TableContainer>
                <Table>
                  <TableHead>
                    <TableRow>
                      <TableCell>
                        <Checkbox
                          checked={isAllAnalysisResultsSelected}
                          onChange={handleSelectAllAnalysisResults}
                        />
                      </TableCell>
                      <TableCell>ID</TableCell>
                      <TableCell>Dataset ID</TableCell>
                      <TableCell>Analysis Type</TableCell>
                      <TableCell>Analysis Date</TableCell>
                      <TableCell>As-of Date</TableCell>
                      <TableCell>Actions</TableCell>
                    </TableRow>
                  </TableHead>
                  <TableBody>
                    {analysisResults.map((result) => (
                      <TableRow key={result.id}>
                        <TableCell>
                          <Checkbox
                            checked={selectedAnalysisResults.includes(result.id)}
                            onChange={() => handleAnalysisResultSelect(result.id)}
                          />
                        </TableCell>
                        <TableCell>{result.id}</TableCell>
                        <TableCell>{result.dataset_id}</TableCell>
                        <TableCell>
                          <Chip
                            label={result.analysis_type.toUpperCase()}
                            color={getModelTypeColor(result.analysis_type)}
                          />
                        </TableCell>
                        <TableCell>{formatDate(result.created_at)}</TableCell>
                        <TableCell>{result.as_of_date ? formatDate(result.as_of_date) : '-'}</TableCell>
                        <TableCell>
                          <Tooltip title="View Result">
                            <IconButton onClick={() => handleViewResultClick(result)}>
                              <ViewIcon />
                            </IconButton>
                          </Tooltip>
                          <Tooltip title="Delete">
                            <IconButton onClick={() => handleDeleteResult(result.id)}>
                              <DeleteIcon />
                            </IconButton>
                          </Tooltip>
                        </TableCell>
                      </TableRow>
                    ))}
                  </TableBody>
                </Table>
              </TableContainer>
            </Box>
          )}
        </>
      )}

      {/* Edit Dialog */}
      <Dialog open={editDialogOpen} onClose={() => setEditDialogOpen(false)}>
        <DialogTitle>Edit Dataset</DialogTitle>
        <DialogContent>
          <TextField
            margin="dense"
            label="Name"
            fullWidth
            value={editName}
            onChange={(e) => setEditName(e.target.value)}
          />
          <TextField
            margin="dense"
            label="Description"
            fullWidth
            multiline
            rows={4}
            value={editDescription}
            onChange={(e) => setEditDescription(e.target.value)}
          />
          <TextField
            margin="dense"
            label="As-of Date"
            type="date"
            fullWidth
            value={editAsOfDate}
            onChange={(e) => setEditAsOfDate(e.target.value)}
            InputLabelProps={{ shrink: true }}
          />
          <Box mt={2}>
            <Checkbox
              checked={editIsBaseline}
              onChange={(e) => setEditIsBaseline(e.target.checked)}
            />
            <Typography component="span">Is Baseline</Typography>
          </Box>
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setEditDialogOpen(false)}>Cancel</Button>
          <Button onClick={handleEditSave} variant="contained">Save</Button>
        </DialogActions>
      </Dialog>

      {/* Delete Dialog */}
      <Dialog open={deleteDialogOpen} onClose={() => setDeleteDialogOpen(false)}>
        <DialogTitle>Confirm Delete</DialogTitle>
        <DialogContent>
          <DialogContentText>
            Are you sure you want to delete the dataset "{selectedDataset?.name}"? This action cannot be undone.
          </DialogContentText>
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setDeleteDialogOpen(false)}>Cancel</Button>
          <Button onClick={handleDeleteConfirm} color="error" variant="contained">
            Delete
          </Button>
        </DialogActions>
      </Dialog>

      {/* View Result Dialog */}
      <Dialog
        open={viewResultDialogOpen}
        onClose={() => setViewResultDialogOpen(false)}
        maxWidth="md"
        fullWidth
      >
        <DialogTitle>Analysis Result</DialogTitle>
        <DialogContent>
          {selectedResult && (
            <Box>
              <Typography variant="h6">Result ID: {selectedResult.id}</Typography>
              <Typography>Dataset ID: {selectedResult.dataset_id}</Typography>
              <Typography>Analysis Type: {selectedResult.analysis_type.toUpperCase()}</Typography>
              <Typography>Analysis Date: {formatDate(selectedResult.created_at)}</Typography>
              <Typography>As-of Date: {selectedResult.as_of_date ? formatDate(selectedResult.as_of_date) : '-'}</Typography>
              <Typography variant="h6" mt={2}>Parameters:</Typography>
              <pre>{JSON.stringify(selectedResult.parameters, null, 2)}</pre>
              <Typography variant="h6" mt={2}>Result Data:</Typography>
              <pre>{JSON.stringify(selectedResult.result_data, null, 2)}</pre>
            </Box>
          )}
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setViewResultDialogOpen(false)}>Close</Button>
        </DialogActions>
      </Dialog>

      {/* Snackbar for Notifications */}
      <Snackbar
        open={snackbarOpen}
        autoHideDuration={6000}
        onClose={handleSnackbarClose}
      >
        <Alert
          onClose={handleSnackbarClose}
          severity={snackbarSeverity}
          sx={{ width: '100%' }}
        >
          {snackbarMessage}
        </Alert>
      </Snackbar>
    </Paper>
  );
};

export default DatabaseManager;