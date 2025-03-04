import React, { useState, useEffect } from 'react';
import {
  Box,
  Typography,
  Paper,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  Button,
  IconButton,
  Dialog,
  DialogActions,
  DialogContent,
  DialogContentText,
  DialogTitle,
  TextField,
  Chip,
  Tabs,
  Tab,
  CircularProgress,
  Tooltip,
  Alert,
  Snackbar,
  Checkbox
} from '@mui/material';
import {
  Delete as DeleteIcon,
  Edit as EditIcon,
  Download as DownloadIcon,
  Visibility as ViewIcon,
  Refresh as RefreshIcon
} from '@mui/icons-material';
import { format } from 'date-fns';
import api from '../utils/api';

interface Dataset {
  id: number;
  name: string;
  description: string;
  file_type: string;
  file_size: number;
  upload_date: string;
  is_baseline: boolean;
  column_names: string[];
  row_count: number;
}

interface AnalysisResult {
  id: number;
  dataset_id: number;
  analysis_type: string;
  result_data: any;
  analysis_date: string;
  parameters: any;
}

const formatFileSize = (bytes: number): string => {
  if (bytes === 0) return '0 Bytes';
  const k = 1024;
  const sizes = ['Bytes', 'KB', 'MB', 'GB'];
  const i = Math.floor(Math.log(bytes) / Math.log(k));
  return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
};

const formatDate = (dateString: string): string => {
  try {
    const date = new Date(dateString);
    return format(date, 'MMM d, yyyy h:mm a');
  } catch (e) {
    return dateString;
  }
};

const DatabaseManager: React.FC = () => {
  const [datasets, setDatasets] = useState<Dataset[]>([]);
  const [analysisResults, setAnalysisResults] = useState<AnalysisResult[]>([]);
  const [loading, setLoading] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);
  const [selectedDataset, setSelectedDataset] = useState<Dataset | null>(null);
  const [selectedResult, setSelectedResult] = useState<AnalysisResult | null>(null);
  const [tabValue, setTabValue] = useState<number>(0);
  const [editDialogOpen, setEditDialogOpen] = useState<boolean>(false);
  const [deleteDialogOpen, setDeleteDialogOpen] = useState<boolean>(false);
  const [viewResultDialogOpen, setViewResultDialogOpen] = useState<boolean>(false);
  const [editName, setEditName] = useState<string>('');
  const [editDescription, setEditDescription] = useState<string>('');
  const [editIsBaseline, setEditIsBaseline] = useState<boolean>(false);
  const [snackbarOpen, setSnackbarOpen] = useState<boolean>(false);
  const [snackbarMessage, setSnackbarMessage] = useState<string>('');
  const [snackbarSeverity, setSnackbarSeverity] = useState<'success' | 'error'>('success');
  
  // New state for selections
  const [selectedDatasets, setSelectedDatasets] = useState<number[]>([]);
  const [selectedAnalysisResults, setSelectedAnalysisResults] = useState<number[]>([]);
  const [isAllDatasetsSelected, setIsAllDatasetsSelected] = useState<boolean>(false);
  const [isAllAnalysisResultsSelected, setIsAllAnalysisResultsSelected] = useState<boolean>(false);

  // Fetch datasets and analysis results
  const fetchData = async () => {
    setLoading(true);
    setError(null);
    try {
      // Fetch datasets
      const datasetsData = await api.getAllDatasets() as { datasets: Dataset[] };
      setDatasets(datasetsData.datasets || []);

      // Fetch analysis results
      const resultsData = await api.getAllAnalysisResults() as { analysis_results: AnalysisResult[] };
      setAnalysisResults(resultsData.analysis_results || []);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'An unknown error occurred');
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

      // Log Headers
      console.log('Response Headers:');
      response.headers.forEach((value, key) => console.log(`${key}: ${value}`));

      // Check response headers for content type
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

      // Ensure response is not HTML
      if (contentType.includes('text/html')) {
        const htmlContent = await response.text();
        console.error('Received HTML response:', htmlContent);
        setSnackbarMessage('Failed to download dataset: Received HTML response instead of file');
        setSnackbarSeverity('error');
        setSnackbarOpen(true);
        return;
      }

      // Extract filename from Content-Disposition or fallback
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

      // Fallback if filename is invalid
      if (!downloadFileName || downloadFileName === 'undefined') {
        downloadFileName = `dataset_${datasetId}.${contentType.includes('csv') ? 'csv' : 'xlsx'}`;
        console.warn(`Filename invalid, using fallback: ${downloadFileName}`);
      }

      // Get Blob and trigger download
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
      });
      
      // Show success message
      setSnackbarMessage('Dataset updated successfully');
      setSnackbarSeverity('success');
      setSnackbarOpen(true);
      
      // Refresh data
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
      
      // Show success message
      setSnackbarMessage('Dataset deleted successfully');
      setSnackbarSeverity('success');
      setSnackbarOpen(true);
      
      // Refresh data
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
      
      // Show success message
      setSnackbarMessage('Analysis result deleted successfully');
      setSnackbarSeverity('success');
      setSnackbarOpen(true);
      
      // Refresh data
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

  // New method to handle dataset selection
  const handleDatasetSelect = (datasetId: number) => {
    const isSelected = selectedDatasets.includes(datasetId);
    if (isSelected) {
      setSelectedDatasets(selectedDatasets.filter(id => id !== datasetId));
      setIsAllDatasetsSelected(false);
    } else {
      setSelectedDatasets([...selectedDatasets, datasetId]);
      setIsAllDatasetsSelected(selectedDatasets.length + 1 === datasets.length);
    }
  };

  // New method to handle analysis result selection
  const handleAnalysisResultSelect = (resultId: number) => {
    const isSelected = selectedAnalysisResults.includes(resultId);
    if (isSelected) {
      setSelectedAnalysisResults(selectedAnalysisResults.filter(id => id !== resultId));
      setIsAllAnalysisResultsSelected(false);
    } else {
      setSelectedAnalysisResults([...selectedAnalysisResults, resultId]);
      setIsAllAnalysisResultsSelected(selectedAnalysisResults.length + 1 === analysisResults.length);
    }
  };

  // New method to toggle select all datasets
  const handleSelectAllDatasets = () => {
    if (isAllDatasetsSelected) {
      setSelectedDatasets([]);
      setIsAllDatasetsSelected(false);
    } else {
      const allDatasetIds = datasets.map(dataset => dataset.id);
      setSelectedDatasets(allDatasetIds);
      setIsAllDatasetsSelected(true);
    }
  };

  // New method to toggle select all analysis results
  const handleSelectAllAnalysisResults = () => {
    if (isAllAnalysisResultsSelected) {
      setSelectedAnalysisResults([]);
      setIsAllAnalysisResultsSelected(false);
    } else {
      const allResultIds = analysisResults.map(result => result.id);
      setSelectedAnalysisResults(allResultIds);
      setIsAllAnalysisResultsSelected(true);
    }
  };

  // New method to handle bulk delete of datasets
  const handleBulkDeleteDatasets = async () => {
    try {
      for (const datasetId of selectedDatasets) {
        await api.deleteDataset(datasetId);
      }
      
      setSnackbarMessage(`${selectedDatasets.length} datasets deleted successfully`);
      setSnackbarSeverity('success');
      setSnackbarOpen(true);
      
      // Refresh data and reset selections
      fetchData();
      setSelectedDatasets([]);
      setIsAllDatasetsSelected(false);
    } catch (err) {
      setSnackbarMessage(err instanceof Error ? err.message : 'An error occurred while deleting datasets');
      setSnackbarSeverity('error');
      setSnackbarOpen(true);
    }
  };

  // New method to handle bulk delete of analysis results
  const handleBulkDeleteAnalysisResults = async () => {
    try {
      for (const resultId of selectedAnalysisResults) {
        await api.deleteAnalysisResult(resultId);
      }
      
      setSnackbarMessage(`${selectedAnalysisResults.length} analysis results deleted successfully`);
      setSnackbarSeverity('success');
      setSnackbarOpen(true);
      
      // Refresh data and reset selections
      fetchData();
      setSelectedAnalysisResults([]);
      setIsAllAnalysisResultsSelected(false);
    } catch (err) {
      setSnackbarMessage(err instanceof Error ? err.message : 'An error occurred while deleting analysis results');
      setSnackbarSeverity('error');
      setSnackbarOpen(true);
    }
  };

  return (
    <Box sx={{ p: 3 }}>
      <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 3 }}>
        <Typography variant="h4" component="h1">
          Database Manager
        </Typography>
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
        <Alert severity="error" sx={{ mb: 3 }}>
          {error}
        </Alert>
      )}

      <Paper sx={{ mb: 3 }}>
        <Tabs value={tabValue} onChange={handleTabChange} centered>
          <Tab label="Datasets" />
          <Tab label="Analysis Results" />
        </Tabs>
      </Paper>

      {loading ? (
        <Box sx={{ display: 'flex', justifyContent: 'center', p: 3 }}>
          <CircularProgress />
        </Box>
      ) : (
        <>
          {/* Datasets Tab */}
          {tabValue === 0 && (
            <>
              <Typography variant="h6" sx={{ mb: 2 }}>
                Datasets ({datasets.length})
                {selectedDatasets.length > 0 && (
                  <Button 
                    variant="contained" 
                    color="error" 
                    sx={{ ml: 2 }}
                    onClick={handleBulkDeleteDatasets}
                  >
                    Delete {selectedDatasets.length} Selected
                  </Button>
                )}
              </Typography>
              {datasets.length === 0 ? (
                <Alert severity="info">No datasets found. Upload a dataset to get started.</Alert>
              ) : (
                <TableContainer component={Paper}>
                  <Table>
                    <TableHead>
                      <TableRow>
                        <TableCell padding="checkbox">
                          <Checkbox
                            indeterminate={selectedDatasets.length > 0 && selectedDatasets.length < datasets.length}
                            checked={isAllDatasetsSelected}
                            onChange={handleSelectAllDatasets}
                          />
                        </TableCell>
                        <TableCell>Dataset ID</TableCell>
                        <TableCell>Name</TableCell>
                        <TableCell>File Type</TableCell>
                        <TableCell>Size</TableCell>
                        <TableCell>Upload Date</TableCell>
                        <TableCell>Baseline</TableCell>
                        <TableCell>Actions</TableCell>
                      </TableRow>
                    </TableHead>
                    <TableBody>
                      {datasets.map((dataset) => (
                        <TableRow key={dataset.id}>
                          <TableCell padding="checkbox">
                            <Checkbox
                              checked={selectedDatasets.includes(dataset.id)}
                              onChange={() => handleDatasetSelect(dataset.id)}
                            />
                          </TableCell>
                          <TableCell>{dataset.id}</TableCell>
                          <TableCell>
                            <Tooltip title={dataset.description || 'No description'}>
                              <Typography>{dataset.name}</Typography>
                            </Tooltip>
                          </TableCell>
                          <TableCell>
                            <Chip 
                              label={dataset.file_type.toUpperCase()} 
                              color={
                                dataset.file_type === 'pd' ? 'primary' :
                                dataset.file_type === 'lgd' ? 'secondary' :
                                dataset.file_type === 'ead' ? 'success' :
                                dataset.file_type === 'macro' ? 'warning' : 'default'
                              }
                              size="small"
                            />
                          </TableCell>

                          
                          <TableCell>{formatFileSize(dataset.file_size)}</TableCell>
                          <TableCell>{formatDate(dataset.upload_date)}</TableCell>
                          <TableCell>{dataset.is_baseline ? 'Yes' : 'No'}</TableCell>
                          <TableCell>
                            <IconButton size="small" onClick={() => handleEditClick(dataset)}>
                              <EditIcon fontSize="small" />
                            </IconButton>
                            <IconButton size="small" onClick={() => handleDownloadClick(dataset.id, dataset.name)}>
                              <DownloadIcon fontSize="small" />
                            </IconButton>
                            <IconButton size="small" onClick={() => handleDeleteClick(dataset)}>
                              <DeleteIcon fontSize="small" />
                            </IconButton>
                          </TableCell>
                        </TableRow>
                      ))}
                    </TableBody>
                  </Table>
                </TableContainer>
              )}
            </>
          )}

          {/* Analysis Results Tab */}
          {tabValue === 1 && (
            <>
              <Typography variant="h6" sx={{ mb: 2 }}>
                Analysis Results ({analysisResults.length})
                {selectedAnalysisResults.length > 0 && (
                  <Button 
                    variant="contained" 
                    color="error" 
                    sx={{ ml: 2 }}
                    onClick={handleBulkDeleteAnalysisResults}
                  >
                    Delete {selectedAnalysisResults.length} Selected
                  </Button>
                )}
              </Typography>
              {analysisResults.length === 0 ? (
                <Alert severity="info">No analysis results found. Analyze datasets to generate results.</Alert>
              ) : (
                <TableContainer component={Paper}>
                  <Table>
                    <TableHead>
                      <TableRow>
                        <TableCell padding="checkbox">
                          <Checkbox
                            indeterminate={selectedAnalysisResults.length > 0 && selectedAnalysisResults.length < analysisResults.length}
                            checked={isAllAnalysisResultsSelected}
                            onChange={handleSelectAllAnalysisResults}
                          />
                        </TableCell>
                        <TableCell>Analysis ID</TableCell>
                        <TableCell>Dataset ID</TableCell>
                        <TableCell>Analysis Type</TableCell>
                        <TableCell>Date</TableCell>
                        <TableCell>Actions</TableCell>
                      </TableRow>
                    </TableHead>
                    <TableBody>
                      {analysisResults.map((result) => (
                        <TableRow key={result.id}>
                          <TableCell padding="checkbox">
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
                              color={
                                result.analysis_type === 'pd' ? 'primary' :
                                result.analysis_type === 'lgd' ? 'secondary' :
                                result.analysis_type === 'ead' ? 'success' :
                                result.analysis_type === 'macro' ? 'warning' : 'default'
                              }
                              size="small"
                            />
                          </TableCell>
                          <TableCell>{formatDate(result.analysis_date)}</TableCell>
                          <TableCell>
                            <IconButton size="small" onClick={() => handleViewResultClick(result)}>
                              <ViewIcon fontSize="small" />
                            </IconButton>
                            <IconButton size="small" onClick={() => handleDeleteResult(result.id)}>
                              <DeleteIcon fontSize="small" />
                            </IconButton>
                          </TableCell>
                        </TableRow>
                      ))}
                    </TableBody>
                  </Table>
                </TableContainer>
              )}
            </>
          )}
        </>
      )}

      {/* Edit Dataset Dialog */}
      <Dialog open={editDialogOpen} onClose={() => setEditDialogOpen(false)}>
        <DialogTitle>Edit Dataset</DialogTitle>
        <DialogContent>
          <TextField
            autoFocus
            margin="dense"
            label="Name"
            fullWidth
            value={editName}
            onChange={(e) => setEditName(e.target.value)}
            sx={{ mb: 2 }}
          />
          <TextField
            margin="dense"
            label="Description"
            fullWidth
            multiline
            rows={3}
            value={editDescription}
            onChange={(e) => setEditDescription(e.target.value)}
            sx={{ mb: 2 }}
          />
          <Box sx={{ display: 'flex', alignItems: 'center' }}>
            <Typography variant="body1" sx={{ mr: 2 }}>Baseline Dataset:</Typography>
            <Button
              variant={editIsBaseline ? "contained" : "outlined"}
              color={editIsBaseline ? "primary" : "inherit"}
              onClick={() => setEditIsBaseline(!editIsBaseline)}
            >
              {editIsBaseline ? "Yes" : "No"}
            </Button>
          </Box>
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setEditDialogOpen(false)}>Cancel</Button>
          <Button onClick={handleEditSave} variant="contained">Save</Button>
        </DialogActions>
      </Dialog>

      {/* Delete Dataset Dialog */}
      <Dialog open={deleteDialogOpen} onClose={() => setDeleteDialogOpen(false)}>
        <DialogTitle>Delete Dataset</DialogTitle>
        <DialogContent>
          <DialogContentText>
            Are you sure you want to delete the dataset "{selectedDataset?.name}"? This action cannot be undone.
          </DialogContentText>
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setDeleteDialogOpen(false)}>Cancel</Button>
          <Button onClick={handleDeleteConfirm} color="error" variant="contained">Delete</Button>
        </DialogActions>
      </Dialog>

      {/* View Analysis Result Dialog */}
      <Dialog
        open={viewResultDialogOpen}
        onClose={() => setViewResultDialogOpen(false)}
        maxWidth="md"
        fullWidth
      >
        <DialogTitle>Analysis Result Details</DialogTitle>
        <DialogContent>
          {selectedResult && (
            <Box>
              <Typography variant="subtitle1">Analysis Type: {selectedResult.analysis_type.toUpperCase()}</Typography>
              <Typography variant="subtitle1">Date: {formatDate(selectedResult.analysis_date)}</Typography>
              <Typography variant="subtitle1">Dataset ID: {selectedResult.dataset_id}</Typography>
              
              <Typography variant="h6" sx={{ mt: 2, mb: 1 }}>Parameters:</Typography>
              <Paper sx={{ p: 2, mb: 2, bgcolor: 'background.default' }}>
                <pre style={{ overflow: 'auto', margin: 0 }}>
                  {JSON.stringify(selectedResult.parameters, null, 2)}
                </pre>
              </Paper>
              
              <Typography variant="h6" sx={{ mt: 2, mb: 1 }}>Results:</Typography>
              <Paper sx={{ p: 2, bgcolor: 'background.default' }}>
                <pre style={{ overflow: 'auto', margin: 0, maxHeight: '400px' }}>
                  {JSON.stringify(selectedResult.result_data, null, 2)}
                </pre>
              </Paper>
            </Box>
          )}
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setViewResultDialogOpen(false)}>Close</Button>
        </DialogActions>
      </Dialog>

      {/* Snackbar for notifications */}
      <Snackbar
        open={snackbarOpen}
        autoHideDuration={6000}
        onClose={handleSnackbarClose}
        anchorOrigin={{ vertical: 'bottom', horizontal: 'center' }}
      >
        <Alert onClose={handleSnackbarClose} severity={snackbarSeverity} sx={{ width: '100%' }}>
          {snackbarMessage}
        </Alert>
      </Snackbar>
    </Box>
  );
};

export default DatabaseManager;
