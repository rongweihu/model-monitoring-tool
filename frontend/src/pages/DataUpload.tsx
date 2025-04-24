import React, { useState } from 'react';
import { Paper, Typography, Box, Button, Alert, Table, TableBody, TableCell, TableContainer, TableHead, TableRow, Select, MenuItem, FormControl, InputLabel, TextField, Tabs, Tab } from '@mui/material';
import { Upload as UploadIcon } from '@mui/icons-material';
import { api } from '../utils/api';
import * as XLSX from 'xlsx';

// === Interfaces ===
interface DataPreview {
  [key: string]: any;
}

interface ColumnMapping {
  [key: string]: string;
}

// === Utility Components ===
interface TabPanelProps {
  children?: React.ReactNode;
  index: string;
  value: string;
}

const TabPanel: React.FC<TabPanelProps> = (props) => {
  const { children, value, index, ...other } = props;

  return (
    <div
      role="tabpanel"
      hidden={value !== index}
      id={`simple-tabpanel-${index}`}
      aria-labelledby={`simple-tab-${index}`}
      {...other}
    >
      {value === index && (
        <Box sx={{ p: 3 }}>
          {children}
        </Box>
      )}
    </div>
  );
};

const UploadSection: React.FC<{
  title: string;
  file: File | undefined;
  onFileChange: (file: File) => void;
  onUpload: () => void;
  type: string;
  dataPreview: DataPreview[];
  columns: string[];
  columnMapping: ColumnMapping;
  setColumnMapping: (mapping: ColumnMapping) => void;
  asOfDate: string;
  setAsOfDate: (date: string) => void;
}> = ({ title, file, onFileChange, onUpload, type, dataPreview, columns, columnMapping, setColumnMapping, asOfDate, setAsOfDate }) => {
  const requiredColumns: { [key: string]: { field: string; label: string }[] } = {
    macro: [
      { field: 'Defaultrate', label: 'Actual Default Rate' },
      { field: 'pred_dr', label: 'Predicted Default Rate' },
      { field: 'snapshot_ccyymm', label: 'Date' },
    ],
    pd: [
      { field: 'PD_1_YR', label: 'Predicted Probability of Default at 1Y' },
      { field: 'DEF_FLAG', label: 'Default Flag at 1Y' },
      { field: 'TTCReportingRating', label: 'Credit Rating' },
    ],
    pd_baseline: [
      { field: 'PD_1_YR', label: 'Predicted Probability of Default at 1Y' },
      { field: 'DEF_FLAG', label: 'Actual Default Flag at 1Y' },
      { field: 'TTCReportingRating', label: 'Credit Rating' },
    ],
    lgd: [
      { field: 'actual_lgd', label: 'Actual LGD' },
      { field: 'predicted_lgd', label: 'Predicted LGD' },
      { field: 'Quarter', label: 'Quarter' },
      { field: 'Portfolio', label: 'Portfolio' },
      { field: 'ModelName', label: 'Model Name' },
    ],
    ead: [
      { field: 'Exposure', label: 'Actual EAD' },
      { field: 'predicted_exposure', label: 'Predicted EAD' },
      { field: 'Quarter', label: 'Quarter' },
      { field: 'Portfolio', label: 'Portfolio' },
      { field: 'ModelName', label: 'Model Name' },
    ],
  };

  const handleColumnChange = (field: string, value: string) => {
    setColumnMapping({ ...columnMapping, [field]: value });
  };

  return (
    <Box sx={{ mb: 4 }}>
     
      <TextField
        label="As-of Date"
        type="date"
        value={asOfDate}
        onChange={(e) => setAsOfDate(e.target.value)}
        fullWidth
        sx={{ mb: 2, maxWidth: 300 }}
        InputLabelProps={{ shrink: true }}
        helperText="Enter the date associated with this data (YYYY-MM-DD)"
      />
      <Box sx={{ display: 'flex', alignItems: 'center', gap: 2, mb: 2 }}>
        <Button
          variant="contained"
          component="label"
          startIcon={<UploadIcon />}
        >
          Choose File
          <input
            type="file"
            hidden
            accept=".xlsx,.csv"
            onChange={(e) => e.target.files && onFileChange(e.target.files[0])}
          />
        </Button>
        <Typography>{file?.name || 'No file chosen'}</Typography>
      </Box>
      
      {dataPreview.length > 0 && (
        <>
          <Typography variant="subtitle1" gutterBottom>Data Preview</Typography>
          <TableContainer component={Paper} sx={{ mb: 2, maxHeight: 300, overflow: 'auto' }}>
            <Table stickyHeader>
              <TableHead>
                <TableRow>
                  {columns.map((col) => (
                    <TableCell key={col}>{col}</TableCell>
                  ))}
                </TableRow>
              </TableHead>
              <TableBody>
                {dataPreview.map((row, index) => (
                  <TableRow key={index}>
                    {columns.map((col) => (
                      <TableCell key={col}>{row[col]}</TableCell>
                    ))}
                  </TableRow>
                ))}
              </TableBody>
            </Table>
          </TableContainer>

          <Typography variant="subtitle1" gutterBottom>Select Columns</Typography>
          <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 2, mb: 2 }}>
            {requiredColumns[type].map(({ field, label }) => (
              <FormControl key={field} sx={{ minWidth: 200 }}>
                <InputLabel>{label}</InputLabel>
                <Select
                  value={columnMapping[field] || ''}
                  onChange={(e) => handleColumnChange(field, e.target.value)}
                  label={label}
                >
                  <MenuItem value=""><em>Select column</em></MenuItem>
                  {columns.map((col) => (
                    <MenuItem key={col} value={col}>{col}</MenuItem>
                  ))}
                </Select>
              </FormControl>
            ))}
          </Box>
        </>
      )}

      <Button
        variant="contained"
        color="primary"
        onClick={onUpload}
        disabled={
          !file ||
          requiredColumns[type].some(({ field }) => !(columnMapping[field]))
        }
      >
        Upload
      </Button>
    </Box>
  );
};

// === Main Component ===
const DataUpload: React.FC = () => {
  const [activeTab, setActiveTab] = useState('macro');
  const [files, setFiles] = useState<{
    macro?: File;
    pd_baseline?: File;
    pd?: File;
    lgd?: File;
    ead?: File;
  }>({});
  const [asOfDates, setAsOfDates] = useState<{
    macro: string;
    pd: string;
    pd_baseline: string;
    lgd: string;
    ead: string;
  }>({
    macro: '',
    pd: '',
    pd_baseline: '',
    lgd: '',
    ead: '',
  });
  const [uploadStatus, setUploadStatus] = useState<{
    success?: string;
    error?: string;
  }>({});
  const [dataPreviews, setDataPreviews] = useState<{
    [key: string]: DataPreview[];
  }>({});
  const [columns, setColumns] = useState<{
    [key: string]: string[];
  }>({});
  const [columnMappings, setColumnMappings] = useState<{
    [key: string]: ColumnMapping;
  }>({
    macro: {},
    pd: {},
    pd_baseline: {},
    lgd: {},
    ead: {},
  });

  const handleFileChange = (type: keyof typeof files) => async (file: File) => {
    setFiles((prev) => ({ ...prev, [type]: file }));
    setUploadStatus({});

    try {
      const data = await file.arrayBuffer();
      const workbook = XLSX.read(data, { type: 'array' });
      const sheetName = workbook.SheetNames[0];
      const worksheet = workbook.Sheets[sheetName];
      const jsonData = XLSX.utils.sheet_to_json(worksheet, { header: 1 });

      const headers = jsonData[0] as string[];
      const rows = jsonData.slice(1).map((row: any[]) =>
        headers.reduce((obj, header, i) => ({ ...obj, [header]: row[i] || '' }), {})
      ).slice(0, 5);

      setDataPreviews((prev) => ({ ...prev, [type]: rows }));
      setColumns((prev) => ({ ...prev, [type]: headers }));
      setColumnMappings((prev) => ({ ...prev, [type]: {} }));
    } catch (error) {
      setUploadStatus({
        error: `Error reading file for ${type.toUpperCase()}: ${
          error instanceof Error ? error.message : 'Unknown error'
        }`,
      });
    }
  };

  const handleAsOfDateChange = (type: keyof typeof asOfDates) => (date: string) => {
    setAsOfDates((prev) => ({ ...prev, [type]: date }));
  };

  const handleUpload = async (type: keyof typeof files) => {
    const file = files[type];
    if (!file) {
      setUploadStatus({ error: `No file selected for ${type.toUpperCase()}` });
      return;
    }

    const columnMapping = columnMappings[type];
    if (!columnMapping || Object.keys(columnMapping).length === 0) {
      setUploadStatus({ error: `Column mapping required for ${type.toUpperCase()}` });
      return;
    }

    const formData = new FormData();
    formData.append('file', file);
    formData.append('column_mapping', JSON.stringify(columnMapping));
    if (asOfDates[type]) {
      formData.append('as_of_date', asOfDates[type]);
    }

    console.log(`Uploading ${type} file:`, file.name);
    console.log('FormData contents:');
    for (const [key, value] of formData.entries()) {
      console.log(`${key}:`, value);
    }

    try {
      const response = await api.uploadFile(type, formData);
      setUploadStatus({ success: `${type.toUpperCase()} data uploaded successfully!` });
      setDataPreviews((prev) => ({ ...prev, [type]: [] }));
      setColumns((prev) => ({ ...prev, [type]: [] }));
      setColumnMappings((prev) => ({ ...prev, [type]: {} }));
      setFiles((prev) => ({ ...prev, [type]: undefined }));
      setAsOfDates((prev) => ({ ...prev, [type]: '' }));
    } catch (error) {
      setUploadStatus({
        error: `Error uploading ${type.toUpperCase()} data: ${
          error instanceof Error ? error.message : 'Unknown error'
        }`,
      });
    }
  };

  const uploadSections = [
    { type: 'macro' as const, title: 'Macro Data' },
    { type: 'pd_baseline' as const, title: 'PD Baseline Data' },
    { type: 'pd' as const, title: 'PD Model Data' },
    { type: 'lgd' as const, title: 'LGD Model Data' },
    { type: 'ead' as const, title: 'EAD Model Data' },
  ];

  return (
    <Paper sx={{ p: 3, width: '100%', maxWidth: 'none', margin: 0, boxShadow: 3, borderRadius: 4 }}>
      <Typography variant="h4" gutterBottom>Data Upload</Typography>

      <Tabs
        value={activeTab}
        onChange={(event, newValue) => setActiveTab(newValue)}
        aria-label="data upload tabs"
        sx={{ mb: 3 }}
      >
        {uploadSections.map(({ type, title }) => (
          <Tab key={type} label={title} value={type} />
        ))}
      </Tabs>

      {uploadStatus.success && (
        <Alert severity="success" sx={{ mb: 2 }}>{uploadStatus.success}</Alert>
      )}
      {uploadStatus.error && (
        <Alert severity="error" sx={{ mb: 2 }}>{uploadStatus.error}</Alert>
      )}

      {uploadSections.map(({ type, title }) => (
        <TabPanel key={type} value={activeTab} index={type}>
          <UploadSection
            type={type}
            title={title}
            file={files[type]}
            onFileChange={handleFileChange(type)}
            onUpload={() => handleUpload(type)}
            dataPreview={dataPreviews[type] || []}
            columns={columns[type] || []}
            columnMapping={columnMappings[type]}
            setColumnMapping={(mapping) =>
              setColumnMappings((prev) => ({ ...prev, [type]: mapping }))
            }
            asOfDate={asOfDates[type]}
            setAsOfDate={handleAsOfDateChange(type)}
          />
        </TabPanel>
      ))}
    </Paper>
  );
};

export default DataUpload;
