import React, { useState } from 'react';
import { Paper, Typography, Box, Button, Alert } from '@mui/material';
import { Upload as UploadIcon } from '@mui/icons-material';
import { api } from '../utils/api';

// === Utility Components ===
const UploadSection: React.FC<{
  type: 'macro' | 'pd' | 'lgd' | 'ead' | 'pd_baseline';
  title: string;
  file: File | undefined;
  onFileChange: (file: File) => void;
  onUpload: () => void;
}> = ({ type, title, file, onFileChange, onUpload }) => (
  <Box sx={{ mb: 4 }}>
    <Typography variant="h6" gutterBottom>{title}</Typography>
    <Box sx={{ display: 'flex', alignItems: 'center', gap: 2 }}>
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
      <Button
        variant="contained"
        color="primary"
        onClick={onUpload}
        disabled={!file}
      >
        Upload
      </Button>
    </Box>
  </Box>
);

// === Main Component ===
const DataUpload: React.FC = () => {
  const [files, setFiles] = useState<{
    macro?: File;
    pd?: File;
    lgd?: File;
    ead?: File;
    pd_baseline?: File;
  }>({});
  const [uploadStatus, setUploadStatus] = useState<{
    success?: string;
    error?: string;
  }>({});

  const handleFileChange = (type: keyof typeof files) => (file: File) => {
    setFiles(prev => ({ ...prev, [type]: file }));
  };

  const handleUpload = async (type: keyof typeof files) => {
    const file = files[type];
    if (!file) return;

    const formData = new FormData();
    formData.append('file', file);

    try {
      await api.uploadFile(type, file);
      setUploadStatus({ success: `${type.toUpperCase()} data uploaded successfully!` });
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
    { type: 'pd' as const, title: 'PD Model Data' },
    { type: 'pd_baseline' as const, title: 'PD Baseline Data (for PSI/CSI)' },
    { type: 'lgd' as const, title: 'LGD Model Data' },
    { type: 'ead' as const, title: 'EAD Model Data' },
  ];

  return (
    <Paper sx={{ p: 3, width: '100%', maxWidth: 'none', margin: 0, boxShadow: 3, borderRadius: 4 }}>
      <Typography variant="h4" gutterBottom>Data Upload</Typography>

      {uploadStatus.success && (
        <Alert severity="success" sx={{ mb: 2 }}>{uploadStatus.success}</Alert>
      )}
      {uploadStatus.error && (
        <Alert severity="error" sx={{ mb: 2 }}>{uploadStatus.error}</Alert>
      )}

      {uploadSections.map(({ type, title }) => (
        <UploadSection
          key={type}
          type={type}
          title={title}
          file={files[type]}
          onFileChange={handleFileChange(type)}
          onUpload={() => handleUpload(type)}
        />
      ))}
    </Paper>
  );
};

export default DataUpload;