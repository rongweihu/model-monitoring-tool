import React, { useState, useEffect } from 'react';
import axios from 'axios';
import {
  Paper, Typography, Box, TextField, Button, Grid,
} from '@mui/material';

// === Interfaces ===
interface ModelCriteria {
  metric: string;
  threshold: number;
  description: string;
}

interface UserThresholdsResponse {
  pdCriteria?: ModelCriteria[];
  macroModelThresholds?: MacroModelThresholds;
  eadThresholds?: ModelCriteria[];
  lgdThresholds?: ModelCriteria[];
}

interface MacroModelThresholds {
  heteroscedasticity_threshold: number;
  rmse_threshold: number;
  r_squared_threshold: number;
  stationarity_threshold: number;
}

// === Utility Components ===
const SectionHeader: React.FC<{ title: string }> = ({ title }) => (
  <Typography
    variant="h6"
    gutterBottom
    sx={{ color: 'primary.dark', mb: 2 }}
  >
    {title}
  </Typography>
);

const ThresholdField: React.FC<{
  label: string;
  value: number;
  onChange: (value: number) => void;
  helperText: string;
  step?: number;
  min?: number;
  max?: number;
}> = ({ label, value, onChange, helperText, step = 0.01, min = 0, max }) => (
  <TextField
    fullWidth
    label={label}
    type="number"
    value={value}
    onChange={(e) => onChange(parseFloat(e.target.value) || 0)}
    helperText={helperText}
    variant="outlined"
    InputProps={{ inputProps: { step, min, max } }}
    sx={{ '& .MuiOutlinedInput-root': { borderRadius: 2 } }}
  />
);

// === Main Component ===
const UserInputs: React.FC = () => {
  const [pdCriteria, setPdCriteria] = useState<ModelCriteria[]>(() => {
    const stored = localStorage.getItem('pdCriteria');
    return stored ? JSON.parse(stored) : [
      { metric: 'Gini Coefficient', threshold: 0.2, description: 'Minimum acceptable Gini value' },
      { metric: 'KS Statistic', threshold: 0.3, description: 'Minimum acceptable KS value' },
    ];
  });

  const [eadThresholds, setEADThresholds] = useState<ModelCriteria[]>(() => {
    const stored = localStorage.getItem('eadThresholds');
    return stored ? JSON.parse(stored) : [
      { metric: 'MAPE', threshold: 0.15, description: 'Maximum acceptable Mean Absolute Percentage Error' },
      { metric: 'R-squared', threshold: 0.8, description: 'Minimum acceptable R-squared value' },
    ];
  });

  const [lgdThresholds, setLGDThresholds] = useState<ModelCriteria[]>(() => {
    const stored = localStorage.getItem('lgdThresholds');
    return stored ? JSON.parse(stored) : [
      { metric: 'MAPE', threshold: 0.15, description: 'Maximum acceptable Mean Absolute Percentage Error' },
      { metric: 'R-squared', threshold: 0.8, description: 'Minimum acceptable R-squared value' },
    ];
  });

  const [macroModelThresholds, setMacroModelThresholds] = useState<MacroModelThresholds>(() => {
    const stored = localStorage.getItem('macroModelThresholds');
    const defaults = {
      heteroscedasticity_threshold: 0.05,
      rmse_threshold: 0.1,
      r_squared_threshold: 0.7,
      stationarity_threshold: 0.05,
    };
    return stored ? { ...defaults, ...JSON.parse(stored) } : defaults;
  });

  // Fetch thresholds from backend
  const fetchUserThresholds = async () => {
    try {
      const response = await axios.get<UserThresholdsResponse>('http://localhost:5000/api/user-thresholds');
      const { pdCriteria, macroModelThresholds, eadThresholds, lgdThresholds } = response.data;

      if (pdCriteria) {
        setPdCriteria(pdCriteria);
        localStorage.setItem('pdCriteria', JSON.stringify(pdCriteria));
      }
      if (macroModelThresholds) {
        setMacroModelThresholds(macroModelThresholds);
        localStorage.setItem('macroModelThresholds', JSON.stringify(macroModelThresholds));
      }
      if (eadThresholds) {
        setEADThresholds(eadThresholds);
        localStorage.setItem('eadThresholds', JSON.stringify(eadThresholds));
      }
      if (lgdThresholds) {
        setLGDThresholds(lgdThresholds);
        localStorage.setItem('lgdThresholds', JSON.stringify(lgdThresholds));
      }
    } catch (error) {
      console.error('Failed to fetch thresholds from backend:', error);
    }
  };

  // Save thresholds to backend and localStorage
  const saveUserThresholds = async () => {
    const payload = {
      pdCriteria,
      macroModelThresholds,
      eadThresholds,
      lgdThresholds,
    };

    const saveToLocalStorage = () => {
      localStorage.setItem('pdCriteria', JSON.stringify(pdCriteria));
      localStorage.setItem('macroModelThresholds', JSON.stringify(macroModelThresholds));
      localStorage.setItem('eadThresholds', JSON.stringify(eadThresholds));
      localStorage.setItem('lgdThresholds', JSON.stringify(lgdThresholds));
    };

    try {
      await axios.post('http://localhost:5000/api/user-thresholds', payload);
      saveToLocalStorage();
      alert('Thresholds saved successfully!');
    } catch {
      saveToLocalStorage();
      alert('Thresholds saved to local storage. Backend sync failed.');
    }
  };

  // Handle threshold changes
  const handleCriteriaChange = (type: 'pd' | 'ead' | 'lgd', index: number, value: string) => {
    const numericValue = parseFloat(value) || 0;
    const updateCriteria = (prev: ModelCriteria[]) => {
      const updated = [...prev];
      updated[index] = { ...updated[index], threshold: numericValue };
      return updated;
    };

    switch (type) {
      case 'pd':
        setPdCriteria(updateCriteria);
        break;
      case 'ead':
        setEADThresholds(updateCriteria);
        break;
      case 'lgd':
        setLGDThresholds(updateCriteria);
        break;
    }
  };

  const handleMacroModelThresholdChange = (field: keyof MacroModelThresholds, value: number) => {
    const updated = { ...macroModelThresholds, [field]: value };
    setMacroModelThresholds(updated);
    localStorage.setItem('macroModelThresholds', JSON.stringify(updated));
    window.dispatchEvent(new Event('macroModelThresholdUpdate'));
  };

  // Render criteria fields
  const renderCriteriaFields = (criteria: ModelCriteria[], type: 'pd' | 'ead' | 'lgd') => (
    criteria.map((criterion, index) => (
      <Grid item xs={12} sm={4} key={criterion.metric}>
        <ThresholdField
          label={criterion.metric}
          value={criterion.threshold}
          onChange={(value) => handleCriteriaChange(type, index, value.toString())}
          helperText={criterion.description}
          max={type === 'pd' ? 1 : undefined}
        />
      </Grid>
    ))
  );

  useEffect(() => {
    fetchUserThresholds();
  }, []);

  return (
    <Paper elevation={3} sx={{ p: 2, bboxShadow: 3, borderRadius: 4 }}>
      <Typography
        variant="h4"
        gutterBottom
        sx={{
          mb: 4,
          textAlign: 'center',
          fontWeight: 600,
          color: 'primary.main',
          borderBottom: '2px solid',
          borderColor: 'primary.main',
          pb: 2,
        }}
      >
        User Input Thresholds
      </Typography>

      <Box sx={{ mb: 4, p: 3, backgroundColor: 'background.paper', borderRadius: 2 }}>
        <SectionHeader title="PD Model Matrices Criteria" />
        <Grid container spacing={2}>
          {renderCriteriaFields(pdCriteria, 'pd')}
        </Grid>
      </Box>

      <Box sx={{ mb: 4, p: 3, backgroundColor: 'background.paper', borderRadius: 2 }}>
        <SectionHeader title="EAD Model Performance Thresholds" />
        <Grid container spacing={2}>
          {renderCriteriaFields(eadThresholds, 'ead')}
        </Grid>
      </Box>

      <Box sx={{ mb: 4, p: 3, backgroundColor: 'background.paper', borderRadius: 2 }}>
        <SectionHeader title="LGD Model Performance Thresholds" />
        <Grid container spacing={2}>
          {renderCriteriaFields(lgdThresholds, 'lgd')}
        </Grid>
      </Box>

      <Box sx={{ mb: 4, p: 3, backgroundColor: 'background.paper', borderRadius: 2 }}>
        <SectionHeader title="Macro Model Thresholds" />
        <Grid container spacing={2}>
          {Object.entries(macroModelThresholds).map(([key, value]) => (
            <Grid item xs={12} sm={4} key={key}>
              <ThresholdField
                label={key.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase())}
                value={value}
                onChange={(val) => handleMacroModelThresholdChange(key as keyof MacroModelThresholds, val)}
                helperText={`Threshold for ${key.replace(/_/g, ' ')}`}
                step={key.includes('threshold') ? 0.01 : 0.1}
                max={key.includes('threshold') ? 1 : undefined}
              />
            </Grid>
          ))}
        </Grid>
      </Box>

      <Box sx={{ display: 'flex', justifyContent: 'center', mt: 4 }}>
        <Button
          variant="contained"
          color="primary"
          onClick={saveUserThresholds}
          sx={{
            px: 4,
            py: 1.5,
            borderRadius: 2,
            fontWeight: 600,
            '&:hover': { transform: 'scale(1.05)', transition: 'transform 0.3s ease' },
          }}
        >
          Save Thresholds
        </Button>
      </Box>
    </Paper>
  );
};

export default UserInputs;