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
  macroThresholds?: ModelCriteria[];
  eadThresholds?: ModelCriteria[];
}

interface MacroModelThresholds {
  normality_threshold: number;
  autocorrelation_threshold: number;
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
  const [pdCriteria, setPdCriteria] = useState<ModelCriteria[]>([
    { metric: 'Gini Coefficient', threshold: 0.2, description: 'Minimum acceptable Gini value' },
    { metric: 'KS Statistic', threshold: 0.3, description: 'Minimum acceptable KS value' },
    { metric: 'PSI', threshold: 0.1, description: 'Maximum acceptable PSI value' },
  ]);

  const [macroThresholds, setMacroThresholds] = useState<ModelCriteria[]>([
    { metric: 'R-squared', threshold: 0.7, description: 'Minimum R-squared value' },
    { metric: 'P-value', threshold: 0.05, description: 'Maximum p-value for significance' },
  ]);

  const [eadThresholds, setEADThresholds] = useState<ModelCriteria[]>([
    { metric: 'MAPE', threshold: 15.0, description: 'Maximum acceptable Mean Absolute Percentage Error' },
    { metric: 'R-squared', threshold: 0.8, description: 'Minimum acceptable R-squared value' },
  ]);

  const [lgdThresholds, setLgdThresholds] = useState<{
    mape: number;
    rSquared: number;
  }>(() => {
    const savedMAPE = localStorage.getItem('lgdMAPEThreshold');
    const savedRSquared = localStorage.getItem('lgdRSquaredThreshold');
    return {
      mape: savedMAPE ? parseFloat(savedMAPE) : 20,
      rSquared: savedRSquared ? parseFloat(savedRSquared) : 0.7,
    };
  });

  const [macroModelThresholds, setMacroModelThresholds] = useState<MacroModelThresholds>(() => {
    const stored = localStorage.getItem('macroModelThresholds');
    const defaults = {
      normality_threshold: 0.05,
      autocorrelation_threshold: 1.5,
      heteroscedasticity_threshold: 0.05,
      rmse_threshold: 0.1,
      r_squared_threshold: 0.7,
      stationarity_threshold: 0.05,
    };
    return stored ? { ...defaults, ...JSON.parse(stored) } : defaults;
  });

  // Fetch thresholds from backend or localStorage
  const fetchUserThresholds = async () => {
    const loadFromLocalStorage = () => {
      const localPd = localStorage.getItem('pdCriteria');
      const localMacro = localStorage.getItem('macroThresholds');
      const localEAD = localStorage.getItem('eadThresholds');
      if (localPd) setPdCriteria(JSON.parse(localPd));
      if (localMacro) setMacroThresholds(JSON.parse(localMacro));
      if (localEAD) {
        const parsedEAD = JSON.parse(localEAD).filter((t: ModelCriteria) => ['MAPE', 'R-squared'].includes(t.metric));
        setEADThresholds(parsedEAD.length > 0 ? parsedEAD : eadThresholds);
      }
    };

    try {
      const response = await axios.get<UserThresholdsResponse>('http://localhost:5000/api/user-thresholds');
      const { pdCriteria, macroThresholds, eadThresholds } = response.data;

      if (pdCriteria) {
        setPdCriteria(pdCriteria);
        localStorage.setItem('pdCriteria', JSON.stringify(pdCriteria));
      }
      if (macroThresholds) {
        setMacroThresholds(macroThresholds);
        localStorage.setItem('macroThresholds', JSON.stringify(macroThresholds));
      }
      if (eadThresholds) {
        const filteredEAD = eadThresholds.filter(t => ['MAPE', 'R-squared'].includes(t.metric));
        setEADThresholds(filteredEAD.length > 0 ? filteredEAD : eadThresholds);
        localStorage.setItem('eadThresholds', JSON.stringify(filteredEAD.length > 0 ? filteredEAD : eadThresholds));
      }
    } catch {
      loadFromLocalStorage(); // Fallback to localStorage
    }
  };

  // Save thresholds to backend and localStorage
  const saveUserThresholds = async () => {
    const payload = {
      pdCriteria,
      macroThresholds,
      eadThresholds: eadThresholds.filter(t => ['MAPE', 'R-squared'].includes(t.metric)),
    };

    const saveToLocalStorage = () => {
      localStorage.setItem('pdCriteria', JSON.stringify(pdCriteria));
      localStorage.setItem('macroThresholds', JSON.stringify(macroThresholds));
      localStorage.setItem('eadThresholds', JSON.stringify(eadThresholds));
      window.dispatchEvent(new StorageEvent('storage', { key: 'eadThresholds', newValue: JSON.stringify(eadThresholds) }));
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
  const handleCriteriaChange = (type: 'pd' | 'macro' | 'ead', index: number, value: string) => {
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
      case 'macro':
        setMacroThresholds(updateCriteria);
        break;
      case 'ead':
        setEADThresholds(updateCriteria);
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
  const renderCriteriaFields = (criteria: ModelCriteria[], type: 'pd' | 'macro' | 'ead') => (
    criteria.map((criterion, index) => (
      <Grid item xs={12} sm={4} key={criterion.metric}>
        <ThresholdField
          label={criterion.metric}
          value={criterion.threshold}
          onChange={(value) => handleCriteriaChange(type, index, value.toString())}
          helperText={criterion.description}
          max={type === 'pd' || type === 'macro' ? 1 : undefined}
        />
      </Grid>
    ))
  );

  useEffect(() => {
    fetchUserThresholds();
  }, []);

  useEffect(() => {
    localStorage.setItem('lgdMAPEThreshold', lgdThresholds.mape.toString());
    localStorage.setItem('lgdRSquaredThreshold', lgdThresholds.rSquared.toString());
    window.dispatchEvent(new Event('lgdThresholdUpdate'));
  }, [lgdThresholds]);

  return (
    <Paper elevation={3} sx={{ p: 2, bboxShadow: 3, borderRadius: 4  }}>
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
        <SectionHeader title="LGD Model Thresholds" />
        <Grid container spacing={2}>
          <Grid item xs={12} sm={6}>
            <ThresholdField
              label="MAPE Threshold (%)"
              value={lgdThresholds.mape}
              onChange={(value) => setLgdThresholds(prev => ({ ...prev, mape: value }))}
              helperText="Maximum acceptable Mean Absolute Percentage Error"
              step={0.1}
              max={100}
            />
          </Grid>
          <Grid item xs={12} sm={6}>
            <ThresholdField
              label="R-squared Threshold"
              value={lgdThresholds.rSquared}
              onChange={(value) => setLgdThresholds(prev => ({ ...prev, rSquared: value }))}
              helperText="Minimum acceptable R-squared value"
              max={1}
            />
          </Grid>
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