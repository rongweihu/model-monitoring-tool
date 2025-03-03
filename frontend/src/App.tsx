import { BrowserRouter as Router, Route, Routes, Link as RouterLink } from 'react-router-dom';
import { Box, CssBaseline, ThemeProvider, createTheme } from '@mui/material';
import { useState, useMemo } from 'react';

import Summary from './pages/Summary';
import PDModel from './pages/PDModel';
import LGDModel from './pages/LGDModel';
import EADModel from './pages/EADModel';
import MacroModel from './pages/MacroModel';

import DataUpload from './pages/DataUpload';
import UserInputs from './pages/UserInputs';
import Introduction from './pages/Introduction';
import DatabaseManager from './pages/DatabaseManager';
import Layout from './components/Layout'; // Correct path to Layout component

function App() {
  const [isDarkMode, setIsDarkMode] = useState(() => {
    const savedTheme = localStorage.getItem('appTheme');
    return savedTheme === 'dark';
  });

  const theme = useMemo(() => 
    createTheme({
      palette: {
        mode: isDarkMode ? 'dark' : 'light',
        primary: {
          main: isDarkMode ? '#90caf9' : '#1976d2',
        },
        background: {
          default: isDarkMode ? '#121212' : '#f4f4f4',
          paper: isDarkMode ? '#1d1d1d' : '#ffffff',
        },
      },
    }),
    [isDarkMode]
  );

  const toggleTheme = () => {
    const newTheme = !isDarkMode;
    setIsDarkMode(newTheme);
    localStorage.setItem('appTheme', newTheme ? 'dark' : 'light');
  };

  return (
    <ThemeProvider theme={theme}>
      <Router>
        <Box sx={{ display: 'flex' }}>
          <CssBaseline />
          <Layout>
            <Routes>
              <Route path="/" element={<Summary />} />
              <Route path="/introduction" element={<Introduction />} />
              <Route path="/data-upload" element={<DataUpload />} />
              <Route path="/user-inputs" element={<UserInputs />} />
              <Route path="/pd" element={<PDModel />} />
              <Route path="/lgd" element={<LGDModel />} />
              <Route path="/ead" element={<EADModel />} />
              <Route path="/macro" element={<MacroModel />} />
              <Route path="/database" element={<DatabaseManager />} />
            </Routes>
          </Layout>
        </Box>
      </Router>
    </ThemeProvider>
  );
}

export default App;
