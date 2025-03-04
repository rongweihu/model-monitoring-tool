import React, { useState } from 'react';
import { 
  Box, 
  CssBaseline, 
  AppBar, 
  Toolbar, 
  Typography, 
  Drawer, 
  List, 
  ListItem, 
  ListItemIcon, 
  ListItemText, 
  Link, 
  Divider, 
  IconButton
} from '@mui/material';
import { 
  Info, 
  Upload, 
  Person, 
  FolderOpen, 
  Menu, 
  CreditScore,  
  MonetizationOn,  
  CompareArrows,  
  Public,
  LightMode,
  DarkMode
} from '@mui/icons-material';
import { Link as RouterLink } from 'react-router-dom';
import { createTheme } from '@mui/material/styles';

// Import logos for different themes
import eyLogo from './ey-logo.png';

const drawerWidth = 240;

const menuItems = [
  { text: 'Introduction', icon: <Info />, path: '/introduction' },
  { text: 'Data Upload', icon: <Upload />, path: '/data-upload' },
  { text: 'User Inputs', icon: <Person />, path: '/user-inputs' },
  { text: 'PD Model', icon: <CreditScore />, path: '/pd' },
  { text: 'Macro Model', icon: <Public />, path: '/macro' },
  { text: 'LGD Model', icon: <MonetizationOn />, path: '/lgd' },
  { text: 'EAD Model', icon: <CompareArrows />, path: '/ead' },
  { text: 'Database Manager', icon: <FolderOpen />, path: '/database' },
];

interface LayoutProps {
  children: React.ReactNode;
}

const Layout: React.FC<LayoutProps> = ({ children }) => {
  const [isMenuVisible, setIsMenuVisible] = useState(true);
  const [isDarkMode, setIsDarkMode] = useState(() => {
    // Check local storage for theme preference on initial load
    const savedTheme = localStorage.getItem('appTheme');
    return savedTheme === 'dark';
  });

  const theme = createTheme({
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
  });

  const toggleMenu = () => {
    setIsMenuVisible(!isMenuVisible);
  };

  const toggleTheme = () => {
    const newTheme = !isDarkMode;
    setIsDarkMode(newTheme);
    // Persist theme preference in local storage
    localStorage.setItem('appTheme', newTheme ? 'dark' : 'light');
    // Force a page refresh to ensure all components update their theme
    window.location.reload();
  };

  return (
    <Box sx={{ display: 'flex', position: 'relative', width: '100%', minHeight: '100vh' }}>
      <CssBaseline />
      <AppBar 
        position="fixed" 
        sx={{ 
          width: `calc(100% - ${isMenuVisible ? drawerWidth : 0}px)`, 
          ml: `${isMenuVisible ? drawerWidth : 0}px`,
          backgroundColor: theme.palette.background.default,
          color: theme.palette.text.primary,
          transition: 'width 225ms cubic-bezier(0, 0, 0.2, 1) 0ms, margin-left 225ms cubic-bezier(0, 0, 0.2, 1) 0ms',
          zIndex: (theme) => theme.zIndex.drawer + 1
        }}
      >
        <Toolbar 
          sx={{ 
            display: 'flex',
            justifyContent: 'space-between',
            alignItems: 'center',
            width: '100%',
            px: 2
          }}
        >
          <Box sx={{ display: 'flex', alignItems: 'center', gap: 2 }}>
            <IconButton
              onClick={toggleMenu}
              size="large"
              edge="start"
              color="inherit"
              aria-label="toggle menu"
            >
              <Menu />
            </IconButton>
            <Typography variant="h6" noWrap component="div">
              Model Monitoring Tool
            </Typography>
          </Box>
          <IconButton
            onClick={toggleTheme}
            size="large"
            edge="end"
            color="inherit"
            aria-label="toggle theme"
          >
            {isDarkMode ? <LightMode /> : <DarkMode />}
          </IconButton>
        </Toolbar>
      </AppBar>
      
      <Drawer
        variant="permanent"
        sx={{
          width: isMenuVisible ? drawerWidth : 0,
          flexShrink: 0,
          transition: 'width 225ms cubic-bezier(0, 0, 0.2, 1) 0ms',
          '& .MuiDrawer-paper': {
            width: drawerWidth,
            boxSizing: 'border-box',
            transform: isMenuVisible ? 'none' : `translateX(-${drawerWidth}px)`,
            transition: 'transform 225ms cubic-bezier(0, 0, 0.2, 1) 0ms',
            overflowX: 'hidden',
            backgroundColor: theme.palette.background.paper,
            color: theme.palette.text.primary,
            borderRight: `1px solid ${theme.palette.divider}`
          },
        }}
        anchor="left"
      >
        <Toolbar>
          <Link 
            href="https://www.ey.com" 
            target="_blank" 
            rel="noopener noreferrer"
            sx={{ 
              display: 'flex', 
              alignItems: 'center',
              justifyContent: 'center',
              width: '100%',
              color: theme.palette.text.primary
            }}
          >
            <img 
              src={eyLogo} 
              alt="EY Logo" 
              style={{ 
                height: '50px', 
                maxWidth: '100%', 
                objectFit: 'contain',
                cursor: 'pointer',
                filter: isDarkMode 
                  ? 'invert(1) brightness(1.5) sepia(0.5) hue-rotate(180deg)' 
                  : 'none',
                transition: 'filter 0.3s ease'
              }} 
            />
          </Link>
        </Toolbar>
        <Divider />
        <List>
          {menuItems.map((item) => (
            <ListItem 
              key={item.path} 
              button 
              component={RouterLink} 
              to={item.path}
              sx={{
                '&:hover': {
                  backgroundColor: theme.palette.action.hover
                }
              }}
            >
              <ListItemIcon sx={{ color: theme.palette.text.secondary }}>
                {item.icon}
              </ListItemIcon>
              <ListItemText 
                primary={item.text} 
                primaryTypographyProps={{ 
                  color: theme.palette.text.primary 
                }} 
              />
            </ListItem>
          ))}
        </List>
      </Drawer>

      
      <Box
        component="main"
        sx={{ 
          flexGrow: 1,
          bgcolor: theme.palette.background.default,
          p: 3,
          marginTop: '64px',
          ml: 0,
          width: isMenuVisible ? `calc(100% - ${drawerWidth}px)` : '100%',
          transition: 'all 225ms cubic-bezier(0, 0, 0.2, 1) 0ms'
        }}
      >
        {children}
      </Box>
    </Box>
  );
};

export default Layout;
