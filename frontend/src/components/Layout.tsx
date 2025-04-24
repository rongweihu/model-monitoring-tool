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
  ListItemButton,
  ListItemIcon, 
  ListItemText, 
  Link, 
  Divider, 
  IconButton,
  ThemeProvider
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
  Dashboard,  
  Public,
  LightMode,
  DarkMode,
  Storage
} from '@mui/icons-material';
import { Link as RouterLink, useLocation } from 'react-router-dom';
import { createTheme } from '@mui/material/styles';

const drawerWidth = 240;

const menuItems = [
  { text: 'Introduction', icon: <Info />, path: '/introduction' },
  { text: 'Data Upload', icon: <Upload />, path: '/data-upload' },
  { text: 'User Inputs', icon: <Person />, path: '/user-inputs' },
  { text: 'PD Model', icon: <CreditScore />, path: '/pd' },
  { text: 'Macro Model', icon: <Public />, path: '/macro' },
  { text: 'LGD Model', icon: <MonetizationOn />, path: '/lgd' },
  { text: 'EAD Model', icon: <CompareArrows />, path: '/ead' },
  { text: 'Summary Dashboard', icon: <Dashboard />, path: '/summary/history' },
  { text: 'Database Manager', icon: <Storage />, path: '/database' },
];

interface LayoutProps {
  children: React.ReactNode;
}

const Layout: React.FC<LayoutProps> = ({ children }) => {
  const [isMenuVisible, setIsMenuVisible] = useState(true);
  const [isDarkMode, setIsDarkMode] = useState(() => {
    const savedTheme = localStorage.getItem('appTheme');
    return savedTheme === 'dark';
  });
  const location = useLocation();

  const theme = createTheme({
    palette: {
      mode: isDarkMode ? 'dark' : 'light',
      primary: {
        main: isDarkMode ? '#FFE600' : '#775C01', // EY Yellow
      },
      secondary: {
        main: isDarkMode ? '#6787f0' : '#D5D5A5',
      },
      background: {
        default: isDarkMode ? '#2E2E38' : '#F5F5F5',
        paper: isDarkMode ? '#1A1A24' : '#FFFFFF',
      },
      text: {
        primary: isDarkMode ? '#FFFFFF' : '#252525',
        secondary: isDarkMode ? '#B0B0B0' : '#666666',
      },
    },
    typography: {
      fontFamily: '"EYInterstate", "Arial", "Helvetica", sans-serif',
      h6: {
        fontWeight: 700,
      },
      body1: {
        fontSize: '1rem',
      },
    },
    components: {
      MuiAppBar: {
        styleOverrides: {
          root: {
            boxShadow: '0 2px 4px rgba(0,0,0,0.1)',
            backgroundColor: isDarkMode ? '#1A1A24' : '#FFFFFF',
          },
        },
      },
      MuiDrawer: {
        styleOverrides: {
          paper: {
            borderRight: 'none',
            boxShadow: '2px 0 4px rgba(0,0,0,0.1)',
            backgroundColor: isDarkMode ? '#1A1A24' : '#FFFFFF',
          },
        },
      },
      MuiListItemButton: {
        styleOverrides: {
          root: {
            '&.Mui-selected': {
              backgroundColor: isDarkMode ? '#FFE600' : '#FFE600',
              color: isDarkMode ? '#1A1A24' : '#252525',
              '& .MuiListItemIcon-root': {
                color: isDarkMode ? '#1A1A24' : '#252525',
              },
            },
            '&:hover': {
              backgroundColor: isDarkMode ? '#33333D' : '#F0F0F0',
            },
          },
        },
      },
    },
  });

  const toggleMenu = () => {
    setIsMenuVisible(!isMenuVisible);
  };

  const toggleTheme = () => {
    const newTheme = !isDarkMode;
    setIsDarkMode(newTheme);
    localStorage.setItem('appTheme', newTheme ? 'dark' : 'light');
  };

  return (
    <ThemeProvider theme={theme}>
      <Box sx={{ display: 'flex', position: 'relative', width: '100%', minHeight: '100vh' }}>
        <CssBaseline />
        <AppBar 
          position="fixed" 
          sx={{ 
            width: `calc(100% - ${isMenuVisible ? drawerWidth : 0}px)`, 
            ml: `${isMenuVisible ? drawerWidth : 0}px`,
            color: theme.palette.text.primary,
            transition: 'width 225ms cubic-bezier(0, 0, 0.2, 1) 0ms, margin-left 225ms cubic-bezier(0, 0, 0.2, 1) 0ms',
            zIndex: (theme) => theme.zIndex.drawer + 1,
          }}
        >
          <Toolbar 
            sx={{ 
              display: 'flex',
              justifyContent: 'space-between',
              alignItems: 'center',
              width: '100%',
              px: 2,
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
              color: theme.palette.text.primary,
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
                color: theme.palette.text.primary,
              }}
            >
              <svg
                width="150"
                height="80"
                viewBox="0 0 1001 403"
                fill="none"
                xmlns="http://www.w3.org/2000/svg"
                xmlSpace="preserve"
                id="ey-icon-main"
                style={{
                  maxWidth: '100%',
                  cursor: 'pointer',
                  transition: 'all 0.3s ease',
                }}
              >
                <path
                  d="m267.91 202.77-33.72 64.77-33.63-64.77h-65.82l69.35 120.09v78.4h59.35v-78.4l69.45-120.09h-64.98Z"
                  fill={isDarkMode ? '#FFFFFF' : '#1a1a24'}
                />
                <path
                  d="M392.42 0 0 143.22 392.42 73.9V0Z"
                  fill={'#ffe600'}
                />
                <path
                  d="M3.43 401.26h158.8v-45.65H62.96v-32.75h71.78v-41.68H62.96v-32.76h79.41l-26.35-45.65H3.43v198.49Z"
                  fill={isDarkMode ? '#FFFFFF' : '#1a1a24'}
                />
                <mask id="ey_icon_fromEY_new_svg__a" maskUnits="userSpaceOnUse" x="392" y="334" width="608" height="68">
                  <path d="M1000 334.54H392.42v66.72H1000v-66.72Z" fill="#fff" />
                </mask>
                <g mask="url(#ey_icon_fromEY_new_svg__a)" fill={isDarkMode ? '#FFFFFF' : '#1a1a24'}>
                  <path d="M865.51 378.72c0 1.43-.1 3.23-.19 3.9h-27.2c.48 6.28 4.76 8.85 9.61 8.85 2.85 0 5.52-.86 7.8-3.14l7.99 6.75c-4.18 5.23-10.56 7.13-16.26 7.13-13.13 0-20.64-10.08-20.64-23.3 0-14.36 8.85-23.49 19.88-23.49 11.79 0 19.02 10.37 19.02 23.3h-.01Zm-27.2-4.85h15.98c-.38-5.04-3.61-8.47-8.18-8.47-5.42 0-7.51 4.76-7.8 8.47ZM1000 378.72c0 1.43-.1 3.23-.19 3.9h-27.2c.48 6.28 4.76 8.85 9.61 8.85 2.85 0 5.52-.86 7.8-3.14l7.99 6.75c-4.18 5.23-10.56 7.13-16.26 7.13-13.13 0-20.64-10.08-20.64-23.3 0-14.36 8.85-23.49 19.88-23.49 11.79 0 19.02 10.37 19.02 23.3h-.01Zm-27.2-4.85h15.98c-.38-5.04-3.61-8.47-8.18-8.47-5.42 0-7.51 4.76-7.8 8.47ZM950.16 385.29l7.42 7.04c-3.8 4.85-9.04 9.89-17.88 9.89-12.27 0-21.31-9.8-21.31-23.3 0-12.36 7.51-23.49 21.5-23.49 7.99 0 13.51 3.61 17.6 9.7l-7.61 7.7c-2.66-3.52-5.52-6.28-10.08-6.28-6.18 0-9.42 5.23-9.42 12.17 0 6.47 2.85 12.27 9.51 12.27 4.09 0 7.51-2.19 10.27-5.71v.01ZM911.54 401.26h-11.6v-24.44c0-6.28-1.43-10.46-7.7-10.46-5.9 0-7.8 3.52-7.8 10.18v24.73h-11.6v-44.89h11.6v3.04c2.38-2.38 5.9-3.99 10.94-3.99 12.17 0 16.17 9.42 16.17 20.45v25.4l-.01-.02ZM766.4 352.47a6.62 6.62 0 0 0 6.66-6.66 6.62 6.62 0 0 0-6.66-6.66 6.62 6.62 0 0 0-6.66 6.66 6.62 6.62 0 0 0 6.66 6.66ZM469.84 352.47a6.62 6.62 0 0 0 6.66-6.66 6.62 6.62 0 0 0-6.66-6.66 6.62 6.62 0 0 0-6.66 6.66 6.62 6.62 0 0 0 6.66 6.66ZM808.06 387.38v-17.12c-2.47-2.66-4.85-3.9-8.08-3.9-6.75 0-8.37 5.33-8.37 11.7 0 7.23 2.09 13.22 8.66 13.22 3.23 0 5.52-1.43 7.8-3.9h-.01Zm11.61 13.88h-11.6v-2.95c-3.8 2.76-6.09 3.9-10.18 3.9-12.94 0-18.45-11.22-18.45-23.78 0-13.6 6.47-23.02 18.17-23.02 3.9 0 7.61 1.05 10.46 3.61v-14.36l11.6-5.8v62.4ZM772.11 356.37h-11.6v44.89h11.6v-44.89ZM753.75 340.39v9.32c-1.71-.57-3.99-.86-5.71-.86-3.33 0-4.85 1.05-4.85 3.9v3.61h9.61v10.84h-9.61v34.05h-11.6V367.2h-6.28v-10.84h6.28v-5.42c0-8.37 5.23-11.79 13.7-11.79 2.47 0 6.09.29 8.47 1.24h-.01ZM719.61 401.26h-11.6v-24.44c0-6.28-1.43-10.46-7.7-10.46-5.9 0-7.8 3.52-7.8 10.18v24.73h-11.6v-44.89h11.6v3.04c2.38-2.38 5.9-3.99 10.94-3.99 12.17 0 16.17 9.42 16.17 20.45v25.4l-.01-.02ZM652.93 355.42c-12.94 0-21.02 10.18-21.02 23.4 0 13.89 8.85 23.4 21.02 23.4s21.02-9.51 21.02-23.4c0-13.89-8.08-23.4-21.02-23.4Zm0 35.57c-7.42 0-9.04-7.13-9.04-12.17 0-6.94 2.85-12.27 9.04-12.27 6.19 0 9.04 5.33 9.04 12.27 0 5.04-1.62 12.17-9.04 12.17ZM620.69 385.29l7.42 7.04c-3.8 4.85-9.04 9.89-17.88 9.89-12.27 0-21.31-9.8-21.31-23.3 0-12.36 7.51-23.49 21.5-23.49 7.99 0 13.51 3.61 17.6 9.7l-7.61 7.7c-2.66-3.52-5.52-6.28-10.08-6.28-6.18 0-9.42 5.23-9.42 12.17 0 6.47 2.85 12.27 9.51 12.27 4.09 0 7.51-2.19 10.27-5.71v.01ZM560.67 401.26h-11.6v-24.44c0-6.28-1.43-10.46-7.7-10.46-5.9 0-7.89 3.52-7.89 10.18v24.73h-11.6v-56.59l11.6-5.8v20.54c2.19-2.66 6.75-3.99 11.22-3.99 11.79 0 15.98 9.32 15.98 20.45v25.4l-.01-.02ZM514.73 387.76l-1.71 11.7c-2.38 1.9-8.08 2.76-11.13 2.76-7.04 0-12.08-5.61-12.08-13.41v-21.59h-7.8v-10.84h7.8v-11.7l11.6-5.8v17.5h13.13v10.84h-13.13v18.45c0 4.09 1.52 5.42 4.28 5.42 2.76 0 7.04-1.43 9.04-3.33ZM475.64 356.37h-11.6v44.89h11.6v-44.89ZM457.95 356.37l-14.17 44.89h-10.65l-7.61-27.29-7.71 27.29h-10.65l-14.08-44.89h12.94l6.75 26.06 7.61-26.06h10.56l7.61 26.06 6.85-26.06h12.55Z" />
                </g>
                <mask id="ey_icon_fromEY_new_svg__b" maskUnits="userSpaceOnUse" x="392" y="248" width="608" height="80">
                  <path d="M1000 248.61H392.42v78.66H1000v-78.66Z" fill="#fff" />
                </mask>
                <g mask="url(#ey_icon_fromEY_new_svg__b)" fill={isDarkMode ? '#FFFFFF' : '#1a1a24'}>
                  <path d="M846.68 310.72h-11.03v-3.14c-2.47 2.76-6.47 4.09-10.46 4.09-11.6 0-15.98-8.37-15.98-20.45v-25.4h11.22v24.44c0 6.09 1.24 10.65 7.51 10.65s7.51-4.76 7.51-10.27v-24.82h11.22v44.89l.01.01ZM881.68 297.59 880.06 309c-2.28 1.81-6.47 2.66-9.23 2.66-6.85 0-12.17-5.14-12.17-13.32v-21.78h-6.47v-10.75h6.47v-11.7l11.22-5.71v17.41h10.65v10.75h-10.65v18.83c0 3.9 1.62 5.33 4.38 5.33s5.61-1.43 7.42-3.14v.01ZM711.24 310.72h-11.22v-24.44c0-6.09-1.33-10.56-7.61-10.56s-7.7 4.09-7.7 10.27v24.73h-11.22v-56.59l11.22-5.71v20.54c2.47-2.47 5.52-4.09 10.56-4.09 11.89 0 15.98 9.13 15.98 20.54v25.3l-.01.01ZM1000 288.17c0 1.43-.1 3.23-.19 3.9h-27.2c.48 6.28 4.76 8.85 9.61 8.85 2.85 0 5.52-.86 7.8-3.14l7.99 6.75c-4.18 5.23-10.56 7.13-16.26 7.13-13.13 0-20.64-10.08-20.64-23.3 0-14.36 8.85-23.49 19.88-23.49 11.79 0 19.02 10.37 19.02 23.3h-.01Zm-27.2-4.85h15.98c-.38-5.04-3.61-8.47-8.18-8.47-5.42 0-7.51 4.76-7.8 8.47ZM924.39 310.72h-11.03v-3.14c-2.47 2.76-6.47 4.09-10.46 4.09-11.6 0-15.98-8.37-15.98-20.45v-25.4h11.22v24.44c0 6.09 1.24 10.65 7.51 10.65s7.51-4.76 7.51-10.27v-24.82h11.22v44.89l.01.01ZM755.85 288.17c0 1.43-.1 3.23-.19 3.9h-27.2c.48 6.28 4.76 8.85 9.61 8.85 2.85 0 5.52-.86 7.8-3.14l7.99 6.75c-4.18 5.23-10.56 7.13-16.26 7.13-13.13 0-20.64-10.08-20.64-23.3 0-14.36 8.85-23.49 19.88-23.49 11.79 0 19.02 10.37 19.02 23.3h-.01Zm-27.21-4.85h15.98c-.38-5.04-3.61-8.47-8.18-8.47-5.42 0-7.51 4.76-7.8 8.47ZM806.35 249.94v9.23c-2.57-.67-4.28-.95-5.8-.95-3.99 0-4.85 1.43-4.85 3.71v3.9h7.61v10.84h-7.61v34.05h-11.22v-34.05h-5.52v-10.84h5.52v-5.42c0-7.8 4.09-11.79 13.6-11.79 3.04 0 5.52.57 8.27 1.33v-.01ZM668.34 297.59 666.72 309c-2.28 1.81-6.28 2.66-9.04 2.66-6.85 0-12.17-5.14-12.17-13.32v-21.78h-7.8v-10.75h7.8v-11.7l11.22-5.71v17.41h10.46v10.75h-10.46v18.83c0 3.9 1.62 5.33 4.38 5.33s5.42-1.43 7.23-3.14v.01ZM616.12 288.17c0 1.43-.1 3.23-.19 3.9h-26.25c.48 6.28 4.85 8.85 9.7 8.85 2.85 0 5.42-.86 7.7-3.14l7.99 6.75c-3.71 4.95-10.46 7.13-16.45 7.13-12.84 0-20.45-10.08-20.45-23.21s8.27-23.59 19.97-23.59c12.55 0 17.98 11.41 17.98 23.3v.01Zm-26.25-4.85h15.98c-.48-5.04-3.42-8.47-8.27-8.47-5.14 0-7.42 4.76-7.7 8.47h-.01ZM572.85 288.37c0 11.51-4.76 23.3-17.5 23.3-4.95 0-7.99-1.81-9.99-3.71v13.7l-11.22 5.61v-61.44h11.22v3.04c2.95-2.66 5.99-3.99 10.18-3.99 12.08 0 17.31 11.41 17.31 23.49Zm-11.61.57c0-6.56-1.71-13.22-8.47-13.22-3.14 0-5.61 1.43-7.42 3.99v17.12c1.81 2.57 4.85 4.09 8.08 4.09 6.28 0 7.8-5.71 7.8-11.98h.01ZM526.34 310.72h-11.22v-3.14c-2.66 2.66-6.09 4.09-10.46 4.09-8.85 0-15.79-5.71-15.79-15.6s6.75-15.12 17.5-15.12c2.85 0 5.8.38 8.75 1.81v-2.38c0-4.18-2.76-5.9-7.7-5.9-3.52 0-6.85.86-10.37 2.95l-4.47-7.99c4.85-3.04 9.51-4.57 15.5-4.57 11.32 0 18.26 5.52 18.26 15.69v30.16Zm-11.23-13.13v-5.71c-2.28-1.43-5.23-1.9-7.51-1.9-4.95 0-7.32 2.19-7.32 5.8 0 3.42 2.19 6.18 6.37 6.18 2.19 0 5.99-.76 8.47-4.38l-.01.01ZM482.68 310.72h-11.22v-24.44c0-6.09-1.33-10.56-7.61-10.56s-7.7 4.09-7.7 10.27v24.73h-11.22v-56.59l11.22-5.71v20.54c2.47-2.47 5.52-4.09 10.56-4.09 11.89 0 15.98 9.13 15.98 20.54v25.3l-.01.01ZM438.74 292.74c0 13.51-10.27 18.93-22.45 18.93-8.75 0-18.45-2.76-23.87-10.75l8.37-7.42c3.9 4.66 9.51 6.85 15.22 6.85 6.75 0 10.65-3.04 10.65-7.32 0-1.71-.67-3.42-3.52-4.85-2.09-1.05-4.66-1.71-9.7-2.95-3.14-.76-9.61-2.28-13.6-5.52-3.99-3.23-5.14-7.89-5.14-11.98 0-12.65 10.84-17.79 21.5-17.79 9.23 0 15.88 3.8 21.21 9.23l-8.37 8.18c-3.9-3.9-7.7-6.09-13.6-6.09-5.04 0-8.75 1.62-8.75 5.8 0 1.81.67 3.04 2.47 4.09 2.09 1.14 5.04 2 9.61 3.14 5.42 1.43 10.46 2.66 14.55 5.99 3.61 2.95 5.42 6.85 5.42 12.46ZM961.57 267.92l-4.58 11.22c-1.71-1.62-3.71-2.76-6.47-2.76-5.33 0-6.65 4.47-6.65 9.89v24.44h-11.22v-44.89h11.22v3.14c2.57-2.57 5.8-4.09 9.42-4.09 3.14 0 5.9.95 8.27 3.04l.01.01Z" />
                </g>
              </svg>
            </Link>
          </Toolbar>
          <Divider />
          <List>
            {menuItems.map((item) => (
              <ListItem key={item.path} disablePadding>
                <ListItemButton
                  component={RouterLink}
                  to={item.path}
                  selected={location.pathname === item.path}
                  sx={{ py: 1 }}
                >
                  <ListItemIcon sx={{ color: 'inherit', minWidth: '40px' }}>
                    {item.icon}
                  </ListItemIcon>
                  <ListItemText 
                    primary={item.text} 
                    primaryTypographyProps={{ 
                      fontWeight: 500,
                      color: 'inherit',
                    }} 
                  />
                </ListItemButton>
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
            transition: 'all 225ms cubic-bezier(0, 0, 0.2, 1) 0ms',
          }}
        >
          {children}
        </Box>
      </Box>
    </ThemeProvider>
  );
};

export default Layout;