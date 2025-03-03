import React from 'react';
import { 
  Card, 
  CardHeader, 
  CardContent 
} from '@mui/material';

interface TestCardProps {
  title: string;
  children: React.ReactNode;
  status?: 'success' | 'warning' | 'error';
}

const TestCard: React.FC<TestCardProps> = ({ 
  title, 
  children, 
  status 
}) => {
  const getStatusColor = () => {
    switch (status) {
      case 'success': return 'success.main';
      case 'warning': return 'warning.main';
      case 'error': return 'error.main';
      default: return 'text.secondary';
    }
  };

  return (
    <Card 
      variant="outlined" 
      sx={{ 
        height: '100%', 
        display: 'flex', 
        flexDirection: 'column',
        transition: 'transform 0.3s ease-in-out',
        '&:hover': {
          transform: 'scale(1.02)',
          boxShadow: 3
        }
      }}
    >
      <CardHeader
        title={title}
        titleTypographyProps={{
          variant: 'h6',
          color: getStatusColor()
        }}
        sx={{ 
          backgroundColor: status ? `${getStatusColor()}10` : 'transparent',
          borderBottom: '1px solid',
          borderBottomColor: status ? getStatusColor() : 'divider'
        }}
      />
      <CardContent sx={{ flexGrow: 1 }}>
        {children}
      </CardContent>
    </Card>
  );
};

export default TestCard;
