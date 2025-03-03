import React, { useState } from 'react';
import {
  Paper, Typography, List, ListItem, ListItemText, Tabs, Tab, Box, Grid,
} from '@mui/material';
import {
  Analytics as AnalyticsIcon, TrendingUp as TrendingUpIcon, Assessment as AssessmentIcon,
  ModelTraining as ModelTrainingIcon, Calculate as CalculateIcon,
} from '@mui/icons-material';
import { useTheme } from '@mui/material/styles';

// === Interfaces ===
interface TabPanelProps {
  children?: React.ReactNode;
  value: number;
  index: number;
}

// === Utility Components ===
const TabPanel: React.FC<TabPanelProps> = ({ children, value, index }) => (
  <Box
    role="tabpanel"
    hidden={value !== index}
    id={`introduction-tabpanel-${index}`}
    aria-labelledby={`introduction-tab-${index}`}
    sx={{
      position: 'relative',
      width: '100%',
      height: '500px',
      overflow: 'hidden',
      backgroundColor: 'transparent',
      p: 3,
    }}
  >
    {value === index && (
      <Box
        sx={{
          height: '100%',
          overflowY: 'auto',
          backgroundColor: 'background.paper',
          borderRadius: 3,
          p: 3,
          boxShadow: 2,
          position: 'absolute',
          top: 0,
          left: 0,
          right: 0,
          bottom: 0,
        }}
      >
        {children}
      </Box>
    )}
  </Box>
);

const SectionHeader: React.FC<{ title: string }> = ({ title }) => (
  <Typography
    variant="h5"
    color="primary"
    gutterBottom
    sx={{ borderBottom: '2px solid', pb: 1, fontWeight: 600 }}
  >
    {title}
  </Typography>
);

const InfoItem: React.FC<{ primary: string; secondary: string }> = ({ primary, secondary }) => (
  <ListItem>
    <ListItemText
      primary={primary}
      secondary={secondary}
      primaryTypographyProps={{ fontWeight: 600 }}
      secondaryTypographyProps={{ component: 'span', sx: { display: 'block', mt: 1, lineHeight: 1.6 } }}
    />
  </ListItem>
);

// === Main Component ===
const Introduction: React.FC = () => {
  const theme = useTheme();
  const [activeTab, setActiveTab] = useState(0);

  const handleTabChange = (_event: React.SyntheticEvent, newValue: number) => {
    setActiveTab(newValue);
  };

  const introText = `Monitoring tool focuses on assessing the different components of ECL on an ongoing basis. 
    The tool compares the performance of each component (PD, LGD and EAD) over different quarters in order to 
    track deterioration of model performance and trigger the requirement for calibration or re-development.`;

  const tabConfig = [
    { label: 'PD', icon: <AnalyticsIcon /> },
    { label: 'Calibration', icon: <TrendingUpIcon /> },
    { label: 'LGD Model', icon: <AssessmentIcon /> },
    { label: 'EAD Model', icon: <ModelTrainingIcon /> },
    { label: 'Macro Model', icon: <CalculateIcon /> },
  ];

  return (
    <Paper
      elevation={3}
      sx={{
        width: '100%',
        maxWidth: 1600,
        margin: '0 auto',
        borderRadius: 4,
        overflow: 'hidden',
        boxShadow: 3,
      }}
    >
      <Box
        sx={{
          background: theme.palette.mode === 'dark'
            ? 'linear-gradient(135deg, #1a1a1a, #121212)'
            : 'linear-gradient(135deg, primary.main, primary.dark)',
          color: theme.palette.mode === 'dark' ? 'white' : 'white',
          p: 6,
          textAlign: 'center',
        }}
      >
        <Typography
          variant="h3"
          gutterBottom
          sx={{
            fontWeight: 700,
            mb: 2,
            color: theme.palette.mode === 'dark' ? 'white' : 'Black',
            textShadow: theme.palette.mode === 'dark' ? 'none' : '2px 2px 4px rgba(0,0,0,0.3)',
            letterSpacing: 1,
          }}
        >
          Model Monitoring Tool
        </Typography>
        <Typography
          variant="body1"
          sx={{
            color: theme.palette.mode === 'dark' ? 'white' : 'Black',
            textAlign: 'left',
            mt: 3,
            mb: 3,
            lineHeight: 1.6,
          }}
        >
          {introText}
          <br /><br />
          The Monitoring tool focuses on the below mentioned aspects:
          <br /><br />
          <strong>Scorecard Model</strong> – A model developed to obtain customer level PD. Metrics such as KS, Gini, PS are used to assess performance and stability of the model. Also, metrics such as IV and CSI are calculated to assess the variables used in the model
          <br /><br />
          <strong>Macroeconomic Model</strong> – A model developed to obtain relationship between portfolio default trend and macro-economic variable. Metrics such as MAPE, R-square are calculated to assess performance of the model. Also, assumption testing, stationarity test is performed to ensure model is relevant after addition of recent data as well
          <br /><br />
          <strong>Calibration</strong> – PD Calibration focuses on checking if the predicted PD is in line with predicted PD
          <br /><br />
          <strong>LGD and EAD</strong> – Performance of LGD and EAD model is assessed through MAPE and actual vs predicted comparison of LGD and EAD components
        </Typography>
      </Box>

      <Tabs
        value={activeTab}
        onChange={handleTabChange}
        variant="fullWidth"
        sx={{
          borderBottom: 1,
          borderColor: 'divider',
          backgroundColor: 'background.default',
        }}
        indicatorColor="primary"
        textColor="primary"
      >
        {tabConfig.map(({ label, icon }, index) => (
          <Tab
            key={label}
            label={label}
            icon={icon}
            iconPosition="start"
            sx={{
              fontWeight: 600,
              '&.Mui-selected': { color: 'primary.main', backgroundColor: 'rgba(0,0,0,0.05)' },
            }}
          />
        ))}
      </Tabs>

      <TabPanel value={activeTab} index={0}>
        <Grid container spacing={4}>
          <Grid item xs={12} md={4}>
            <SectionHeader title="Scorecard Model - Discriminatory Power Test" />
            <List>
              <InfoItem
                primary="GINI - Cumulative Accuracy Profile (CAP) curve"
                secondary="Gini Coefficient is the ratio of model lift over random model and lift of best model over random model. Gini is calculated as 2*Area Under CAP curve -1. Higher the GINI coefficient, closer is the model in perfectly discriminating bad accounts from good accounts."
              />
              <InfoItem
                primary="Kolmogorov – Smirnov (KS) Test"
                secondary="KS Test measures the maximum deviation between the percentage cumulative bads and percentage cumulative goods. It identifies the maximum vertical separation (percentage) between the cumulative percentages of Goods vs. Bads."
              />
            </List>
          </Grid>
          <Grid item xs={12} md={4}>
            <SectionHeader title="Stability Test" />
            <List>
              <InfoItem
                primary="Population Stability Index (PSI)"
                secondary="Population stability index is used to measure the stability of the model by quantifying and measuring the difference in the distance (distributional shift) between the two distributions: current and baseline. PSI is calculated using the below formula:\nPSI = Σ (% of Current - % of Base) × ln(% of Current / % of Base)"
              />
            </List>
          </Grid>
          <Grid item xs={12} md={4}>
            <SectionHeader title="Scorecard Model - Variable Assessment" />
            <List>
              <InfoItem
                primary="Information Value (IV)"
                secondary="Information value corresponding to a variable is used to assess the predictive power. It is calculated as:\nIV = Σ ( % of Good - % of Bad ) × ln( % of Good / % of Bad )"
              />
              <InfoItem
                primary="Character Stability Index (CSI)"
                secondary="Character stability index is used to measure the stability of the variable by quantifying and measuring the difference in the distance (distributional shift) between the two distributions: current and baseline. CSI is calculated using the below formula:\nCSI = Σ (% of Current - % of Base) × ln(% of Current / % of Base)"
              />
            </List>
          </Grid>
        </Grid>
      </TabPanel>

      <TabPanel value={activeTab} index={1}>
        <SectionHeader title="Model Calibration / Accuracy Tests" />
        <Grid container spacing={3}>
          <Grid item xs={12}>
            <List>
              <InfoItem
                primary="Binomial Test"
                secondary="This test is used to assess the performance of calibration in each of the score bands individually. This is achieved by constructing an acceptable upper bound of observed default rates around the expected default rate in each score band."
              />
              <InfoItem
                primary="Hosmer – Lemeshow (HL) Test"
                secondary="The Calibration is assessed by estimating the Hosmer Lemeshow test Statistic (HL). HL statistic follows a (chi-square) distribution with degrees of freedom (df) k, where k is the (number of bands – 2). Higher the value of observed HL statistic, it indicates the observed and expected PD is more far from each other and so the test fails."
              />
            </List>
          </Grid>
        </Grid>
      </TabPanel>

      <TabPanel value={activeTab} index={2}>
        <SectionHeader title="LGD Model" />
        <Grid container spacing={3}>
          <Grid item xs={12}>
            <List>
              <InfoItem
                primary="Mean Absolute Percentage Error (MAPE) Test for LGD"
                secondary="This test is used to assess the performance of calibration in each of the score bands individually. This is achieved by constructing an acceptable upper bound of observed default rates around the expected default rate in each score band."
              />
            </List>
          </Grid>
        </Grid>
      </TabPanel>

      <TabPanel value={activeTab} index={3}>
        <SectionHeader title="EAD Model" />
        <Grid container spacing={3}>
          <Grid item xs={12}>
            <List>
              <InfoItem
                primary="Mean Absolute Percentage Error (MAPE) Test for EAD"
                secondary="This test is used to assess the performance of calibration in each of the score bands individually. This is achieved by constructing an acceptable upper bound of observed default rates around the expected default rate in each score band."
              />
            </List>
          </Grid>
        </Grid>
      </TabPanel>

      <TabPanel value={activeTab} index={4}>
        <SectionHeader title="Macroeconomic Model" />
        <Grid container spacing={3}>
          <Grid item xs={12}>
            <List>
              <InfoItem
                primary="Normality Assumption Testing (for Linear Regression)"
                secondary="The Anderson–Darling test is a statistical test of whether a given sample of data is drawn from a given probability distribution. In its basic form, the test assumes that there are no parameters to be estimated in the distribution being tested, in which case the test and its set of critical values is distribution-free. However, the test is most often used in contexts where a family of distributions is being tested, in which case the parameters of that family need to be estimated and account must be taken of this in adjusting either the test-statistic or its critical values. When applied to testing whether a normal distribution adequately describes a set of data, it is one of the most powerful statistical tools for detecting most departures from normality."
              />
              <InfoItem
                primary="Auto-correlation Assumption Testing"
                secondary="Absence of auto – correlation is a very important assumption for linear regression model. Trend in error indicates the dependency on lagged observations In presence of auto correlation, the obtained estimates are no longer efficient. Durbin Watson test is used to detect presence of autocorrelation"
              />
              <InfoItem
                primary="Heteroscedasticity - Assumption Testing"
                secondary="Errors are said to have constant variance across all the observation In absence of homoscedasticity, the observed estimate would be unbiased and consistent but no longer best, which means it will not have minimum variance in class of all unbiased estimates. The Breusch–Pagan test is used to test for heteroskedasticity in a linear regression model"
              />
              <InfoItem
                primary="Adjusted R-square and RMSE – Model Performance"
                secondary="Adjusted R-square explains the proportion of variance in PD that is explained by the model. Higher the adjusted R-square, better is the performance of the model. RMSE compares the actual PD and predicted PD. RMSE is calculated as: Lower the RMSE, better is the accuracy of the model."
              />
              <InfoItem
                primary="Stationarity Test for Macro Variables"
                secondary={[
                  "Several tests are commonly employed:",
                  "• <strong>Augmented Dickey-Fuller (ADF) Test</strong>: The ADF test is a widely used statistical test that checks for the presence of a unit root in a univariate time series. A unit root indicates non-stationarity. The null hypothesis of the ADF test is that the series has a unit root, while the alternative hypothesis suggests stationarity.",
                  "• <strong>Kwiatkowski-Phillips-Schmidt-Shin (KPSS) Test</strong>: In contrast to the ADF test, the KPSS test has a null hypothesis of stationarity. It tests whether a time series is stationary around a deterministic trend. If the KPSS test statistic exceeds a critical value, we reject the null hypothesis and conclude that the series is non-stationary.",
                  "• <strong>Zivot-Andrews Test</strong>: This test extends the ADF test by allowing for a structural break in the time series. It tests for a unit root while accounting for potential shifts in the mean or trend, making it particularly useful for macroeconomic data that may experience sudden changes due to policy shifts or economic events.",
                  "• <strong>Phillips-Perron (PP) Test</strong>: Similar to the ADF test, the Phillips-Perron test checks for a unit root in a time series. However, it adjusts for serial correlation and heteroskedasticity in the error terms, providing a more robust alternative to the ADF test.",
                ].map((item, index) => (
                  <div key={index} dangerouslySetInnerHTML={{ __html: item }} />
                ))}
              />
            </List>
          </Grid>
        </Grid>
      </TabPanel>
    </Paper>
  );
};

export default Introduction;