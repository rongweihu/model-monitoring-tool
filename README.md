# Model Monitoring Tool (MMT)

[![Version](https://img.shields.io/badge/version-1.0.0-blue.svg)](https://github.com/yourusername/mmt)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

A comprehensive tool for monitoring and analyzing credit risk models, including PD (Probability of Default), LGD (Loss Given Default), EAD (Exposure at Default), and Macroeconomic models. This application helps financial institutions assess model performance, calibration, and stability over time to meet regulatory requirements and improve risk management practices.

![MMT Dashboard](https://via.placeholder.com/800x450.png?text=MMT+Dashboard)

## 🚀 Features

### 📊 Model Performance Monitoring
- **PD Model Analysis**
  - Gini coefficient and KS statistic calculation
  - Population Stability Index (PSI) monitoring
  - Hosmer-Lemeshow test for calibration
  - Binomial test by rating
  - Variable assessment with Information Value (IV) and Characteristic Stability Index (CSI)
  - Weight of Evidence (WOE) visualization

- **Macro Model Analysis**
  - Stationarity tests (ADF, KPSS, Zivot-Andrews, Phillips-Perron)
  - Normality, autocorrelation, and heteroscedasticity tests
  - Time series visualization and forecasting
  - Model performance metrics (R-squared, RMSE)

- **LGD Model Analysis**
  - Recovery rate analysis
  - Mean Absolute Percentage Error (MAPE) calculation
  - R-squared performance metrics
  - Segmentation analysis

- **EAD Model Analysis**
  - Credit Conversion Factor (CCF) distribution
  - MAPE and R-squared metrics
  - Utilization rate analysis

### 📈 Data Visualization
- Interactive charts and graphs with Recharts
- Time series analysis with trend visualization
- Distribution analysis with histograms and density plots
- Segmentation analysis with heatmaps
- Weight of Evidence (WOE) plot visualization for both numeric and categorical variables

### 🎨 User Experience
- Intuitive dark/light theme toggle
- Responsive design for desktop and tablet
- Consistent error handling with informative messages
- Reusable UI components for a cohesive experience
- Database-driven file storage for improved data management

### 📝 Statistical Analysis
- Binomial tests for calibration assessment
- Stationarity tests for time series analysis
- Performance metrics calculation (Gini, KS, MAPE, R-squared)
- Model validation tests with configurable thresholds

## 🔍 Key Features in Detail

### Weight of Evidence (WOE) Plot Visualization

The application supports WOE plot visualizations for both numeric and categorical variables, enhancing the analysis of variable importance in predictive modeling.

- **Numeric Variables**: WOE plots display mean values on the x-axis
- **Categorical Variables**: WOE plots show categories on the x-axis
- **Interactive Plots**: Responsive and interactive visualizations
- **Automatic Binning**: Smart binning for numeric variables
- **Tooltip Information**: Detailed WOE values and bin information on hover

### Database-Driven File Storage

The application stores files directly in the database rather than the filesystem:

- **Centralized Storage**: All data stored in a single database
- **Reduced Dependencies**: No filesystem management required
- **Improved Data Management**: Easier backup and migration
- **Simplified Workflow**: Streamlined data retrieval for analysis

## 🛠️ Tech Stack

### Backend
- **Python 3.8+**
- **Flask**: Web framework
- **SQLAlchemy**: ORM for database interactions
- **NumPy & Pandas**: Data analysis
- **SciPy & Statsmodels**: Statistical computing
- **Scikit-learn**: Machine learning algorithms
- **WOE-Binning**: Weight of Evidence calculation

### Frontend
- **React 18** with TypeScript
- **Material-UI v5**: Component library
- **Recharts**: Data visualization
- **Axios**: HTTP client
- **React Router v6**: Routing
- **Notistack**: Notification system

## 📋 Prerequisites

- Python 3.8 or higher
- Node.js 16 or higher
- npm 8 or higher

## 🔧 Installation

### Clone the Repository

```bash
git clone https://github.com/yourusername/mmt.git
cd mmt
```

### Backend Setup

1. Navigate to the backend directory:
   ```bash
   cd backend
   ```

2. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Set up environment variables (create a .env file):
   ```
   FLASK_APP=app.py
   FLASK_ENV=development
   DATABASE_URL=sqlite:///mmt.db
   ```

5. Initialize the database:
   ```bash
   python -c "from app import db; db.create_all()"
   ```

6. Start the Flask server:
   ```bash
   python app.py
   ```
   The server will run on http://localhost:5000

### Frontend Setup

1. Navigate to the frontend directory:
   ```bash
   cd frontend
   ```

2. Install dependencies:
   ```bash
   npm install
   ```

3. Start the development server:
   ```bash
   npm run dev
   ```
   The application will be available at http://localhost:5173

## 🚀 Production Deployment

### Backend

1. Install Gunicorn:
   ```bash
   pip install gunicorn
   ```

2. Create a WSGI entry point (wsgi.py):
   ```python
   from app import app
   
   if __name__ == "__main__":
       app.run()
   ```

3. Run with Gunicorn:
   ```bash
   gunicorn --bind 0.0.0.0:5000 wsgi:app
   ```

### Frontend

1. Build the production version:
   ```bash
   npm run build
   ```

2. Serve the build directory with a static file server or configure your web server (Nginx, Apache) to serve the files.

## 📖 Usage

1. **Upload Data**: Start by uploading your model data files through the Data Upload page
2. **Configure Thresholds**: Set model thresholds and criteria in User Inputs
3. **Analyze Models**: Navigate through different sections to analyze model performance:
   - **Summary**: Overview of key metrics across all models
   - **PD Model**: Detailed PD model analysis with discrimination and calibration tests
   - **Macro Model**: Macroeconomic model validation with statistical tests
   - **LGD Model**: LGD performance metrics and segmentation analysis
   - **EAD Model**: EAD and CCF analysis with performance metrics
   - **Database Manager**: Manage uploaded datasets and analysis results

## 🔌 API Reference

### Data Upload Endpoints
- `POST /api/upload/pd`: Upload PD model data files
- `POST /api/upload/lgd`: Upload LGD model data files
- `POST /api/upload/ead`: Upload EAD model data files
- `POST /api/upload/macro`: Upload macroeconomic data files

### Analysis Endpoints
- `POST /api/analyze/pd`: Analyze PD model performance
- `POST /api/analyze/lgd`: Analyze LGD model performance
- `POST /api/analyze/ead`: Analyze EAD model performance
- `POST /api/analyze/macro`: Analyze macro model performance
- `POST /api/analyze/rating_transition`: Analyze rating transitions
- `POST /api/analyze/stability`: Calculate stability metrics

### Configuration Endpoints
- `GET /api/thresholds`: Get user-defined thresholds
- `POST /api/thresholds`: Save user-defined thresholds
- `GET /api/summary`: Get summary data
- `GET /api/lgd/options`: Get LGD filter options

## 🧪 Testing

### Backend Tests

```bash
cd backend
pytest
```

### Frontend Tests

```bash
cd frontend
npm test
```

## 📝 Documentation

Comprehensive documentation is available in the `/docs` directory.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👥 Authors

- **Your Name** - *Initial work* - [YourGitHub](https://github.com/yourusername)

## 🙏 Acknowledgments

- Hat tip to anyone whose code was used
- Inspiration
- etc.
