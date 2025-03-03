# MMT Backend

[![Python](https://img.shields.io/badge/Python-3.9+-blue?logo=python)](https://www.python.org/)
[![Flask](https://img.shields.io/badge/Flask-2.2.3-000000?logo=flask)](https://flask.palletsprojects.com/)
[![SQLAlchemy](https://img.shields.io/badge/SQLAlchemy-2.0.4-red?logo=sqlalchemy)](https://www.sqlalchemy.org/)
[![Pandas](https://img.shields.io/badge/Pandas-1.5.3-150458?logo=pandas)](https://pandas.pydata.org/)

This is the backend application for the Model Monitoring Tool (MMT), built with Flask and SQLAlchemy. The backend provides a RESTful API for model performance monitoring, statistical analysis, and data management for credit risk models.

## 🚀 Features

### 📊 Statistical Analysis
- **Stationarity Tests**: ADF, KPSS, and Phillips-Perron tests for time series data
- **Performance Metrics**: Comprehensive metrics for PD, LGD, EAD, and Macro models
- **Stability Analysis**: PSI, CSI, and other stability metrics
- **Weight of Evidence**: WOE calculation and visualization for both numeric and categorical variables
- **Rating Transitions**: Analysis of rating migrations over time

### 💾 Data Management
- **Database Storage**: Direct storage of file content in the database
- **Data Processing**: Efficient processing of large datasets
- **File Validation**: Input validation for uploaded files
- **Data Transformation**: Preprocessing and transformation of raw data

### 🔄 API Endpoints
- **RESTful Design**: Well-structured endpoints following REST principles
- **CORS Support**: Cross-origin resource sharing for frontend integration
- **Error Handling**: Comprehensive error handling and reporting
- **Response Formatting**: Consistent JSON response format

### 🛡️ Security
- **Input Validation**: Validation of all input data
- **Error Handling**: Secure error handling that doesn't expose sensitive information
- **Database Security**: Parameterized queries to prevent SQL injection

## 🛠️ Tech Stack

- **Flask**: Lightweight web framework for Python
- **SQLAlchemy**: SQL toolkit and Object-Relational Mapping (ORM)
- **Pandas**: Data manipulation and analysis library
- **NumPy**: Numerical computing library
- **SciPy**: Scientific computing and statistics
- **Statsmodels**: Statistical models and tests
- **Scikit-learn**: Machine learning algorithms
- **Arch**: Time series analysis
- **WOE-Binning**: Weight of Evidence calculations
- **Gunicorn**: WSGI HTTP Server for production deployment

## 📍 Project Structure

```
backend/
├── app.py                # Main Flask application with API endpoints
├── models.py             # SQLAlchemy database models
├── requirements.txt      # Python dependencies
├── uploads/              # Directory for temporary file storage
├── data/                 # Sample data and test files
└── tests/                # Unit and integration tests
```

## 🔧 Key Components

### Database Models

The application uses SQLAlchemy ORM with the following main models:

- **Dataset**: Stores uploaded file content, metadata, and file type
- **AnalysisResult**: Stores analysis results for different model types
- **UserThreshold**: Stores user-defined thresholds for model monitoring

### API Endpoints

The backend provides the following main API endpoints:

#### Data Upload
- `POST /api/upload/pd`: Upload PD model data files
- `POST /api/upload/lgd`: Upload LGD model data files
- `POST /api/upload/ead`: Upload EAD model data files
- `POST /api/upload/macro`: Upload macroeconomic data files

#### Analysis
- `POST /api/analyze/pd`: Analyze PD model performance
- `POST /api/analyze/lgd`: Analyze LGD model performance
- `POST /api/analyze/ead`: Analyze EAD model performance
- `POST /api/analyze/macro`: Analyze macro model performance
- `POST /api/analyze/rating_transition`: Analyze rating transitions
- `POST /api/analyze/stability`: Calculate stability metrics

#### Configuration
- `GET /api/thresholds`: Get user-defined thresholds
- `POST /api/thresholds`: Save user-defined thresholds
- `GET /api/summary`: Get summary data
- `GET /api/lgd/options`: Get LGD filter options

### Statistical Analysis Functions

The backend implements various statistical analysis functions:

- **Stationarity Tests**: `run_stationarity_tests()`, `run_adf_test()`, `run_kpss_test()`, `run_pp_test()`
- **Performance Metrics**: `calculate_performance_metrics()`, `calculate_discrimination()`, `calculate_calibration()`
- **Stability Analysis**: `calculate_psi()`, `calculate_csi()`, `calculate_stability_metrics()`
- **Weight of Evidence**: `calculate_information_value()`, `calculate_woe()`

## 💻 Development

### Prerequisites

- Python 3.9 or higher
- pip (Python package installer)
- virtualenv (recommended)

### Installation

1. Clone the repository and navigate to the backend directory:
   ```bash
   git clone https://github.com/yourusername/mmt.git
   cd mmt/backend
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

4. Start the Flask development server:
   ```bash
   python app.py
   ```
   The API will be available at http://localhost:5000

### Available Commands

- Start development server: `python app.py`
- Run tests: `pytest`
- Start production server: `gunicorn -w 4 app:app`

## 🚀 Production Deployment

For production deployment, it's recommended to use Gunicorn as a WSGI HTTP server:

1. Install Gunicorn (already included in requirements.txt):
   ```bash
   pip install gunicorn
   ```

2. Start the production server:
   ```bash
   gunicorn -w 4 -b 0.0.0.0:5000 app:app
   ```

### Deployment Considerations

- Use environment variables for configuration
- Set `DEBUG=False` in production
- Configure proper logging
- Use a reverse proxy like Nginx for SSL termination and load balancing
- Set up proper database backups

## 🔗 API Documentation

### Data Upload Endpoints

#### `POST /api/upload/pd`
Upload PD model data files.

**Request:**
- Form data with file attachment

**Response:**
```json
{
  "success": true,
  "message": "File uploaded successfully",
  "dataset_id": 123
}
```

#### `POST /api/upload/lgd`
Upload LGD model data files.

**Request:**
- Form data with file attachment

**Response:**
```json
{
  "success": true,
  "message": "File uploaded successfully",
  "dataset_id": 124
}
```

### Analysis Endpoints

#### `POST /api/analyze/pd`
Analyze PD model performance.

**Request:**
```json
{
  "dataset_id": 123
}
```

**Response:**
```json
{
  "success": true,
  "results": {
    "discrimination": {...},
    "calibration": {...},
    "stability": {...},
    "variable_importance": [...]
  }
}
```

#### `POST /api/analyze/macro`
Analyze macroeconomic model performance.

**Request:**
```json
{
  "dataset_id": 125
}
```

**Response:**
```json
{
  "success": true,
  "results": {
    "stationarity_tests": [...],
    "performance_metrics": {...},
    "time_series_data": [...]
  }
}
```

### Configuration Endpoints

#### `GET /api/thresholds`
Get user-defined thresholds.

**Response:**
```json
{
  "success": true,
  "thresholds": {
    "auc_threshold": 0.7,
    "gini_threshold": 0.4,
    "psi_threshold": 0.1,
    ...
  }
}
```

#### `POST /api/thresholds`
Save user-defined thresholds.

**Request:**
```json
{
  "auc_threshold": 0.75,
  "gini_threshold": 0.45,
  "psi_threshold": 0.15,
  ...
}
```

**Response:**
```json
{
  "success": true,
  "message": "Thresholds saved successfully"
}
```

## 🔍 Testing

The backend includes unit and integration tests using pytest:

```bash
pytest
```

### Test Coverage

To generate a test coverage report:

```bash
pytest --cov=. --cov-report=html
```

This will generate an HTML coverage report in the `htmlcov` directory.

## 📝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](../LICENSE) file for details.
