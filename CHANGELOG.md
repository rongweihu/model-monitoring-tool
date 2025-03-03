# Changelog

All notable changes to the Model Monitoring Tool (MMT) will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.0] - 2023-11-15

### Added
- Comprehensive model monitoring capabilities for PD, LGD, EAD, and Macro models
- Interactive data visualizations using Recharts
- Database storage for uploaded files and analysis results
- User-configurable thresholds for model performance metrics
- Weight of Evidence (WOE) plot visualization for both numeric and categorical variables
- Stationarity tests for macroeconomic variables (ADF, KPSS, Phillips-Perron)
- Summary dashboard with key metrics from all models
- Dark/light theme toggle with persistent user preference
- Responsive design for various screen sizes
- Comprehensive documentation (README, API documentation)

### Changed
- Refactored file storage mechanism to store files directly in the database
- Improved error handling and user feedback
- Enhanced UI/UX with Material-UI components
- Optimized data processing for large datasets

### Fixed
- Resolved issues with data validation during file upload
- Fixed calculation errors in certain statistical tests
- Addressed browser compatibility issues
- Improved error handling for edge cases

## [0.9.0] - 2023-10-01

### Added
- Initial implementation of PD model analysis
- Basic LGD and EAD model analysis
- File upload functionality
- Simple data visualization
- Basic user interface with React

### Changed
- Improved data processing pipeline
- Enhanced statistical analysis functions

### Fixed
- Various bug fixes and performance improvements

## [0.8.0] - 2023-09-15

### Added
- Project initialization
- Basic Flask API setup
- Database models with SQLAlchemy
- Frontend scaffolding with React and TypeScript
