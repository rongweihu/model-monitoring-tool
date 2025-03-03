# Model Monitoring Tool (MMT) Publication Package

This package contains the Model Monitoring Tool (MMT), a comprehensive application for monitoring and analyzing credit risk models. This publication-ready version has been optimized for distribution and includes all necessary files to run the application.

## Package Contents

The package includes the following components:

### Documentation
- `README.md`: Main project documentation
- `CHANGELOG.md`: History of changes and version information
- `CONTRIBUTING.md`: Guidelines for contributing to the project
- `LICENSE`: MIT license file
- `INSTALL.md`: Installation instructions

### Backend
- `backend/app.py`: Main Flask application with API endpoints
- `backend/models.py`: SQLAlchemy database models
- `backend/requirements.txt`: Python dependencies
- `backend/README.md`: Backend-specific documentation
- `backend/data/`: Sample data files for testing
- `backend/uploads/`: Directory for temporary file storage

### Frontend
- `frontend/src/`: React application source code
- `frontend/package.json`: Node.js dependencies and scripts
- `frontend/tsconfig.json`: TypeScript configuration
- `frontend/vite.config.ts`: Vite build configuration
- `frontend/README.md`: Frontend-specific documentation

## Key Features

### Database-Driven File Storage
The application stores files directly in the database rather than the filesystem:
- Centralized storage for improved data management
- Reduced filesystem dependencies
- Simplified backup and migration processes

### Weight of Evidence (WOE) Visualization
The application includes WOE plot visualization for both numeric and categorical variables:
- Interactive charts using Recharts
- Automatic binning for numeric variables
- Support for categorical variables
- Detailed tooltips with WOE values and bin information

### Comprehensive Model Analysis
- PD Model: Discrimination, calibration, and stability metrics
- Macro Model: Stationarity tests and performance metrics
- LGD Model: Recovery rate analysis and segmentation
- EAD Model: CCF distribution and utilization rate analysis

## Installation

Please refer to the `INSTALL.md` file for detailed installation instructions.

## Optimization Notes

This package has been optimized for distribution by:

1. Removing unnecessary files:
   - Node modules and virtual environment directories
   - Cache files and temporary data
   - Development and build artifacts
   - IDE-specific configuration files
   - System files (.DS_Store, etc.)

2. Including only essential sample data files

3. Providing clear documentation for setup and usage

## Version Information

This is version 1.0.0 of the Model Monitoring Tool. For a detailed list of features and changes, please refer to the `CHANGELOG.md` file.

## License

This project is licensed under the MIT License - see the `LICENSE` file for details.
