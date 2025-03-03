#!/bin/bash

# Create a temporary directory for the package
TEMP_DIR=$(mktemp -d)
PACKAGE_NAME="mmt_v1.0.0"
PACKAGE_DIR="$TEMP_DIR/$PACKAGE_NAME"

# Create directory structure
mkdir -p "$PACKAGE_DIR"
mkdir -p "$PACKAGE_DIR/backend"
mkdir -p "$PACKAGE_DIR/backend/data"
mkdir -p "$PACKAGE_DIR/frontend/src"
mkdir -p "$PACKAGE_DIR/frontend/src/components"
mkdir -p "$PACKAGE_DIR/frontend/src/pages"
mkdir -p "$PACKAGE_DIR/frontend/src/utils"
mkdir -p "$PACKAGE_DIR/frontend/src/assets"

# Copy root level files
cp /Users/rongweihu/mmt/README.md "$PACKAGE_DIR/"
cp /Users/rongweihu/mmt/CHANGELOG.md "$PACKAGE_DIR/"
cp /Users/rongweihu/mmt/CONTRIBUTING.md "$PACKAGE_DIR/"
cp /Users/rongweihu/mmt/LICENSE "$PACKAGE_DIR/"
cp /Users/rongweihu/mmt/requirements.txt "$PACKAGE_DIR/"

# Copy backend files
cp /Users/rongweihu/mmt/backend/app.py "$PACKAGE_DIR/backend/"
cp /Users/rongweihu/mmt/backend/models.py "$PACKAGE_DIR/backend/"
cp /Users/rongweihu/mmt/backend/README.md "$PACKAGE_DIR/backend/"
cp /Users/rongweihu/mmt/backend/requirements.txt "$PACKAGE_DIR/backend/"

# Copy sample data files (limited set)
cp /Users/rongweihu/mmt/backend/data/Data_PD_Model.csv "$PACKAGE_DIR/backend/data/"
cp /Users/rongweihu/mmt/backend/data/MacroData_predicted_defaultrate.xlsx "$PACKAGE_DIR/backend/data/"
cp /Users/rongweihu/mmt/backend/data/lgd_mock_data.csv "$PACKAGE_DIR/backend/data/"
cp /Users/rongweihu/mmt/backend/data/EAD_Mock_Data.xlsx "$PACKAGE_DIR/backend/data/"

# Create empty uploads directory
mkdir -p "$PACKAGE_DIR/backend/uploads"
touch "$PACKAGE_DIR/backend/uploads/.gitkeep"

# Copy frontend configuration files
cp /Users/rongweihu/mmt/frontend/package.json "$PACKAGE_DIR/frontend/"
cp /Users/rongweihu/mmt/frontend/tsconfig.json "$PACKAGE_DIR/frontend/"
cp /Users/rongweihu/mmt/frontend/tsconfig.node.json "$PACKAGE_DIR/frontend/"
cp /Users/rongweihu/mmt/frontend/vite.config.ts "$PACKAGE_DIR/frontend/"
cp /Users/rongweihu/mmt/frontend/README.md "$PACKAGE_DIR/frontend/"

# Copy frontend source files
cp /Users/rongweihu/mmt/frontend/src/*.tsx "$PACKAGE_DIR/frontend/src/"
cp /Users/rongweihu/mmt/frontend/src/*.css "$PACKAGE_DIR/frontend/src/"
cp /Users/rongweihu/mmt/frontend/src/vite-env.d.ts "$PACKAGE_DIR/frontend/src/"

# Copy components, pages, and utils
cp /Users/rongweihu/mmt/frontend/src/components/*.tsx "$PACKAGE_DIR/frontend/src/components/"
cp /Users/rongweihu/mmt/frontend/src/pages/*.tsx "$PACKAGE_DIR/frontend/src/pages/"
cp /Users/rongweihu/mmt/frontend/src/utils/*.ts "$PACKAGE_DIR/frontend/src/utils/"

# Copy assets (excluding .DS_Store)
find /Users/rongweihu/mmt/frontend/src/assets -type f -not -name ".DS_Store" -exec cp {} "$PACKAGE_DIR/frontend/src/assets/" \;

# Create a README file for the package
cat > "$PACKAGE_DIR/INSTALL.md" << 'EOF'
# Model Monitoring Tool (MMT) Installation Guide

This guide will help you set up and run the Model Monitoring Tool on your system.

## Prerequisites

- Python 3.9 or higher
- Node.js 16 or higher
- npm 8 or higher

## Backend Setup

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

4. Start the Flask development server:
   ```bash
   python app.py
   ```
   The API will be available at http://localhost:5000

## Frontend Setup

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

## Production Deployment

For production deployment, please refer to the README.md file for detailed instructions.

## Sample Data

Sample data files are provided in the `backend/data` directory. You can use these files to test the application.
EOF

# Create zip file
cd "$TEMP_DIR"
zip -r "$PACKAGE_NAME.zip" "$PACKAGE_NAME"

# Move zip file to user's home directory
mv "$PACKAGE_NAME.zip" /Users/rongweihu/

# Clean up
rm -rf "$TEMP_DIR"

echo "Package created: /Users/rongweihu/$PACKAGE_NAME.zip"
