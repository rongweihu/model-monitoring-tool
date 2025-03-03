#!/bin/bash

# Create clean project directory
CLEAN_DIR="/Users/rongweihu/mmt_clean"
mkdir -p "$CLEAN_DIR"/{frontend,backend,backend/data}

# Copy essential files
echo "Copying essential files..."

# Root files
cp /Users/rongweihu/mmt/README.md "$CLEAN_DIR/"
cp /Users/rongweihu/mmt/requirements.txt "$CLEAN_DIR/"

# Backend files
cp /Users/rongweihu/mmt/backend/app.py "$CLEAN_DIR/backend/"
cp /Users/rongweihu/mmt/backend/requirements.txt "$CLEAN_DIR/backend/"

# Sample data files (keeping one of each type)
cp /Users/rongweihu/mmt/backend/data/Data_PD_Model.csv "$CLEAN_DIR/backend/data/"
cp /Users/rongweihu/mmt/backend/data/Data_PD_Model_pvs_qtr.csv "$CLEAN_DIR/backend/data/"
cp /Users/rongweihu/mmt/backend/data/EAD_Mock_Data.xlsx "$CLEAN_DIR/backend/data/"
cp /Users/rongweihu/mmt/backend/data/MacroData_predicted_defaultrate.xlsx "$CLEAN_DIR/backend/data/"
cp /Users/rongweihu/mmt/backend/data/lgd_mock_data.csv "$CLEAN_DIR/backend/data/"

# Create uploads directory
mkdir -p "$CLEAN_DIR/backend/uploads"

# Frontend files
cp /Users/rongweihu/mmt/frontend/README.md "$CLEAN_DIR/frontend/"
cp /Users/rongweihu/mmt/frontend/package.json "$CLEAN_DIR/frontend/"
cp /Users/rongweihu/mmt/frontend/tsconfig.json "$CLEAN_DIR/frontend/"
cp /Users/rongweihu/mmt/frontend/tsconfig.app.json "$CLEAN_DIR/frontend/"
cp /Users/rongweihu/mmt/frontend/tsconfig.node.json "$CLEAN_DIR/frontend/"
cp /Users/rongweihu/mmt/frontend/vite.config.ts "$CLEAN_DIR/frontend/"
cp /Users/rongweihu/mmt/frontend/index.html "$CLEAN_DIR/frontend/"

# Create frontend directory structure
mkdir -p "$CLEAN_DIR/frontend/src/"{components,pages,utils,assets}

# Copy frontend source files
cp -r /Users/rongweihu/mmt/frontend/src/components/* "$CLEAN_DIR/frontend/src/components/"
cp -r /Users/rongweihu/mmt/frontend/src/pages/* "$CLEAN_DIR/frontend/src/pages/"
cp -r /Users/rongweihu/mmt/frontend/src/utils/* "$CLEAN_DIR/frontend/src/utils/"
cp -r /Users/rongweihu/mmt/frontend/src/assets/* "$CLEAN_DIR/frontend/src/assets/" 2>/dev/null || true
cp /Users/rongweihu/mmt/frontend/src/App.tsx "$CLEAN_DIR/frontend/src/"
cp /Users/rongweihu/mmt/frontend/src/App.css "$CLEAN_DIR/frontend/src/"
cp /Users/rongweihu/mmt/frontend/src/index.css "$CLEAN_DIR/frontend/src/"
cp /Users/rongweihu/mmt/frontend/src/main.tsx "$CLEAN_DIR/frontend/src/"
cp /Users/rongweihu/mmt/frontend/src/vite-env.d.ts "$CLEAN_DIR/frontend/src/"

# Remove .DS_Store files
find "$CLEAN_DIR" -name ".DS_Store" -delete

echo "Project cleaned and copied to $CLEAN_DIR"
echo "To publish the project, compress this directory:"
echo "cd $CLEAN_DIR && zip -r ../mmt_project.zip ."
