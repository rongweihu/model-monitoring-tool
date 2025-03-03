from flask_sqlalchemy import SQLAlchemy
from datetime import datetime
import json

db = SQLAlchemy()

class Dataset(db.Model):
    """Model for storing uploaded dataset information"""
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(255), nullable=False)
    description = db.Column(db.Text, nullable=True)
    file_content = db.Column(db.LargeBinary, nullable=False)  # Store file as BLOB
    file_type = db.Column(db.String(50), nullable=False)  # 'pd', 'lgd', 'ead', 'macro', etc.
    file_size = db.Column(db.Integer, nullable=False)  # Size in bytes
    upload_date = db.Column(db.DateTime, default=datetime.utcnow)
    is_baseline = db.Column(db.Boolean, default=False)
    column_names = db.Column(db.Text, nullable=True)  # JSON string of column names
    row_count = db.Column(db.Integer, nullable=True)
    
    # Relationships
    analysis_results = db.relationship('AnalysisResult', backref='dataset', lazy=True)
    
    def __repr__(self):
        return f'<Dataset {self.name}>'
    
    def to_dict(self):
        return {
            'id': self.id,
            'name': self.name,
            'description': self.description,
            'file_type': self.file_type,
            'file_size': self.file_size,
            'upload_date': self.upload_date.isoformat(),
            'is_baseline': self.is_baseline,
            'column_names': json.loads(self.column_names) if self.column_names else [],
            'row_count': self.row_count
        }

class AnalysisResult(db.Model):
    """Model for storing analysis results"""
    id = db.Column(db.Integer, primary_key=True)
    dataset_id = db.Column(db.Integer, db.ForeignKey('dataset.id'), nullable=False)
    analysis_type = db.Column(db.String(50), nullable=False)  # 'pd', 'lgd', 'ead', 'macro', etc.
    result_data = db.Column(db.Text, nullable=False)  # JSON string of analysis results
    analysis_date = db.Column(db.DateTime, default=datetime.utcnow)
    parameters = db.Column(db.Text, nullable=True)  # JSON string of parameters used for analysis
    
    def __repr__(self):
        return f'<AnalysisResult {self.id} for Dataset {self.dataset_id}>'
    
    def to_dict(self):
        return {
            'id': self.id,
            'dataset_id': self.dataset_id,
            'analysis_type': self.analysis_type,
            'result_data': json.loads(self.result_data),
            'analysis_date': self.analysis_date.isoformat(),
            'parameters': json.loads(self.parameters) if self.parameters else {}
        }

class UserThreshold(db.Model):
    """Model for storing user-defined thresholds"""
    id = db.Column(db.Integer, primary_key=True)
    threshold_type = db.Column(db.String(50), nullable=False)  # 'pd', 'lgd', 'ead', 'macro', etc.
    metric_name = db.Column(db.String(100), nullable=False)
    threshold_value = db.Column(db.Float, nullable=False)
    description = db.Column(db.Text, nullable=True)
    created_date = db.Column(db.DateTime, default=datetime.utcnow)
    updated_date = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    def __repr__(self):
        return f'<UserThreshold {self.threshold_type}.{self.metric_name}>'
    
    def to_dict(self):
        return {
            'id': self.id,
            'threshold_type': self.threshold_type,
            'metric_name': self.metric_name,
            'threshold_value': self.threshold_value,
            'description': self.description,
            'created_date': self.created_date.isoformat(),
            'updated_date': self.updated_date.isoformat()
        }