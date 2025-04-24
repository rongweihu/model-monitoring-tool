from datetime import datetime
import json
from flask_sqlalchemy import SQLAlchemy

db = SQLAlchemy()

class Dataset(db.Model):
    __tablename__ = "dataset"
    
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(255), nullable=False)
    description = db.Column(db.Text)
    file_content = db.Column(db.LargeBinary, nullable=False)
    file_type = db.Column(db.String(50), nullable=False)  # e.g., 'pd', 'lgd', 'ead', 'macro'
    file_size = db.Column(db.Integer, nullable=False)
    upload_date = db.Column(db.DateTime, default=db.func.current_timestamp())
    is_baseline = db.Column(db.Boolean, default=False)
    column_names = db.Column(db.Text)  # JSON string of column names
    row_count = db.Column(db.Integer)
    column_mapping = db.Column(db.Text)  # JSON string of column mappings
    as_of_date = db.Column(db.Date, nullable=True)  # New field for as-of date

    def to_dict(self):
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "file_type": self.file_type,
            "file_size": self.file_size,
            "upload_date": self.upload_date.isoformat() if self.upload_date else None,
            "is_baseline": self.is_baseline,
            "column_names": self.column_names,
            "row_count": self.row_count,
            "column_mapping": self.column_mapping,
            "as_of_date": self.as_of_date.isoformat() if self.as_of_date else None
        }

class AnalysisResult(db.Model):
    __tablename__ = "analysis_result"
    
    id = db.Column(db.Integer, primary_key=True)
    dataset_id = db.Column(db.Integer, db.ForeignKey("dataset.id"), nullable=False)
    analysis_type = db.Column(db.String(50), nullable=False)  # e.g., 'pd', 'lgd', 'ead', 'macro'
    result_data = db.Column(db.Text, nullable=False)  # JSON string of analysis results
    parameters = db.Column(db.Text)  # JSON string of analysis parameters
    created_at = db.Column(db.DateTime, default=db.func.current_timestamp())

    dataset = db.relationship("Dataset", backref=db.backref("analysis_results", lazy=True))

    def to_dict(self):
        return {
            "id": self.id,
            "dataset_id": self.dataset_id,
            "analysis_type": self.analysis_type,
            "result_data": self.result_data,
            "parameters": self.parameters,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "as_of_date": self.dataset.as_of_date.isoformat() if self.dataset and self.dataset.as_of_date else None
        }

class UserThreshold(db.Model):
    """Model for storing user-defined thresholds"""

    id = db.Column(db.Integer, primary_key=True)
    threshold_type = db.Column(
        db.String(50), nullable=False
    )  # 'pd', 'lgd', 'ead', 'macro', etc.
    metric_name = db.Column(db.String(100), nullable=False)
    threshold_value = db.Column(db.Float, nullable=False)
    description = db.Column(db.Text, nullable=True)
    created_date = db.Column(db.DateTime, default=datetime.utcnow)
    updated_date = db.Column(
        db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow
    )

    def __repr__(self):
        return f"<UserThreshold {self.threshold_type}.{self.metric_name}>"

    def to_dict(self):
        return {
            "id": self.id,
            "threshold_type": self.threshold_type,
            "metric_name": self.metric_name,
            "threshold_value": self.threshold_value,
            "description": self.description,
            "created_date": self.created_date.isoformat(),
            "updated_date": self.updated_date.isoformat(),
        }