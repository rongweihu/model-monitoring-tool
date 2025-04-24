from flask import Flask
from models import db
import os

app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = "sqlite:///" + os.path.join(
    os.path.dirname(__file__), "mmt_database.db"
)
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db.init_app(app)

with app.app_context():
    db.drop_all()  # Drops all tables
    db.create_all()  # Creates all tables based on models