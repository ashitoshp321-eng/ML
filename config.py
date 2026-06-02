import os

class Config:
    # Base Directory
    BASE_DIR = os.path.abspath(os.path.dirname(__file__))
    
    # Flask configuration
    SECRET_KEY = os.environ.get('SECRET_KEY', 'ml_testing_platform_secret_key_129847129')
    DEBUG = True
    
    # Paths for uploads, reports, models, datasets
    UPLOAD_FOLDER = os.path.join(BASE_DIR, 'uploads')
    REPORTS_FOLDER = os.path.join(BASE_DIR, 'reports')
    SAVED_MODELS_FOLDER = os.path.join(BASE_DIR, 'saved_models')
    DATASETS_FOLDER = os.path.join(BASE_DIR, 'datasets')
    
    # Firebase configuration
    FIREBASE_CREDENTIALS = os.path.join(BASE_DIR, 'firebase-credentials.json')
    LOCAL_DB_PATH = os.path.join(BASE_DIR, 'database', 'local_db.json')
    
    # Allowed extensions
    ALLOWED_EXTENSIONS = {'csv'}

# Ensure folders exist
for folder in [Config.UPLOAD_FOLDER, Config.REPORTS_FOLDER, Config.SAVED_MODELS_FOLDER, Config.DATASETS_FOLDER, os.path.dirname(Config.LOCAL_DB_PATH)]:
    os.makedirs(folder, exist_ok=True)
