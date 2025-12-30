import os
from pathlib import Path

BASE_DIR = Path(__file__).parent

class Config:
    SECRET_KEY = os.environ.get('SECRET_KEY', 'dev-secret-key-change-in-production')
    UPLOAD_FOLDER = BASE_DIR / 'uploads'
    MAX_CONTENT_LENGTH = 16 * 1024 * 1024
    ALLOWED_EXTENSIONS = {'.tif', '.tiff', '.png', '.jpg', '.jpeg'}

    MODEL_PATH = BASE_DIR / 'model' / 'model_076.pth'
    DEVICE = 'cpu'
    CLASS_NAMES = ['not_rain', 'medium_rain', 'heavy_rain']

    RAINFALL_DATA_PATH = BASE_DIR / 'rainfall_data.csv'

    IMG_SIZE = (224, 224)
    NORMALIZE_MEAN = [0.485, 0.456, 0.406]
    NORMALIZE_STD = [0.229, 0.224, 0.225]

class DevelopmentConfig(Config):
    DEBUG = True
    TESTING = False

class ProductionConfig(Config):
    DEBUG = False
    TESTING = False

config = {
    'development': DevelopmentConfig,
    'production': ProductionConfig,
    'default': DevelopmentConfig
}
