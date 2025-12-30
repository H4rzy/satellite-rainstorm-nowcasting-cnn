import os
import logging
from pathlib import Path
import torch
from torchvision import models
import torch.nn as nn
from flask import Flask
from flask_cors import CORS

from config import config
from routes.main_routes import main_bp
from services.prediction_service import PredictionService
from services.data_service import RainfallDataService
from predict import load_model

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('app.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def create_app(config_name='default'):
    app = Flask(__name__)

    app.config.from_object(config[config_name])

    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

    CORS(app)

    logger.info("Loading prediction model...")
    try:
        model = load_model(str(app.config['MODEL_PATH']), device=app.config['DEVICE'])
        logger.info("Model loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise

    app.prediction_service = PredictionService(
        model=model,
        device=app.config['DEVICE'],
        class_names=app.config['CLASS_NAMES']
    )
    logger.info("Prediction service initialized")

    app.data_service = RainfallDataService(str(app.config['RAINFALL_DATA_PATH']))
    logger.info("Data service initialized")

    app.register_blueprint(main_bp)
    logger.info("Blueprints registered")

    @app.errorhandler(404)
    def not_found(error):
        logger.warning(f"404 error: {error}")
        return "Page not found", 404

    @app.errorhandler(500)
    def internal_error(error):
        logger.error(f"500 error: {error}")
        return "Internal server error", 500

    return app


if __name__ == '__main__':
    env = os.environ.get('FLASK_ENV', 'development')

    app = create_app(env)

    logger.info(f"Starting application in {env} mode")
    app.run(debug=app.config['DEBUG'])
