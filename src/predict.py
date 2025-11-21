import joblib
import pandas as pd
import os
from src.utils import setup_logger
from src.features import prepare_features

logger = setup_logger("predict")

def load_model(model_dir="models"):
    """Load trained model."""
    model_path = os.path.join(model_dir, "model.joblib")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found at {model_path}")
        
    logger.info(f"Loading model from {model_path}")
    return joblib.load(model_path)

def make_predictions(df, model=None):
    """Make predictions using the trained model."""
    if model is None:
        model = load_model()
        
    # Preprocess features (assuming no target column in prediction data)
    # Note: In a real scenario, we need to ensure the schema matches exactly what was trained.
    # This simple implementation assumes the input df has the raw columns that need processing.
    
    logger.info("Preparing features for prediction")
    X = prepare_features(df)
    
    # Align columns with model if possible (simple check)
    # Ideally, we should save the feature names during training and enforce them here.
    # For this MVP, we assume the pipeline handles it or the user provides matching data.
    
    logger.info("Making predictions")
    predictions = model.predict(X)
    
    return predictions
