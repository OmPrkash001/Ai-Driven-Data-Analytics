import pandas as pd
import numpy as np
import joblib
import json
import os
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, f1_score, classification_report
from xgboost import XGBRegressor, XGBClassifier
from lightgbm import LGBMRegressor, LGBMClassifier
from src.utils import setup_logger, ensure_directories

logger = setup_logger("train_model")

def detect_problem_type(y):
    """
    Detect if the problem is regression or classification.
    """
    if y.dtype in [np.int64, np.int32, object] and y.nunique() < 20:
        return "classification"
    else:
        return "regression"

def train_model(X_train, y_train, problem_type):
    """Train model based on problem type."""
    logger.info(f"Training model for {problem_type}")
    
    if problem_type == "classification":
        # Use LightGBM for classification
        model = LGBMClassifier(random_state=42)
    else:
        # Use XGBoost for regression
        model = XGBRegressor(random_state=42)
        
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test, problem_type):
    """Evaluate model and return metrics."""
    logger.info("Evaluating model")
    y_pred = model.predict(X_test)
    
    metrics = {}
    
    if problem_type == "classification":
        metrics['accuracy'] = accuracy_score(y_test, y_pred)
        metrics['f1_score'] = f1_score(y_test, y_pred, average='weighted')
        metrics['report'] = classification_report(y_test, y_pred, output_dict=True)
    else:
        metrics['mse'] = mean_squared_error(y_test, y_pred)
        metrics['rmse'] = np.sqrt(metrics['mse'])
        metrics['r2'] = r2_score(y_test, y_pred)
        
    return metrics

def save_artifacts(model, metrics, output_dir="models"):
    """Save model and metrics."""
    ensure_directories([output_dir])
    
    model_path = os.path.join(output_dir, "model.joblib")
    metrics_path = os.path.join(output_dir, "metrics.json")
    
    joblib.dump(model, model_path)
    logger.info(f"Model saved to {model_path}")
    
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=4)
    logger.info(f"Metrics saved to {metrics_path}")

def run_training(df, target_col):
    """Run full training pipeline."""
    from src.features import prepare_features, split_data
    
    logger.info(f"Starting training pipeline for target: {target_col}")
    
    X, y = prepare_features(df, target_col)
    problem_type = detect_problem_type(y)
    logger.info(f"Detected problem type: {problem_type}")
    
    X_train, X_test, y_train, y_test = split_data(X, y)
    
    model = train_model(X_train, y_train, problem_type)
    metrics = evaluate_model(model, X_test, y_test, problem_type)
    
    metrics['problem_type'] = problem_type
    save_artifacts(model, metrics)
    
    return metrics
