import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from src.utils import setup_logger

logger = setup_logger("features")

def extract_date_features(df):
    """Extract features from datetime columns."""
    logger.info("Extracting date features")
    
    # Attempt to identify datetime columns
    # This is a heuristic: look for 'date' or 'time' in column name or object cols that parse easily
    for col in df.columns:
        if df[col].dtype == 'object' or np.issubdtype(df[col].dtype, np.datetime64):
            try:
                # Try converting to datetime
                temp_col = pd.to_datetime(df[col], errors='ignore')
                if np.issubdtype(temp_col.dtype, np.datetime64):
                    df[col] = temp_col
                    df[f'{col}_year'] = df[col].dt.year
                    df[f'{col}_month'] = df[col].dt.month
                    df[f'{col}_day'] = df[col].dt.day
                    df[f'{col}_dayofweek'] = df[col].dt.dayofweek
                    df[f'{col}_is_weekend'] = df[col].dt.dayofweek.isin([5, 6]).astype(int)
                    
                    # Drop original date column if it's not the target (assuming target won't be a date)
                    # But for safety, let's keep it or drop it based on user preference. 
                    # For now, we'll drop the original to avoid issues in ML models that can't handle datetime objects directly.
                    df = df.drop(columns=[col])
                    logger.info(f"Decomposed {col} into year, month, day, dayofweek")
            except Exception:
                pass
                
    return df

def prepare_features(df, target_col=None):
    """
    Prepare features for training.
    If target_col is provided, splits into X and y.
    """
    logger.info("Preparing features")
    
    df = extract_date_features(df)
    
    # One-hot encoding for categorical variables
    df = pd.get_dummies(df, drop_first=True)
    
    if target_col:
        if target_col not in df.columns:
            raise ValueError(f"Target column {target_col} not found in dataframe")
            
        X = df.drop(columns=[target_col])
        y = df[target_col]
        return X, y
    else:
        return df

def split_data(X, y, test_size=0.2, random_state=42):
    """Split data into train and test sets."""
    logger.info(f"Splitting data with test_size={test_size}")
    return train_test_split(X, y, test_size=test_size, random_state=random_state)
