import pandas as pd
import numpy as np
import os
from src.utils import setup_logger

logger = setup_logger("etl")

def load_data(filepath):
    """Load data from CSV file."""
    logger.info(f"Loading data from {filepath}")
    try:
        df = pd.read_csv(filepath)
        logger.info(f"Loaded {len(df)} rows and {len(df.columns)} columns")
        return df
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        raise

def clean_columns(df):
    """Clean column names: strip whitespace, lower case, replace spaces with underscores."""
    logger.info("Cleaning column names")
    df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_')
    return df

def handle_missing_values(df):
    """Handle missing values: drop rows with too many missing, fill others."""
    logger.info("Handling missing values")
    
    # Drop columns with > 50% missing
    threshold = len(df) * 0.5
    df = df.dropna(thresh=threshold, axis=1)
    
    # Fill numeric with median, categorical with mode
    for col in df.columns:
        if df[col].dtype in [np.float64, np.int64]:
            df[col] = df[col].fillna(df[col].median())
        else:
            df[col] = df[col].fillna(df[col].mode()[0])
            
    return df

def cap_outliers(df):
    """Cap outliers using IQR method for numeric columns."""
    logger.info("Capping outliers using IQR")
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    
    for col in numeric_cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        df[col] = np.clip(df[col], lower_bound, upper_bound)
        
    return df

def run_etl(filepath):
    """Run full ETL pipeline."""
    logger.info("Starting ETL pipeline")
    
    df = load_data(filepath)
    df = clean_columns(df)
    
    # Trim whitespace from string columns
    df = df.apply(lambda x: x.str.strip() if x.dtype == "object" else x)
    
    df = handle_missing_values(df)
    df = cap_outliers(df)
    
    logger.info("ETL pipeline completed")
    logger.info(f"Final shape: {df.shape}")
    
    # Summary output
    print(df.describe())
    
    return df

if __name__ == "__main__":
    # Example usage
    import sys
    if len(sys.argv) > 1:
        run_etl(sys.argv[1])
    else:
        print("Please provide a file path")
