"""
Utility functions for data loading, preprocessing, and evaluation.

This module provides helper functions for:
- Loading and processing well-log data
- Feature scaling and normalization
- Data splitting for training and testing
- Model evaluation metrics
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def load_data(path: str) -> pd.DataFrame:
    """
    Load data from a CSV file.
    
    Parameters
    ----------
    path : str
        Path to the CSV file
        
    Returns
    -------
    pd.DataFrame
        Loaded data
        
    Raises
    ------
    FileNotFoundError
        If the file does not exist
    pd.errors.ParserError
        If the file cannot be parsed
    """
    try:
        df = pd.read_csv(path)
        logger.info(f"✓ Data loaded successfully from {path}. Shape: {df.shape}")
        return df
    except FileNotFoundError:
        logger.error(f"✗ File not found: {path}")
        raise
    except pd.errors.ParserError as e:
        logger.error(f"✗ Error parsing CSV file: {e}")
        raise


def check_data_quality(df: pd.DataFrame) -> dict:
    """
    Check data quality and return a report.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe
        
    Returns
    -------
    dict
        Dictionary containing quality metrics
    """
    quality_report = {
        'total_rows': len(df),
        'total_columns': len(df.columns),
        'missing_values': df.isnull().sum().to_dict(),
        'duplicates': df.duplicated().sum(),
        'dtypes': df.dtypes.to_dict()
    }
    
    logger.info(f"Data Quality Report:\n{quality_report}")
    return quality_report


def handle_missing_values(df: pd.DataFrame, method: str = 'mean') -> pd.DataFrame:
    """
    Handle missing values in the dataframe.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe
    method : str, default='mean'
        Method to handle missing values ('mean', 'median', 'drop')
        
    Returns
    -------
    pd.DataFrame
        Dataframe with missing values handled
    """
    if method == 'mean':
        df = df.fillna(df.mean())
    elif method == 'median':
        df = df.fillna(df.median())
    elif method == 'drop':
        df = df.dropna()
    else:
        logger.warning(f"Unknown method {method}, using 'mean'")
        df = df.fillna(df.mean())
    
    logger.info(f"✓ Missing values handled using '{method}' method")
    return df


def split_data(df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42) -> tuple:
    """
    Split data into training and testing sets.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe with features and target
    test_size : float, default=0.2
        Proportion of data to use for testing
    random_state : int, default=42
        Random seed for reproducibility
        
    Returns
    -------
    tuple
        (X_train, X_test, y_train, y_test)
    """
    if 'Porosity' not in df.columns:
        logger.error("✗ 'Porosity' column not found in dataframe")
        raise ValueError("Target column 'Porosity' not found")
    
    X = df.drop("Porosity", axis=1)
    y = df["Porosity"]
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    logger.info(f"✓ Data split: Train={len(X_train)}, Test={len(X_test)}")
    return X_train, X_test, y_train, y_test


def scale_features(X_train: pd.DataFrame, X_test: pd.DataFrame, 
                   method: str = 'standard') -> tuple:
    """
    Scale features using StandardScaler or MinMaxScaler.
    
    Parameters
    ----------
    X_train : pd.DataFrame
        Training features
    X_test : pd.DataFrame
        Testing features
    method : str, default='standard'
        Scaling method ('standard' or 'minmax')
        
    Returns
    -------
    tuple
        (X_train_scaled, X_test_scaled, scaler)
    """
    if method == 'standard':
        scaler = StandardScaler()
    elif method == 'minmax':
        scaler = MinMaxScaler()
    else:
        logger.warning(f"Unknown scaling method {method}, using 'standard'")
        scaler = StandardScaler()
    
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns)
    X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns)
    
    logger.info(f"✓ Features scaled using '{method}' scaling")
    return X_train_scaled, X_test_scaled, scaler


def evaluate_model(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """
    Evaluate model predictions using multiple metrics.
    
    Parameters
    ----------
    y_true : np.ndarray
        True target values
    y_pred : np.ndarray
        Predicted values
        
    Returns
    -------
    dict
        Dictionary containing evaluation metrics
    """
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    
    metrics = {
        'MSE': mse,
        'RMSE': rmse,
        'MAE': mae,
        'R2_Score': r2
    }
    
    logger.info(f"✓ Model Evaluation Metrics:")
    logger.info(f"  RMSE: {rmse:.4f}")
    logger.info(f"  MAE: {mae:.4f}")
    logger.info(f"  R² Score: {r2:.4f}")
    
    return metrics


def get_feature_info(df: pd.DataFrame) -> dict:
    """
    Get detailed information about features.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe
        
    Returns
    -------
    dict
        Feature information including statistics and data types
    """
    feature_info = {}
    
    for column in df.columns:
        if df[column].dtype in ['float64', 'int64']:
            feature_info[column] = {
                'dtype': str(df[column].dtype),
                'mean': float(df[column].mean()),
                'std': float(df[column].std()),
                'min': float(df[column].min()),
                'max': float(df[column].max()),
                'missing': int(df[column].isnull().sum())
            }
    
    return feature_info


if __name__ == "__main__":
    # Example usage
    print("Utility module for Hydrocarbon Porosity Predictor")
    print("=" * 50)
    print("\nAvailable functions:")
    print("- load_data(path)")
    print("- check_data_quality(df)")
    print("- handle_missing_values(df, method)")
    print("- split_data(df, test_size, random_state)")
    print("- scale_features(X_train, X_test, method)")
    print("- evaluate_model(y_true, y_pred)")
    print("- get_feature_info(df)")
