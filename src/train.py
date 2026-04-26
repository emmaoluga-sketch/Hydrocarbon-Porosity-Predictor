"""
Training script for Hydrocarbon Porosity Predictor.

This script trains machine learning models to predict rock porosity from well-log data.
Supports multiple algorithms: XGBoost, Random Forest, and Linear Regression.

Usage:
    python train.py
    python train.py --model xgboost --output models/model.pkl
"""

import argparse
import joblib
import logging
import sys
from pathlib import Path

import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from xgboost import XGBRegressor

from utils import (
    load_data,
    check_data_quality,
    handle_missing_values,
    split_data,
    scale_features,
    evaluate_model
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ModelTrainer:
    """
    Trainer class for machine learning models.
    """
    
    def __init__(self, model_type: str = 'xgboost', random_state: int = 42):
        """
        Initialize the trainer.
        
        Parameters
        ----------
        model_type : str
            Type of model ('xgboost', 'random_forest', 'linear_regression')
        random_state : int
            Random seed for reproducibility
        """
        self.model_type = model_type
        self.random_state = random_state
        self.model = None
        self.scaler = None
        self.metrics = None
        
    def build_model(self):
        """Build the selected model."""
        if self.model_type == 'xgboost':
            self.model = XGBRegressor(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=self.random_state,
                verbose=0
            )
            logger.info("✓ XGBoost model initialized")
            
        elif self.model_type == 'random_forest':
            self.model = RandomForestRegressor(
                n_estimators=100,
                max_depth=15,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=self.random_state,
                n_jobs=-1
            )
            logger.info("✓ Random Forest model initialized")
            
        elif self.model_type == 'linear_regression':
            self.model = LinearRegression()
            logger.info("✓ Linear Regression model initialized")
            
        else:
            logger.error(f"✗ Unknown model type: {self.model_type}")
            raise ValueError(f"Unknown model type: {self.model_type}")
    
    def train(self, X_train, y_train):
        """
        Train the model.
        
        Parameters
        ----------
        X_train : pd.DataFrame
            Training features
        y_train : pd.Series
            Training target
        """
        logger.info(f"Training {self.model_type} model...")
        self.model.fit(X_train, y_train)
        logger.info(f"✓ {self.model_type} model trained successfully")
    
    def predict(self, X):
        """
        Make predictions.
        
        Parameters
        ----------
        X : pd.DataFrame
            Input features
            
        Returns
        -------
        np.ndarray
            Predictions
        """
        return self.model.predict(X)
    
    def evaluate(self, X_test, y_test):
        """
        Evaluate model performance.
        
        Parameters
        ----------
        X_test : pd.DataFrame
            Test features
        y_test : pd.Series
            Test target
            
        Returns
        -------
        dict
            Evaluation metrics
        """
        y_pred = self.predict(X_test)
        self.metrics = evaluate_model(y_test, y_pred)
        return self.metrics
    
    def get_feature_importance(self, feature_names):
        """
        Get feature importance (for tree-based models).
        
        Parameters
        ----------
        feature_names : list
            Names of features
            
        Returns
        -------
        dict
            Feature importance scores
        """
        if hasattr(self.model, 'feature_importances_'):
            importances = self.model.feature_importances_
            feature_importance = dict(zip(feature_names, importances))
            feature_importance = dict(sorted(
                feature_importance.items(),
                key=lambda x: x[1],
                reverse=True
            ))
            return feature_importance
        else:
            logger.warning("Model does not have feature_importances_ attribute")
            return {}
    
    def save_model(self, filepath: str):
        """
        Save model to disk.
        
        Parameters
        ----------
        filepath : str
            Path to save the model
        """
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.model, filepath)
        logger.info(f"✓ Model saved to {filepath}")


def main():
    """Main training function."""
    
    parser = argparse.ArgumentParser(
        description='Train Hydrocarbon Porosity Predictor model'
    )
    parser.add_argument(
        '--data',
        type=str,
        default='data/sample_data.csv',
        help='Path to training data (default: data/sample_data.csv)'
    )
    parser.add_argument(
        '--model',
        type=str,
        choices=['xgboost', 'random_forest', 'linear_regression'],
        default='xgboost',
        help='Model type to train (default: xgboost)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='models/model.pkl',
        help='Path to save trained model (default: models/model.pkl)'
    )
    parser.add_argument(
        '--test-size',
        type=float,
        default=0.2,
        help='Proportion of data for testing (default: 0.2)'
    )
    parser.add_argument(
        '--random-state',
        type=int,
        default=42,
        help='Random seed (default: 42)'
    )
    parser.add_argument(
        '--scale',
        action='store_true',
        help='Scale features before training'
    )
    
    args = parser.parse_args()
    
    logger.info("=" * 60)
    logger.info("HYDROCARBON POROSITY PREDICTOR - TRAINING")
    logger.info("=" * 60)
    
    try:
        # Load data
        logger.info(f"Loading data from {args.data}...")
        df = load_data(args.data)
        
        # Check data quality
        check_data_quality(df)
        
        # Handle missing values
        df = handle_missing_values(df, method='mean')
        
        # Split data
        X_train, X_test, y_train, y_test = split_data(
            df,
            test_size=args.test_size,
            random_state=args.random_state
        )
        
        # Scale features if requested
        if args.scale:
            logger.info("Scaling features...")
            X_train, X_test, scaler = scale_features(
                X_train, X_test, method='standard'
            )
        
        # Build and train model
        logger.info(f"Training {args.model} model...")
        trainer = ModelTrainer(
            model_type=args.model,
            random_state=args.random_state
        )
        trainer.build_model()
        trainer.train(X_train, y_train)
        
        # Evaluate on test set
        logger.info("Evaluating on test set...")
        metrics = trainer.evaluate(X_test, y_test)
        
        # Display results
        logger.info("\n" + "=" * 60)
        logger.info("TRAINING RESULTS")
        logger.info("=" * 60)
        logger.info(f"Model: {args.model}")
        logger.info(f"Training samples: {len(X_train)}")
        logger.info(f"Test samples: {len(X_test)}")
        logger.info("\nEvaluation Metrics:")
        logger.info(f"  RMSE: {metrics['RMSE']:.4f}")
        logger.info(f"  MAE:  {metrics['MAE']:.4f}")
        logger.info(f"  R²:   {metrics['R2_Score']:.4f}")
        
        # Feature importance
        if hasattr(trainer.model, 'feature_importances_'):
            logger.info("\nTop Features:")
            importance = trainer.get_feature_importance(X_train.columns)
            for i, (feature, score) in enumerate(importance.items(), 1):
                logger.info(f"  {i}. {feature}: {score:.4f}")
        
        # Save model
        logger.info(f"\nSaving model to {args.output}...")
        trainer.save_model(args.output)
        
        logger.info("\n✓ Training completed successfully!")
        logger.info("=" * 60)
        
        return 0
        
    except Exception as e:
        logger.error(f"\n✗ Error during training: {e}")
        logger.exception("Full traceback:")
        return 1


if __name__ == "__main__":
    sys.exit(main())
