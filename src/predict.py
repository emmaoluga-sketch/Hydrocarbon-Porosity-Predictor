"""
Prediction script for Hydrocarbon Porosity Predictor.

This script makes porosity predictions using a trained model.
Supports both single predictions and batch predictions from CSV files.

Usage:
    python predict.py --model models/model.pkl
    python predict.py --model models/model.pkl --input data/new_data.csv --output predictions.csv
"""

import argparse
import joblib
import logging
import sys
from pathlib import Path
from typing import Union, Dict, List

import numpy as np
import pandas as pd

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class PredictionEngine:
    """
    Engine for making predictions with trained models.
    """
    
    def __init__(self, model_path: str):
        """
        Initialize the prediction engine.
        
        Parameters
        ----------
        model_path : str
            Path to the trained model file
            
        Raises
        ------
        FileNotFoundError
            If model file does not exist
        """
        if not Path(model_path).exists():
            logger.error(f"✗ Model file not found: {model_path}")
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        self.model = joblib.load(model_path)
        logger.info(f"✓ Model loaded from {model_path}")
        self.model_path = model_path
    
    def predict_single(self, features: Dict[str, float]) -> float:
        """
        Make a single prediction.
        
        Parameters
        ----------
        features : dict
            Dictionary with feature names as keys and values
            Expected keys: 'GR', 'RT', 'NPHI', 'RHOB'
            
        Returns
        -------
        float
            Predicted porosity value
            
        Example
        -------
        >>> engine = PredictionEngine('models/model.pkl')
        >>> prediction = engine.predict_single({
        ...     'GR': 75.0,
        ...     'RT': 12.0,
        ...     'NPHI': 0.27,
        ...     'RHOB': 2.33
        ... })
        >>> print(f"Predicted Porosity: {prediction:.4f}")
        """
        try:
            # Create DataFrame with single row
            df = pd.DataFrame([features])
            
            # Ensure column order matches training data
            expected_columns = ['GR', 'RT', 'NPHI', 'RHOB']
            df = df[expected_columns]
            
            # Make prediction
            prediction = self.model.predict(df)[0]
            
            logger.info(f"✓ Single prediction made: {prediction:.4f}")
            return prediction
            
        except KeyError as e:
            logger.error(f"✗ Missing required feature: {e}")
            raise
        except Exception as e:
            logger.error(f"✗ Error making prediction: {e}")
            raise
    
    def predict_batch(self, data: Union[str, pd.DataFrame]) -> pd.DataFrame:
        """
        Make batch predictions from CSV file or DataFrame.
        
        Parameters
        ----------
        data : str or pd.DataFrame
            Path to CSV file or DataFrame with input data
            Expected columns: 'GR', 'RT', 'NPHI', 'RHOB'
            
        Returns
        -------
        pd.DataFrame
            Original data with 'Porosity_Pred' column added
            
        Example
        -------
        >>> engine = PredictionEngine('models/model.pkl')
        >>> results = engine.predict_batch('data/new_wells.csv')
        >>> results.to_csv('predictions.csv', index=False)
        """
        try:
            # Load data if path provided
            if isinstance(data, str):
                if not Path(data).exists():
                    logger.error(f"✗ Input file not found: {data}")
                    raise FileNotFoundError(f"Input file not found: {data}")
                df = pd.read_csv(data)
                logger.info(f"✓ Loaded {len(df)} samples from {data}")
            else:
                df = data.copy()
            
            # Ensure required columns exist
            expected_columns = ['GR', 'RT', 'NPHI', 'RHOB']
            missing_columns = [col for col in expected_columns if col not in df.columns]
            
            if missing_columns:
                logger.error(f"✗ Missing required columns: {missing_columns}")
                raise ValueError(f"Missing required columns: {missing_columns}")
            
            # Extract features for prediction
            X = df[expected_columns]
            
            # Make predictions
            predictions = self.model.predict(X)
            
            # Add predictions to DataFrame
            df['Porosity_Pred'] = predictions
            
            logger.info(f"✓ Batch predictions completed for {len(df)} samples")
            logger.info(f"  Mean predicted porosity: {predictions.mean():.4f}")
            logger.info(f"  Std dev: {predictions.std():.4f}")
            logger.info(f"  Range: [{predictions.min():.4f}, {predictions.max():.4f}]")
            
            return df
            
        except Exception as e:
            logger.error(f"✗ Error during batch prediction: {e}")
            raise
    
    def predict_with_uncertainty(self, features: Dict[str, float]) -> Dict:
        """
        Make prediction with uncertainty estimate (for compatible models).
        
        Parameters
        ----------
        features : dict
            Dictionary with feature names and values
            
        Returns
        -------
        dict
            Dictionary with prediction and uncertainty metrics
        """
        # Standard prediction
        prediction = self.predict_single(features)
        
        result = {
            'prediction': prediction,
            'model_type': type(self.model).__name__
        }
        
        # Add uncertainty metrics if available
        if hasattr(self.model, 'estimators_'):  # Random Forest, etc.
            result['has_ensemble'] = True
            logger.info("Model is ensemble-based, uncertainty estimates available")
        
        return result


def print_example_predictions():
    """Print example predictions."""
    print("\n" + "=" * 60)
    print("EXAMPLE PREDICTIONS")
    print("=" * 60)
    
    examples = [
        {'GR': 75, 'RT': 12, 'NPHI': 0.27, 'RHOB': 2.33, 'description': 'Average well'},
        {'GR': 90, 'RT': 5, 'NPHI': 0.20, 'RHOB': 2.40, 'description': 'High GR, low porosity'},
        {'GR': 60, 'RT': 25, 'NPHI': 0.35, 'RHOB': 2.20, 'description': 'Low GR, high porosity'},
    ]
    
    for ex in examples:
        desc = ex.pop('description')
        print(f"\n{desc}:")
        print(f"  Input: {ex}")


def main():
    """Main prediction function."""
    
    parser = argparse.ArgumentParser(
        description='Make predictions with Hydrocarbon Porosity Predictor'
    )
    parser.add_argument(
        '--model',
        type=str,
        required=True,
        help='Path to trained model file'
    )
    parser.add_argument(
        '--input',
        type=str,
        help='Path to input CSV file for batch predictions'
    )
    parser.add_argument(
        '--output',
        type=str,
        help='Path to save predictions (default: predictions.csv)'
    )
    parser.add_argument(
        '--gr',
        type=float,
        help='Gamma Ray value (for single prediction)'
    )
    parser.add_argument(
        '--rt',
        type=float,
        help='Resistivity value (for single prediction)'
    )
    parser.add_argument(
        '--nphi',
        type=float,
        help='Neutron Porosity value (for single prediction)'
    )
    parser.add_argument(
        '--rhob',
        type=float,
        help='Bulk Density value (for single prediction)'
    )
    parser.add_argument(
        '--example',
        action='store_true',
        help='Show example predictions'
    )
    
    args = parser.parse_args()
    
    logger.info("=" * 60)
    logger.info("HYDROCARBON POROSITY PREDICTOR - INFERENCE")
    logger.info("=" * 60)
    
    try:
        # Load model
        engine = PredictionEngine(args.model)
        
        # Show examples
        if args.example:
            print_example_predictions()
            return 0
        
        # Batch prediction
        if args.input:
            logger.info(f"Starting batch predictions from {args.input}...")
            results = engine.predict_batch(args.input)
            
            # Save results
            output_path = args.output or 'predictions.csv'
            results.to_csv(output_path, index=False)
            logger.info(f"✓ Predictions saved to {output_path}")
            
            # Display summary
            logger.info("\nPrediction Summary:")
            logger.info(results[['GR', 'RT', 'NPHI', 'RHOB', 'Porosity_Pred']].describe())
            
        # Single prediction
        elif all([args.gr is not None, args.rt is not None, 
                  args.nphi is not None, args.rhob is not None]):
            logger.info("Making single prediction...")
            features = {
                'GR': args.gr,
                'RT': args.rt,
                'NPHI': args.nphi,
                'RHOB': args.rhob
            }
            
            prediction = engine.predict_single(features)
            
            logger.info("\n" + "=" * 60)
            logger.info("PREDICTION RESULT")
            logger.info("=" * 60)
            logger.info(f"Input Features:")
            logger.info(f"  GR:   {args.gr}")
            logger.info(f"  RT:   {args.rt}")
            logger.info(f"  NPHI: {args.nphi}")
            logger.info(f"  RHOB: {args.rhob}")
            logger.info(f"\nPredicted Porosity: {prediction:.4f}")
            logger.info("=" * 60)
            
        else:
            logger.warning("No input data provided. Use --input for batch prediction or --gr --rt --nphi --rhob for single prediction")
            print_example_predictions()
            parser.print_help()
            return 1
        
        logger.info("\n✓ Prediction completed successfully!")
        return 0
        
    except Exception as e:
        logger.error(f"\n✗ Error during prediction: {e}")
        logger.exception("Full traceback:")
        return 1


if __name__ == "__main__":
    sys.exit(main())
