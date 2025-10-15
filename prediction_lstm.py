import pandas as pd
import numpy as np
import tensorflow as tf
import pickle
from pathlib import Path
from typing import Tuple, Dict, Any, List


class LSTMPredictor:
    def __init__(self, 
                 model_path: str = "model/multi_output_lstm_h5_lookback5.keras",
                 scaler_path: str = "model/scaler_multi.pkl"):
        self.model_path = Path(model_path)
        self.scaler_path = Path(scaler_path)
        self.model = None
        self.scaler = None
        self.lookback = 5
        self.horizon = 5
        self.target_column = 'intensity_nca'
        self.required_features = ['nca', 'gdp', 'growth_gdp', 'growth_nca', 
                                 'loggrowth_gdp', 'loggrowth_nca', 'intensity_nca']
        self.load_model()
    
    def load_model(self):
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model not found at {self.model_path}")
        
        if not self.scaler_path.exists():
            raise FileNotFoundError(f"Scaler not found at {self.scaler_path}")
        
        # Load model
        self.model = tf.keras.models.load_model(str(self.model_path))
        print(f"LSTM model loaded from {self.model_path}")
        print(f"Input shape: {self.model.input_shape}")
        print(f"Output shape: {self.model.output_shape}")
        
        # Load scaler
        with open(self.scaler_path, 'rb') as f:
            self.scaler = pickle.load(f)
        print(f"Scaler loaded from {self.scaler_path}")
    
    def get_model_info(self) -> Dict[str, Any]:
        return {
            "model_name": "Multi-Output LSTM",
            "model_type": "Time Series Forecasting",
            "lookback": self.lookback,
            "horizon": self.horizon,
            "target": self.target_column,
            "features": self.required_features,
            "n_features": len(self.required_features),
            "input_shape": str(self.model.input_shape),
            "output_shape": str(self.model.output_shape),
            "description": f"Predicts {self.horizon} future years using {self.lookback} past years"
        }
    
    def validate_and_prepare_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Validate and prepare data with required features.
        """
        df = df.copy()
        
        # Check for required base features
        base_features = ['nca', 'gdp', 'growth_gdp', 'growth_nca', 
                        'loggrowth_gdp', 'loggrowth_nca']
        missing = [f for f in base_features if f not in df.columns]
        
        if missing:
            raise ValueError(
                f"Missing required columns: {', '.join(missing)}\n\n"
                f"Required columns: {', '.join(base_features)}"
            )
        
        # Calculate intensity_nca if not present
        if 'intensity_nca' not in df.columns:
            df['intensity_nca'] = df['nca'] / df['gdp']
            df['intensity_nca'] = df['intensity_nca'].replace([np.inf, -np.inf], np.nan)
        
        # Sort by year and code if available
        if 'code' in df.columns and 'year' in df.columns:
            df = df.sort_values(['code', 'year'])
        elif 'year' in df.columns:
            df = df.sort_values('year')
        
        # Select only required features
        df_features = df[self.required_features].copy()
        
        # Fill missing values
        df_features = df_features.fillna(df_features.mean())
        
        return df_features
    
    def create_sequences(self, data: np.ndarray) -> Tuple[np.ndarray, List[int]]:
        """
        Create sequences for LSTM prediction.
        
        Args:
            data: Scaled array of shape (n_samples, n_features)
        
        Returns:
            Tuple of (sequences, start_indices)
            - sequences: Array of shape (n_sequences, lookback, n_features)
            - start_indices: List of starting indices for each sequence
        """
        sequences = []
        start_indices = []
        
        for i in range(len(data) - self.lookback + 1):
            sequence = data[i:i + self.lookback]
            sequences.append(sequence)
            start_indices.append(i + self.lookback)
        
        return np.array(sequences, dtype=np.float32), start_indices
    
    def predict(self, df: pd.DataFrame) -> Tuple[np.ndarray, List[int], Dict[str, Any]]:
        """
        Make predictions on the dataset.
        
        Args:
            df: Input dataframe
        
        Returns:
            Tuple of (predictions, start_years, model_info)
            - predictions: Array of shape (n_sequences, horizon)
            - start_years: List of years where predictions start
            - model_info: Model information dictionary
        """
        try:
            df_features = self.validate_and_prepare_data(df)
            
            if len(df_features) < self.lookback:
                raise ValueError(
                    f"Not enough data for prediction. Need at least {self.lookback} rows, got {len(df_features)}.\n"
                    f"This model requires {self.lookback} consecutive years to predict the next {self.horizon} years."
                )
            
            print(f"Data shape before scaling: {df_features.shape}")
            
            # Scale the features using the fitted scaler
            scaled_data = self.scaler.transform(df_features.values)
            print(f"Data scaled successfully")
            
            # Create sequences from scaled data
            X, start_indices = self.create_sequences(scaled_data)
            
            print(f"Created {len(X)} sequences of shape {X.shape}")
            
            # Make predictions
            predictions = self.model.predict(X, verbose=0)
            
            print(f"Predictions shape: {predictions.shape}")
            print(f"Predictions (scaled): min={predictions.min():.4f}, max={predictions.max():.4f}")
            
            # Inverse transform predictions back to original scale
            # The target 'intensity_nca' is at index 6 (last feature) in the scaler
            target_idx = self.required_features.index(self.target_column)
            target_mean = self.scaler.mean_[target_idx]
            target_scale = self.scaler.scale_[target_idx]
            
            # Apply inverse transformation: original = (scaled * scale) + mean
            predictions = predictions * target_scale + target_mean
            
            print(f"Predictions (original scale): min={predictions.min():.4f}, max={predictions.max():.4f}")
            
            # Get start years if available
            start_years = []
            if 'year' in df.columns:
                years = df['year'].values
                start_years = [int(years[idx]) if idx < len(years) else idx for idx in start_indices]
            else:
                start_years = start_indices
            
            model_info = self.get_model_info()
            
            return predictions, start_years, model_info
        
        except Exception as e:
            raise Exception(f"Prediction error: {str(e)}")
    
    def format_predictions_as_dataframe(self, predictions: np.ndarray, 
                                       start_years: List[int],
                                       original_df: pd.DataFrame = None) -> pd.DataFrame:
        """
        Format predictions as a readable DataFrame.
        
        Args:
            predictions: Array of shape (n_sequences, horizon)
            start_years: List of starting years
            original_df: Original dataframe (optional, for additional context)
        
        Returns:
            DataFrame with predictions
        """
        results = []
        
        for i, (pred, start_year) in enumerate(zip(predictions, start_years)):
            base_row = {
                'sequence_id': i + 1,
                'base_year_end': start_year - 1,
                'prediction_start_year': start_year
            }
            
            # Add context from original data if available
            if original_df is not None and 'code' in original_df.columns:
                idx = start_year - 1 if isinstance(start_year, int) else i + self.lookback - 1
                if idx < len(original_df):
                    if 'code' in original_df.columns:
                        base_row['code'] = original_df.iloc[idx]['code']
                    if 'countryname' in original_df.columns:
                        base_row['countryname'] = original_df.iloc[idx]['countryname']
            
            # Add predictions for each future year
            for j, value in enumerate(pred, 1):
                base_row[f'{self.target_column}_year_{j}'] = float(value)
                base_row[f'predicted_year_{j}'] = start_year + j - 1
            
            results.append(base_row)
        
        return pd.DataFrame(results)
    
    def save_predictions(self, predictions_df: pd.DataFrame, output_path: str) -> str:
        """Save predictions to CSV."""
        predictions_df.to_csv(output_path, index=False)
        return output_path
