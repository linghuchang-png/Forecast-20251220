"""
Data cleaning utilities for the forecasting application.

This module handles cleaning of data including handling
missing values, outliers, and data normalization.
"""
import pandas as pd
import numpy as np
from datetime import datetime
from scipy import stats


class DataCleaner:
    """Handles data cleaning operations for forecasting."""
    
    def __init__(self):
        """Initialize the DataCleaner."""
        # Define supported strategies for missing value handling
        self.missing_strategies = {
            'numeric': ['mean', 'median', 'zero', 'interpolate', 'drop', 'ffill', 'bfill'],
            'categorical': ['mode', 'drop', 'ffill', 'bfill']
        }
        
        # Define supported strategies for outlier handling
        self.outlier_strategies = ['clip', 'remove', 'winsorize', 'iqr']
    
    def clean_dataset(self, data, config=None):
        """Clean the dataset based on provided configuration.

        Args:
            data: pandas.DataFrame to clean
            config: Dictionary with column-specific cleaning configurations
                   Format: {column_name: {'missing': strategy, 'outlier': strategy}}
                   If None, will use default strategies

        Returns:
            pandas.DataFrame: Cleaned data
        """
        if data is None or len(data) == 0:
            return data
        
        # Make a copy to avoid modifying the original
        cleaned_data = data.copy()
        
        # If no config provided, create default config
        if config is None:
            config = self._create_default_config(cleaned_data)
        
        # Handle missing values first
        for column, strategies in config.items():
            if column in cleaned_data.columns:
                # Handle missing values
                if 'missing' in strategies and strategies['missing'] is not None:
                    cleaned_data = self._handle_missing_values(
                        cleaned_data, column, strategies['missing']
                    )
                
                # Handle outliers if column is numeric
                if (
                    'outlier' in strategies and 
                    strategies['outlier'] is not None and 
                    pd.api.types.is_numeric_dtype(cleaned_data[column])
                ):
                    cleaned_data = self._handle_outliers(
                        cleaned_data, column, strategies['outlier']
                    )
        
        # Remove rows that still have missing values in key columns
        key_columns = ['date', 'price', 'lead_time']
        key_columns = [col for col in key_columns if col in cleaned_data.columns]
        
        if key_columns:
            cleaned_data = cleaned_data.dropna(subset=key_columns)
        
        return cleaned_data
    
    def _create_default_config(self, data):
        """Create a default cleaning configuration.

        Args:
            data: pandas.DataFrame to create config for

        Returns:
            dict: Default cleaning configuration
        """
        config = {}
        
        for column in data.columns:
            column_config = {}
            
            # Set missing value strategy based on data type
            if pd.api.types.is_numeric_dtype(data[column]):
                if column in ['price', 'lead_time']:
                    column_config['missing'] = 'median'  # Use median for price and lead time
                else:
                    column_config['missing'] = 'mean'    # Use mean for other numeric columns
                
                # Set outlier strategy for numeric columns
                if column in ['price', 'lead_time']:
                    column_config['outlier'] = 'winsorize'  # Use winsorize for price and lead time
                else:
                    column_config['outlier'] = 'clip'       # Use clip for other numeric columns
            
            elif pd.api.types.is_datetime64_any_dtype(data[column]):
                column_config['missing'] = 'drop'  # Drop rows with missing dates
            
            else:  # Categorical or string columns
                column_config['missing'] = 'mode'  # Use mode for categorical columns
            
            config[column] = column_config
        
        return config
    
    def _handle_missing_values(self, data, column, strategy):
        """Handle missing values in a column.

        Args:
            data: pandas.DataFrame containing the column
            column: Name of the column to process
            strategy: Strategy to use for missing values

        Returns:
            pandas.DataFrame: Data with missing values handled
        """
        if data[column].isna().sum() == 0:
            return data  # No missing values to handle
        
        # Make a copy to avoid modifying the original
        result = data.copy()
        
        if strategy == 'drop':
            # Drop rows with missing values in this column
            result = result.dropna(subset=[column])
        
        elif strategy == 'mean' and pd.api.types.is_numeric_dtype(result[column]):
            # Replace with mean
            mean_value = result[column].mean()
            result[column] = result[column].fillna(mean_value)
        
        elif strategy == 'median' and pd.api.types.is_numeric_dtype(result[column]):
            # Replace with median
            median_value = result[column].median()
            result[column] = result[column].fillna(median_value)
        
        elif strategy == 'mode':
            # Replace with mode (works for any data type)
            mode_value = result[column].mode().iloc[0]
            result[column] = result[column].fillna(mode_value)
        
        elif strategy == 'zero' and pd.api.types.is_numeric_dtype(result[column]):
            # Replace with zero
            result[column] = result[column].fillna(0)
        
        elif strategy == 'interpolate' and pd.api.types.is_numeric_dtype(result[column]):
            # Linear interpolation
            result[column] = result[column].interpolate(method='linear')
            
            # Fill any remaining NaNs at the beginning or end
            result[column] = result[column].fillna(method='bfill').fillna(method='ffill')
        
        elif strategy == 'ffill':
            # Forward fill (use previous value)
            result[column] = result[column].fillna(method='ffill')
            
            # If there are still NaNs at the beginning, use backward fill
            if result[column].isna().any():
                result[column] = result[column].fillna(method='bfill')
        
        elif strategy == 'bfill':
            # Backward fill (use next value)
            result[column] = result[column].fillna(method='bfill')
            
            # If there are still NaNs at the end, use forward fill
            if result[column].isna().any():
                result[column] = result[column].fillna(method='ffill')
        
        # If any missing values remain, apply a fallback strategy
        if result[column].isna().any():
            if pd.api.types.is_numeric_dtype(result[column]):
                # For numeric columns, use median as fallback
                median_value = result[column].median()
                if pd.isna(median_value):  # If median is also NaN, use 0
                    median_value = 0
                result[column] = result[column].fillna(median_value)
            else:
                # For non-numeric columns, drop rows as fallback
                result = result.dropna(subset=[column])
        
        return result
    
    def _handle_outliers(self, data, column, strategy, **kwargs):
        """Handle outliers in a column.

        Args:
            data: pandas.DataFrame containing the column
            column: Name of the column to process
            strategy: Strategy to use for outlier detection and handling
            **kwargs: Additional parameters for the strategy

        Returns:
            pandas.DataFrame: Data with outliers handled
        """
        if not pd.api.types.is_numeric_dtype(data[column]):
            return data  # Only handle outliers for numeric columns
        
        # Make a copy to avoid modifying the original
        result = data.copy()
        
        # Get outlier bounds using IQR method by default
        q1 = result[column].quantile(0.25)
        q3 = result[column].quantile(0.75)
        iqr = q3 - q1
        
        # Default: 1.5 * IQR above Q3 or below Q1
        lower_bound = q1 - (1.5 * iqr)
        upper_bound = q3 + (1.5 * iqr)
        
        # For price and lead time, ensure lower bound is not negative
        if column in ['price', 'lead_time']:
            lower_bound = max(0, lower_bound)
        
        if strategy == 'clip':
            # Clip values outside bounds
            result[column] = result[column].clip(lower=lower_bound, upper=upper_bound)
        
        elif strategy == 'remove':
            # Remove rows with outliers
            result = result[(result[column] >= lower_bound) & (result[column] <= upper_bound)]
        
        elif strategy == 'winsorize':
            # Winsorize (set outliers to percentile values)
            lower_pct = kwargs.get('lower_pct', 0.05)
            upper_pct = kwargs.get('upper_pct', 0.95)
            
            lower_val = result[column].quantile(lower_pct)
            upper_val = result[column].quantile(upper_pct)
            
            # For price and lead time, ensure lower bound is not negative
            if column in ['price', 'lead_time']:
                lower_val = max(0, lower_val)
            
            result[column] = result[column].clip(lower=lower_val, upper=upper_val)
        
        elif strategy == 'iqr':
            # Just use IQR method (already calculated bounds)
            result = result[(result[column] >= lower_bound) & (result[column] <= upper_bound)]
        
        return result
    
    def normalize_data(self, data, columns=None, method='minmax'):
        """Normalize numeric data to a standard scale.

        Args:
            data: pandas.DataFrame to normalize
            columns: List of columns to normalize, or None for all numeric columns
            method: Normalization method ('minmax', 'zscore', or 'robust')

        Returns:
            pandas.DataFrame: Normalized data
        """
        # Make a copy to avoid modifying the original
        result = data.copy()
        
        # If no columns specified, use all numeric columns
        if columns is None:
            columns = result.select_dtypes(include=['number']).columns.tolist()
        else:
            # Filter out non-numeric columns
            columns = [col for col in columns if col in result.columns and pd.api.types.is_numeric_dtype(result[col])]
        
        # Apply normalization method
        for column in columns:
            if method == 'minmax':
                # Min-max scaling to [0, 1] range
                min_val = result[column].min()
                max_val = result[column].max()
                
                # Avoid division by zero
                if max_val > min_val:
                    result[column] = (result[column] - min_val) / (max_val - min_val)
                else:
                    result[column] = 0  # If all values are the same
            
            elif method == 'zscore':
                # Z-score normalization
                mean = result[column].mean()
                std = result[column].std()
                
                # Avoid division by zero
                if std > 0:
                    result[column] = (result[column] - mean) / std
                else:
                    result[column] = 0  # If all values are the same
            
            elif method == 'robust':
                # Robust scaling using median and IQR
                median = result[column].median()
                q1 = result[column].quantile(0.25)
                q3 = result[column].quantile(0.75)
                iqr = q3 - q1
                
                # Avoid division by zero
                if iqr > 0:
                    result[column] = (result[column] - median) / iqr
                else:
                    result[column] = 0  # If all values are the same
        
        return result
    
    def detect_anomalies(self, data, column, method='zscore', threshold=3.0):
        """Detect anomalies in a column.

        Args:
            data: pandas.DataFrame containing the column
            column: Name of the column to analyze
            method: Anomaly detection method ('zscore', 'iqr', or 'isolation_forest')
            threshold: Threshold for z-score or IQR method

        Returns:
            pandas.Series: Boolean mask where True indicates anomalies
        """
        if not pd.api.types.is_numeric_dtype(data[column]):
            return pd.Series([False] * len(data))  # Only detect anomalies for numeric columns
        
        if method == 'zscore':
            # Z-score method
            z_scores = np.abs((data[column] - data[column].mean()) / data[column].std())
            return z_scores > threshold
        
        elif method == 'iqr':
            # IQR method
            q1 = data[column].quantile(0.25)
            q3 = data[column].quantile(0.75)
            iqr = q3 - q1
            
            lower_bound = q1 - (threshold * iqr)
            upper_bound = q3 + (threshold * iqr)
            
            return (data[column] < lower_bound) | (data[column] > upper_bound)
        
        elif method == 'isolation_forest':
            try:
                from sklearn.ensemble import IsolationForest
                
                # Reshape for sklearn
                X = data[column].values.reshape(-1, 1)
                
                # Fit Isolation Forest
                iso_forest = IsolationForest(contamination=0.05, random_state=42)
                predictions = iso_forest.fit_predict(X)
                
                # -1 indicates anomaly, 1 indicates normal
                return predictions == -1
    
            except ImportError:
                print("Scikit-learn is required for Isolation Forest method.")
                # Fall back to z-score method
                z_scores = np.abs((data[column] - data[column].mean()) / data[column].std())
                return z_scores > threshold
        
        else:
            # Default to z-score method
            z_scores = np.abs((data[column] - data[column].mean()) / data[column].std())
            return z_scores > threshold