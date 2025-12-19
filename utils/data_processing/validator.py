"""
Data validation utilities for the forecasting application.

This module handles validation of input data to ensure it meets
the requirements for forecasting.
"""
import pandas as pd
import numpy as np
from datetime import datetime


class DataValidator:
    """Validates data for forecasting requirements."""
    
    def __init__(self):
        """Initialize the DataValidator."""
        # Define required columns for different validation levels
        self.required_columns = {
            'basic': ['date'],
            'forecasting': ['date', 'price', 'lead_time']
        }
        
        # Minimum number of data points needed for forecasting
        self.min_data_points = 12
    
    def validate_data(self, data, level='basic'):
        """Validate data according to specified level.

        Args:
            data: pandas.DataFrame to validate
            level: Validation level ('basic' or 'forecasting')

        Returns:
            dict: Validation result with 'valid' flag and 'message'
        """
        if data is None or len(data) == 0:
            return {"valid": False, "message": "Data is empty or None"}
        
        # Determine required columns based on validation level
        required_cols = self.required_columns.get(level, self.required_columns['basic'])
        
        # Check for required columns
        missing_cols = [col for col in required_cols if col not in data.columns]
        if missing_cols:
            return {
                "valid": False, 
                "message": f"Missing required columns: {', '.join(missing_cols)}"
            }
        
        # Validate date column
        date_col = 'date'
        if date_col in data.columns:
            if not self._validate_date_column(data, date_col):
                return {
                    "valid": False,
                    "message": f"Invalid date column format or values in '{date_col}'"
                }
        
        # For forecasting level, validate price and lead time columns
        if level == 'forecasting':
            # Validate price column
            price_col = 'price'
            if price_col in data.columns:
                if not self._validate_numeric_column(data, price_col):
                    return {
                        "valid": False,
                        "message": f"Invalid price column format or values in '{price_col}'"
                    }
            
            # Validate lead_time column
            lead_time_col = 'lead_time'
            if lead_time_col in data.columns:
                if not self._validate_numeric_column(data, lead_time_col):
                    return {
                        "valid": False,
                        "message": f"Invalid lead_time column format or values in '{lead_time_col}'"
                    }
            
            # Check if there are enough data points for forecasting
            if len(data) < self.min_data_points:
                return {
                    "valid": False,
                    "message": f"Insufficient data points for forecasting. Need at least {self.min_data_points}, found {len(data)}"
                }
        
        # If all validations passed, return success
        return {"valid": True, "message": "Data validation successful"}
    
    def _validate_date_column(self, data, column):
        """Validate that a column contains valid date values.

        Args:
            data: pandas.DataFrame containing the column
            column: Name of the column to validate

        Returns:
            bool: True if column is valid, False otherwise
        """
        # Check if the column is already in datetime format
        if pd.api.types.is_datetime64_any_dtype(data[column]):
            return True
        
        # Try to convert to datetime
        try:
            pd.to_datetime(data[column])
            return True
        except:
            return False
    
    def _validate_numeric_column(self, data, column):
        """Validate that a column contains valid numeric values.

        Args:
            data: pandas.DataFrame containing the column
            column: Name of the column to validate

        Returns:
            bool: True if column is valid, False otherwise
        """
        # Check if the column is already numeric
        if pd.api.types.is_numeric_dtype(data[column]):
            # Check for NaN values
            if data[column].isna().sum() > 0:
                return False
            
            # Check for negative values in price or lead_time columns
            if column in ['price', 'lead_time'] and (data[column] < 0).any():
                return False
            
            return True
        
        # Try to convert to numeric
        try:
            pd.to_numeric(data[column])
            return True
        except:
            return False
    
    def get_data_quality_report(self, data):
        """Generate a detailed report on data quality issues.

        Args:
            data: pandas.DataFrame to analyze

        Returns:
            dict: Report containing quality metrics and issues
        """
        report = {
            "row_count": len(data),
            "column_count": len(data.columns),
            "missing_values": {},
            "negative_values": {},
            "duplicates": 0,
            "date_range": None,
            "issues": []
        }
        
        # Check for missing values in each column
        for col in data.columns:
            missing_count = data[col].isna().sum()
            if missing_count > 0:
                report["missing_values"][col] = missing_count
                report["issues"].append(f"Column '{col}' has {missing_count} missing values")
        
        # Check for negative values in numeric columns
        for col in data.select_dtypes(include=['number']).columns:
            if col in ['price', 'lead_time']:  # These columns should not be negative
                neg_count = (data[col] < 0).sum()
                if neg_count > 0:
                    report["negative_values"][col] = neg_count
                    report["issues"].append(f"Column '{col}' has {neg_count} negative values")
        
        # Check for duplicates
        duplicates = data.duplicated().sum()
        report["duplicates"] = duplicates
        if duplicates > 0:
            report["issues"].append(f"Data contains {duplicates} duplicate rows")
        
        # Get date range if date column exists
        if 'date' in data.columns:
            if pd.api.types.is_datetime64_any_dtype(data['date']):
                report["date_range"] = {
                    "start": data['date'].min().strftime('%Y-%m-%d'),
                    "end": data['date'].max().strftime('%Y-%m-%d'),
                    "periods": len(data['date'].unique())
                }
            else:
                report["issues"].append("Date column is not in datetime format")
        
        # Check for enough data points per group if grouping columns exist
        group_cols = [col for col in data.columns if col in ['part_number', 'vendor', 'country']]
        if group_cols and 'date' in data.columns:
            # Check each group for enough data points
            for group_col in group_cols:
                group_counts = data.groupby(group_col).size()
                small_groups = group_counts[group_counts < self.min_data_points]
                if not small_groups.empty:
                    report["issues"].append(f"{len(small_groups)} values in '{group_col}' have fewer than {self.min_data_points} data points")
        
        return report
    
    def suggest_fixes(self, data):
        """Suggest fixes for common data issues.

        Args:
            data: pandas.DataFrame to analyze

        Returns:
            dict: Suggested fixes for various issues
        """
        suggestions = {}
        
        # Generate quality report
        report = self.get_data_quality_report(data)
        
        # Suggest fixes for missing values
        if report["missing_values"]:
            missing_fixes = {}
            for col, count in report["missing_values"].items():
                if col in ['price', 'lead_time']:
                    if count / len(data) < 0.1:  # Less than 10% missing
                        missing_fixes[col] = "Replace with mean or median"
                    else:
                        missing_fixes[col] = "Consider removing rows or using more advanced imputation"
                elif col == 'date':
                    missing_fixes[col] = "Remove rows with missing dates"
                else:
                    missing_fixes[col] = "Fill with mode or remove rows"
            
            suggestions["missing_values"] = missing_fixes
        
        # Suggest fixes for negative values
        if report["negative_values"]:
            negative_fixes = {}
            for col, count in report["negative_values"].items():
                negative_fixes[col] = "Replace with absolute values or remove rows"
            
            suggestions["negative_values"] = negative_fixes
        
        # Suggest fixes for duplicates
        if report["duplicates"] > 0:
            suggestions["duplicates"] = "Remove duplicate rows"
        
        # Suggest fixes for date range issues
        if 'date' in data.columns and report.get("date_range") is None:
            suggestions["date_format"] = "Convert date column to datetime format"
        
        # Suggest grouping strategies if multiple categorical columns exist
        categorical_cols = data.select_dtypes(include=['object', 'category']).columns
        if len(categorical_cols) > 2:
            suggestions["grouping"] = f"Consider grouping by one or more of: {', '.join(categorical_cols)}"
        
        return suggestions