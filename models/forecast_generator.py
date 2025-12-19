"""
Forecast generator module that integrates statistical models with optimization.

This module handles the complete forecasting workflow from data preparation
to forecast generation and optimization.
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import concurrent.futures
import warnings

from models.statistical import ForecasterFactory
from models.optimization import OptimizerFactory, BusinessConstraintGenerator
from models.time_series_analyzer import TimeSeriesAnalyzer


class ForecastGenerator:
    """Generates price and lead time forecasts based on historical data."""
    
    def __init__(self, price_model, leadtime_model, optimizer=None):
        """Initialize the ForecastGenerator.

        Args:
            price_model: Statistical forecasting model for price
            leadtime_model: Statistical forecasting model for lead time
            optimizer: Optional optimizer for applying business constraints
        """
        self.price_model = price_model
        self.leadtime_model = leadtime_model
        self.optimizer = optimizer
        self.time_series_analyzer = TimeSeriesAnalyzer()
    
    def generate_forecast(self, data, forecast_horizon=12, price_col='price', 
                         leadtime_col='lead_time', date_col='date', 
                         group_columns=None, apply_optimization=True,
                         custom_constraints=None, parallel=True):
        """Generate forecasts for price and lead time.

        Args:
            data: pandas.DataFrame with historical data
            forecast_horizon: Number of periods to forecast
            price_col: Name of the price column
            leadtime_col: Name of the lead time column
            date_col: Name of the date column
            group_columns: List of columns to group by
            apply_optimization: Whether to apply optimization
            custom_constraints: Custom constraints for optimization
            parallel: Whether to use parallel processing for groups

        Returns:
            dict: Forecast results
        """
        # Ensure the date column is in datetime format
        if not pd.api.types.is_datetime64_any_dtype(data[date_col]):
            data = data.copy()
            data[date_col] = pd.to_datetime(data[date_col])
        
        # Set a proper index for forecasting
        data = data.sort_values(date_col)
        
        # Create result container
        results = {
            'price_forecast': None,
            'leadtime_forecast': None
        }
        
        # If we have group columns, forecast for each group
        if group_columns and len(group_columns) > 0:
            if all(col in data.columns for col in group_columns):
                price_forecasts = []
                leadtime_forecasts = []
                
                # Get unique combinations of group values
                groups = data.groupby(group_columns)
                
                if parallel and len(groups) > 1:
                    # Parallel processing for multiple groups
                    with concurrent.futures.ThreadPoolExecutor(max_workers=min(8, len(groups))) as executor:
                        # Create forecast tasks
                        tasks = []
                        for name, group_data in groups:
                            if isinstance(name, tuple):
                                group_dict = {group_columns[i]: name[i] for i in range(len(group_columns))}
                            else:
                                group_dict = {group_columns[0]: name}
                                
                            task = executor.submit(
                                self._generate_group_forecast,
                                group_data,
                                forecast_horizon,
                                price_col,
                                leadtime_col,
                                date_col,
                                group_dict,
                                apply_optimization,
                                custom_constraints
                            )
                            tasks.append(task)
                        
                        # Process results as they complete
                        for future in concurrent.futures.as_completed(tasks):
                            try:
                                price_forecast, leadtime_forecast = future.result()
                                
                                if price_forecast is not None:
                                    price_forecasts.append(price_forecast)
                                
                                if leadtime_forecast is not None:
                                    leadtime_forecasts.append(leadtime_forecast)
                            except Exception as e:
                                print(f"Error in group forecast: {str(e)}")
                else:
                    # Sequential processing for groups
                    for name, group_data in groups:
                        try:
                            if isinstance(name, tuple):
                                group_dict = {group_columns[i]: name[i] for i in range(len(group_columns))}
                            else:
                                group_dict = {group_columns[0]: name}
                                
                            price_forecast, leadtime_forecast = self._generate_group_forecast(
                                group_data,
                                forecast_horizon,
                                price_col,
                                leadtime_col,
                                date_col,
                                group_dict,
                                apply_optimization,
                                custom_constraints
                            )
                            
                            if price_forecast is not None:
                                price_forecasts.append(price_forecast)
                            
                            if leadtime_forecast is not None:
                                leadtime_forecasts.append(leadtime_forecast)
                                
                        except Exception as e:
                            print(f"Error forecasting for group {name}: {str(e)}")
                
                # Combine group forecasts into a single dataframe
                if price_forecasts:
                    results['price_forecast'] = pd.concat(price_forecasts, ignore_index=False)
                
                if leadtime_forecasts:
                    results['leadtime_forecast'] = pd.concat(leadtime_forecasts, ignore_index=False)
            else:
                print(f"Warning: Group columns {group_columns} not found in data. Ignoring grouping.")
                # Generate forecast without grouping
                results = self._generate_overall_forecast(
                    data,
                    forecast_horizon,
                    price_col,
                    leadtime_col,
                    date_col,
                    apply_optimization,
                    custom_constraints
                )
        else:
            # Generate forecast without grouping
            results = self._generate_overall_forecast(
                data,
                forecast_horizon,
                price_col,
                leadtime_col,
                date_col,
                apply_optimization,
                custom_constraints
            )
        
        return results
    
    def _generate_overall_forecast(self, data, forecast_horizon, price_col,
                                 leadtime_col, date_col, apply_optimization,
                                 custom_constraints):
        """Generate forecasts for the entire dataset.

        Args:
            data: pandas.DataFrame with historical data
            forecast_horizon: Number of periods to forecast
            price_col: Name of the price column
            leadtime_col: Name of the lead time column
            date_col: Name of the date column
            apply_optimization: Whether to apply optimization
            custom_constraints: Custom constraints for optimization

        Returns:
            dict: Forecast results
        """
        results = {
            'price_forecast': None,
            'leadtime_forecast': None
        }
        
        # Create time series for price and lead time
        price_series = self._create_time_series(data, date_col, price_col)
        leadtime_series = self._create_time_series(data, date_col, leadtime_col)
        
        # Generate price forecast
        if price_series is not None:
            try:
                # Generate price forecast
                price_forecast = self._generate_forecast_for_series(
                    price_series,
                    self.price_model,
                    forecast_horizon,
                    apply_optimization,
                    custom_constraints.get('price') if custom_constraints else None,
                    'price'
                )
                
                # Convert to dataframe
                price_df = price_forecast.reset_index()
                price_df.columns = [date_col, price_col]
                
                results['price_forecast'] = price_df
                
            except Exception as e:
                print(f"Error generating price forecast: {str(e)}")
        
        # Generate lead time forecast
        if leadtime_series is not None:
            try:
                # Generate lead time forecast
                leadtime_forecast = self._generate_forecast_for_series(
                    leadtime_series,
                    self.leadtime_model,
                    forecast_horizon,
                    apply_optimization,
                    custom_constraints.get('leadtime') if custom_constraints else None,
                    'leadtime'
                )
                
                # Convert to dataframe
                leadtime_df = leadtime_forecast.reset_index()
                leadtime_df.columns = [date_col, leadtime_col]
                
                results['leadtime_forecast'] = leadtime_df
                
            except Exception as e:
                print(f"Error generating lead time forecast: {str(e)}")
        
        return results
    
    def _generate_group_forecast(self, group_data, forecast_horizon, price_col,
                               leadtime_col, date_col, group_dict, apply_optimization,
                               custom_constraints):
        """Generate forecasts for a specific group.

        Args:
            group_data: pandas.DataFrame with data for a specific group
            forecast_horizon: Number of periods to forecast
            price_col: Name of the price column
            leadtime_col: Name of the lead time column
            date_col: Name of the date column
            group_dict: Dictionary with group column values
            apply_optimization: Whether to apply optimization
            custom_constraints: Custom constraints for optimization

        Returns:
            tuple: Price and lead time forecast dataframes
        """
        # Create time series for price and lead time
        price_series = self._create_time_series(group_data, date_col, price_col)
        leadtime_series = self._create_time_series(group_data, date_col, leadtime_col)
        
        price_forecast_df = None
        leadtime_forecast_df = None
        
        # Generate price forecast
        if price_series is not None and len(price_series) > 0:
            try:
                # Generate custom constraints if not provided
                if custom_constraints is None or 'price' not in custom_constraints:
                    group_constraints = BusinessConstraintGenerator.generate_constraints(
                        group_data, price_col
                    )
                else:
                    group_constraints = custom_constraints.get('price')
                
                # Generate price forecast
                price_forecast = self._generate_forecast_for_series(
                    price_series,
                    self.price_model,
                    forecast_horizon,
                    apply_optimization,
                    group_constraints,
                    'price'
                )
                
                # Convert to dataframe
                price_forecast_df = price_forecast.reset_index()
                price_forecast_df.columns = [date_col, price_col]
                
                # Add group columns
                for col, val in group_dict.items():
                    price_forecast_df[col] = val
                
            except Exception as e:
                print(f"Error generating price forecast for group {group_dict}: {str(e)}")
        
        # Generate lead time forecast
        if leadtime_series is not None and len(leadtime_series) > 0:
            try:
                # Generate custom constraints if not provided
                if custom_constraints is None or 'leadtime' not in custom_constraints:
                    group_constraints = BusinessConstraintGenerator.generate_constraints(
                        group_data, leadtime_col
                    )
                else:
                    group_constraints = custom_constraints.get('leadtime')
                
                # Generate lead time forecast
                leadtime_forecast = self._generate_forecast_for_series(
                    leadtime_series,
                    self.leadtime_model,
                    forecast_horizon,
                    apply_optimization,
                    group_constraints,
                    'leadtime'
                )
                
                # Convert to dataframe
                leadtime_forecast_df = leadtime_forecast.reset_index()
                leadtime_forecast_df.columns = [date_col, leadtime_col]
                
                # Add group columns
                for col, val in group_dict.items():
                    leadtime_forecast_df[col] = val
                
            except Exception as e:
                print(f"Error generating lead time forecast for group {group_dict}: {str(e)}")
        
        return price_forecast_df, leadtime_forecast_df
    
    def _create_time_series(self, data, date_col, value_col):
        """Create a time series for forecasting.

        Args:
            data: pandas.DataFrame with historical data
            date_col: Name of the date column
            value_col: Name of the value column

        Returns:
            pandas.Series: Time series with date index and values
        """
        if date_col not in data.columns or value_col not in data.columns:
            return None
        
        # Create a copy to avoid modifying original data
        df = data[[date_col, value_col]].copy()
        
        # Convert date to datetime if needed
        if not pd.api.types.is_datetime64_any_dtype(df[date_col]):
            df[date_col] = pd.to_datetime(df[date_col])
        
        # Ensure values are numeric
        if not pd.api.types.is_numeric_dtype(df[value_col]):
            try:
                df[value_col] = pd.to_numeric(df[value_col])
            except Exception as e:
                print(f"Error converting {value_col} to numeric: {str(e)}")
                return None
        
        # Remove rows with NaN values
        df = df.dropna()
        
        # If multiple values per date, aggregate
        if df.groupby(date_col).size().max() > 1:
            series = df.groupby(date_col)[value_col].mean()
        else:
            # Set date as index
            df = df.set_index(date_col)
            series = df[value_col]
        
        # Sort by date
        series = series.sort_index()
        
        return series
    
    def _generate_forecast_for_series(self, time_series, model, horizon,
                                     apply_optimization, constraints, forecast_type):
        """Generate a forecast for a single time series.

        Args:
            time_series: pandas.Series with historical data
            model: Statistical forecasting model
            horizon: Number of periods to forecast
            apply_optimization: Whether to apply optimization
            constraints: Custom constraints for optimization
            forecast_type: Type of forecast ('price' or 'leadtime')

        Returns:
            pandas.Series: Forecast values
        """
        # Clone the model to avoid side effects
        model_instance = type(model)(**model.get_params())
        
        # Fit the model
        model_instance.fit(time_series)
        
        # Generate forecast
        forecast = model_instance.forecast(horizon)
        
        # Apply optimization if requested
        if apply_optimization and self.optimizer is not None:
            try:
                optimized_forecast = self.optimizer.optimize_forecast(
                    forecast,
                    historical_data=time_series,
                    custom_constraints=constraints
                )
                
                forecast = optimized_forecast
                
            except Exception as e:
                print(f"Error optimizing {forecast_type} forecast: {str(e)}")
                warnings.warn(f"Optimization failed: {str(e)}. Using unoptimized forecast.")
        
        return forecast
    
    def evaluate_forecast(self, forecast, actual, metric='rmse'):
        """Evaluate forecast accuracy.

        Args:
            forecast: pandas.Series with forecast values
            actual: pandas.Series with actual values
            metric: Metric to use ('rmse', 'mae', 'mape')

        Returns:
            float: Evaluation result
        """
        # Ensure indexes match
        common_index = forecast.index.intersection(actual.index)
        
        if len(common_index) == 0:
            return None
        
        forecast = forecast.loc[common_index]
        actual = actual.loc[common_index]
        
        if metric == 'rmse':
            # Root Mean Squared Error
            return np.sqrt(((forecast - actual) ** 2).mean())
        
        elif metric == 'mae':
            # Mean Absolute Error
            return (forecast - actual).abs().mean()
        
        elif metric == 'mape':
            # Mean Absolute Percentage Error
            # Avoid division by zero
            mask = actual != 0
            return (((forecast[mask] - actual[mask]).abs() / actual[mask]) * 100).mean()
        
        else:
            raise ValueError(f"Unknown metric: {metric}")
    
    def get_forecast_seasonality(self, historical_data, column, date_col='date'):
        """Analyze seasonality pattern in data.

        Args:
            historical_data: pandas.DataFrame with historical data
            column: Column to analyze
            date_col: Date column name

        Returns:
            dict: Seasonality information
        """
        # Create a time series
        series = self._create_time_series(historical_data, date_col, column)
        
        if series is None or len(series) < 12:
            return None
        
        # Detect seasonality
        try:
            period = self.time_series_analyzer.detect_seasonality(series)
            
            # Decompose time series
            decomposition = self.time_series_analyzer.decompose_time_series(
                series, period=period
            )
            
            # Extract seasonality
            if 'seasonal' in decomposition:
                seasonal = decomposition['seasonal']
                
                # Get seasonal pattern by month (for monthly data)
                if isinstance(series.index, pd.DatetimeIndex) and period == 12:
                    seasonal_pattern = seasonal.groupby(seasonal.index.month).mean()
                    
                    # Normalize to percentage difference from average
                    normalized = (seasonal_pattern / seasonal_pattern.mean() - 1) * 100
                    
                    return {
                        'period': period,
                        'seasonal_pattern': normalized.to_dict(),
                        'amplitude': (seasonal.max() - seasonal.min()) / 2,
                        'peak_month': normalized.idxmax(),
                        'trough_month': normalized.idxmin()
                    }
                else:
                    return {
                        'period': period,
                        'amplitude': (seasonal.max() - seasonal.min()) / 2
                    }
        except Exception as e:
            print(f"Error analyzing seasonality: {str(e)}")
            
        return None