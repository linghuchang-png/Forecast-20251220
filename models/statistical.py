"""
Statistical forecasting models for time series data.

This module implements various statistical forecasting methods
including moving average, exponential smoothing, and seasonal naive models.
"""
import pandas as pd
import numpy as np
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from datetime import datetime, timedelta
from abc import ABC, abstractmethod


class BaseForecaster(ABC):
    """Base class for all forecasting models."""
    
    @abstractmethod
    def fit(self, time_series):
        """Fit the model to historical data."""
        pass
    
    @abstractmethod
    def forecast(self, horizon):
        """Generate forecast for specified horizon."""
        pass
    
    @abstractmethod
    def get_params(self):
        """Get the model parameters."""
        pass
    
    def _prepare_time_series(self, time_series):
        """Prepare time series for forecasting."""
        # Ensure the time series is valid
        if not isinstance(time_series, pd.Series):
            raise TypeError("time_series must be a pandas Series")
        
        # Make a copy to avoid modifying the original
        ts = time_series.copy()
        
        # Sort by index
        ts = ts.sort_index()
        
        # Handle missing values by linear interpolation
        if ts.isna().any():
            ts = ts.interpolate(method='linear')
            # Fill any remaining NaNs at the beginning or end
            ts = ts.fillna(method='bfill').fillna(method='ffill')
        
        return ts
    
    def _create_forecast_index(self, last_date, horizon):
        """Create a date index for the forecast."""
        if isinstance(last_date, pd.Timestamp):
            # Try to infer frequency from the fitted data
            if hasattr(self, 'time_series') and len(self.time_series) > 1:
                freq = pd.infer_freq(self.time_series.index)
                
                if freq is None:
                    # Try to determine frequency from the last two dates
                    td = self.time_series.index[-1] - self.time_series.index[-2]
                    if td.days >= 28 and td.days <= 31:
                        freq = 'MS'  # Monthly
                    elif td.days >= 89 and td.days <= 92:
                        freq = 'QS'  # Quarterly
                    elif td.days == 7:
                        freq = 'W'   # Weekly
                    elif td.days == 1:
                        freq = 'D'   # Daily
                    else:
                        freq = 'D'   # Default to daily
                
                forecast_index = pd.date_range(
                    start=last_date + pd.Timedelta(days=1),
                    periods=horizon,
                    freq=freq
                )
            else:
                # Default to monthly frequency
                forecast_index = pd.date_range(
                    start=last_date + pd.Timedelta(days=1),
                    periods=horizon,
                    freq='MS'
                )
        else:
            # Non-datetime index, create a numeric continuation
            last_idx = self.time_series.index[-1] if hasattr(self, 'time_series') else last_date
            if isinstance(last_idx, (int, float)):
                forecast_index = range(last_idx + 1, last_idx + 1 + horizon)
            else:
                # Create a simple sequential index
                forecast_index = range(
                    len(self.time_series) if hasattr(self, 'time_series') else 0, 
                    len(self.time_series) if hasattr(self, 'time_series') else 0 + horizon
                )
        
        return forecast_index


class MovingAverageForecaster(BaseForecaster):
    """Moving Average forecasting model."""
    
    def __init__(self, window_size=3):
        """Initialize the MovingAverageForecaster.

        Args:
            window_size: Size of the moving window
        """
        self.window_size = max(1, window_size)
        self.time_series = None
        self.last_values = None
        self.fitted = False
    
    def fit(self, time_series):
        """Fit the model to historical data.

        Args:
            time_series: pandas.Series with datetime index

        Returns:
            self: Fitted forecaster
        """
        self.time_series = self._prepare_time_series(time_series)
        
        # Calculate moving average
        if len(self.time_series) >= self.window_size:
            # Store last 'window_size' values for forecasting
            self.last_values = self.time_series.iloc[-self.window_size:].values
            self.fitted = True
        else:
            # Not enough data for the specified window size
            # Just use all available data
            self.last_values = self.time_series.values
            self.fitted = True
            
        return self
    
    def forecast(self, horizon):
        """Generate forecast for specified horizon.

        Args:
            horizon: Number of periods to forecast

        Returns:
            pandas.Series: Forecast values
        """
        if not self.fitted:
            raise ValueError("Model must be fitted before forecasting")
        
        # Create forecast index
        forecast_index = self._create_forecast_index(
            self.time_series.index[-1],
            horizon
        )
        
        # Moving average is simply the average of the last window_size values
        # repeated for the forecast horizon
        forecast_value = np.mean(self.last_values)
        
        # Create forecast series
        forecast = pd.Series(
            data=[forecast_value] * horizon,
            index=forecast_index
        )
        
        return forecast
    
    def get_params(self):
        """Get the model parameters.

        Returns:
            dict: Model parameters
        """
        return {'window_size': self.window_size}


class ExponentialSmoothingForecaster(BaseForecaster):
    """Exponential Smoothing forecasting model using statsmodels."""
    
    def __init__(self, smoothing_level=0.3, smoothing_trend=0.1, smoothing_seasonal=0.1,
                trend=None, seasonal=None, seasonal_periods=12, damped_trend=False):
        """Initialize the ExponentialSmoothingForecaster.

        Args:
            smoothing_level: Alpha parameter (level smoothing)
            smoothing_trend: Beta parameter (trend smoothing)
            smoothing_seasonal: Gamma parameter (seasonal smoothing)
            trend: Type of trend ('add', 'mul', or None)
            seasonal: Type of seasonality ('add', 'mul', or None)
            seasonal_periods: Number of periods in a seasonal cycle
            damped_trend: Whether to use damped trend
        """
        self.smoothing_level = smoothing_level
        self.smoothing_trend = smoothing_trend
        self.smoothing_seasonal = smoothing_seasonal
        self.trend = trend
        self.seasonal = seasonal
        self.seasonal_periods = seasonal_periods
        self.damped_trend = damped_trend
        
        self.time_series = None
        self.model = None
        self.fitted_model = None
        self.fitted = False
    
    def fit(self, time_series):
        """Fit the model to historical data.

        Args:
            time_series: pandas.Series with datetime index

        Returns:
            self: Fitted forecaster
        """
        self.time_series = self._prepare_time_series(time_series)
        
        # Adjust seasonal_periods if not enough data
        if len(self.time_series) < self.seasonal_periods * 2:
            if len(self.time_series) <= 3:
                # Too little data, disable seasonality
                self.seasonal = None
                self.seasonal_periods = 0
            else:
                # Reduce seasonal periods
                self.seasonal_periods = min(self.seasonal_periods, len(self.time_series) // 2)
        
        # Create and fit the model
        try:
            # Use the statsmodels implementation
            self.model = ExponentialSmoothing(
                self.time_series,
                trend=self.trend,
                seasonal=self.seasonal,
                seasonal_periods=self.seasonal_periods if self.seasonal else None,
                damped_trend=self.damped_trend
            )
            
            # Fit the model with provided smoothing parameters
            self.fitted_model = self.model.fit(
                smoothing_level=self.smoothing_level,
                smoothing_trend=self.smoothing_trend if self.trend else None,
                smoothing_seasonal=self.smoothing_seasonal if self.seasonal else None,
                optimized=False
            )
            
            self.fitted = True
            
        except Exception as e:
            # Fall back to simple exponential smoothing
            print(f"Error fitting Holt-Winters: {str(e)}")
            print("Falling back to simple exponential smoothing")
            
            try:
                self.model = ExponentialSmoothing(
                    self.time_series,
                    trend=None,
                    seasonal=None
                )
                
                self.fitted_model = self.model.fit(
                    smoothing_level=self.smoothing_level,
                    optimized=False
                )
                
                self.fitted = True
                
            except Exception as e2:
                print(f"Error fitting simple exponential smoothing: {str(e2)}")
                raise ValueError(f"Could not fit exponential smoothing model: {str(e2)}")
        
        return self
    
    def forecast(self, horizon):
        """Generate forecast for specified horizon.

        Args:
            horizon: Number of periods to forecast

        Returns:
            pandas.Series: Forecast values
        """
        if not self.fitted:
            raise ValueError("Model must be fitted before forecasting")
        
        try:
            # Use statsmodels forecast method
            forecast = self.fitted_model.forecast(horizon)
            return forecast
        except Exception as e:
            print(f"Error forecasting with statsmodels: {str(e)}")
            
            # Fall back to manual forecasting
            # Create forecast index
            forecast_index = self._create_forecast_index(
                self.time_series.index[-1],
                horizon
            )
            
            # Just use the last value repeated
            last_value = self.time_series.iloc[-1]
            forecast = pd.Series(
                data=[last_value] * horizon,
                index=forecast_index
            )
            
            return forecast
    
    def get_params(self):
        """Get the model parameters.

        Returns:
            dict: Model parameters
        """
        return {
            'smoothing_level': self.smoothing_level,
            'smoothing_trend': self.smoothing_trend,
            'smoothing_seasonal': self.smoothing_seasonal,
            'trend': self.trend,
            'seasonal': self.seasonal,
            'seasonal_periods': self.seasonal_periods,
            'damped_trend': self.damped_trend
        }


class SeasonalNaiveForecaster(BaseForecaster):
    """Seasonal Naive forecasting model."""
    
    def __init__(self, seasonal_periods=12):
        """Initialize the SeasonalNaiveForecaster.

        Args:
            seasonal_periods: Number of periods in a seasonal cycle
        """
        self.seasonal_periods = seasonal_periods
        self.time_series = None
        self.last_season = None
        self.fitted = False
    
    def fit(self, time_series):
        """Fit the model to historical data.

        Args:
            time_series: pandas.Series with datetime index

        Returns:
            self: Fitted forecaster
        """
        self.time_series = self._prepare_time_series(time_series)
        
        # Check if we have at least one full season of data
        if len(self.time_series) >= self.seasonal_periods:
            # Store the last full season of values
            self.last_season = self.time_series.iloc[-self.seasonal_periods:].values
            self.fitted = True
        else:
            # Not enough data for a full season
            # Just use all available data and repeat it
            self.last_season = self.time_series.values
            self.fitted = True
        
        return self
    
    def forecast(self, horizon):
        """Generate forecast for specified horizon.

        Args:
            horizon: Number of periods to forecast

        Returns:
            pandas.Series: Forecast values
        """
        if not self.fitted:
            raise ValueError("Model must be fitted before forecasting")
        
        # Create forecast index
        forecast_index = self._create_forecast_index(
            self.time_series.index[-1],
            horizon
        )
        
        # Repeat the last season values as needed
        num_repeats = int(np.ceil(horizon / len(self.last_season)))
        repeated_values = np.tile(self.last_season, num_repeats)
        
        # Take only the needed number of values
        forecast_values = repeated_values[:horizon]
        
        # Create forecast series
        forecast = pd.Series(
            data=forecast_values,
            index=forecast_index
        )
        
        return forecast
    
    def get_params(self):
        """Get the model parameters.

        Returns:
            dict: Model parameters
        """
        return {'seasonal_periods': self.seasonal_periods}


class ForecasterFactory:
    """Factory for creating forecasting models."""
    
    @staticmethod
    def create_forecaster(model_type, **kwargs):
        """Create a forecaster instance.

        Args:
            model_type: Type of forecaster to create
            **kwargs: Parameters for the forecaster

        Returns:
            BaseForecaster: An instance of the requested forecaster
        """
        model_type = model_type.lower()
        
        if model_type == 'moving_average':
            return MovingAverageForecaster(**kwargs)
        
        elif model_type == 'exponential_smoothing':
            return ExponentialSmoothingForecaster(**kwargs)
        
        elif model_type == 'seasonal_naive':
            return SeasonalNaiveForecaster(**kwargs)
        
        else:
            raise ValueError(f"Unknown forecaster type: {model_type}")