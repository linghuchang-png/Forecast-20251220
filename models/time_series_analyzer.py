"""
Time Series analysis for decomposing trends and seasonality.

This module handles decomposition of time series data into trend, 
seasonal, and residual components for analysis and forecasting.
"""
import pandas as pd
import numpy as np
from statsmodels.tsa.seasonal import seasonal_decompose
import matplotlib.pyplot as plt
from datetime import datetime, timedelta


class TimeSeriesAnalyzer:
    """Analyzes time series data for patterns and components."""
    
    def __init__(self):
        """Initialize the TimeSeriesAnalyzer."""
        pass
    
    def decompose_time_series(self, time_series, period=12, model='additive'):
        """Decompose a time series into trend, seasonal, and residual components.

        Args:
            time_series: pandas.Series with datetime index
            period: Number of periods in a seasonal cycle (12 for monthly)
            model: Decomposition model ('additive' or 'multiplicative')

        Returns:
            dict: Decomposition components
        """
        # Ensure the time series is valid
        if not isinstance(time_series, pd.Series):
            raise TypeError("time_series must be a pandas Series")
        
        if not pd.api.types.is_datetime64_dtype(time_series.index):
            raise TypeError("time_series index must be a datetime index")
        
        # Sort by index
        time_series = time_series.sort_index()
        
        # Handle missing values by linear interpolation
        if time_series.isna().any():
            time_series = time_series.interpolate(method='linear')
        
        # For very short time series, adjust the period
        if len(time_series) < period * 2:
            period = min(period, len(time_series) // 2)
            if period < 2:
                period = 2
        
        # Perform decomposition
        try:
            result = seasonal_decompose(
                time_series, 
                model=model, 
                period=period,
                extrapolate_trend='freq'
            )
            
            return {
                'trend': result.trend,
                'seasonal': result.seasonal,
                'residual': result.resid,
                'original': time_series
            }
        except Exception as e:
            print(f"Error in seasonal decomposition: {str(e)}")
            # Return a simplified decomposition
            return self._simple_decomposition(time_series, period, model)
    
    def _simple_decomposition(self, time_series, period=12, model='additive'):
        """Simplified decomposition when statsmodels fails.

        Args:
            time_series: pandas.Series with datetime index
            period: Number of periods in a seasonal cycle (12 for monthly)
            model: Decomposition model ('additive' or 'multiplicative')

        Returns:
            dict: Simplified decomposition components
        """
        # Create a moving average for trend
        trend = time_series.rolling(window=period, center=True).mean()
        
        # Fill missing values at the ends
        trend = trend.interpolate(method='linear').fillna(method='bfill').fillna(method='ffill')
        
        # Detrend the series
        if model == 'additive':
            detrended = time_series - trend
        else:  # multiplicative
            detrended = time_series / trend
        
        # Create a seasonal component (average by period)
        if 'month' in str(time_series.index.dtype):
            # For monthly data, use month as seasonal index
            seasonal_index = time_series.index.month
        elif 'day' in str(time_series.index.dtype):
            # For daily data, use day of week
            seasonal_index = time_series.index.dayofweek
        else:
            # Create a sequential period index
            seasonal_index = np.arange(len(time_series)) % period
        
        # Group by seasonal index and get mean of detrended values
        seasonal_means = detrended.groupby(seasonal_index).mean()
        
        # Create seasonal component by mapping back from seasonal index
        seasonal = pd.Series(
            index=time_series.index,
            data=[seasonal_means[idx] for idx in seasonal_index]
        )
        
        # Calculate residual
        if model == 'additive':
            residual = time_series - trend - seasonal
        else:  # multiplicative
            residual = time_series / (trend * seasonal)
        
        return {
            'trend': trend,
            'seasonal': seasonal,
            'residual': residual,
            'original': time_series
        }
    
    def extract_features(self, time_series, period=12):
        """Extract features from time series for forecasting.

        Args:
            time_series: pandas.Series with datetime index
            period: Number of periods in a seasonal cycle (12 for monthly)

        Returns:
            dict: Extracted features
        """
        features = {}
        
        # Basic statistics
        features['mean'] = time_series.mean()
        features['median'] = time_series.median()
        features['std'] = time_series.std()
        features['min'] = time_series.min()
        features['max'] = time_series.max()
        
        # Trend features
        try:
            # Linear regression slope
            x = np.arange(len(time_series))
            y = time_series.values
            slope, intercept = np.polyfit(x, y, 1)
            features['trend_slope'] = slope
            
            # Positive slope indicates upward trend
            features['trend_direction'] = 'increasing' if slope > 0 else 'decreasing'
            
            # Trend strength - R² of the linear regression
            y_hat = slope * x + intercept
            ss_total = np.sum((y - y.mean()) ** 2)
            ss_residual = np.sum((y - y_hat) ** 2)
            r_squared = 1 - (ss_residual / ss_total)
            features['trend_strength'] = r_squared
            
        except Exception as e:
            print(f"Error calculating trend features: {str(e)}")
            features['trend_slope'] = 0
            features['trend_direction'] = 'unknown'
            features['trend_strength'] = 0
        
        # Seasonality features
        try:
            # Decompose the series
            decomposition = self.decompose_time_series(time_series, period=period)
            seasonal = decomposition['seasonal']
            
            # Seasonal amplitude - half the range of the seasonal component
            amplitude = (seasonal.max() - seasonal.min()) / 2
            features['seasonal_amplitude'] = amplitude
            
            # Seasonal strength - variance of seasonal component / variance of deseasonalized series
            seasonal_var = seasonal.var()
            deseasonal_var = (time_series - seasonal).var()
            if deseasonal_var > 0:
                seasonal_strength = seasonal_var / (seasonal_var + deseasonal_var)
            else:
                seasonal_strength = 0
                
            features['seasonal_strength'] = seasonal_strength
            
            # Get peaks and troughs months (for monthly data)
            if period == 12 and len(seasonal) >= 12:
                avg_by_month = seasonal.groupby(seasonal.index.month).mean()
                features['peak_month'] = avg_by_month.idxmax()
                features['trough_month'] = avg_by_month.idxmin()
            
        except Exception as e:
            print(f"Error calculating seasonality features: {str(e)}")
            features['seasonal_amplitude'] = 0
            features['seasonal_strength'] = 0
            features['peak_month'] = None
            features['trough_month'] = None
        
        # Autocorrelation features
        try:
            # First-order autocorrelation
            acf1 = time_series.autocorr(lag=1)
            features['autocorrelation_lag1'] = acf1
            
            # Autocorrelation at seasonal lag
            if len(time_series) > period:
                acf_seasonal = time_series.autocorr(lag=period)
                features['autocorrelation_seasonal'] = acf_seasonal
            else:
                features['autocorrelation_seasonal'] = 0
                
        except Exception as e:
            print(f"Error calculating autocorrelation features: {str(e)}")
            features['autocorrelation_lag1'] = 0
            features['autocorrelation_seasonal'] = 0
        
        # Volatility features
        features['volatility'] = time_series.std() / time_series.mean() if time_series.mean() != 0 else 0
        
        # Recent trend
        if len(time_series) >= 3:
            last_values = time_series.values[-3:]
            if last_values[2] > last_values[1] > last_values[0]:
                features['recent_trend'] = 'increasing'
            elif last_values[2] < last_values[1] < last_values[0]:
                features['recent_trend'] = 'decreasing'
            else:
                features['recent_trend'] = 'mixed'
        else:
            features['recent_trend'] = 'unknown'
        
        return features
    
    def detect_seasonality(self, time_series, max_lag=24):
        """Detect seasonality period in a time series.

        Args:
            time_series: pandas.Series with datetime index
            max_lag: Maximum lag to consider for seasonality

        Returns:
            int: Detected seasonality period
        """
        # Ensure enough data
        if len(time_series) < max_lag * 2:
            max_lag = len(time_series) // 2
        
        if max_lag < 2:
            return 1  # Not enough data for seasonality detection
        
        # Calculate autocorrelation for different lags
        acf = [time_series.autocorr(lag=lag) for lag in range(1, max_lag + 1)]
        
        # Find peaks in autocorrelation
        peaks = []
        for i in range(1, len(acf) - 1):
            if acf[i] > acf[i-1] and acf[i] > acf[i+1] and acf[i] > 0.2:
                peaks.append((i + 1, acf[i]))  # lag and correlation
        
        if not peaks:
            # No significant peak found
            # Check if monthly (12) or quarterly (4) based on time index
            if isinstance(time_series.index, pd.DatetimeIndex):
                freq = pd.infer_freq(time_series.index)
                if freq == 'MS' or freq == 'M':
                    return 12  # Monthly data
                elif freq in ['QS', 'Q']:
                    return 4  # Quarterly data
                else:
                    return 1  # Default to no seasonality
            else:
                return 1  # Default to no seasonality
        
        # Sort peaks by correlation value
        peaks.sort(key=lambda x: x[1], reverse=True)
        
        # Return the lag with highest autocorrelation
        return peaks[0][0]
    
    def forecast_naive(self, time_series, forecast_horizon, method='seasonal_naive', seasonal_period=12):
        """Create a naive forecast based on historical data.

        Args:
            time_series: pandas.Series with datetime index
            forecast_horizon: Number of periods to forecast
            method: Forecasting method ('naive', 'seasonal_naive', 'average')
            seasonal_period: Number of periods in a seasonal cycle

        Returns:
            pandas.Series: Naive forecast
        """
        # Ensure the time series is valid
        if not isinstance(time_series, pd.Series):
            raise TypeError("time_series must be a pandas Series")
        
        # Sort by index
        time_series = time_series.sort_index()
        
        # Handle missing values by linear interpolation
        if time_series.isna().any():
            time_series = time_series.interpolate(method='linear')
        
        # Create forecast index continuing from the last date
        last_date = time_series.index[-1]
        
        if isinstance(last_date, pd.Timestamp):
            # Infer frequency from the index
            freq = pd.infer_freq(time_series.index)
            
            if freq is None:
                # Try to determine frequency from the last two dates
                if len(time_series) >= 2:
                    td = time_series.index[-1] - time_series.index[-2]
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
                else:
                    freq = 'D'  # Default to daily
            
            forecast_index = pd.date_range(
                start=last_date + pd.Timedelta(days=1),
                periods=forecast_horizon,
                freq=freq
            )
        else:
            # Non-datetime index, create a numeric continuation
            last_idx = time_series.index[-1]
            if isinstance(last_idx, (int, float)):
                forecast_index = range(last_idx + 1, last_idx + 1 + forecast_horizon)
            else:
                # Create a simple sequential index
                forecast_index = range(len(time_series), len(time_series) + forecast_horizon)
        
        # Create the forecast based on the specified method
        if method == 'naive':
            # Use the last value for all future periods
            last_value = time_series.iloc[-1]
            forecast = pd.Series([last_value] * forecast_horizon, index=forecast_index)
            
        elif method == 'seasonal_naive':
            # Use values from the last season
            if len(time_series) < seasonal_period:
                # Fall back to simple naive if not enough data
                last_value = time_series.iloc[-1]
                forecast = pd.Series([last_value] * forecast_horizon, index=forecast_index)
            else:
                # Get the last season values and repeat as needed
                last_season = time_series.iloc[-seasonal_period:].values
                forecast_values = np.tile(last_season, (forecast_horizon // seasonal_period) + 1)[:forecast_horizon]
                forecast = pd.Series(forecast_values, index=forecast_index)
            
        elif method == 'average':
            # Use the average of all historical values
            avg_value = time_series.mean()
            forecast = pd.Series([avg_value] * forecast_horizon, index=forecast_index)
            
        else:
            # Default to naive method
            last_value = time_series.iloc[-1]
            forecast = pd.Series([last_value] * forecast_horizon, index=forecast_index)
        
        return forecast