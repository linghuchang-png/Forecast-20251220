"""
Session state management for the forecasting application.

This module handles the application's state across different pages,
providing functions to initialize and update the session state.
"""
import streamlit as st


def initialize_session_state():
    """Initialize session state variables if they don't exist."""
    if 'data' not in st.session_state:
        st.session_state.data = None
    
    if 'data_clean' not in st.session_state:
        st.session_state.data_clean = None
    
    if 'forecasts' not in st.session_state:
        st.session_state.forecasts = None
    
    if 'error_message' not in st.session_state:
        st.session_state.error_message = None
    
    if 'success_message' not in st.session_state:
        st.session_state.success_message = None
    
    if 'forecast_params' not in st.session_state:
        st.session_state.forecast_params = {
            'model_type': 'exponential_smoothing',
            'horizon': 12,
            'date_col': 'date',
            'price_col': 'price',
            'leadtime_col': 'lead_time',
            'group_columns': [],
            'apply_optimization': True,
            'min_price': 0.0,
            'max_price': 0.0,
            'smoothness_price': 5.0,
            'preserve_seasonality': True,
            'min_leadtime': 0.0,
            'max_leadtime': 0.0,
            'smoothness_leadtime': 3.0,
            'monotonic_leadtime': None
        }
    
    if 'model_params' not in st.session_state:
        st.session_state.model_params = {
            'exponential_smoothing': {
                'smoothing_level': 0.3,
                'smoothing_trend': 0.1,
                'smoothing_seasonal': 0.1,
                'trend': 'add',
                'seasonal': 'add',
                'seasonal_periods': 12
            },
            'moving_average': {
                'window_size': 3
            },
            'seasonal_naive': {
                'seasonal_periods': 12
            }
        }


def set_error(message):
    """Set an error message in the session state."""
    st.session_state.error_message = message


def get_error():
    """Get the current error message from the session state."""
    return st.session_state.error_message


def set_success(message):
    """Set a success message in the session state."""
    st.session_state.success_message = message


def get_success():
    """Get the current success message from the session state."""
    return st.session_state.success_message


def clear_messages():
    """Clear both error and success messages."""
    st.session_state.error_message = None
    st.session_state.success_message = None


def update_forecast_params(new_params):
    """Update the forecast parameters in the session state."""
    st.session_state.forecast_params.update(new_params)


def update_model_params(model_type, new_params):
    """Update the model parameters for a specific model type."""
    if model_type in st.session_state.model_params:
        st.session_state.model_params[model_type].update(new_params)
    else:
        st.session_state.model_params[model_type] = new_params