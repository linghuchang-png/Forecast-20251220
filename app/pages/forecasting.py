"""
Forecasting page for the forecasting application.

This module handles forecasting parameter configuration and forecast generation.
"""
import os
import sys
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px 
import plotly.graph_objects as go
import time
from datetime import datetime, timedelta

# Add parent directory to path for importing
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

# Import models and utilities
from models.statistical import ForecasterFactory
from models.optimization import OptimizerFactory, BusinessConstraintGenerator
from models.forecast_generator import ForecastGenerator
from utils.data_processing.cleaner import DataCleaner
from app.state.session import set_error, set_success, clear_messages, update_forecast_params, update_model_params


def show_forecasting_page():
    """Display the forecasting page."""
    st.header("Forecast Generation")
    
    # Clear any previous messages
    clear_messages()
    
    # Check if data is loaded
    if st.session_state.data is None:
        st.warning("Please upload data or generate sample data on the Data Upload page.")
        if st.button("Go to Data Upload"):
            st.query_params(page="data_upload")
            st.rerun()
        return
    
    # Use cleaned data if available, otherwise use original data
    if st.session_state.data_clean is not None:
        data = st.session_state.data_clean
        st.success("Using cleaned data for forecasting.")
    else:
        data = st.session_state.data
        st.info("Using original data for forecasting. Consider cleaning the data first.")
    
    # Create two columns for configurations
    col1, col2 = st.columns([2, 1])
    
    # Left column: Forecast parameters
    with col1:
        show_forecast_parameters(data)
    
    # Right column: Model parameters
    with col2:
        show_model_parameters()
    
    # Forecasting button
    if st.button("Generate Forecast", type="primary", key="generate_forecast"):
        with st.spinner("Generating forecasts... This may take a moment."):
            try:
                generate_forecast(data)
                st.success("Forecast generated successfully!")
            except Exception as e:
                st.error(f"Error generating forecast: {str(e)}")
    
    # Display forecast results if available
    if st.session_state.forecasts is not None:
        show_forecast_results()         


def show_forecast_parameters(data):
    """Show forecast parameter configuration options."""
    st.subheader("Forecast Parameters")
    
    # Get current parameters
    params = st.session_state.forecast_params
    
    # Model selection
    model_type = st.selectbox(
        "Forecasting Method:",
        options=["exponential_smoothing", "moving_average", "seasonal_naive"],
        format_func=lambda x: {
            "exponential_smoothing": "Exponential Smoothing", 
            "moving_average": "Moving Average",
            "seasonal_naive": "Seasonal Naive"
        }.get(x, x),
        index=["exponential_smoothing", "moving_average", "seasonal_naive"].index(params.get('model_type', 'exponential_smoothing')),
        key="model_type"
    )
    
    # Forecast horizon
    horizon = st.slider(
        "Forecast Horizon (months):",
        min_value=1,
        max_value=24,
        value=params.get('horizon', 12),
        key="horizon"
    )
    
    # Column selection
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # Date column selection
        date_options = [col for col in data.columns if 'date' in col.lower() or 
                        pd.api.types.is_datetime64_any_dtype(data[col])]
        
        if not date_options and 'date' in data.columns:
            date_options = ['date']
            
        if date_options:
            date_col = st.selectbox(
                "Date Column:", 
                options=date_options,
                index=0 if params.get('date_col') not in date_options else date_options.index(params.get('date_col')),
                key="date_col"
            )
        else:
            date_col = st.text_input("Date Column:", value=params.get('date_col', 'date'), key="date_col")
            st.warning("No date column detected. Please ensure the specified column exists.")
    
    with col2:
        # Price column selection
        price_options = [col for col in data.columns if 'price' in col.lower() or 
                        (pd.api.types.is_numeric_dtype(data[col]) and 'date' not in col.lower())]
        
        if price_options:
            price_col = st.selectbox(
                "Price Column:", 
                options=price_options,
                index=0 if params.get('price_col') not in price_options else price_options.index(params.get('price_col')),
                key="price_col"
            )
        else:
            price_col = st.text_input("Price Column:", value=params.get('price_col', 'price'), key="price_col")
            st.warning("No suitable price column detected. Please ensure the specified column exists.")
    
    with col3:
        # Lead time column selection
        leadtime_options = [col for col in data.columns if 'lead' in col.lower() or 'time' in col.lower() or
                          (pd.api.types.is_numeric_dtype(data[col]) and col != price_col and 'date' not in col.lower())]
        
        if leadtime_options:
            leadtime_col = st.selectbox(
                "Lead Time Column:", 
                options=leadtime_options,
                index=0 if params.get('leadtime_col') not in leadtime_options else leadtime_options.index(params.get('leadtime_col')),
                key="leadtime_col"
            )
        else:
            leadtime_col = st.text_input("Lead Time Column:", value=params.get('leadtime_col', 'lead_time'), key="leadtime_col")
            st.warning("No suitable lead time column detected. Please ensure the specified column exists.")
    
    # Grouping options
    st.subheader("Grouping Options")
    
    group_options = [col for col in data.columns if col not in [date_col, price_col, leadtime_col] and
                    pd.api.types.is_object_dtype(data[col]) and
                    data[col].nunique() <= 50]  # Limit to columns with reasonable number of unique values
    
    if group_options:
        group_columns = st.multiselect(
            "Group forecasts by:",
            options=group_options,
            default=params.get('group_columns', []),
            key="group_columns",
            help="Select columns to group forecasts by. This will generate separate forecasts for each combination."
        )
        
        if group_columns:
            # Show number of groups that will be generated
            group_count = data.groupby(group_columns).ngroups
            st.info(f"This will generate forecasts for {group_count} different groups.")
            
            # Warning if too many groups
            if group_count > 20:
                st.warning("Large number of groups detected. Forecasting may take a long time.")
    else:
        group_columns = []
        st.info("No suitable grouping columns detected.")
    
    # Optimization options
    st.subheader("Optimization Options")
    
    apply_optimization = st.checkbox(
        "Apply business constraints to forecasts",
        value=params.get('apply_optimization', True),
        key="apply_optimization",
        help="Use optimization to ensure forecasts adhere to business constraints like smoothness and bounds."
    )
    
    # Show advanced optimization options if enabled
    if apply_optimization:
        with st.expander("Advanced Optimization Settings"):
            # Price constraints
            st.write("Price Constraints")
            
            col1, col2 = st.columns(2)
            
            with col1:
                min_price = st.number_input(
                    "Minimum Price (0 = auto):",
                    min_value=0.0,
                    value=params.get('min_price', 0.0),
                    key="min_price"
                )
                
                smoothness_price = st.number_input(
                    "Price Smoothness Factor:",
                    min_value=0.1,
                    max_value=100.0,
                    value=params.get('smoothness_price', 5.0),
                    help="Maximum allowed change between consecutive periods",
                    key="smoothness_price"
                )
            
            with col2:
                max_price = st.number_input(
                    "Maximum Price (0 = auto):",
                    min_value=0.0,
                    value=params.get('max_price', 0.0),
                    key="max_price"
                )
                
                preserve_seasonality = st.checkbox(
                    "Preserve Seasonality",
                    value=params.get('preserve_seasonality', True),
                    key="preserve_seasonality"
                )
            
            # Lead time constraints
            st.write("Lead Time Constraints")
            
            col1, col2 = st.columns(2)
            
            with col1:
                min_leadtime = st.number_input(
                    "Minimum Lead Time (0 = auto):",
                    min_value=0.0,
                    value=params.get('min_leadtime', 0.0),
                    key="min_leadtime"
                )
                
                smoothness_leadtime = st.number_input(
                    "Lead Time Smoothness Factor:",
                    min_value=0.1,
                    max_value=100.0,
                    value=params.get('smoothness_leadtime', 3.0),
                    help="Maximum allowed change between consecutive periods",
                    key="smoothness_leadtime"
                )
            
            with col2:
                max_leadtime = st.number_input(
                    "Maximum Lead Time (0 = auto):",
                    min_value=0.0,
                    value=params.get('max_leadtime', 0.0),
                    key="max_leadtime"
                )
                
                monotonic_leadtime = st.selectbox(
                    "Enforce Trend:",
                    options=[None, "increasing", "decreasing"],
                    index=0,
                    key="monotonic_leadtime"
                )
    
    # Update forecast parameters in session state
    updated_params = {
        'model_type': model_type,
        'horizon': horizon,
        'date_col': date_col,
        'price_col': price_col,
        'leadtime_col': leadtime_col,
        'group_columns': group_columns,
        'apply_optimization': apply_optimization
    }
    
    if apply_optimization:
        updated_params.update({
            'min_price': min_price,
            'max_price': max_price,
            'smoothness_price': smoothness_price,
            'preserve_seasonality': preserve_seasonality,
            'min_leadtime': min_leadtime,
            'max_leadtime': max_leadtime,
            'smoothness_leadtime': smoothness_leadtime,
            'monotonic_leadtime': monotonic_leadtime
        })
    
    update_forecast_params(updated_params)


def show_model_parameters():
    """Show model-specific parameter configuration options."""
    st.subheader("Model Parameters")
    
    # Get current model type
    model_type = st.session_state.forecast_params.get('model_type', 'exponential_smoothing')
    
    # Get model parameters
    model_params = st.session_state.model_params.get(model_type, {})
    
    # Display parameters based on model type
    if model_type == 'exponential_smoothing':
        # Smoothing parameters
        st.write("Smoothing Parameters")
        
        alpha = st.slider(
            "Level (Alpha):",
            min_value=0.01,
            max_value=0.99,
            value=model_params.get('smoothing_level', 0.3),
            format="%.2f",
            key="alpha"
        )
        
        beta = st.slider(
            "Trend (Beta):",
            min_value=0.01,
            max_value=0.99,
            value=model_params.get('smoothing_trend', 0.1),
            format="%.2f",
            key="beta"
        )
        
        gamma = st.slider(
            "Seasonal (Gamma):",
            min_value=0.01,
            max_value=0.99,
            value=model_params.get('smoothing_seasonal', 0.1),
            format="%.2f",
            key="gamma"
        )
        
        # Trend and seasonality options
        st.write("Trend and Seasonality")
        
        trend = st.selectbox(
            "Trend Type:",
            options=["add", "mul", None],
            index=["add", "mul", None].index(model_params.get('trend', 'add')),
            format_func=lambda x: {
                "add": "Additive", 
                "mul": "Multiplicative",
                None: "None"
            }.get(x, x),
            key="trend"
        )
        
        seasonal = st.selectbox(
            "Seasonality Type:",
            options=["add", "mul", None],
            index=["add", "mul", None].index(model_params.get('seasonal', 'add')),
            format_func=lambda x: {
                "add": "Additive", 
                "mul": "Multiplicative",
                None: "None"
            }.get(x, x),
            key="seasonal"
        )
        
        seasonal_periods = st.slider(
            "Seasonal Periods:",
            min_value=2,
            max_value=24,
            value=model_params.get('seasonal_periods', 12),
            key="seasonal_periods"
        )
        
        # Update model parameters
        updated_params = {
            'smoothing_level': alpha,
            'smoothing_trend': beta,
            'smoothing_seasonal': gamma,
            'trend': trend,
            'seasonal': seasonal,
            'seasonal_periods': seasonal_periods
        }
        update_model_params(model_type, updated_params)
        
    elif model_type == 'moving_average':
        # Window size
        window_size = st.slider(
            "Window Size:",
            min_value=1,
            max_value=24,
            value=model_params.get('window_size', 3),
            key="window_size"
        )
        
        # Update model parameters
        updated_params = {'window_size': window_size}
        update_model_params(model_type, updated_params)
        
    elif model_type == 'seasonal_naive':
        # Seasonal periods
        seasonal_periods = st.slider(
            "Seasonal Periods:",
            min_value=2,
            max_value=24,
            value=model_params.get('seasonal_periods', 12),
            key="seasonal_periods"
        )
        
        # Update model parameters
        updated_params = {'seasonal_periods': seasonal_periods}
        update_model_params(model_type, updated_params)
        
    # Add parameter descriptions
    with st.expander("Parameter Descriptions"):
        if model_type == 'exponential_smoothing':
            st.markdown("""
            **Level (Alpha)**: Controls how much the forecast responds to recent observations. Higher values give more weight to recent data.
            
            **Trend (Beta)**: Controls how much the trend component responds to recent changes. Higher values make the trend more responsive.
            
            **Seasonal (Gamma)**: Controls how much the seasonal component changes over time. Higher values allow seasonality to change more quickly.
            
            **Trend Type**:
            - **Additive**: Trend changes by constant amount (better for stable series)
            - **Multiplicative**: Trend changes by percentage (better for growing series)
            - **None**: No trend component
            
            **Seasonality Type**:
            - **Additive**: Seasonal effects are consistent in magnitude
            - **Multiplicative**: Seasonal effects increase/decrease with the level
            - **None**: No seasonality component
            
            **Seasonal Periods**: Number of observations in one seasonal cycle (12 for monthly data)
            """)
            
        elif model_type == 'moving_average':
            st.markdown("""
            **Window Size**: Number of periods to include in the moving average calculation. Larger values create smoother forecasts that are less responsive to recent changes.
            """)
            
        elif model_type == 'seasonal_naive':
            st.markdown("""
            **Seasonal Periods**: Number of observations in one seasonal cycle (12 for monthly data). The forecast for a given period will be equal to the most recent observation from the same season.
            """)


def generate_forecast(data):
    """Generate forecast based on configured parameters."""
    # Get forecast parameters
    params = st.session_state.forecast_params
    model_params = st.session_state.model_params.get(params['model_type'], {})
    
    # Create forecaster based on model type
    if params['model_type'] == 'exponential_smoothing':
        price_model = ForecasterFactory.create_forecaster('exponential_smoothing', **model_params)
        leadtime_model = ForecasterFactory.create_forecaster('exponential_smoothing', **model_params)
    elif params['model_type'] == 'moving_average':
        price_model = ForecasterFactory.create_forecaster('moving_average', **model_params)
        leadtime_model = ForecasterFactory.create_forecaster('moving_average', **model_params)
    elif params['model_type'] == 'seasonal_naive':
        price_model = ForecasterFactory.create_forecaster('seasonal_naive', **model_params)
        leadtime_model = ForecasterFactory.create_forecaster('seasonal_naive', **model_params)
    else:
        # Default to exponential smoothing
        price_model = ForecasterFactory.create_forecaster('exponential_smoothing')
        leadtime_model = ForecasterFactory.create_forecaster('exponential_smoothing')
    
    # Create optimizer if optimization is enabled
    optimizer = None
    if params['apply_optimization']:
        optimizer = OptimizerFactory.create_optimizer('cp_sat')
        
        # Create custom constraints if specified
        custom_constraints = None
        if any(k in params for k in ['min_price', 'max_price', 'smoothness_price']):
            price_constraints = {}
            
            # Add price constraints
            if params.get('min_price', 0) > 0:
                price_constraints['min_value'] = params['min_price']
            
            if params.get('max_price', 0) > 0:
                price_constraints['max_value'] = params['max_price']
            
            if 'smoothness_price' in params:
                price_constraints['smoothness_factor'] = params['smoothness_price']
            
            if 'preserve_seasonality' in params:
                price_constraints['preserve_seasonality'] = params['preserve_seasonality']
            
            # Add lead time constraints
            leadtime_constraints = {}
            
            if params.get('min_leadtime', 0) > 0:
                leadtime_constraints['min_value'] = params['min_leadtime']
            
            if params.get('max_leadtime', 0) > 0:
                leadtime_constraints['max_value'] = params['max_leadtime']
            
            if 'smoothness_leadtime' in params:
                leadtime_constraints['smoothness_factor'] = params['smoothness_leadtime']
            
            if params.get('monotonic_leadtime'):
                leadtime_constraints['monotonic'] = params['monotonic_leadtime']
            
            custom_constraints = {
                'price': price_constraints,
                'leadtime': leadtime_constraints
            }
    
    # Create forecast generator
    forecast_generator = ForecastGenerator(
        price_model=price_model,
        leadtime_model=leadtime_model,
        optimizer=optimizer
    )
    
    # Generate forecast
    st.text("Generating forecasts...")
    progress_bar = st.progress(0)
    
    # Show a progress bar while generating forecasts
    for i in range(5):
        time.sleep(0.1)
        progress_bar.progress((i + 1) / 5)
        
    # Generate forecasts
    forecasts = forecast_generator.generate_forecast(
        data,
        forecast_horizon=params['horizon'],
        price_col=params['price_col'],
        leadtime_col=params['leadtime_col'],
        date_col=params['date_col'],
        group_columns=params.get('group_columns', []),
        apply_optimization=params.get('apply_optimization', True),
        custom_constraints=custom_constraints
    )
    
    # Complete progress bar
    time.sleep(0.1)
    progress_bar.progress(1.0)
    
    # Store forecasts in session state
    st.session_state.forecasts = forecasts


def show_forecast_results():
    """Display the forecast results."""
    forecasts = st.session_state.forecasts
    
    if not forecasts:
        st.warning("No forecast results available.")
        return
    
    st.subheader("Forecast Results")
    
    # Get parameters
    params = st.session_state.forecast_params
    price_col = params['price_col']
    leadtime_col = params['leadtime_col']
    
    # Get forecasts
    price_forecast = forecasts.get('price_forecast')
    leadtime_forecast = forecasts.get('leadtime_forecast')
    
    # Check if we have group columns
    group_columns = params.get('group_columns', [])
    has_groups = len(group_columns) > 0 and all(col in price_forecast.columns for col in group_columns)
    
    # Create tabs for price and lead time forecasts
    forecast_tabs = st.tabs(["Price Forecast", "Lead Time Forecast", "Summary Statistics"])
    
    # Price Forecast tab
    with forecast_tabs[0]:
        st.write("### Price Forecast")
        
        if has_groups:
            # Select group to display
            group_filters = {}
            cols = st.columns(min(3, len(group_columns)))
            
            for i, col_name in enumerate(group_columns):
                with cols[i % len(cols)]:
                    # Get unique values for this column
                    unique_values = sorted(price_forecast[col_name].unique())
                    
                    # Create a selectbox for this column
                    selected = st.selectbox(
                        f"Select {col_name}:",
                        options=unique_values,
                        key=f"price_group_{col_name}"
                    )
                    
                    group_filters[col_name] = selected
            
            # Filter forecast data
            filtered_price = price_forecast.copy()
            for col, val in group_filters.items():
                filtered_price = filtered_price[filtered_price[col] == val]
            
            # Show forecast
            if len(filtered_price) > 0:
                # Create chart
                fig = px.line(
                    filtered_price, 
                    x='date', 
                    y=price_col,
                    title=f"Price Forecast for {', '.join([f'{k}: {v}' for k, v in group_filters.items()])}"
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Show forecast data
                st.write("Forecast Data:")
                st.dataframe(filtered_price, use_container_width=True)
            else:
                st.warning("No forecasts available for the selected filters.")
        else:
            # Show single forecast
            fig = px.line(
                price_forecast, 
                x='date', 
                y=price_col,
                title="Price Forecast"
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Show forecast data
            st.write("Forecast Data:")
            st.dataframe(price_forecast, use_container_width=True)
    
    # Lead Time Forecast tab
    with forecast_tabs[1]:
        st.write("### Lead Time Forecast")
        
        if has_groups:
            # Select group to display
            group_filters = {}
            cols = st.columns(min(3, len(group_columns)))
            
            for i, col_name in enumerate(group_columns):
                with cols[i % len(cols)]:
                    # Get unique values for this column
                    unique_values = sorted(leadtime_forecast[col_name].unique())
                    
                    # Create a selectbox for this column
                    selected = st.selectbox(
                        f"Select {col_name}:",
                        options=unique_values,
                        key=f"leadtime_group_{col_name}"
                    )
                    
                    group_filters[col_name] = selected
            
            # Filter forecast data
            filtered_leadtime = leadtime_forecast.copy()
            for col, val in group_filters.items():
                filtered_leadtime = filtered_leadtime[filtered_leadtime[col] == val]
            
            # Show forecast
            if len(filtered_leadtime) > 0:
                # Create chart
                fig = px.line(
                    filtered_leadtime, 
                    x='date', 
                    y=leadtime_col,
                    title=f"Lead Time Forecast for {', '.join([f'{k}: {v}' for k, v in group_filters.items()])}"
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Show forecast data
                st.write("Forecast Data:")
                st.dataframe(filtered_leadtime, use_container_width=True)
            else:
                st.warning("No forecasts available for the selected filters.")
        else:
            # Show single forecast
            fig = px.line(
                leadtime_forecast, 
                x='date', 
                y=leadtime_col,
                title="Lead Time Forecast"
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Show forecast data
            st.write("Forecast Data:")
            st.dataframe(leadtime_forecast, use_container_width=True)
    
    # Summary Statistics tab
    with forecast_tabs[2]:
        st.write("### Forecast Summary Statistics")
        
        # Price statistics
        st.write("#### Price Forecast Statistics")
        
        if has_groups:
            # Group statistics
            price_stats = price_forecast.groupby(group_columns)[price_col].agg(
                ['count', 'min', 'max', 'mean', 'median', 'std']
            ).reset_index()
            
            st.dataframe(price_stats, use_container_width=True)
            
            # Show distribution by group
            if len(group_columns) == 1:
                # Create box plot by group
                fig = px.box(
                    price_forecast, 
                    x=group_columns[0], 
                    y=price_col,
                    title="Price Forecast Distribution by Group"
                )
                
                st.plotly_chart(fig, use_container_width=True)
        else:
            # Overall statistics
            price_stats = {
                'Count': len(price_forecast),
                'Min': price_forecast[price_col].min(),
                'Max': price_forecast[price_col].max(),
                'Mean': price_forecast[price_col].mean(),
                'Median': price_forecast[price_col].median(),
                'Std Dev': price_forecast[price_col].std()
            }
            
            st.dataframe(pd.DataFrame([price_stats]), use_container_width=True)
        
        # Lead time statistics
        st.write("#### Lead Time Forecast Statistics")
        
        if has_groups:
            # Group statistics
            leadtime_stats = leadtime_forecast.groupby(group_columns)[leadtime_col].agg(
                ['count', 'min', 'max', 'mean', 'median', 'std']
            ).reset_index()
            
            st.dataframe(leadtime_stats, use_container_width=True)
            
            # Show distribution by group
            if len(group_columns) == 1:
                # Create box plot by group
                fig = px.box(
                    leadtime_forecast, 
                    x=group_columns[0], 
                    y=leadtime_col,
                    title="Lead Time Forecast Distribution by Group"
                )
                
                st.plotly_chart(fig, use_container_width=True)
        else:
            # Overall statistics
            leadtime_stats = {
                'Count': len(leadtime_forecast),
                'Min': leadtime_forecast[leadtime_col].min(),
                'Max': leadtime_forecast[leadtime_col].max(),
                'Mean': leadtime_forecast[leadtime_col].mean(),
                'Median': leadtime_forecast[leadtime_col].median(),
                'Std Dev': leadtime_forecast[leadtime_col].std()
            }
            
            st.dataframe(pd.DataFrame([leadtime_stats]), use_container_width=True)
    
    # Actions
    st.subheader("Actions")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("View Detailed Visualizations", type="primary"):
            # Navigate to visualization page
            st.query_params["page"] = "visualization"
            st.rerun()
    
    with col2:
        if st.button("Export Results"):
            # Navigate to export page
            st.query_params(page="export")
            st.rerun()