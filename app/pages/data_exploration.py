"""
Data Exploration page for the forecasting application.

This module handles data exploration, cleaning, and analysis functionality.
"""
import os
import sys
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime

# Add parent directory to path for importing
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

# Import utilities
from utils.data_processing.cleaner import DataCleaner
from utils.data_processing.validator import DataValidator
from models.time_series_analyzer import TimeSeriesAnalyzer
from app.state.session import set_error, set_success, clear_messages


def show_data_exploration_page():
    """Display the data exploration page."""
    st.header("Data Exploration & Cleaning")
    
    # Clear any previous messages
    clear_messages()
    
    # Check if data is loaded
    if st.session_state.data is None:
        st.warning("Please upload data or generate sample data on the Data Upload page.")
        if st.button("Go to Data Upload"):
            st.query_params(page="data_upload")
            st.rerun()
        return
    
    # Get data
    data = st.session_state.data
    
    # Create tabs for different exploration/cleaning tasks
    tabs = st.tabs([
        "Data Overview", 
        "Data Cleaning", 
        "Time Series Analysis", 
        "Correlation Analysis"
    ])
    
    # Data Overview tab
    with tabs[0]:
        show_data_overview(data)
    
    # Data Cleaning tab
    with tabs[1]:
        show_data_cleaning(data)
    
    # Time Series Analysis tab
    with tabs[2]:
        show_time_series_analysis(data)
    
    # Correlation Analysis tab
    with tabs[3]:
        show_correlation_analysis(data)


def show_data_overview(data):
    """Show data overview with summary statistics and charts."""
    st.subheader("Data Overview")
    
    # Display data shape and time range
    row_count, col_count = data.shape
    st.write(f"Dataset has {row_count} rows and {col_count} columns")
    
    if 'date' in data.columns:
        date_col = 'date'
        if not pd.api.types.is_datetime64_any_dtype(data[date_col]):
            data[date_col] = pd.to_datetime(data[date_col])
            
        date_range = f"{data[date_col].min().strftime('%Y-%m-%d')} to {data[date_col].max().strftime('%Y-%m-%d')}"
        st.write(f"Date range: {date_range}")
    
    # Data filtering
    st.subheader("Data Filtering")
    
    filter_cols = []
    if 'part_number' in data.columns:
        filter_cols.append('part_number')
    if 'vendor' in data.columns:
        filter_cols.append('vendor')
    if 'country' in data.columns:
        filter_cols.append('country')
    
    filtered_data = data.copy()
    
    if filter_cols:
        # Create expander for filters to save space
        with st.expander("Filter Data", expanded=True):
            # Create columns for filters
            columns = st.columns(min(3, len(filter_cols)))
            
            selected_filters = {}
            
            # Add each filter as a multi-select
            for i, col in enumerate(filter_cols):
                with columns[i % len(columns)]:
                    # Limit options to top 50 for performance
                    top_values = data[col].value_counts().head(50).index.tolist()
                    
                    if len(top_values) > 1:
                        selected = st.multiselect(
                            f"Select {col}:",
                            options=top_values,
                            default=None,
                            key=f"filter_{col}"
                        )
                        
                        if selected:
                            selected_filters[col] = selected
            
            # Apply filters if any selected
            for col, values in selected_filters.items():
                filtered_data = filtered_data[filtered_data[col].isin(values)]
            
            # Show filter info
            if selected_filters:
                st.info(f"Showing {len(filtered_data)} of {len(data)} rows based on filters")
    
    # Data preview
    st.subheader("Data Preview")
    st.dataframe(filtered_data.head(10), use_container_width=True)
    
    # Summary statistics for numeric columns
    st.subheader("Summary Statistics")
    
    numeric_cols = filtered_data.select_dtypes(include=['number']).columns
    if not numeric_cols.empty:
        stats = filtered_data[numeric_cols].describe().T
        st.dataframe(stats, use_container_width=True)
    else:
        st.info("No numeric columns available for summary statistics.")
    
    # Plot distributions for key columns
    st.subheader("Data Distributions")
    
    # Price distribution
    if 'price' in filtered_data.columns:
        show_distribution_chart(filtered_data, 'price')
    
    # Lead time distribution
    if 'lead_time' in filtered_data.columns:
        show_distribution_chart(filtered_data, 'lead_time')
    
    # Missing values
    st.subheader("Missing Values")
    
    missing_data = filtered_data.isna().sum()
    missing_pct = (filtered_data.isna().sum() / len(filtered_data)) * 100
    
    missing_df = pd.DataFrame({
        'Missing Count': missing_data,
        'Missing Percent': missing_pct
    }).sort_values('Missing Count', ascending=False)
    
    if missing_df['Missing Count'].sum() > 0:
        st.dataframe(missing_df[missing_df['Missing Count'] > 0], use_container_width=True)
        
        # Plot missing values
        missing_cols = missing_df[missing_df['Missing Count'] > 0].index.tolist()
        if missing_cols:
            fig = px.bar(
                missing_df[missing_df['Missing Count'] > 0].reset_index(),
                x='index',
                y='Missing Percent',
                title="Percentage of Missing Values by Column",
                labels={'index': 'Column', 'Missing Percent': 'Missing (%)'},
                color='Missing Percent',
                color_continuous_scale='Reds'
            )
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.success("No missing values found in the dataset!")


def show_distribution_chart(data, column):
    """Show distribution chart for the given column."""
    if column not in data.columns:
        return
    
    # Calculate basic stats
    mean_val = data[column].mean()
    median_val = data[column].median()
    
    # Create histogram with KDE
    fig = px.histogram(
        data, 
        x=column, 
        marginal="box", 
        title=f"{column.capitalize()} Distribution",
        histnorm='probability density',
        opacity=0.7
    )
    
    # Add mean and median lines
    fig.add_vline(x=mean_val, line_dash="solid", line_width=2, line_color="red", 
                  annotation_text=f"Mean: {mean_val:.2f}", annotation_position="top right")
    
    fig.add_vline(x=median_val, line_dash="dash", line_width=2, line_color="green", 
                  annotation_text=f"Median: {median_val:.2f}", annotation_position="top left")
    
    st.plotly_chart(fig, use_container_width=True)


def show_data_cleaning(data):
    """Show data cleaning options and apply cleaning operations."""
    st.subheader("Data Cleaning")
    
    st.markdown("""
    This section allows you to clean your data by handling missing values and outliers. 
    The cleaned dataset will be used for forecasting.
    """)
    
    # Check if cleaned data already exists
    if st.session_state.data_clean is not None:
        with st.expander("Current Cleaning Status"):
            st.success("Data has been cleaned. You can apply different cleaning strategies if needed.")
            st.write("Original data shape:", data.shape)
            st.write("Cleaned data shape:", st.session_state.data_clean.shape)
    
    # Create two columns for cleaning strategies
    col1, col2 = st.columns(2)
    
    # Column 1: Missing value strategies
    with col1:
        st.subheader("Missing Value Strategies")
        
        # Identify columns with missing values
        missing_cols = data.columns[data.isna().any()].tolist()
        
        if not missing_cols:
            st.success("No missing values to handle!")
            missing_strategies = {}
        else:
            st.write("Select strategies for handling missing values:")
            
            # Create strategy selection for each column with missing values
            missing_strategies = {}
            for col in missing_cols:
                # Determine appropriate strategies based on column type
                if pd.api.types.is_numeric_dtype(data[col]):
                    options = ['mean', 'median', 'zero', 'drop', 'interpolate']
                else:
                    options = ['mode', 'drop', 'ffill', 'bfill']
                
                missing_strategies[col] = st.selectbox(
                    f"Strategy for {col}:", 
                    options=options,
                    key=f"missing_{col}"
                )
    
    # Column 2: Outlier handling strategies
    with col2:
        st.subheader("Outlier Handling Strategies")
        
        # Identify numeric columns for outlier handling
        numeric_cols = data.select_dtypes(include=['number']).columns.tolist()
        
        if not numeric_cols:
            st.warning("No numeric columns for outlier handling!")
            outlier_strategies = {}
        else:
            st.write("Select strategies for handling outliers:")
            
            # Create strategy selection for each numeric column
            outlier_strategies = {}
            for col in numeric_cols:
                outlier_strategies[col] = st.selectbox(
                    f"Strategy for {col}:", 
                    options=['none', 'clip', 'remove', 'winsorize'],
                    key=f"outlier_{col}"
                )
    
    # Combine strategies into config
    cleaning_config = {}
    for col in data.columns:
        cleaning_config[col] = {
            'missing': missing_strategies.get(col, None),
            'outlier': outlier_strategies.get(col, None) if outlier_strategies.get(col) != 'none' else None
        }
    
    # Create a button to apply cleaning
    if st.button("Apply Data Cleaning", type="primary"):
        with st.spinner("Cleaning data..."):
            try:
                # Apply data cleaning
                cleaner = DataCleaner()
                cleaned_data = cleaner.clean_dataset(data, cleaning_config)
                
                # Store cleaned data in session state
                st.session_state.data_clean = cleaned_data
                
                # Show success message
                set_success("Data cleaning completed successfully!")
                
                # Clear forecasts as the data has changed
                st.session_state.forecasts = None
                
                # Show results
                st.subheader("Cleaning Results")
                
                # Compare original and cleaned data
                comparison = pd.DataFrame({
                    'Original Missing': data.isna().sum(),
                    'Cleaned Missing': cleaned_data.isna().sum(),
                    'Original Records': len(data),
                    'Cleaned Records': len(cleaned_data)
                })
                
                st.dataframe(comparison, use_container_width=True)
                
                # Show preview of cleaned data
                st.subheader("Cleaned Data Preview")
                st.dataframe(cleaned_data.head(10), use_container_width=True)
                
            except Exception as e:
                set_error(f"Error during data cleaning: {str(e)}")


def show_time_series_analysis(data):
    """Show time series analysis for price and lead time."""
    st.subheader("Time Series Analysis")
    
    # Check if date column exists
    if 'date' not in data.columns:
        st.warning("No date column found in the data. Time series analysis requires a date column.")
        return
    
    # Ensure date column is datetime
    if not pd.api.types.is_datetime64_any_dtype(data['date']):
        data = data.copy()
        data['date'] = pd.to_datetime(data['date'])
    
    # Get time-based columns
    time_series_cols = []
    if 'price' in data.columns:
        time_series_cols.append('price')
    if 'lead_time' in data.columns:
        time_series_cols.append('lead_time')
    
    if not time_series_cols:
        st.warning("Neither price nor lead_time columns found for time series analysis.")
        return
    
    # Let user select a column
    selected_col = st.selectbox(
        "Select column for time series analysis:", 
        options=time_series_cols
    )
    
    # Let user select aggregation level
    aggregation_level = st.selectbox(
        "Select time aggregation level:", 
        options=['day', 'week', 'month', 'quarter']
    )
    
    # Apply filters if available
    filter_cols = []
    if 'part_number' in data.columns:
        filter_cols.append('part_number')
    if 'vendor' in data.columns:
        filter_cols.append('vendor')
    if 'country' in data.columns:
        filter_cols.append('country')
    
    filtered_data = data.copy()
    
    if filter_cols:
        # Create expander for filters to save space
        with st.expander("Filter Data"):
            # Create columns for filters
            columns = st.columns(min(3, len(filter_cols)))
            
            selected_filters = {}
            
            # Add each filter as a multi-select
            for i, col in enumerate(filter_cols):
                with columns[i % len(columns)]:
                    # Limit options to top 50 for performance
                    top_values = data[col].value_counts().head(50).index.tolist()
                    
                    if len(top_values) > 1:
                        selected = st.multiselect(
                            f"Select {col}:",
                            options=top_values,
                            default=None,
                            key=f"ts_filter_{col}"
                        )
                        
                        if selected:
                            selected_filters[col] = selected
            
            # Apply filters if any selected
            for col, values in selected_filters.items():
                filtered_data = filtered_data[filtered_data[col].isin(values)]
    
    # Check if we have enough data
    if len(filtered_data) < 5:
        st.warning("Not enough data for time series analysis after filtering. Please adjust filters.")
        return
    
    # Aggregate data
    if aggregation_level == 'day':
        # No aggregation needed for daily data
        ts_df = filtered_data.copy()
    else:
        # Set frequency based on selected level
        if aggregation_level == 'week':
            freq = 'W'
        elif aggregation_level == 'month':
            freq = 'MS'
        else:  # quarter
            freq = 'QS'
        
        # Resample and aggregate
        ts_df = filtered_data.copy()
        ts_df = ts_df.set_index('date')
        ts_df = ts_df.resample(freq)[selected_col].mean().reset_index()
    
    # Show time series plot
    fig = px.line(ts_df, x='date', y=selected_col, 
                  title=f"{selected_col.capitalize()} Time Series ({aggregation_level.capitalize()} Aggregation)")
    
    # Add trendline
    fig.update_traces(line=dict(width=2))
    
    # Add moving averages
    ma_windows = [3, 6, 12]
    for window in ma_windows:
        if len(ts_df) > window:
            ma = ts_df[selected_col].rolling(window=window).mean()
            fig.add_trace(go.Scatter(
                x=ts_df['date'],
                y=ma,
                mode='lines',
                name=f'{window}-Period Moving Avg',
                line=dict(width=1, dash='dash')
            ))
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Time Series Decomposition if enough data
    if len(ts_df) >= 14:  # Need enough data for decomposition
        st.subheader("Time Series Decomposition")
        
        # Let user select period for decomposition
        period_options = [3, 6, 12]
        if len(ts_df) >= 24:
            period_options.extend([24, 52])
            
        period = st.select_slider(
            "Select seasonality period for decomposition:",
            options=period_options,
            value=min(12, max(period_options))
        )
        
        try:
            # Perform decomposition
            ts_analyzer = TimeSeriesAnalyzer()
            ts_data = ts_df.set_index('date')[selected_col]
            
            # Make sure the index is datetime
            ts_data.index = pd.to_datetime(ts_data.index)
            
            decomposition = ts_analyzer.decompose_time_series(ts_data, period=period)
            
            # Plot decomposition components
            st.write("Trend Component")
            fig_trend = px.line(x=decomposition['trend'].index, y=decomposition['trend'].values,
                                labels={'x': 'Date', 'y': 'Trend'})
            st.plotly_chart(fig_trend, use_container_width=True)
            
            st.write("Seasonal Component")
            fig_seasonal = px.line(x=decomposition['seasonal'].index, y=decomposition['seasonal'].values,
                                  labels={'x': 'Date', 'y': 'Seasonality'})
            st.plotly_chart(fig_seasonal, use_container_width=True)
            
            st.write("Residual Component")
            fig_residual = px.line(x=decomposition['residual'].index, y=decomposition['residual'].values,
                                  labels={'x': 'Date', 'y': 'Residual'})
            st.plotly_chart(fig_residual, use_container_width=True)
            
        except Exception as e:
            st.error(f"Error performing time series decomposition: {str(e)}")


def show_correlation_analysis(data):
    """Show correlation analysis between variables."""
    st.subheader("Correlation Analysis")
    
    # Get numeric columns
    numeric_cols = data.select_dtypes(include=['number']).columns.tolist()
    
    if len(numeric_cols) < 2:
        st.warning("At least two numeric columns are required for correlation analysis.")
        return
    
    # Create correlation matrix
    correlation = data[numeric_cols].corr()
    
    # Plot heatmap
    fig = px.imshow(
        correlation, 
        text_auto=True, 
        aspect="auto", 
        color_continuous_scale='RdBu_r',
        title="Correlation Matrix",
        labels=dict(color="Correlation")
    )
    
    # Update layout
    fig.update_layout(
        height=max(500, len(numeric_cols) * 30),
        width=max(500, len(numeric_cols) * 30)
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Scatter plot for selected variables
    st.subheader("Relationship Explorer")
    
    col1, col2 = st.columns(2)
    
    with col1:
        x_var = st.selectbox("Select X variable:", numeric_cols, index=0)
    with col2:
        y_var = st.selectbox("Select Y variable:", numeric_cols, 
                             index=min(1, len(numeric_cols)-1))
    
    # Choose color variable if available
    color_var = None
    if 'part_number' in data.columns:
        color_var = 'part_number'
    elif 'vendor' in data.columns:
        color_var = 'vendor'
    
    # Let user choose whether to use color
    use_color = False
    if color_var:
        use_color = st.checkbox("Color points by category", value=True)
        if use_color:
            color_var_options = [col for col in data.columns if col not in [x_var, y_var] 
                                and col != 'date' and data[col].nunique() < 30]
            if color_var_options:
                color_var = st.selectbox("Select category for coloring:", color_var_options)
            else:
                use_color = False
                st.info("No suitable categorical variables found for coloring.")
    
    # Create scatter plot
    if use_color:
        fig = px.scatter(
            data, 
            x=x_var, 
            y=y_var, 
            color=color_var, 
            title=f"Relationship between {x_var} and {y_var}",
            trendline="ols" if len(data) < 5000 else None,  # Only add trendline if not too many points
            opacity=0.7
        )
    else:
        fig = px.scatter(
            data, 
            x=x_var, 
            y=y_var, 
            title=f"Relationship between {x_var} and {y_var}",
            trendline="ols",
            opacity=0.7
        )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Calculate and display correlation coefficient
    corr = data[[x_var, y_var]].corr().iloc[0, 1]
    st.write(f"Correlation coefficient: **{corr:.4f}**")
    
    if abs(corr) < 0.3:
        st.write("🔵 Weak correlation")
    elif abs(corr) < 0.7:
        st.write("🟠 Moderate correlation")
    else:
        st.write("🔴 Strong correlation")