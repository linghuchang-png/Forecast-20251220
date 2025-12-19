"""
Visualization page for the forecasting application.

This module handles detailed visualization and analysis of forecast results.
"""
import os
import sys
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import io
import base64

# Add parent directory to path for importing
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

# Import utilities and visualization components
from app.visualization.prediction_viz import ForecastVisualizer
from app.state.session import set_error, set_success, clear_messages


def show_visualization_page():
    """Display the visualization page."""
    st.header("Forecast Visualizations")
    
    # Clear any previous messages
    clear_messages()
    
    # Check if forecasts are available
    if st.session_state.forecasts is None:
        st.warning("No forecast data available. Please generate forecasts first.")
        if st.button("Go to Forecasting"):
            st.query_params(page="forecasting")
            st.rerun()
        return
    
    # Get forecast data
    forecasts = st.session_state.forecasts
    
    # Get parameters
    params = st.session_state.forecast_params
    date_col = params.get('date_col', 'date')
    price_col = params.get('price_col', 'price')
    leadtime_col = params.get('leadtime_col', 'lead_time')
    group_columns = params.get('group_columns', [])
    
    # Extract price and lead time forecasts
    price_forecast = forecasts.get('price_forecast')
    leadtime_forecast = forecasts.get('leadtime_forecast')
    
    # Get historical data
    historical_data = None
    if st.session_state.data_clean is not None:
        historical_data = st.session_state.data_clean
    else:
        historical_data = st.session_state.data
    
    # Create visualization tabs
    tabs = st.tabs([
        "Time Series Visualization", 
        "Comparative Analysis", 
        "Forecast Statistics", 
        "Custom Visualization"
    ])
    
    # Time Series Visualization tab
    with tabs[0]:
        show_time_series_viz(
            historical_data, 
            price_forecast, 
            leadtime_forecast, 
            date_col, 
            price_col, 
            leadtime_col, 
            group_columns
        )
    
    # Comparative Analysis tab
    with tabs[1]:
        show_comparative_analysis(
            price_forecast, 
            leadtime_forecast, 
            date_col, 
            price_col, 
            leadtime_col, 
            group_columns
        )
    
    # Forecast Statistics tab
    with tabs[2]:
        show_forecast_statistics(
            price_forecast, 
            leadtime_forecast,
            date_col, 
            price_col, 
            leadtime_col, 
            group_columns
        )
    
    # Custom Visualization tab
    with tabs[3]:
        show_custom_visualization(
            historical_data, 
            price_forecast, 
            leadtime_forecast,
            date_col, 
            price_col, 
            leadtime_col, 
            group_columns
        )


def show_time_series_viz(historical_data, price_forecast, leadtime_forecast, 
                         date_col, price_col, leadtime_col, group_columns):
    """Show time series visualization with historical data and forecasts."""
    st.subheader("Time Series Visualization")
    
    # Check if group columns exist
    has_groups = len(group_columns) > 0 and all(col in price_forecast.columns for col in group_columns)
    
    # Create filters if groups exist
    selected_groups = {}
    if has_groups:
        st.write("#### Filter by Group")
        # Create columns for filters
        cols = st.columns(min(3, len(group_columns)))
        
        for i, col_name in enumerate(group_columns):
            with cols[i % len(cols)]:
                # Get unique values for this column
                unique_values = sorted(price_forecast[col_name].unique())
                
                # Create a selectbox for this column
                selected = st.selectbox(
                    f"Select {col_name}:",
                    options=unique_values,
                    key=f"ts_viz_{col_name}"
                )
                
                selected_groups[col_name] = selected
    
    # Create visualization options
    st.write("#### Visualization Options")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        show_price = st.checkbox("Show Price Forecast", value=True, key="show_price_ts")
    
    with col2:
        show_lead_time = st.checkbox("Show Lead Time Forecast", value=True, key="show_lead_time_ts")
    
    with col3:
        show_historical = st.checkbox("Include Historical Data", value=True, key="show_historical_ts")
    
    # Filter forecast data if groups exist
    if has_groups:
        filtered_price = price_forecast.copy()
        filtered_lead_time = leadtime_forecast.copy()
        
        # Apply group filters
        for col, val in selected_groups.items():
            filtered_price = filtered_price[filtered_price[col] == val]
            filtered_lead_time = filtered_lead_time[filtered_lead_time[col] == val]
        
        # Filter historical data if showing historical
        if show_historical and historical_data is not None:
            filtered_historical = historical_data.copy()
            for col, val in selected_groups.items():
                if col in filtered_historical.columns:
                    filtered_historical = filtered_historical[filtered_historical[col] == val]
        else:
            filtered_historical = None
    else:
        filtered_price = price_forecast
        filtered_lead_time = leadtime_forecast
        filtered_historical = historical_data if show_historical else None
    
    # Create time series visualizations
    if show_price:
        st.write("### Price Forecast")
        visualize_time_series(
            filtered_historical, 
            filtered_price, 
            date_col, 
            price_col,
            title=f"Price Forecast over Time" + (f" for {', '.join([f'{k}: {v}' for k, v in selected_groups.items()])}" if has_groups else ""),
            show_historical=show_historical
        )
    
    if show_lead_time:
        st.write("### Lead Time Forecast")
        visualize_time_series(
            filtered_historical, 
            filtered_lead_time, 
            date_col, 
            leadtime_col,
            title=f"Lead Time Forecast over Time" + (f" for {', '.join([f'{k}: {v}' for k, v in selected_groups.items()])}" if has_groups else ""),
            show_historical=show_historical
        )


def visualize_time_series(historical_data, forecast_data, date_col, value_col, title, show_historical=True):
    """Create a time series visualization with historical and forecast data."""
    # Create figure
    fig = go.Figure()
    
    # Add historical data if available and requested
    if show_historical and historical_data is not None:
        # Ensure date column is datetime
        historical = historical_data.copy()
        if not pd.api.types.is_datetime64_any_dtype(historical[date_col]):
            historical[date_col] = pd.to_datetime(historical[date_col])
        
        # Check if value column exists in historical data
        if value_col in historical.columns:
            # Group by date if multiple records per date
            if historical.groupby(date_col).size().max() > 1:
                historical = historical.groupby(date_col)[value_col].mean().reset_index()
            
            # Sort by date
            historical = historical.sort_values(by=date_col)
            
            # Add historical trace
            fig.add_trace(go.Scatter(
                x=historical[date_col], 
                y=historical[value_col],
                mode='lines+markers',
                name='Historical',
                line=dict(color='blue', width=2, dash='dash'),
                marker=dict(size=8, color='blue', symbol='circle')
            ))
    
    # Add forecast data
    forecast = forecast_data.copy()
    
    # Ensure date column is datetime
    if not pd.api.types.is_datetime64_any_dtype(forecast[date_col]):
        forecast[date_col] = pd.to_datetime(forecast[date_col])
    
    # Sort by date
    forecast = forecast.sort_values(by=date_col)
    
    # Add forecast trace
    fig.add_trace(go.Scatter(
        x=forecast[date_col], 
        y=forecast[value_col],
        mode='lines+markers',
        name='Forecast',
        line=dict(color='red', width=2),
        marker=dict(size=8, color='red', symbol='circle')
    ))
    
    # Update layout
    fig.update_layout(
        title=title,
        xaxis_title='Date',
        yaxis_title=value_col.capitalize(),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        template="plotly_white",
        height=500
    )
    
    # Display figure
    st.plotly_chart(fig, use_container_width=True)
    
    # Add option to download the visualization
    add_download_button(fig, title.replace(" ", "_").lower() + ".png", "Download Chart")


def show_comparative_analysis(price_forecast, leadtime_forecast, date_col, price_col, leadtime_col, group_columns):
    """Show comparative analysis between different groups."""
    st.subheader("Comparative Analysis")
    
    # Check if group columns exist
    if not (len(group_columns) > 0 and all(col in price_forecast.columns for col in group_columns)):
        st.info("Comparative analysis requires grouping columns. Please generate forecasts with group columns first.")
        return
    
    # Create visualization options
    st.write("#### Visualization Options")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        show_price = st.checkbox("Show Price Comparison", value=True, key="show_price_comp")
    
    with col2:
        show_lead_time = st.checkbox("Show Lead Time Comparison", value=True, key="show_lead_time_comp")
    
    with col3:
        # Select primary grouping column for comparison
        comp_group_col = st.selectbox(
            "Compare by:",
            options=group_columns,
            key="compare_group"
        )
    
    # Filter by secondary groups if more than one group column
    secondary_filters = {}
    if len(group_columns) > 1:
        st.write("#### Filter by Secondary Groups")
        
        # Create columns for filters
        sec_cols = st.columns(min(3, len(group_columns) - 1))
        
        i = 0
        for col_name in group_columns:
            if col_name != comp_group_col:
                with sec_cols[i % len(sec_cols)]:
                    # Get unique values for this column
                    unique_values = sorted(price_forecast[col_name].unique())
                    
                    # Add an "All" option
                    options = ["All"] + unique_values
                    
                    # Create a selectbox for this column
                    selected = st.selectbox(
                        f"Select {col_name}:",
                        options=options,
                        key=f"comp_sec_{col_name}"
                    )
                    
                    if selected != "All":
                        secondary_filters[col_name] = selected
                
                i += 1
    
    # Get top N groups to compare
    top_n = st.slider(
        "Number of groups to compare:",
        min_value=2,
        max_value=min(10, price_forecast[comp_group_col].nunique()),
        value=min(5, price_forecast[comp_group_col].nunique()),
        step=1,
        key="top_n_groups"
    )
    
    # Select specific date for bar chart comparison
    forecast_dates = price_forecast[date_col].sort_values().unique()
    if len(forecast_dates) > 0:
        selected_date = st.select_slider(
            "Select date for comparison:",
            options=forecast_dates,
            value=forecast_dates[-1] if len(forecast_dates) > 0 else None,
            key="comp_date"
        )
    else:
        selected_date = None
        st.warning("No dates available in forecast data.")
        return
    
    # Apply secondary filters
    filtered_price = price_forecast.copy()
    filtered_lead_time = leadtime_forecast.copy()
    
    for col, val in secondary_filters.items():
        filtered_price = filtered_price[filtered_price[col] == val]
        filtered_lead_time = filtered_lead_time[filtered_lead_time[col] == val]
    
    # Find top N groups based on average price/lead time
    if show_price:
        top_price_groups = filtered_price.groupby(comp_group_col)[price_col].mean().nlargest(top_n).index.tolist()
    else:
        top_price_groups = []
    
    if show_lead_time:
        top_lead_time_groups = filtered_lead_time.groupby(comp_group_col)[leadtime_col].mean().nlargest(top_n).index.tolist()
    else:
        top_lead_time_groups = []
    
    # Combine top groups
    top_groups = list(set(top_price_groups + top_lead_time_groups))[:top_n]
    
    # Filter to top groups and selected date
    if selected_date:
        date_filtered_price = filtered_price[
            (filtered_price[date_col] == selected_date) & 
            (filtered_price[comp_group_col].isin(top_groups))
        ]
        
        date_filtered_lead_time = filtered_lead_time[
            (filtered_lead_time[date_col] == selected_date) & 
            (filtered_lead_time[comp_group_col].isin(top_groups))
        ]
    else:
        date_filtered_price = filtered_price[filtered_price[comp_group_col].isin(top_groups)]
        date_filtered_lead_time = filtered_lead_time[filtered_lead_time[comp_group_col].isin(top_groups)]
    
    # Create bar chart comparisons for selected date
    if show_price:
        st.write(f"### Price Comparison ({selected_date.strftime('%Y-%m-%d') if isinstance(selected_date, (pd.Timestamp, datetime)) else selected_date})")
        
        if not date_filtered_price.empty:
            fig = px.bar(
                date_filtered_price.sort_values(price_col, ascending=False), 
                x=comp_group_col, 
                y=price_col,
                color=comp_group_col,
                title=f"Price Comparison by {comp_group_col}" + 
                      (f" (Filtered by {', '.join([f'{k}: {v}' for k, v in secondary_filters.items()])})" if secondary_filters else "")
            )
            
            # Update layout
            fig.update_layout(
                xaxis_title=comp_group_col,
                yaxis_title=price_col.capitalize(),
                template="plotly_white",
                height=500
            )
            
            # Display figure
            st.plotly_chart(fig, use_container_width=True)
            
            # Add download button
            add_download_button(fig, f"price_comparison_{comp_group_col}.png", "Download Chart")
        else:
            st.info("No price data available for the selected filters.")
    
    if show_lead_time:
        st.write(f"### Lead Time Comparison ({selected_date.strftime('%Y-%m-%d') if isinstance(selected_date, (pd.Timestamp, datetime)) else selected_date})")
        
        if not date_filtered_lead_time.empty:
            fig = px.bar(
                date_filtered_lead_time.sort_values(leadtime_col, ascending=False), 
                x=comp_group_col, 
                y=leadtime_col,
                color=comp_group_col,
                title=f"Lead Time Comparison by {comp_group_col}" + 
                      (f" (Filtered by {', '.join([f'{k}: {v}' for k, v in secondary_filters.items()])})" if secondary_filters else "")
            )
            
            # Update layout
            fig.update_layout(
                xaxis_title=comp_group_col,
                yaxis_title=leadtime_col.capitalize(),
                template="plotly_white",
                height=500
            )
            
            # Display figure
            st.plotly_chart(fig, use_container_width=True)
            
            # Add download button
            add_download_button(fig, f"lead_time_comparison_{comp_group_col}.png", "Download Chart")
        else:
            st.info("No lead time data available for the selected filters.")
    
    # Time series comparison across groups
    st.write("### Comparison Over Time")
    
    # Filter to top groups for time series
    time_filtered_price = filtered_price[filtered_price[comp_group_col].isin(top_groups)]
    time_filtered_lead_time = filtered_lead_time[filtered_lead_time[comp_group_col].isin(top_groups)]
    
    if show_price and not time_filtered_price.empty:
        fig = px.line(
            time_filtered_price, 
            x=date_col, 
            y=price_col, 
            color=comp_group_col,
            title=f"Price Forecast Comparison by {comp_group_col}" + 
                  (f" (Filtered by {', '.join([f'{k}: {v}' for k, v in secondary_filters.items()])})" if secondary_filters else ""),
            markers=True
        )
        
        # Update layout
        fig.update_layout(
            xaxis_title="Date",
            yaxis_title=price_col.capitalize(),
            template="plotly_white",
            height=500
        )
        
        # Display figure
        st.plotly_chart(fig, use_container_width=True)
        
        # Add download button
        add_download_button(fig, f"price_forecast_comparison_{comp_group_col}.png", "Download Chart")
    
    if show_lead_time and not time_filtered_lead_time.empty:
        fig = px.line(
            time_filtered_lead_time, 
            x=date_col, 
            y=leadtime_col, 
            color=comp_group_col,
            title=f"Lead Time Forecast Comparison by {comp_group_col}" + 
                  (f" (Filtered by {', '.join([f'{k}: {v}' for k, v in secondary_filters.items()])})" if secondary_filters else ""),
            markers=True
        )
        
        # Update layout
        fig.update_layout(
            xaxis_title="Date",
            yaxis_title=leadtime_col.capitalize(),
            template="plotly_white",
            height=500
        )
        
        # Display figure
        st.plotly_chart(fig, use_container_width=True)
        
        # Add download button
        add_download_button(fig, f"lead_time_forecast_comparison_{comp_group_col}.png", "Download Chart")


def show_forecast_statistics(price_forecast, leadtime_forecast, date_col, price_col, leadtime_col, group_columns):
    """Show statistical analysis of forecast data."""
    st.subheader("Forecast Statistics")
    
    # Create tabs for price and lead time statistics
    stats_tabs = st.tabs(["Price Statistics", "Lead Time Statistics"])
    
    # Price Statistics tab
    with stats_tabs[0]:
        st.write("### Price Forecast Statistics")
        
        # Overall statistics
        st.write("#### Overall Statistics")
        
        # Calculate basic statistics
        price_stats = {
            'Minimum': price_forecast[price_col].min(),
            'Maximum': price_forecast[price_col].max(),
            'Average': price_forecast[price_col].mean(),
            'Median': price_forecast[price_col].median(),
            'Std. Deviation': price_forecast[price_col].std(),
            'Range': price_forecast[price_col].max() - price_forecast[price_col].min()
        }
        
        # Display as dataframe
        st.dataframe(pd.DataFrame([price_stats]), use_container_width=True)
        
        # Create distribution visualization
        st.write("#### Price Distribution")
        
        fig = px.histogram(
            price_forecast, 
            x=price_col,
            marginal="box", 
            title="Price Forecast Distribution",
            histnorm='probability density',
            opacity=0.7
        )
        
        # Add mean and median lines
        fig.add_vline(x=price_stats['Average'], line_dash="solid", line_width=2, line_color="red", 
                      annotation_text=f"Mean: {price_stats['Average']:.2f}", annotation_position="top right")
        
        fig.add_vline(x=price_stats['Median'], line_dash="dash", line_width=2, line_color="green", 
                      annotation_text=f"Median: {price_stats['Median']:.2f}", annotation_position="top left")
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Group statistics if group columns exist
        if len(group_columns) > 0 and all(col in price_forecast.columns for col in group_columns):
            st.write("#### Statistics by Group")
            
            # Select primary grouping column
            primary_group = st.selectbox(
                "Group by:",
                options=group_columns,
                key="price_stats_group"
            )
            
            # Calculate group statistics
            group_stats = price_forecast.groupby(primary_group)[price_col].agg([
                ('Minimum', 'min'),
                ('Maximum', 'max'),
                ('Average', 'mean'),
                ('Median', 'median'),
                ('Std. Deviation', 'std'),
                ('Range', lambda x: x.max() - x.min())
            ]).reset_index()
            
            # Display group statistics
            st.dataframe(group_stats, use_container_width=True)
            
            # Create box plot by group
            st.write("#### Price Distribution by Group")
            
            fig = px.box(
                price_forecast, 
                x=primary_group, 
                y=price_col,
                color=primary_group,
                title=f"Price Distribution by {primary_group}"
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Add download button
            add_download_button(fig, f"price_distribution_{primary_group}.png", "Download Chart")
    
    # Lead Time Statistics tab
    with stats_tabs[1]:
        st.write("### Lead Time Forecast Statistics")
        
        # Overall statistics
        st.write("#### Overall Statistics")
        
        # Calculate basic statistics
        leadtime_stats = {
            'Minimum': leadtime_forecast[leadtime_col].min(),
            'Maximum': leadtime_forecast[leadtime_col].max(),
            'Average': leadtime_forecast[leadtime_col].mean(),
            'Median': leadtime_forecast[leadtime_col].median(),
            'Std. Deviation': leadtime_forecast[leadtime_col].std(),
            'Range': leadtime_forecast[leadtime_col].max() - leadtime_forecast[leadtime_col].min()
        }
        
        # Display as dataframe
        st.dataframe(pd.DataFrame([leadtime_stats]), use_container_width=True)
        
        # Create distribution visualization
        st.write("#### Lead Time Distribution")
        
        fig = px.histogram(
            leadtime_forecast, 
            x=leadtime_col,
            marginal="box", 
            title="Lead Time Forecast Distribution",
            histnorm='probability density',
            opacity=0.7
        )
        
        # Add mean and median lines
        fig.add_vline(x=leadtime_stats['Average'], line_dash="solid", line_width=2, line_color="red", 
                      annotation_text=f"Mean: {leadtime_stats['Average']:.2f}", annotation_position="top right")
        
        fig.add_vline(x=leadtime_stats['Median'], line_dash="dash", line_width=2, line_color="green", 
                      annotation_text=f"Median: {leadtime_stats['Median']:.2f}", annotation_position="top left")
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Group statistics if group columns exist
        if len(group_columns) > 0 and all(col in leadtime_forecast.columns for col in group_columns):
            st.write("#### Statistics by Group")
            
            # Select primary grouping column
            primary_group = st.selectbox(
                "Group by:",
                options=group_columns,
                key="leadtime_stats_group"
            )
            
            # Calculate group statistics
            group_stats = leadtime_forecast.groupby(primary_group)[leadtime_col].agg([
                ('Minimum', 'min'),
                ('Maximum', 'max'),
                ('Average', 'mean'),
                ('Median', 'median'),
                ('Std. Deviation', 'std'),
                ('Range', lambda x: x.max() - x.min())
            ]).reset_index()
            
            # Display group statistics
            st.dataframe(group_stats, use_container_width=True)
            
            # Create box plot by group
            st.write("#### Lead Time Distribution by Group")
            
            fig = px.box(
                leadtime_forecast, 
                x=primary_group, 
                y=leadtime_col,
                color=primary_group,
                title=f"Lead Time Distribution by {primary_group}"
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Add download button
            add_download_button(fig, f"lead_time_distribution_{primary_group}.png", "Download Chart")


def show_custom_visualization(historical_data, price_forecast, leadtime_forecast, 
                             date_col, price_col, leadtime_col, group_columns):
    """Show custom visualization options."""
    st.subheader("Custom Visualization")
    
    st.write("""
    This section allows you to create custom visualizations based on your specific needs.
    Select the chart type, data, and display options to generate your custom visualization.
    """)
    
    # Select chart type
    chart_type = st.selectbox(
        "Select Chart Type:",
        options=["Line Chart", "Bar Chart", "Scatter Plot", "Box Plot", "Heatmap", "Bubble Chart"],
        key="custom_chart_type"
    )
    
    # Create tabs for price, lead time, and combined visualizations
    custom_tabs = st.tabs(["Price Data", "Lead Time Data", "Combined Data"])
    
    # Price Data tab
    with custom_tabs[0]:
        create_custom_chart(
            chart_type, 
            historical_data, 
            price_forecast, 
            date_col, 
            price_col, 
            group_columns,
            "Price"
        )
    
    # Lead Time Data tab
    with custom_tabs[1]:
        create_custom_chart(
            chart_type, 
            historical_data, 
            leadtime_forecast, 
            date_col, 
            leadtime_col, 
            group_columns,
            "Lead Time"
        )
    
    # Combined Data tab
    with custom_tabs[2]:
        create_combined_chart(
            chart_type, 
            historical_data, 
            price_forecast, 
            leadtime_forecast,
            date_col, 
            price_col, 
            leadtime_col, 
            group_columns
        )


def create_custom_chart(chart_type, historical_data, forecast_data, date_col, value_col, group_columns, data_type):
    """Create a custom chart based on selected options."""
    st.write(f"### Custom {data_type} Visualization")
    
    # Determine if we have groups
    has_groups = len(group_columns) > 0 and all(col in forecast_data.columns for col in group_columns)
    
    # Data source options
    data_source = st.radio(
        "Data Source:",
        options=["Forecast Only", "Historical Only", "Both Historical and Forecast"],
        index=0,
        key=f"custom_{data_type.lower()}_source",
        horizontal=True
    )
    
    # Prepare data based on selection
    if data_source == "Forecast Only":
        plot_data = forecast_data.copy()
        data_label = "Forecast"
    elif data_source == "Historical Only":
        if historical_data is not None and value_col in historical_data.columns:
            plot_data = historical_data.copy()
            data_label = "Historical"
        else:
            st.warning(f"No historical {data_type.lower()} data available.")
            return
    else:  # Both
        if historical_data is not None and value_col in historical_data.columns:
            # Create combined dataframe with a source column
            historical = historical_data.copy()
            historical['data_source'] = 'Historical'
            
            forecast = forecast_data.copy()
            forecast['data_source'] = 'Forecast'
            
            # Ensure both have the same columns for concatenation
            required_cols = [date_col, value_col, 'data_source']
            if has_groups:
                required_cols.extend(group_columns)
            
            historical_subset = historical[required_cols]
            forecast_subset = forecast[required_cols]
            
            plot_data = pd.concat([historical_subset, forecast_subset], ignore_index=True)
            data_label = "data_source"  # Use the data_source column for color
        else:
            st.warning(f"No historical {data_type.lower()} data available. Using forecast data only.")
            plot_data = forecast_data.copy()
            plot_data['data_source'] = 'Forecast'
            data_label = "data_source"
    
    # Filter options if we have groups
    if has_groups:
        st.write("#### Filter Data")
        
        # Create select/multiselect for each group column
        filters = {}
        use_multiselect = st.checkbox("Enable multi-select filtering", key=f"custom_{data_type.lower()}_multiselect")
        
        cols = st.columns(min(3, len(group_columns)))
        
        for i, col_name in enumerate(group_columns):
            with cols[i % len(cols)]:
                unique_values = sorted(plot_data[col_name].unique())
                
                if use_multiselect:
                    selected = st.multiselect(
                        f"Select {col_name}:",
                        options=unique_values,
                        default=unique_values[:1],
                        key=f"custom_{data_type.lower()}_filter_{col_name}"
                    )
                    
                    if selected:
                        filters[col_name] = selected
                else:
                    # Add "All" option for selectbox
                    options = ["All"] + unique_values
                    
                    selected = st.selectbox(
                        f"Select {col_name}:",
                        options=options,
                        key=f"custom_{data_type.lower()}_filter_{col_name}"
                    )
                    
                    if selected != "All":
                        filters[col_name] = [selected]
        
        # Apply filters
        filtered_data = plot_data.copy()
        for col, values in filters.items():
            filtered_data = filtered_data[filtered_data[col].isin(values)]
        
        if filtered_data.empty:
            st.warning("No data available for the selected filters.")
            return
    else:
        filtered_data = plot_data
    
    # Chart specific options
    if chart_type in ["Line Chart", "Bar Chart", "Scatter Plot"]:
        # Choose X-axis
        if chart_type == "Bar Chart":
            x_options = [date_col] + (group_columns if has_groups else [])
        else:
            x_options = [date_col]
            
        x_col = st.selectbox(
            "X-axis:",
            options=x_options,
            index=0,
            key=f"custom_{data_type.lower()}_x"
        )
        
        # Choose Y-axis (always value_col for single data type charts)
        y_col = value_col
        
        # Choose color column
        color_options = ["None"] + (["data_source"] if 'data_source' in filtered_data.columns else [])
        if has_groups:
            color_options.extend(group_columns)
            
        color_col = st.selectbox(
            "Color by:",
            options=color_options,
            index=1 if "data_source" in color_options else 0,
            key=f"custom_{data_type.lower()}_color"
        )
        
        if color_col == "None":
            color_col = None
    
    # Create the chart based on type
    if chart_type == "Line Chart":
        if x_col == date_col:
            # Sort by date
            filtered_data = filtered_data.sort_values(by=date_col)
        
        # Create the chart
        fig = px.line(
            filtered_data, 
            x=x_col, 
            y=y_col,
            color=color_col,
            markers=True,
            title=f"Custom {data_type} Line Chart"
        )
    
    elif chart_type == "Bar Chart":
        if x_col == date_col and len(filtered_data) > 20:
            st.warning(f"Too many dates for a clear bar chart. Consider filtering the data or using a line chart instead.")
        
        # Group data if using date as x-axis and there are many dates
        if x_col == date_col and len(filtered_data) > 20:
            # Convert to datetime if needed
            if not pd.api.types.is_datetime64_any_dtype(filtered_data[date_col]):
                filtered_data[date_col] = pd.to_datetime(filtered_data[date_col])
                
            # Create month column
            filtered_data['month'] = filtered_data[date_col].dt.to_period('M')
            
            # Group by month and other columns
            group_cols = ['month']
            if color_col and color_col != "None" and color_col != date_col:
                group_cols.append(color_col)
                
            # Aggregate
            grouped = filtered_data.groupby(group_cols)[y_col].mean().reset_index()
            
            # Convert month back to string
            grouped['month'] = grouped['month'].astype(str)
            
            # Create the chart
            fig = px.bar(
                grouped, 
                x='month', 
                y=y_col,
                color=color_col if color_col != date_col else None,
                barmode='group' if color_col and color_col != "None" and color_col != date_col else None,
                title=f"Custom {data_type} Bar Chart (Monthly Aggregation)"
            )
        else:
            # Create the chart
            fig = px.bar(
                filtered_data, 
                x=x_col, 
                y=y_col,
                color=color_col,
                barmode='group' if color_col and color_col != "None" and color_col != x_col else None,
                title=f"Custom {data_type} Bar Chart"
            )
    
    elif chart_type == "Scatter Plot":
        # Create the chart
        fig = px.scatter(
            filtered_data, 
            x=x_col, 
            y=y_col,
            color=color_col,
            opacity=0.7,
            title=f"Custom {data_type} Scatter Plot"
        )
    
    elif chart_type == "Box Plot":
        # Choose group column for box plot
        if has_groups:
            group_col = st.selectbox(
                "Group by:",
                options=group_columns,
                key=f"custom_{data_type.lower()}_box_group"
            )
        elif 'data_source' in filtered_data.columns:
            group_col = 'data_source'
        else:
            st.warning("Box plot requires grouping. No group columns available.")
            return
        
        # Create the chart
        fig = px.box(
            filtered_data, 
            x=group_col, 
            y=y_col,
            color=group_col,
            title=f"Custom {data_type} Box Plot by {group_col}"
        )
    
    elif chart_type == "Heatmap":
        if not has_groups:
            st.warning("Heatmap requires at least one grouping column.")
            return
        
        # Choose group columns for heatmap
        if len(group_columns) >= 2:
            heatmap_x = st.selectbox(
                "X-axis Group:",
                options=group_columns,
                index=0,
                key=f"custom_{data_type.lower()}_heatmap_x"
            )
            
            heatmap_y = st.selectbox(
                "Y-axis Group:",
                options=[col for col in group_columns if col != heatmap_x],
                index=0,
                key=f"custom_{data_type.lower()}_heatmap_y"
            )
            
            # Aggregate data
            heatmap_data = filtered_data.groupby([heatmap_x, heatmap_y])[y_col].mean().reset_index()
            
            # Pivot data for heatmap
            heatmap_pivot = heatmap_data.pivot(index=heatmap_y, columns=heatmap_x, values=y_col)
            
            # Create heatmap
            fig = px.imshow(
                heatmap_pivot,
                labels=dict(x=heatmap_x, y=heatmap_y, color=f"Avg. {y_col}"),
                x=heatmap_pivot.columns,
                y=heatmap_pivot.index,
                aspect="auto",
                title=f"Custom {data_type} Heatmap by {heatmap_x} and {heatmap_y}"
            )
        else:
            # Use date (month) as second dimension
            if not pd.api.types.is_datetime64_any_dtype(filtered_data[date_col]):
                filtered_data[date_col] = pd.to_datetime(filtered_data[date_col])
                
            # Create month column
            filtered_data['month'] = filtered_data[date_col].dt.to_period('M').astype(str)
            
            heatmap_x = 'month'
            heatmap_y = group_columns[0]
            
            # Aggregate data
            heatmap_data = filtered_data.groupby([heatmap_x, heatmap_y])[y_col].mean().reset_index()
            
            # Pivot data for heatmap
            heatmap_pivot = heatmap_data.pivot(index=heatmap_y, columns=heatmap_x, values=y_col)
            
            # Create heatmap
            fig = px.imshow(
                heatmap_pivot,
                labels=dict(x=heatmap_x, y=heatmap_y, color=f"Avg. {y_col}"),
                x=heatmap_pivot.columns,
                y=heatmap_pivot.index,
                aspect="auto",
                title=f"Custom {data_type} Heatmap by Month and {heatmap_y}"
            )
    
    elif chart_type == "Bubble Chart":
        if not has_groups:
            st.warning("Bubble chart works best with grouping columns.")
            
        # Choose bubble size metric
        size_options = ["count", "std", "range"]
        size_metric = st.selectbox(
            "Bubble Size Metric:",
            options=size_options,
            index=0,
            key=f"custom_{data_type.lower()}_bubble_size"
        )
        
        # Prepare data
        if has_groups:
            # Choose group column for bubble chart
            bubble_group = st.selectbox(
                "Group by:",
                options=group_columns,
                key=f"custom_{data_type.lower()}_bubble_group"
            )
            
            # Aggregate data
            if size_metric == "count":
                bubble_data = filtered_data.groupby(bubble_group)[y_col].agg(['mean', 'count']).reset_index()
                bubble_data.columns = [bubble_group, 'y', 'size']
            elif size_metric == "std":
                bubble_data = filtered_data.groupby(bubble_group)[y_col].agg(['mean', 'std']).reset_index()
                bubble_data.columns = [bubble_group, 'y', 'size']
                # Replace NaN with 0
                bubble_data['size'] = bubble_data['size'].fillna(0)
            else:  # range
                bubble_data = filtered_data.groupby(bubble_group)[y_col].agg(['mean', lambda x: x.max() - x.min()]).reset_index()
                bubble_data.columns = [bubble_group, 'y', 'size']
            
            # Create bubble chart
            fig = px.scatter(
                bubble_data,
                x=bubble_group,
                y='y',
                size='size',
                color=bubble_group,
                hover_name=bubble_group,
                size_max=40,
                title=f"Custom {data_type} Bubble Chart by {bubble_group} (Size: {size_metric})"
            )
        else:
            # Use months as x-axis
            if not pd.api.types.is_datetime64_any_dtype(filtered_data[date_col]):
                filtered_data[date_col] = pd.to_datetime(filtered_data[date_col])
                
            # Create month column
            filtered_data['month'] = filtered_data[date_col].dt.to_period('M').astype(str)
            
            # Aggregate data
            if size_metric == "count":
                bubble_data = filtered_data.groupby('month')[y_col].agg(['mean', 'count']).reset_index()
                bubble_data.columns = ['month', 'y', 'size']
            elif size_metric == "std":
                bubble_data = filtered_data.groupby('month')[y_col].agg(['mean', 'std']).reset_index()
                bubble_data.columns = ['month', 'y', 'size']
                # Replace NaN with 0
                bubble_data['size'] = bubble_data['size'].fillna(0)
            else:  # range
                bubble_data = filtered_data.groupby('month')[y_col].agg(['mean', lambda x: x.max() - x.min()]).reset_index()
                bubble_data.columns = ['month', 'y', 'size']
            
            # Create bubble chart
            fig = px.scatter(
                bubble_data,
                x='month',
                y='y',
                size='size',
                color='month',
                hover_name='month',
                size_max=40,
                title=f"Custom {data_type} Bubble Chart by Month (Size: {size_metric})"
            )
    
    # Update layout
    fig.update_layout(
        xaxis_title=x_col if chart_type in ["Line Chart", "Bar Chart", "Scatter Plot"] else "",
        yaxis_title=y_col if chart_type in ["Line Chart", "Bar Chart", "Scatter Plot", "Box Plot"] else "",
        template="plotly_white",
        height=500
    )
    
    # Display figure
    st.plotly_chart(fig, use_container_width=True)
    
    # Add download button
    chart_title = f"custom_{data_type.lower()}_{chart_type.lower().replace(' ', '_')}"
    add_download_button(fig, f"{chart_title}.png", "Download Chart")


def create_combined_chart(chart_type, historical_data, price_forecast, leadtime_forecast,
                         date_col, price_col, leadtime_col, group_columns):
    """Create combined chart with price and lead time data."""
    st.write("### Combined Price and Lead Time Visualization")
    
    # Only certain chart types are supported for combined visualization
    if chart_type not in ["Line Chart", "Scatter Plot", "Bubble Chart"]:
        st.warning(f"{chart_type} is not supported for combined visualization. Please use Line Chart, Scatter Plot, or Bubble Chart.")
        return
    
    # Data source options
    data_source = st.radio(
        "Data Source:",
        options=["Forecast Only", "Historical Only", "Both Historical and Forecast"],
        index=0,
        key="custom_combined_source",
        horizontal=True
    )
    
    # Prepare price data
    if data_source == "Forecast Only":
        price_data = price_forecast.copy()
        leadtime_data = leadtime_forecast.copy()
        data_label = "Forecast"
    elif data_source == "Historical Only":
        if historical_data is not None and price_col in historical_data.columns and leadtime_col in historical_data.columns:
            price_data = historical_data.copy()
            leadtime_data = historical_data.copy()
            data_label = "Historical"
        else:
            st.warning("No historical data available with both price and lead time.")
            return
    else:  # Both
        if historical_data is not None and price_col in historical_data.columns and leadtime_col in historical_data.columns:
            # Create combined dataframes with a source column
            hist_price = historical_data.copy()
            hist_price['data_source'] = 'Historical'
            
            fore_price = price_forecast.copy()
            fore_price['data_source'] = 'Forecast'
            
            hist_leadtime = historical_data.copy()
            hist_leadtime['data_source'] = 'Historical'
            
            fore_leadtime = leadtime_forecast.copy()
            fore_leadtime['data_source'] = 'Forecast'
            
            # Ensure both have the same columns for concatenation
            required_cols = [date_col, price_col, leadtime_col, 'data_source']
            if len(group_columns) > 0:
                required_cols.extend(group_columns)
            
            # Subset historical data
            if all(col in hist_price.columns for col in required_cols):
                hist_price_subset = hist_price[required_cols]
                hist_leadtime_subset = hist_leadtime[required_cols]
            else:
                missing_cols = [col for col in required_cols if col not in hist_price.columns]
                st.warning(f"Historical data missing columns: {missing_cols}. Using forecast data only.")
                price_data = fore_price
                leadtime_data = fore_leadtime
                data_label = 'data_source'
                return
            
            # Subset forecast data
            fore_price_subset = fore_price[[col for col in required_cols if col != leadtime_col]]
            fore_leadtime_subset = fore_leadtime[[col for col in required_cols if col != price_col]]
            
            # Combine data
            price_data = pd.concat([hist_price_subset, fore_price_subset], ignore_index=True)
            leadtime_data = pd.concat([hist_leadtime_subset, fore_leadtime_subset], ignore_index=True)
            data_label = "data_source"
        else:
            st.warning("No historical data available with both price and lead time. Using forecast data only.")
            price_data = price_forecast.copy()
            price_data['data_source'] = 'Forecast'
            
            leadtime_data = leadtime_forecast.copy()
            leadtime_data['data_source'] = 'Forecast'
            
            data_label = "data_source"
    
    # Determine if we have groups
    has_groups = len(group_columns) > 0 and all(col in price_data.columns for col in group_columns)
    
    # Filter options if we have groups
    if has_groups:
        st.write("#### Filter Data")
        
        # Create selectbox for each group column
        filters = {}
        cols = st.columns(min(3, len(group_columns)))
        
        for i, col_name in enumerate(group_columns):
            with cols[i % len(cols)]:
                # Get unique values for this column
                unique_values = sorted(price_data[col_name].unique())
                
                # Add "All" option
                options = ["All"] + unique_values
                
                # Create a selectbox for this column
                selected = st.selectbox(
                    f"Select {col_name}:",
                    options=options,
                    key=f"custom_combined_filter_{col_name}"
                )
                
                if selected != "All":
                    filters[col_name] = selected
        
        # Apply filters
        filtered_price = price_data.copy()
        filtered_leadtime = leadtime_data.copy()
        
        for col, val in filters.items():
            filtered_price = filtered_price[filtered_price[col] == val]
            filtered_leadtime = filtered_leadtime[filtered_leadtime[col] == val]
    else:
        filtered_price = price_data
        filtered_leadtime = leadtime_data
    
    # Chart specific options
    if chart_type == "Scatter Plot":
        # Create scatter plot of price vs. lead time
        st.write("#### Price vs. Lead Time Relationship")
        
        # Combine price and lead time data
        combined_data = pd.merge(
            filtered_price[[date_col, price_col] + (group_columns if has_groups else []) + 
                           (['data_source'] if 'data_source' in filtered_price.columns else [])], 
            filtered_leadtime[[date_col, leadtime_col]], 
            on=date_col
        )
        
        # Choose color column
        color_options = ["None"]
        if 'data_source' in combined_data.columns:
            color_options.append('data_source')
        if has_groups:
            color_options.extend(group_columns)
            
        color_col = st.selectbox(
            "Color by:",
            options=color_options,
            index=1 if 'data_source' in color_options else 0,
            key="custom_combined_scatter_color"
        )
        
        if color_col == "None":
            color_col = None
        
        # Create scatter plot
        fig = px.scatter(
            combined_data, 
            x=price_col, 
            y=leadtime_col,
            color=color_col,
            opacity=0.7,
            trendline="ols" if combined_data.shape[0] < 5000 else None,
            title="Price vs. Lead Time Relationship"
        )
        
        # Update layout
        fig.update_layout(
            xaxis_title=price_col.capitalize(),
            yaxis_title=leadtime_col.capitalize(),
            template="plotly_white",
            height=500
        )
    
    elif chart_type == "Line Chart":
        # Create dual-axis line chart
        st.write("#### Price and Lead Time Trends")
        
        # Ensure date column is datetime
        if not pd.api.types.is_datetime64_any_dtype(filtered_price[date_col]):
            filtered_price[date_col] = pd.to_datetime(filtered_price[date_col])
            
        if not pd.api.types.is_datetime64_any_dtype(filtered_leadtime[date_col]):
            filtered_leadtime[date_col] = pd.to_datetime(filtered_leadtime[date_col])
        
        # Sort by date
        filtered_price = filtered_price.sort_values(by=date_col)
        filtered_leadtime = filtered_leadtime.sort_values(by=date_col)
        
        # Choose color/grouping option
        color_options = ["None"]
        if 'data_source' in filtered_price.columns:
            color_options.append('data_source')
        if has_groups:
            color_options.extend(group_columns)
            
        group_by = st.selectbox(
            "Group by:",
            options=color_options,
            index=0,
            key="custom_combined_line_group"
        )
        
        if group_by == "None":
            # Simple dual-axis chart (no grouping)
            fig = make_subplots(specs=[[{"secondary_y": True}]])
            
            # Add price trace
            fig.add_trace(
                go.Scatter(
                    x=filtered_price[date_col], 
                    y=filtered_price[price_col],
                    name=price_col.capitalize(),
                    mode='lines+markers',
                    line=dict(color='blue', width=2),
                    marker=dict(size=8, color='blue')
                ),
                secondary_y=False
            )
            
            # Add lead time trace
            fig.add_trace(
                go.Scatter(
                    x=filtered_leadtime[date_col], 
                    y=filtered_leadtime[leadtime_col],
                    name=leadtime_col.capitalize(),
                    mode='lines+markers',
                    line=dict(color='red', width=2),
                    marker=dict(size=8, color='red')
                ),
                secondary_y=True
            )
            
            # Update layout
            fig.update_layout(
                title="Price and Lead Time Trends",
                template="plotly_white",
                height=500,
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="right",
                    x=1
                )
            )
            
            # Update axes
            fig.update_xaxes(title_text="Date")
            fig.update_yaxes(title_text=price_col.capitalize(), secondary_y=False)
            fig.update_yaxes(title_text=leadtime_col.capitalize(), secondary_y=True)
        
        else:
            # Grouped dual-axis chart
            # Create figure
            fig = go.Figure()
            
            # Get unique groups
            group_values = sorted(filtered_price[group_by].unique())
            
            # Add traces for each group
            for i, group_val in enumerate(group_values):
                # Filter data for this group
                price_group = filtered_price[filtered_price[group_by] == group_val]
                leadtime_group = filtered_leadtime[filtered_leadtime[group_by] == group_val]
                
                # Use different color for each group
                price_color = px.colors.qualitative.Plotly[i % len(px.colors.qualitative.Plotly)]
                leadtime_color = px.colors.qualitative.Set2[i % len(px.colors.qualitative.Set2)]
                
                # Add price trace
                fig.add_trace(go.Scatter(
                    x=price_group[date_col],
                    y=price_group[price_col],
                    mode='lines+markers',
                    name=f"{price_col.capitalize()} - {group_val}",
                    line=dict(color=price_color, width=2, dash='solid'),
                    marker=dict(size=8, color=price_color)
                ))
                
                # Add lead time trace
                fig.add_trace(go.Scatter(
                    x=leadtime_group[date_col],
                    y=leadtime_group[leadtime_col],
                    mode='lines+markers',
                    name=f"{leadtime_col.capitalize()} - {group_val}",
                    line=dict(color=leadtime_color, width=2, dash='dash'),
                    marker=dict(size=8, color=leadtime_color, symbol='square')
                ))
            
            # Update layout
            fig.update_layout(
                title=f"Price and Lead Time Trends by {group_by}",
                xaxis_title="Date",
                yaxis_title="Value",
                template="plotly_white",
                height=600,
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="right",
                    x=1
                )
            )
    
    elif chart_type == "Bubble Chart":
        st.write("#### Price and Lead Time Bubble Chart")
        
        # Combine price and lead time data
        combined_data = pd.merge(
            filtered_price[[date_col, price_col] + (group_columns if has_groups else []) + 
                          (['data_source'] if 'data_source' in filtered_price.columns else [])], 
            filtered_leadtime[[date_col, leadtime_col]], 
            on=date_col
        )
        
        # Create a size column (could be quantity if available)
        combined_data['size'] = 10  # Default size
        
        # Choose color column
        color_options = ["None"]
        if 'data_source' in combined_data.columns:
            color_options.append('data_source')
        if has_groups:
            color_options.extend(group_columns)
            
        color_col = st.selectbox(
            "Color by:",
            options=color_options,
            index=1 if 'data_source' in color_options else 0,
            key="custom_combined_bubble_color"
        )
        
        if color_col == "None":
            color_col = None
        
        # Create bubble chart
        fig = px.scatter(
            combined_data, 
            x=price_col, 
            y=leadtime_col,
            size='size',
            color=color_col,
            hover_name=date_col if pd.api.types.is_datetime64_any_dtype(combined_data[date_col]) else None,
            size_max=30,
            opacity=0.7,
            title="Price vs. Lead Time Bubble Chart"
        )
        
        # Update layout
        fig.update_layout(
            xaxis_title=price_col.capitalize(),
            yaxis_title=leadtime_col.capitalize(),
            template="plotly_white",
            height=500
        )
    
    # Display figure
    st.plotly_chart(fig, use_container_width=True)
    
    # Add download button
    chart_title = f"combined_{chart_type.lower().replace(' ', '_')}"
    add_download_button(fig, f"{chart_title}.png", "Download Chart")


def add_download_button(fig, filename, button_text):
    """Add download button for the Plotly figure."""
    # Create a buffer to save the image
    buf = io.BytesIO()
    
    # Save figure as PNG
    fig.write_image(buf, format='png', width=1200, height=800, scale=2)
    buf.seek(0)
    
    # Encode the image to base64
    encoded_image = base64.b64encode(buf.getvalue()).decode()
    
    # Create download link
    href = f'<a href="data:image/png;base64,{encoded_image}" download="{filename}">{button_text}</a>'
    
    # Display download link
    st.markdown(href, unsafe_allow_html=True)
    
    # Add some space
    st.write("")