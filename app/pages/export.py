"""
Export page for the forecasting application.

This module handles the export of forecasts and visualizations in various formats.
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
from fpdf import FPDF
import matplotlib.pyplot as plt
import json
import tempfile

# Add parent directory to path for importing
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

# Import utilities
from app.visualization.prediction_viz import ForecastVisualizer
from app.state.session import set_error, set_success, clear_messages


def show_export_page():
    """Display the export page."""
    st.header("Export Results")
    
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
    
    # Create tabs for different export options
    tabs = st.tabs([
        "Data Export", 
        "Visualization Export", 
        "Report Export"
    ])
    
    # Data Export tab
    with tabs[0]:
        show_data_export_options(price_forecast, leadtime_forecast, date_col, price_col, leadtime_col, group_columns)
    
    # Visualization Export tab
    with tabs[1]:
        show_visualization_export_options(price_forecast, leadtime_forecast, date_col, price_col, leadtime_col, group_columns)
    
    # Report Export tab
    with tabs[2]:
        show_report_export_options(price_forecast, leadtime_forecast, date_col, price_col, leadtime_col, group_columns)


def show_data_export_options(price_forecast, leadtime_forecast, date_col, price_col, leadtime_col, group_columns):
    """Show options for exporting forecast data."""
    st.subheader("Export Forecast Data")
    
    # Data selection options
    st.write("#### Select Data to Export")
    
    export_price = st.checkbox("Export Price Forecast", value=True, key="export_price")
    export_leadtime = st.checkbox("Export Lead Time Forecast", value=True, key="export_leadtime")
    combine_data = st.checkbox("Combine Price and Lead Time in One Export", value=True, key="combine_data")
    
    # Format options
    st.write("#### Export Format")
    
    export_format = st.radio(
        "Select Export Format:",
        options=["CSV", "Excel", "JSON"],
        index=1,  # Default to Excel
        horizontal=True,
        key="export_format"
    )
    
    # Filter options if we have group columns
    has_groups = len(group_columns) > 0 and all(col in price_forecast.columns for col in group_columns)
    
    if has_groups:
        st.write("#### Filter Data")
        
        # Option to filter or export all groups
        export_all_groups = st.checkbox("Export All Groups", value=True, key="export_all_groups")
        
        if not export_all_groups:
            # Create filters for each group column
            filters = {}
            cols = st.columns(min(3, len(group_columns)))
            
            for i, col_name in enumerate(group_columns):
                with cols[i % len(cols)]:
                    # Get unique values for this column
                    unique_values = sorted(price_forecast[col_name].unique())
                    
                    # Create a multiselect for this column
                    selected = st.multiselect(
                        f"Select {col_name}:",
                        options=unique_values,
                        default=unique_values[:1],
                        key=f"export_filter_{col_name}"
                    )
                    
                    if selected:
                        filters[col_name] = selected
            
            # Apply filters
            if filters:
                filtered_price = price_forecast.copy()
                filtered_leadtime = leadtime_forecast.copy()
                
                for col, values in filters.items():
                    filtered_price = filtered_price[filtered_price[col].isin(values)]
                    filtered_leadtime = filtered_leadtime[filtered_leadtime[col].isin(values)]
            else:
                filtered_price = price_forecast
                filtered_leadtime = leadtime_forecast
        else:
            filtered_price = price_forecast
            filtered_leadtime = leadtime_forecast
    else:
        filtered_price = price_forecast
        filtered_leadtime = leadtime_forecast
    
    # Date range options
    st.write("#### Date Range")
    
    # Convert date column to datetime if needed
    if not pd.api.types.is_datetime64_any_dtype(filtered_price[date_col]):
        filtered_price[date_col] = pd.to_datetime(filtered_price[date_col])
    
    if not pd.api.types.is_datetime64_any_dtype(filtered_leadtime[date_col]):
        filtered_leadtime[date_col] = pd.to_datetime(filtered_leadtime[date_col])
    
    # Get min and max dates
    min_date = min(filtered_price[date_col].min(), filtered_leadtime[date_col].min())
    max_date = max(filtered_price[date_col].max(), filtered_leadtime[date_col].max())
    
    # Date range selector
    export_all_dates = st.checkbox("Export All Dates", value=True, key="export_all_dates")
    
    if not export_all_dates:
        date_range = st.date_input(
            "Select Date Range:",
            value=(min_date.date(), max_date.date()),
            min_value=min_date.date(),
            max_value=max_date.date(),
            key="export_date_range"
        )
        
        if len(date_range) == 2:
            start_date, end_date = date_range
            
            # Convert to datetime for filtering
            start_datetime = pd.Timestamp(start_date)
            end_datetime = pd.Timestamp(end_date)
            
            # Apply date filters
            filtered_price = filtered_price[(filtered_price[date_col] >= start_datetime) & 
                                           (filtered_price[date_col] <= end_datetime)]
            
            filtered_leadtime = filtered_leadtime[(filtered_leadtime[date_col] >= start_datetime) & 
                                                 (filtered_leadtime[date_col] <= end_datetime)]
    
    # Column selection options
    st.write("#### Column Selection")
    
    # Get available columns
    available_columns = list(filtered_price.columns)
    if combine_data:
        available_columns = list(set(available_columns + list(filtered_leadtime.columns)))
    
    # Always include date and key columns
    default_columns = [date_col, price_col, leadtime_col] + group_columns
    default_columns = [col for col in default_columns if col in available_columns]
    
    # Remove duplicates while preserving order
    default_columns = list(dict.fromkeys(default_columns))
    
    # Column selection
    selected_columns = st.multiselect(
        "Select Columns to Include:",
        options=available_columns,
        default=default_columns,
        key="export_columns"
    )
    
    # Export button and actions
    st.write("#### Generate Export")
    
    export_filename = st.text_input(
        "Export Filename (without extension):",
        value="forecast_export",
        key="export_filename"
    )
    
    if st.button("Generate Export", type="primary"):
        if not export_price and not export_leadtime:
            st.error("Please select at least one data type to export (Price or Lead Time).")
            return
        
        if not selected_columns:
            st.error("Please select at least one column to export.")
            return
        
        try:
            # Prepare data for export
            if combine_data and export_price and export_leadtime:
                # Create combined dataframe with both price and lead time data
                common_columns = [col for col in filtered_price.columns if col in filtered_leadtime.columns]
                
                # Select columns to keep from each dataset
                price_cols = [col for col in selected_columns if col in filtered_price.columns]
                leadtime_cols = [col for col in selected_columns if col in filtered_leadtime.columns]
                
                # Ensure we have the date column to merge on
                if date_col not in common_columns:
                    st.error(f"Cannot merge datasets: {date_col} column not found in both datasets.")
                    return
                
                # Handle case where price_col or leadtime_col is not in the selected columns
                if price_col not in price_cols:
                    price_cols.append(price_col)
                if leadtime_col not in leadtime_cols:
                    leadtime_cols.append(leadtime_col)
                
                # Select columns from each dataset
                price_data = filtered_price[price_cols].copy()
                leadtime_data = filtered_leadtime[leadtime_cols].copy()
                
                # Group by necessary columns based on selected columns
                group_cols = [col for col in common_columns if col in selected_columns]
                
                # Combine datasets
                export_data = pd.merge(price_data, leadtime_data, on=group_cols)
            elif export_price and not export_leadtime:
                # Export only price forecast
                available_cols = [col for col in selected_columns if col in filtered_price.columns]
                export_data = filtered_price[available_cols].copy()
            elif export_leadtime and not export_price:
                # Export only lead time forecast
                available_cols = [col for col in selected_columns if col in filtered_leadtime.columns]
                export_data = filtered_leadtime[available_cols].copy()
            else:
                st.error("Please select at least one data type to export (Price or Lead Time).")
                return
            
            # Create export
            if export_format == "CSV":
                csv_export = create_csv_export(export_data)
                create_download_link(csv_export, f"{export_filename}.csv", "Download CSV")
            elif export_format == "Excel":
                excel_export = create_excel_export(export_data, export_price, export_leadtime, 
                                                  price_forecast, leadtime_forecast, selected_columns)
                create_download_link(excel_export, f"{export_filename}.xlsx", "Download Excel")
            elif export_format == "JSON":
                json_export = create_json_export(export_data)
                create_download_link(json_export, f"{export_filename}.json", "Download JSON")
            
            # Show success message
            set_success(f"Export generated successfully! Click the download link to save the file.")
            
        except Exception as e:
            set_error(f"Error generating export: {str(e)}")
            raise e


def create_csv_export(data):
    """Create a CSV export from the given data."""
    # Create a buffer to save the CSV
    buf = io.StringIO()
    
    # Save dataframe as CSV
    data.to_csv(buf, index=False)
    buf.seek(0)
    
    return buf.getvalue()


def create_excel_export(data, export_price, export_leadtime, price_forecast, leadtime_forecast, selected_columns):
    """Create an Excel export with multiple sheets."""
    # Create a buffer to save the Excel file
    buf = io.BytesIO()
    
    # Create Excel writer
    with pd.ExcelWriter(buf, engine='xlsxwriter') as writer:
        # Write combined data to main sheet
        data.to_excel(writer, sheet_name='Forecast Data', index=False)
        
        # Add metadata sheet
        metadata = pd.DataFrame([{
            'Creation Date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'Data Points': len(data),
            'Selected Columns': ', '.join(selected_columns),
            'Includes Price Data': 'Yes' if export_price else 'No',
            'Includes Lead Time Data': 'Yes' if export_leadtime else 'No',
            'Total Price Forecasts': len(price_forecast) if export_price else 0,
            'Total Lead Time Forecasts': len(leadtime_forecast) if export_leadtime else 'No',
        }])
        metadata.to_excel(writer, sheet_name='Metadata', index=False)
        
        # Add statistics sheet
        if export_price or export_leadtime:
            stats_data = []
            
            if export_price and 'price' in data.columns:
                price_stats = {
                    'Metric': 'Price',
                    'Min': data['price'].min(),
                    'Max': data['price'].max(),
                    'Mean': data['price'].mean(),
                    'Median': data['price'].median(),
                    'Std Dev': data['price'].std()
                }
                stats_data.append(price_stats)
            
            if export_leadtime and 'lead_time' in data.columns:
                leadtime_stats = {
                    'Metric': 'Lead Time',
                    'Min': data['lead_time'].min(),
                    'Max': data['lead_time'].max(),
                    'Mean': data['lead_time'].mean(),
                    'Median': data['lead_time'].median(),
                    'Std Dev': data['lead_time'].std()
                }
                stats_data.append(leadtime_stats)
            
            if stats_data:
                stats_df = pd.DataFrame(stats_data)
                stats_df.to_excel(writer, sheet_name='Statistics', index=False)
    
    # Get the bytes data
    buf.seek(0)
    
    return buf.getvalue()


def create_json_export(data):
    """Create a JSON export from the given data."""
    # Convert dates to strings for JSON serialization
    for col in data.columns:
        if pd.api.types.is_datetime64_any_dtype(data[col]):
            data[col] = data[col].dt.strftime('%Y-%m-%d')
    
    # Create a buffer to save the JSON
    json_data = data.to_json(orient='records', date_format='iso')
    
    return json_data


def show_visualization_export_options(price_forecast, leadtime_forecast, date_col, price_col, leadtime_col, group_columns):
    """Show options for exporting visualizations."""
    st.subheader("Export Visualizations")
    
    # Create visualization selection
    st.write("#### Select Visualization Type")
    
    viz_type = st.selectbox(
        "Visualization Type:",
        options=["Time Series", "Comparison", "Distribution", "Combined"],
        key="export_viz_type"
    )
    
    # Get historical data if needed
    historical_data = None
    if st.session_state.data_clean is not None:
        historical_data = st.session_state.data_clean
    else:
        historical_data = st.session_state.data
    
    # Filter options if we have group columns
    has_groups = len(group_columns) > 0 and all(col in price_forecast.columns for col in group_columns)
    
    # Time Series visualization options
    if viz_type == "Time Series":
        st.write("#### Time Series Options")
        
        # Data selection
        export_price_ts = st.checkbox("Include Price", value=True, key="export_price_ts")
        export_leadtime_ts = st.checkbox("Include Lead Time", value=True, key="export_leadtime_ts")
        include_historical = st.checkbox("Include Historical Data", value=True, key="include_historical_ts")
        
        # Group selection if available
        selected_groups = {}
        if has_groups:
            st.write("#### Group Selection")
            
            # Create filters for each group column
            cols = st.columns(min(3, len(group_columns)))
            
            for i, col_name in enumerate(group_columns):
                with cols[i % len(cols)]:
                    # Get unique values for this column
                    unique_values = sorted(price_forecast[col_name].unique())
                    
                    # Add "All" option
                    options = ["All"] + unique_values
                    
                    # Create a selectbox for this column
                    selected = st.selectbox(
                        f"Select {col_name}:",
                        options=options,
                        key=f"export_ts_group_{col_name}"
                    )
                    
                    if selected != "All":
                        selected_groups[col_name] = selected
            
        # Format options
        st.write("#### Export Format")
        
        export_format_ts = st.radio(
            "Select Image Format:",
            options=["PNG", "JPEG", "PDF", "SVG"],
            index=0,
            horizontal=True,
            key="export_format_ts"
        )
        
        # Quality/resolution options
        resolution = st.select_slider(
            "Image Resolution:",
            options=["Low (800x600)", "Medium (1200x800)", "High (1600x1000)", "Very High (2000x1200)"],
            value="Medium (1200x800)",
            key="export_resolution_ts"
        )
        
        # Get dimensions from resolution setting
        width, height = get_resolution_dimensions(resolution)
        
        # Export filename
        st.write("#### Export Filename")
        
        export_filename_ts = st.text_input(
            "Export Filename (without extension):",
            value="time_series_visualization",
            key="export_filename_ts"
        )
        
        # Generate export button
        if st.button("Generate Visualization", key="generate_ts_viz"):
            if not export_price_ts and not export_leadtime_ts:
                st.error("Please select at least one data type to include (Price or Lead Time).")
                return
            
            # Create the time series visualization
            try:
                if export_price_ts:
                    # Filter forecast data if groups selected
                    filtered_price = price_forecast.copy()
                    filtered_historical = historical_data.copy() if include_historical else None
                    
                    for col, val in selected_groups.items():
                        filtered_price = filtered_price[filtered_price[col] == val]
                        if include_historical and filtered_historical is not None:
                            if col in filtered_historical.columns:
                                filtered_historical = filtered_historical[filtered_historical[col] == val]
                    
                    # Create visualization
                    fig_price = create_time_series_viz(
                        filtered_historical, 
                        filtered_price, 
                        date_col, 
                        price_col,
                        title=f"Price Forecast over Time" + (f" for {', '.join([f'{k}: {v}' for k, v in selected_groups.items()])}" if selected_groups else ""),
                        show_historical=include_historical
                    )
                    
                    # Create download link
                    img_bytes_price = export_figure(fig_price, export_format_ts, width, height)
                    create_download_link(img_bytes_price, f"{export_filename_ts}_price.{export_format_ts.lower()}", 
                                        f"Download Price {export_format_ts}")
                
                if export_leadtime_ts:
                    # Filter forecast data if groups selected
                    filtered_leadtime = leadtime_forecast.copy()
                    filtered_historical = historical_data.copy() if include_historical else None
                    
                    for col, val in selected_groups.items():
                        filtered_leadtime = filtered_leadtime[filtered_leadtime[col] == val]
                        if include_historical and filtered_historical is not None:
                            if col in filtered_historical.columns:
                                filtered_historical = filtered_historical[filtered_historical[col] == val]
                    
                    # Create visualization
                    fig_leadtime = create_time_series_viz(
                        filtered_historical, 
                        filtered_leadtime, 
                        date_col, 
                        leadtime_col,
                        title=f"Lead Time Forecast over Time" + (f" for {', '.join([f'{k}: {v}' for k, v in selected_groups.items()])}" if selected_groups else ""),
                        show_historical=include_historical
                    )
                    
                    # Create download link
                    img_bytes_leadtime = export_figure(fig_leadtime, export_format_ts, width, height)
                    create_download_link(img_bytes_leadtime, f"{export_filename_ts}_leadtime.{export_format_ts.lower()}", 
                                        f"Download Lead Time {export_format_ts}")
                
                # Show success message
                set_success(f"Visualization(s) generated successfully! Click the download link(s) to save.")
                
            except Exception as e:
                set_error(f"Error generating visualization: {str(e)}")
                raise e
    
    # Comparison visualization options
    elif viz_type == "Comparison":
        if not has_groups:
            st.warning("Comparison visualization requires group columns. Please generate forecasts with group columns first.")
            return
        
        st.write("#### Comparison Options")
        
        # Data selection
        export_price_comp = st.checkbox("Include Price", value=True, key="export_price_comp")
        export_leadtime_comp = st.checkbox("Include Lead Time", value=True, key="export_leadtime_comp")
        
        # Select primary grouping column for comparison
        comp_group_col = st.selectbox(
            "Compare by:",
            options=group_columns,
            key="export_comp_group"
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
                            key=f"export_comp_sec_{col_name}"
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
            key="export_top_n_groups"
        )
        
        # Select specific date for bar chart comparison
        forecast_dates = price_forecast[date_col].sort_values().unique()
        if len(forecast_dates) > 0:
            selected_date = st.select_slider(
                "Select date for comparison:",
                options=forecast_dates,
                value=forecast_dates[-1] if len(forecast_dates) > 0 else None,
                key="export_comp_date"
            )
        else:
            selected_date = None
            st.warning("No dates available in forecast data.")
            return
        
        # Format options
        st.write("#### Export Format")
        
        export_format_comp = st.radio(
            "Select Image Format:",
            options=["PNG", "JPEG", "PDF", "SVG"],
            index=0,
            horizontal=True,
            key="export_format_comp"
        )
        
        # Quality/resolution options
        resolution_comp = st.select_slider(
            "Image Resolution:",
            options=["Low (800x600)", "Medium (1200x800)", "High (1600x1000)", "Very High (2000x1200)"],
            value="Medium (1200x800)",
            key="export_resolution_comp"
        )
        
        # Get dimensions from resolution setting
        width, height = get_resolution_dimensions(resolution_comp)
        
        # Export filename
        st.write("#### Export Filename")
        
        export_filename_comp = st.text_input(
            "Export Filename (without extension):",
            value="comparison_visualization",
            key="export_filename_comp"
        )
        
        # Generate export button
        if st.button("Generate Visualization", key="generate_comp_viz"):
            if not export_price_comp and not export_leadtime_comp:
                st.error("Please select at least one data type to include (Price or Lead Time).")
                return
            
            # Create comparison visualization
            try:
                # Apply secondary filters
                filtered_price = price_forecast.copy()
                filtered_leadtime = leadtime_forecast.copy()
                
                for col, val in secondary_filters.items():
                    filtered_price = filtered_price[filtered_price[col] == val]
                    filtered_leadtime = filtered_leadtime[filtered_leadtime[col] == val]
                
                # Find top N groups based on average price/lead time
                if export_price_comp:
                    top_price_groups = filtered_price.groupby(comp_group_col)[price_col].mean().nlargest(top_n).index.tolist()
                else:
                    top_price_groups = []
                
                if export_leadtime_comp:
                    top_leadtime_groups = filtered_leadtime.groupby(comp_group_col)[leadtime_col].mean().nlargest(top_n).index.tolist()
                else:
                    top_leadtime_groups = []
                
                # Combine top groups
                top_groups = list(set(top_price_groups + top_leadtime_groups))[:top_n]
                
                # Filter to top groups and selected date
                if selected_date:
                    date_filtered_price = filtered_price[
                        (filtered_price[date_col] == selected_date) & 
                        (filtered_price[comp_group_col].isin(top_groups))
                    ]
                    
                    date_filtered_leadtime = filtered_leadtime[
                        (filtered_leadtime[date_col] == selected_date) & 
                        (filtered_leadtime[comp_group_col].isin(top_groups))
                    ]
                else:
                    date_filtered_price = filtered_price[filtered_price[comp_group_col].isin(top_groups)]
                    date_filtered_leadtime = filtered_leadtime[filtered_leadtime[comp_group_col].isin(top_groups)]
                
                # Create bar chart comparisons for selected date
                if export_price_comp and not date_filtered_price.empty:
                    fig_price_comp = px.bar(
                        date_filtered_price.sort_values(price_col, ascending=False), 
                        x=comp_group_col, 
                        y=price_col,
                        color=comp_group_col,
                        title=f"Price Comparison by {comp_group_col}" + 
                              (f" (Filtered by {', '.join([f'{k}: {v}' for k, v in secondary_filters.items()])})" if secondary_filters else "")
                    )
                    
                    # Update layout
                    fig_price_comp.update_layout(
                        xaxis_title=comp_group_col,
                        yaxis_title=price_col.capitalize(),
                        template="plotly_white",
                        height=height,
                        width=width
                    )
                    
                    # Create download link
                    img_bytes_price_comp = export_figure(fig_price_comp, export_format_comp, width, height)
                    create_download_link(img_bytes_price_comp, f"{export_filename_comp}_price_bar.{export_format_comp.lower()}", 
                                        f"Download Price Bar Chart {export_format_comp}")
                
                if export_leadtime_comp and not date_filtered_leadtime.empty:
                    fig_leadtime_comp = px.bar(
                        date_filtered_leadtime.sort_values(leadtime_col, ascending=False), 
                        x=comp_group_col, 
                        y=leadtime_col,
                        color=comp_group_col,
                        title=f"Lead Time Comparison by {comp_group_col}" + 
                              (f" (Filtered by {', '.join([f'{k}: {v}' for k, v in secondary_filters.items()])})" if secondary_filters else "")
                    )
                    
                    # Update layout
                    fig_leadtime_comp.update_layout(
                        xaxis_title=comp_group_col,
                        yaxis_title=leadtime_col.capitalize(),
                        template="plotly_white",
                        height=height,
                        width=width
                    )
                    
                    # Create download link
                    img_bytes_leadtime_comp = export_figure(fig_leadtime_comp, export_format_comp, width, height)
                    create_download_link(img_bytes_leadtime_comp, f"{export_filename_comp}_leadtime_bar.{export_format_comp.lower()}", 
                                        f"Download Lead Time Bar Chart {export_format_comp}")
                
                # Create line chart comparisons across groups
                if export_price_comp:
                    # Filter to top groups for time series
                    time_filtered_price = filtered_price[filtered_price[comp_group_col].isin(top_groups)]
                    
                    if not time_filtered_price.empty:
                        fig_price_time = px.line(
                            time_filtered_price, 
                            x=date_col, 
                            y=price_col, 
                            color=comp_group_col,
                            title=f"Price Forecast Comparison by {comp_group_col}" + 
                                  (f" (Filtered by {', '.join([f'{k}: {v}' for k, v in secondary_filters.items()])})" if secondary_filters else ""),
                            markers=True
                        )
                        
                        # Update layout
                        fig_price_time.update_layout(
                            xaxis_title="Date",
                            yaxis_title=price_col.capitalize(),
                            template="plotly_white",
                            height=height,
                            width=width
                        )
                        
                        # Create download link
                        img_bytes_price_time = export_figure(fig_price_time, export_format_comp, width, height)
                        create_download_link(img_bytes_price_time, f"{export_filename_comp}_price_line.{export_format_comp.lower()}", 
                                            f"Download Price Line Chart {export_format_comp}")
                
                if export_leadtime_comp:
                    # Filter to top groups for time series
                    time_filtered_leadtime = filtered_leadtime[filtered_leadtime[comp_group_col].isin(top_groups)]
                    
                    if not time_filtered_leadtime.empty:
                        fig_leadtime_time = px.line(
                            time_filtered_leadtime, 
                            x=date_col, 
                            y=leadtime_col, 
                            color=comp_group_col,
                            title=f"Lead Time Forecast Comparison by {comp_group_col}" + 
                                  (f" (Filtered by {', '.join([f'{k}: {v}' for k, v in secondary_filters.items()])})" if secondary_filters else ""),
                            markers=True
                        )
                        
                        # Update layout
                        fig_leadtime_time.update_layout(
                            xaxis_title="Date",
                            yaxis_title=leadtime_col.capitalize(),
                            template="plotly_white",
                            height=height,
                            width=width
                        )
                        
                        # Create download link
                        img_bytes_leadtime_time = export_figure(fig_leadtime_time, export_format_comp, width, height)
                        create_download_link(img_bytes_leadtime_time, f"{export_filename_comp}_leadtime_line.{export_format_comp.lower()}", 
                                            f"Download Lead Time Line Chart {export_format_comp}")
                
                # Show success message
                set_success(f"Visualization(s) generated successfully! Click the download link(s) to save.")
                
            except Exception as e:
                set_error(f"Error generating visualization: {str(e)}")
                raise e
    
    # Distribution visualization options
    elif viz_type == "Distribution":
        st.write("#### Distribution Options")
        
        # Data selection
        export_price_dist = st.checkbox("Include Price", value=True, key="export_price_dist")
        export_leadtime_dist = st.checkbox("Include Lead Time", value=True, key="export_leadtime_dist")
        
        # Group selection if available
        group_by = None
        if has_groups:
            st.write("#### Group Selection")
            
            # Select whether to group
            use_groups = st.checkbox("Group by Category", value=True, key="use_groups_dist")
            
            if use_groups:
                # Select grouping column
                group_by = st.selectbox(
                    "Group by:",
                    options=group_columns,
                    key="export_dist_group"
                )
            
        # Format options
        st.write("#### Export Format")
        
        export_format_dist = st.radio(
            "Select Image Format:",
            options=["PNG", "JPEG", "PDF", "SVG"],
            index=0,
            horizontal=True,
            key="export_format_dist"
        )
        
        # Quality/resolution options
        resolution_dist = st.select_slider(
            "Image Resolution:",
            options=["Low (800x600)", "Medium (1200x800)", "High (1600x1000)", "Very High (2000x1200)"],
            value="Medium (1200x800)",
            key="export_resolution_dist"
        )
        
        # Get dimensions from resolution setting
        width, height = get_resolution_dimensions(resolution_dist)
        
        # Export filename
        st.write("#### Export Filename")
        
        export_filename_dist = st.text_input(
            "Export Filename (without extension):",
            value="distribution_visualization",
            key="export_filename_dist"
        )
        
        # Generate export button
        if st.button("Generate Visualization", key="generate_dist_viz"):
            if not export_price_dist and not export_leadtime_dist:
                st.error("Please select at least one data type to include (Price or Lead Time).")
                return
            
            # Create distribution visualization
            try:
                # For price distribution
                if export_price_dist:
                    if group_by and has_groups:
                        # Create box plot by group
                        fig_price_dist = px.box(
                            price_forecast, 
                            x=group_by, 
                            y=price_col,
                            color=group_by,
                            title=f"Price Distribution by {group_by}"
                        )
                    else:
                        # Create histogram
                        price_stats = {
                            'Minimum': price_forecast[price_col].min(),
                            'Maximum': price_forecast[price_col].max(),
                            'Average': price_forecast[price_col].mean(),
                            'Median': price_forecast[price_col].median()
                        }
                        
                        fig_price_dist = px.histogram(
                            price_forecast, 
                            x=price_col,
                            marginal="box", 
                            title="Price Forecast Distribution",
                            histnorm='probability density',
                            opacity=0.7
                        )
                        
                        # Add mean and median lines
                        fig_price_dist.add_vline(x=price_stats['Average'], line_dash="solid", line_width=2, line_color="red", 
                                              annotation_text=f"Mean: {price_stats['Average']:.2f}", annotation_position="top right")
                        
                        fig_price_dist.add_vline(x=price_stats['Median'], line_dash="dash", line_width=2, line_color="green", 
                                              annotation_text=f"Median: {price_stats['Median']:.2f}", annotation_position="top left")
                    
                    # Update layout
                    fig_price_dist.update_layout(
                        height=height,
                        width=width,
                        template="plotly_white"
                    )
                    
                    # Create download link
                    img_bytes_price_dist = export_figure(fig_price_dist, export_format_dist, width, height)
                    create_download_link(img_bytes_price_dist, f"{export_filename_dist}_price.{export_format_dist.lower()}", 
                                        f"Download Price Distribution {export_format_dist}")
                
                # For lead time distribution
                if export_leadtime_dist:
                    if group_by and has_groups:
                        # Create box plot by group
                        fig_leadtime_dist = px.box(
                            leadtime_forecast, 
                            x=group_by, 
                            y=leadtime_col,
                            color=group_by,
                            title=f"Lead Time Distribution by {group_by}"
                        )
                    else:
                        # Create histogram
                        leadtime_stats = {
                            'Minimum': leadtime_forecast[leadtime_col].min(),
                            'Maximum': leadtime_forecast[leadtime_col].max(),
                            'Average': leadtime_forecast[leadtime_col].mean(),
                            'Median': leadtime_forecast[leadtime_col].median()
                        }
                        
                        fig_leadtime_dist = px.histogram(
                            leadtime_forecast, 
                            x=leadtime_col,
                            marginal="box", 
                            title="Lead Time Forecast Distribution",
                            histnorm='probability density',
                            opacity=0.7
                        )
                        
                        # Add mean and median lines
                        fig_leadtime_dist.add_vline(x=leadtime_stats['Average'], line_dash="solid", line_width=2, line_color="red", 
                                              annotation_text=f"Mean: {leadtime_stats['Average']:.2f}", annotation_position="top right")
                        
                        fig_leadtime_dist.add_vline(x=leadtime_stats['Median'], line_dash="dash", line_width=2, line_color="green", 
                                              annotation_text=f"Median: {leadtime_stats['Median']:.2f}", annotation_position="top left")
                    
                    # Update layout
                    fig_leadtime_dist.update_layout(
                        height=height,
                        width=width,
                        template="plotly_white"
                    )
                    
                    # Create download link
                    img_bytes_leadtime_dist = export_figure(fig_leadtime_dist, export_format_dist, width, height)
                    create_download_link(img_bytes_leadtime_dist, f"{export_filename_dist}_leadtime.{export_format_dist.lower()}", 
                                        f"Download Lead Time Distribution {export_format_dist}")
                
                # Show success message
                set_success(f"Visualization(s) generated successfully! Click the download link(s) to save.")
                
            except Exception as e:
                set_error(f"Error generating visualization: {str(e)}")
                raise e
    
    # Combined visualization options
    elif viz_type == "Combined":
        st.write("#### Combined Visualization Options")
        
        # Chart type
        chart_type = st.selectbox(
            "Chart Type:",
            options=["Scatter Plot", "Bubble Chart"],
            key="export_combined_type"
        )
        
        # Group selection if available
        color_by = None
        if has_groups:
            st.write("#### Group Selection")
            
            # Select whether to use color grouping
            use_color = st.checkbox("Color by Category", value=True, key="use_color_combined")
            
            if use_color:
                # Select color column
                color_options = ["None"] + group_columns
                if 'data_source' in price_forecast.columns:
                    color_options.append('data_source')
                    
                color_by = st.selectbox(
                    "Color by:",
                    options=color_options,
                    index=1,
                    key="export_combined_color"
                )
                
                if color_by == "None":
                    color_by = None
        
        # Format options
        st.write("#### Export Format")
        
        export_format_combined = st.radio(
            "Select Image Format:",
            options=["PNG", "JPEG", "PDF", "SVG"],
            index=0,
            horizontal=True,
            key="export_format_combined"
        )
        
        # Quality/resolution options
        resolution_combined = st.select_slider(
            "Image Resolution:",
            options=["Low (800x600)", "Medium (1200x800)", "High (1600x1000)", "Very High (2000x1200)"],
            value="Medium (1200x800)",
            key="export_resolution_combined"
        )
        
        # Get dimensions from resolution setting
        width, height = get_resolution_dimensions(resolution_combined)
        
        # Export filename
        st.write("#### Export Filename")
        
        export_filename_combined = st.text_input(
            "Export Filename (without extension):",
            value="combined_visualization",
            key="export_filename_combined"
        )
        
        # Generate export button
        if st.button("Generate Visualization", key="generate_combined_viz"):
            # Create combined visualization
            try:
                # Combine price and lead time data
                combined_data = pd.merge(
                    price_forecast[[date_col, price_col] + (group_columns if has_groups else []) + 
                                  (['data_source'] if 'data_source' in price_forecast.columns else [])], 
                    leadtime_forecast[[date_col, leadtime_col]], 
                    on=date_col
                )
                
                if chart_type == "Scatter Plot":
                    # Create scatter plot of price vs. lead time
                    fig_combined = px.scatter(
                        combined_data, 
                        x=price_col, 
                        y=leadtime_col,
                        color=color_by,
                        opacity=0.7,
                        trendline="ols" if combined_data.shape[0] < 5000 else None,
                        title="Price vs. Lead Time Relationship"
                    )
                    
                    # Update layout
                    fig_combined.update_layout(
                        xaxis_title=price_col.capitalize(),
                        yaxis_title=leadtime_col.capitalize(),
                        template="plotly_white",
                        height=height,
                        width=width
                    )
                
                elif chart_type == "Bubble Chart":
                    # Create a size column (could be quantity if available)
                    combined_data['size'] = 10  # Default size
                    
                    # Create bubble chart
                    fig_combined = px.scatter(
                        combined_data, 
                        x=price_col, 
                        y=leadtime_col,
                        size='size',
                        color=color_by,
                        hover_name=date_col if pd.api.types.is_datetime64_any_dtype(combined_data[date_col]) else None,
                        size_max=30,
                        opacity=0.7,
                        title="Price vs. Lead Time Bubble Chart"
                    )
                    
                    # Update layout
                    fig_combined.update_layout(
                        xaxis_title=price_col.capitalize(),
                        yaxis_title=leadtime_col.capitalize(),
                        template="plotly_white",
                        height=height,
                        width=width
                    )
                
                # Create download link
                img_bytes_combined = export_figure(fig_combined, export_format_combined, width, height)
                create_download_link(img_bytes_combined, f"{export_filename_combined}.{export_format_combined.lower()}", 
                                    f"Download Combined Visualization {export_format_combined}")
                
                # Show success message
                set_success(f"Visualization generated successfully! Click the download link to save.")
                
            except Exception as e:
                set_error(f"Error generating visualization: {str(e)}")
                raise e


def show_report_export_options(price_forecast, leadtime_forecast, date_col, price_col, leadtime_col, group_columns):
    """Show options for exporting complete reports."""
    st.subheader("Export Complete Reports")
    
    # Report content options
    st.write("#### Report Content")
    
    include_price = st.checkbox("Include Price Forecasts", value=True, key="report_include_price")
    include_leadtime = st.checkbox("Include Lead Time Forecasts", value=True, key="report_include_leadtime")
    include_data_overview = st.checkbox("Include Data Overview", value=True, key="report_include_overview")
    include_statistics = st.checkbox("Include Statistical Analysis", value=True, key="report_include_stats")
    include_visualizations = st.checkbox("Include Visualizations", value=True, key="report_include_viz")
    include_metadata = st.checkbox("Include Report Metadata", value=True, key="report_include_metadata")
    
    # Report format
    st.write("#### Report Format")
    
    report_format = st.radio(
        "Select Report Format:",
        options=["PDF", "Excel Report", "HTML"],
        index=0,
        horizontal=True,
        key="report_format"
    )
    
    # Filter options if we have group columns
    has_groups = len(group_columns) > 0 and all(col in price_forecast.columns for col in group_columns)
    
    # Group selection for report
    selected_groups = {}
    if has_groups:
        st.write("#### Group Selection")
        
        # Option to generate reports for all groups or specific groups
        report_all_groups = st.radio(
            "Generate For:",
            options=["All Groups", "Selected Groups", "One Consolidated Report"],
            index=2,
            horizontal=True,
            key="report_group_option"
        )
        
        if report_all_groups == "Selected Groups":
            # Create filters for each group column
            cols = st.columns(min(3, len(group_columns)))
            
            for i, col_name in enumerate(group_columns):
                with cols[i % len(cols)]:
                    # Get unique values for this column
                    unique_values = sorted(price_forecast[col_name].unique())
                    
                    # Create a multiselect for this column
                    selected = st.multiselect(
                        f"Select {col_name}:",
                        options=unique_values,
                        default=unique_values[:1],
                        key=f"report_filter_{col_name}"
                    )
                    
                    if selected:
                        selected_groups[col_name] = selected
    
    # Export filename
    st.write("#### Export Filename")
    
    report_filename = st.text_input(
        "Export Filename (without extension):",
        value="forecast_report",
        key="report_filename"
    )
    
    # Generate report button
    if st.button("Generate Report", key="generate_report", type="primary"):
        if not include_price and not include_leadtime:
            st.error("Please include at least one of Price Forecasts or Lead Time Forecasts.")
            return
        
        # Create report
        try:
            # Get historical data
            historical_data = None
            if st.session_state.data_clean is not None:
                historical_data = st.session_state.data_clean
            else:
                historical_data = st.session_state.data
            
            # Filter data for specific groups if needed
            if has_groups and report_all_groups == "Selected Groups" and selected_groups:
                # Apply filters
                filtered_price = price_forecast.copy()
                filtered_leadtime = leadtime_forecast.copy()
                filtered_historical = historical_data.copy() if historical_data is not None else None
                
                for col, values in selected_groups.items():
                    filtered_price = filtered_price[filtered_price[col].isin(values)]
                    filtered_leadtime = filtered_leadtime[filtered_leadtime[col].isin(values)]
                    if filtered_historical is not None and col in filtered_historical.columns:
                        filtered_historical = filtered_historical[filtered_historical[col].isin(values)]
            else:
                filtered_price = price_forecast
                filtered_leadtime = leadtime_forecast
                filtered_historical = historical_data
            
            # Create report based on format
            if report_format == "PDF":
                # Create PDF report
                pdf_bytes = create_pdf_report(
                    filtered_price, filtered_leadtime, filtered_historical,
                    date_col, price_col, leadtime_col, group_columns,
                    include_price, include_leadtime, include_data_overview,
                    include_statistics, include_visualizations, include_metadata,
                    report_all_groups if has_groups else "One Consolidated Report",
                    selected_groups
                )
                
                # Create download link
                create_download_link(pdf_bytes, f"{report_filename}.pdf", "Download PDF Report")
            
            elif report_format == "Excel Report":
                # Create Excel report
                excel_bytes = create_excel_report(
                    filtered_price, filtered_leadtime, filtered_historical,
                    date_col, price_col, leadtime_col, group_columns,
                    include_price, include_leadtime, include_data_overview,
                    include_statistics, include_metadata,
                    report_all_groups if has_groups else "One Consolidated Report",
                    selected_groups
                )
                
                # Create download link
                create_download_link(excel_bytes, f"{report_filename}.xlsx", "Download Excel Report")
            
            elif report_format == "HTML":
                # Create HTML report
                html_bytes = create_html_report(
                    filtered_price, filtered_leadtime, filtered_historical,
                    date_col, price_col, leadtime_col, group_columns,
                    include_price, include_leadtime, include_data_overview,
                    include_statistics, include_visualizations, include_metadata,
                    report_all_groups if has_groups else "One Consolidated Report",
                    selected_groups
                )
                
                # Create download link
                create_download_link(html_bytes, f"{report_filename}.html", "Download HTML Report")
            
            # Show success message
            set_success(f"Report generated successfully! Click the download link to save.")
            
        except Exception as e:
            set_error(f"Error generating report: {str(e)}")
            raise e


def create_time_series_viz(historical_data, forecast_data, date_col, value_col, title, show_historical=True):
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
        template="plotly_white"
    )
    
    return fig


def export_figure(fig, format_type, width, height):
    """Export a Plotly figure to the specified format."""
    # Create a buffer to save the image
    buf = io.BytesIO()
    
    # Convert format type to kaleido format
    kaleido_format = format_type.lower()
    if kaleido_format == 'jpeg':
        kaleido_format = 'jpg'
    
    # Save figure in the specified format
    fig.write_image(buf, format=kaleido_format, width=width, height=height, scale=2)
    buf.seek(0)
    
    return buf.getvalue()


def create_pdf_report(price_data, leadtime_data, historical_data, date_col, price_col, leadtime_col, 
                     group_columns, include_price, include_leadtime, include_overview,
                     include_statistics, include_viz, include_metadata, group_option, selected_groups):
    """Create a PDF report with the specified content."""
    # Create a PDF document
    pdf = FPDF()
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=15)
    
    # Add title
    pdf.set_font("Arial", 'B', 16)
    pdf.cell(0, 10, "Forecast Report", ln=True, align='C')
    pdf.ln(5)
    
    # Add metadata if included
    if include_metadata:
        pdf.set_font("Arial", 'B', 12)
        pdf.cell(0, 10, "Report Information", ln=True)
        pdf.set_font("Arial", '', 10)
        pdf.cell(0, 8, f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", ln=True)
        pdf.cell(0, 8, f"Data points: Price - {len(price_data)}, Lead Time - {len(leadtime_data)}", ln=True)
        
        if len(group_columns) > 0:
            group_info = []
            for col in group_columns:
                if col in price_data.columns:
                    unique_count = price_data[col].nunique()
                    group_info.append(f"{col} ({unique_count} unique values)")
            
            if group_info:
                pdf.cell(0, 8, f"Group columns: {', '.join(group_info)}", ln=True)
        
        pdf.ln(5)
    
    # Add data overview if included
    if include_overview:
        pdf.add_page()
        pdf.set_font("Arial", 'B', 12)
        pdf.cell(0, 10, "Data Overview", ln=True)
        
        # Basic data info
        pdf.set_font("Arial", '', 10)
        
        if include_price:
            date_range = ""
            if pd.api.types.is_datetime64_any_dtype(price_data[date_col]):
                min_date = price_data[date_col].min()
                max_date = price_data[date_col].max()
                date_range = f"{min_date.strftime('%Y-%m-%d')} to {max_date.strftime('%Y-%m-%d')}"
            else:
                date_range = f"{price_data[date_col].min()} to {price_data[date_col].max()}"
            
            pdf.cell(0, 8, f"Price forecast date range: {date_range}", ln=True)
            pdf.cell(0, 8, f"Price forecast points: {len(price_data)}", ln=True)
            pdf.ln(3)
        
        if include_leadtime:
            date_range = ""
            if pd.api.types.is_datetime64_any_dtype(leadtime_data[date_col]):
                min_date = leadtime_data[date_col].min()
                max_date = leadtime_data[date_col].max()
                date_range = f"{min_date.strftime('%Y-%m-%d')} to {max_date.strftime('%Y-%m-%d')}"
            else:
                date_range = f"{leadtime_data[date_col].min()} to {leadtime_data[date_col].max()}"
            
            pdf.cell(0, 8, f"Lead time forecast date range: {date_range}", ln=True)
            pdf.cell(0, 8, f"Lead time forecast points: {len(leadtime_data)}", ln=True)
        
        pdf.ln(5)
    
    # Add statistics if included
    if include_statistics:
        pdf.add_page()
        pdf.set_font("Arial", 'B', 12)
        pdf.cell(0, 10, "Statistical Analysis", ln=True)
        
        if include_price:
            pdf.set_font("Arial", 'B', 10)
            pdf.cell(0, 8, "Price Forecast Statistics", ln=True)
            pdf.set_font("Arial", '', 10)
            
            # Calculate price statistics
            price_stats = {
                'Minimum': price_data[price_col].min(),
                'Maximum': price_data[price_col].max(),
                'Average': price_data[price_col].mean(),
                'Median': price_data[price_col].median(),
                'Std. Deviation': price_data[price_col].std()
            }
            
            for stat, value in price_stats.items():
                pdf.cell(0, 6, f"{stat}: {value:.2f}", ln=True)
            
            pdf.ln(5)
        
        if include_leadtime:
            pdf.set_font("Arial", 'B', 10)
            pdf.cell(0, 8, "Lead Time Forecast Statistics", ln=True)
            pdf.set_font("Arial", '', 10)
            
            # Calculate lead time statistics
            leadtime_stats = {
                'Minimum': leadtime_data[leadtime_col].min(),
                'Maximum': leadtime_data[leadtime_col].max(),
                'Average': leadtime_data[leadtime_col].mean(),
                'Median': leadtime_data[leadtime_col].median(),
                'Std. Deviation': leadtime_data[leadtime_col].std()
            }
            
            for stat, value in leadtime_stats.items():
                pdf.cell(0, 6, f"{stat}: {value:.2f}", ln=True)
        
        pdf.ln(5)
    
    # Add visualizations if included
    if include_viz:
        # Add time series visualizations
        if include_price:
            pdf.add_page()
            pdf.set_font("Arial", 'B', 12)
            pdf.cell(0, 10, "Price Forecast Visualization", ln=True)
            
            # Create price time series visualization
            fig = create_time_series_viz(
                historical_data, 
                price_data, 
                date_col, 
                price_col,
                title="Price Forecast over Time",
                show_historical=True
            )
            
            # Save to temp file
            with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as temp_file:
                fig.write_image(temp_file.name, width=800, height=600)
                pdf.image(temp_file.name, x=10, y=None, w=190)
                os.unlink(temp_file.name)
            
            pdf.ln(5)
        
        if include_leadtime:
            pdf.add_page()
            pdf.set_font("Arial", 'B', 12)
            pdf.cell(0, 10, "Lead Time Forecast Visualization", ln=True)
            
            # Create lead time time series visualization
            fig = create_time_series_viz(
                historical_data, 
                leadtime_data, 
                date_col, 
                leadtime_col,
                title="Lead Time Forecast over Time",
                show_historical=True
            )
            
            # Save to temp file
            with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as temp_file:
                fig.write_image(temp_file.name, width=800, height=600)
                pdf.image(temp_file.name, x=10, y=None, w=190)
                os.unlink(temp_file.name)
            
            pdf.ln(5)
        
        # Add combined visualization
        if include_price and include_leadtime:
            pdf.add_page()
            pdf.set_font("Arial", 'B', 12)
            pdf.cell(0, 10, "Price vs. Lead Time Relationship", ln=True)
            
            # Combine price and lead time data
            combined_data = pd.merge(
                price_data[[date_col, price_col]],
                leadtime_data[[date_col, leadtime_col]],
                on=date_col
            )
            
            # Create scatter plot
            fig = px.scatter(
                combined_data,
                x=price_col,
                y=leadtime_col,
                opacity=0.7,
                trendline="ols",
                title="Price vs. Lead Time Relationship"
            )
            
            # Save to temp file
            with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as temp_file:
                fig.write_image(temp_file.name, width=800, height=600)
                pdf.image(temp_file.name, x=10, y=None, w=190)
                os.unlink(temp_file.name)
            
            pdf.ln(5)
    
    # Save PDF to memory
    pdf_output = io.BytesIO()
    pdf_output.write(pdf.output(dest='S').encode('latin1'))
    pdf_output.seek(0)
    
    return pdf_output.getvalue()


def create_excel_report(price_data, leadtime_data, historical_data, date_col, price_col, leadtime_col,
                     group_columns, include_price, include_leadtime, include_overview,
                     include_statistics, include_metadata, group_option, selected_groups):
    """Create an Excel report with multiple sheets."""
    # Create a buffer to save the Excel file
    buf = io.BytesIO()
    
    # Create Excel writer
    with pd.ExcelWriter(buf, engine='xlsxwriter') as writer:
        # Add metadata sheet if included
        if include_metadata:
            metadata = pd.DataFrame([{
                'Creation Date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'Data Points': len(price_data) if include_price else 0,
                'Data Points (Lead Time)': len(leadtime_data) if include_leadtime else 0,
                'Selected Columns': ', '.join([date_col, price_col, leadtime_col] + group_columns),
                'Includes Price Data': 'Yes' if include_price else 'No',
                'Includes Lead Time Data': 'Yes' if include_leadtime else 'No',
                'Group Columns': ', '.join(group_columns) if group_columns else 'None',
                'Group Option': group_option
            }])
            metadata.to_excel(writer, sheet_name='Metadata', index=False)
        
        # Add data overview if included
        if include_overview:
            # Create overview data
            overview_data = []
            
            if include_price:
                date_range = ""
                if pd.api.types.is_datetime64_any_dtype(price_data[date_col]):
                    min_date = price_data[date_col].min()
                    max_date = price_data[date_col].max()
                    date_range = f"{min_date.strftime('%Y-%m-%d')} to {max_date.strftime('%Y-%m-%d')}"
                else:
                    date_range = f"{price_data[date_col].min()} to {price_data[date_col].max()}"
                
                overview_data.append({
                    'Metric': 'Price Forecast',
                    'Date Range': date_range,
                    'Data Points': len(price_data),
                    'Min Value': price_data[price_col].min(),
                    'Max Value': price_data[price_col].max(),
                    'Average Value': price_data[price_col].mean()
                })
            
            if include_leadtime:
                date_range = ""
                if pd.api.types.is_datetime64_any_dtype(leadtime_data[date_col]):
                    min_date = leadtime_data[date_col].min()
                    max_date = leadtime_data[date_col].max()
                    date_range = f"{min_date.strftime('%Y-%m-%d')} to {max_date.strftime('%Y-%m-%d')}"
                else:
                    date_range = f"{leadtime_data[date_col].min()} to {leadtime_data[date_col].max()}"
                
                overview_data.append({
                    'Metric': 'Lead Time Forecast',
                    'Date Range': date_range,
                    'Data Points': len(leadtime_data),
                    'Min Value': leadtime_data[leadtime_col].min(),
                    'Max Value': leadtime_data[leadtime_col].max(),
                    'Average Value': leadtime_data[leadtime_col].mean()
                })
            
            if overview_data:
                overview_df = pd.DataFrame(overview_data)
                overview_df.to_excel(writer, sheet_name='Data Overview', index=False)
        
        # Add forecast data sheets
        if include_price:
            price_data.to_excel(writer, sheet_name='Price Forecast', index=False)
        
        if include_leadtime:
            leadtime_data.to_excel(writer, sheet_name='Lead Time Forecast', index=False)
        
        # Add statistics if included
        if include_statistics:
            stats_data = []
            
            if include_price:
                price_stats = {
                    'Metric': 'Price',
                    'Min': price_data[price_col].min(),
                    'Max': price_data[price_col].max(),
                    'Mean': price_data[price_col].mean(),
                    'Median': price_data[price_col].median(),
                    'Std Dev': price_data[price_col].std(),
                    'Range': price_data[price_col].max() - price_data[price_col].min(),
                    'Count': len(price_data)
                }
                stats_data.append(price_stats)
            
            if include_leadtime:
                leadtime_stats = {
                    'Metric': 'Lead Time',
                    'Min': leadtime_data[leadtime_col].min(),
                    'Max': leadtime_data[leadtime_col].max(),
                    'Mean': leadtime_data[leadtime_col].mean(),
                    'Median': leadtime_data[leadtime_col].median(),
                    'Std Dev': leadtime_data[leadtime_col].std(),
                    'Range': leadtime_data[leadtime_col].max() - leadtime_data[leadtime_col].min(),
                    'Count': len(leadtime_data)
                }
                stats_data.append(leadtime_stats)
            
            if stats_data:
                stats_df = pd.DataFrame(stats_data)
                stats_df.to_excel(writer, sheet_name='Statistics', index=False)
                
                # Add group statistics if available
                if group_columns and (group_option == "One Consolidated Report" or group_option == "All Groups"):
                    for group_col in group_columns:
                        # Create group statistics dataframes
                        group_stats_price = None
                        group_stats_leadtime = None
                        
                        if include_price:
                            group_stats_price = price_data.groupby(group_col)[price_col].agg([
                                ('Min', 'min'),
                                ('Max', 'max'),
                                ('Mean', 'mean'),
                                ('Median', 'median'),
                                ('Std Dev', 'std'),
                                ('Count', 'count')
                            ]).reset_index()
                        
                        if include_leadtime:
                            group_stats_leadtime = leadtime_data.groupby(group_col)[leadtime_col].agg([
                                ('Min', 'min'),
                                ('Max', 'max'),
                                ('Mean', 'mean'),
                                ('Median', 'median'),
                                ('Std Dev', 'std'),
                                ('Count', 'count')
                            ]).reset_index()
                        
                        # Write to sheets
                        if group_stats_price is not None:
                            group_stats_price.to_excel(writer, sheet_name=f'Price by {group_col}', index=False)
                        
                        if group_stats_leadtime is not None:
                            group_stats_leadtime.to_excel(writer, sheet_name=f'Lead Time by {group_col}', index=False)
        
        # Add combined data if both price and lead time are included
        if include_price and include_leadtime:
            # Combine price and lead time data
            combined_data = pd.merge(
                price_data[[date_col, price_col] + group_columns],
                leadtime_data[[date_col, leadtime_col]],
                on=date_col
            )
            
            combined_data.to_excel(writer, sheet_name='Combined Data', index=False)
    
    # Get the output
    buf.seek(0)
    return buf.getvalue()


def create_html_report(price_data, leadtime_data, historical_data, date_col, price_col, leadtime_col,
                     group_columns, include_price, include_leadtime, include_overview,
                     include_statistics, include_visualizations, include_metadata,
                     group_option, selected_groups):
    """Create an HTML report with interactive visualizations."""
    # Start HTML content
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Forecast Report</title>
        <meta charset="utf-8">
        <meta name="viewport" content="width=device-width, initial-scale=1">
        <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 20px; }}
            h1, h2, h3 {{ color: #2c3e50; }}
            .section {{ margin-bottom: 30px; }}
            table {{ border-collapse: collapse; width: 100%; margin-bottom: 20px; }}
            th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
            th {{ background-color: #f2f2f2; }}
            tr:nth-child(even) {{ background-color: #f9f9f9; }}
            .chart-container {{ height: 500px; margin-bottom: 30px; }}
            .header {{ background-color: #2c3e50; color: white; padding: 20px; margin-bottom: 20px; }}
            .footer {{ background-color: #f2f2f2; padding: 15px; text-align: center; margin-top: 30px; }}
        </style>
    </head>
    <body>
        <div class="header">
            <h1>Forecast Report</h1>
            <p>Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        </div>
    """
    
    # Add metadata if included
    if include_metadata:
        html_content += """
        <div class="section">
            <h2>Report Information</h2>
            <table>
                <tr><th>Parameter</th><th>Value</th></tr>
        """
        
        html_content += f"<tr><td>Generated on</td><td>{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</td></tr>"
        html_content += f"<tr><td>Price Data Points</td><td>{len(price_data) if include_price else 'Not included'}</td></tr>"
        html_content += f"<tr><td>Lead Time Data Points</td><td>{len(leadtime_data) if include_leadtime else 'Not included'}</td></tr>"
        
        if group_columns:
            html_content += f"<tr><td>Group Columns</td><td>{', '.join(group_columns)}</td></tr>"
            html_content += f"<tr><td>Group Option</td><td>{group_option}</td></tr>"
        
        html_content += """
            </table>
        </div>
        """
    
    # Add data overview if included
    if include_overview:
        html_content += """
        <div class="section">
            <h2>Data Overview</h2>
        """
        
        if include_price:
            date_range = ""
            if pd.api.types.is_datetime64_any_dtype(price_data[date_col]):
                min_date = price_data[date_col].min()
                max_date = price_data[date_col].max()
                date_range = f"{min_date.strftime('%Y-%m-%d')} to {max_date.strftime('%Y-%m-%d')}"
            else:
                date_range = f"{price_data[date_col].min()} to {price_data[date_col].max()}"
            
            html_content += """
            <h3>Price Forecast Overview</h3>
            <table>
                <tr><th>Metric</th><th>Value</th></tr>
            """
            
            html_content += f"<tr><td>Date Range</td><td>{date_range}</td></tr>"
            html_content += f"<tr><td>Data Points</td><td>{len(price_data)}</td></tr>"
            html_content += f"<tr><td>Minimum Price</td><td>{price_data[price_col].min():.2f}</td></tr>"
            html_content += f"<tr><td>Maximum Price</td><td>{price_data[price_col].max():.2f}</td></tr>"
            html_content += f"<tr><td>Average Price</td><td>{price_data[price_col].mean():.2f}</td></tr>"
            
            html_content += """
            </table>
            """
        
        if include_leadtime:
            date_range = ""
            if pd.api.types.is_datetime64_any_dtype(leadtime_data[date_col]):
                min_date = leadtime_data[date_col].min()
                max_date = leadtime_data[date_col].max()
                date_range = f"{min_date.strftime('%Y-%m-%d')} to {max_date.strftime('%Y-%m-%d')}"
            else:
                date_range = f"{leadtime_data[date_col].min()} to {leadtime_data[date_col].max()}"
            
            html_content += """
            <h3>Lead Time Forecast Overview</h3>
            <table>
                <tr><th>Metric</th><th>Value</th></tr>
            """
            
            html_content += f"<tr><td>Date Range</td><td>{date_range}</td></tr>"
            html_content += f"<tr><td>Data Points</td><td>{len(leadtime_data)}</td></tr>"
            html_content += f"<tr><td>Minimum Lead Time</td><td>{leadtime_data[leadtime_col].min():.2f}</td></tr>"
            html_content += f"<tr><td>Maximum Lead Time</td><td>{leadtime_data[leadtime_col].max():.2f}</td></tr>"
            html_content += f"<tr><td>Average Lead Time</td><td>{leadtime_data[leadtime_col].mean():.2f}</td></tr>"
            
            html_content += """
            </table>
            """
        
        html_content += """
        </div>
        """
    
    # Add statistics if included
    if include_statistics:
        html_content += """
        <div class="section">
            <h2>Statistical Analysis</h2>
        """
        
        if include_price:
            # Calculate price statistics
            price_stats = {
                'Minimum': price_data[price_col].min(),
                'Maximum': price_data[price_col].max(),
                'Average': price_data[price_col].mean(),
                'Median': price_data[price_col].median(),
                'Std. Deviation': price_data[price_col].std(),
                'Range': price_data[price_col].max() - price_data[price_col].min()
            }
            
            html_content += """
            <h3>Price Forecast Statistics</h3>
            <table>
                <tr><th>Metric</th><th>Value</th></tr>
            """
            
            for stat, value in price_stats.items():
                html_content += f"<tr><td>{stat}</td><td>{value:.2f}</td></tr>"
            
            html_content += """
            </table>
            """
            
            # Add group statistics if available
            if group_columns and (group_option == "One Consolidated Report" or group_option == "All Groups"):
                for group_col in group_columns:
                    # Create group statistics
                    group_stats = price_data.groupby(group_col)[price_col].agg([
                        ('Minimum', 'min'),
                        ('Maximum', 'max'),
                        ('Average', 'mean'),
                        ('Median', 'median'),
                        ('Std. Deviation', 'std')
                    ]).reset_index()
                    
                    html_content += f"""
                    <h3>Price Statistics by {group_col}</h3>
                    <table>
                        <tr>
                            <th>{group_col}</th>
                            <th>Minimum</th>
                            <th>Maximum</th>
                            <th>Average</th>
                            <th>Median</th>
                            <th>Std. Deviation</th>
                        </tr>
                    """
                    
                    for _, row in group_stats.iterrows():
                        html_content += f"""
                        <tr>
                            <td>{row[group_col]}</td>
                            <td>{row['Minimum']:.2f}</td>
                            <td>{row['Maximum']:.2f}</td>
                            <td>{row['Average']:.2f}</td>
                            <td>{row['Median']:.2f}</td>
                            <td>{row['Std. Deviation']:.2f}</td>
                        </tr>
                        """
                    
                    html_content += """
                    </table>
                    """
        
        if include_leadtime:
            # Calculate lead time statistics
            leadtime_stats = {
                'Minimum': leadtime_data[leadtime_col].min(),
                'Maximum': leadtime_data[leadtime_col].max(),
                'Average': leadtime_data[leadtime_col].mean(),
                'Median': leadtime_data[leadtime_col].median(),
                'Std. Deviation': leadtime_data[leadtime_col].std(),
                'Range': leadtime_data[leadtime_col].max() - leadtime_data[leadtime_col].min()
            }
            
            html_content += """
            <h3>Lead Time Forecast Statistics</h3>
            <table>
                <tr><th>Metric</th><th>Value</th></tr>
            """
            
            for stat, value in leadtime_stats.items():
                html_content += f"<tr><td>{stat}</td><td>{value:.2f}</td></tr>"
            
            html_content += """
            </table>
            """
            
            # Add group statistics if available
            if group_columns and (group_option == "One Consolidated Report" or group_option == "All Groups"):
                for group_col in group_columns:
                    # Create group statistics
                    group_stats = leadtime_data.groupby(group_col)[leadtime_col].agg([
                        ('Minimum', 'min'),
                        ('Maximum', 'max'),
                        ('Average', 'mean'),
                        ('Median', 'median'),
                        ('Std. Deviation', 'std')
                    ]).reset_index()
                    
                    html_content += f"""
                    <h3>Lead Time Statistics by {group_col}</h3>
                    <table>
                        <tr>
                            <th>{group_col}</th>
                            <th>Minimum</th>
                            <th>Maximum</th>
                            <th>Average</th>
                            <th>Median</th>
                            <th>Std. Deviation</th>
                        </tr>
                    """
                    
                    for _, row in group_stats.iterrows():
                        html_content += f"""
                        <tr>
                            <td>{row[group_col]}</td>
                            <td>{row['Minimum']:.2f}</td>
                            <td>{row['Maximum']:.2f}</td>
                            <td>{row['Average']:.2f}</td>
                            <td>{row['Median']:.2f}</td>
                            <td>{row['Std. Deviation']:.2f}</td>
                        </tr>
                        """
                    
                    html_content += """
                    </table>
                    """
        
        html_content += """
        </div>
        """
    
    # Add visualizations if included
    if include_visualizations:
        html_content += """
        <div class="section">
            <h2>Visualizations</h2>
        """
        
        # Add time series visualizations
        if include_price:
            # Create price time series visualization
            fig = create_time_series_viz(
                historical_data,
                price_data,
                date_col,
                price_col,
                title="Price Forecast over Time",
                show_historical=True
            )
            
            # Convert to HTML
            html_content += """
            <h3>Price Forecast Visualization</h3>
            <div class="chart-container" id="price-chart"></div>
            <script>
                var priceData =
            """
            
            html_content += fig.to_json()
            
            html_content += """
                ;
                Plotly.newPlot('price-chart', priceData.data, priceData.layout);
            </script>
            """
        
        if include_leadtime:
            # Create lead time time series visualization
            fig = create_time_series_viz(
                historical_data,
                leadtime_data,
                date_col,
                leadtime_col,
                title="Lead Time Forecast over Time",
                show_historical=True
            )
            
            # Convert to HTML
            html_content += """
            <h3>Lead Time Forecast Visualization</h3>
            <div class="chart-container" id="leadtime-chart"></div>
            <script>
                var leadtimeData =
            """
            
            html_content += fig.to_json()
            
            html_content += """
                ;
                Plotly.newPlot('leadtime-chart', leadtimeData.data, leadtimeData.layout);
            </script>
            """
        
        # Add combined visualization
        if include_price and include_leadtime:
            # Combine price and lead time data
            combined_data = pd.merge(
                price_data[[date_col, price_col]],
                leadtime_data[[date_col, leadtime_col]],
                on=date_col
            )
            
            # Create scatter plot
            fig = px.scatter(
                combined_data,
                x=price_col,
                y=leadtime_col,
                opacity=0.7,
                trendline="ols",
                title="Price vs. Lead Time Relationship"
            )
            
            # Convert to HTML
            html_content += """
            <h3>Price vs. Lead Time Relationship</h3>
            <div class="chart-container" id="combined-chart"></div>
            <script>
                var combinedData =
            """
            
            html_content += fig.to_json()
            
            html_content += """
                ;
                Plotly.newPlot('combined-chart', combinedData.data, combinedData.layout);
            </script>
            """
        
        html_content += """
        </div>
        """
    
    # Add forecasted data tables
    html_content += """
    <div class="section">
        <h2>Forecast Data</h2>
    """
    
    if include_price:
        html_content += """
        <h3>Price Forecast Data (Sample)</h3>
        <table>
            <tr>
        """
        
        # Add headers
        for col in price_data.columns:
            html_content += f"<th>{col}</th>"
        
        html_content += "</tr>"
        
        # Add rows (limit to 10 for performance)
        for _, row in price_data.head(10).iterrows():
            html_content += "<tr>"
            for col in price_data.columns:
                if pd.api.types.is_datetime64_any_dtype(row[col]):
                    html_content += f"<td>{row[col].strftime('%Y-%m-%d')}</td>"
                elif pd.api.types.is_numeric_dtype(type(row[col])):
                    html_content += f"<td>{row[col]:.2f}</td>"
                else:
                    html_content += f"<td>{row[col]}</td>"
            html_content += "</tr>"
        
        html_content += """
        </table>
        """
    
    if include_leadtime:
        html_content += """
        <h3>Lead Time Forecast Data (Sample)</h3>
        <table>
            <tr>
        """
        
        # Add headers
        for col in leadtime_data.columns:
            html_content += f"<th>{col}</th>"
        
        html_content += "</tr>"
        
        # Add rows (limit to 10 for performance)
        for _, row in leadtime_data.head(10).iterrows():
            html_content += "<tr>"
            for col in leadtime_data.columns:
                if pd.api.types.is_datetime64_any_dtype(row[col]):
                    html_content += f"<td>{row[col].strftime('%Y-%m-%d')}</td>"
                elif pd.api.types.is_numeric_dtype(type(row[col])):
                    html_content += f"<td>{row[col]:.2f}</td>"
                else:
                    html_content += f"<td>{row[col]}</td>"
            html_content += "</tr>"
        
        html_content += """
        </table>
        """
    
    html_content += """
    </div>
    """
    
    # Close HTML
    html_content += """
        <div class="footer">
            <p>Generated by Price & Lead Time Forecasting Application</p>
        </div>
    </body>
    </html>
    """
    
    # Convert to bytes
    return html_content.encode('utf-8')


def create_download_link(content, filename, link_text):
    """Create a download link for the given content."""
    # Convert content to base64
    if isinstance(content, bytes):
        b64 = base64.b64encode(content).decode()
    else:
        b64 = base64.b64encode(content.encode('utf-8')).decode()
    
    # Determine MIME type
    mime_type = 'application/octet-stream'
    if filename.endswith('.csv'):
        mime_type = 'text/csv'
    elif filename.endswith('.xlsx'):
        mime_type = 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
    elif filename.endswith('.json'):
        mime_type = 'application/json'
    elif filename.endswith('.html'):
        mime_type = 'text/html'
    elif filename.endswith('.pdf'):
        mime_type = 'application/pdf'
    elif filename.endswith('.png'):
        mime_type = 'image/png'
    elif filename.endswith('.jpg') or filename.endswith('.jpeg'):
        mime_type = 'image/jpeg'
    elif filename.endswith('.svg'):
        mime_type = 'image/svg+xml'
    
    # Create href
    href = f'<a href="data:{mime_type};base64,{b64}" download="{filename}" target="_blank">{link_text}</a>'
    
    # Display link
    st.markdown(href, unsafe_allow_html=True)


def get_resolution_dimensions(resolution_str):
    """Get width and height from resolution string."""
    if resolution_str == "Low (800x600)":
        return 800, 600
    elif resolution_str == "Medium (1200x800)":
        return 1200, 800
    elif resolution_str == "High (1600x1000)":
        return 1600, 1000
    elif resolution_str == "Very High (2000x1200)":
        return 2000, 1200
    else:
        # Default to medium
        return 1200, 800