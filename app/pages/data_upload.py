"""
Data Upload page for the forecasting application.

This module handles file upload and sample data generation functionality.
"""
import os
import sys
import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import io

# Add parent directory to path for importing
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

# Import utilities
from utils.data_processing.loader import DataLoader
from utils.data_processing.validator import DataValidator
from app.state.session import set_error, set_success, clear_messages


def show_data_upload_page():
    """Display the data upload page."""
    st.header("Data Upload")
    
    # Clear any previous messages
    clear_messages()
    
    # Create tabs for different upload options
    upload_tabs = st.tabs(["Upload File", "Generate Sample Data", "Data Configuration"])
    
    # Upload File tab
    with upload_tabs[0]:
        show_file_upload()
    
    # Generate Sample Data tab
    with upload_tabs[1]:
        show_sample_generation()
    
    # Data Configuration tab
    with upload_tabs[2]:
        show_data_configuration()
    
    # Show data preview if data is loaded
    if st.session_state.data is not None:
        st.header("Data Preview")
        st.dataframe(st.session_state.data.head(10), use_container_width=True)
        
        # Show data info
        with st.expander("Data Information", expanded=False):
            # Get basic info
            data = st.session_state.data
            info = {
                'Rows': len(data),
                'Columns': len(data.columns),
                'Column Names': ', '.join(data.columns.tolist()),
                'Missing Values': data.isna().sum().sum(),
            }
            
            # Find date column if exists
            date_cols = [col for col in data.columns if 'date' in col.lower()]
            if date_cols:
                date_col = date_cols[0]
                if pd.api.types.is_datetime64_any_dtype(data[date_col]):
                    info['Date Range'] = f"{data[date_col].min().strftime('%Y-%m-%d')} to {data[date_col].max().strftime('%Y-%m-%d')}"
                else:
                    info['Date Range'] = "Date column not in datetime format"
            else:
                info['Date Range'] = "No date column found"
            
            # Display info
            for key, value in info.items():
                st.write(f"**{key}:** {value}")
            
            # Display column types
            st.write("**Column Types:**")
            st.write(data.dtypes)


def show_file_upload():
    """Show file upload interface."""
    st.subheader("Upload Data File")
    
    st.write("""
    Upload your historical data file in CSV or Excel format. 
    The file should contain date, price, and lead time columns, as well as 
    optional grouping columns such as part number, vendor, and country.
    """)
    
    # File uploader
    uploaded_file = st.file_uploader(
        "Choose a file",
        type=["csv", "xlsx", "xls"],
        help="Upload CSV or Excel file"
    )
    
    # Process the uploaded file
    if uploaded_file is not None:
        try:
            # Create DataLoader instance
            loader = DataLoader()
            
            # Determine file type from extension
            file_extension = uploaded_file.name.split('.')[-1].lower()
            
            # Load data based on file type
            if file_extension == "csv":
                # Get separator options
                separator_options = [",", ";", "\t", "|"]
                separator = st.selectbox("CSV separator:", separator_options, index=0)
                
                # Get encoding options
                encoding_options = ["utf-8", "iso-8859-1", "latin1", "cp1252"]
                encoding = st.selectbox("File encoding:", encoding_options, index=0)
                
                # Load CSV
                data = loader.load_csv(uploaded_file, separator=separator, encoding=encoding)
            else:
                # Get sheet options
                with pd.ExcelFile(uploaded_file) as xls:
                    sheet_names = xls.sheet_names
                
                # Let user select a sheet
                selected_sheet = st.selectbox("Select Excel sheet:", sheet_names, index=0)
                
                # Load Excel
                data = loader.load_excel(uploaded_file, sheet_name=selected_sheet)
            
            # Validate date columns
            if data is not None:
                # Look for date columns
                date_col_candidates = [col for col in data.columns if 'date' in col.lower()]
                
                if date_col_candidates:
                    # If date columns found, let user select one
                    date_col = st.selectbox(
                        "Select date column:",
                        date_col_candidates,
                        index=0,
                        help="Select the column that contains date information"
                    )
                    
                    # Convert to datetime
                    try:
                        data[date_col] = pd.to_datetime(data[date_col])
                        st.success(f"Successfully converted {date_col} to datetime format.")
                    except Exception as e:
                        st.warning(f"Could not convert {date_col} to datetime: {str(e)}")
                else:
                    st.warning("No date column detected. Please ensure your data contains a date column.")
            
            # Button to load the data
            if st.button("Load Data", type="primary"):
                if data is not None:
                    # Validate data
                    validator = DataValidator()
                    validation_result = validator.validate_data(data)
                    
                    if validation_result["valid"]:
                        # Store data in session state
                        st.session_state.data = data
                        st.session_state.data_clean = None  # Reset cleaned data
                        st.session_state.forecasts = None  # Reset forecasts
                        
                        # Show success message
                        set_success(f"Successfully loaded {len(data)} records from {uploaded_file.name}")
                        
                        # Provide next steps
                        st.info("Continue to Data Exploration to analyze and prepare your data for forecasting.")
                    else:
                        # Show validation errors
                        set_error(f"Data validation failed: {validation_result['message']}")
                else:
                    set_error("Failed to load data. Please check the file format and try again.")
        
        except Exception as e:
            set_error(f"Error loading file: {str(e)}")


def show_sample_generation():
    """Show sample data generation interface."""
    st.subheader("Generate Sample Data")
    
    st.write("""
    Generate sample data to test the application. This will create synthetic price and
    lead time data with realistic patterns that you can use to explore the forecasting
    capabilities of the application.
    """)
    
    # Sample data parameters
    col1, col2, col3 = st.columns(3)
    
    with col1:
        num_parts = st.number_input(
            "Number of Part Numbers:",
            min_value=1,
            max_value=50,
            value=5,
            help="Number of unique part numbers to generate"
        )
    
    with col2:
        num_vendors = st.number_input(
            "Number of Vendors:",
            min_value=1,
            max_value=20,
            value=3,
            help="Number of unique vendors to generate"
        )
    
    with col3:
        num_countries = st.number_input(
            "Number of Countries:",
            min_value=1,
            max_value=20,
            value=3,
            help="Number of unique countries to generate"
        )
    
    # Time period parameters
    col1, col2, col3 = st.columns(3)
    
    with col1:
        start_date = st.date_input(
            "Start Date:",
            value=datetime.now() - timedelta(days=365*2),
            help="Start date for the sample data"
        )
    
    with col2:
        end_date = st.date_input(
            "End Date:",
            value=datetime.now(),
            help="End date for the sample data"
        )
    
    with col3:
        frequency = st.selectbox(
            "Data Frequency:",
            options=["Daily", "Weekly", "Monthly"],
            index=2,
            help="Frequency of the time series data"
        )
    
    # Price and lead time parameters
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        min_price = st.number_input(
            "Minimum Price:",
            min_value=0.1,
            max_value=1000.0,
            value=10.0,
            help="Minimum price value"
        )
    
    with col2:
        max_price = st.number_input(
            "Maximum Price:",
            min_value=0.1,
            max_value=10000.0,
            value=100.0,
            help="Maximum price value"
        )
    
    with col3:
        min_lead_time = st.number_input(
            "Minimum Lead Time:",
            min_value=1,
            max_value=100,
            value=7,
            help="Minimum lead time value (days)"
        )
    
    with col4:
        max_lead_time = st.number_input(
            "Maximum Lead Time:",
            min_value=1,
            max_value=365,
            value=60,
            help="Maximum lead time value (days)"
        )
    
    # Pattern parameters
    col1, col2 = st.columns(2)
    
    with col1:
        include_trend = st.checkbox(
            "Include Trend",
            value=True,
            help="Include upward or downward trends in the data"
        )
    
    with col2:
        include_seasonality = st.checkbox(
            "Include Seasonality",
            value=True,
            help="Include seasonal patterns in the data"
        )
    
    # Generate button
    if st.button("Generate Sample Data", type="primary"):
        try:
            # Create DataLoader instance
            loader = DataLoader()
            
            # Convert frequency string to pandas frequency string
            freq_map = {
                "Daily": "D",
                "Weekly": "W",
                "Monthly": "MS"
            }
            pd_freq = freq_map[frequency]
            
            # Generate sample data
            data = loader.generate_sample_data(
                start_date=start_date,
                end_date=end_date,
                frequency=pd_freq,
                num_parts=num_parts,
                num_vendors=num_vendors,
                num_countries=num_countries,
                min_price=min_price,
                max_price=max_price,
                min_lead_time=min_lead_time,
                max_lead_time=max_lead_time,
                include_trend=include_trend,
                include_seasonality=include_seasonality
            )
            
            # Store data in session state
            st.session_state.data = data
            st.session_state.data_clean = None  # Reset cleaned data
            st.session_state.forecasts = None  # Reset forecasts
            
            # Show success message
            set_success(f"Successfully generated {len(data)} records of sample data")
            
            # Provide next steps
            st.info("Continue to Data Exploration to analyze and prepare your data for forecasting.")
        
        except Exception as e:
            set_error(f"Error generating sample data: {str(e)}")


def show_data_configuration():
    """Show data configuration interface."""
    st.subheader("Data Configuration")
    
    st.write("""
    Configure your data columns to ensure the application correctly identifies
    date, price, lead time, and grouping columns for forecasting.
    """)
    
    # Check if data is loaded
    if st.session_state.data is None:
        st.warning("No data loaded. Please upload a file or generate sample data first.")
        return
    
    # Get the loaded data
    data = st.session_state.data
    
    # Column configuration
    st.write("#### Column Mapping")
    
    # Date column
    date_options = [col for col in data.columns if 'date' in col.lower() or pd.api.types.is_datetime64_any_dtype(data[col])]
    if not date_options and 'date' in data.columns:
        date_options = ['date']
    
    if date_options:
        date_col = st.selectbox(
            "Date Column:",
            options=date_options,
            index=date_options.index('date') if 'date' in date_options else 0,
            help="Column containing the date values"
        )
    else:
        date_col = st.selectbox(
            "Date Column:",
            options=data.columns.tolist(),
            help="Column containing the date values"
        )
    
    # Price column
    price_options = [col for col in data.columns if 'price' in col.lower()]
    if not price_options and 'price' in data.columns:
        price_options = ['price']
    
    if price_options:
        price_col = st.selectbox(
            "Price Column:",
            options=price_options,
            index=price_options.index('price') if 'price' in price_options else 0,
            help="Column containing the price values"
        )
    else:
        # Show only numeric columns for price
        numeric_cols = data.select_dtypes(include=['number']).columns.tolist()
        price_col = st.selectbox(
            "Price Column:",
            options=numeric_cols if numeric_cols else data.columns.tolist(),
            help="Column containing the price values"
        )
    
    # Lead time column
    leadtime_options = [col for col in data.columns if 'lead' in col.lower() or 'time' in col.lower()]
    if not leadtime_options and 'lead_time' in data.columns:
        leadtime_options = ['lead_time']
    
    if leadtime_options:
        leadtime_col = st.selectbox(
            "Lead Time Column:",
            options=leadtime_options,
            index=leadtime_options.index('lead_time') if 'lead_time' in leadtime_options else 0,
            help="Column containing the lead time values"
        )
    else:
        # Show only numeric columns for lead time that are not the price column
        numeric_cols = [col for col in data.select_dtypes(include=['number']).columns.tolist() if col != price_col]
        leadtime_col = st.selectbox(
            "Lead Time Column:",
            options=numeric_cols if numeric_cols else [col for col in data.columns if col != price_col],
            help="Column containing the lead time values"
        )
    
    # Grouping columns
    st.write("#### Grouping Columns")
    
    # Suggest possible grouping columns (categorical columns)
    categorical_cols = data.select_dtypes(include=['object', 'category']).columns.tolist()
    # Add columns with low cardinality (less than 20% of total rows)
    for col in data.select_dtypes(include=['number', 'integer']).columns:
        if col not in [price_col, leadtime_col] and data[col].nunique() < len(data) * 0.2:
            categorical_cols.append(col)
    
    # Remove date column from categorical columns if it exists
    if date_col in categorical_cols:
        categorical_cols.remove(date_col)
    
    # Let user select grouping columns
    if categorical_cols:
        group_columns = st.multiselect(
            "Select Grouping Columns:",
            options=categorical_cols,
            default=[col for col in categorical_cols if col.lower() in ['part_number', 'vendor', 'country']],
            help="Columns to group by when forecasting"
        )
    else:
        st.info("No categorical columns detected for grouping.")
        group_columns = []
    
    # Apply configuration button
    if st.button("Apply Configuration", type="primary"):
        try:
            # Check if date column is in datetime format
            if not pd.api.types.is_datetime64_any_dtype(data[date_col]):
                try:
                    data[date_col] = pd.to_datetime(data[date_col])
                    st.success(f"Successfully converted {date_col} to datetime format.")
                except Exception as e:
                    st.warning(f"Could not convert {date_col} to datetime: {str(e)}")
            
            # Check if price and lead time columns are numeric
            if not pd.api.types.is_numeric_dtype(data[price_col]):
                try:
                    data[price_col] = pd.to_numeric(data[price_col])
                    st.success(f"Successfully converted {price_col} to numeric format.")
                except Exception as e:
                    st.warning(f"Could not convert {price_col} to numeric: {str(e)}")
            
            if not pd.api.types.is_numeric_dtype(data[leadtime_col]):
                try:
                    data[leadtime_col] = pd.to_numeric(data[leadtime_col])
                    st.success(f"Successfully converted {leadtime_col} to numeric format.")
                except Exception as e:
                    st.warning(f"Could not convert {leadtime_col} to numeric: {str(e)}")
            
            # Rename columns if needed
            if date_col != 'date' or price_col != 'price' or leadtime_col != 'lead_time':
                data_renamed = data.copy()
                column_map = {}
                
                if date_col != 'date':
                    column_map[date_col] = 'date'
                
                if price_col != 'price':
                    column_map[price_col] = 'price'
                
                if leadtime_col != 'lead_time':
                    column_map[leadtime_col] = 'lead_time'
                
                if column_map:
                    data_renamed = data_renamed.rename(columns=column_map)
                    
                    # Update group columns if they were renamed
                    new_group_columns = []
                    for col in group_columns:
                        if col in column_map:
                            new_group_columns.append(column_map[col])
                        else:
                            new_group_columns.append(col)
                    
                    # Update session state
                    st.session_state.data = data_renamed
                    group_columns = new_group_columns
                    
                    set_success("Successfully renamed columns to standard names.")
            
            # Update forecast parameters with selected group columns
            from app.state.session import update_forecast_params
            update_forecast_params({
                'date_col': 'date' if date_col in column_map else date_col,
                'price_col': 'price' if price_col in column_map else price_col,
                'leadtime_col': 'lead_time' if leadtime_col in column_map else leadtime_col,
                'group_columns': group_columns
            })
            
            # Show success message
            set_success("Data configuration applied successfully.")
            
            # Provide next steps
            st.info("Continue to Data Exploration to analyze and prepare your data for forecasting.")
        
        except Exception as e:
            set_error(f"Error applying configuration: {str(e)}")