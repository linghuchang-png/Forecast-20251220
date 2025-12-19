"""
Main application for the forecasting tool.

This module serves as the entry point for the Streamlit application 
and handles navigation between different pages.
"""
import os
import sys
import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime

# Add parent directory to path for importing
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import pages
from app.pages.data_upload import show_data_upload_page
from app.pages.data_exploration import show_data_exploration_page
from app.pages.forecasting import show_forecasting_page
from app.pages.visualization import show_visualization_page
from app.pages.export import show_export_page

# Import utilities
from app.state.session import initialize_session_state, get_error, get_success, clear_messages

# Application title and config
st.set_page_config(
    page_title="Price & Lead Time Forecasting Tool",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)


def main():
    """Run the main application."""
    # Initialize session state if needed
    initialize_session_state()
    
    # Add custom CSS
    add_custom_css()
    
    # Create application header
    st.title("Price & Lead Time Forecasting Tool")
    st.markdown("""
    This application helps you forecast product prices and delivery lead times 
    using historical data and statistical forecasting methods.
    """)
    
    # Create sidebar navigation
    with st.sidebar:
        st.header("Navigation")
        
        # FIX: Get the current page directly (no [0] needed anymore)
        current_page = st.query_params.get('page', 'data_upload')
        
        # Navigation buttons
        pages = {
            "data_upload": "📤 Data Upload",
            "data_exploration": "🔍 Data Exploration",
            "forecasting": "📈 Forecasting",
            "visualization": "📊 Visualization",
            "export": "📥 Export Results",
            "help": "❓ Help & Documentation"
        }
        
        # Create navigation tabs
        selected_page = None
        for page_id, page_name in pages.items():
            if st.button(page_name, key=f"nav_{page_id}",
                       help=f"Navigate to {page_name} page",
                       use_container_width=True, 
                       type="primary" if page_id == current_page else "secondary"):
                selected_page = page_id
        
        # FIX: Change page if a navigation button was clicked
        if selected_page is not None and selected_page != current_page:
            st.query_params["page"] = selected_page
            st.rerun()  # Corrected from experimental_rerun earlier
        
        # Show application status
        show_application_status()
    
    # Display error/success messages if any
    if get_error():
        st.error(get_error())
    
    if get_success():
        st.success(get_success())
    
    # Show the selected page
    if current_page == "data_upload":
        show_data_upload_page()
    elif current_page == "data_exploration":
        show_data_exploration_page()
    elif current_page == "forecasting":
        show_forecasting_page()
    elif current_page == "visualization":
        show_visualization_page()
    elif current_page == "export":
        show_export_page()
    elif current_page == "help":
        show_help_page()
    else:
        show_data_upload_page()


def show_application_status():
    """Display the current application status."""
    st.sidebar.header("Application Status")
    
    # Check if data is loaded
    if st.session_state.data is not None:
        st.sidebar.success(f"✅ Data loaded: {len(st.session_state.data)} records")
    else:
        st.sidebar.warning("❌ No data loaded")
    
    # Check if data is cleaned
    if st.session_state.data_clean is not None:
        st.sidebar.success(f"✅ Data cleaned: {len(st.session_state.data_clean)} records")
    else:
        st.sidebar.info("⚠️ Data not yet cleaned")
    
    # Check if forecasts are generated
    if st.session_state.forecasts is not None:
        price_forecast = st.session_state.forecasts.get('price_forecast')
        leadtime_forecast = st.session_state.forecasts.get('leadtime_forecast')
        
        if price_forecast is not None:
            st.sidebar.success(f"✅ Price forecast generated: {len(price_forecast)} points")
        
        if leadtime_forecast is not None:
            st.sidebar.success(f"✅ Lead time forecast generated: {len(leadtime_forecast)} points")
    else:
        st.sidebar.info("⚠️ No forecasts generated")
    
    # Add a divider
    st.sidebar.markdown("---")
    
    # Show application info
    st.sidebar.markdown("### Application Info")
    st.sidebar.info(f"""
    **Version:** 1.0.0  
    **Date:** {datetime.now().strftime('%Y-%m-%d')}  
    **Libraries:** Streamlit, Pandas, NumPy, Plotly, OR-Tools
    """)


def show_help_page():
    """Display the help and documentation page."""
    st.header("Help & Documentation")
    
    # Create tabs for different help sections
    help_tabs = st.tabs([
        "Getting Started", 
        "Data Requirements", 
        "Forecasting Methods", 
        "Troubleshooting",
        "About"
    ])
    
    # Getting Started tab
    with help_tabs[0]:
        st.subheader("Getting Started")
        
        st.markdown("""
        ### Using the Application
        
        This application is designed to forecast product prices and delivery lead times based on historical data.
        The application workflow consists of the following steps:
        
        1. **Data Upload**: Upload your historical data in CSV or Excel format, or generate sample data for testing.
        2. **Data Exploration**: Explore and clean your data to prepare it for forecasting.
        3. **Forecasting**: Configure and generate forecasts using statistical methods.
        4. **Visualization**: Visualize and analyze the forecast results.
        5. **Export**: Export the forecasts and visualizations in various formats.
        
        ### Navigation
        
        Use the sidebar navigation menu to move between different sections of the application.
        Each page has specific functionality related to the forecasting workflow.
        """)
        
        st.info("""
        **Tip**: Begin by uploading your data or generating sample data on the Data Upload page.
        Then follow the workflow through Exploration, Forecasting, Visualization, and Export.
        """)
    
    # Data Requirements tab
    with help_tabs[1]:
        st.subheader("Data Requirements")
        
        st.markdown("""
        ### Required Data Format
        
        The application expects your data to have the following columns:
        
        - **Date Column**: A column containing dates in a standard format (e.g., YYYY-MM-DD).
        - **Price Column**: A numeric column containing historical prices.
        - **Lead Time Column**: A numeric column containing historical lead times.
        - **Grouping Columns (optional)**: Columns for grouping data, such as:
            - Part Number
            - Vendor
            - Country
            - Any other categorical variables
        
        ### Sample Data Format
        
        Here's an example of the expected data format:
        
        | date       | part_number | vendor | country | price | lead_time | quantity |
        |------------|------------|--------|---------|-------|-----------|----------|
        | 2023-01-01 | AB12345    | VendorA| USA     | 15.75 | 14        | 100      |
        | 2023-01-15 | AB12345    | VendorA| USA     | 16.25 | 15        | 150      |
        | 2023-02-01 | CD67890    | VendorB| Germany | 22.50 | 21        | 75       |
        | ...        | ...        | ...    | ...     | ...   | ...       | ...      |
        
        ### Data Quality
        
        For best results, ensure your data:
        - Has consistent time intervals (daily, weekly, monthly)
        - Covers a sufficient historical period (at least 1 year recommended)
        - Has minimal missing values
        - Does not contain outliers that could skew forecasts
        
        You can use the Data Exploration page to identify and fix data quality issues.
        """)
    
    # Forecasting Methods tab
    with help_tabs[2]:
        st.subheader("Forecasting Methods")
        
        st.markdown("""
        ### Available Forecasting Methods
        
        This application uses statistical forecasting methods combined with optimization techniques:
        
        #### Exponential Smoothing
        
        **Exponential smoothing** gives more weight to recent observations and less weight to older ones.
        It's suitable for data with trends and seasonality.
        
        Parameters:
        - **Alpha (Level)**: Controls how much the forecast responds to recent observations
        - **Beta (Trend)**: Controls how much the trend component responds to recent changes
        - **Gamma (Seasonal)**: Controls how much the seasonal component changes over time
        
        #### Moving Average
        
        **Moving average** uses the average of a fixed number of the most recent observations.
        It's simple and works well for stable data without strong trends or seasonality.
        
        Parameters:
        - **Window Size**: Number of periods to include in the moving average
        
        #### Seasonal Naive
        
        **Seasonal naive** forecasting uses the most recent observation from the same season.
        It's simple but effective for strongly seasonal data.
        
        Parameters:
        - **Seasonal Periods**: Number of observations in one seasonal cycle
        
        ### Optimization with OR-Tools
        
        After generating statistical forecasts, the application uses Google OR-Tools to apply business constraints:
        
        - **Min/Max Bounds**: Ensure forecasts stay within reasonable bounds
        - **Smoothness**: Prevent unrealistic jumps between periods
        - **Trend Enforcement**: Optionally enforce increasing or decreasing trends
        - **Seasonality Preservation**: Maintain seasonal patterns from the statistical forecast
        """)
    
    # Troubleshooting tab
    with help_tabs[3]:
        st.subheader("Troubleshooting")
        
        st.markdown("""
        ### Common Issues and Solutions
        
        #### Data Loading Problems
        
        **Issue**: Application fails to load data or shows errors during data upload.
        
        **Solution**:
        - Ensure your file is in CSV or Excel format
        - Check that your column names don't contain special characters
        - Verify that date columns are in a standard format (YYYY-MM-DD)
        - Make sure numeric columns contain only numbers
        
        #### Forecasting Errors
        
        **Issue**: Forecasting fails or produces unexpected results.
        
        **Solution**:
        - Check for missing values in your data (use Data Exploration page)
        - Ensure you have sufficient historical data (at least 2x the forecast horizon)
        - Try a different forecasting method
        - Adjust forecasting parameters (smoothing values, window size)
        - If using groups, check that you have enough data points per group
        
        #### Visualization Issues
        
        **Issue**: Visualizations don't display or look incorrect.
        
        **Solution**:
        - Verify that forecasts were generated successfully
        - Try refreshing the page
        - If using filters, make sure the selected filters have data
        - Check for outliers that might be skewing the scale of visualizations
        
        #### Export Problems
        
        **Issue**: Unable to export results.
        
        **Solution**:
        - Make sure you have generated forecasts before trying to export
        - Try a different export format
        - If exporting visualizations, try a lower resolution
        - For PDF exports, ensure you have selected reasonable content to include
        
        ### Still Need Help?
        
        If you continue to experience issues:
        
        1. Check the application documentation and user guide
        2. Try with a smaller dataset to isolate the problem
        3. Clear your browser cache and restart the application
        4. Contact technical support with details about your issue
        """)
    
    # About tab
    with help_tabs[4]:
        st.subheader("About")
        
        st.markdown("""
        ### Price & Lead Time Forecasting Tool
        
        **Version**: 1.0.0  
        **Release Date**: December 2025
        
        This application was developed to help businesses forecast product pricing and 
        delivery lead times based on historical data. It combines statistical forecasting
        methods with optimization techniques to generate accurate and business-realistic
        forecasts.
        
        ### Features
        
        - Data loading and preprocessing
        - Interactive data exploration and cleaning
        - Multiple statistical forecasting methods
        - Business constraint optimization
        - Interactive visualizations
        - Comprehensive export options
        
        ### Technologies Used
        
        - **Streamlit**: Web application framework
        - **Pandas & NumPy**: Data manipulation and analysis
        - **Plotly**: Interactive data visualization
        - **Google OR-Tools**: Optimization engine
        - **Scikit-learn**: Statistical modeling
        
        ### License
        
        This application is licensed under the MIT License.
        
        &copy; 2025 Price & Lead Time Forecasting Tool
        """)


def add_custom_css():
    """Add custom CSS to improve the application appearance."""
    st.markdown("""
    <style>
    /* Main container */
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    
    /* Headers */
    h1, h2, h3, h4, h5, h6 {
        color: #1E88E5;
    }
    
    /* Sidebar */
    .sidebar .sidebar-content {
        background-color: #f8f9fa;
    }
    
    /* Buttons */
    .stButton button {
        margin-bottom: 0.5rem;
    }
    
    /* Status indicators */
    .success {
        color: #28a745;
    }
    
    .warning {
        color: #ffc107;
    }
    
    .error {
        color: #dc3545;
    }
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
    }
    
    .stTabs [data-baseweb="tab"] {
        padding: 10px 16px;
        border-radius: 4px 4px 0 0;
    }
    
    /* Tables */
    .dataframe {
        font-size: 0.9rem;
    }
    
    /* Plots */
    .js-plotly-plot {
        margin-bottom: 1.5rem;
    }
    
    /* Input fields */
    .stSelectbox, .stMultiSelect {
        margin-bottom: 1rem;
    }
    
    /* Sliders */
    .stSlider {
        padding-top: 0.5rem;
        padding-bottom: 1.5rem;
    }
    </style>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()