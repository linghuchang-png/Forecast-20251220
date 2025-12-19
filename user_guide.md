# Price & Lead Time Forecasting Tool - User Guide

## Table of Contents
1. [Introduction](#introduction)
2. [Getting Started](#getting-started)
3. [Data Requirements](#data-requirements)
4. [Application Workflow](#application-workflow)
5. [Data Upload](#data-upload)
6. [Data Exploration & Cleaning](#data-exploration--cleaning)
7. [Forecasting Configuration](#forecasting-configuration)
8. [Analyzing Forecast Results](#analyzing-forecast-results)
9. [Visualizations](#visualizations)
10. [Exporting Results](#exporting-results)
11. [Tips & Best Practices](#tips--best-practices)
12. [Troubleshooting](#troubleshooting)

## Introduction

The Price & Lead Time Forecasting Tool helps you generate accurate forecasts for product pricing and delivery lead times using historical data. The application uses statistical forecasting methods combined with optimization techniques to produce realistic forecasts that respect business constraints.

Key features:
- Data validation and cleaning
- Interactive exploratory data analysis
- Statistical forecasting with multiple algorithms
- Business constraint optimization
- Comprehensive visualizations
- Export capabilities for forecasts and charts

## Getting Started

### System Requirements
- Windows 10 or later
- Python 3.9 or later
- 4GB RAM minimum (8GB recommended)
- 1GB free disk space

### Installation
1. Ensure Python 3.9+ is installed on your system
2. Run `setup.bat` to create a virtual environment and install dependencies
3. Run `run.bat` to start the application

The application will open in your default web browser at http://localhost:8501.

## Data Requirements

### Required Data Format
Your data must include the following columns:
- **Date**: Timestamps for each observation
- **Price**: Historical price values
- **Lead Time**: Historical lead time values in days

### Optional Columns
- **Part Number**: Product identifiers
- **Vendor**: Supplier information
- **Country**: Geographic location
- **Quantity**: Order quantities

### Sample Data Format

| date       | part_number | vendor  | country  | price  | lead_time | quantity |
|------------|-------------|---------|----------|--------|-----------|----------|
| 2024-01-01 | A12345      | Vendor1 | USA      | 125.50 | 14        | 100      |
| 2024-01-15 | A12345      | Vendor1 | USA      | 126.75 | 15        | 150      |
| 2024-02-01 | B67890      | Vendor2 | Germany  | 233.25 | 21        | 75       |

### Data Quality Considerations
For best results:
- Ensure regular time intervals (daily, weekly, monthly)
- Provide at least 12 historical data points per group
- Avoid large gaps in the time series
- Clean obvious outliers before uploading

## Application Workflow

The application follows a sequential workflow:

1. **Data Upload**: Upload your historical data file or generate sample data
2. **Data Exploration**: Analyze and clean your data
3. **Forecasting**: Configure and generate forecasts
4. **Visualization**: Explore forecast results visually
5. **Export**: Export forecasts and visualizations

Navigate between these steps using the sidebar menu.

## Data Upload

### Uploading Data Files
1. Navigate to the **Data Upload** page
2. Select the **Upload File** tab
3. Click "Browse files" and select your CSV or Excel file
4. Select the appropriate separator and encoding options for CSVs
5. For Excel files, select the sheet containing your data
6. Click "Load Data" to process the file

### Generating Sample Data
If you don't have your own data:
1. Select the **Generate Sample Data** tab
2. Configure parameters for the sample data:
   - Number of part numbers, vendors, and countries
   - Date range and data frequency
   - Min/max values for price and lead time
   - Optional trend and seasonality patterns
3. Click "Generate Sample Data"

### Data Configuration
After loading data:
1. Go to the **Data Configuration** tab
2. Map your columns to expected fields:
   - Select the date column
   - Select the price column
   - Select the lead time column
   - Select grouping columns (part_number, vendor, country, etc.)
3. Click "Apply Configuration"

## Data Exploration & Cleaning

### Data Overview
The Data Exploration page provides:
- Summary statistics for your dataset
- Distribution visualizations
- Missing value analysis
- Interactive filtering by categories

### Data Cleaning
To clean your data:
1. Select the **Data Cleaning** tab
2. Choose strategies for handling missing values:
   - For numeric columns: mean, median, zero, interpolation
   - For categorical columns: mode, most frequent value
3. Choose strategies for handling outliers:
   - Winsorizing (capping at percentiles)
   - Removal
   - IQR-based detection
4. Click "Apply Data Cleaning"

### Time Series Analysis
To analyze time patterns:
1. Select the **Time Series Analysis** tab
2. Choose the column to analyze (price or lead time)
3. Select time aggregation level (day, week, month)
4. Apply filters if needed
5. The system will display:
   - Time series trend
   - Moving averages
   - Seasonal decomposition

## Forecasting Configuration

### Basic Settings
1. Select the **Forecasting** page
2. Choose the statistical method:
   - **Exponential Smoothing**: Adaptable for data with trends and seasonality
   - **Moving Average**: Simple technique for stable data
   - **Seasonal Naive**: Effective for strongly seasonal data
3. Set the forecast horizon (number of months to forecast)
4. Confirm column mappings for date, price, and lead time
5. Select grouping columns if needed

### Advanced Settings
#### Model Parameters
Configure algorithm-specific parameters:
- **Exponential Smoothing**:
  - Alpha (level): Controls weight given to recent observations
  - Beta (trend): Controls trend responsiveness
  - Gamma (seasonal): Controls seasonal pattern changes
  - Trend and seasonality types (additive/multiplicative)

- **Moving Average**:
  - Window size: Number of periods to average

- **Seasonal Naive**:
  - Seasonal periods: Length of one seasonal cycle

#### Optimization Options
Apply business constraints to make forecasts more realistic:
- Price min/max bounds
- Smoothness factors (limiting changes between periods)
- Trend enforcement (increasing/decreasing)
- Seasonality preservation

### Generating Forecasts
Click "Generate Forecast" to create forecasts based on your configuration.

## Analyzing Forecast Results

After generating forecasts, you'll see:

### Forecast Overview
- Forecast time series charts
- Summary statistics
- Comparison with historical data

### Forecast Details
- Key metrics like minimum, maximum, average values
- Growth rates and trends
- Seasonal patterns

### Group Analysis
If you used grouping columns:
- Comparison across different groups
- Rankings by price or lead time
- Performance metrics

## Visualizations

The Visualization page offers various ways to explore your forecasts:

### Time Series Visualization
- Historical and forecast data on the same chart
- Ability to filter by groups
- Trend lines and confidence intervals

### Comparative Analysis
- Compare forecasts across different groups
- Bar charts for specific time points
- Line charts for trends over time

### Forecast Statistics
- Distribution analysis
- Box plots by group
- Correlation analysis

### Custom Visualization
- Create your own visualizations
- Mix and match different data elements
- Customize chart types and formatting

## Exporting Results

### Data Export
Export your forecast data in various formats:
1. Select the **Export** page
2. Choose **Data Export** tab
3. Select what to include (price, lead time, or both)
4. Choose the export format:
   - CSV: Simple tabular format
   - Excel: Multiple sheets with metadata
   - JSON: Structured data format
5. Apply filters if needed
6. Click "Generate Export"

### Visualization Export
Save charts as images:
1. Select the **Visualization Export** tab
2. Choose the visualization type
3. Configure the visualization
4. Select image format (PNG, JPEG, SVG, PDF)
5. Set resolution options
6. Click "Generate Visualization"

### Report Export
Create comprehensive reports:
1. Select the **Report Export** tab
2. Choose content to include
3. Select report format (PDF, Excel, HTML)
4. Configure group settings
5. Click "Generate Report"

## Tips & Best Practices

### Data Preparation
- Ensure consistent time intervals
- Clean obvious errors before uploading
- Aggregate data if too detailed (e.g., daily to monthly)
- Provide at least 2x the forecast horizon in historical data

### Forecasting Strategy
- Start with exponential smoothing for most cases
- Use moving average for stable data without trends
- Seasonal naive works well for strong seasonal patterns
- Apply optimization to ensure forecasts meet business rules

### Group Management
- Use fewer grouping dimensions for more reliable forecasts
- Ensure sufficient data points per group (12+ recommended)
- Consider hierarchical forecasting for many groups

### Performance Tips
- Filter large datasets before uploading
- Limit the number of groups in visualization
- Export complex visualizations in sections

## Troubleshooting

For common issues and their solutions, please refer to the [TROUBLESHOOTING.md](TROUBLESHOOTING.md) document included with the application.

### Common Issues
- **Data Upload Failures**: Check file format, encoding, and size
- **Forecasting Errors**: Ensure sufficient historical data and proper column mapping
- **Visualization Problems**: Try different browsers or reduce data volume
- **Export Issues**: Verify all required dependencies are installed

For additional help, please contact technical support with:
- A description of the issue
- Steps to reproduce the problem
- Any error messages
- Sample data (if possible)