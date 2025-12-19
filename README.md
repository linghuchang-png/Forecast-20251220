# Price & Lead Time Forecasting Tool

![Version](https://img.shields.io/badge/version-1.0.0-blue)
![Python](https://img.shields.io/badge/python-3.9+-green)
![Streamlit](https://img.shields.io/badge/streamlit-1.32.0-red)
![OR-Tools](https://img.shields.io/badge/ortools-9.8-orange)

An interactive application for forecasting product prices and delivery lead times using statistical methods with business constraint optimization.

## 📋 Overview

This application helps businesses forecast product pricing and delivery lead times using historical data. It combines statistical forecasting methods with optimization techniques to generate accurate and business-realistic forecasts without requiring machine learning expertise.

### Key Features

- 📤 **Data Import**: Load CSV/Excel files or generate sample data
- 🔍 **Data Exploration**: Analyze and clean data with interactive visualizations
- 📈 **Forecasting**: Multiple statistical algorithms with configurable parameters
- 🧮 **Optimization**: Apply business constraints to ensure realistic forecasts
- 📊 **Visualization**: Interactive charts and comparative analysis
- 📥 **Export**: Save forecasts and visualizations in various formats

## 🚀 Quick Start

### Prerequisites

- Windows 10 or later
- Python 3.9 or later (with pip)
- 4GB RAM minimum (8GB recommended)

### Installation

#### Standard Installation
1. Clone or download this repository
2. Run `create_directories.bat` to set up the directory structure
3. Run `setup.bat` to create a virtual environment and install dependencies
4. Run `run.bat` to start the application

#### Alternative Installation (if you encounter pyproject.toml errors)
If you encounter errors related to "Preparing metadata (pyproject.toml)" during installation:

1. Run `create_directories.bat` to set up the directory structure
2. Run `install_deps.bat` instead of `setup.bat` (this installs dependencies incrementally)
3. Run `run.bat` to start the application

#### Fixing Installation Issues
If you're still experiencing issues after trying the alternative installation:

1. Run `fix_installation.bat` - a dedicated script for resolving pyproject.toml errors
2. This script specifically addresses common package build issues with special flags
3. Run `run.bat` to start the application after the fixes are applied

#### Python 3.14 Compatibility
If you're using Python 3.14 and encounter specific errors like `module 'pkgutil' has no attribute 'ImpImporter'`:

1. Run `python314_numpy_fix.bat` - a specialized script for Python 3.14 compatibility
2. This script applies patches to work around removed functionality in Python 3.14
3. After successful NumPy installation, run `fix_installation.bat` to complete the setup
4. Run `run.bat` to start the application

For detailed troubleshooting of installation issues, refer to the [Troubleshooting Guide](TROUBLESHOOTING.md).

The application will open in your default web browser at http://localhost:8501.

### First-Time Use

1. Navigate to the **Data Upload** page
2. Either upload your data file or generate sample data
3. Go to **Data Exploration** to analyze and clean your data
4. Proceed to **Forecasting** to configure and generate forecasts
5. Use **Visualization** to explore results visually
6. Export your forecasts using the **Export** page

## 📁 Directory Structure

```
forecast/
├── app/                    # Streamlit UI components
│   ├── main.py             # Application entry point
│   ├── state/              # Session state management
│   ├── pages/              # Page modules
│   └── visualization/      # Visualization components
├── models/                 # Forecasting models
│   ├── statistical.py      # Statistical algorithms
│   ├── optimization.py     # OR-Tools optimization
│   └── forecast_generator.py # Forecast generation
├── utils/                  # Utility modules
│   └── data_processing/    # Data handling utilities
├── data/                   # Data storage directory
│   └── sample/             # Sample data files
├── setup.bat               # Setup script
├── run.bat                 # Application launch script
├── requirements.txt        # Dependencies
├── user_guide.md           # Detailed user guide
├── TROUBLESHOOTING.md      # Troubleshooting help
└── project_summary.md      # Technical overview
```

## 📊 Sample Workflow

1. **Upload Data**: Load a CSV or Excel file with historical data containing:
   - Date column
   - Price column
   - Lead time column
   - Optional grouping columns (part number, vendor, country)

2. **Explore & Clean**: Analyze your data, identify patterns, and handle missing values

3. **Generate Forecasts**: Configure the forecasting parameters:
   - Select statistical method (exponential smoothing, moving average, seasonal naive)
   - Set forecast horizon (number of months to forecast)
   - Apply business constraints (min/max bounds, smoothness, trends)
   - Select grouping dimensions (if needed)

4. **Visualize Results**: Explore the forecast results with interactive charts:
   - Time series visualization
   - Comparative analysis across groups
   - Statistical distributions
   - Custom visualizations

5. **Export Forecasts**: Save the results in your preferred format:
   - Data formats: CSV, Excel, JSON
   - Visualization formats: PNG, JPEG, SVG, PDF
   - Reports: PDF, Excel, HTML

## 📚 Documentation

For more detailed information, refer to:

- [User Guide](user_guide.md): Comprehensive instructions for using the application
- [Troubleshooting](TROUBLESHOOTING.md): Solutions for common issues
- [Project Summary](project_summary.md): Technical overview of the application

## 🔧 Troubleshooting

If you encounter issues with installation or running the application, we provide a comprehensive suite of troubleshooting tools:

### Automated Troubleshooting

For the simplest experience, run the all-in-one troubleshooter:

```
troubleshoot_all.bat
```

This script will:
1. Diagnose installation issues
2. Generate a visual diagnostic report
3. Apply targeted fixes
4. Verify the fixes
5. Provide clear next steps

### Step-by-Step Troubleshooting

If you prefer a more controlled approach, you can use these tools individually:

1. **Diagnose Installation Issues**:
   ```
   diagnose_installation.bat
   ```
   This will test each dependency installation individually and create detailed logs.

2. **Fix Common Issues**:
   ```
   fix_installation.bat
   ```
   This applies targeted fixes for common pyproject.toml errors.

3. **Generate Visual Report**:
   A diagnostic report is automatically generated when running the diagnosis tool, but you can also manually generate it:
   ```
   python generate_diagnostic_report.py
   ```

### Common Issues

1. **pyproject.toml errors**: These are common with packages like OR-Tools and Kaleido. Use the fix_installation.bat script to resolve.

2. **Build dependencies**: Some packages may require Microsoft Visual C++ Build Tools. The troubleshooting process will identify such issues.

3. **Python version incompatibility**: This application works best with Python 3.8-3.11. Using Python 3.14 may require additional configuration.

For more detailed information, refer to the [Troubleshooting Guide](TROUBLESHOOTING.md).

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 📞 Support

For additional support:
- Review the documentation
- Check for common issues in the Troubleshooting Guide
- Contact technical support with a description of your issue

---

**Developed for forecasting price and lead time data to facilitate effective business planning and decision-making.**