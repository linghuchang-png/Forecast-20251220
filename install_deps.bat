@echo off
REM Simplified dependency installation script for Price & Lead Time Forecasting Tool
REM This script installs dependencies incrementally to avoid pyproject.toml errors

echo ======================================================
echo   Price & Lead Time Forecasting - Simplified Setup
echo ======================================================
echo.

REM Check if Python is installed
where python >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo Error: Python is not installed or not in the PATH.
    echo Please install Python 3.9 or later from https://www.python.org/downloads/
    echo and make sure to check "Add Python to PATH" during installation.
    pause
    exit /b 1
)

REM Create virtual environment if it doesn't exist
if not exist "venv\" (
    echo Creating virtual environment...
    python -m venv venv
) else (
    echo Virtual environment already exists.
)

REM Activate virtual environment
echo Activating virtual environment...
call venv\Scripts\activate.bat

REM Check if activation succeeded
if %ERRORLEVEL% NEQ 0 (
    echo Error: Failed to activate virtual environment.
    pause
    exit /b 1
)

echo.
echo Installing base requirements...

REM Update pip, setuptools and wheel first (critical for avoiding pyproject.toml errors)
echo.
echo Step 1: Updating pip, setuptools and wheel...
python -m pip install --upgrade pip setuptools wheel

echo.
echo Step 2: Installing core data processing packages...
pip install pandas==2.0.3 numpy==1.24.3 scipy==1.10.1

echo.
echo Step 3: Installing visualization packages...
pip install plotly==5.14.1 matplotlib==3.7.2

echo.
echo Step 4: Installing statistical packages...
pip install statsmodels==0.14.0

echo.
echo Step 5: Installing optimization package...
pip install ortools==9.6.2534

echo.
echo Step 6: Installing file handling packages...
pip install openpyxl==3.1.2 xlrd==2.0.1 fpdf==1.7.2

echo.
echo Step 7: Installing Streamlit...
pip install streamlit==1.25.0

echo.
echo Step 8: Installing scikit-learn...
pip install scikit-learn==1.3.0

echo.
echo Optional: Installing kaleido for image export (may fail on some systems)...
pip install kaleido==0.2.1
if %ERRORLEVEL% NEQ 0 (
    echo Note: Kaleido installation failed. This is non-critical.
    echo Image exports to PNG/JPEG may not work, but PDF exports will still function.
    echo You can try installing it manually later with: pip install kaleido
)

echo.
echo ======================================================
echo    Installation complete! 
echo ======================================================
echo.
echo If you encountered any "pyproject.toml" errors:
echo 1. Try running "pip install [problematic-package] --no-build-isolation"
echo 2. For persistent issues, check the TROUBLESHOOTING.md file
echo.
echo To run the application, use the run.bat script.
echo.

pause