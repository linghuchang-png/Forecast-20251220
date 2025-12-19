@echo off
REM Fix script for pyproject.toml installation issues
REM This script specifically targets packages that commonly cause pyproject.toml errors

echo ======================================================
echo   Price & Lead Time Forecasting - Installation Fix
echo ======================================================
echo.
echo This script will help resolve "pyproject.toml" errors during installation.
echo.
echo WARNING: Make sure you've already run either setup.bat or install_deps.bat
echo before running this fix script.
echo.
pause

REM Create virtual environment if it doesn't exist
if not exist "venv\" (
    echo Virtual environment not found. Creating one now...
    python -m venv venv
    if %ERRORLEVEL% NEQ 0 (
        echo Error: Failed to create virtual environment.
        echo Please ensure Python is properly installed.
        pause
        exit /b 1
    )
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
echo ======================================================
echo   Fixing Common pyproject.toml Installation Issues
echo ======================================================
echo.

echo Step 1: Ensuring base build requirements are installed...
echo.

echo Fixing 'Cannot import setuptools.build_meta' error...
echo Uninstalling setuptools (if present)...
pip uninstall -y setuptools

echo Installing specific setuptools version (59.6.0) known to work...
pip install setuptools==59.6.0

echo Verifying setuptools.build_meta is accessible...
python -c "import setuptools.build_meta; print('setuptools.build_meta successfully imported')" 2>nul
if %ERRORLEVEL% NEQ 0 (
    echo WARNING: setuptools.build_meta still cannot be imported.
    echo Attempting alternative setuptools installation...
    pip uninstall -y setuptools
    pip install setuptools==58.1.0
    python -c "import setuptools.build_meta; print('setuptools.build_meta now successfully imported')" 2>nul
    if %ERRORLEVEL% NEQ 0 (
        echo WARNING: Still cannot import setuptools.build_meta.
        echo This may affect building packages from source.
    )
)

echo Installing other build tools...
pip install --upgrade pip wheel build

echo.
echo Step 2: Installing potential problematic packages with special flags...

echo.
echo Trying kaleido installation with no-build-isolation...
pip install kaleido --no-build-isolation
if %ERRORLEVEL% NEQ 0 (
    echo Trying alternative kaleido installation method...
    pip install kaleido==0.1.0 --no-deps
    if %ERRORLEVEL% NEQ 0 (
        echo Kaleido installation failed. This is non-critical.
        echo Image export to PNG/JPEG may be limited. PDF exports will still work.
    )
)

echo.
echo Trying ortools installation with no-build-isolation...
pip install ortools==9.6.2534 --no-build-isolation
if %ERRORLEVEL% NEQ 0 (
    echo Trying alternative ortools installation...
    pip install ortools==9.6.2534 --no-deps
    if %ERRORLEVEL% NEQ 0 (
        echo WARNING: OR-Tools installation failed. 
        echo Optimization functionality may be limited.
    )
)

echo.
echo Checking scikit-learn installation...
python -c "import sklearn" >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo Re-installing scikit-learn with compatible dependencies...
    pip install scikit-learn==1.3.0 --no-build-isolation
)

echo.
echo Step 3: Verifying critical packages...

echo Testing pandas import...
python -c "import pandas" >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo Pandas not properly installed. Attempting to fix...
    pip install pandas==2.0.3 --no-build-isolation
)

echo Testing numpy import...
python -c "import numpy" >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo Numpy not properly installed. Attempting to fix...
    pip install numpy==1.24.3 --no-build-isolation
)

echo Testing streamlit import...
python -c "import streamlit" >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo Streamlit not properly installed. Attempting to fix...
    pip install streamlit==1.25.0 --no-build-isolation
)

echo.
echo ======================================================
echo   Installation checks complete!
echo ======================================================
echo.
echo If you still experience issues:
echo 1. Try installing packages individually with: 
echo    pip install package-name --no-build-isolation
echo 2. Consider using a different Python version (3.9 is recommended)
echo 3. Check the TROUBLESHOOTING.md file for more information
echo.
echo To run the application, use the run.bat script.
echo.

pause