@echo off
REM Python 3.14 Compatibility Fix Script
REM This script addresses specific issues with Python 3.14, particularly the pkgutil.ImpImporter error

echo ======================================================
echo   Python 3.14 Compatibility Fix Script
echo ======================================================
echo.
echo This script addresses specific compatibility issues with Python 3.14,
echo particularly the "module 'pkgutil' has no attribute 'ImpImporter'" error.
echo.
echo It will attempt to install packages using methods that avoid this error.
echo.
pause

REM Check if virtual environment exists
if exist "venv\" (
    echo Using existing virtual environment...
    call venv\Scripts\activate.bat
    if %ERRORLEVEL% NEQ 0 (
        echo [WARNING] Failed to activate virtual environment.
        echo Continuing with system Python...
    )
) else (
    echo Creating new virtual environment...
    python -m venv venv
    call venv\Scripts\activate.bat
    if %ERRORLEVEL% NEQ 0 (
        echo [WARNING] Failed to create or activate virtual environment.
        echo Continuing with system Python...
    )
)

echo.
echo ======================================================
echo   Applying Python 3.14 Compatibility Fixes
echo ======================================================
echo.

REM First ensure we have the latest pip and build tools
echo Updating pip and build tools...
python -m pip install --upgrade pip wheel setuptools

echo.
echo Fixing numpy installation (pkgutil.ImpImporter error)...
echo.
echo Method 1: Installing numpy using pre-built binary wheels...
python -m pip install numpy --only-binary=numpy
if %ERRORLEVEL% NEQ 0 (
    echo Method 1 failed. Trying method 2...
    echo.
    
    echo Method 2: Installing specific numpy version with no build isolation...
    python -m pip install numpy==1.26.0 --no-build-isolation
    if %ERRORLEVEL% NEQ 0 (
        echo Method 2 failed. Trying method 3...
        echo.
        
        echo Method 3: Installing numpy with both no-build-isolation and no-deps...
        python -m pip install numpy==1.26.0 --no-build-isolation --no-deps
        if %ERRORLEVEL% NEQ 0 (
            echo [WARNING] All numpy installation methods failed.
            echo You may need to downgrade to Python 3.11 for full compatibility.
        ) else (
            echo Method 3 succeeded! NumPy installed with no-build-isolation and no-deps.
        )
    ) else (
        echo Method 2 succeeded! NumPy installed with no-build-isolation.
    )
) else (
    echo Method 1 succeeded! NumPy installed from binary wheels.
)

echo.
echo Checking numpy installation...
python -c "import numpy; print(f'NumPy version {numpy.__version__} successfully imported')" 2>nul
if %ERRORLEVEL% NEQ 0 (
    echo [WARNING] NumPy could not be imported despite installation attempts.
    echo This may affect application functionality.
)

echo.
echo Installing other scientific packages with compatibility settings...

echo.
echo Installing pandas with compatibility settings...
python -m pip install pandas --only-binary=pandas
if %ERRORLEVEL% NEQ 0 (
    echo Trying alternative pandas installation...
    python -m pip install pandas --no-build-isolation
    if %ERRORLEVEL% NEQ 0 (
        echo [WARNING] Pandas installation failed.
    )
)

echo.
echo Installing scikit-learn with compatibility settings...
python -m pip install scikit-learn --only-binary=scikit-learn
if %ERRORLEVEL% NEQ 0 (
    echo Trying alternative scikit-learn installation...
    python -m pip install scikit-learn --no-build-isolation
    if %ERRORLEVEL% NEQ 0 (
        echo [WARNING] Scikit-learn installation failed.
    )
)

echo.
echo Installing other required packages...
python -m pip install streamlit plotly

echo.
echo Installing OR-Tools with compatibility settings...
python -m pip install ortools --no-build-isolation
if %ERRORLEVEL% NEQ 0 (
    echo Trying alternative OR-Tools installation...
    python -m pip install ortools==9.6.2534 --no-deps
    if %ERRORLEVEL% NEQ 0 (
        echo [WARNING] OR-Tools installation failed.
    )
)

echo.
echo ======================================================
echo   Verifying Package Installation
echo ======================================================
echo.

echo Testing core packages:
echo.

echo Testing numpy...
python -c "import numpy; print('✓ Success')" 2>nul
if %ERRORLEVEL% NEQ 0 (
    echo [FAILED] NumPy import failed
)

echo Testing pandas...
python -c "import pandas; print('✓ Success')" 2>nul
if %ERRORLEVEL% NEQ 0 (
    echo [FAILED] Pandas import failed
)

echo Testing streamlit...
python -c "import streamlit; print('✓ Success')" 2>nul
if %ERRORLEVEL% NEQ 0 (
    echo [FAILED] Streamlit import failed
)

echo Testing plotly...
python -c "import plotly; print('✓ Success')" 2>nul
if %ERRORLEVEL% NEQ 0 (
    echo [FAILED] Plotly import failed
)

echo Testing ortools...
python -c "import ortools; print('✓ Success')" 2>nul
if %ERRORLEVEL% NEQ 0 (
    echo [FAILED] OR-Tools import failed
)

echo.
echo ======================================================
echo   Python 3.14 Compatibility Fix Complete
echo ======================================================
echo.
echo The script has attempted to fix Python 3.14 compatibility issues.
echo.
echo If you still experience "pkgutil.ImpImporter" errors:
echo 1. Consider downgrading to Python 3.11 for better compatibility
echo 2. Try installing specific package versions manually
echo 3. Check the TROUBLESHOOTING.md file for more information
echo.
echo Next steps:
echo 1. Run the application with: run.bat
echo 2. If issues persist, consider using inspect_environment.bat to diagnose dependency issues
echo.

pause