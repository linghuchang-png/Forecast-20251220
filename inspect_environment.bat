@echo off
REM Environment Inspector Launcher
REM This script runs the Python environment inspector tool

echo ======================================================
echo   Price & Lead Time Forecasting - Environment Inspector
echo ======================================================
echo.
echo This tool will analyze your Python environment and generate a visual report
echo showing installed packages, dependencies, and potential conflicts.
echo.
echo This is especially useful for diagnosing which packages might be causing
echo installation or dependency issues.
echo.
pause

REM Check if virtual environment exists and activate it if it does
if exist "venv\" (
    echo Activating virtual environment...
    call venv\Scripts\activate.bat
    if %ERRORLEVEL% NEQ 0 (
        echo [WARNING] Could not activate virtual environment.
        echo Will use system Python instead.
    ) else (
        echo Using virtual environment Python.
    )
) else (
    echo No virtual environment found, using system Python.
)

echo.
echo Running Python Environment Inspector...
echo.

python venv_inspector.py

echo.
echo If the report did not open automatically, you can find it at:
echo logs\environment_report.html
echo.

pause