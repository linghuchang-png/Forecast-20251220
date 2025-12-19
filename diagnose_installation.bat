@echo off
REM Diagnostic script for installation issues
REM This script identifies which packages are causing pyproject.toml errors

echo ======================================================
echo   Price & Lead Time Forecasting - Installation Diagnosis
echo ======================================================
echo.
echo This script will diagnose which packages are causing installation issues.
echo It will create detailed logs to help identify the problematic packages.
echo.
pause

REM Create logs directory if it doesn't exist
if not exist "logs\" mkdir logs

echo Capturing Python and pip version information...
python --version > logs\python_version.log 2>&1
pip --version > logs\pip_version.log 2>&1
echo.

echo ======================================================
echo   Testing individual package installations
echo ======================================================
echo.

echo Testing basic dependencies...
pip install --upgrade pip setuptools wheel build > logs\base_deps.log 2>&1
echo Base dependencies test completed. [logs\base_deps.log]

echo.
echo Testing core packages individually...

echo Testing streamlit...
pip install streamlit==1.25.0 > logs\streamlit_install.log 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo [FAILED] Streamlit installation failed. See logs\streamlit_install.log
) else (
    echo [SUCCESS] Streamlit installed successfully.
)

echo Testing pandas...
pip install pandas==2.0.3 > logs\pandas_install.log 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo [FAILED] Pandas installation failed. See logs\pandas_install.log
) else (
    echo [SUCCESS] Pandas installed successfully.
)

echo Testing numpy...
pip install numpy==1.24.3 > logs\numpy_install.log 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo [FAILED] Numpy installation failed. See logs\numpy_install.log
) else (
    echo [SUCCESS] Numpy installed successfully.
)

echo Testing plotly...
pip install plotly==5.14.1 > logs\plotly_install.log 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo [FAILED] Plotly installation failed. See logs\plotly_install.log
) else (
    echo [SUCCESS] Plotly installed successfully.
)

echo Testing scikit-learn...
pip install scikit-learn==1.3.0 > logs\sklearn_install.log 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo [FAILED] Scikit-learn installation failed. See logs\sklearn_install.log
) else (
    echo [SUCCESS] Scikit-learn installed successfully.
)

echo Testing ortools...
pip install ortools==9.6.2534 > logs\ortools_install.log 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo [FAILED] OR-Tools installation failed. See logs\ortools_install.log
    echo Testing OR-Tools with no-build-isolation...
    pip install ortools==9.6.2534 --no-build-isolation > logs\ortools_no_build_isolation.log 2>&1
    if %ERRORLEVEL% NEQ 0 (
        echo [FAILED] OR-Tools installation with --no-build-isolation failed.
    ) else (
        echo [SUCCESS] OR-Tools installed successfully with --no-build-isolation.
    )
) else (
    echo [SUCCESS] OR-Tools installed successfully.
)

echo Testing kaleido...
pip install kaleido==0.2.1 > logs\kaleido_install.log 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo [FAILED] Kaleido installation failed. See logs\kaleido_install.log
    echo Testing kaleido with no-build-isolation...
    pip install kaleido==0.2.1 --no-build-isolation > logs\kaleido_no_build_isolation.log 2>&1
    if %ERRORLEVEL% NEQ 0 (
        echo [FAILED] Kaleido installation with --no-build-isolation failed.
    ) else (
        echo [SUCCESS] Kaleido installed successfully with --no-build-isolation.
    )
) else (
    echo [SUCCESS] Kaleido installed successfully.
)

echo.
echo ======================================================
echo   Capturing dependency information
echo ======================================================
echo.

echo Generating pip dependency tree (if pip-dep-tree is available)...
pip install pipdeptree > logs\pipdeptree_install.log 2>&1
pipdeptree > logs\dependency_tree.log 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo Could not generate dependency tree. pipdeptree is not installed.
) else (
    echo Dependency tree generated. See logs\dependency_tree.log
)

echo.
echo ======================================================
echo   Diagnosis Summary
echo ======================================================
echo.
echo Diagnosis complete! Log files have been created in the 'logs' directory.

echo.
echo Generating visual diagnostic report...
python generate_diagnostic_report.py
if %ERRORLEVEL% NEQ 0 (
    echo.
    echo Could not generate visual report. Python may not be available.
    echo.
    echo The following logs may help identify installation issues:
    echo - logs\python_version.log - Python version information
    echo - logs\pip_version.log - Pip version information
    echo - logs\*_install.log - Individual package installation logs
    echo - logs\dependency_tree.log - Package dependency tree (if available)
) else (
    echo.
    echo Visual report generated successfully!
    echo Opening report in browser...
    start logs\installation_report.html
)

echo.
echo Most common issues:
echo 1. OR-Tools and Kaleido packages often have pyproject.toml build issues
echo 2. Some packages require Microsoft Visual C++ Build Tools
echo 3. Certain package versions may be incompatible with your Python version
echo.
echo Recommended next steps:
echo 1. Review the diagnostic report for detailed information
echo 2. Run fix_installation.bat to apply targeted fixes
echo 3. If problems persist, try using a different Python version (3.9 or 3.10)
echo.

pause