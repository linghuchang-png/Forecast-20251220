# Price & Lead Time Forecasting - Troubleshooting Guide

This document provides solutions for common issues encountered during installation and usage of the forecasting application, with special focus on pyproject.toml errors.

## Table of Contents
1. [Installation Issues](#installation-issues)
2. [pyproject.toml Errors](#pyprojecttoml-errors)
3. [Package Dependencies Issues](#package-dependencies-issues)
4. [Application Startup Problems](#application-startup-problems)
5. [Data Loading Issues](#data-loading-issues)

## Installation Issues

### Common Installation Error Messages

#### Error: 'pip' is not recognized as an internal or external command

**Cause**: Python's Scripts directory is not in your system PATH.

**Solution**:
1. Use the absolute path to call pip:
   ```
   C:\Path\To\Python\python.exe -m pip install streamlit
   ```
2. Or add Python's Scripts directory to your PATH:
   - Find your Python installation directory, e.g., `C:\Python310`
   - Add `C:\Python310\Scripts` to your system PATH environment variable

### Python Version Compatibility

Our application has been tested on:
- **Optimal compatibility**: Python 3.8 - 3.11
- **Usable with potential issues**: Python 3.12 - 3.14
- **Not supported**: Python 3.7 and earlier

If you're experiencing issues with Python 3.14, we recommend:
1. Try our compatibility solutions in the troubleshooting scripts
2. Or consider installing Python 3.10, which has the best compatibility

## pyproject.toml Errors

### Error: "Preparing metadata (pyproject.toml) ... error"

This is a common error when installing packages that require build dependencies.

**Sample Error Message**:
```
Preparing metadata (pyproject.toml) ... error
error: subprocess-exited-with-error
```

### Error: "Cannot import 'setuptools.build_meta'"

This specific error is a common manifestation of pyproject.toml issues:

**Sample Error Message**:
```
pip._vendor.pyproject_hooks._impl.BackendUnavailable: Cannot import 'setuptools.build_meta'
```

**What This Means**:
This error indicates that pip is unable to find or import the setuptools.build_meta module, which is required to build packages from source using pyproject.toml. This often happens with:
- Incompatible setuptools versions
- Setuptools installation corruption
- Conflicts between different versions of build tools

**Specific Solution for 'setuptools.build_meta' Error**:

1. Run our fix script which includes specific handling for this error:
   ```
   fix_installation.bat
   ```

2. Or manually fix with these steps:
   
   a. Uninstall current setuptools:
   ```
   pip uninstall -y setuptools
   ```
   
   b. Install a compatible setuptools version:
   ```
   pip install setuptools==59.6.0
   ```
   
   c. If that doesn't resolve the issue, try an alternative version:
   ```
   pip uninstall -y setuptools
   pip install setuptools==58.1.0
   ```
   
   d. Verify the fix worked by running:
   ```
   python -c "import setuptools.build_meta; print('setuptools.build_meta successfully imported')"
   ```

**Causes**:
1. Missing build dependencies
2. Conflicts between build systems
3. Incompatible package versions
4. Local build environment issues
5. Pip version incompatibilities

### Recommended Troubleshooting Steps

#### Step 1: Diagnose the Issue
Run our diagnostic script to identify exactly which packages are causing the problem:
```
diagnose_installation.bat
```
This script:
- Tests each package installation individually
- Creates detailed logs in the `logs` directory
- Identifies specifically which packages are failing
- Attempts alternative installation methods automatically
- Provides a summary of issues found

#### Step 2: Apply Targeted Fixes
After identifying the problematic packages, run our fix script:
```
fix_installation.bat
```
This script:
- Installs essential build tools
- Handles problematic packages with special flags
- Verifies critical package installations
- Provides alternative installation methods for packages

#### Step 3: Alternative Installation Approaches

If the automated scripts don't resolve the issue:

1. Install Core Build Tools First
```
python -m pip install --upgrade pip wheel setuptools build
```

2. Try Alternative Installation Methods
For specific problematic packages:
```
pip install package-name --no-build-isolation
```
Or:
```
pip install package-name==specific-version --no-deps
```

3. Install Dependencies Incrementally
Instead of installing all dependencies at once, install them one by one:
```
python -m pip install streamlit
python -m pip install pandas
python -m pip install numpy
# etc.
```

4. Use `install_deps.bat` Instead of `setup.bat`
The `install_deps.bat` script is designed to install dependencies incrementally with appropriate flags.

#### Step 4: Inspect Python Environment
If you're still experiencing issues, use the environment inspector tool to get a visual analysis of your Python environment:

```
inspect_environment.bat
```

This tool:
- Analyzes your Python environment and installed packages
- Identifies dependency conflicts between packages
- Checks if key application packages are properly installed
- Generates a comprehensive visual HTML report
- Helps pinpoint which specific package conflicts might be causing problems

#### Step 5: Check Log Files
After running the diagnostic script:
- Review logs in the `logs` directory
- Check which specific packages failed installation
- Note any error patterns that might indicate system-level issues

## Package Dependencies Issues

### Python 3.14 Specific Issues

#### AttributeError: module 'pkgutil' has no attribute 'ImpImporter'

This error occurs specifically in Python 3.14 when building packages from source:

**Sample Error Message**:
```
AttributeError: module 'pkgutil' has no attribute 'ImpImporter'. Did you mean: 'zipimporter'?
```

**What This Means**:
Python 3.14 removed the deprecated `pkgutil.ImpImporter`, but some packages still rely on it during their build process. This commonly happens when:

- The package uses setuptools to build from source
- The package has compiled C extensions
- The package's build process uses pkg_resources or setuptools internals
- The code relies on the old import system rather than importlib

**Automated Solution**:

We've created dedicated tools to address this specific Python 3.14 issue:

1. Run the specialized NumPy fix script:
   ```
   python314_numpy_fix.bat
   ```

   This script:
   - Applies a compatibility patch for pkgutil.ImpImporter
   - Tries multiple installation methods automatically
   - Provides a fallback to manual wheel installation if needed
   - Verifies the installation worked correctly

**Manual Solutions**:

If the automated script doesn't work, you can try these approaches in order:

1. Use pre-built wheels:
   ```
   pip install numpy --only-binary=numpy
   ```
   This avoids building from source entirely.

2. Install a specific version with no-build-isolation:
   ```
   pip install numpy==1.26.0 --no-build-isolation
   ```

3. Apply the patch manually and install:
   ```
   python -c "import pkgutil_compat_patch; pkgutil_compat_patch.apply_numpy_build_patch()" && pip install numpy
   ```

4. Install with both no-build-isolation and no-deps:
   ```
   pip install numpy --no-build-isolation --no-deps
   ```

5. Download a pre-built wheel directly from PyPI and install it:
   ```
   pip install path/to/downloaded/numpy-1.26.0-cp314-cp314-win_amd64.whl
   ```

6. Consider downgrading to Python 3.11 which has better compatibility with scientific packages

**Technical Details**:

The `pkgutil_compat_patch.py` script provides compatibility by:
- Creating a minimal implementation of the removed ImpImporter class
- Patching the pkgutil module to include this class
- Providing fallback implementations of related functions
- Monkeypatching Python's import system to handle the problematic imports

This allows packages that depend on the removed functionality to build successfully without needing to modify their source code.

### Common Package Installation Problems

#### OR-Tools Installation Issues
OR-Tools can be problematic due to complex C++ dependencies.

**Solutions**:
1. Install with specific flags:
   ```
   pip install ortools==9.6.2534 --no-build-isolation
   ```
2. Use the fix_installation.bat script

#### Kaleido Installation Issues (for Image Export)
Kaleido is used for visualization exports and may fail to build.

**Solutions**:
1. Try alternative installation:
   ```
   pip install kaleido==0.1.0 --no-deps
   ```
2. Note that PDF exports will still work even without Kaleido

## Application Startup Problems

If the application fails to start:

1. Verify Streamlit is working:
   ```
   python -m streamlit hello
   ```

2. Check if all dependencies are installed:
   ```
   python -c "import pandas, numpy, streamlit, plotly"
   ```

3. Review log files for error messages

4. If the browser doesn't open automatically, manually navigate to:
   ```
   http://localhost:8501
   ```

## Data Loading Issues

If data loading fails:

1. Verify CSV file encoding (UTF-8 recommended)
2. Confirm the separator is correct (comma, semicolon, etc.)
3. Check if data format meets application requirements
4. Try using sample data to verify the application works correctly

---

If these solutions don't resolve your issues, please provide detailed error information and environment details for further assistance.