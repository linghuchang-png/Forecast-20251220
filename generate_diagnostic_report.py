"""
Diagnostic Report Generator for Installation Issues
--------------------------------------------------

This script parses the installation logs from diagnose_installation.bat
and generates an HTML report with visual indicators of installation status
and specific recommendations.
"""

import os
import re
import sys
import datetime
from pathlib import Path

# HTML template for the report
HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>Installation Diagnostic Report</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            line-height: 1.6;
            max-width: 1000px;
            margin: 0 auto;
            padding: 20px;
            color: #333;
        }
        h1, h2, h3 {
            color: #2c3e50;
        }
        .timestamp {
            color: #7f8c8d;
            font-style: italic;
            margin-bottom: 20px;
        }
        .summary {
            background-color: #f9f9f9;
            padding: 15px;
            border-radius: 5px;
            margin-bottom: 20px;
        }
        .package-grid {
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(300px, 1fr));
            gap: 15px;
            margin-bottom: 30px;
        }
        .package-card {
            border: 1px solid #ddd;
            border-radius: 5px;
            padding: 15px;
        }
        .package-card h3 {
            margin-top: 0;
            border-bottom: 1px solid #eee;
            padding-bottom: 5px;
        }
        .status {
            display: flex;
            align-items: center;
            font-weight: bold;
            margin: 10px 0;
        }
        .status-icon {
            width: 20px;
            height: 20px;
            border-radius: 50%;
            margin-right: 10px;
        }
        .success {
            background-color: #2ecc71;
        }
        .failure {
            background-color: #e74c3c;
        }
        .warning {
            background-color: #f39c12;
        }
        .recommendations {
            background-color: #e8f4f8;
            padding: 15px;
            border-radius: 5px;
            margin-top: 20px;
        }
        .next-steps {
            margin-top: 25px;
            background-color: #f0f9ff;
            padding: 15px;
            border-radius: 5px;
        }
        .next-steps ol {
            margin: 10px 0;
            padding-left: 25px;
        }
        code {
            background-color: #f8f8f8;
            padding: 2px 5px;
            border-radius: 3px;
            font-family: Consolas, monospace;
            font-size: 90%;
        }
        pre {
            background-color: #f8f8f8;
            padding: 10px;
            border-radius: 5px;
            overflow-x: auto;
            font-family: Consolas, monospace;
            font-size: 90%;
        }
    </style>
</head>
<body>
    <h1>Price & Lead Time Forecasting - Installation Diagnostic Report</h1>
    <div class="timestamp">Generated on: {timestamp}</div>
    
    <div class="summary">
        <h2>Diagnostic Summary</h2>
        <p>{summary}</p>
        <p><strong>Python Version:</strong> {python_version}</p>
        <p><strong>Pip Version:</strong> {pip_version}</p>
        <p><strong>Overall Status:</strong> {overall_status}</p>
    </div>
    
    <h2>Package Installation Status</h2>
    <div class="package-grid">
        {package_status}
    </div>
    
    <div class="recommendations">
        <h2>Recommendations</h2>
        <p>{recommendations}</p>
        <div class="next-steps">
            <h3>Next Steps</h3>
            <ol>
                {next_steps}
            </ol>
        </div>
    </div>
</body>
</html>
"""

PACKAGE_CARD_TEMPLATE = """
<div class="package-card">
    <h3>{package_name}</h3>
    <div class="status">
        <div class="status-icon {status_class}"></div>
        {status_text}
    </div>
    <p>{details}</p>
    {alternate_status}
</div>
"""

def read_log_file(file_path):
    """Read a log file and return its contents or None if the file doesn't exist."""
    try:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            return f.read()
    except FileNotFoundError:
        return None

def extract_python_version(logs_dir):
    """Extract Python version from the log file."""
    log_content = read_log_file(os.path.join(logs_dir, "python_version.log"))
    if log_content:
        match = re.search(r'Python\s+(\d+\.\d+\.\d+)', log_content)
        if match:
            return match.group(1)
    return "Unknown"

def extract_pip_version(logs_dir):
    """Extract pip version from the log file."""
    log_content = read_log_file(os.path.join(logs_dir, "pip_version.log"))
    if log_content:
        match = re.search(r'pip\s+(\d+\.\d+\.\d+)', log_content)
        if match:
            return match.group(1)
    return "Unknown"

def check_package_status(logs_dir):
    """Check the status of each package installation."""
    packages = {
        "streamlit": {"log_file": "streamlit_install.log", "friendly_name": "Streamlit"},
        "pandas": {"log_file": "pandas_install.log", "friendly_name": "Pandas"},
        "numpy": {"log_file": "numpy_install.log", "friendly_name": "Numpy"},
        "plotly": {"log_file": "plotly_install.log", "friendly_name": "Plotly"},
        "scikit-learn": {"log_file": "sklearn_install.log", "friendly_name": "Scikit-learn"},
        "ortools": {"log_file": "ortools_install.log", "friendly_name": "OR-Tools"},
        "kaleido": {"log_file": "kaleido_install.log", "friendly_name": "Kaleido"}
    }
    
    results = {}
    failed_packages = []
    
    for pkg, info in packages.items():
        log_path = os.path.join(logs_dir, info["log_file"])
        log_content = read_log_file(log_path)
        
        status = {"success": False, "details": "No log file found"}
        
        if log_content:
            if "Successfully installed" in log_content:
                status["success"] = True
                status["details"] = "Package installed successfully."
            else:
                status["success"] = False
                status["details"] = "Installation failed. "
                
                # Check for specific error patterns
                if "pyproject.toml" in log_content:
                    status["details"] += "pyproject.toml error detected."
                elif "Microsoft Visual C++" in log_content:
                    status["details"] += "Microsoft Visual C++ build tools required."
                elif "CondaError" in log_content:
                    status["details"] += "Conda environment conflict detected."
                
                # Check for alternative installation method
                alt_log_path = os.path.join(logs_dir, f"{pkg}_no_build_isolation.log")
                alt_log_content = read_log_file(alt_log_path)
                
                if alt_log_content and "Successfully installed" in alt_log_content:
                    status["alt_success"] = True
                    status["alt_details"] = "Alternative installation method succeeded."
                else:
                    status["alt_success"] = False
                    failed_packages.append(pkg)
        
        results[pkg] = status
    
    return results, failed_packages

def generate_package_cards(package_status):
    """Generate HTML for package status cards."""
    cards = []
    
    for pkg_name, pkg_info in package_status.items():
        status_class = "success" if pkg_info["success"] else "failure"
        status_text = "Successfully Installed" if pkg_info["success"] else "Installation Failed"
        
        alternate_status = ""
        if not pkg_info["success"] and pkg_info.get("alt_success"):
            alternate_status = f"""
            <div class="status">
                <div class="status-icon success"></div>
                Alternative Method: Success
            </div>
            <p>{pkg_info["alt_details"]}</p>
            """
        
        card = PACKAGE_CARD_TEMPLATE.format(
            package_name=pkg_name,
            status_class=status_class,
            status_text=status_text,
            details=pkg_info["details"],
            alternate_status=alternate_status
        )
        cards.append(card)
    
    return "".join(cards)

def generate_recommendations(failed_packages, python_version):
    """Generate recommendations based on analysis."""
    if not failed_packages:
        return "All packages installed successfully! No specific recommendations needed."
    
    recommendations = []
    
    # General recommendation
    recommendations.append(
        "Based on the diagnostic results, there are issues with installing "
        f"the following packages: {', '.join(failed_packages)}."
    )
    
    # Python version recommendation
    python_major_minor = '.'.join(python_version.split('.')[:2]) if python_version != "Unknown" else "Unknown"
    if python_major_minor not in ["3.8", "3.9", "3.10", "3.11"]:
        recommendations.append(
            f"Your Python version ({python_version}) may not be optimal for this application. "
            "We recommend using Python 3.9 or 3.10 for best compatibility."
        )
    
    # Package specific recommendations
    if "ortools" in failed_packages:
        recommendations.append(
            "OR-Tools installation failed. This is a common issue due to complex C++ dependencies. "
            "Try running fix_installation.bat which includes special handling for OR-Tools."
        )
    
    if "kaleido" in failed_packages:
        recommendations.append(
            "Kaleido installation failed. This package is used for exporting visualizations as images. "
            "The application will still work, but image export functionality may be limited."
        )
    
    # General build issues
    if any(pkg in failed_packages for pkg in ["scikit-learn", "ortools"]):
        recommendations.append(
            "Some packages require build tools. Ensure you have Microsoft Visual C++ Build Tools installed, "
            "which are required for compiling certain Python packages."
        )
    
    return " ".join(recommendations)

def generate_next_steps(failed_packages):
    """Generate next steps based on diagnostics."""
    if not failed_packages:
        return "<li>No issues detected. Proceed with using the application.</li>"
    
    steps = []
    
    steps.append("<li>Run <code>fix_installation.bat</code> to apply targeted fixes for the identified issues.</li>")
    
    if "ortools" in failed_packages:
        steps.append(
            "<li>For OR-Tools specifically, try manual installation: <br>"
            "<code>pip install ortools==9.6.2534 --no-build-isolation</code></li>"
        )
    
    if len(failed_packages) > 2:
        steps.append(
            "<li>Consider using a different Python version (3.9 or 3.10) which generally has better "
            "compatibility with the required packages.</li>"
        )
    
    steps.append("<li>After applying fixes, try running the application with <code>run.bat</code>.</li>")
    
    return "".join(steps)

def generate_report(logs_dir="logs"):
    """Generate an HTML diagnostic report."""
    if not os.path.exists(logs_dir):
        print(f"Error: Logs directory '{logs_dir}' not found.")
        print("Please run diagnose_installation.bat first.")
        return False
    
    python_version = extract_python_version(logs_dir)
    pip_version = extract_pip_version(logs_dir)
    
    package_status, failed_packages = check_package_status(logs_dir)
    package_cards_html = generate_package_cards(package_status)
    
    # Generate overall summary
    total_packages = len(package_status)
    failed_count = len(failed_packages)
    success_count = total_packages - failed_count
    
    if failed_count == 0:
        overall_status = "✅ All packages installed successfully"
        summary = f"All {total_packages} packages were successfully installed. The application should work without issues."
    elif failed_count <= 2:
        overall_status = "⚠️ Some packages failed to install"
        summary = f"{success_count} out of {total_packages} packages installed successfully. " \
                f"There are {failed_count} problematic packages that need attention."
    else:
        overall_status = "❌ Multiple installation failures"
        summary = f"Only {success_count} out of {total_packages} packages installed successfully. " \
                f"{failed_count} packages failed to install. Significant issues need to be addressed."
    
    recommendations = generate_recommendations(failed_packages, python_version)
    next_steps = generate_next_steps(failed_packages)
    
    # Generate the report
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    report_html = HTML_TEMPLATE.format(
        timestamp=timestamp,
        python_version=python_version,
        pip_version=pip_version,
        summary=summary,
        overall_status=overall_status,
        package_status=package_cards_html,
        recommendations=recommendations,
        next_steps=next_steps
    )
    
    # Save the report
    report_path = os.path.join(logs_dir, "installation_report.html")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_html)
    
    print(f"Diagnostic report generated: {os.path.abspath(report_path)}")
    return True

if __name__ == "__main__":
    logs_directory = "logs"
    if len(sys.argv) > 1:
        logs_directory = sys.argv[1]
    
    success = generate_report(logs_directory)
    
    if success:
        print("\nTo view the report, open the HTML file in your browser.")
        print("You can either:")
        print("1. Navigate to the logs directory and open installation_report.html")
        print(f"2. Open this file directly: {os.path.abspath(os.path.join(logs_directory, 'installation_report.html'))}")
    else:
        print("\nReport generation failed. Please run diagnose_installation.bat first to generate logs.")