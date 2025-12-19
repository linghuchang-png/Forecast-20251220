"""
Virtual Environment Inspector Tool
---------------------------------

This tool analyzes the Python virtual environment structure and dependencies to help
identify potential conflicts or issues that might affect the forecasting application.
It generates a visual representation of the installed packages and their dependencies.
"""

import os
import sys
import json
import subprocess
import platform
from datetime import datetime
from pathlib import Path

def get_python_info():
    """Get basic Python environment information."""
    info = {
        "python_version": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "system": platform.system(),
        "platform": platform.platform(),
        "executable": sys.executable,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    return info

def get_pip_list():
    """Get list of installed packages using pip list."""
    try:
        result = subprocess.run(
            [sys.executable, "-m", "pip", "list", "--format=json"],
            capture_output=True,
            text=True,
            check=True
        )
        packages = json.loads(result.stdout)
        return packages
    except Exception as e:
        print(f"Error getting pip list: {e}")
        return []

def get_package_dependencies(package_name):
    """Get dependencies for a specific package."""
    try:
        result = subprocess.run(
            [sys.executable, "-m", "pip", "show", package_name],
            capture_output=True,
            text=True,
            check=True
        )
        
        # Parse the output to get dependencies
        lines = result.stdout.split('\n')
        dependencies = []
        
        for line in lines:
            if line.startswith("Requires:"):
                deps = line.replace("Requires:", "").strip()
                if deps:
                    dependencies = [dep.strip() for dep in deps.split(',')]
                break
        
        return dependencies
    except Exception as e:
        print(f"Error getting dependencies for {package_name}: {e}")
        return []

def build_dependency_graph(packages):
    """Build a graph of package dependencies."""
    graph = {}
    
    for pkg in packages:
        pkg_name = pkg["name"]
        graph[pkg_name] = {
            "version": pkg["version"],
            "dependencies": get_package_dependencies(pkg_name)
        }
    
    return graph

def detect_conflicts(graph):
    """Identify potential dependency conflicts."""
    conflicts = []
    
    # This is a simplified conflict detection - in a real scenario you'd 
    # need to check version compatibility more thoroughly
    for pkg_name, pkg_data in graph.items():
        for dep in pkg_data["dependencies"]:
            if dep in graph:
                # Check if this dependency is required by multiple packages
                dependent_pkgs = [p for p, data in graph.items() 
                                  if p != dep and dep in data["dependencies"]]
                if len(dependent_pkgs) > 1:
                    conflicts.append({
                        "package": dep,
                        "version": graph[dep]["version"] if dep in graph else "unknown",
                        "required_by": dependent_pkgs
                    })
    
    # Remove duplicates
    unique_conflicts = []
    conflict_signatures = set()
    
    for conflict in conflicts:
        signature = (conflict["package"], tuple(sorted(conflict["required_by"])))
        if signature not in conflict_signatures:
            conflict_signatures.add(signature)
            unique_conflicts.append(conflict)
    
    return unique_conflicts

def check_key_packages():
    """Check if key application packages are installed."""
    key_packages = [
        "streamlit", "pandas", "numpy", "plotly", "ortools"
    ]
    
    results = {}
    for pkg in key_packages:
        try:
            __import__(pkg)
            results[pkg] = {"installed": True, "status": "OK"}
        except ImportError:
            results[pkg] = {"installed": False, "status": "Missing"}
        
    return results

def generate_html_report(info, packages, graph, conflicts, key_packages):
    """Generate an HTML report with the environment information."""
    html = f"""<!DOCTYPE html>
<html>
<head>
    <title>Python Environment Inspector Report</title>
    <style>
        body {{
            font-family: Arial, sans-serif;
            line-height: 1.6;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            color: #333;
        }}
        h1, h2, h3 {{
            color: #2c3e50;
        }}
        .timestamp {{
            color: #7f8c8d;
            font-style: italic;
            margin-bottom: 20px;
        }}
        .info-section {{
            background-color: #f8f9fa;
            padding: 15px;
            border-radius: 5px;
            margin-bottom: 20px;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin-bottom: 20px;
        }}
        th, td {{
            padding: 8px 12px;
            text-align: left;
            border-bottom: 1px solid #ddd;
        }}
        th {{
            background-color: #f2f2f2;
        }}
        tr:hover {{
            background-color: #f5f5f5;
        }}
        .status-ok {{
            color: #28a745;
            font-weight: bold;
        }}
        .status-warning {{
            color: #ffc107;
            font-weight: bold;
        }}
        .status-error {{
            color: #dc3545;
            font-weight: bold;
        }}
        .dependency-map {{
            margin: 20px 0;
            padding: 15px;
            background-color: #f0f7ff;
            border-radius: 5px;
        }}
        .main-package {{
            font-weight: bold;
        }}
        .package-card {{
            border: 1px solid #ddd;
            border-radius: 5px;
            margin-bottom: 10px;
            padding: 10px;
            background-color: #fff;
        }}
        .conflict-section {{
            background-color: #fff8f8;
            padding: 15px;
            border-radius: 5px;
            margin: 20px 0;
        }}
        .key-package-section {{
            background-color: #f0fff0;
            padding: 15px; 
            border-radius: 5px;
            margin: 20px 0;
        }}
        .footer {{
            margin-top: 30px;
            padding-top: 10px;
            border-top: 1px solid #eee;
            color: #777;
            font-size: 0.9em;
        }}
    </style>
</head>
<body>
    <h1>Python Environment Inspector Report</h1>
    <div class="timestamp">Generated on: {info["timestamp"]}</div>
    
    <div class="info-section">
        <h2>System Information</h2>
        <table>
            <tr><td><strong>Python Version</strong></td><td>{info["python_version"]}</td></tr>
            <tr><td><strong>Python Implementation</strong></td><td>{info["python_implementation"]}</td></tr>
            <tr><td><strong>System</strong></td><td>{info["system"]}</td></tr>
            <tr><td><strong>Platform</strong></td><td>{info["platform"]}</td></tr>
            <tr><td><strong>Python Executable</strong></td><td>{info["executable"]}</td></tr>
        </table>
    </div>
    
    <div class="key-package-section">
        <h2>Key Application Packages</h2>
        <table>
            <tr>
                <th>Package</th>
                <th>Status</th>
            </tr>
    """
    
    for pkg, status in key_packages.items():
        status_class = "status-ok" if status["installed"] else "status-error"
        html += f"""
            <tr>
                <td>{pkg}</td>
                <td class="{status_class}">{status["status"]}</td>
            </tr>"""
    
    html += """
        </table>
    </div>
    """
    
    if conflicts:
        html += """
    <div class="conflict-section">
        <h2>Potential Dependency Conflicts</h2>
        <table>
            <tr>
                <th>Package</th>
                <th>Version</th>
                <th>Required By</th>
            </tr>
        """
        
        for conflict in conflicts:
            html += f"""
            <tr>
                <td>{conflict["package"]}</td>
                <td>{conflict["version"]}</td>
                <td>{", ".join(conflict["required_by"])}</td>
            </tr>"""
        
        html += """
        </table>
        <p><em>Note: These are potential version conflicts that might need attention if you experience issues.</em></p>
    </div>
    """
    
    html += """
    <h2>Installed Packages</h2>
    <p>Total packages installed: """ + str(len(packages)) + """</p>
    
    <table>
        <tr>
            <th>Package</th>
            <th>Version</th>
        </tr>
    """
    
    for pkg in sorted(packages, key=lambda x: x["name"].lower()):
        html += f"""
        <tr>
            <td>{pkg["name"]}</td>
            <td>{pkg["version"]}</td>
        </tr>"""
    
    html += """
    </table>
    
    <h2>Dependency Graph</h2>
    <p>This section shows the dependencies between packages. Only packages with dependencies are shown.</p>
    <div class="dependency-map">
    """
    
    # Generate dependency graph HTML
    for pkg_name, pkg_data in sorted(graph.items()):
        if pkg_data["dependencies"]:
            html += f"""
            <div class="package-card">
                <div class="main-package">{pkg_name} ({pkg_data["version"]})</div>
                <div>Dependencies:</div>
                <ul>"""
            
            for dep in pkg_data["dependencies"]:
                dep_version = graph[dep]["version"] if dep in graph else "unknown"
                html += f"""
                    <li>{dep} ({dep_version})</li>"""
            
            html += """
                </ul>
            </div>
            """
    
    html += """
    </div>
    
    <div class="footer">
        <p>This report was generated by the Environment Inspector Tool for the Price & Lead Time Forecasting Application.</p>
    </div>
</body>
</html>
    """
    
    return html

def main():
    # Create logs directory if it doesn't exist
    logs_dir = Path("logs")
    logs_dir.mkdir(exist_ok=True)
    
    print("Gathering Python environment information...")
    python_info = get_python_info()
    
    print("Getting list of installed packages...")
    packages = get_pip_list()
    
    print("Building dependency graph...")
    graph = build_dependency_graph(packages)
    
    print("Detecting potential conflicts...")
    conflicts = detect_conflicts(graph)
    
    print("Checking key application packages...")
    key_packages = check_key_packages()
    
    print("Generating HTML report...")
    html_report = generate_html_report(python_info, packages, graph, conflicts, key_packages)
    
    # Save the report
    report_path = logs_dir / "environment_report.html"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(html_report)
    
    print(f"\nEnvironment inspection complete!")
    print(f"Report generated: {report_path.absolute()}")
    
    # Try to open the report in a browser
    try:
        import webbrowser
        webbrowser.open(str(report_path.absolute()))
        print("Report opened in your web browser.")
    except:
        print("Could not automatically open the report. Please open it manually.")

if __name__ == "__main__":
    print("=" * 60)
    print("  Python Virtual Environment Inspector")
    print("=" * 60)
    print("This tool analyzes your Python environment and identifies potential issues.")
    print("It will create a detailed HTML report for troubleshooting.")
    print("-" * 60)
    
    main()