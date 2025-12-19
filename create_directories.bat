@echo off
REM Script to create the directory structure for the forecasting application

echo Creating directory structure for Price & Lead Time Forecasting Application...

REM Create main directories
mkdir app
mkdir app\state
mkdir app\pages
mkdir app\visualization
mkdir models
mkdir utils
mkdir utils\data_processing
mkdir data
mkdir data\sample

REM Create placeholder for sample data
echo date,part_number,vendor,country,price,lead_time,quantity > data\sample\example_data.csv
echo 2024-01-01,A12345,Vendor1,USA,125.50,14,100 >> data\sample\example_data.csv
echo 2024-01-15,A12345,Vendor1,USA,126.75,15,150 >> data\sample\example_data.csv
echo 2024-02-01,B67890,Vendor2,Germany,233.25,21,75 >> data\sample\example_data.csv

REM Create __init__.py files to make directories proper Python packages
echo. > app\__init__.py
echo. > app\state\__init__.py
echo. > app\pages\__init__.py
echo. > app\visualization\__init__.py
echo. > models\__init__.py
echo. > utils\__init__.py
echo. > utils\data_processing\__init__.py

echo Directory structure created successfully!
echo.
echo Run setup.bat to set up the Python environment.
echo.

pause