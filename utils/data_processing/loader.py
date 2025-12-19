"""
Data loading and sample data generation utilities.

This module handles loading data from various file formats
and generating synthetic data for testing.
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random


class DataLoader:
    """Handles loading data from files and generating sample data."""
    
    def load_csv(self, file_obj, separator=",", encoding="utf-8"):
        """Load data from a CSV file.

        Args:
            file_obj: File object or path to the CSV file
            separator: Column separator character (default: ",")
            encoding: File encoding (default: "utf-8")

        Returns:
            pandas.DataFrame: Loaded data or None if loading fails
        """
        try:
            data = pd.read_csv(file_obj, sep=separator, encoding=encoding)
            return self._preprocess_data(data)
        except Exception as e:
            print(f"Error loading CSV file: {str(e)}")
            return None
    
    def load_excel(self, file_obj, sheet_name=0):
        """Load data from an Excel file.

        Args:
            file_obj: File object or path to the Excel file
            sheet_name: Sheet name or index to read (default: 0)

        Returns:
            pandas.DataFrame: Loaded data or None if loading fails
        """
        try:
            data = pd.read_excel(file_obj, sheet_name=sheet_name)
            return self._preprocess_data(data)
        except Exception as e:
            print(f"Error loading Excel file: {str(e)}")
            return None
    
    def _preprocess_data(self, data):
        """Perform basic preprocessing on loaded data.

        Args:
            data: pandas.DataFrame to preprocess

        Returns:
            pandas.DataFrame: Preprocessed data
        """
        # Convert column names to lowercase and replace spaces with underscores
        data.columns = [col.lower().replace(' ', '_') for col in data.columns]
        
        # Try to convert date columns to datetime
        for col in data.columns:
            if 'date' in col.lower():
                try:
                    data[col] = pd.to_datetime(data[col])
                except:
                    pass  # Skip if conversion fails
        
        return data
    
    def generate_sample_data(self, start_date, end_date, frequency="MS", num_parts=5,
                            num_vendors=3, num_countries=3, min_price=10.0, max_price=100.0,
                            min_lead_time=7, max_lead_time=60, include_trend=True,
                            include_seasonality=True):
        """Generate synthetic data for testing.

        Args:
            start_date: Start date for the data
            end_date: End date for the data
            frequency: Data frequency (D=daily, W=weekly, MS=monthly start)
            num_parts: Number of unique part numbers to generate
            num_vendors: Number of unique vendors to generate
            num_countries: Number of unique countries to generate
            min_price: Minimum price value
            max_price: Maximum price value
            min_lead_time: Minimum lead time value (days)
            max_lead_time: Maximum lead time value (days)
            include_trend: Whether to include trends in the data
            include_seasonality: Whether to include seasonality in the data

        Returns:
            pandas.DataFrame: Generated sample data
        """
        # Convert dates if they're string or datetime.date
        if isinstance(start_date, str):
            start_date = pd.to_datetime(start_date)
        elif not isinstance(start_date, pd.Timestamp):
            start_date = pd.to_datetime(start_date)
            
        if isinstance(end_date, str):
            end_date = pd.to_datetime(end_date)
        elif not isinstance(end_date, pd.Timestamp):
            end_date = pd.to_datetime(end_date)
        
        # Generate date range
        dates = pd.date_range(start=start_date, end=end_date, freq=frequency)
        
        # Generate part numbers, vendors, and countries
        part_numbers = [f"PART{i+1:05d}" for i in range(num_parts)]
        vendors = [f"Vendor{chr(65+i)}" for i in range(min(num_vendors, 26))]
        
        # Use realistic country names
        countries = ["USA", "China", "Germany", "Japan", "UK", "France", 
                    "Italy", "Canada", "Mexico", "Brazil", "India", 
                    "South Korea", "Spain", "Australia", "Netherlands"]
        if num_countries > len(countries):
            countries = countries + [f"Country{i+1}" for i in range(num_countries - len(countries))]
        else:
            countries = countries[:num_countries]
        
        # Generate all combinations of date, part, vendor, country
        data_records = []
        
        for part in part_numbers:
            # Each part has a different base price and lead time
            base_price = np.random.uniform(min_price, max_price)
            base_lead_time = np.random.uniform(min_lead_time, max_lead_time)
            
            # Each part might have different trend and seasonality factors
            trend_factor = np.random.uniform(0.002, 0.01) if include_trend else 0
            trend_direction = np.random.choice([-1, 1])
            seasonal_amplitude_price = np.random.uniform(0.05, 0.15) * base_price if include_seasonality else 0
            seasonal_amplitude_lead = np.random.uniform(0.05, 0.15) * base_lead_time if include_seasonality else 0
            
            # Random phase shift for seasonality
            phase_shift = np.random.uniform(0, 2*np.pi)
            
            # Each vendor might adjust prices differently
            for vendor in vendors:
                vendor_factor = np.random.uniform(0.8, 1.2)
                
                # Each country might have different lead times
                for country in countries:
                    country_factor = np.random.uniform(0.8, 1.2)
                    
                    # Generate time series data for this combination
                    for i, date in enumerate(dates):
                        # Add trend component if enabled
                        trend = trend_factor * trend_direction * i if include_trend else 0
                        
                        # Add seasonal component if enabled
                        # Using different seasonal patterns for price and lead time
                        if include_seasonality:
                            # Annual seasonality (12-month cycle)
                            month = date.month
                            seasonal_price = seasonal_amplitude_price * np.sin(2*np.pi*month/12 + phase_shift)
                            seasonal_lead = seasonal_amplitude_lead * np.sin(2*np.pi*month/12 + phase_shift + np.pi/2)
                        else:
                            seasonal_price = 0
                            seasonal_lead = 0
                        
                        # Add some random noise
                        price_noise = np.random.normal(0, base_price * 0.03)
                        lead_noise = np.random.normal(0, base_lead_time * 0.03)
                        
                        # Calculate final price and lead time values
                        price = base_price * vendor_factor * (1 + trend + seasonal_price/base_price) + price_noise
                        lead_time = base_lead_time * country_factor * (1 + trend + seasonal_lead/base_lead_time) + lead_noise
                        
                        # Ensure values are within bounds
                        price = max(min_price, min(max_price, price))
                        lead_time = max(min_lead_time, min(max_lead_time, lead_time))
                        
                        # Add some random quantity values
                        quantity = np.random.randint(50, 500)
                        
                        # Create record
                        record = {
                            'date': date,
                            'part_number': part,
                            'vendor': vendor,
                            'country': country,
                            'price': round(price, 2),
                            'lead_time': round(lead_time, 1),
                            'quantity': quantity
                        }
                        
                        data_records.append(record)
        
        # Convert to DataFrame
        df = pd.DataFrame(data_records)
        
        # Sort by date
        df = df.sort_values('date')
        
        return df