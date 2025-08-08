"""
Data collection module for commodity price forecasting.
"""
import logging
import time
import requests
import pandas as pd
from typing import List, Dict, Optional
from datetime import datetime, timedelta
import os
from config import config

logger = logging.getLogger(__name__)

class DataCollector:
    """Handles data collection from the API with proper error handling."""
    
    def __init__(self):
        self.api_key = config.data.api_key
        self.base_url = config.data.api_base_url
        self.limit = config.data.api_limit
        
        if not self.api_key:
            raise ValueError("API_KEY environment variable is required")
    
    def _make_request(self, params: Dict, max_retries: int = 3) -> Optional[Dict]:
        """Make API request with retry logic."""
        for attempt in range(max_retries):
            try:
                response = requests.get(
                    self.base_url, 
                    params=params, 
                    timeout=30
                )
                response.raise_for_status()
                return response.json()
            
            except requests.exceptions.RequestException as e:
                logger.warning(f"API request failed (attempt {attempt + 1}/{max_retries}): {e}")
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)  # Exponential backoff
                else:
                    logger.error(f"API request failed after {max_retries} attempts")
                    return None
    
    def collect_data(
        self, 
        commodity: str = None, 
        district: str = None, 
        state: str = None,
        max_iterations: int = 5
    ) -> pd.DataFrame:
        """
        Collect data from the API with proper pagination.
        
        Args:
            commodity: Commodity name to filter
            district: District name to filter
            state: State name to filter
            max_iterations: Maximum number of API calls to prevent infinite loops
            
        Returns:
            DataFrame with collected data
        """
        commodity = commodity or config.data.commodity
        district = district or config.data.district
        state = state or config.data.state
        
        logger.info(f"Starting data collection for {commodity} in {district}, {state}")
        
        all_data = []
        offset = 0
        iteration = 0
        
        while iteration < max_iterations:
            params = {
                "api-key": self.api_key,
                "format": "json",
                "limit": self.limit,
                "offset": offset,
                "filters[Commodity]": commodity,
                "filters[District]": district,
                "filters[State]": state
            }
            
            response_data = self._make_request(params)
            
            if not response_data or "records" not in response_data:
                logger.warning(f"No valid response received at offset {offset}")
                break
            
            records = response_data["records"]
            
            if not records:
                logger.info("No more records to fetch")
                break
            
            all_data.extend(records)
            logger.info(f"Collected {len(all_data)} records so far")
            
            offset += self.limit
            iteration += 1
            
            # Small delay to be respectful to the API
            time.sleep(0.1)
        
        if not all_data:
            logger.error("No data collected")
            return pd.DataFrame()
        
        df = pd.DataFrame(all_data)
        logger.info(f"Successfully collected {len(df)} records")
        
        return df
    
    def save_data(self, df: pd.DataFrame, filename: str = None) -> str:
        """Save collected data to file."""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"commodity_data_{timestamp}.csv"
        
        filepath = os.path.join(config.data.data_path, filename)
        df.to_csv(filepath, index=False)
        logger.info(f"Data saved to {filepath}")
        
        return filepath
    
    def load_data(self, filename: str) -> pd.DataFrame:
        """Load data from file."""
        filepath = os.path.join(config.data.data_path, filename)
        
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Data file not found: {filepath}")
        
        df = pd.read_csv(filepath)
        logger.info(f"Loaded {len(df)} records from {filepath}")
        
        return df
    
    def preprocess_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Preprocess the collected data."""
        logger.info("Starting data preprocessing")
        
        # Convert date column
        if 'Arrival_Date' in df.columns:
            df['Arrival_Date'] = pd.to_datetime(df['Arrival_Date'])
        
        # Remove duplicates
        initial_count = len(df)
        df = df.drop_duplicates()
        logger.info(f"Removed {initial_count - len(df)} duplicate records")
        
        # Sort by date
        df = df.sort_values('Arrival_Date')
        
        # Handle missing values
        missing_counts = df.isnull().sum()
        if missing_counts.sum() > 0:
            logger.warning(f"Missing values found: {missing_counts[missing_counts > 0]}")
        
        # Filter out records with missing prices
        df = df.dropna(subset=['Modal_Price'])
        
        logger.info(f"Preprocessing complete. Final dataset has {len(df)} records")
        
        return df


def main():
    """Main function to run data collector directly."""
    import argparse
    import logging
    
    # Set up logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    parser = argparse.ArgumentParser(description='Collect commodity price data')
    parser.add_argument('--commodity', default='Groundnut', help='Commodity name (default: Groundnut)')
    parser.add_argument('--district', default='Mumbai', help='District name (default: Mumbai)')
    parser.add_argument('--state', default='Maharashtra', help='State name (default: Maharashtra)')
    parser.add_argument('--output', help='Output filename (optional)')
    parser.add_argument('--preprocess', action='store_true', help='Preprocess the data after collection')
    
    args = parser.parse_args()
    
    try:
        # Initialize data collector
        print("Initializing data collector...")
        collector = DataCollector()
        
        # Collect data
        print(f"Collecting data for {args.commodity} in {args.district}, {args.state}...")
        df = collector.collect_data(
            commodity=args.commodity,
            district=args.district,
            state=args.state
        )
        
        if df.empty:
            print("No data collected!")
            return
        
        print(f"Collected {len(df)} records")
        
        # Preprocess if requested
        if args.preprocess:
            print("Preprocessing data...")
            df = collector.preprocess_data(df)
            print(f"After preprocessing: {len(df)} records")
        
        # Save data
        filename = args.output or f"{args.commodity.lower()}_data.csv"
        filepath = collector.save_data(df, filename)
        print(f"Data saved to: {filepath}")
        
        # Show sample
        print("\nSample data:")
        print(df.head())
        
    except Exception as e:
        print(f"Error: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main()) 