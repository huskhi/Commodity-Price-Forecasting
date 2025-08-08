#!/usr/bin/env python3
"""
Simple test script for data collector.
Run this from the root directory: python test_data_collector.py
"""
import logging
from data_collector import DataCollector

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def test_data_collector():
    try:
        print("Testing data collector...")
        
        # Initialize collector
        collector = DataCollector()
        print("✓ DataCollector initialized successfully")
        
        # Test with a small collection
        print("Collecting sample data...")
        df = collector.collect_data(
            commodity="Groundnut",
            district="Mumbai", 
            state="Maharashtra",
            max_iterations=1  # Just one API call for testing
        )
        
        if not df.empty:
            print(f"✓ Successfully collected {len(df)} records")
            print("\nSample data:")
            print(df.head())
        else:
            print("⚠ No data collected (this might be normal if API key is not set)")
            
    except Exception as e:
        
        print(f"✗ Error: {e}")
        print("\nThis might be because:")
        print("1. API_KEY environment variable is not set")
        print("2. Network connection issues")
        print("3. API service is down")

if __name__ == "__main__":
    test_data_collector() 