#!/usr/bin/env python3
import argparse
import logging
from src.data_collector import DataCollector

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def main():
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