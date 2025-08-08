#!/usr/bin/env python3
import pandas as pd
import argparse
import sys
import os
import shutil

# Import all models
from src.models.xgboost_model import XGBoostForecaster
from src.models.lstm_model import LSTMForecaster
from src.models.arima_model import ARIMAForecaster

def create_results_folder():
    """Create results folder if it doesn't exist."""
    results_dir = './results'
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
        print(f"Created results directory: {results_dir}")
    return results_dir

def move_existing_forecasts(results_dir):
    """Move existing forecast files to results folder."""
    forecast_files = ['lstm_forecast.csv', 'arima_forecast.csv', 'xgboost_forecast.csv']
    
    for file in forecast_files:
        if os.path.exists(file):
            destination = os.path.join(results_dir, file)
            shutil.move(file, destination)
            print(f"Moved {file} to {destination}")

def load_data():
    """Load and prepare the data."""
    df = pd.read_csv('./Groundnut_data_filtered.csv')
    df['Arrival_Date'] = pd.to_datetime(df['Arrival_Date'])
    df = df.set_index('Arrival_Date')
    return df

def run_model(model_type, df):
    """Run a specific model and return forecast."""
    if model_type == 'xgboost':
        model = XGBoostForecaster(crop_name='Groundnut')
    elif model_type == 'lstm':
        model = LSTMForecaster(crop_name='Groundnut')
    elif model_type == 'arima':
        model = ARIMAForecaster(crop_name='Groundnut')
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    print(f"Training {model_type.upper()} model...")
    trained_model = model.train(df)
    
    print(f"Making {model_type.upper()} predictions...")
    forecast = model.predict(df)
    
    return forecast

def main():
    parser = argparse.ArgumentParser(description='Run commodity price forecasting models')
    parser.add_argument('--model', choices=['xgboost', 'lstm', 'arima', 'all'], 
                       default='all', help='Model to run (default: all)')
    
    args = parser.parse_args()
    
    # Create results folder and move existing files
    results_dir = create_results_folder()
    move_existing_forecasts(results_dir)
    
    # Load data
    print("Loading data...")
    df = load_data()
    print(f"Data loaded: {len(df)} records")
    
    # Run models
    if args.model == 'all':
        models = ['xgboost', 'lstm', 'arima']
    else:
        models = [args.model]
    
    for model_type in models:
        print(f"\n{'='*50}")
        print(f"Running {model_type.upper()} model")
        print(f"{'='*50}")
        
        try:
            forecast = run_model(model_type, df)
            print(f"{model_type.upper()} Forecast:")
            print(forecast)
            
            # Save results to results folder
            filename = f'{model_type}_forecast.csv'
            filepath = os.path.join(results_dir, filename)
            forecast.to_csv(filepath, index=False)
            print(f"\nSaved to: {filepath}")
            
        except Exception as e:
            print(f"Error running {model_type.upper()}: {e}")
            continue
    
    print(f"\n{'='*50}")
    print("All models completed!")
    print(f"All results saved in: {results_dir}")

if __name__ == "__main__":
    main() 