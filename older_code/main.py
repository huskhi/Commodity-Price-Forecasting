#!/usr/bin/env python3
"""
Main application for commodity price forecasting.
"""
import argparse
import logging
import sys
import os
from datetime import datetime
from typing import Optional

import pandas as pd

from src.config import config
from src.data_collector import DataCollector
from src.models.ensemble_model import EnsembleForecaster
from src.models.lstm_model import LSTMForecaster
from src.models.xgboost_model import XGBoostForecaster
from src.models.arima_model import ARIMAForecaster

def setup_logging(log_level: str = "INFO") -> None:
    """Setup logging configuration."""
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format=config.logging.log_format,
        handlers=[
            logging.FileHandler(config.logging.log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )

def load_or_collect_data(
    data_collector: DataCollector,
    data_file: Optional[str] = None,
    force_collect: bool = False
) -> pd.DataFrame:
    """Load existing data or collect new data."""
    if data_file and os.path.exists(data_file) and not force_collect:
        logger.info(f"Loading existing data from {data_file}")
        df = data_collector.load_data(data_file)
    else:
        logger.info("Collecting new data from API")
        df = data_collector.collect_data()
        
        if df.empty:
            raise ValueError("No data collected from API")
        
        # Save collected data
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"commodity_data_{timestamp}.csv"
        data_collector.save_data(df, filename)
    
    # Preprocess data
    df = data_collector.preprocess_data(df)
    
    # Set index
    if 'Arrival_Date' in df.columns:
        df['Arrival_Date'] = pd.to_datetime(df['Arrival_Date'])
        df = df.set_index('Arrival_Date')
    
    return df

def train_models(df: pd.DataFrame, models: list, crop_name: str) -> dict:
    """Train specified models."""
    results = {}
    
    if 'lstm' in models:
        logger.info("Training LSTM model")
        lstm_forecaster = LSTMForecaster(crop_name=crop_name)
        lstm_model = lstm_forecaster.train(df)
        results['lstm'] = {'status': 'success', 'model': lstm_model}
    
    if 'xgboost' in models:
        logger.info("Training XGBoost model")
        xgb_forecaster = XGBoostForecaster(crop_name=crop_name)
        xgb_model = xgb_forecaster.train(df)
        results['xgboost'] = {'status': 'success', 'model': xgb_model}
    
    if 'arima' in models:
        logger.info("Training ARIMA model")
        arima_forecaster = ARIMAForecaster(crop_name=crop_name)
        arima_model = arima_forecaster.train(df)
        results['arima'] = {'status': 'success', 'model': arima_model}
    
    return results

def make_predictions(df: pd.DataFrame, models: list, crop_name: str) -> dict:
    """Make predictions using specified models."""
    predictions = {}
    
    if 'lstm' in models:
        logger.info("Making LSTM predictions")
        lstm_forecaster = LSTMForecaster(crop_name=crop_name)
        lstm_forecast = lstm_forecaster.predict(df)
        predictions['lstm'] = lstm_forecast
    
    if 'xgboost' in models:
        logger.info("Making XGBoost predictions")
        xgb_forecaster = XGBoostForecaster(crop_name=crop_name)
        xgb_forecast = xgb_forecaster.predict(df)
        predictions['xgboost'] = xgb_forecast
    
    if 'arima' in models:
        logger.info("Making ARIMA predictions")
        arima_forecaster = ARIMAForecaster(crop_name=crop_name)
        arima_forecast = arima_forecaster.predict(df)
        predictions['arima'] = arima_forecast
    
    return predictions

def evaluate_models(df: pd.DataFrame, models: list, crop_name: str) -> dict:
    """Evaluate specified models."""
    results = {}
    
    if 'lstm' in models:
        logger.info("Evaluating LSTM model")
        lstm_forecaster = LSTMForecaster(crop_name=crop_name)
        lstm_metrics = lstm_forecaster.evaluate(df)
        results['lstm'] = lstm_metrics
    
    if 'xgboost' in models:
        logger.info("Evaluating XGBoost model")
        xgb_forecaster = XGBoostForecaster(crop_name=crop_name)
        xgb_metrics = xgb_forecaster.evaluate(df)
        results['xgboost'] = xgb_metrics
    
    if 'arima' in models:
        logger.info("Evaluating ARIMA model")
        arima_forecaster = ARIMAForecaster(crop_name=crop_name)
        arima_metrics = arima_forecaster.evaluate(df)
        results['arima'] = arima_metrics
    
    return results

def main():
    """Main function."""
    parser = argparse.ArgumentParser(description='Commodity Price Forecasting System')
    
    # Data options
    parser.add_argument('--data-file', type=str, help='Path to existing data file')
    parser.add_argument('--force-collect', action='store_true', help='Force data collection from API')
    
    # Model options
    parser.add_argument('--models', nargs='+', 
                       choices=['lstm', 'xgboost', 'arima', 'ensemble'],
                       default=['ensemble'],
                       help='Models to use (default: ensemble)')
    parser.add_argument('--crop-name', type=str, default='Groundnut',
                       help='Crop name for forecasting (default: Groundnut)')
    
    # Action options
    parser.add_argument('--train', action='store_true', help='Train models')
    parser.add_argument('--predict', action='store_true', help='Make predictions')
    parser.add_argument('--evaluate', action='store_true', help='Evaluate models')
    parser.add_argument('--plot', action='store_true', help='Generate plots')
    
    # Output options
    parser.add_argument('--output-dir', type=str, default='results',
                       help='Output directory for results')
    parser.add_argument('--log-level', type=str, default='INFO',
                       choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                       help='Logging level (default: INFO)')
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.log_level)
    global logger
    logger = logging.getLogger(__name__)
    
    logger.info("Starting Commodity Price Forecasting System")
    logger.info(f"Models: {args.models}")
    logger.info(f"Crop: {args.crop_name}")
    
    try:
        # Initialize data collector
        data_collector = DataCollector()
        
        # Load or collect data
        df = load_or_collect_data(data_collector, args.data_file, args.force_collect)
        logger.info(f"Data loaded: {len(df)} records")
        
        # Create output directory
        os.makedirs(args.output_dir, exist_ok=True)
        
        # Train models
        if args.train:
            logger.info("Training models...")
            train_results = train_models(df, args.models, args.crop_name)
            logger.info("Model training completed")
        
        # Make predictions
        if args.predict:
            logger.info("Making predictions...")
            predictions = make_predictions(df, args.models, args.crop_name)
            
            # Save predictions
            for model_name, forecast in predictions.items():
                output_file = os.path.join(args.output_dir, f"{model_name}_forecast.csv")
                forecast.to_csv(output_file, index=False)
                logger.info(f"{model_name.upper()} forecast saved to {output_file}")
            
            # Generate ensemble if requested
            if 'ensemble' in args.models:
                logger.info("Generating ensemble forecast...")
                ensemble_forecaster = EnsembleForecaster(crop_name=args.crop_name)
                ensemble_forecast = ensemble_forecaster.ensemble_predict(df)
                
                output_file = os.path.join(args.output_dir, "ensemble_forecast.csv")
                ensemble_forecast.to_csv(output_file, index=False)
                logger.info(f"Ensemble forecast saved to {output_file}")
        
        # Evaluate models
        if args.evaluate:
            logger.info("Evaluating models...")
            evaluation_results = evaluate_models(df, args.models, args.crop_name)
            
            # Save evaluation results
            import json
            output_file = os.path.join(args.output_dir, "evaluation_results.json")
            with open(output_file, 'w') as f:
                json.dump(evaluation_results, f, indent=2)
            logger.info(f"Evaluation results saved to {output_file}")
        
        # Generate plots
        if args.plot:
            logger.info("Generating plots...")
            
            if 'ensemble' in args.models:
                ensemble_forecaster = EnsembleForecaster(crop_name=args.crop_name)
                
                # Ensemble forecast plot
                plot_file = os.path.join(args.output_dir, "ensemble_forecast.png")
                ensemble_forecaster.plot_ensemble_forecast(df, save_path=plot_file)
                
                # Model comparison plot
                comparison_file = os.path.join(args.output_dir, "model_comparison.png")
                ensemble_forecaster.plot_model_comparison(df, save_path=comparison_file)
            else:
                # Individual model plots
                for model_name in args.models:
                    if model_name == 'lstm':
                        forecaster = LSTMForecaster(crop_name=args.crop_name)
                        forecast = forecaster.predict(df)
                        plot_file = os.path.join(args.output_dir, f"{model_name}_forecast.png")
                        forecaster.plot_forecast(df, forecast, save_path=plot_file)
                    
                    elif model_name == 'xgboost':
                        forecaster = XGBoostForecaster(crop_name=args.crop_name)
                        forecast = forecaster.predict(df)
                        plot_file = os.path.join(args.output_dir, f"{model_name}_forecast.png")
                        forecaster.plot_forecast(df, forecast, save_path=plot_file)
                    
                    elif model_name == 'arima':
                        forecaster = ARIMAForecaster(crop_name=args.crop_name)
                        forecast = forecaster.predict(df)
                        plot_file = os.path.join(args.output_dir, f"{model_name}_forecast.png")
                        forecaster.plot_forecast(df, forecast, save_path=plot_file)
        
        logger.info("Commodity Price Forecasting System completed successfully")
        
    except Exception as e:
        logger.error(f"Error in main execution: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 