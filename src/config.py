"""
Configuration settings for the commodity price forecasting system.
"""
import os
from typing import Dict, Any
from dataclasses import dataclass

# Load environment variables from .env file if it exists
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # dotenv not installed, continue without it

@dataclass
class ModelConfig:
    """Configuration for model parameters."""
    # LSTM Configuration
    lstm_lookback_days: int = 30
    lstm_forecast_days: int = 7
    lstm_max_trials: int = 2
    lstm_epochs: int = 10
    lstm_validation_split: float = 0.2
    
    # XGBoost Configuration
    xgb_input_seq_len: int = 30
    xgb_target_seq_len: int = 7
    xgb_step_size: int = 1
    xgb_lags: list = None
    
    # ARIMA Configuration
    arima_order: tuple = (1, 1, 1)
    arima_seasonal_order: tuple = (1, 1, 1, 12)
    arima_forecast_days: int = 30

@dataclass
class DataConfig:
    """Configuration for data processing."""
    data_path: str = "data/"
    models_path: str = "models/"
    logs_path: str = "logs/"
    results_path: str = "results/"
    
    # API Configuration
    api_key: str = os.getenv("API_KEY", "")
    api_base_url: str = "https://api.data.gov.in/resource/35985678-0d79-46b4-9ed6-6f13308a1d24"
    api_limit: int = 10000
    
    # Data filters
    commodity: str = "Groundnut"
    district: str = "Rajkot"
    state: str = "Gujarat"

@dataclass
class LoggingConfig:
    """Configuration for logging."""
    log_level: str = "INFO"
    log_format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    log_file: str = "logs/commodity_forecast.log"

class Config:
    """Main configuration class."""
    def __init__(self):
        self.model = ModelConfig()
        self.data = DataConfig()
        self.logging = LoggingConfig()
        
        # Create necessary directories
        self._create_directories()
    
    def _create_directories(self):
        """Create necessary directories if they don't exist."""
        directories = [
            self.data.data_path,
            self.data.models_path,
            self.data.logs_path,
            self.data.results_path
        ]
        
        for directory in directories:
            os.makedirs(directory, exist_ok=True)

# Global configuration instance
config = Config() 