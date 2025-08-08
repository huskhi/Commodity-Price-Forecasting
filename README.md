# Commodity Price Forecasting System

A  machine learning system for forecasting commodity prices using multiple models - LSTM, XGBoost, ARIMA. 

##Input Data:
Currently, the input data is being taken from data.gov.in wesbite (https://www.data.gov.in/resource/variety-wise-daily-market-prices-data-commodity). You can generate a free API key by authenticating yourself on the website. You can add filters such as commodity, state, district (and arrival date) in the API.

#Expected Input format:
```
date,commodity,modal_price
2023-01-01,mustard,5400
2023-01-02,mustard,5412
```

## 🚀 Features

- **Multiple ML Models**: LSTM, XGBoost, and ARIMA forecasting
- **Ensemble Learning**: Combine predictions from multiple models
- **Production Ready**: Logging, configuration management, error handling
- **Data Collection**: Robust API integration with retry logic
- **Model Persistence**: Save and load trained models
- **Evaluation Metrics**: MSE, MAE, MAPE for model comparison
- **Visualization**: Plot forecasts and model comparisons
- **CLI Interface**: Easy-to-use command line tools

## 📁 Project Structure

```
├── src/
│   ├── config.py              # Configuration management
│   ├── data_collector.py      # Data collection from API
│   └── models/
│       ├── lstm_model.py      # LSTM forecaster
│       ├── xgboost_model.py   # XGBoost forecaster
│       ├── arima_model.py     # ARIMA forecaster
│       └── ensemble_model.py  # Ensemble methods
├── run_models.py              # Main execution script
├── run_data_collector.py      # Data collection script
├── requirements.txt           # Python dependencies
├── .gitignore                # Git ignore rules
└── README.md                 # This file
```

## 🛠️ Installation

1. **Clone the repository:**
   ```bash
   git clone huskhi/Commodity-Price-Forecasting
   cd commodity-price-forecasting
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up API key:**
   Create a `.env` file in the root directory:
   ```
   API_KEY=your_api_key_here
   ```

## 🚀 Usage

### Run All Models
```bash
python run_models.py
```

### Run Specific Model
```bash
python run_models.py --model xgboost
python run_models.py --model lstm
python run_models.py --model arima
```

### Collect Data
```bash
python run_data_collector.py
```

### Collect Data with Custom Parameters
```bash
python run_data_collector.py --commodity "Rice" --district "Delhi" --state "Delhi" --preprocess
```

## 📊 Models

### LSTM (Long Short-Term Memory)
- Deep learning model for time series forecasting
- Hyperparameter tuning with Keras Tuner
- Sequence-based predictions

### XGBoost
- Gradient boosting for time series
- Feature engineering with lag features
- Time-based features (day, month, weekend, etc.)

### ARIMA (AutoRegressive Integrated Moving Average)
- Statistical time series model
- Handles trend and seasonality
- Configurable order parameters

### Ensemble
- Combines predictions from all models
- Methods: weighted average, simple average, median
- Improved accuracy and robustness

## 📈 Output

- **Forecast Files**: `results/xgboost_forecast.csv`, `results/lstm_forecast.csv`, `results/arima_forecast.csv`
- **Models**: Saved in `models/` directory
- **Logs**: Detailed logging in `logs/` directory
- **Data**: Collected data in `data/` directory

## ⚙️ Configuration

Edit `src/config.py` to customize:
- Model parameters (lookback days, forecast days, etc.)
- API settings
- File paths
- Logging configuration

## 🔧 Development

### Project Structure
- **Modular Design**: Each model is a separate class
- **Configuration Management**: Centralized config using dataclasses
- **Error Handling**: Comprehensive exception handling
- **Logging**: Structured logging throughout

### Adding New Models
1. Create a new model class in `src/models/`
2. Implement `train()`, `predict()`, `evaluate()`, and `plot_forecast()` methods
3. Add to `run_models.py` and `ensemble_model.py`

## 📝 License

This project is licensed under the terms of the MIT License.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request


**Note**: This system is designed for educational and research purposes. Always validate predictions before making financial decisions. 