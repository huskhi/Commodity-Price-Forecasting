# Commodity Price Forecasting System

A production-ready machine learning system for forecasting commodity prices using multiple models including LSTM, XGBoost, and ARIMA.

## Features

- **Multiple Models**: LSTM, XGBoost, and ARIMA forecasting models
- **Ensemble Learning**: Combines predictions from multiple models for better accuracy
- **Data Collection**: Automated data collection from government APIs
- **Production Ready**: Proper error handling, logging, and configuration management
- **CLI Interface**: Easy-to-use command-line interface
- **Visualization**: Comprehensive plotting and analysis tools
- **Model Persistence**: Save and load trained models
- **Evaluation**: Comprehensive model evaluation metrics

## Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Setup

1. Clone the repository:
```bash
git clone <repository-url>
cd commodity-price-forecasting
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Set up environment variables:
```bash
# Create a .env file
echo "API_KEY=your_api_key_here" > .env
```

## Usage

### Basic Usage

Train models and make predictions:
```bash
python main.py --train --predict --plot --models ensemble
```

### Advanced Usage

1. **Train models only**:
```bash
python main.py --train --models lstm xgboost arima
```

2. **Make predictions with existing data**:
```bash
python main.py --predict --data-file data/commodity_data.csv --models ensemble
```

3. **Evaluate model performance**:
```bash
python main.py --evaluate --models lstm xgboost arima
```

4. **Force data collection from API**:
```bash
python main.py --force-collect --predict --plot
```

5. **Use specific models**:
```bash
python main.py --models lstm xgboost --predict --plot
```

### Command Line Options

- `--data-file`: Path to existing data file
- `--force-collect`: Force data collection from API
- `--models`: Models to use (lstm, xgboost, arima, ensemble)
- `--crop-name`: Crop name for forecasting (default: Groundnut)
- `--train`: Train models
- `--predict`: Make predictions
- `--evaluate`: Evaluate models
- `--plot`: Generate plots
- `--output-dir`: Output directory for results (default: results)
- `--log-level`: Logging level (DEBUG, INFO, WARNING, ERROR)

## Project Structure

```
commodity-price-forecasting/
├── src/
│   ├── __init__.py
│   ├── config.py              # Configuration management
│   ├── data_collector.py      # Data collection and preprocessing
│   └── models/
│       ├── __init__.py
│       ├── lstm_model.py      # LSTM forecasting model
│       ├── xgboost_model.py   # XGBoost forecasting model
│       ├── arima_model.py     # ARIMA forecasting model
│       └── ensemble_model.py  # Ensemble model
├── data/                      # Data storage
├── models/                    # Trained models
├── logs/                      # Log files
├── results/                   # Output results
├── main.py                    # Main application
├── requirements.txt           # Dependencies
└── README.md                  # This file
```

## Models

### LSTM Model
- Deep learning model for time series forecasting
- Automatic hyperparameter tuning
- Sequence-based predictions
- Handles long-term dependencies

### XGBoost Model
- Gradient boosting model
- Feature engineering with lag and time features
- Robust to outliers
- Fast training and prediction

### ARIMA Model
- Statistical time series model
- Handles seasonality and trends
- Automatic parameter selection
- Model diagnostics

### Ensemble Model
- Combines predictions from multiple models
- Weighted averaging for optimal results
- Reduces overfitting
- Improves prediction accuracy

## Configuration

The system uses a centralized configuration system in `src/config.py`. Key configuration options:

- **Model Parameters**: Lookback periods, forecast horizons, hyperparameters
- **Data Settings**: API endpoints, data filters, file paths
- **Logging**: Log levels, file paths, formats

## API Configuration

To use the data collection feature, you need an API key from the government data portal. Set it as an environment variable:

```bash
export API_KEY="your_api_key_here"
```

Or create a `.env` file:
```
API_KEY=your_api_key_here
```

## Output Files

The system generates several output files:

- **Forecast CSV files**: Predictions from each model
- **Evaluation JSON**: Model performance metrics
- **Plot images**: Visualization of forecasts and comparisons
- **Log files**: Detailed execution logs

## Model Evaluation

The system provides comprehensive evaluation metrics:

- **MSE (Mean Squared Error)**: Measures prediction accuracy
- **MAE (Mean Absolute Error)**: Measures absolute prediction errors
- **MAPE (Mean Absolute Percentage Error)**: Measures relative prediction errors

## Development

### Running Tests
```bash
pytest tests/
```

### Code Formatting
```bash
black src/
flake8 src/
```

### Type Checking
```bash
mypy src/
```

## Production Deployment

### Docker Deployment
```bash
# Build Docker image
docker build -t commodity-forecast .

# Run container
docker run -e API_KEY=your_key commodity-forecast --predict --plot
```

### Web API (Future Enhancement)
The system can be extended to provide a REST API for real-time predictions.

## Troubleshooting

### Common Issues

1. **API Key Error**: Ensure your API key is set correctly
2. **Memory Issues**: Reduce batch sizes or use smaller models
3. **Training Time**: Use fewer hyperparameter trials for faster training

### Logs
Check the log files in the `logs/` directory for detailed error information.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Government of India for providing the commodity price data API
- Open source community for the machine learning libraries used

## Support

For support and questions, please open an issue on the GitHub repository. 