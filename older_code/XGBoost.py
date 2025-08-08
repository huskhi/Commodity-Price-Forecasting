import numpy as np
import pandas as pd
import logging
from typing import List, Tuple, Optional
import os
import json
from xgboost import XGBRegressor
from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV
import matplotlib.pyplot as plt
import joblib

# Set up logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class XGBoostTimeSeriesPreprocessor:
    def __init__(
        self,
        input_seq_len: int,
        target_seq_len: int,
        step_size: int = 1,
        lags: Optional[List[int]] = None,
        use_time_features: bool = True,
    ):
        self.input_seq_len = input_seq_len
        self.target_seq_len = target_seq_len
        self.step_size = step_size
        self.lags = lags if lags else []
        self.use_time_features = use_time_features

    def get_indices(self, data_len: int) -> List[Tuple[int, int]]:
        """
        Generate rolling window start and end indices.
        """
        window_size = self.input_seq_len + self.target_seq_len
        stop_position = data_len - 1
        subseq_first_idx = 0
        subseq_last_idx = window_size
        indices = []

        while subseq_last_idx <= stop_position:
            indices.append((subseq_first_idx, subseq_last_idx))
            subseq_first_idx += self.step_size
            subseq_last_idx += self.step_size

        return indices

    def prepare_features(
        self, data: pd.Series, indices: List[Tuple[int, int]]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare X and y arrays for XGBoost using rolling window approach.
        """
        logger.info("Preparing data...")
        all_x = []
        all_y = []

        data_np = data.to_numpy()

        for start, end in indices:
            seq = data_np[start:end]
            x = seq[:self.input_seq_len]
            y = seq[self.input_seq_len:self.input_seq_len + self.target_seq_len]

            all_x.append(x)
            all_y.append(y)

        logger.info("Finished preparing data.")
        return np.array(all_x), np.array(all_y)

    def add_lag_features(self, data: pd.Series) -> pd.DataFrame:
        """
        Add lag-based features to a time series.
        """
        df = pd.DataFrame({'target': data})
        for lag in self.lags:
            df[f'lag_{lag}'] = df['target'].shift(lag)

        df = df.dropna()
        return df

    def add_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add time-based features like day, month, weekend, etc.
        Requires datetime index.
        """
        if not isinstance(df.index, pd.DatetimeIndex):
            raise ValueError("Index must be a DatetimeIndex to add time features.")

        df["dayofweek"] = df.index.dayofweek
        df["month"] = df.index.month
        df["quarter"] = df.index.quarter
        df["is_weekend"] = df.index.dayofweek.isin([5, 6]).astype(int)
        return df

    def generate_dataset(self, data: pd.Series) -> Tuple[np.ndarray, np.ndarray]:
        """
        End-to-end generation of X and y dataset.
        """
        # Add lag features if specified
        if self.lags:
            df = self.add_lag_features(data)
        else:
            df = pd.DataFrame({'target': data})

        # Add time features if specified
        if self.use_time_features:
            df = self.add_time_features(df)

        indices = self.get_indices(len(df))
        x, y = self.prepare_features(df["target"], indices)
        return x, y

import pandas as pd
df = pd.read_csv('Groundnut_data_filtered.csv')
df['Arrival_Date'] = pd.to_datetime(df['Arrival_Date'])
df = df.set_index('Arrival_Date')
series = df["Modal_Price"]  # or whatever your price column is named
print(series)
# Load your time series as a pd.Series with datetime index
# series = pd.read_csv("Groundnut_data_filtered.csv", index_col="date", parse_dates=True)["price"]

preprocessor = XGBoostTimeSeriesPreprocessor(
    input_seq_len=30,
    target_seq_len=7,
    step_size=1,
    lags=[1, 7, 14],
    use_time_features=True
)

X, y = preprocessor.generate_dataset(series)
print(X , y)

X_train, y_train = X[:-1], y[:-1]
X_pred_input = X[-1:]

# Check if best hyperparameters are saved
param_path = "best_xgb_params.json"
if os.path.exists(param_path):
    with open(param_path, "r") as f:
        best_params = json.load(f)
    print("Loaded saved hyperparameters.")
else:
    print("No saved hyperparameters found. Running tuning...")
    param_grid = {
        "n_estimators": [100, 300, 500],
        "learning_rate": [0.01, 0.05, 0.1],
        "max_depth": [3, 5, 7],
        "subsample": [0.7, 0.9, 1.0],
        "colsample_bytree": [0.7, 0.9, 1.0]
    }

    model = XGBRegressor(objective="reg:squarederror", random_state=42)
    tscv = TimeSeriesSplit(n_splits=5)
    search = RandomizedSearchCV(
        estimator=model,
        param_distributions=param_grid,
        n_iter=20,
        cv=tscv,
        scoring='neg_mean_squared_error',
        verbose=1,
        random_state=42,
        n_jobs=-1
    )
    search.fit(X_train, y_train)
    best_params = search.best_params_

    with open(param_path, "w") as f:
        json.dump(best_params, f)

# Train final model
final_model = XGBRegressor(**best_params)
final_model.fit(X_train, y_train)

# Save model
joblib.dump(final_model, "xgb_model.pkl")

# Predict next 7 days
y_pred = final_model.predict(X_pred_input)
print("Next 7-day forecast:", y_pred)

future_dates = pd.date_range(start=series.index[-1] + pd.Timedelta(days=1), periods=7)
plt.figure(figsize=(10, 5))
plt.plot(series[-30:], label="Historical Prices")
plt.plot(future_dates, y_pred.flatten(), label="Forecast", marker='o')
plt.xlabel("Date")
plt.ylabel("Price")
plt.title("7-Day Forecast")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()