import pandas as pd
import numpy as np
import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import MeanAbsolutePercentageError
from sklearn.preprocessing import MinMaxScaler
import keras_tuner as kt
from tensorflow.keras.callbacks import EarlyStopping


class LSTMForecaster:
    def __init__(self, n_lookback=30, n_forecast=7, crop_name='crop_name', root_path='./'):
        self.n_lookback = n_lookback
        self.n_forecast = n_forecast
        self.crop_name = crop_name
        self.root_path = root_path
        self.scaler = MinMaxScaler()
        self.seq_len = n_lookback

    def create_sequences(self, data):
        X, Y = [], []
        for i in range(self.n_lookback, len(data) - self.n_forecast + 1):
            X.append(data[i - self.n_lookback: i , 0])
            Y.append(data[i: i + self.n_forecast , 0])
        return np.array(X), np.array(Y)

    def build_model(self, hp):
        model = Sequential()
        for i in range(hp.Int('num_layers', 1, 3)):
            units = hp.Int(f'units_{i}', 32, 128, step=32)
            return_seq = i < hp.get('num_layers') - 1
            if i == 0:
                model.add(LSTM(units, return_sequences=return_seq, input_shape=(self.seq_len, 1)))
            else:
                model.add(LSTM(units, return_sequences=return_seq))
            dropout_rate = hp.Float(f'dropout_{i}', 0.0, 0.5, step=0.1)
            if dropout_rate > 0:
                model.add(Dropout(dropout_rate))
        model.add(Dense(self.n_forecast))
        model.compile(optimizer=Adam(learning_rate=hp.Float('lr', 1e-4, 1e-2, sampling='log')),
                      loss=MeanAbsolutePercentageError())
        return model

    def run_tuner(self, x_train, y_train):
        tuner = kt.RandomSearch(
            self.build_model,
            objective='val_loss',
            max_trials=2,
            executions_per_trial=2,
            directory=self.root_path+'Models/',
            project_name=f"tune_{self.crop_name}_{self.n_forecast}"
        )
        tuner.search(x_train, y_train, epochs=2, validation_split=0.2,
                     callbacks=[EarlyStopping(patience=5)], verbose=1)
        return tuner

    def get_best_model(self, x_train, y_train):
        model_path = f'{self.root_path}Models/LSTM_{self.crop_name}_{self.n_forecast}.h5'
        hyperparam_path = f'{self.root_path}Models/LSTM_{self.crop_name}_{self.n_forecast}_hp.json'
        if os.path.exists(model_path):
            model = load_model(model_path, custom_objects={'MeanAbsolutePercentageError': MeanAbsolutePercentageError})
            with open(hyperparam_path, 'r') as f:
                best_hp = json.load(f)
        else:
            tuner = self.run_tuner(x_train, y_train)
            best_model = tuner.get_best_models(1)[0]
            best_hp = tuner.get_best_hyperparameters(1)[0].values
            best_model.save(model_path)
            with open(hyperparam_path, 'w') as f:
                json.dump(best_hp, f)
            model = best_model
        return model

    def forecast(self, df):
        df_scaled = self.scaler.fit_transform(df[['Price']])
        x_all, y_all = self.create_sequences(df_scaled)
        x_all = x_all.reshape((x_all.shape[0], x_all.shape[1], 1))
        model = self.get_best_model(x_all, y_all)
        y_pred_scaled = model.predict(x_all[-1].reshape(1, self.n_lookback, 1)).reshape(-1, 1)
        y_pred = self.scaler.inverse_transform(y_pred_scaled).flatten()
        dates = pd.date_range(start=df.index[-1] + pd.Timedelta(days=1), periods=self.n_forecast)
        forecast_df = pd.DataFrame({'Date': dates, 'Predicted': y_pred})

        plt.figure(figsize=(12, 6))
        plt.plot(df.index[-30:], df['Price'].values[-30:], label='Actual')
        plt.plot(forecast_df['Date'], forecast_df['Predicted'], label='Forecast')
        plt.title('LSTM Forecast')
        plt.legend()
        plt.show()

        return forecast_df


# --- USAGE EXAMPLE ---
# prices = pd.read_csv('path_to_data.csv')
# prices['Date'] = pd.to_datetime(prices['Date'])
# prices.set_index('Date', inplace=True)
import pandas as pd
df = pd.read_csv('Groundnut_data_filtered.csv')
df['Arrival_Date'] = pd.to_datetime(df['Arrival_Date'])
df['Price'] = df['Modal_Price']
df = df.set_index('Arrival_Date')
series = df["Modal_Price"]  # or whatever your price column is named
forecaster = LSTMForecaster()
forecast = forecaster.forecast(df)
