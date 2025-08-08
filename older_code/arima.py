from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import adfuller
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('Groundnut_data_filtered.csv')

model = SARIMAX(
    df['Modal_Price'], 
    order=(1,1,1),              # ARIMA part
    seasonal_order=(1,1,1,12)   # SARIMA part
)
model_fit = model.fit()

print(f"AIC: {model_fit.aic}")
print(f"BIC: {model_fit.bic}")

forecast = model_fit.forecast(steps=30)


plt.figure(figsize=(10,5))
plt.plot(df['Modal_Price'], label='Observed')
plt.plot(forecast, label='Forecast', color='red')
plt.legend()
plt.show()

plot_acf(df['Modal_Price'].diff().dropna(), lags=40)
plot_pacf(df['Modal_Price'].diff().dropna(), lags=40)
plt.show()

result = adfuller(df['Modal_Price'])
print('ADF Statistic:', result[0])
print('p-value:', result[1])