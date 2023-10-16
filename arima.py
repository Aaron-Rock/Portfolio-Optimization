
# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX
from pmdarima import auto_arima


data = pd.read_csv('stocks/AAPL_data.csv')

data['date'] = pd.to_datetime(data['date'])
data.set_index('date', inplace=True)

plt.figure(figsize=(12, 6))
plt.plot(data['adjclose'], label='Adjusted Close Price')
plt.title('AAPL Adjusted Close Price Over Time')
plt.legend()
plt.show()


from statsmodels.tsa.stattools import adfuller

diff = data['adjclose'].diff().dropna()

result = adfuller(diff)
print(f'ADF Statistic: {result[0]}')
print(f'p-value: {result[1]}')

if result[1] > 0.05:
    diff = diff.diff().dropna()
    result = adfuller(diff)
    print(f'ADF Statistic after differencing: {result[0]}')
    print(f'p-value after differencing: {result[1]}')
model = auto_arima(diff, seasonal=False, trace=True)
p, d, q = model.order
exog_data = data[['volume', 'open', 'low', 'high']]

forecast_periods = 30  
forecasted_prices = []
sarima_model = SARIMAX(data['adjclose'], exog=exog_data)

for i in range(forecast_periods):
    last_date = data.index[-1] + pd.DateOffset(days=i)
    exog_forecast_step = exog_data.loc[:last_date]
    
    sarima_result = sarima_model.fit(disp=False)
    forecast_step = sarima_result.forecast(steps=1, exog=exog_forecast_step[-1:])
    forecasted_prices.append(forecast_step.values[0])
    data.loc[last_date] = forecast_step.values[0]
    exog_data = data[['volume', 'open', 'low', 'high']]
forecasted_prices_series = pd.Series(forecasted_prices, index=pd.date_range(start=data.index[-forecast_periods], periods=forecast_periods))
print("Forecasted Adjusted Close Prices:")
print(forecasted_prices_series)

# %%
from yahooquery import Ticker
import pandas as pd

symbol = 'AAPL'

start_date = '2023-08-01'
end_date = '2023-08-31'

aapl_ticker = Ticker(symbol)

historical_data = aapl_ticker.history(
    start=start_date,
    end=end_date,
    interval='1d'  
)
# %%
august_days = [
    "2023-08-01",
    "2023-08-02",
    "2023-08-03",
    "2023-08-04",
    "2023-08-07",
    "2023-08-08",
    "2023-08-09",
    "2023-08-10",
    "2023-08-11",
    "2023-08-14",
    "2023-08-15",
    "2023-08-16",
    "2023-08-17",
    "2023-08-21",
    "2023-08-22",
    "2023-08-23",
    "2023-08-24",
    "2023-08-25",
    "2023-08-28",
    "2023-08-29",
    "2023-08-30"
]

forecasted_prices_series.index = pd.to_datetime(forecasted_prices_series.index)

#%%

plt.figure(figsize=(12, 6))
plt.plot(forecasted_prices_series.index, forecasted_prices_series.values, label='Forecasted Adjusted Close Price', color='red')
plt.title('AAPL Forecasted Adjusted Close Price for August')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
plt.grid(True)
plt.show()

#%%
import yfinance as yf
import matplotlib.pyplot as plt


apple = yf.Ticker("AAPL")
hist = apple.history(start="2023-08-01", end="2023-09-01")


plt.figure(figsize=(12, 6))
hist['Close'].plot(title='Apple Adjusted Closing Price - August 2023')
plt.xlabel('Date')
plt.ylabel('Adjusted Closing Price')
plt.grid(True)
plt.show()
# %%
