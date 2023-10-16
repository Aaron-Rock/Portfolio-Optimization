
# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX
from pmdarima import auto_arima

# Load the data
data = pd.read_csv('stocks/AAPL_data.csv')

# Convert the 'date' column to datetime and set it as the index
data['date'] = pd.to_datetime(data['date'])
data.set_index('date', inplace=True)

# Visualize the adjusted close price
plt.figure(figsize=(12, 6))
plt.plot(data['adjclose'], label='Adjusted Close Price')
plt.title('AAPL Adjusted Close Price Over Time')
plt.legend()
plt.show()

# Check for stationarity
from statsmodels.tsa.stattools import adfuller

diff = data['adjclose'].diff().dropna()

result = adfuller(diff)
print(f'ADF Statistic: {result[0]}')
print(f'p-value: {result[1]}')

# If necessary, perform differencing to achieve stationarity
if result[1] > 0.05:
    diff = diff.diff().dropna()
    result = adfuller(diff)
    print(f'ADF Statistic after differencing: {result[0]}')
    print(f'p-value after differencing: {result[1]}')

# Find the best ARIMA model using auto_arima
model = auto_arima(diff, seasonal=False, trace=True)

# Get the order of the best model
p, d, q = model.order

# Create exogenous variables (volume, open, low, high)
exog_data = data[['volume', 'open', 'low', 'high']]

# Specify the forecast periods
forecast_periods = 30  # Number of periods to forecast

# Initialize an empty list to store forecasted prices
forecasted_prices = []

# Initialize the model with the initial data
sarima_model = SARIMAX(data['adjclose'], exog=exog_data)

# Forecast one step ahead for each period
for i in range(forecast_periods):
    # Determine the last date in your data
    last_date = data.index[-1] + pd.DateOffset(days=i)
    
    # Slice the exogenous data up to the current forecast date
    exog_forecast_step = exog_data.loc[:last_date]
    
    # Make the forecast for the current step
    sarima_result = sarima_model.fit(disp=False)
    forecast_step = sarima_result.forecast(steps=1, exog=exog_forecast_step[-1:])
    
    # Append the forecasted value to the list
    forecasted_prices.append(forecast_step.values[0])
    
    # Append the actual value to the data for re-fitting the model
    data.loc[last_date] = forecast_step.values[0]
    
    # Update the exogenous data
    exog_data = data[['volume', 'open', 'low', 'high']]

# Convert the list of forecasted prices to a pandas Series
forecasted_prices_series = pd.Series(forecasted_prices, index=pd.date_range(start=data.index[-forecast_periods], periods=forecast_periods))

# Print the 30 forecasted adjusted close prices
print("Forecasted Adjusted Close Prices:")
print(forecasted_prices_series)

# %%
from yahooquery import Ticker
import pandas as pd

# Define the ticker symbol for Apple
symbol = 'AAPL'

# Define the date range
start_date = '2023-08-01'
end_date = '2023-08-31'

# Create a Ticker object for Apple
aapl_ticker = Ticker(symbol)

# Get historical data for adjusted close prices
historical_data = aapl_ticker.history(
    start=start_date,
    end=end_date,
    interval='1d'  # Daily interval
)
#historical_data.reset_index(inplace=False)  # Reset the index
historical_data = historical_data.drop('symbol', axis=1) 
# Extract adjusted close prices from the historical data
adj_close_prices = historical_data['adjclose']
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


#%%
# Convert the index of forecasted_prices_series to datetime
forecasted_prices_series.index = pd.to_datetime(forecasted_prices_series.index)

# Convert the index of adj_close_prices to datetime
adj_close_prices.index = pd.to_datetime(adj_close_prices.index)

#%%
# Plot adjusted close prices vs forecasted prices
plt.figure(figsize=(12, 6))
plt.plot(forecasted_prices_series.index, forecasted_prices_series.values, label='Forecasted Adjusted Close Price', color='red')
plt.title('AAPL Forecasted Adjusted Close Price for August')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
plt.grid(True)
plt.show()

# %%
plt.figure(figsize=(12, 6))
plt.plot(forecasted_prices_series.index, historical_data['adjclose'].to_list(), label='Forecasted Adjusted Close Price', color='red')
plt.title('AAPL Forecasted Adjusted Close Price for August')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
plt.grid(True)
plt.show()
# %%
