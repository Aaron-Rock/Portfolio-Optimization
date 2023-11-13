
#%%
# Import necessary libraries
import pandas as pd
import numpy as np
import yfinance as yf
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.io as pio
import plotly.graph_objects as go
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import mean_absolute_error, mean_squared_error
from keras.regularizers import l2
from sklearn.metrics import r2_score
#%%
# data = pd.read_csv('apple_lstm.csv', parse_dates=['date'], index_col='date') 
# data = pd.read_csv('ba_lstm.csv', parse_dates=['date'], index_col='date') 
data = pd.read_csv('model_files/aapl_lstm.csv', parse_dates=['date'], index_col='date') 
data = data.iloc[:-200]
test_data = data[:-120]
#%%

# selected_features = [
#     'open', 'high', 'low', 'volume', 'DayOverDayReturn', 'SMA50', 'SMA200', 'EMA', '30day_Volatility',
#     'Stochastic_K', 'Stochastic_D', 'BollingerUpper', 'BollingerLower',
#     'GDP', 'UNRATE', 'CPIAUCSL', 'world_index', 'close'
# ]
selected_features = [
    'volume','DayOverDayReturn', 'SMA50', 'EMA', '30day_Volatility',
    'Stochastic_K', 'Stochastic_D', 'GDP', 'UNRATE', 'CPIAUCSL', 'world_index', 'EBITDA', 'EBIT','Revenue','close'
]

data = data.filter(selected_features)
dataset = data.values
#%%
# Scale the data
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(dataset)

# Create the training data set
training_data_len = int(np.ceil(len(dataset) * .95))
train_data = scaled_data[0:int(training_data_len), :]

# Split the data into x_train and y_train data sets
x_train, y_train = [], []
look_back = 1000 #90
forecast_horizon = 200  # Assuming you're still predicting 'close' price for 120 days

for i in range(look_back, len(train_data) - forecast_horizon + 1):
    x_train.append(train_data[i-look_back:i])  # Past 90 days' data
    y_train.append(train_data[i:i+forecast_horizon, -1])  # Next 120 days' close prices as label

x_train, y_train = np.array(x_train), np.array(y_train)
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], len(selected_features)))
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.2, random_state=42)
#%%
# Build the LSTM model
model = Sequential()
# model.add(LSTM(100, return_sequences=True, input_shape=(look_back, len(selected_features)), kernel_regularizer=l2(0.01), recurrent_regularizer=l2(0.01)))
model.add(LSTM(100, return_sequences=True, input_shape=(look_back, len(selected_features))))
model.add(LSTM(50, return_sequences=False))
# model.add(Dense(25))  # Can be tuned
model.add(Dense(forecast_horizon)) 
model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
model.summary()

early_stopping = EarlyStopping(
    monitor='val_loss',  # Monitor the validation loss
    patience=10,         # Number of epochs with no improvement after which training will be stopped
    restore_best_weights=True  # Restores model weights from the epoch with the best value of the monitored metric
)

# Train the model with the early stopping callback
history = model.fit(
    x_train, y_train,
    epochs=10,
    batch_size=32,
    validation_data=(x_val, y_val),
    callbacks=[early_stopping]  # Include the callback in the callbacks list
)


# Plot the training and validation loss
plt.figure(figsize=(10, 6))
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend()
plt.show()

#%%
# last_90_days_data = dataset[-90:, :]
# last_90_days_scaled = scaler.transform(last_90_days_data)
num_features = scaled_data.shape[1] 
last_look_back_days_data = dataset[-look_back:, :]
last_look_back_days_scaled = scaler.transform(last_look_back_days_data)
X_test = last_look_back_days_scaled.reshape(1, look_back, num_features)

# Generate the predictions
pred_price_scaled = model.predict(X_test)

# Prepare a dummy array for inverse scaling
# Make sure the array has the same number of columns as the training feature set
dummy_array = np.zeros((len(pred_price_scaled[0]), scaled_data.shape[1]))

# Insert the predicted 'close' price into the corresponding place in the dummy array
# Assuming 'close' is the last column in the dataset
dummy_array[:, -1] = pred_price_scaled[0]

# Inverse transform to get the actual predictions
pred_close_price = scaler.inverse_transform(dummy_array)[:, -1]

# Ensure the first predicted price aligns with the last actual price
last_actual_price = dataset[-1, -1]  # Get the last actual 'close' price (non-scaled)
pred_close_price[0] = last_actual_price  # Set the first prediction to the last actual price

# Generate the dates for the forecast
future_dates = pd.date_range(start=data.index[-1] + pd.Timedelta(days=1), periods=forecast_horizon, freq='B')

forecasted_prices = pd.Series(pred_close_price.flatten())
# Now you can apply the exponential weighted moving average (EWMA)
smoothed_prices_ewm = forecasted_prices.ewm(span=5).mean()
#%%
# forecasted_prices = pd.Series(pred_close_price.flatten())

# # Now you can apply the exponential weighted moving average (EWMA)
# smoothed_prices_ewm = forecasted_prices.ewm(span=5).mean()
#%%
# Generate dates for future predictions
# future_dates = pd.date_range(start=data.index[-1], periods=120, freq='B')


# actual_data = yf.download('NKE', start=future_dates.min(), end=future_dates.max())

# # Extract the actual closing prices
# actual_closing_prices = actual_data['Close']

# # Align the actual closing prices with the predicted dates
# # Ensure the indexes match: 'future_dates' should correspond to the index of 'actual_closing_prices'
# actual_closing_prices = actual_closing_prices.reindex(future_dates, method='nearest')

# # Calculate the performance metrics
# mae = mean_absolute_error(actual_closing_prices, pred_close_price)
# rmse = mean_squared_error(actual_closing_prices, pred_close_price, squared=False)

# print(f"Backtest Mean Absolute Error: {mae}")
# print(f"Backtest Root Mean Squared Error: {rmse}")

last_date = data.index[-1]
future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=forecast_horizon, freq='B')

# Download actual data for the predicted period
actual_data = yf.download('AAPL', start=future_dates[0], end=future_dates[-1])

# Extract the actual closing prices
actual_closing_prices = actual_data['Close']

# Reindex the actual closing prices to match the predicted dates
actual_closing_prices = actual_closing_prices.reindex(future_dates, method='nearest')

# Calculate performance metrics
mae = mean_absolute_error(actual_closing_prices, pred_close_price)
rmse = mean_squared_error(actual_closing_prices, pred_close_price, squared=False)

print(f"Backtest Mean Absolute Error: {mae}")
print(f"Backtest Root Mean Squared Error: {rmse}")

mae = mean_absolute_error(actual_closing_prices, smoothed_prices_ewm)
rmse = mean_squared_error(actual_closing_prices, smoothed_prices_ewm, squared=False)

print(f"Backtest Mean Absolute Error Smoothed: {mae}")
print(f"Backtest Root Mean Squared Error Smoothed: {rmse}")

#%%
r_squared = r2_score(actual_closing_prices, pred_close_price)
print(f"Backtest R-squared: {r_squared}")

# Calculate R-squared for the smoothed predictions
r_squared_smoothed = r2_score(actual_closing_prices, smoothed_prices_ewm.values)
print(f"Backtest R-squared Smoothed: {r_squared_smoothed}")
#%%
# Visualise the predicted stock price
trace_pred = go.Scatter(
    x=future_dates,
    y=pred_close_price.flatten(),
    mode='lines',
    name='Predicted Future Prices'
)

# Existing data traces
trace_actual = go.Scatter(
    x=data.index,
    y=data['close'],
    mode='lines',
    name='Actual Prices'
)

fig = go.Figure(data=[trace_actual, trace_pred])
fig.update_layout(
    title='Stock: AAPL - 200-Day Future Price Prediction',
    xaxis=dict(title='Date', showgrid=True, gridcolor='lightgray'),
    yaxis=dict(title='Price USD ($)', showgrid=True, gridcolor='lightgray'),
    plot_bgcolor='white',
    paper_bgcolor='white',
    showlegend=True,
)
fig.show()
# %%
# Trace for the smoothed predicted prices
trace_smoothed_pred = go.Scatter(
    x=future_dates,
    y=smoothed_prices_ewm.values,
    mode='lines',
    name='Smoothed Predicted Prices'
)

# Existing data traces
trace_actual = go.Scatter(
    x=data.index,
    y=data['close'],
    mode='lines',
    name='Actual Prices'
)

# Create the figure and add only the actual and smoothed traces
fig = go.Figure(data=[trace_actual, trace_smoothed_pred])

# Update the layout
fig.update_layout(
    title='Stock: AAPL - Actual Prices vs Smoothed Predicted Prices',
    xaxis=dict(title='Date', showgrid=True, gridcolor='lightgray'),
    yaxis=dict(title='Price USD ($)', showgrid=True, gridcolor='lightgray'),
    plot_bgcolor='white',
    paper_bgcolor='white',
    showlegend=True,
)

# Show the figure
fig.show()
# %%
