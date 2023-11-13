
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
#%%
data = pd.read_csv('apple_lstm.csv', parse_dates=['date'], index_col='date') 
data = data.iloc[:-120]
test_data = data[:-120]
#%%

# selected_features = [
#     'open', 'high', 'low', 'volume', 'DayOverDayReturn', 'SMA50', 'SMA200', 'EMA', '30day_Volatility',
#     'Stochastic_K', 'Stochastic_D', 'BollingerUpper', 'BollingerLower',
#     'GDP', 'UNRATE', 'CPIAUCSL', 'world_index', 'close'
# ]
selected_features = [
    'volume', 'DayOverDayReturn', 'SMA50', 'EMA', '30day_Volatility',
    'Stochastic_K', 'Stochastic_D', 'GDP', 'UNRATE', 'CPIAUCSL', 'world_index', 'close'
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
look_back = 90
forecast_horizon = 120  # Assuming you're still predicting 'close' price for 120 days

for i in range(look_back, len(train_data) - forecast_horizon + 1):
    x_train.append(train_data[i-look_back:i])  # Past 90 days' data
    y_train.append(train_data[i:i+forecast_horizon, -1])  # Next 120 days' close prices as label

x_train, y_train = np.array(x_train), np.array(y_train)
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], len(selected_features)))
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.2, random_state=42)
#%%
# Build the LSTM model
model = Sequential()
model.add(LSTM(100, return_sequences=True, input_shape=(look_back, len(selected_features))))
model.add(LSTM(50, return_sequences=False))
# model.add(Dense(25))  # Can be tuned
model.add(Dense(120)) 
model.compile(optimizer=Adam(learning_rate=0.0001), loss='mse')
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
    batch_size=8,
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
last_90_days_data = dataset[-90:, :]
last_90_days_scaled = scaler.transform(last_90_days_data)
num_features = scaled_data.shape[1] 
X_test = last_90_days_scaled.reshape(1, 90, num_features)

pred_price_scaled = model.predict(X_test)
dummy_array = np.zeros((pred_price_scaled.shape[1], scaled_data.shape[1]))
close_index = -1
dummy_array[:, close_index] = pred_price_scaled[0, :]
pred_close_price = scaler.inverse_transform(dummy_array)[:, close_index]

#%%
# Generate dates for future predictions
future_dates = pd.date_range(start=data.index[-1], periods=120, freq='B')


actual_data = yf.download('AAPL', start=future_dates.min(), end=future_dates.max())

# Extract the actual closing prices
actual_closing_prices = actual_data['Close']

# Align the actual closing prices with the predicted dates
# Ensure the indexes match: 'future_dates' should correspond to the index of 'actual_closing_prices'
actual_closing_prices = actual_closing_prices.reindex(future_dates, method='nearest')

# Calculate the performance metrics
mae = mean_absolute_error(actual_closing_prices, pred_close_price)
rmse = mean_squared_error(actual_closing_prices, pred_close_price, squared=False)

print(f"Backtest Mean Absolute Error: {mae}")
print(f"Backtest Root Mean Squared Error: {rmse}")
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
    title='Stock: AAPL - 120-Day Future Price Prediction',
    xaxis=dict(title='Date', showgrid=True, gridcolor='lightgray'),
    yaxis=dict(title='Price USD ($)', showgrid=True, gridcolor='lightgray'),
    plot_bgcolor='white',
    paper_bgcolor='white',
    showlegend=True,
)
fig.show()
# %%
import yfinance as yf
import pandas as pd

# Define the ticker symbol and the start and end dates based on your 'future_dates'
ticker_symbol = 'AAPL'
start_date = future_dates.min().strftime('%Y-%m-%d')  # format as string
end_date = future_dates.max().strftime('%Y-%m-%d')  # format as string

# Fetch the historical data
data = yf.download(ticker_symbol, start=start_date, end=end_date)

# Extract the closing prices
closing_prices = data['Close']

# If 'future_dates' is a pandas Series or DatetimeIndex, you can reindex 'closing_prices' to match those dates
# This assumes 'future_dates' are business days. If they include non-business days, this step may require adjustment.
closing_prices = closing_prices.reindex(future_dates, method='nearest')
# %%
