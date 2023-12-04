#%%
from keras.models import load_model
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
import yfinance as yf
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.io as pio
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
#%%
stock_files = [
    'aapl_lstm.csv','amgn_lstm.csv', 'axp_lstm.csv', 'ba_lstm.csv','cat_lstm.csv', 'crm_lstm.csv', 'csco_lstm.csv',
    'cvx_lstm.csv', 'gs_lstm.csv','hd_lstm.csv', 'hon_lstm.csv', 'ibm_lstm.csv','jnj_lstm.csv', 'jpm_lstm.csv',
    'ko_lstm.csv', 'mcd_lstm.csv', 'mmm_lstm.csv',
    'mrk_lstm.csv', 'msft_lstm.csv', 'nke_lstm.csv',
    'pg_lstm.csv', 'trv_lstm.csv', 'unh_lstm.csv',
    'v_lstm.csv', 'vz_lstm.csv', 'wba_lstm.csv',
    'wmt_lstm.csv'
]


stock_files_dict = {file.split('_')[0].upper(): file for file in stock_files}
results_df = pd.DataFrame(columns=[
    'Stock Symbol', 'Last Adj Close', 'Last Forecasted Price', 
    'RMSE Smoothed', 'MSE Smoothed', 'MAE Smoothed', 'Forecasted Difference','Returns', 'Return %', 'Risk'
])

adjclose_prices_dict = {}

for ticker, file in stock_files_dict.items():
    print(f"Processing {ticker}")
    model = load_model(f'{ticker}_model.h5')
    data = pd.read_csv(f'model_files/{file}', parse_dates=['date'], index_col='date') 
    data = data.iloc[:-410]
    
    selected_features = [
        'open', 'high', 'low', 'volume', 'DayOverDayReturn','Revenue', 'EBITDA',
    'SMA50', 'SMA200', '30day_Volatility', 'sp500', 'consumer_services', 'consumer_goods', 'nasdaq', 'technology', 'financials', 'Open_Close_Diff', 'High_Low_Diff',
        'GDP', 'UNRATE', 'CPIAUCSL', 'world_index','adjclose'
    ]

    data = data.filter(selected_features)
    dataset = data.values
    # Scale the data
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(dataset)

    # Create the training data set
    training_data_len = int(np.ceil(len(dataset) * .95))
    train_data = scaled_data[0:int(training_data_len), :]

    # Split the data into x_train and y_train data sets
    x_train, y_train = [], []
    look_back = 1000
    forecast_horizon = 90 
    daily_returns = data['adjclose'].pct_change()
    last_1000_days_returns = daily_returns.tail(look_back)
    risk = daily_returns.std()

    for i in range(look_back, len(train_data) - forecast_horizon + 1):
        x_train.append(train_data[i-look_back:i]) 
        y_train.append(train_data[i:i+forecast_horizon, -1])

    x_train, y_train = np.array(x_train), np.array(y_train)
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], len(selected_features)))
    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.2, random_state=42, shuffle=True)
    
    num_features = scaled_data.shape[1] 
    last_look_back_days_data = dataset[-look_back:, :]
    last_look_back_days_scaled = scaler.transform(last_look_back_days_data)
    X_test = last_look_back_days_scaled.reshape(1, look_back, num_features)

    # Generate the predictions
    pred_price_scaled = model.predict(X_test)
    dummy_array = np.zeros((len(pred_price_scaled[0]), scaled_data.shape[1]))
    dummy_array[:, -1] = pred_price_scaled[0]

    # Inverse transform to get the actual predictions
    pred_close_price = scaler.inverse_transform(dummy_array)[:, -1]

    last_actual_price = dataset[-1, -1] 
    pred_close_price[0] = last_actual_price  
   
    future_dates = pd.date_range(start=data.index[-1] + pd.Timedelta(days=1), periods=forecast_horizon, freq='B')

    forecasted_prices = pd.Series(pred_close_price.flatten(), index=future_dates)
    smoothed_prices_ewm = forecasted_prices.ewm(span=5).mean()

    last_date = data.index[-1]
    future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=forecast_horizon, freq='B')

    actual_data = yf.download(ticker, start=future_dates[0], end=future_dates[-1])

    actual_closing_prices = actual_data['Adj Close']

    actual_closing_prices = actual_closing_prices.reindex(future_dates, method='nearest')
    adjclose_prices_dict[ticker] = actual_closing_prices
    
    # Calculate performance metrics
    mae = mean_absolute_error(actual_closing_prices, pred_close_price)
    mse = mean_squared_error(actual_closing_prices, pred_close_price)
    rmse = mean_squared_error(actual_closing_prices, pred_close_price, squared=False)

    print(f"Backtest Mean Absolute Error: {mae}")
    print(f"Backtest Mean Squared Error: {mse}")
    print(f"Backtest Root Mean Squared Error: {rmse}")

    mae_smoothed = mean_absolute_error(actual_closing_prices, smoothed_prices_ewm)
    mse_smoothed = mean_squared_error(actual_closing_prices, smoothed_prices_ewm)
    rmse_smoothed = mean_squared_error(actual_closing_prices, smoothed_prices_ewm, squared=False)

    print(f"Backtest Mean Absolute Error Smoothed: {mae}")
    print(f"Backtest Mean Squared Error Smoothed: {mse}")
    print(f"Backtest Root Mean Squared Error Smoothed: {rmse}")

    final_diff = smoothed_prices_ewm[-1] -actual_closing_prices[-1]
    print(f'Final Price Difference: {final_diff}')
    
    #Adding the New Dataframes
    last_adj_close = data['adjclose'].iloc[-1]
    last_forecasted_price = pred_close_price[-1]
    returns = last_forecasted_price - last_adj_close
    percent_return = (returns / last_adj_close) * 100
    
    new_row = {
        'Stock Symbol': ticker,
        'Last Adj Close': last_adj_close,
        'Last Forecasted Price': last_forecasted_price,
        'RMSE Smoothed': rmse_smoothed,
        'MSE Smoothed': mse_smoothed,
        'MAE Smoothed': mae_smoothed,
        'Forecasted Difference': final_diff,
        'Returns': returns,
        'Return %': percent_return,
        'Risk': risk
    }

    results_df = results_df.append(new_row, ignore_index=True)
    
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
        y=data['adjclose'],
        mode='lines',
        name='Actual Prices'
    )

    fig = go.Figure(data=[trace_actual, trace_pred])
    fig.update_layout(
        title=f'Stock: {ticker} - 90-Day Future Price Prediction',
        xaxis=dict(title='Date', showgrid=True, gridcolor='lightgray'),
        yaxis=dict(title='Price USD ($)', showgrid=True, gridcolor='lightgray'),
        plot_bgcolor='white',
        paper_bgcolor='white',
        showlegend=True,
    )
    fig.show()

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
        y=data['adjclose'],
        mode='lines',
        name='Actual Prices'
    )
    
    trace_actual_prices = go.Scatter(
    x=actual_closing_prices.index,
    y=actual_closing_prices.values,
    mode='lines',
    name='Actual Closing Prices'
)

    # Create the figure and add only the actual and smoothed traces
    fig = go.Figure(data=[trace_actual, trace_smoothed_pred])
    fig.add_trace(trace_actual_prices)

    # Update the layout
    fig.update_layout(
        title=f'Stock: {ticker} - Actual Prices vs Smoothed Predicted Prices',
        xaxis=dict(title='Date', showgrid=True, gridcolor='lightgray'),
        yaxis=dict(title='Price USD ($)', showgrid=True, gridcolor='lightgray'),
        plot_bgcolor='white',
        paper_bgcolor='white',
        showlegend=True,
    )

    fig.show()
# %%
results_df.to_csv('results.csv')
# %%
saved_future_dates = future_dates
stocks_df = pd.DataFrame(adjclose_prices_dict, index=saved_future_dates)
stocks_df.to_csv('stock_backtest.csv')