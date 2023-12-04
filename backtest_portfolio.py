
#%%
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
from scipy.stats import linregress

# Load data
weights_file_path = 'optimal_weights.csv'
prices_file_path = 'stock_backtest.csv'

optimal_weights_df = pd.read_csv(weights_file_path)
stock_prices_df = pd.read_csv(prices_file_path)

# Process stock prices DataFrame
stock_prices_df.rename(columns={'Unnamed: 0': 'Date'}, inplace=True)
stock_prices_df.set_index('Date', inplace=True)
stock_prices_df.index = pd.to_datetime(stock_prices_df.index)

# Calculate daily returns
daily_returns = stock_prices_df.pct_change().dropna()

# Optimized portfolio
optimal_weights_df.set_index('Stock', inplace=True)
optimal_weights_df = optimal_weights_df.reindex(daily_returns.columns)
portfolio_daily_returns = (daily_returns * optimal_weights_df['Optimal Weight']).sum(axis=1)
portfolio_cumulative_returns = (1 + portfolio_daily_returns).cumprod()

# S&P 500 data
sp500_ticker = '^GSPC'
start_date = stock_prices_df.index.min().strftime('%Y-%m-%d')
end_date = stock_prices_df.index.max().strftime('%Y-%m-%d')
sp500_data = yf.download(sp500_ticker, start=start_date, end=end_date)
sp500_daily_returns = sp500_data['Adj Close'].pct_change().dropna()
sp500_cumulative_returns = (1 + sp500_daily_returns).cumprod()

#%% Dow
djia_ticker = '^DJI'
start_date = stock_prices_df.index.min().strftime('%Y-%m-%d')
end_date = stock_prices_df.index.max().strftime('%Y-%m-%d')
djia_data = yf.download(djia_ticker, start=start_date, end=end_date)
djia_daily_returns = djia_data['Adj Close'].pct_change().dropna()
djia_cumulative_returns = (1 + djia_daily_returns).cumprod()

# Performance metrics function
def calculate_performance_metrics(cumulative_returns, daily_returns):
    days_per_year = 252
    annualized_return = cumulative_returns.iloc[-1] ** (days_per_year / len(cumulative_returns)) - 1
    annualized_volatility = np.sqrt(np.mean(daily_returns ** 2)) * np.sqrt(days_per_year)
    sharpe_ratio = annualized_return / annualized_volatility
    
    return {
        'Annualized Return': annualized_return,
        'Annualized Volatility': annualized_volatility,
        'Sharpe Ratio': sharpe_ratio
    }

# Calculate summaries
optimized_summary = calculate_performance_metrics(portfolio_cumulative_returns, portfolio_daily_returns)
sp500_summary = calculate_performance_metrics(sp500_cumulative_returns, sp500_daily_returns)
dow_summary = calculate_performance_metrics(djia_cumulative_returns, djia_daily_returns)

# Print summaries
print("Optimized Portfolio Summary:")
print(optimized_summary)
print("\nS&P 500 Summary:")
print(sp500_summary)
print("\nDow Summary:")
print(dow_summary)

# Plot cumulative returns
plt.figure(figsize=(14, 7))
plt.plot(portfolio_cumulative_returns, label='Optimized Portfolio Cumulative Returns')
plt.plot(sp500_cumulative_returns, label='S&P 500 Cumulative Returns', linestyle='-.')
plt.plot(djia_cumulative_returns, label='Dow Cumulative Returns', linestyle=':')
plt.title('Comparison of Cumulative Returns Over Time')
plt.xlabel('Date')
plt.ylabel('Cumulative Returns')
plt.legend()
plt.grid(True)
plt.show()

# R-squared calculation
aligned_returns = pd.DataFrame({
    'Portfolio': portfolio_daily_returns,
    'SP500': sp500_daily_returns
}).dropna()
slope, intercept, r_value, p_value, std_err = linregress(aligned_returns['SP500'], aligned_returns['Portfolio'])
r_squared = r_value ** 2
print("R-squared:", r_squared)

risk_free_rate = 0.01
beta = slope
annualized_market_return = sp500_summary['Annualized Return']
alpha = optimized_summary['Annualized Return'] - (risk_free_rate + beta * (annualized_market_return - risk_free_rate))
print("Alpha:", alpha)

# Time period return calculation
cumulative_return_optimized_portfolio = (1 + portfolio_daily_returns).prod() - 1
cumulative_return_sp500 = (1 + sp500_daily_returns).prod() - 1
cumulative_return_dow = (1 + djia_daily_returns).prod() - 1


#%%
annualized_volatility_optimized = np.sqrt(np.mean(portfolio_daily_returns ** 2)) * np.sqrt(252)

risk_free_rate = 0.01
cumulative_risk_free_rate = (1 + risk_free_rate) ** (len(portfolio_daily_returns)/252) - 1
sharpe_ratio_optimized = (cumulative_return_optimized_portfolio - cumulative_risk_free_rate) / annualized_volatility_optimized

print("Cumulative Return - Optimized Portfolio:", cumulative_return_optimized_portfolio)
print("Cumulative Return - SP500:", cumulative_return_sp500)
print("Cumulative Return - Dow:", cumulative_return_dow)

# %%
import plotly.graph_objects as go


fig = go.Figure()


fig.add_trace(go.Scatter(x=portfolio_cumulative_returns.index, y=portfolio_cumulative_returns,
                         mode='lines', name='Optimized Portfolio Cumulative Returns'))
fig.add_trace(go.Scatter(x=sp500_cumulative_returns.index, y=sp500_cumulative_returns,
                         mode='lines', name='S&P 500 Cumulative Returns', line=dict(dash='dash')))
fig.add_trace(go.Scatter(x=djia_cumulative_returns.index, y=djia_cumulative_returns,
                         mode='lines', name='Dow Cumulative Returns', line=dict(dash='dot')))


fig.update_layout(title='Comparison of Cumulative Returns Over Time',
                  xaxis_title='Date',
                  yaxis_title='Cumulative Returns',
                  legend_title='Legends',
                  template='plotly_white')

fig.show()
# %%
# %%
