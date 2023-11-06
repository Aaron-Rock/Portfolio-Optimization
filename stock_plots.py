#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# %%
aapl = pd.read_csv('stocks/AAPL_data.csv', index_col='Unnamed: 0')
amgn = pd.read_csv('stocks/AMGN_data.csv', index_col='Unnamed: 0')
axp = pd.read_csv('stocks/AXP_data.csv', index_col='Unnamed: 0')
ba = pd.read_csv('stocks/BA_data.csv', index_col='Unnamed: 0')
cat = pd.read_csv('stocks/CAT_data.csv', index_col='Unnamed: 0')
crm = pd.read_csv('stocks/CRM_data.csv', index_col='Unnamed: 0')
csco = pd.read_csv('stocks/CSCO_data.csv', index_col='Unnamed: 0')
cvx = pd.read_csv('stocks/CVX_data.csv', index_col='Unnamed: 0')
dis = pd.read_csv('stocks/DIS_data.csv', index_col='Unnamed: 0')
gs = pd.read_csv('stocks/GS_data.csv', index_col='Unnamed: 0')
hd = pd.read_csv('stocks/HD_data.csv', index_col='Unnamed: 0')
hon = pd.read_csv('stocks/HON_data.csv', index_col='Unnamed: 0')
ibm = pd.read_csv('stocks/IBM_data.csv', index_col='Unnamed: 0')
intc = pd.read_csv('stocks/INTC_data.csv', index_col='Unnamed: 0')
jpm = pd.read_csv('stocks/JPM_data.csv', index_col='Unnamed: 0')
jnj = pd.read_csv('stocks/JNJ_data.csv', index_col='Unnamed: 0')
ko = pd.read_csv('stocks/KO_data.csv', index_col='Unnamed: 0')
mcd = pd.read_csv('stocks/MCD_data.csv', index_col='Unnamed: 0')
mmm = pd.read_csv('stocks/MMM_data.csv', index_col='Unnamed: 0')
mrk = pd.read_csv('stocks/MRK_data.csv', index_col='Unnamed: 0')
msft = pd.read_csv('stocks/MSFT_data.csv', index_col='Unnamed: 0')
nke = pd.read_csv('stocks/NKE_data.csv', index_col='Unnamed: 0')
pg = pd.read_csv('stocks/PG_data.csv', index_col='Unnamed: 0')
trv = pd.read_csv('stocks/TRV_data.csv', index_col='Unnamed: 0')
unh = pd.read_csv('stocks/UNH_data.csv', index_col='Unnamed: 0')
v = pd.read_csv('stocks/V_data.csv', index_col='Unnamed: 0')
vz = pd.read_csv('stocks/VZ_data.csv', index_col='Unnamed: 0')
wba = pd.read_csv('stocks/WBA_data.csv', index_col='Unnamed: 0')
wmt = pd.read_csv('stocks/WMT_data.csv', index_col='Unnamed: 0')
#%%
portfolio = pd.read_csv('portfolio.csv')
portfolio['Date'] = pd.to_datetime(portfolio['Date'])
portfolio.set_index('Date', inplace=True)
portfolio = portfolio.drop(columns=["Unnamed: 0"])
#%%
stocks = [aapl, amgn, axp, ba, cat, crm, csco, cvx, dis, gs, hd, hon, ibm, intc, jpm, jnj, ko, mcd, mmm, mrk, msft, nke, pg, trv, unh, v, vz, wba, wmt]
#%%
for stock in stocks:
    stock.index = pd.to_datetime(stock.index)
    stock.reset_index(inplace=True)
    stock.rename(columns={'index': 'date'}, inplace=True) 
    stock.set_index('date', inplace=True)


# %%
def close_plot(stock, stock_name):
    plt.figure(figsize=(10, 6))
    plt.plot(stock.index, stock['close'])
    plt.title(f'{stock_name} Price Time Series')
    plt.xlabel('Date')
    plt.ylabel('Close Price')
    plt.grid()
    plt.minorticks_on()
    plt.tight_layout()
    plt.show()
# %%
def adj_close_plot(stock, stock_name):
    plt.figure(figsize=(10, 6))
    plt.plot(stock.index, stock['adjclose'])
    plt.title(f'{stock_name} Price Time Series')
    plt.xlabel('Date')
    plt.ylabel('Adjusted Close Price')
    plt.grid()
    plt.minorticks_on()
    plt.tight_layout()
    plt.show()
# %%
def moving_average_50(stock, stock_name):
    plt.figure(figsize=(10, 6))
    plt.plot(stock.index, stock['SMA50'])
    plt.title(f'{stock_name} Price Time Series 50 Day Moving Average')
    plt.xlabel('Date')
    plt.ylabel('Moving Average')
    plt.grid()
    plt.minorticks_on()
    plt.tight_layout()
    plt.show()
# %%
def close_ma50(stock, stock_name):
    plt.figure(figsize=(10, 6))
    plt.plot(stock.index, stock['adjclose'], label='Adjusted Close Price', color='blue')
    plt.plot(stock.index, stock['SMA50'], label='50-Day Moving Average', color='orange')
    plt.title(f'{stock_name} Price Time Series')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    plt.grid()
    plt.minorticks_on()
    plt.tight_layout()
    plt.show()

# %%
def daily_returns(stock, stock_name):
    plt.figure(figsize=(10, 6))
    plt.plot(stock.index, stock['DayOverDayReturn'])
    plt.title(f'{stock_name} Price Time Series Day Over Day Returns')
    plt.xlabel('Date')
    plt.ylabel('Day Over Day Returns')
    plt.legend()
    plt.grid()
    plt.minorticks_on()
    plt.tight_layout()
    plt.show()
# %%
def close_daily_returns(stock, stock_name):
    plt.figure(figsize=(10, 6))
    plt.plot(stock.index, stock['adjclose'], label='Adjusted Close Price', color='blue')
    plt.plot(stock.index, stock['DayOverDayReturn'], label='Day Over Day Returns', color='orange')
    plt.title(f'{stock_name} Price Time Series')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    plt.grid()
    plt.minorticks_on()
    plt.tight_layout()
    plt.show()

# %%
def volume(stock, stock_name):
    plt.figure(figsize=(10, 6))
    plt.plot(stock.index, stock['volume'])
    plt.title(f'{stock_name} Volume Time Series')
    plt.xlabel('Date')
    plt.ylabel('Volume')
    plt.legend()
    plt.grid()
    plt.minorticks_on()
    plt.tight_layout()
    plt.show()
# %%
def moving_average(stock, stock_name):
    plt.figure(figsize=(10, 6))
    plt.plot(stock.index, stock['EMA'])
    plt.title(f'{stock_name} Price Time Series Moving Average')
    plt.xlabel('Date')
    plt.ylabel('EMA')
    plt.legend()
    plt.grid()
    plt.minorticks_on()
    plt.tight_layout()
    plt.show()
# %%
def moving_average_200(stock, stock_name):
    plt.figure(figsize=(10, 6))
    plt.plot(stock.index, stock['SMA200'])
    plt.title(f'{stock_name} Price Time Series 200 Day Moving Average')
    plt.xlabel('Date')
    plt.ylabel('Moving Average')
    plt.legend()
    plt.grid()
    plt.minorticks_on()
    plt.tight_layout()
    plt.show()

# %%
def close_ma200(stock, stock_name):
    plt.figure(figsize=(10, 6))
    plt.plot(stock.index, stock['adjclose'], label='Adjusted Close Price', color='blue')
    plt.plot(stock.index, stock['SMA200'], label='200-Day Moving Average', color='orange')
    plt.title(f'{stock_name} Price Time Series')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    plt.grid()
    plt.minorticks_on()
    plt.tight_layout()
    plt.show()
# %%
def close_ma50_ma200(stock, stock_name):
    plt.figure(figsize=(10, 6))
    plt.plot(stock.index, stock['adjclose'], label='Adjusted Close Price', color='blue')
    plt.plot(stock.index, stock['SMA50'], label='50-Day Moving Average', color='orange')
    plt.plot(stock.index, stock['SMA200'], label='200-Day Moving Average', color='green')
    plt.title(f'{stock_name} Price Time Series')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    plt.grid()
    plt.minorticks_on()
    plt.tight_layout()
    plt.show()

# %%
def volatility_30(stock, stock_name):
    plt.figure(figsize=(10, 6))
    plt.plot(stock.index, stock['30day_Volatility'])
    plt.title(f'{stock_name} Price 30 Day Volatility')
    plt.xlabel('Date')
    plt.ylabel('Volatility')
    plt.legend()
    plt.grid()
    plt.minorticks_on()
    plt.tight_layout()
    plt.show()
# %%
def stock_data(stock, stock_name):
    plt.figure(figsize=(20, 16))
    plt.plot(stock.index, stock['adjclose'], label='Adjusted Close Price', color='blue')
    plt.plot(stock.index, stock['EMA'], label='Exponential Moving Average', color='purple')
    plt.plot(stock.index, stock['SMA50'], label='50-Day Simple Moving Average', color='orange')
    plt.plot(stock.index, stock['SMA200'], label='200-Day Simple Moving Average', color='green')
    plt.plot(stock.index, stock['BollingerUpper'], label='Bollinger Upper Band', color='red', linestyle='--')
    plt.plot(stock.index, stock['BollingerLower'], label='Bollinger Lower Band', color='purple', linestyle='--')
    plt.title(f'{stock_name} Price Time Series')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    plt.grid()
    plt.minorticks_on()
    plt.tight_layout()
    plt.show()

# %%
