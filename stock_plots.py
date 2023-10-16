#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# %%
aapl = pd.read_csv('stocks/AAPL_data.csv')
amgn = pd.read_csv('stocks/AMGN_data.csv')
axp = pd.read_csv('stocks/AXP_data.csv')
ba = pd.read_csv('stocks/BA_data.csv')
cat = pd.read_csv('stocks/CAT_data.csv')
crm = pd.read_csv('stocks/CRM_data.csv')
csco = pd.read_csv('stocks/CSCO_data.csv')
cvx = pd.read_csv('stocks/CVX_data.csv')
dis = pd.read_csv('stocks/DIS_data.csv')
gs = pd.read_csv('stocks/GS_data.csv')
hd = pd.read_csv('stocks/HD_data.csv')
hon = pd.read_csv('stocks/HON_data.csv')
ibm = pd.read_csv('stocks/IBM_data.csv')
intc = pd.read_csv('stocks/INTC_data.csv')
jpm = pd.read_csv('stocks/JPM_data.csv')
jnj = pd.read_csv('stocks/JNJ_data.csv')
ko = pd.read_csv('stocks/KO_data.csv')
mcd = pd.read_csv('stocks/MCD_data.csv')
mmm = pd.read_csv('stocks/MMM_data.csv')
mrk = pd.read_csv('stocks/MRK_data.csv')
msft = pd.read_csv('stocks/MSFT_data.csv')
nke = pd.read_csv('stocks/NKE_data.csv')
pg = pd.read_csv('stocks/PG_data.csv')
trv = pd.read_csv('stocks/TRV_data.csv')
unh = pd.read_csv('stocks/UNH_data.csv')
v = pd.read_csv('stocks/V_data.csv')
vz = pd.read_csv('stocks/VZ_data.csv')
wba = pd.read_csv('stocks/WBA_data.csv')
wmt = pd.read_csv('stocks/WMT_data.csv')
#%%
portfolio = pd.read_csv('portfolio.csv')
portfolio['Date'] = pd.to_datetime(portfolio['Date'])
portfolio.set_index('Date', inplace=True)
portfolio = portfolio.drop(columns=["Unnamed: 0"])
#%%
stocks = [aapl, amgn, axp, ba, cat, crm, csco, cvx, dis, gs, hd, hon, ibm, intc, jpm, jnj, ko, mcd, mmm, mrk, msft, nke, pg, trv, unh, v, vz, wba, wmt]
#%%
for stock in stocks:
    stock['date'] = pd.to_datetime(stock['date'])
    stock.set_index('date', inplace=True)


# %%
def close_plot(stock):
    plt.figure(figsize=(10, 6))
    plt.plot(stock.index, stock['close'])
    plt.title(f'{stock.iloc[1, 0]} Price Time Series')
    plt.xlabel('Date')
    plt.ylabel('Close Price')
    plt.legend()
    plt.grid()
    plt.minorticks_on()
    plt.tight_layout()
    plt.show()
# %%
def adj_close_plot(stock):
    plt.figure(figsize=(10, 6))
    plt.plot(stock.index, stock['adjclose'])
    plt.title(f'{stock.iloc[1, 0]} Price Time Series')
    plt.xlabel('Date')
    plt.ylabel('Adjusted Close Price')
    plt.legend()
    plt.grid()
    plt.minorticks_on()
    plt.tight_layout()
    plt.show()
# %%
def moving_average_50(stock):
    plt.figure(figsize=(10, 6))
    plt.plot(stock.index, stock['SMA50'])
    plt.title(f'{stock.iloc[1, 0]} Price Time Series 50 Day Moving Average')
    plt.xlabel('Date')
    plt.ylabel('Moving Average')
    plt.legend()
    plt.grid()
    plt.minorticks_on()
    plt.tight_layout()
    plt.show()
# %%
def close_ma50(stock):
    plt.figure(figsize=(10, 6))
    plt.plot(stock.index, stock['adjclose'], label='Adjusted Close Price', color='blue')
    plt.plot(stock.index, stock['SMA50'], label='50-Day Moving Average', color='orange')
    plt.title(f'{stock.iloc[1, 0]} Price Time Series')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    plt.grid()
    plt.minorticks_on()
    plt.tight_layout()
    plt.show()

# %%
def daily_returns(stock):
    plt.figure(figsize=(10, 6))
    plt.plot(stock.index, stock['DayOverDayReturn'])
    plt.title(f'{stock.iloc[1, 0]} Price Time Series Day Over Day Returns')
    plt.xlabel('Date')
    plt.ylabel('Day Over Day Returns')
    plt.legend()
    plt.grid()
    plt.minorticks_on()
    plt.tight_layout()
    plt.show()
# %%
def close_daily_returns(stock):
    plt.figure(figsize=(10, 6))
    plt.plot(stock.index, stock['adjclose'], label='Adjusted Close Price', color='blue')
    plt.plot(stock.index, stock['DayOverDayReturn'], label='Day Over Day Returns', color='orange')
    plt.title(f'{stock.iloc[1, 0]} Price Time Series')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    plt.grid()
    plt.minorticks_on()
    plt.tight_layout()
    plt.show()

# %%
def volume(stock):
    plt.figure(figsize=(10, 6))
    plt.plot(stock.index, stock['volume'])
    plt.title(f'{stock.iloc[1, 0]} Volume Time Series')
    plt.xlabel('Date')
    plt.ylabel('Volume')
    plt.legend()
    plt.grid()
    plt.minorticks_on()
    plt.tight_layout()
    plt.show()
# %%
def moving_average(stock):
    plt.figure(figsize=(10, 6))
    plt.plot(stock.index, stock['EMA'])
    plt.title(f'{stock.iloc[1, 0]} Price Time Series Moving Average')
    plt.xlabel('Date')
    plt.ylabel('EMA')
    plt.legend()
    plt.grid()
    plt.minorticks_on()
    plt.tight_layout()
    plt.show()
# %%
def moving_average_200(stock):
    plt.figure(figsize=(10, 6))
    plt.plot(stock.index, stock['SMA200'])
    plt.title(f'{stock.iloc[1, 0]} Price Time Series 200 Day Moving Average')
    plt.xlabel('Date')
    plt.ylabel('Moving Average')
    plt.legend()
    plt.grid()
    plt.minorticks_on()
    plt.tight_layout()
    plt.show()

# %%
def close_ma200(stock):
    plt.figure(figsize=(10, 6))
    plt.plot(stock.index, stock['adjclose'], label='Adjusted Close Price', color='blue')
    plt.plot(stock.index, stock['SMA200'], label='200-Day Moving Average', color='orange')
    plt.title(f'{stock.iloc[1, 0]} Price Time Series')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    plt.grid()
    plt.minorticks_on()
    plt.tight_layout()
    plt.show()
# %%
def close_ma50_ma200(stock):
    plt.figure(figsize=(10, 6))
    plt.plot(stock.index, stock['adjclose'], label='Adjusted Close Price', color='blue')
    plt.plot(stock.index, stock['SMA50'], label='50-Day Moving Average', color='orange')
    plt.plot(stock.index, stock['SMA200'], label='200-Day Moving Average', color='green')
    plt.title(f'{stock.iloc[1, 0]} Price Time Series')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    plt.grid()
    plt.minorticks_on()
    plt.tight_layout()
    plt.show()

# %%
def volatility_30(stock):
    plt.figure(figsize=(10, 6))
    plt.plot(stock.index, stock['30day_Volatility'])
    plt.title(f'{stock.iloc[1, 0]} Price 30 Day Volatility')
    plt.xlabel('Date')
    plt.ylabel('Volatility')
    plt.legend()
    plt.grid()
    plt.minorticks_on()
    plt.tight_layout()
    plt.show()
# %%
def stock_data(stock):
    plt.figure(figsize=(20, 16))
    plt.plot(stock.index, stock['adjclose'], label='Adjusted Close Price', color='blue')
    plt.plot(stock.index, stock['EMA'], label='Moving Average', color='purple')
    plt.plot(stock.index, stock['SMA50'], label='50-Day Moving Average', color='orange')
    plt.plot(stock.index, stock['SMA200'], label='200-Day Moving Average', color='green')
    plt.plot(stock.index, stock['BollingerUpper'], label='Bollinger Upper', color='red', linestyle='--')
    plt.plot(stock.index, stock['BollingerLower'], label='Bollinger Lower', color='purple', linestyle='--')
    plt.title(f'{stock.iloc[1, 0]} Price Time Series')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    plt.grid()
    plt.minorticks_on()
    plt.tight_layout()
    plt.show()

# %%
