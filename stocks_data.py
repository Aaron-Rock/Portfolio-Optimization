#%%
import pandas as pd
import numpy as np
from yahooquery import Ticker
#%%
ticker_names = pd.read_csv('ticker_names.csv')
ticker_list = ticker_names['Symbol'].to_list()
#%%
for stock in ticker_list:
    ticker = Ticker(stock)
    historical_data = ticker.history(start="2013-07-31", end="2023-08-02")
    df = pd.DataFrame(historical_data)
    df['DayOverDayReturn'] = df['close'].pct_change() * 100
    df['Adj_DayOverDayReturn'] = df['adjclose'].pct_change() * 100
    df['DayOverDayReturn'] = df['DayOverDayReturn'].fillna(0)
    df['Adj_DayOverDayReturn'] = df['Adj_DayOverDayReturn'].fillna(0)
    df['SMA50'] = df['adjclose'].rolling(window=50).mean()
    for i in range(50):
        df['SMA50'].iloc[i] = df['adjclose'].iloc[:i+1].mean()
    df['SMA200'] = df['adjclose'].rolling(window=200).mean()
    for i in range(200):
        df['SMA200'].iloc[i] = df['adjclose'].iloc[:i+1].mean()
    df['EMA'] = df['adjclose'].ewm(span=14, adjust=False).mean()
    df['30day_Volatility'] = df['adjclose'].pct_change().rolling(window=30).std() * np.sqrt(30)
    df['30day_Volatility'] =  df['30day_Volatility'].fillna(0)
    df['Stochastic_K'] = 100 * (df['adjclose'] - df['low'].rolling(window=14).min()) / (
            df['high'].rolling(window=14).max() - df['low'].rolling(window=14).min())
    df['Stochastic_D'] = df['Stochastic_K'].rolling(window=3).mean()
    df['Stochastic_K'] = df['Stochastic_K'].fillna(0)
    df['Stochastic_D'] = df['Stochastic_D'].fillna(0)
    rolling_mean = df['adjclose'].rolling(window=20).mean()
    rolling_std = df['adjclose'].rolling(window=20).std()
    df['BollingerUpper'] = rolling_mean + (rolling_std * 2)
    df['BollingerLower'] = rolling_mean - (rolling_std * 2)
    df['BollingerUpper'] = df['BollingerUpper'].fillna(0)
    df['BollingerLower'] = df['BollingerLower'].fillna(0)
    df.to_csv(f'stocks/{stock}_data.csv')

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
# %%
portfolio = pd.DataFrame()
portfolio['Date'] = aapl['date']
for stock in ticker_list:
    listing = pd.read_csv(f'stocks/{stock}_data.csv')
    portfolio[stock] = listing['adjclose']
portfolio.to_csv('portfolio.csv')
# %%
