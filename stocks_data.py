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
    historical_data = ticker.history(period='5y')
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
    print("Working on "+stock)
    df.to_csv(f'{stock}_data.csv')

# %%
aapl = pd.read_csv('AAPL_data.csv')
amgn = pd.read_csv('AMGN_data.csv')
axp = pd.read_csv('AXP_data.csv')
ba = pd.read_csv('BA_data.csv')
cat = pd.read_csv('CAT_data.csv')
crm = pd.read_csv('CRM_data.csv')
csco = pd.read_csv('CSCO_data.csv')
cvx = pd.read_csv('CVX_data.csv')
dis = pd.read_csv('DIS_data.csv')
gs = pd.read_csv('GS_data.csv')
hd = pd.read_csv('HD_data.csv')
hon = pd.read_csv('HON_data.csv')
ibm = pd.read_csv('IBM_data.csv')
intc = pd.read_csv('INTC_data.csv')
jpm = pd.read_csv('JPM_data.csv')
jnj = pd.read_csv('JNJ_data.csv')
ko = pd.read_csv('KO_data.csv')
mcd = pd.read_csv('MCD_data.csv')
mmm = pd.read_csv('MMM_data.csv')
mrk = pd.read_csv('MRK_data.csv')
msft = pd.read_csv('MSFT_data.csv')
nke = pd.read_csv('NKE_data.csv')
pg = pd.read_csv('PG_data.csv')
trv = pd.read_csv('TRV_data.csv')
unh = pd.read_csv('UNH_data.csv')
v = pd.read_csv('V_data.csv')
vz = pd.read_csv('VZ_data.csv')
wba = pd.read_csv('WBA_data.csv')
wmt = pd.read_csv('WMT_data.csv')
# %%
portfolio = pd.DataFrame()
portfolio['Date'] = aapl['date']
for stock in ticker_list:
    listing = pd.read_csv(f'{stock}_data.csv')
    portfolio[stock] = listing['adjclose']
portfolio.to_csv('portfolio.csv')
# %%
