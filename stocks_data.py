#%%
import pandas as pd
import numpy as np
from yahooquery import Ticker
import yfinance as yf
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
gdp = pd.read_csv('GDP.csv')
gdp['DATE'] = pd.to_datetime(gdp['DATE'])
filtered_gdp = gdp[gdp['DATE'] > '2013-04-01']
print(filtered_gdp)
# %%
data = {
    'DATE': ['2013-07-01', '2013-10-01', '2014-01-01', '2014-04-01', '2014-07-01', '2014-10-01', '2015-01-01', '2015-04-01', 
            '2015-07-01', '2015-10-01', '2016-01-01', '2016-04-01', '2016-07-01', '2016-10-01', '2017-01-01', 
            '2017-04-01', '2017-07-01', '2017-10-01', '2018-01-01', '2018-04-01', '2018-07-01', '2018-10-01', 
            '2019-01-01', '2019-04-01', '2019-07-01', '2019-10-01', '2020-01-01', '2020-04-01', '2020-07-01', 
            '2020-10-01', '2021-01-01', '2021-04-01', '2021-07-01', '2021-10-01', '2022-01-01', '2022-04-01', 
            '2022-07-01', '2022-10-01', '2023-01-01', '2023-04-01', '2023-07-01', '2023-10-01'],
    'GDP': [16953.838, 17192.019, 17197.738, 17518.508, 17804.228, 17912.079, 18063.529, 18279.784, 18401.626, 18435.137, 
            18525.933, 18711.702, 18892.639, 19089.379, 19280.084, 19438.643, 19692.595, 20037.088, 20328.553, 
            20580.912, 20798.730, 20917.867, 21104.133, 21384.775, 21694.282, 21902.390, 21706.513, 19913.143, 
            21647.640, 22024.502, 22600.185, 23292.362, 23828.973, 24654.603, 25029.116, 25544.273, 25994.639, 
            26408.405, 26813.601, 27063.012, 27171.264, 27279.949]
}
gdp = pd.DataFrame(data)
gdp['DATE'] = pd.to_datetime(gdp['DATE'])
gdp = gdp.rename(columns={'DATE': 'date'})
gdp_daily = gdp.set_index('date').resample('D').ffill().reset_index()

# %%
gdp_daily['date'] = pd.to_datetime(gdp_daily['date'])
aapl['date'] = pd.to_datetime(aapl['date'])
merged_data = aapl.merge(gdp_daily, on='date', how='left')
# %%
cpi = pd.read_csv('CPIAUCSL.csv')
filtered_cpi = cpi[cpi['DATE'] >= '2013-07-01']
filtered_cpi['DATE'] = pd.to_datetime(filtered_cpi['DATE'])
filtered_cpi = filtered_cpi.rename(columns={'DATE': 'date'})
# %%
cpi_daily = filtered_cpi.set_index('date').resample('D').ffill().reset_index()
# %%
merged_data_aapl = merged_data.merge(cpi_daily, on='date', how='left')
# %%
unemp = pd.read_csv('UNRATE.csv')
filtered_unemp = unemp[unemp['DATE'] >= '2013-07-01']
filtered_unemp['DATE'] = pd.to_datetime(filtered_unemp['DATE'])
filtered_unemp = filtered_unemp.rename(columns={'DATE': 'date'})
# %%
unemp_daily = filtered_unemp.set_index('date').resample('D').ffill().reset_index()
# %%
merged_data_aapl_un = merged_data_aapl.merge(unemp_daily, on='date', how='left')
# %%
gdp_column = merged_data_aapl_un['GDP']
unemp_column = merged_data_aapl_un['UNRATE']
cpi_column = merged_data_aapl_un['CPIAUCSL']
# %%
stocks = [aapl, amgn, axp, ba, cat, crm, csco, cvx, dis, gs, hd, hon, ibm, intc, jpm, jnj, ko, mcd, mmm, mrk, msft, nke, pg, trv, unh, v, vz, wba, wmt]

cols_to_add = ['GDP', 'UNRATE', 'CPIAUCSL']

for idx, stock in enumerate(stocks):
    stock['date'] = pd.to_datetime(stock['date'])
    updated_stock = stock.merge(merged_data_aapl_un[['date'] + cols_to_add], on='date', how='left')
    stocks[idx] = updated_stock
    

aapl, amgn, axp, ba, cat, crm, csco, cvx, dis, gs, hd, hon, ibm, intc, jpm, jnj, ko, mcd, mmm, mrk, msft, nke, pg, trv, unh, v, vz, wba, wmt = stocks

# %%
def fetch_data(ticker_symbol, start_date, end_date):
    ticker = Ticker(ticker_symbol)
    data = ticker.history(start=start_date, end=end_date, interval='1d')
    return data
nasdaq_data = fetch_data("^IXIC", "2013-07-31", "2023-08-02")
nasdaq_data = nasdaq_data.reset_index()
nasdaq_close = nasdaq_data['close']

sp500_data = fetch_data("^GSPC", "2013-07-31", "2023-08-02")
sp500_data = sp500_data.reset_index()
sp500_close = sp500_data['close']

world_data = fetch_data('URTH', "2013-07-31", "2023-08-02")
world_data = world_data.reset_index()
world_close = world_data['close']


# %%
for idx, stock_df in enumerate(stocks):
    stock_df = stock_df.reset_index()
    sp500_data['date'] = pd.to_datetime(sp500_data['date'])
    nasdaq_data['date'] = pd.to_datetime(nasdaq_data['date'])
    world_data['date'] = pd.to_datetime(world_data['date'])
    stock_df['date'] = pd.to_datetime(stock_df['date'])
    stock_df['sp500'] = sp500_close
    stock_df['nasdaq'] = nasdaq_close
for stock in stocks:
    stock['world_index'] = world_close

#%%


#%%
symbs = ['aapl', 'amgn', 'axp', 'ba', 'cat', 'crm', 'csco', 'cvx', 'dis', 'gs', 'hd', 'hon', 'ibm', 'intc', 'jpm', 'jnj', 'ko', 'mcd', 'mmm', 'mrk', 'msft', 'nke', 'pg', 'trv', 'unh', 'v', 'vz', 'wba', 'wmt']
for symb in symbs:
    eval(symb).to_csv(f'stocks/{symb}_data.csv')
#%%
# Sector and Industry Data
consumer_goods = pd.read_csv('consumer_goods.csv')
consumer_services = pd.read_excel('consumer_services.xlsx')
technology = pd.read_excel('technology_inx.xlsx')
financials = pd.read_excel('financials_inx.xlsx')
#%%
for stock in stocks:
    stock['consumer_goods'] = consumer_goods['Close']
    stock['consumer_services'] = consumer_services['price']
    stock['technology'] = technology['price']
    stock['financials'] = financials['price']


#%%
# Apple
apple_revenue = pd.read_excel('aaple_revenue.xlsx')
apple_revenue = apple_revenue.rename(columns={'Date': 'date'})
apple_revenue = apple_revenue.set_index('date').resample('D').ffill().reset_index()
apple_revenue['date'] = pd.to_datetime(apple_revenue['date'])
apple_revenue = apple_revenue[(apple_revenue['date'] >= '2013-07-31') & (apple_revenue['date'] <= '2023-08-02')]
aapl = aapl.merge(apple_revenue, on='date', how='left')





# %%
