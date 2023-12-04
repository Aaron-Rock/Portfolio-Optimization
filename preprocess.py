#%%
import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import adfuller
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
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
stocks = [aapl, amgn, axp, ba, cat, crm, csco, cvx, dis, gs, hd, hon, ibm, intc, jpm, jnj, ko, mcd, mmm, mrk, msft, nke, pg, trv, unh, v, vz, wba, wmt]

percentage_columns = [
    'Revenue Growth (YoY)', 'Net Income Growth', 'EPS Growth', 'Dividend Growth', 
    'Gross Margin', 'Operating Margin', 'Profit Margin', 'Free Cash Flow Margin', 
    'Effective Tax Rate', 'Shares Change', 'EBITDA Margin', 'EBIT Margin'
]

f_columns = ['Revenue', 'Cost of Revenue', 'Gross Profit', 'Selling, General & Admin', 'Research & Development', 'Operating Expenses', 'Operating Income', 'Interest Expense / Income', 'Other Expense / Income', 'Pretax Income', 'Income Tax', 'Net Income', 'Shares Outstanding (Basic)', 'Shares Outstanding (Diluted)', 'Free Cash Flow', 'EBITDA', 'Depreciation & Amortization', 'consumer_goods','EBIT', 'Other Operating Expenses', 'Dividend Per Share']
#%%
for stock in stocks:
    stock.rename(columns={'Unnamed: 0': 'date'}, inplace=True)
    stock['date'] = pd.to_datetime(stock['date'])
    stock.set_index('date', inplace=True)

    for col in percentage_columns:
        if col in stock.columns: 
            # Replace non-numeric values with NaN
            stock[col] = pd.to_numeric(stock[col].str.replace('%', ''), errors='coerce') / 100
    for col_f in f_columns:
        if col_f in stock.columns and stock[col_f].dtype == 'object':
            stock[col_f] = pd.to_numeric(stock[col_f].str.replace(',', ''), errors='coerce')

# %%
def adf_test(series):
    result = adfuller(series, autolag='AIC')
    print(f'ADF Statistic: {result[0]}')
    print(f'p-value: {result[1]}')
    for key, value in result[4].items():
        print(f'Critical Values ({key}): {value}')
        
#%%
stocks_dict = {
    'aapl': aapl, 
    'amgn': amgn, 
    'axp': axp, 
    'ba': ba, 
    'cat': cat, 
    'crm': crm, 
    'csco': csco, 
    'cvx': cvx, 
    'dis': dis, 
    'gs': gs, 
    'hd': hd, 
    'hon': hon, 
    'ibm': ibm, 
    'intc': intc, 
    'jpm': jpm, 
    'jnj': jnj, 
    'ko': ko, 
    'mcd': mcd, 
    'mmm': mmm, 
    'mrk': mrk, 
    'msft': msft, 
    'nke': nke, 
    'pg': pg, 
    'trv': trv, 
    'unh': unh, 
    'v': v, 
    'vz': vz, 
    'wba': wba, 
    'wmt': wmt
}
#%%
for key, stock in stocks_dict.items():
    print()
    print('------------------------------------------')
    print(f"Running ADF Test for {key}")
    if 'adjclose' in stock.columns:
        adf_test(stock['adjclose'])
    else:
        print(f"'adjclose' column not found in {key}")
# %%

# %%
aapl.to_csv('apple_lstm.csv')
ba.to_csv('ba_lstm.csv')
mrk.to_csv('mrk_lstm.csv')
#%%
for key, stock in stocks_dict.items():
    stock.to_csv(f'model_files/{key}_lstm.csv')
# %%
