# %%
from bs4 import BeautifulSoup
import pandas as pd

# %%
tickers = pd.read_csv('ticker_names.csv')
tickers = tickers['Symbol'].to_list()
tickers = [x.lower() for x in tickers]
# %%
for ticker in tickers:
    url = f'https://stockanalysis.com/stocks/{ticker}/financials/?p=quarterly'
    response = requests.get(url)
    soup = BeautifulSoup(response.content, 'html.parser')
    headers = [th.text.strip() for th in soup.select('thead th')]
    rows = soup.select('tbody tr')
    table_data = []
    for row in rows:
        table_row = []
        for td in row.select('td'):
            table_row.append(td.text.strip())
        table_data.append(table_row)
    df = pd.DataFrame(table_data, columns=headers)
    df.to_csv(f'{ticker}_financial_sheet.csv', index=False)
    
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
