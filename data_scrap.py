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
    df = df.T
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
wba_financial = pd.read_csv('wba_financial_sheet.csv')
v_financial = pd.read_csv('v_financial_sheet.csv')
crm_financial = pd.read_csv('crm_financial_sheet.csv')
cvx_financial = pd.read_csv('cvx_financial_sheet.csv')
pg_financial = pd.read_csv('pg_financial_sheet.csv')
vz_financial = pd.read_csv('vz_financial_sheet.csv')
wmt_financial = pd.read_csv('wmt_financial_sheet.csv')
unh_financial = pd.read_csv('unh_financial_sheet.csv')
trv_financial = pd.read_csv('trv_financial_sheet.csv')
mcd_financial = pd.read_csv('mcd_financial_sheet.csv')
mmm_financial = pd.read_csv('mmm_financial_sheet.csv')
nke_financial = pd.read_csv('nke_financial_sheet.csv')
mrk_financial = pd.read_csv('mrk_financial_sheet.csv')
msft_financial = pd.read_csv('msft_financial_sheet.csv')
jpm_financial = pd.read_csv('jpm_financial_sheet.csv')
ko_financial = pd.read_csv('ko_financial_sheet.csv')
jnj_financial = pd.read_csv('jnj_financial_sheet.csv')
gs_financial = pd.read_csv('gs_financial_sheet.csv')
hd_financial = pd.read_csv('hd_financial_sheet.csv')
hon_financial = pd.read_csv('hon_financial_sheet.csv')
ibm_financial = pd.read_csv('ibm_financial_sheet.csv')
intc_financial = pd.read_csv('intc_financial_sheet.csv')
dis_financial = pd.read_csv('dis_financial_sheet.csv')
cat_financial = pd.read_csv('cat_financial_sheet.csv')
csco_financial = pd.read_csv('csco_financial_sheet.csv')
axp_financial = pd.read_csv('axp_financial_sheet.csv')
ba_financial = pd.read_csv('ba_financial_sheet.csv')
amgn_financial = pd.read_csv('amgn_financial_sheet.csv')
aapl_financial = pd.read_csv('aapl_financial_sheet.csv')

# %%
dates = [
    "8/31/23", "5/31/23", "2/28/23", "11/30/22", "8/31/22", 
    "5/31/22", "2/28/22", "11/30/21", "8/31/21", "5/31/21", 
    "2/28/21", "11/30/20", "8/31/20", "5/31/20", "2/29/20", 
    "11/30/19", "8/31/19", "5/31/19", "2/28/19", "11/30/18", 
    "8/31/18", "5/31/18", "2/28/18", "11/30/17", "8/31/17", 
    "5/31/17", "2/28/17", "11/30/16", "8/31/16", "5/31/16", 
    "2/29/16", "11/30/15", "8/31/15", "5/31/15", "2/28/15", 
    "11/30/14", "8/31/14", "5/31/14", "2/28/14", "11/30/13"
]

# %%
financial_variables_list = [wba_financial, v_financial, crm_financial, cvx_financial, 
                           pg_financial, vz_financial, wmt_financial, unh_financial, 
                           trv_financial, mcd_financial, mmm_financial, nke_financial, 
                           mrk_financial, msft_financial, jpm_financial, ko_financial, 
                           jnj_financial, gs_financial, hd_financial, hon_financial, 
                           ibm_financial, intc_financial, dis_financial, cat_financial, 
                           csco_financial, axp_financial, ba_financial, amgn_financial, 
                           aapl_financial]

start_date = '07/31/2013'
end_date = '08/01/2023'

for idx, financial_df in enumerate(financial_variables_list):
    financial_df = financial_df.drop(financial_df.tail(1).index)
    new_headers = financial_df.iloc[0]
    financial_df = financial_df[1:]
    financial_df.columns = new_headers
    financial_df.index = dates
    financial_df = financial_df.reset_index().rename(columns={'index': 'Date'})
    financial_df['Date'] = pd.to_datetime(financial_df['Date'])
    financial_df = financial_df.set_index('Date').resample('D').ffill()
    if financial_df.index.min() > pd.Timestamp(start_date):
        earliest_row = financial_df.iloc[0]
        date_range = pd.date_range(start=start_date, end=financial_df.index.min() - pd.Timedelta(days=1))
        backfill_df = pd.DataFrame([earliest_row] * len(date_range), index=date_range, columns=financial_df.columns)
        financial_df = pd.concat([backfill_df, financial_df])
    financial_df = financial_df[start_date:end_date]
    financial_variables_list[idx] = financial_df

wba_financial, v_financial, crm_financial, cvx_financial, pg_financial, vz_financial, wmt_financial, unh_financial, trv_financial, mcd_financial, mmm_financial, nke_financial, mrk_financial, msft_financial, jpm_financial, ko_financial, jnj_financial, gs_financial, hd_financial, hon_financial, ibm_financial, intc_financial, dis_financial, cat_financial, csco_financial, axp_financial, ba_financial, amgn_financial, aapl_financial = financial_variables_list

# %%

