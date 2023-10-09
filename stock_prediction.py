#%%
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import LSTM, Dense, Input, concatenate
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
from sklearn.svm import SVR
from sklearn.linear_model import Lasso, Ridge
import statsmodels.api as sm
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
import seaborn as sns
from statsmodels.stats.outliers_influence import variance_inflation_factor

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
for stock in stocks:
    stock['date'] = pd.to_datetime(stock['date'])
    stock.set_index('date', inplace=True)
# %%
features = [
    'open', 'high', 'low', 'close', 'volume', 'adjclose', 'DayOverDayReturn', 'Adj_DayOverDayReturn',
    'SMA50', 'SMA200', '30day_Volatility', 'Stochastic_K',
    'Stochastic_D'
]


#%%
#
#
#
#
aapl = aapl[features]
#%%
aapl['target'] = aapl['close'].shift(-1)  
aapl = aapl.dropna()
#%%
plt.figure(figsize=(20, 10))
correlation_matrix = aapl.corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.show()
#%%
vif = pd.DataFrame()
vif["Feature"] = aapl.columns
vif["VIF"] = [variance_inflation_factor(X.values, i) for i in range(aapl.shape[1])]
print(vif)
#%%
scaler = StandardScaler()
aapl = pd.DataFrame(scaler.fit_transform(aapl.iloc[:, 1:20]), columns=aapl.iloc[:, 1:20].columns)


#%%
X = aapl.iloc[:, 1:14]
y = aapl['target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=False)
#%%
aapl = aapl.fillna(0)
#%%
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
future_predictions = model.predict(X_test)
#%%
plt.figure(figsize=(12, 6))
plt.plot(aapl.index[-len(y_test):], y_test, label='Actual Prices', color='b')
plt.plot(aapl.index[-len(y_test):], future_predictions, label='Predicted Prices', color='r')
plt.xlabel('Time')
plt.ylabel('aapl Price')
plt.legend()
plt.title('Actual vs. Predicted AAPL Stock Prices')
plt.show()

mae = mean_absolute_error(y_test, future_predictions)
print(f'Mean Absolute Error (MAE) for {aapl.iloc[1, 0]} aapl: {mae:.2f}')

#%%
svm_model = SVR(kernel='linear')
svm_model.fit(X_train, y_train)
svm_predictions = svm_model.predict(X_test)
svm_mae = mean_absolute_error(y_test, svm_predictions)
print(f'Mean Absolute Error (MAE) for SVM Regression: {svm_mae:.2f}')
svm_r2 = r2_score(y_test, svm_predictions)


print(f'Mean Absolute Error (MAE) for SVM Regression: {svm_mae:.2f}')
print(f'R-squared (R^2) for SVM Regression: {svm_r2:.2f}')


if svm_model.kernel == 'linear':
    coef = svm_model.coef_[0]
    intercept = svm_model.intercept_[0]
    print("\nSVM Regression Model Coefficients:")
    for i, c in enumerate(coef):
        print(f'Coefficient for Feature{i+1}: {c:.4f}')
    print(f'Intercept: {intercept:.4f}')
#%%
plt.figure(figsize=(12, 6))
plt.plot(aapl.index[-len(y_test):], y_test, label='Actual Prices', color='b')
plt.plot(aapl.index[-len(y_test):], svm_predictions, label='Predicted Prices (SVM)', color='r')
plt.xlabel('Time')
plt.ylabel('aapl Price')
plt.legend()
plt.title('Actual vs. Predicted AAPL Stock Prices (SVM)')
plt.show()


# %%
lasso_model = Lasso(alpha=0.01)
lasso_model.fit(X_train, y_train)
lasso_predictions = lasso_model.predict(X_test)

mae = mean_absolute_error(y_test, lasso_predictions)
print(f'Mean Absolute Error (MAE) for AAPL: {mae:.2f}')

plt.figure(figsize=(12, 6))
plt.plot(aapl.index[-len(y_test):], y_test, label='Actual Prices', color='b')
plt.plot(aapl.index[-len(y_test):], lasso_predictions, label='Predicted Prices (Lasso)', color='r')
plt.xlabel('Time')
plt.ylabel('AAPL Price')
plt.legend()
plt.title('Actual vs. Predicted AAPL Stock Prices (Lasso Regression)')
plt.show()
# %%
ridge_model = Ridge(alpha=100) 
ridge_model.fit(X_train, y_train)
ridge_predictions = ridge_model.predict(X_test)

mae = mean_absolute_error(y_test, ridge_predictions)
print(f'Mean Absolute Error (MAE) for AAPL: {mae:.2f}')
plt.figure(figsize=(12, 6))
plt.plot(aapl.index[-len(y_test):], y_test, label='Actual Prices', color='b')
plt.plot(aapl.index[-len(y_test):], ridge_predictions, label='Predicted Prices (Ridge)', color='r')
plt.xlabel('Time')
plt.ylabel('AAPL Price')
plt.legend()
plt.title('Actual vs. Predicted AAPL Stock Prices (Ridge Regression)')
plt.show()

# %%


# %%
