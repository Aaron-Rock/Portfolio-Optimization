#%%
import xgboost as xgb
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant
from sklearn.decomposition import PCA
# %%
data = pd.read_csv('model_files/aapl_lstm.csv')
#%%
data.drop('date', axis=1, inplace=True)
data.drop('close', axis=1, inplace=True)


#%%
X = data.drop(columns='adjclose')
y = data['adjclose']  

#%%
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True)
#%%
dtrain = xgb.DMatrix(X_train, label=y_train)
dtest = xgb.DMatrix(X_test, label=y_test)

params = {
    'max_depth': 5,
    'eta': 0.1,
    'objective': 'reg:squarederror',
    'eval_metric': 'rmse'
}
#%%
num_round = 100
bst = xgb.train(params, dtrain, num_round)
xgb.plot_importance(bst,max_num_features=25)
plt.show()

# %%
bst = xgb.train(params, dtrain, num_round)

# Getting the feature importance
feature_importance = bst.get_score(importance_type='weight')

# Sorting features by their importance scores
sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)

# Extracting only the feature names from the top 25 features
top_25_feature_names = [feature for feature, _ in sorted_features[:10]]

# Printing the list of feature names
print("Top 25 Features:")
print(top_25_feature_names)
# %%
['DayOverDayReturn', 'low', 'high', 'open', 'Stochastic_K', 'volume', 'BollingerUpper', 'EMA', 'Stochastic_D', 'BollingerLower']

# %%
