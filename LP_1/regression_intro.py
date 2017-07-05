import pandas as pd
import quandl, math
import numpy as np
from sklearn import preprocessing, cross_validation, svm
from sklearn.linear_model import LinearRegression

quandl.ApiConfig.api_key = 'YN8WTw6K5y_1zjry3eJu'
df = quandl.get_table("WIKI/PRICES")

df = df[['adj_open', 'adj_high', 'adj_low', 'adj_close', 'adj_volume']]
df['HL_PCT'] = (df['adj_high'] - df['adj_close']) / df['adj_close'] * 100.0
df['PCT_change'] = (df['adj_close'] - df['adj_open']) / df['adj_open'] * 100.0

df = df[['adj_close', 'HL_PCT', 'PCT_change', 'adj_volume']]

forecast_col = 'adj_close'

# make all na as outliers
df.fillna(-99999, inplace=True)

# data prediction length (0.1 = 10%, 0.2 = 20% and so on ..)
forecast_out = int(math.ceil(0.01*len(df)))
print forecast_out

# label >> shifting the forecast column up by forecast out length

df['label'] = df[forecast_col].shift(-forecast_out)
df.dropna(inplace=True)

#feature >> all except label
x = np.array(df.drop(['label'], 1))

# label
y = np.array(df['label'])

x = preprocessing.scale(x)
y = np.array(df['label'])

x_train, x_test, y_train, y_test = cross_validation.train_test_split(x, y, test_size=0.2)

clf = LinearRegression(n_jobs=-1)
clf.fit(x_train, y_train)

accuracy = clf.score(x_test, y_test)

print accuracy