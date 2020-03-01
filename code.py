# --------------
# Import Libraries
import os
# import warnings.filterwarnings('ignore')
import pandas as pd
import numpy as np
import warnings

# Code starts here
df = pd.read_csv(path)

df.columns = map(str.lower, df.columns)
# df.columns = map(str.lower, df.columns)
df.columns = df.columns.str.replace(' ','_')
print(df.head())
# df = df.fillna('np.nan')
# df[df.isna()]
# Code ends here


# --------------
from sklearn.model_selection import train_test_split
df.set_index(keys='serial_number',inplace=True,drop=True)


# Code starts
df['established_date'] = pd.to_datetime(df['established_date'])
df['acquired_date'] = pd.to_datetime(df['acquired_date'])
X = df.iloc[:,:-1]
y =df['2016_deposits']
X_train,X_val,y_train,y_val = train_test_split(X,y,test_size=.25,random_state=3)
# Code ends here


# --------------
# time_col = X_train.select_dtypes(exclude=[np.number,'O']).columns
time_col = ['established_date', 'acquired_date']

# Code starts here
# for col in time_col:
for df in [X_train,X_val]:
    for col in time_col:
        new_col_name = 'since_'+col
        df[new_col_name] = pd.datetime.now() - df[col]
        df[new_col_name] = df[new_col_name].apply(lambda x:float(x.days)/365)
        df.drop(col,axis=1,inplace=True)


# Code ends here


# --------------
from sklearn.preprocessing import LabelEncoder
cat = X_train.select_dtypes(include='O').columns.tolist()

# Code starts here
X_train.fillna(0, inplace=True)
X_val.fillna(0, inplace=True)

le = LabelEncoder()
for df in [X_train, X_val]:
    for col in cat:
        df[col] = le.fit_transform(df[col])

X_train_temp = pd.get_dummies(data = X_train, columns = cat)
X_val_temp = pd.get_dummies(data = X_val, columns = cat)

print(X_train_temp.head())

# Code ends here


# --------------
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error

# Code starts here
dt=DecisionTreeRegressor(random_state=5)
dt.fit(X_train,y_train)

accuracy = dt.score(X_val, y_val)

y_pred = dt.predict(X_val)
rmse = np.sqrt(mean_squared_error(y_pred, y_val))



# --------------
from xgboost import XGBRegressor


# Code starts here
xgb=XGBRegressor(max_depth=50,learning_rate=0.83,n_estimators=100)
xgb.fit(X_train,y_train)
accuracy=xgb.score(X_val, y_val)
y_pred=xgb.predict(X_val)
rmse=np.sqrt(mean_squared_error(y_pred, y_val))


# Code ends here


