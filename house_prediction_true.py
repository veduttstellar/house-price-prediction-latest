import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.datasets import fetch_california_housing
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor, AdaBoostRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR


house = fetch_california_housing()
# print(house)

data = pd.DataFrame(house.data, columns=house.feature_names)
print(data)
# print(data.shape)


data["Price"] = house.target
#print(data.head())
#print(data.shape)

X = data.drop("Price", axis=1)
y = data["Price"]
# print(X.shape)
# print(y.shape)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
# print(X_train.shape)
# print(X_test.shape）
# print(y_train.shape)
# print(y_test.shape)

model = RandomForestRegressor()


model.fit(X_train, y_train)


y_pred=model.predict(X_test)
mse=mean_squared_error(y_test,y_pred)
print(mse)

