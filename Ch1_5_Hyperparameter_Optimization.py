import pandas as pd
import matplotlib.pyplot as plt
#import numpy as np
#from scipy.spatial import distance as dst
from sklearn.neighbors import KNeighborsRegressor as KNR
from sklearn.metrics import mean_squared_error as MSE
# not in the sample code but might be useful
from sklearn.metrics import root_mean_squared_error as RMSE


# Hyperparameter Optimization
train_df = pd.read_csv('dc_airbnb_train.csv')
test_df = pd.read_csv('dc_airbnb_test.csv')
features = ['accommodates', 'bedrooms', 'bathrooms', 'number_of_reviews']
hyper_params = [1, 2, 3, 4, 5]
mse_values = list()
# not in the sample code but might be useful
rmse_values = list()

for hp in hyper_params:
    knn = KNR(n_neighbors=hp, algorithm='brute')
    knn.fit(train_df[features], train_df['price'])
    predictions = knn.predict(test_df[features])
    mse = MSE(test_df['price'], predictions)
    mse_values.append(mse)
    # not in the sample code but might be useful
    print(f"k = {hp} MSE: {mse}")
    rmse = RMSE(test_df['price'], predictions)
    rmse_values.append(rmse)
    print(f"k = {hp} RMSE: {rmse}")

print(f"List of MSE values - k from 1 to 5: {mse_values}")
# not in the sample code but might be useful
print(f"List of RMSE values - k from 1 to 5: {rmse_values}")

# Expanding Grid Search
hyper_params_20 = [x for x in range(1, 21)]
mse_values_20 = list()
# not in the sample code but might be useful
rmse_values_20 = list()

for hp in hyper_params_20:
    knn20 = KNR(n_neighbors=hp, algorithm='brute')
    knn20.fit(train_df[features], train_df['price'])
    predictions_20 = knn20.predict(test_df[features])
    mse_20 = MSE(test_df['price'], predictions_20)
    mse_values_20.append(mse_20)
    # not in the sample code but might be useful
    print(f"k = {hp} MSE20: {mse_20}")
    rmse_20 = RMSE(test_df['price'], predictions_20)
    rmse_values_20.append(rmse_20)
    print(f"k = {hp} RMSE20: {rmse_20}")

print(f"List of MSE values - k from 1 to 20: {mse_values_20}")
# not in the sample code but might be useful
print(f"List of RMSE values - k from 1 to 20: {rmse_values_20}")

# Visualizing Hyperparameter Values with matplotlib.pyplot
plt.scatter(hyper_params_20, mse_values_20)
plt.show()
# not in the sample code but might be useful
plt.scatter(hyper_params_20, rmse_values_20)
plt.show()
