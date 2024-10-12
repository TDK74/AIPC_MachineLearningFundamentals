import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsRegressor as KNR
from sklearn.metrics import mean_squared_error as MSE


dc_listings = pd.read_csv("dc_airbnb.csv")
stripped_commas = dc_listings['price'].str.replace(',', '')
stripped_dollars = stripped_commas.str.replace('$', '')
dc_listings['price'] = stripped_dollars.astype('float')
shuffled_index = np.random.permutation(dc_listings.index)
dc_listings = dc_listings.reindex(shuffled_index)

# splitting dataset 50/50 for holdout validation
split_one = dc_listings.iloc[0 : 1862].copy()
split_two = dc_listings.iloc[1862 : ].copy()

# Holdout Validation
train_one = split_one
test_one = split_two
train_two = split_two
test_two = split_one

# First half
model = KNR()
model.fit(train_one[['accommodates']], train_one['price'])
test_one['predicted_price'] = model.predict(test_one[['accommodates']])
iteration_one_rmse = MSE(test_one['price'], test_one['predicted_price']) ** 0.5

# Second half
model.fit(train_two[['accommodates']], train_two['price'])
test_two['predicted_price'] = model.predict(test_two[['accommodates']])
iteration_two_rmse = MSE(test_two['price'], test_two['predicted_price']) ** 0.5

# Average RMSError
avg_rmse = np.mean([iteration_two_rmse, iteration_one_rmse])

print(f"Iteration one RMSE: {iteration_one_rmse}")
print(f"Iteration two RMSE: {iteration_two_rmse}")
print(f"Average RMSE: {avg_rmse}")

# K-Fold Cross Validation
dc_listings.loc[dc_listings.index[0 : 745], "fold"] = 1
dc_listings.loc[dc_listings.index[745 : 1490], "fold"] = 2
dc_listings.loc[dc_listings.index[1490 : 2234], "fold"] = 3
dc_listings.loc[dc_listings.index[2234 : 2978], "fold"] = 4
dc_listings.loc[dc_listings.index[2978 : 3723], "fold"] = 5

print(dc_listings["fold"].value_counts())
print("\n Num of missing values: ", dc_listings["fold"].isnull().sum())
