import pandas as pd
import numpy as np
from scipy.spatial import distance
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error
from sklearn.metrics import root_mean_squared_error

np.random.seed(1)

dc_listings = pd.read_csv('dc_airbnb.csv')
dc_listings = dc_listings.loc[np.random.permutation(len(dc_listings))]
stripped_commas = dc_listings['price'].str.replace(',', '')
stripped_dollars = stripped_commas.str.replace('$', '')
dc_listings['price'] = stripped_dollars.astype('float')
print(dc_listings.info())

# Removing Features from the DataFrame
drop_columns = ['room_type', 'city', 'state', 'latitude', 'longitude', 'zipcode',
                'host_response_rate', 'host_acceptance_rate', 'host_listings_count']
dc_listings = dc_listings.drop(drop_columns, axis=1)
print(dc_listings.isnull().sum())

# Handling Missing Values
dc_listings = dc_listings.drop(['cleaning_fee', 'security_deposit'], axis=1)
dc_listings = dc_listings.dropna(axis=0)
print(dc_listings.isnull().sum())

# Normalize Columns
# Subtract each value in the column by the mean.
first_transform = dc_listings['maximum_nights'] - dc_listings['maximum_nights'].mean()
# Divide each value in the column by the standard deviation
normalized_col = first_transform / first_transform.std()
# Also possible :
# normalized_col = first_transform / dc_listings['maximum_nights'].std()
normalized_listings = (dc_listings - dc_listings.mean() / (dc_listings.std()))
normalized_listings['price'] = dc_listings['price']
print(normalized_listings.head(3))

# Euclidean Distance for Multivariate Case
# a simple example
first_list = [-0.596544, -0.439151]
second_list = [-0.596544, 0.412923]
dist = distance.euclidean(first_list, second_list)
print(f"Euclidean distance: {dist}")

# actual example
first_listing = normalized_listings.iloc[0][['accommodates', 'bathrooms']]
fifth_listing = normalized_listings.iloc[4][['accommodates', 'bathrooms']]
first_fifth_distance = distance.euclidean(first_listing, fifth_listing)
print(f"First fifth distance: {first_fifth_distance}")

# Introduction to Scikit-learn - Fitting a Model and Making Predictions.
# Split the full dataser into train and test sets.
train_df = normalized_listings.iloc[0: 2792]
test_df = normalized_listings.iloc[2792 : ]

# Matrix-like object, containing just the 2 columns of interest from training set.
train_features = train_df[['accommodates', 'bathrooms']]

# List-like object, containing just the target column 'price'.
train_target = train_df['price']

# Pass everything into the fit method.
knn = KNeighborsRegressor(algorithm='brute')
knn.fit(train_features, train_target)
predictions_brute = knn.predict(test_df[['accommodates', 'bathrooms']])
print(f"Length of predictions 1st time: {len(predictions_brute)}")

# ONE MORE WAY.
train_columns = ['accommodates', 'bathrooms']

# Instantiate ML model.
knn = KNeighborsRegressor(n_neighbors=5, algorithm='brute')

# Fit the model to data.
knn.fit(train_df[train_columns], train_df['price'])

# Use model to make predictions.
predictions = knn.predict(test_df[train_columns])
print(f"Length of predictions 2nd time: {len(predictions)}")

# Calculating MSE and RMSE using Scikit-Learn
#knn = KNeighborsRegressor(n_neighbors=5, algorithm='brute', metric='euclidean') # same results as w/o metric='euclidean'

two_features_mse = mean_squared_error(test_df['price'], predictions)
two_features_rmse1 = two_features_mse ** (1 / 2)
two_features_rmse2 = root_mean_squared_error(test_df['price'], predictions)
print(f"Two features RMSE: {two_features_mse}")
print(f"Two features RMSE1: {two_features_rmse1}")
print(f"Two features RMSE2: {two_features_rmse2}")

# Using More Features - accommodates, bedrooms, bathrooms, number_of_reviews
features = ['accommodates', 'bedrooms', 'bathrooms', 'number_of_reviews']
#knn = KNeighborsRegressor(n_neighbors=5, algorithm='brute')
knn.fit(train_df[features], train_df['price'])
four_predictions = knn.predict(test_df[features])
four_mse = mean_squared_error(test_df['price'], four_predictions)
four_rsme = four_mse ** 0.5
print(f"Four features MSE: {four_mse}")
print(f"Four features RMSE: {four_rsme}")
