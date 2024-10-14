import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsRegressor as KNR
from sklearn.metrics import mean_squared_error as MSE


# dataset is used from here: https://archive.ics.uci.edu/dataset/10/automobile and all credits go to
# Creators - Jeffrey Schlimmer

# Introduction To The Dataset
pd.options.display.max_columns = 99

columns = ['symboling', 'normalized-losses', 'make', 'fuel-type', 'aspiration', 'num-of-doors', 'body-style',
        'drive-wheels', 'engine-location', 'wheel-base', 'length', 'width', 'height', 'curb-weight',
        'engine-type', 'num-of-cylinders', 'engine-size', 'fuel-system', 'bore', 'stroke', 'compression-rate',
        'horsepower', 'peak-rpm', 'city-mpg', 'highway-mpg', 'price']
cars = pd.read_csv('imports-85.data', names=columns)

cars.head()

# Select only the columns with continuous values from imports-85.names
continuous_values_columns = ['normalized-losses', 'wheel-base', 'length', 'width', 'height', 'curb-weight',
                             'engine-size', 'bore', 'stroke', 'compression-rate', 'horsepower', 'peak-rpm',
                             'city-mpg', 'highway-mpg', 'price']
numeric_cars = cars[continuous_values_columns]

numeric_cars.head()

# Data Cleaning
numeric_cars = numeric_cars.replace('?', np.nan)
numeric_cars.head()
numeric_cars = numeric_cars.astype('float')
numeric_cars.isnull().sum()
# 'price' is the column to be predicted -> removing any rows with missing 'price' values.
numeric_cars = numeric_cars.dropna(subset=['price'])
numeric_cars.isnull().sum()
# Replace missing values in other columns using column means.
numeric_cars = numeric_cars.fillna(numeric_cars.mean())
# Confirm that there are no more missing values.
numeric_cars.isnull().sum()
# Normalize all columnns to range from 0 to 1 except the target column.
price_column = numeric_cars['price']
numeric_cars = (numeric_cars - numeric_cars.min()) / (numeric_cars.max() - numeric_cars.min())
numeric_cars['price'] = price_column

# Univariate Model
# Default value of k hyperparameter
def knn_train_test_def(train_col, target_col, df):
    knn = KNR()
    np.random.seed(1)

    # Randomize order of rows in DataFrame.
    shuffled_index = np.random.permutation(df.index)
    rand_df = df.reindex(shuffled_index)

    # Divide number of rows in half and round.
    last_train_row = int(len(rand_df) / 2)

    # Select the first half, and set as training set.
    # Select the second half, and set as test set.
    train_df = rand_df.iloc[0 : last_train_row]
    test_df = rand_df.iloc[last_train_row : ]

    # Fit a KNN model using default k value.
    knn.fit(train_df[[train_col]], train_df[target_col])

    # Make predictions using model.
    predicted_labels = knn.predict(test_df[[train_col]])

    # Calculate and return RMSE.
    mse_def = MSE(test_df[target_col], predicted_labels)
    rmse_def = np.sqrt(mse_def)

    return rmse_def

rmse_results_def = {}
train_columns_def = numeric_cars.columns.drop('price')

# For each column (minus 'price'), train a model, return RMSE value
# and add to the dictionary 'rmse_results'.
for column in train_columns_def:
    rmse_val_def = knn_train_test_def(column, 'price', numeric_cars)
    rmse_results_def[column] = rmse_val_def

# Create a Series object from the dictionary so
# we can easily view the results, sort, etc.
rmse_results_series = pd.Series(rmse_results_def)
rmse_results_series.sort_values() # for run with option -> 'Run current File in Interactive Window'


# Hyperparameter optimization
def knn_train_test_hpo(train_col, target_col, df):
    np.random.seed(1)

    # Randomize order of rows in Dataframe.
    shuffled_index = np.random.permutation(df.index)
    rand_df = df.reindex(shuffled_index)

    # Divide number of rows in half and round.
    last_train_row = int(len(rand_df) / 2)

    # Select the first half, and set as training set.
    # Select the second half, and set as test set.
    train_df = rand_df.iloc[0 : last_train_row]
    test_df = rand_df.iloc[last_train_row : ]

    k_values = [x for x in range(1, 10, 2)]
    k_rmses = {}

    for k in k_values:
        # Fit model using k nearest neighbors.
        knn = KNR(n_neighbors=k)
        knn.fit(train_df[[train_col]], train_df[[target_col]])

        # Make predictions using model.
        predicted_labels = knn.predict(test_df[[train_col]])

        # Calculate and return RMSE.
        mse_hpo = MSE(test_df[target_col], predicted_labels)
        rmse_hpo = np.sqrt(mse_hpo)

        k_rmses[k] = rmse_hpo

    return k_rmses

k_rmse_results = {}

# For each column (minus 'price'), train a model, return RMSE value
# and add to the dictionary 'rmse_results'.
train_columns_hpo = numeric_cars.columns.drop('price')

for column in train_columns_hpo:
    rmse_val_hpo = knn_train_test_hpo(column, 'price', numeric_cars)
    k_rmse_results[column] = rmse_val_hpo

# for normal run
print(f"RMSE results by hyperparameter k: {k_rmse_results}")

# for run with option -> 'Run current File in Interactive Window'
# (sample code is actually for Jupyter .ipynb)
k_rmse_results
