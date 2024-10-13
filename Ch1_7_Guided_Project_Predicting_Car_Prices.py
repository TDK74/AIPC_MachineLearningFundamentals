import pandas as pd
import numpy as np


# dataset is used from here: https://archive.ics.uci.edu/dataset/10/automobile and all credits go to
# Creators - Jeffrey Schlimmer

# Introduction To The Dataset
pd.options.display.max_columns = 99

cols = ['symboling', 'normalized-losses', 'make', 'fuel-type', 'aspiration', 'num-of-doors', 'body-style',
        'drive-wheels', 'engine-location', 'wheel-base', 'length', 'width', 'height', 'curb-weight',
        'engine-type', 'num-of-cylinders', 'engine-size', 'fuel-system', 'bore', 'stroke', 'compression-rate',
        'horsepower', 'peak-rpm', 'city-mpg', 'highway-mpg', 'price']
cars = pd.read_csv('imports-85.data', names=cols)

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
# `price` is the column to be predicted -> removing any rows with missing `price` values.
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
