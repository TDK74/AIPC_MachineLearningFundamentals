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

