import pandas as pd


"""
Citation:
R. Quinlan. "Auto MPG," UCI Machine Learning Repository, 1993.
[Online].
Available: https://doi.org/10.24432/C5859H.
"""

# Select the names of the columns from auto-mpg.names
columns = ['mpg', 'cylinders', 'displacement', 'horsepower', 'weight',
           'acceleration', 'model year', 'origin', 'car name']
#cars = pd.read_csv("auto-mpg.data", names=columns, delim_whitespace=True)
# delim_whitespace= is deprecated and will be removed in a future version
cars = pd.read_csv("auto-mpg.data", names=columns, sep="\s+")
print(cars.head())

# Origin is: 1. North America, 2. Europe, 3. Asia
unique_regions = cars["origin"].unique()
#unique_regions = sorted(cars["origin"].unique()) # sorted unique regions
print(f"Origin is:\n {unique_regions}")
