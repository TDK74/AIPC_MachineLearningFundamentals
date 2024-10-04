import pandas as pd
import numpy as np

dc_listings = pd.read_csv('dc_airbnb.csv')
print(dc_listings.iloc[0])

# K-nearest Neighbours and Euclidean distance
our_acc_values = 3
first_living_space_valus = dc_listings.iloc[0]['accommodates']
first_distance  = np.abs(first_living_space_valus - our_acc_values)
print(first_distance)

# Calculate distance for all observations
new_listings = 3
dc_listings['distance'] = dc_listings['accommodates'].apply(lambda x: np.abs(x - new_listings))
print(dc_listings['distance'].value_counts())
print(dc_listings[dc_listings['distance'] == 0]['accommodates'])

# Randomizing and sorting
np.random.seed(1)
dc_listings = dc_listings.loc[np.random.permutation(len(dc_listings))]
dc_listings = dc_listings.sort_values('distance')
print(dc_listings.iloc[0:10]['price'])

# Average price
stripped_commas = dc_listings['price'].str.replace(',', '')
#print(stripped_commas)
stripped_dollars = stripped_commas.str.replace('$', '')
#print(stripped_dollars)
dc_listings['price'] = stripped_dollars.astype('float')
#print(dc_listings['price'])
mean_price= dc_listings.iloc[0:5]['price'].mean()
print(mean_price)
