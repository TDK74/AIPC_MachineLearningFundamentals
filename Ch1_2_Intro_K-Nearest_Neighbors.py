import pandas as pd
import numpy as np

dc_listings = pd.read_csv('dc_airbnb.csv')
print(dc_listings.iloc[0])

our_acc_values = 3
first_living_space_valus = dc_listings.iloc[0]['accommodates']
first_distance  = np.abs(first_living_space_valus - our_acc_values)
print(first_distance)

new_listings = 3
dc_listings['distance'] = dc_listings['accommodates'].apply(lambda x: np.abs(x - new_listings))
print(dc_listings['distance'].value_counts())
print(dc_listings[dc_listings['distance'] == 0]['accommodates'])
