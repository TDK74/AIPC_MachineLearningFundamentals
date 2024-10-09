import pandas as pd
import numpy as np


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
