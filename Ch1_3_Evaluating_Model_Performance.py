import pandas as pd
import numpy as np

dc_listings = pd.read_csv("dc_airbnb.csv")

stripped_commas = dc_listings['price'].str.replace(',', '')
stripped_dollars = stripped_commas.str.replace('$', '')
dc_listings['price'] = stripped_dollars.astype('float')

# split the dataset into 2 partitions - the training set and the test set
train_df = dc_listings.iloc[0 : 2792]
#test_df = dc_listings.iloc[2792 : ]
test_df = dc_listings.iloc[2792 : ].copy()  # Explicitly create a copy

# def predict_price(new_listing):
#     # DataFrame.copy() performs a deep copy
#     temp_df = dc_listings.copy()
#     temp_df['distance'] = temp_df['accommodates'].apply(lambda x: np.abs(x - new_listing))
#     temp_df = temp_df.sort_values('distance')
#     nearest_neightbor_prices = temp_df.iloc[0 : 5]['price']
#     predicted_price = nearest_neightbor_prices.mean()

#     return(predicted_price)


def predict_price(new_listing):
    temp_df = train_df.copy()
    temp_df['distance'] = temp_df['accommodates'].apply(lambda x: np.abs(x - new_listing))
    temp_df = temp_df.sort_values('distance')
    nearest_neighbor_price = temp_df.iloc[0: 5]['price']
    predicted_price = nearest_neighbor_price.mean()

    return(predicted_price)


test_df['predicted_price'] = test_df['accommodates'].apply(predict_price)
# Use .loc to avoid SettingWithCopyWarning - didn't work but for the record...
#test_df.loc[2792: , 'predicted_price'] = test_df['accommodates'].apply(predict_price)

# Error metrics for the predictions - mean absolute error.
test_df['error'] = np.absolute(test_df['predicted_price'] - test_df['price'])
mae = test_df['error'].mean()
print(mae)

# Error metrics for the predictions - mean square error.
test_df['square error'] = (test_df['predicted_price'] - test_df['price']) ** 2
mse = test_df['square error'].mean()
print(mse)
