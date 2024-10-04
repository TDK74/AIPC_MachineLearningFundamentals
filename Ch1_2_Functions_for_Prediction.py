import pandas as pd
import numpy as np

#  Brought along the changes we made to the 'dc_listing' Dataframe.
dc_listings = pd.read_csv('dc_airbnb.csv')
stripped_commas = dc_listings['price'.str.replace]