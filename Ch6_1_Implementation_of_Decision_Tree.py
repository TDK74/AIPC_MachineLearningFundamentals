import numpy as np
import pandas as pd
import matplotlib.pyplot as mtp
from sklearn.model_selection import train_test_split as tts
from sklearn.preprocessing import StandardScaler as StScl


# importing dataset
data_set = pd.read_csv("user_data.csv")

# extracting indepedent and dependent variable
x = data_set.iloc[:, [2, 3]].values
y = data_set.iloc[:, 4].values

# splitting the dataset into train and test set
x_train, x_test, y_train, y_test = tts(x, y, test_size=0.25, random_state=0)

# feature scaling
st_x = StScl()
x_train = st_x.fit_transform(x_train)
x_test = st_x.transform(x_test)
# print(x_train[:10])
# print('\n')
# print(x_test[:10])