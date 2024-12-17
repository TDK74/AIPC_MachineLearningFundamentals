import pandas as pd


"""
MORE INFO AND CREDITS:
Ames, Iowa: Alternative to the Boston Housing Data as an End of Semester Regression Project
Dean De Cock
https://www.tandfonline.com/doi/abs/10.1080/10691898.2011.11889627
"""

data = pd.read_csv('AmesHousing.txt', delimiter="\t")
train = data[0 : 1460]
test = data[1460 : ]

train_null_counts = train.isnull().sum()
print(train_null_counts)
df_no_mv = train[train_null_counts[train_null_counts == 0].index]
print(df_no_mv)
