import pandas as pd


"""
MORE INFO AND CREDITS:
Data Sets and Stories
Dean De Cock
Ames, Iowa: Alternative to the Boston Housing Data as an End of Semester Regression Project
http://jse.amstat.org/v19n3/decock.pdf
"""

# dataset = pd.read_csv("http://jse.amstat.org/v19n3/decock/AmesHousing.txt", sep='\t')
# print(dataset.shape)
# print(dataset.columns)

data = pd.read_csv('AmesHousing.txt', delimiter="\t")
# print(data.shape)
# print(data.columns)
train = data[0:1460]
test = data[1460:]

# Solution code
numerical_train = train.select_dtypes(include=['int', 'float'])
numerical_train = numerical_train.drop(['PID', 'Year Built', 'Year Remod/Add',
                                'Garage Yr Blt', 'Mo Sold', 'Yr Sold'], axis=1)
null_series = numerical_train.isnull().sum()
full_cols_series = null_series[null_series == 0]
print(full_cols_series)
