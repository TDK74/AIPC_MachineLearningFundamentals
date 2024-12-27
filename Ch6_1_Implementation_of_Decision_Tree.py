import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split as tts
from sklearn.preprocessing import StandardScaler as StScl
from sklearn.tree import DecisionTreeClassifier as DcsTCls
from sklearn.metrics import confusion_matrix as c_mtx
from matplotlib.colors import ListedColormap as LstClm


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

# fitting Decision Tree Classifier to the training set
classifier = DcsTCls(criterion='entropy', random_state=0)
classifier.fit(x_train, y_train)

# predicting the test set result
y_pred = classifier.predict(x_test)

# creation of Confusion matrix
cm = c_mtx(y_test, y_pred)

# visualizing the training set result
x_set, y_set = x_train, y_train
x1, x2 = np.meshgrid(np.arange(start=x_set[:, 0].min() - 1, stop=x_set[:, 0].max() + 1, step=0.01),
                            np.arange(start=x_set[:, 1].min() - 1, stop=x_set[:, 1].max() + 1, step=0.01))

plt.contourf(x1, x2, classifier.predict(np.array([x1.ravel(), x2.ravel()]).T).reshape(x1.shape),
             alpha=0.75, cmap=LstClm(('purple', 'green')))
plt.xlim(x1.min(), x1.max())
plt.ylim(x2.min(), x2.max())

for i, j in enumerate(np.unique(y_set)):
    plt.scatter(x_set[y_set == j, 0], x_set[y_set == j, 1], c=LstClm(('purple', 'green'))(i), label=j)

plt.title('Decision Tree Classification (Training set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()


