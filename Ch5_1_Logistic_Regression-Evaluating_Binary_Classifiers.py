import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression as LogR


admissions = pd.read_csv('admissions.csv')
model = LogR()
model.fit(admissions[["gpa"]], admissions["admit"])

labels = model.predict(admissions[["gpa"]])
admissions["predicted_label"] = labels
print(admissions["predicted_label"].value_counts())
print(admissions.head())
