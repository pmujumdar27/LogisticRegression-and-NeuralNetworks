import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer

from logisticRegression.logisticRegression import LogisticRegression
from metrics import *

data = load_breast_cancer()

# print(data['data'])

df = pd.DataFrame(data['data'], columns=data['feature_names'])
y = pd.Series(data['target'])

print(df.head())
print(y)

LR = LogisticRegression()

LR.fit_2class_unreg(df, y, 10)

y_hat = LR.predict_2class(df)

acc = accuracy(y_hat, y)
# prec = precision(y_hat, y)
# rec = recall(y_hat, y)

print("Accuracy: ", acc)
# print("Precision: ", prec)
# print("Recall: ", rec)