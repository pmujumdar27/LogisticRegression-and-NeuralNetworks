import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from tqdm import tqdm
from logisticRegression.multiclassLR import MulticlassLR
from metrics import *

from sklearn.datasets import load_digits
from sklearn.preprocessing import MinMaxScaler

normalized = MinMaxScaler(feature_range=(-1,1))

digits_dataset = load_digits()

data = digits_dataset['data']
data = normalized.fit_transform(data)
target = digits_dataset['target']

df = pd.DataFrame(data, columns=digits_dataset['feature_names'])
df['target'] = target

X = df.drop(['target'], axis=1)
y = df['target']

num_classes = len(digits_dataset['target_names'])

MLR = MulticlassLR(num_classes)

# MLR.fit(X, y, 10, n_iter=50, lr=0.1)
MLR.fit_autograd(X, y, 10, n_iter=50, lr=0.1)
MLR.plot_loss_history()

y_hat = MLR.predict(X)

acc = accuracy(y_hat, y)

print("Accuracy: ", acc)

# print(digits_dataset)