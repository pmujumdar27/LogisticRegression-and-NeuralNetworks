import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from tqdm import tqdm

from neuralNetworks.neuralNet import NN
import metrics

from sklearn.datasets import load_digits, load_boston
from sklearn.preprocessing import MinMaxScaler

from sklearn.model_selection import KFold

normalized = MinMaxScaler(feature_range=(-1,1))

digits_dataset = load_digits()
boston_dataset = load_boston()

dataset = 'Boston'

data = digits_dataset['data']

if dataset=='Boston':
    data = load_boston()['data']

data = normalized.fit_transform(data)

# target = digits_dataset['target']
target = boston_dataset['target']

# df = pd.DataFrame(data, columns=digits_dataset['feature_names'])
df = pd.DataFrame(data, columns=boston_dataset['feature_names'])

df['target'] = target

X = df.drop(['target'], axis=1)
y = df['target']

X = np.array(X)
y = np.array(y)

num_classes = len(digits_dataset['target_names'])

kf = KFold(n_splits=3)

for train_index, test_index in kf.split(X, y):
    # Code taken from official documentation of sklearn
    
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

    config = {
        'n_hidden': 2,
        'n_neurons': [8, 8],
        'activations': ['relu', 'relu'],
        'num_classes': None,
        'last_activ': None,
        'n_iter': 200,
        'lr': 0.01
        }

    model = NN(
        n_features=X.shape[1],
        n_hidden=config['n_hidden'],
        n_neurons=config['n_neurons'],
        activations=config['activations'],
        num_classes=config['num_classes'],
        last_activ=config['last_activ']
    )

    model.fit(X_train, y_train, 10, config['n_iter'], lr=config['lr'])
    y_hat = model.predict(X_test)
    
    # acc = metrics.accuracy(pd.Series(y_hat), pd.Series(y_test))
    rmse = metrics.rmse(pd.Series(y_hat), pd.Series(y_test))

    # print("Accuracy: ", acc)
    print("RMSE: ", rmse)

# print(model.weights)