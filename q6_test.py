import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from tqdm import tqdm
from pprint import pprint

from neuralNetworks.neuralNet import NN
import metrics

from sklearn.datasets import load_digits, load_boston
from sklearn.preprocessing import MinMaxScaler

from sklearn.model_selection import KFold

normalized = MinMaxScaler(feature_range=(-1,1))

digits_dataset = load_digits()
boston_dataset = load_boston()

def test_digits(k):
    data = digits_dataset['data']
    data = normalized.fit_transform(data)
    target = digits_dataset['target']
    df = pd.DataFrame(data, columns=digits_dataset['feature_names'])
    df['target'] = target
    X = df.drop(['target'], axis=1)
    y = df['target']
    X = np.array(X)
    y = np.array(y)
    num_classes = len(digits_dataset['target_names'])

    best_acc = 0
    accuracies = []

    kf = KFold(n_splits=k)

    config = {
        'n_hidden': 0,
        'n_neurons': [],
        'activations': [],
        'num_classes': num_classes,
        'last_activ': 'softmax',
        'n_iter': 120,
        'lr': 0.02
        }

    print("Current Config: ")
    pprint(config)
    print()

    for train_index, test_index in kf.split(X, y):
        # Code taken from official documentation of sklearn

        print()
        
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]        

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

        acc = metrics.accuracy(pd.Series(y_hat), pd.Series(y_test))
        # print("Accuracy: ", acc)
        accuracies.append(acc)
        best_acc = max(best_acc, acc)

    print()
    print("The accuracies for the {} folds: ".format(k))
    print(accuracies)
    print()
    print("Best accuracy: ", best_acc)
    print()

def test_boston(k):
    data = boston_dataset['data']
    data = normalized.fit_transform(data)
    target = boston_dataset['target']
    df = pd.DataFrame(data, columns=boston_dataset['feature_names'])
    df['target'] = target
    X = df.drop(['target'], axis=1)
    y = df['target']
    X = np.array(X)
    y = np.array(y)
    num_classes = None

    kf = KFold(n_splits=k)

    config = {
        'n_hidden': 2,
        'n_neurons': [8, 8],
        'activations': ['relu', 'relu'],
        'num_classes': num_classes,
        'last_activ': None,
        'n_iter': 100,
        'lr': 0.01
        }

    print('Current config: ')
    pprint(config)
    print()

    rmses = []
    best_rmse = 100

    for train_index, test_index in kf.split(X, y):
        # Code taken from official documentation of sklearn

        print()
        
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

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

        rmse = metrics.rmse(pd.Series(y_hat), pd.Series(y_test))
        # print("RMSE: ", rmse)
        rmses.append(rmse)
        best_rmse = min(best_rmse, rmse)
    
    print()
    print("The RMSE in the {} folds: ".format(k))
    print(rmses)
    print("The best RMSE: ", best_rmse)

# test_digits(3)
test_boston(3)