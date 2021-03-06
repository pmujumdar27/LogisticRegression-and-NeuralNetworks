import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import MinMaxScaler

from tqdm import tqdm

from logisticRegression.logisticRegression import LogisticRegression
from metrics import *


def cross_validation(k, df):
    print('{} fold cross validation:'.format(k))
    fold_size = len(df)/k
    best_lr = None
    accs = []
    best_acc = 0

    for fold in tqdm(range(k)):
        l_fold = int(fold_size*fold)
        r_fold = int(fold_size*(fold+1))

        train = df[0:l_fold].append(df[r_fold:len(df)])
        test = df[l_fold:r_fold]

        x_train = train.drop(["target"], axis=1)
        y_train = train["target"]
        x_test = test.drop(['target'], axis=1)
        y_test = test['target']

        LR = LogisticRegression()

        # LR.fit_2class_unreg(x_train, y_train, 10, n_iter=200)
        LR.fit_2class_autograd(x_train, y_train, 10, n_iter = 200, reg_type=None)

        y_hat = np.array(LR.predict_2class(x_test))

        acc = accuracy(y_hat, pd.Series(np.array(y_test)))
        accs.append(acc)
        if(acc > best_acc):
            best_acc = acc
            best_lr = LR

    return best_lr, best_acc, accs

def plot_db_test(X, y):
    LR = LogisticRegression()
    LR.fit_2class_unreg(X, y, 10, n_iter=200)

    print("Plotting decision boundary")
    fig = LR.plot_2d_boundary(X, y)

    fig.savefig('plots/decision_boundary_2cls.png')
    print("Plotting successful!")

data = load_breast_cancer()

normalized = MinMaxScaler(feature_range=(-1,1))

data_arr = data['data']
y = pd.Series(data['target'])
data_arr = normalized.fit_transform(data_arr)
df = pd.DataFrame(data_arr, columns=data['feature_names'])

X_db = df[[df.columns[0], df.columns[1]]]
df['target'] = data['target']

print('--------------------------------------------------')

_, best_accuracy, accs = cross_validation(3, df)
print()
print("Accuracies: ")
print(accs)
print()
print("Best Accuracy: ", best_accuracy)

print('--------------------------------------------------')

y_db = df['target']

plot_db_test(pd.DataFrame(X_db), y_db)