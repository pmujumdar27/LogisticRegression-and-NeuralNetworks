import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from tqdm import tqdm
from logisticRegression.multiclassLR import MulticlassLR
from metrics import *

from sklearn.datasets import load_digits
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix

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

'''
MLR.fit(X, y, 10, n_iter=50, lr=0.1)
# MLR.fit_autograd(X, y, 10, n_iter=50, lr=0.1)
MLR.plot_loss_history()

y_hat = MLR.predict(X)

acc = accuracy(y_hat, y)

print("Accuracy: ", acc)

# print(digits_dataset)
'''

skf = StratifiedKFold(n_splits=4)

cm_best = None
best_acc = 0
cm_tot = np.zeros((10,10))

for train_index, test_index in skf.split(X, y):
    # Code taken from official documentation https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.StratifiedKFold.html
    # print("Train: ", train_index, "Test: ", test_index)
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y[train_index], y[test_index]

    MLR = MulticlassLR(num_classes)

    MLR.fit(X_train, y_train, 10, n_iter=50, lr=0.1)

    y_hat = MLR.predict(X_test)

    acc = accuracy(y_hat, y_test)

    cm = confusion_matrix(y_hat, y_test)
    cm_tot += cm
    print("Accuracy: ", acc)

    if acc > best_acc:
        best_acc = acc
        y_test_best = y_test
        y_hat_best = y_hat

        cm_best = cm
        # cm_best = cm

print(cm_best)
print()
print("Best accuracy: ", best_acc)
sns.heatmap(cm_best, annot=True)
plt.title("Confusion matrix visualized")
plt.show()

print(cm_tot)
sns.heatmap(cm_tot, annot=True)
plt.show()