import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import KFold
import seaborn as sns
from sklearn.metrics import confusion_matrix

from tqdm import tqdm

from logisticRegression.logisticRegression import LogisticRegression
from metrics import *


# k-fold cross-validation
def cross_validation(k, df, lam, reg_type):
	fold_size = len(df)/k
	best_lr = None
	best_accuracy = 0
	avg_acc = 0

	for fold in range(k):
		print("		Inner Fold: ", fold+1)
		l_fold = int(fold_size*fold)
		r_fold = int(fold_size*(fold+1))

		train = df[0:l_fold].append(df[r_fold:len(df)])
		test = df[l_fold:r_fold]

		x_train = train.drop(["target"], axis=1)
		y_train = train["target"]
		x_test = test.drop(["target"], axis=1)
		y_test = test["target"]

		LR = LogisticRegression()

		# LR.fit_2class_unreg(x_train, y_train, 10, n_iter=200)
		LR.fit_2class_autograd(x_train, y_train, 10, n_iter = 200, reg_type=reg_type, lam=lam)

		y_hat = np.array(LR.predict_2class(x_test))

		acc = accuracy(y_hat, pd.Series(np.array(y_test)))
		avg_acc += acc
		if(acc > best_accuracy):
			best_accuracy = acc
			best_lr = LR
	avg_acc/=k
	return best_lr, best_accuracy, avg_acc

def nested_cross_val(k, df, min_lambda, max_lambda, reg_type):
	fold_size = len(df)/k
	best_lr = None
	best_accuracy = 0
	final_lambda = None
	for fold in range(k):
		print('[Logs] Outer fold: ', fold+1)

		l_fold = int(fold_size*fold)
		r_fold = int(fold_size*(fold+1))

		train = df[0:l_fold].append(df[r_fold:len(df)])
		test = df[l_fold:r_fold]

		x_train = train.drop(["target"], axis=1)
		y_train = train["target"]
		x_test = test.drop(["target"], axis=1)
		y_test = test["target"]

		best_avg_val_acc = 0
		best_lambda = None

		for lam in range(min_lambda, max_lambda):
			_, _, avg_val_acc = cross_validation(k, train, lam, reg_type)
			if avg_val_acc>best_avg_val_acc:
				best_lambda = lam

		LR = LogisticRegression()

		# LR.fit_2class_unreg(x_train, y_train, 10, n_iter=200)
		LR.fit_2class_autograd(x_train, y_train, 10, n_iter = 200, reg_type=reg_type, lam=lam)

		y_hat = np.array(LR.predict_2class(x_test))
		acc = accuracy(y_hat, pd.Series(list(y_test)))

		if(acc>best_accuracy):
			best_accuracy = acc
			best_lr = LR
			final_lambda = best_lambda
	return final_lambda, best_accuracy, best_lr



data = load_breast_cancer()

normalized = MinMaxScaler(feature_range=(-1,1))

data_arr = data['data']
y = pd.Series(data['target'])
data_arr = normalized.fit_transform(data_arr)
df = pd.DataFrame(data_arr, columns=data['feature_names'])

df['target'] = data['target']

print('\n------------------Nested Cross Validation------------------')
best_lambda, best_n_acc, best_n_LR = nested_cross_val(4, df, 1, 4, 'L1')
print("Best lambda is: ", best_lambda)
print("The best accuracy is: ", best_n_acc)

# print(best_n_LR.coef_)
imp_feat_no = np.argmax(abs(best_n_LR.coef_.reshape(-1)))
print("The most important featue no. is: ", imp_feat_no)