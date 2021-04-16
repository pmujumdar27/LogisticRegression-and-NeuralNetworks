import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from .utils import *

import autograd.numpy as anp
from autograd import grad

class MulticlassLR():
    def __init__(self, num_classes):
        self.coef_ = None
        self.num_classes = num_classes
        self.theta_history = []

    def softmax(self, z):
        z -= np.max(z) #regularization, if we don't do this, overflow might happen (almost always happens)
        return np.exp(z) / np.sum(np.exp(z))

    def hypothesis(self, X, theta):
        return self.softmax(X @ theta)

    def indicator(self, y):
        ind = np.zeros((len(y), self.num_classes))

        for sample in range(len(y)):
            ind[sample][y[sample]] = 1

        return ind

    def fit(self, X, y, batch_size, n_iter=100, lr=0.01):
        n, m = X.shape

        X_new = pd.concat([pd.Series(np.ones(n)),X],axis=1)

        self.coef_ = np.zeros((m+1,self.num_classes))
        theta = self.coef_
        curr_lr = lr

        for iter in range(1, n_iter+1):
            # y_hat = self.hypothesis(X_new, theta)
            
            # loss = 0

            # for i in range(n):
            #     for k in range(self.num_classes):
            #         indicator_tmp = 0
            #         if y[i]==k:
            #             indicator_tmp = 1
            #         loss += -1 * indicator_tmp * np.log(y_hat.iloc[i][k])

            # print("Iteration: {}, Loss: {}".format(iter, loss))

            self.theta_history.append(theta.copy())

            for batch in range(0, m, batch_size):
                X_batch = np.array(X_new.iloc[batch:batch+batch_size])
                y_batch = np.array(y.iloc[batch:batch+batch_size]).reshape((len(X_batch), 1))

                curr_sample_size = len(X_batch)

                # update coeffs
                # theta -= (1/curr_sample_size)*curr_lr*np.matmul(X_batch.T, error_batch)
                probab = self.hypothesis(X_batch, theta)
                theta -= (1/curr_sample_size)*curr_lr*(X_batch.T @ (probab - self.indicator(y_batch)))

        self.coef_ = theta

    def predict(self, X):
        X_new = pd.concat([pd.Series(np.ones(X.shape[0])),X],axis=1)

        # pred = self.hypothesis(X_new, self.coef_)

        preds = []

        for sample in range(X_new.shape[0]):
            pred = self.hypothesis(X_new.iloc[sample], self.coef_)
            preds.append(np.argmax(pred))

        return pd.Series(preds)