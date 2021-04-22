import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

from .utils import *

import autograd.numpy as anp
from autograd import grad

class MulticlassLR():
    def __init__(self, num_classes):
        self.coef_ = None
        self.num_classes = num_classes
        self.theta_history = []
        self.loss_history = []

    def softmax_vec(self, z):
        tmp = z-max(z)
        return np.exp(tmp)/np.sum(np.exp(tmp))

    def softmax(self, z):
        tmp = z
        for i in range(z.shape[0]):
            tmp[i]-=max(tmp[i])
            tmp[i] = np.exp(tmp[i])
            tmp[i]/=tmp[i].sum()
        return tmp

    def anp_hypothesis(self, X, theta):
        hyp = anp.exp(anp.dot(np.array(X), theta))
        den = anp.sum(hyp, axis=1).reshape(-1,1)
        return hyp/den

    def hypothesis(self, X, theta, single=False):
        if single:
            return self.softmax_vec(X @ theta)
        return self.softmax(X @ theta)

    def indicator(self, y):
        ind = np.zeros((len(y), self.num_classes))

        for sample in range(len(y)):
            ind[sample][y[sample]] = 1

        return ind

    def xentropy_loss(self, X, y, theta):

        hyp = self.hypothesis(X, theta)

        indic = self.indicator(y)

        loss = -np.sum(indic * np.log(hyp))

        # loss = 0
        # for i in range(X.shape[0]):
        #     for j in range(self.num_classes):
        #         loss -= (y[i]==j)*np.log(hyp[i][j])

        return loss


    def anp_xentropy_loss(self, X, y, theta):
        hyp = self.anp_hypothesis(X, theta)

        # indic = self.indicator(y)

        # loss = -np.sum(indic * np.log(hyp))

        loss = 0
        for i in range(X.shape[0]):
            for j in range(self.num_classes):
                loss -= (y[i]==j)*anp.log(hyp[i][j])

        return loss

    def fit(self, X, y, batch_size, n_iter=100, lr=0.01):
        n, m = X.shape

        n,m = X.shape
        X = np.array(X)
        y = np.array(y)
        o = np.ones(n)
        X = np.concatenate((o.reshape(-1,1), X), axis=1)

        num_samples = X.shape[0]
        
        num_features = X.shape[1]

        self.coef_ = np.zeros((num_features,self.num_classes))

        theta = self.coef_
        curr_lr = lr

        for iter in tqdm(range(1, n_iter+1)):

            if iter%10 == 0:
                loss = self.xentropy_loss(np.array(X), np.array(y), theta)
                print("Iteration: {}, Loss: {}".format(iter, loss))
                self.loss_history.append(loss)
                self.theta_history.append(theta.copy())

            for batch in range(0, n, batch_size):
                X_batch = np.array(X[batch:batch+batch_size])
                y_batch = np.array(y[batch:batch+batch_size]).reshape((len(X_batch), 1))

                curr_sample_size = len(X_batch)

                probab = self.hypothesis(X_batch, theta)
                theta -= (1/curr_sample_size)*curr_lr*(X_batch.T @ (probab - self.indicator(y_batch)))

                self.coef_ = theta

    def fit_autograd(self, X, y, batch_size, n_iter=100, lr=0.01):
        # n, m = X.shape

        # X_new = pd.concat([pd.Series(np.ones(n)),X],axis=1)

        n, m = X.shape

        n,m = X.shape
        X = np.array(X)
        y = np.array(y)
        o = np.ones(n)
        X = np.concatenate((o.reshape(-1,1), X), axis=1)

        num_samples = X.shape[0]
        
        num_features = X.shape[1]

        self.coef_ = anp.zeros((m+1,self.num_classes))
        theta = self.coef_
        curr_lr = lr

        loss_grad = grad(self.anp_xentropy_loss, argnum=2)

        for iter in tqdm(range(1, n_iter+1)):

            if iter%10 == 0:
                loss = self.xentropy_loss(np.array(X), np.array(y), theta)
                print("Iteration: {}, Loss: {}".format(iter, loss))
                self.loss_history.append(loss)
                self.theta_history.append(theta.copy())

            for batch in range(0, n, batch_size):
                X_batch = np.array(X[batch:batch+batch_size])
                y_batch = np.array(y[batch:batch+batch_size]).reshape((len(X_batch), 1))

                curr_sample_size = len(X_batch)

                probab = self.hypothesis(X_batch, theta)
                theta -= (1/curr_sample_size)*curr_lr*loss_grad(X_batch, y_batch, theta)

        self.coef_ = theta

    def predict(self, X):
        # X_new = pd.concat([pd.Series(np.ones(X.shape[0])),X],axis=1)

        n,m = X.shape
        X = np.array(X)
        # y = np.array(y)
        o = np.ones(n)
        X_new = np.concatenate((o.reshape(-1,1), X), axis=1)

        # pred = self.hypothesis(X_new, self.coef_)

        preds = []

        for sample in range(X_new.shape[0]):
            pred = self.hypothesis(X_new[sample], self.coef_, single=True)
            preds.append(np.argmax(pred))

        return np.array(preds)

    def plot_loss_history(self):
        plt.plot([10*i for i in range(1, len(self.loss_history)+1)], self.loss_history)
        plt.xlabel("Iterations")
        plt.ylabel("Crossentropy Loss")
        plt.show()