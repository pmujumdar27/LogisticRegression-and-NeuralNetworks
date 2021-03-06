import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from .utils import *

import autograd.numpy as anp
from autograd import grad

class LogisticRegression():
    def __init__(self):
        self.coef_ =        None
        self.theta_history = []

    def fit_2class_unreg(self, X, y, batch_size, n_iter=100, lr=0.01):
        '''
        Function to fit using manually calculated gradients
        X: input samples dataframe
        y: target labels series
        batch_size: batch size
        n_iter: number of iterations
        lr: learning rate
        '''
        n,m = X.shape
        X = np.array(X)
        y = np.array(y)
        o = np.ones(n)
        X = np.concatenate((o.reshape(-1,1), X), axis=1)

        num_samples = X.shape[0]
        
        num_features = X.shape[1]

        self.coef_ = np.zeros((num_features,1))
        theta = self.coef_
        curr_lr = lr

        for iter in range(1, n_iter+1):

            self.theta_history.append(theta.copy())

            # batched gradient descent
            for batch in range(0, num_samples, batch_size):
                X_batch = np.array(X[batch:batch+batch_size])
                y_batch = np.array(y[batch:batch+batch_size]).reshape((len(X_batch), 1))

                curr_sample_size = len(X_batch)

                y_hat_batch = sigmoid(np.matmul(X_batch, theta))
                error_batch = y_hat_batch - y_batch

                # update coeffs
                theta -= (1/curr_sample_size)*curr_lr*np.matmul(X_batch.T, error_batch)
                

        self.coef_ = theta


    def anp_loss(self, X, y, theta, reg_type, lam):
        '''
        Function to calculate crossentropy loss
        X: input samples dataframe
        y: target labels series
        theta: weights
        reg_type: type of regularization (None, L1 or L2)
        lam: lambda for regularization
        '''
        y_hat = anp_sigmoid(anp.matmul(X,theta))
        cost = -1 * (anp.dot(y.T, anp.log(y_hat)) + anp.dot((1 - y).T, anp.log(1 - y_hat)))
        if reg_type == 'L1':
            # cost += lam * anp.linalg.norm(theta, ord=1)
            cost += lam * anp.sum(abs(theta))
        elif reg_type == 'L2':
            # cost += lam * anp.linalg.norm(theta, ord=2)
            cost += lam * anp.dot(theta.reshape(-1), theta.reshape(-1).T)
        return cost        

    def fit_2class_autograd(self, X, y, batch_size, n_iter=100, lr=0.01, reg_type=None, lam=0):
        '''
        Function to fit using manually calculated gradients
        X: input samples dataframe
        y: target labels series
        batch_size: batch size
        n_iter: number of iterations
        lr: learning rate
        reg_type: type of regularization (None, L1 or L2)
        lam: lambda for regularization
        '''
        n, m = X.shape
        X = np.array(X)
        y = np.array(y)
        o = np.ones(n)
        X = np.concatenate((o.reshape(-1,1), X), axis=1)

        num_samples = X.shape[0]
        
        num_features = X.shape[1]

        self.coef_ = anp.zeros((num_features,1))
        theta = self.coef_
        curr_lr = lr

        loss_grad = grad(self.anp_loss, argnum=2)

        for iter in range(1, n_iter+1):
            # print(theta)

            for batch in range(0, num_samples, batch_size):
                X_batch = anp.array(X[batch:batch+batch_size])
                y_batch = anp.array(y[batch:batch+batch_size]).reshape((len(X_batch), 1))

                curr_sample_size = len(X_batch)

                y_hat_batch = anp.matmul(X_batch, theta)

                # update coeffs
                theta -= (1/curr_sample_size)*curr_lr*loss_grad(X_batch, y_batch, theta, reg_type, lam)

        self.coef_ = theta


    def predict_2class(self, X):
        '''
        Funtion to run the Logistic Regression on a data point

        :param X: pd.DataFrame with rows as samples and columns as features

        :return: y: pd.Series with rows corresponding to output variable. The output variable in a row is the prediction for sample in corresponding row in X.
        '''

        n,_ = X.shape
        X = np.array(X)
        o = np.ones(n)
        X_dash = np.concatenate((o.reshape(-1,1), X), axis=1)

        pred = sigmoid(np.dot(X_dash, self.coef_))

        return pd.Series([1 if i > 0.5 else 0 for i in pred])

    def plot_2d_boundary(self, X, y):
        '''
        Function to plot 2d decision boundary
        Inputs:
        X: pd.DataFrame with rows as samples and columns as features [Give only 2 features (columns)]
        y: pd.Series with rows corresponding to output variable
        '''
        X_np = X.to_numpy()
        min1, max1 = X_np[:, 0].min()-1, X_np[:, 0].max()+1
        min2, max2 = X_np[:, 1].min()-1, X_np[:, 1].max()+1
        x1grid = np.arange(min1, max1, 0.1)
        x2grid = np.arange(min2, max2, 0.1)

        xx, yy = np.meshgrid(x1grid, x2grid)

        r1, r2 = xx.flatten(), yy.flatten()
        r1, r2 = r1.reshape((len(r1), 1)), r2.reshape((len(r2), 1))

        grid = np.hstack((r1,r2))

        fig1, axs1 = plt.subplots()
        axs1.set(xlabel='X1', ylabel='X2')
        fig1.suptitle('Decision boundary for the two classes')

        self.fit_2class_unreg(X, y, 10)
        y_hat = self.predict_2class(grid)
        zz = pd.Series(y_hat).to_numpy().reshape(xx.shape)

        axs1.contourf(xx, yy, zz, cmap='Paired')

        for class_value in [0,1]:
            row_ix = y == class_value
            axs1.scatter(X[row_ix][X.columns[0]], X[row_ix][X.columns[1]], cmap='Paired')

        return fig1