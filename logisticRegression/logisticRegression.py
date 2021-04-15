import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from .utils import *

import autograd.numpy as anp
from autograd import grad

class LogisticRegression():
    def __init__(self, bias=False):
        self.coef_ = None
        self.use_bias = bias
        self.bias = None
        
        self.theta_history = []

    def fit_2class_unreg(self, X, y,batch_size, n_iter=100, lr=0.01):
        '''
        Function to train model using vectorised gradient descent.

        :param X: pd.DataFrame with rows as samples and columns as features (shape: (n_samples, n_features))
        :param y: pd.Series with rows corresponding to output (shape: (n_samples,))
        :param batch_size: int specifying the batch size. Batch size can only be between 1 and number of samples in data.
        :param n_iter: number of iterations (default: 100)
        :param lr: learning rate (default: 0.01)
        :param lr_type: If lr_type = 'constant', then the learning rate remains constant,
                        if lr_type = 'inverse', then learning rate = lr / t, where t = current iteration number

        :return None
        '''

        num_samples = X.shape[0]
        
        num_features = X.shape[1]

        self.coef_ = np.zeros((num_features,1))
        theta = self.coef_
        curr_lr = lr

        if self.use_bias:
            self.bias = 0
            b = self.bias

        for iter in range(1, n_iter+1):
            # print("Iteration: {}".format(iter), theta)

            self.theta_history.append(theta.copy())

            for batch in range(0, num_samples, batch_size):
                X_batch = np.array(X.iloc[batch:batch+batch_size])
                y_batch = np.array(y.iloc[batch:batch+batch_size]).reshape((len(X_batch), 1))

                curr_sample_size = len(X_batch)

                y_hat_batch = sigmoid(np.matmul(X_batch, theta))
                error_batch = y_hat_batch - y_batch

                # update coeffs
                theta -= (1/curr_sample_size)*curr_lr*np.matmul(X_batch.T, error_batch)
                if self.use_bias:
                    b -= (1/curr_sample_size)*curr_lr*np.sum(error_batch)
                

        self.coef_ = theta
        if self.use_bias:
            self.bias = b


    def anp_loss(self, X, y, theta):
        y_hat = anp_sigmoid(anp.matmul(X,theta))
        cost = y*anp.log(anp_sigmoid(y_hat)) + (1-y)*anp.log(1-anp_sigmoid(y_hat))
        return -anp.sum(cost)/len(X)

    def anp_l1_loss(self, X, y, theta, lam):
        y_hat = anp_sigmoid(anp.matmul(X,theta))
        cost = y*anp.log(anp_sigmoid(y_hat)) + (1-y)*anp.log(1-anp_sigmoid(y_hat))

        return (lam*anp.absolute(theta))/len(X) - anp.sum(cost)/len(X)

    def anp_l2_loss(self, X, y, theta, lam):
        y_hat = anp_sigmoid(anp.matmul(X,theta))
        cost = y*anp.log(anp_sigmoid(y_hat)) + (1-y)*anp.log(1-anp_sigmoid(y_hat))

        return lam*anp.dot(theta, theta.T)/len(X) - anp.sum(cost)/len(X)
        

    def fit_2class_autograd(self, X, y, batch_size, n_iter=100, lr=0.01, reg_type=None, lam=0):
        '''
        Function to train model using gradient descent with Autograd to compute the gradients.
        Autograd reference: https://github.com/HIPS/autograd

        :param X: pd.DataFrame with rows as samples and columns as features (shape: (n_samples, n_features))
        :param y: pd.Series with rows corresponding to output (shape: (n_samples,))
        :param batch_size: int specifying the  batch size. Batch size can only be between 1 and number of samples in data.
        :param n_iter: number of iterations (default: 100)
        :param lr: learning rate (default: 0.01)

        :return None
        '''

        num_samples = X.shape[0]
        
        num_features = X.shape[1]

        self.coef_ = anp.zeros((num_features,1))
        theta = self.coef_
        curr_lr = lr

        if reg_type is None:
            loss_grad = grad(self.anp_loss, argnum=2)
        elif reg_type == 'L1':
            loss_grad = grad(self.anp_l1_loss, argnum=2)
        elif reg_type == "L2":
            loss_grad = grad(self.anp_l2_loss, argnum=2)

        for iter in range(1, n_iter+1):
            # print(theta)

            for batch in range(0, num_samples, batch_size):
                X_batch = anp.array(X.iloc[batch:batch+batch_size])
                y_batch = anp.array(y.iloc[batch:batch+batch_size]).reshape((len(X_batch), 1))

                curr_sample_size = len(X_batch)

                y_hat_batch = anp.matmul(X_batch, theta)
                # error_batch = y_hat_batch - y_batch

                # update coeffs
                theta -= (1/curr_sample_size)*curr_lr*loss_grad(X_batch, y_batch, theta, lam)

        self.coef_ = theta


    def predict_2class(self, X):
        '''
        Funtion to run the Logistic Regression on a data point

        :param X: pd.DataFrame with rows as samples and columns as features

        :return: y: pd.Series with rows corresponding to output variable. The output variable in a row is the prediction for sample in corresponding row in X.
        '''

        X_dash = X

        bias = self.bias
        if bias is None:
            bias = 0

        pred = sigmoid(np.dot(X_dash, self.coef_)+bias)

        return pd.Series([1 if i > 0.5 else 0 for i in pred])

    def plot_2d_boundary(self, X, y):
        '''
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