import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from .utils import *

import autograd.numpy as anp
from autograd import grad

class LogisticRegression():
    def __init__(self):
        self.coef_ = None
        self.theta_history = []

    def fit_2class_unreg(self, X, y,batch_size, n_iter=100, lr=0.01, lr_type='constant'):
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

        for iter in range(1, n_iter+1):
            # print("Iteration: {}".format(iter), theta)

            self.theta_history.append(theta.copy())

            if lr_type!='constant':
                curr_lr = lr/iter

            for batch in range(0, num_samples, batch_size):
                X_batch = np.array(X.iloc[batch:batch+batch_size])
                y_batch = np.array(y.iloc[batch:batch+batch_size]).reshape((len(X_batch), 1))

                curr_sample_size = len(X_batch)

                y_hat_batch = sigmoid(np.matmul(X_batch, theta))
                error_batch = y_hat_batch - y_batch

                # update coeffs
                theta -= (1/curr_sample_size)*curr_lr*np.matmul(X_batch.T, error_batch)
                

        self.coef_ = theta


    def anp_loss(self, X, y, theta):
        y_hat = sigmoid(anp.matmul(X,theta))
        cost = y*np.log(sigmoid(y_hat)) + (1-y)*np.log(1-sigmoid(y_hat))
        return -np.sum(cost)/len(X)
        

    def fit_2class_unreg_autograd(self, X, y, batch_size, n_iter=100, lr=0.01, lr_type='constant'):
        '''
        Function to train model using gradient descent with Autograd to compute the gradients.
        Autograd reference: https://github.com/HIPS/autograd

        :param X: pd.DataFrame with rows as samples and columns as features (shape: (n_samples, n_features))
        :param y: pd.Series with rows corresponding to output (shape: (n_samples,))
        :param batch_size: int specifying the  batch size. Batch size can only be between 1 and number of samples in data.
        :param n_iter: number of iterations (default: 100)
        :param lr: learning rate (default: 0.01)
        :param lr_type: If lr_type = 'constant', then the learning rate remains constant,
                        if lr_type = 'inverse', then learning rate = lr / t, where t = current iteration number

        :return None
        '''

        num_samples = X.shape[0]
        
        num_features = X.shape[1]

        self.coef_ = anp.zeros((num_features,1))
        theta = self.coef_
        curr_lr = lr

        loss_grad = grad(self.anp_loss, argnum=2)

        for iter in range(1, n_iter+1):
            # print(theta)
            if lr_type!='constant':
                curr_lr = lr/iter

            for batch in range(0, num_samples, batch_size):
                X_batch = anp.array(X.iloc[batch:batch+batch_size])
                y_batch = anp.array(y.iloc[batch:batch+batch_size]).reshape((len(X_batch), 1))

                curr_sample_size = len(X_batch)

                y_hat_batch = anp.matmul(X_batch, theta)
                # error_batch = y_hat_batch - y_batch

                # update coeffs
                theta -= (1/curr_sample_size)*curr_lr*loss_grad(X_batch, y_batch, theta)

        self.coef_ = theta


    def predict_2class(self, X):
        '''
        Funtion to run the LinearRegression on a data point

        :param X: pd.DataFrame with rows as samples and columns as features

        :return: y: pd.Series with rows corresponding to output variable. The output variable in a row is the prediction for sample in corresponding row in X.
        '''

        X_dash = X

        pred = sigmoid(np.dot(X_dash, self.coef_))

        return pd.Series([1 if i > 0.5 else 0 for i in pred])