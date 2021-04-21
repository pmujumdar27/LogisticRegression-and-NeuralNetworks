# import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
import autograd.numpy as np
from autograd import grad

class NN():
    def __init__(self, n_features, n_hidden, n_neurons, activations, num_classes=None, last_activ=None):
        '''
        '''
        # o = np.ones(n)
        # X = np.concatenate((o.reshape(-1,1), X), axis=1)
        # self.X = np.concatenate((np.ones(X.shape[0]).reshape(-1,1), X), axis=1)
        # self.X = X
        # self.y = y
        # self.n_samples = self.X.shape[0]
        self.n_features = n_features
        self.n_hidden = n_hidden
        self.n_neurons = n_neurons
        self.activations = activations
        self.num_classes = num_classes
        self.last_activ = last_activ
        self.weights = []

        assert(self.n_hidden == len(n_neurons) == len(activations))

        self.init_params()

    def init_params(self):
        '''
        '''
        if self.n_hidden > 0:
            first_w = np.random.randn(*(self.n_features+1, self.n_neurons[0]))
            self.weights.append(first_w)

        for curr_layer in range(1, self.n_hidden):
            n_prev = self.n_neurons[curr_layer-1]
            n_curr = self.n_neurons[curr_layer]

            self.weights.append((np.random.randn( *(n_prev+1, n_curr ))))

        if self.num_classes is None:
            self.activations.append(self.last_activ)
            prev_shape = self.n_features
            if self.n_hidden > 0:
                prev_shape = self.n_neurons[-1]
            last_w = np.random.randn( *(prev_shape+1, 1) ).reshape(-1, 1)
        else:
            self.activations.append('softmax')
            prev_shape = self.n_features
            if self.n_hidden > 0:
                prev_shape = self.n_neurons[-1]
            last_w = np.random.randn( *(prev_shape+1, self.num_classes) )

        self.weights.append(last_w)

    def relu(self, z):
        return (z > 0).astype(float)

    def sigmoid(self, z):
        return 1 / np.exp(-z)

    def softmax(self, z):
        z -= np.max( z, axis=1, keepdims=True)
        tmp = np.exp(z)
        return tmp / np.sum(tmp, axis=1, keepdims=True)

    def forward_pass(self, X, weights):
        '''
        '''
        z = X
        
        for i in range(self.n_hidden):
            z = np.concatenate((np.ones(z.shape[0]).reshape(-1,1), z), axis=1)
            z = (np.dot(z, weights[i]))

            if self.activations[i] == 'relu':
                z = self.relu(z)
            elif self.activations[i] == 'sigmoid':
                z = self.sigmoid(z)

        z = np.concatenate((np.ones(z.shape[0]).reshape(-1,1), z), axis=1)
        z = (np.dot(z, weights[self.n_hidden]))

        if self.last_activ is None:
            pass
        elif self.last_activ == "softmax":
            z = self.softmax(z)
        
        return z

    def indicator(self, y, y_hat):
        ind = np.zeros(y_hat.shape)

        for sample in range(len(y)):
            ind[sample][y[sample]] = 1

        return ind

    def loss(self, weights, X, y):
        '''
        '''
        y_hat = self.forward_pass(X, weights) + 1e-8
        if self.last_activ is None:
            loss = np.sqrt(np.sum(np.square(y - y_hat)) / len(y_hat))
        else:
            # loss is softmax and it is a classification problem
            indic = self.indicator(y, y_hat)

            loss = - np.sum( indic * np.log(y_hat))

        return loss

    def fit(self, X, y, batch_size, n_iter=100, lr=0.01):
        '''
        '''
        self.X = X
        self.y = y
        
        n = X.shape[0]

        assert (X.shape[1] == self.n_features)

        self.init_params()

        weights_grad = grad(self.loss, argnum=0)

        for iter in range(n_iter):

            loss = self.loss(self.weights, X, y)

            if iter % 10 == 0:
                print("[Logs] Iteration: {} | Loss: {}".format(iter, loss))

            for batch in range(0, n, batch_size):
                X_batch = X[batch : batch+batch_size]
                y_batch = y[batch : batch+batch_size]

                curr_sample_size = len(X_batch)

                curr_weights_grad = weights_grad(self.weights, X_batch, y_batch)

                for layer in range(len(self.weights)):
                    self.weights[layer] -= (lr*curr_weights_grad[layer])/curr_sample_size

    def predict(self, X):
        '''
        '''
        y_hat = self.forward_pass(X, self.weights)
        if self.num_classes is None:
            return y_hat.reshape(-1)
        else:
            return np.argmax(y_hat, axis=1)
        # return y_hat

'''
X = np.random.randn(10, 5)
y = np.array([np.random.randint(4) for i in range(10)])
print(X.shape)
print(y.shape)

n_hidden = 3
n_neurons = [3,4,2]
n_features = 5
activations = ['relu', 'relu', 'relu']
num_classes = 4
last_activ = None

nn = NN(n_features, n_hidden, n_neurons, activations, None, None)
# nn = NN(n_features, n_hidden, n_neurons, activations, 5, 'softmax')

tmp = nn.forward_pass(X, nn.weights)
# print(tmp)
print(nn.softmax(tmp))

print("Loss: ")
print(nn.loss(nn.weights, X, y))

nn.fit(X, y, 10, n_iter=1000)
y_hat = nn.predict(X)
print(y_hat)
'''