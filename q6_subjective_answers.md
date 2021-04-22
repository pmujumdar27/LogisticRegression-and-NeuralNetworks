# ES654-2021 Assignment 3

*Pushkar Mujumdar* - *18110132*

------

> Implemented a fully connected Neural Network by writing forward pass manually and using Autorgrad for backpropagation gradients  
> Tested the above code for classification on Digits Dataset and for regression on Boston housing dataset both using 3-fold Cross Validation

Results on Digits Dataset for 10 class classification using ```0 hidden layers``` and 3 fold CV:
```
Current Config: 
{'activations': [],      
 'last_activ': 'softmax',
 'lr': 0.02,
 'n_hidden': 0,
 'n_iter': 120,
 'n_neurons': [],        
 'num_classes': 10}

The accuracies for the 3 folds: 
[0.9131886477462438, 0.9115191986644408, 0.8964941569282137]

Best accuracy:  0.9131886477462438
```

Results on Boston Housing dataset for regression using ```2 hidden layers``` and 3 fold CV:
```
Current config: 
{'activations': ['relu', 'relu'],
 'last_activ': None,
 'lr': 0.01,
 'n_hidden': 2,
 'n_iter': 100,
 'n_neurons': [8, 8],
 'num_classes': None}

The RMSE in the 3 folds: 
[5.754130195701396, 9.623466091756738, 6.9685388928632825]

The best RMSE:  5.754130195701396
```