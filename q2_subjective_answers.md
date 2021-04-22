# ES654-2021 Assignment 3

*Pushkar Mujumdar* - *18110132*

------

> Implemented L1 and L2 regularized logistic regression for 2 classes

- We can use L1 regularization to find the feature importance, as it adds an L1 norm of the weights to the loss function. The less important features tend to have 0 weights.

### Breast Cancer Dataset

Following are the results for ```L1 Regularized``` loss calculation:
```
Best lambda is:  3
The best accuracy is:  0.8342307692307325
The most important features turn out to be 1 and 15. (Calculated over multiple experiments)
```

Following are the results for ```L2 Regularized``` loss calculation:  
```
Best lambda is:  3
The best accuracy is:  0.8802816901408451
```