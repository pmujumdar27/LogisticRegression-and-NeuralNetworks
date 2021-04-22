# ES654-2021 Assignment 3

*Pushkar Mujumdar* - *18110132*

------

> Time and Space Complexity analysis of Logistic regression

---
### Assumptions:
- Let the number of features = ```d```
- Let the number of samples = ```n```
- Let the number of classes = ```k```
- Let number of iteratons = ```n_iter```

---

### Time complexity analysis

For one sample during a particular iteration of fitting, we calculate the loss using ```(d+1) x k``` indicator matrix.

We do this for each sample in each iteration.

Everything else is similar to linear regression using gradient descent.

Therefor theoretical time complexity of ```k-class Logistic Regression``` for fitting is:
```
O(n*d*k*n_iter)
```

For prediction of one sample, we multiply the sample with the weights which is a ```(d+1) x k``` matrix.

So for ```n``` samples, space complexity of prediction is:
```
O(n*d*k)
```

---

### Space complexity analysis

The space complexity ```k-class Logistic Regression``` for fit is:
```
O(n*(k+d))
```

The space complexity for predict is:
```
O(n*(k+d))
```

---