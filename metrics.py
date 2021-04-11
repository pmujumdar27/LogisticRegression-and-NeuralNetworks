import pandas as pd
import numpy as np

def accuracy(y_hat, y):
    """
    Function to calculate the accuracy

    Inputs:
    > y_hat: pd.Series of predictions
    > y: pd.Series of ground truth
    Output:
    > Returns the accuracy as float
    """
    """
    The following assert checks if sizes of y_hat and y are equal.
    Students are required to add appropriate assert checks at places to
    ensure that the function does not fail in corner cases.
    """
    assert(y_hat.size == y.size)
    # TODO: Write here
    assert(y.size>0)
    if True in (y==y_hat).unique():
        return (((y_hat==y).value_counts()[True])/y.size)
    else:
        return 0

def precision(y_hat, y, cls):
    """
    Function to calculate the precision

    Inputs:
    > y_hat: pd.Series of predictions
    > y: pd.Series of ground truth
    > cls: The class chosen
    Output:
    > Returns the precision as float
    """
    assert(y_hat.size == y.size)
    retrieved = (y_hat==cls)
    relevant = (y==cls)
    cnt = (retrieved==True) & (relevant==True)
    tot = retrieved.sum()
    # assert(tot!=0)
    if(tot==0):
        return 0
    return cnt.sum()/tot

def recall(y_hat, y, cls):
    """
    Function to calculate the recall

    Inputs:
    > y_hat: pd.Series of predictions
    > y: pd.Series of ground truth
    > cls: The class chosen
    Output:
    > Returns the recall as float
    """
    assert(y_hat.size == y.size)
    retrieved = (y_hat==cls)
    relevant = (y==cls)
    cnt = (retrieved==True) & (relevant==True)
    tot = relevant.sum()
    assert(tot>0)
    return cnt.sum()/tot

# def rmse(y_hat, y):
#     """
#     Function to calculate the root-mean-squared-error(rmse)

#     Inputs:
#     > y_hat: pd.Series of predictions
#     > y: pd.Series of ground truth
#     Output:
#     > Returns the rmse as float
#     """
#     assert(y_hat.size == y.size)
#     assert(y.size>0)
#     return np.sqrt(((y_hat-y)**2).sum()/y.size)

# def mae(y_hat, y):
#     """
#     Function to calculate the mean-absolute-error(mae)

#     Inputs:
#     > y_hat: pd.Series of predictions
#     > y: pd.Series of ground truth
#     Output:
#     > Returns the mae as float
#     """
#     assert(y_hat.size == y.size)
#     assert(y.size>0)
#     return (abs(y_hat-y)).sum()/y.size
