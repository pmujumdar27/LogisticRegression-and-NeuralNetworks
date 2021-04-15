import numpy as np
import pandas as pd
import autograd.numpy as anp

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def anp_sigmoid(z):
    return 1 / (1 + anp.exp(-z))