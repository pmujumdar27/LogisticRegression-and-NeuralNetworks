import numpy as np
import pandas as pd

def sigmoid(z):
    return 1 / (1 + np.e**(-z))