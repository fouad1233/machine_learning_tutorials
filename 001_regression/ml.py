import numpy as np  
import matplotlib.pyplot as plt

def LinearRegression(x : np.array , y : np.array):
    # Calculate the mean of x and y
    x_mean = np.mean(x)
    y_mean = np.mean(y)
    
    N = len(x)

    # Calculate the terms needed for the numator and denominator of beta
    numerator = np.sum(x * y) - N * x_mean * y_mean
    denominator = np.sum(x * x) - N * (x_mean**2)

    # Calculate beta
    w1 = numerator / denominator

    # Calculate alpha
    w0 = y_mean - (w1 * x_mean)

    return w0, w1