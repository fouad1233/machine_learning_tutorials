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

def mean_squared_error(y_true, y_pred):
    return np.mean((y_true - y_pred)**2)


def true_positive(y_true, y_pred):
    TP = np.sum((y_true == 1) & (y_pred == 1))
    return TP
def true_negative(y_true, y_pred):
    TN = np.sum((y_true == 0) & (y_pred == 0))
    return TN
def false_positive(y_true, y_pred):
    FP = np.sum((y_true == 0) & (y_pred == 1))
    return FP
def false_negative(y_true, y_pred):
    FN = np.sum((y_true == 1) & (y_pred == 0))
    return FN

def classification_accuracy(y_true, y_pred):
    return np.mean(y_true == y_pred)
def precision(y_true, y_pred):
    TP = true_positive(y_true, y_pred)
    FP = false_positive(y_true, y_pred)
    return TP / (TP + FP)
def true_positive_rate(y_true, y_pred):
    TP = true_positive(y_true, y_pred)
    FN = false_negative(y_true, y_pred)
    return TP / (TP + FN)
def false_positive_rate(y_true, y_pred):
    FP = false_positive(y_true, y_pred)
    TN = true_negative(y_true, y_pred)
    return FP / (FP + TN)
