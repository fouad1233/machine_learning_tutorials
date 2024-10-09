import numpy as np  
import matplotlib.pyplot as plt

def LinearRegression(x : np.ndarray , r : np.ndarray):
    # Calculate the mean of x and r
    x_mean = np.mean(x)
    r_mean = np.mean(r)
    
    N = len(x)

    numerator = np.sum(x * r) - N * x_mean * r_mean
    denominator = np.sum(x * x) - N * (x_mean**2)

    # Calculate w1
    w1 = numerator / denominator

    # Calculate w0
    w0 = r_mean - (w1 * x_mean)

    return w0, w1
def regression(x : np.ndarray , r : np.ndarray, degree : int):
    X = np.array([x**i for i in range(degree )]).T
    #print(X)
    w = np.linalg.inv(X.T @ (X))@ X.T @ r
    return w
    

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
