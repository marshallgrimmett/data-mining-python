###############################     INSTALLATION/PREP     ################
# This is a code template for logistic regression using stochastic gradient ascent to be completed by you 
# in CSI 431/531 @ UAlbany
#


###############################     IMPORTS     ##########################
import numpy as np
import pandas as pd
import math as mt
from numpy import linalg as li
import matplotlib.pyplot as plt


###############################     FUNCTION DEFINITOPNS   ##########################

"""
Receives data point x and coefficient parameter values w 
Returns the predicted label yhat, according to logistic regression.
"""
def predict(x, w):
    """
      TODO
    """
    yhat = 1 / (1 + np.exp(-np.dot(w.T, x)))
    return(yhat)
      
    
"""
Receives data point (x), data label (y), and coefficient parameter values (w) 
Computes and returns the gradient of negative log likelihood at point (x)
"""
def gradient(x, y, w):
    """
      TODO
    """
    temp = y - predict(x, w)
    grad = temp * x
    return(grad)


"""
Receives the predicted labels (y_hat), and the actual data labels (y)
Computes and returns the cross-entropy loss
"""
def cross_entropy(y_hat, y):
    """
      TODO
    """
    cross_ent = 0
    for i in range(len(y)):
        cross_ent += y[i]*np.log(1/y_hat[i]) + (1-y[i])*np.log(1/(1-y_hat[i]))
        # break
    return(cross_ent)


"""
Receives data set (X), dataset labels (y), learning rate (step size) (psi), stopping criterion (for change in norm of ws) (epsilon), and maximum number of epochs (max_epochs)
Computes and returns the vector of coefficient parameters (w), and the list of cross-entropy losses for all epochs (cross_ent)
"""
def logisticRegression_SGA(X, y, psi, epsilon, max_epochs):
    """
      TODO
      NOTE: remember to either shuffle the data points or select data points randomly in each internal iteration.
      NOTE: stopping criterion: stop iterating if norm of change in w (norm(w-w_old)) is less than epsilon, or the number of epochs (iterations over the whole dataset) is more than maximum number of epochs
    """ 

    t = 0
    w = np.zeros(len(X.columns))
    wtemp = w
    cross_ent = []

    while t < max_epochs:
        wtemp = w
        y_hat = np.zeros(len(y))
        for i, x in X.sample(frac=1).iterrows():
            w = w + psi * gradient(x, y[i], w)
            y_hat[i] = predict(x, w)
        cross_ent.append(cross_entropy(y_hat, y))
        t += 1
        if np.linalg.norm(w - wtemp) <= epsilon:
          break

    return(w,cross_ent)
  
  
if __name__ == '__main__':  
    ## initializations and config
    psi = 0.3 # learning rate or step size
    epsilon = .5 # used in SGA's stopping criterion to define the minimum change in norm of w
    max_epochs = 8 # used in SGA's stopping criterion to define the maximum number of epochs (iterations) over the whole dataset
    
    ## loading the data
    df_train = pd.read_csv("cancer-data-train.csv", header=None)
    df_test = pd.read_csv("cancer-data-test.csv", header=None)
    
    ## split into features and labels
    X_train, y_train = df_train.iloc[:, :-1], df_train.iloc[:, -1].astype("category").cat.codes #Convert string labels to numeric
    X_test, y_test = df_test.iloc[:, :-1], df_test.iloc[:, -1].astype("category").cat.codes

    ## normalize X
    X_train = (X_train - X_train.mean()) / X_train.std()
    X_test = (X_test - X_test.mean()) / X_test.std()

    ## augmenting train data with 1 (X0)
    X_train.insert(0,'',1)
    X_test.insert(0,'',1)

    ## learning logistic regression parameters
    [w, cross_ent] = logisticRegression_SGA(X_train, y_train, psi, epsilon, max_epochs)
    
    ## plotting the cross-entropy across epochs to see how it changes
    plt.plot(cross_ent, 'o', color='black')
    plt.show()
    
    """
      TODO: calculate and print the average cross-entropy error for training data (cross-entropy error/number of data points)
    """
    avg_cross_ent = cross_ent[len(cross_ent)-1] / len(X_train)       # select the last value in cross_ent, since that is our final cross entropy error
    print(avg_cross_ent)
    
    """
      TODO: predict the labels for test data points using the learned w's
    """
    y_hat = []
    for i, x in X_test.iterrows():
        y_hat.append(predict(x, w))
    
    """
      TODO: calculate and print the average cross-entropy error for testing data (cross-entropy error/number of data points)
    """
    cross_ent_test = cross_entropy(y_hat, y_test)
    avg_cross_ent_test = cross_ent_test / len(X_test)
    print(avg_cross_ent_test)