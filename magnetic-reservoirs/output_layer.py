"""
    This file contains the 2 methods to train the output layer of the predicted_classes_vectorervoir computer. Either it uses 
    a linear regression or a ridge regression. The "classifier" ones are for classification tasks and the 
    "predicter" ones are for prediction tasks.
    The methods are just slightly adapted from the scikit-learn methods. For classifying, we one-hot encode 
    the argmax of the predicted_classes_vectorults to get only [0 ... 0 1 0 ... 0] type vectors. For predicting, ...

    Parameters:
                n_features : int : dimension of one switching probability 
                                example: psw is a scalar then n_features = 1
                                example: psw is a 5-dimensional vector then n_features = 5
                beta : float : Tikhonov regularization parameter, usually 0.<beta<=1., 
                                regularizing slightly reduces accuracy but leads to 
                                less overfitting hence more robustness
                training_target : np.array : true values of the classification/prediction 
                                            in the training stage
                training_Pswitch : np.array : array of switching probabilities corresponding
                                            to each training target value
                test_Pswitch : np.array : array of switching probabilities corresponding to 
                                        each value we want to predict/classify in the testing phase
    
    Returns:
                predicted_classes_vector :  np.array : prediction of the classifying vector (contains all the predicted classes of our model)
                predicted_data_point : float : prediction of the next data point in a time series 
                 
"""

import numpy as np
from sklearn.linear_model import LinearRegression, Ridge


class classifier_linear_regression:

    def __init__(self, n_features = 1):
        self.n_features = n_features
        self.reg = LinearRegression()

    def train(self, training_target, training_Pswitch):
        X = training_Pswitch
        y = training_target
        if self.n_features == 1:
            self.reg.fit(X.reshape(-1,1),y)
        else:
            self.reg.fit(X,y)

    def matrix(self):
        return self.reg.coef_, self.reg.intercept_

    def test(self, test_Pswitch):
        X_test = test_Pswitch
        if self.n_features == 1:
            predicted_classes_vector = self.reg.predict(X_test.reshape(-1,1))
        else:
            predicted_classes_vector = self.reg.predict(X_test)
        for r in predicted_classes_vector:
            for j in range(len(r)):
                if j == np.argmax(r):
                    r[j] = 1.
                else:
                    r[j] = 0.

        return predicted_classes_vector
    



class classifier_ridge_regression: 

    def __init__(self, beta, n_features):
        self.n_features = n_features
        self.clf = Ridge(alpha = beta)

    def train(self, training_target, training_Pswitch):
        y = training_target
        X = training_Pswitch
        if self.n_features == 1:
            self.clf.fit(X.reshape(-1,1),y)
        else:
            self.clf.fit(X,y)

    def matrix(self):
        return self.clf.coef_, self.clf.intercept_

    def test(self, test_Pswitch):
        X_test = test_Pswitch
        if self.n_features == 1:
            predicted_classes_vector = self.clf.predict(X_test.reshape(-1,1))
        else:
            predicted_classes_vector = self.clf.predict(X_test)
        for r in predicted_classes_vector:
            for j in range(len(r)):
                if j == np.argmax(r):
                    r[j] = 1.
                else:
                    r[j] = 0.
                    
        return predicted_classes_vector 
    
    
class prediction_linear_regression:

    def __init__(self, n_features = 1):
        self.n_features = n_features
        self.reg = LinearRegression()

    def train(self, training_target, training_Pswitch):
        X = np.array(training_Pswitch[3:])
        y = np.array(training_target[3:])
        if self.n_features == 1:
            self.reg.fit(X.reshape(-1,1),y)
        else:
            self.reg.fit(X,y)

    def matrix(self):
        return self.reg.coef_, self.reg.intercept_

    def predict(self, test_Pswitch):
        X_test = np.array(test_Pswitch)
        if self.n_features == 1:
            predicted_data_point = self.reg.predict(X_test.reshape(-1,1))
        else:
            predicted_data_point = self.reg.predict(X_test)

        return predicted_data_point