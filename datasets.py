"""
Here are all the datasets used in the experiments.
For classification tasks, the datasets are NUMPY ARRAYS.
For prediction tasks, the datasets are LISTS.

For classification, each function returns the training and testing data, targets, and psw.

NB : The streams of 01, 011 and 0011 are in the helper_functions.py file.
"""

import numpy as np
from sklearn.datasets import load_digits # it is the skdigits dataset
from sklearn.model_selection import train_test_split
import helper_functions as hf
from keras.datasets import mnist

rng = np.random.default_rng(2025) # to always get the same dataset 

def mnist_model(n_samples: int = 1000, test_ratio: float = 0.2, crop: bool = True, crop_list: list = [144, 704]): # classification
    """
    Function : 
                Preprocesses the MNIST dataset for training and testing.
    Parameters :
                n_samples : int : number of total (training + testing) samples 
                               
                test_ratio : float : between 0 and 1 : percentage of the total number 
                                    of samples used for testing the model (doing the 
                                    actual prediction/classification)
    Returns :
                X_train : np.array : array of all the training inputs (send to mtj)
                X_test : np.array : array of all the testing inputs
                y_train : np.array : array of all the training targets to be put in the regression models
                y_test : np.array : array of all the testing targets
                p_train : np.array : array of all the set-to-zero training psw, to be updated with the correct 
                                values from the experiment (already has the right size)
                p_test : np.array : array of all the set-to-zero testing psw
    """
    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    X = np.concatenate((X_train, X_test), axis=0)
    y = np.concatenate((y_train, y_test), axis=0)
    if crop:
        X_flattened = np.array([x.flatten()[crop_list[0]:crop_list[1]] for x in X])  # Flatten the images
    else:
        X_flattened = np.array([x.flatten() for x in X])
    n_prior = len(y)
    y_onehot = hf.one_hot(y) # Transforming digit target (3, 8) into vector with 0 and 1 at the digit's position
    p_switch = np.zeros(n_prior)
    if n_samples > n_prior:
        train_size = int(n_prior*(1-test_ratio))
        test_size = int(n_prior*test_ratio)
    else:
        train_size = int(n_samples*(1-test_ratio))
        test_size = int(n_samples*test_ratio)

    return train_test_split(X_flattened, y_onehot, p_switch, train_size=train_size, test_size=test_size, shuffle=False)

def mnist_model_38(n_samples, test_ratio): # classification
    """
    Function : 
                Preprocesses the MNIST dataset for training and testing, focusing on digits 3 and 8.
    """
    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    X = np.concatenate((X_train, X_test), axis=0)
    y = np.concatenate((y_train, y_test), axis=0)
    X_flattened = np.array([x.flatten() for x in X])  # Flatten the images
    X_new = X_flattened[(y==3)|(y==8)] # only the 3 and the 8
    y_new = y[(y==3)|(y==8)]
    n_prior = len(y_new)
    y_onehot = hf.one_hot(y_new) # Transforming digit target (3, 8) into vector with 0 and 1 at the digit's position
    p_switch = np.zeros(n_prior)
    if n_samples > n_prior:
        train_size = int(n_prior*(1-test_ratio))
        test_size = int(n_prior*test_ratio)
    else:
        train_size = int(n_samples*(1-test_ratio))
        test_size = int(n_samples*test_ratio)

    return train_test_split(X_new, y_onehot, p_switch, train_size=train_size, test_size=test_size, shuffle=False)

def horizontal_vertical(n_samples, test_size=.3, lamb=.2): # classification
    """
    Function : vertical and horizontal bars dataset,
                the psw is a dummy psw that is 
                above 0.5 for horizontal
                and below 0.5 for vertical. 
                This psw can be used without the mtj 
                or is updated by the mtj values during 
                the experiment
    Parameters :
                n_samples : int : number of total (training + testing) samples 
                                = number of vertical and horizontal bars to create
                test_size : float : between 0 and 1 : percentage of the total number 
                                    of samples used for testing the model (doing the 
                                    actual prediction/classification)
                lamb : float : lambda of Poisson law, only relevant if one wants to 
                                quantitatively determine the difficulty to separate
                                the psw as a function of the "mixing" (one class switching
                                too much in the domain of the other)
    Returns :
                X : np.array : array of all the training inputs (send to mtj)
                                example : [[0,0,1,1], [0,1,0,1], [0,1,0,1], [0,0,1,1], ...]
                Xt : np.array : array of all the testing inputs
                y : np.array : array of all the training targets to be put in the regression models
                                example : [[1 0], [0, 1], [0,1], [1,0], ...]
                yt : np.array : array of all the testing targets
                p : np.array : array of all the dummy training psw, to be updated with the correct 
                                values from the experiment (already has the right size)
                pt : np.array : array of all the dummy testing psw, to be updated with the correct 
                                values from the experiment (already has the right size)
    """
    rand = rng.random(n_samples)
    input_in_mtj = np.zeros((n_samples, 4)) # 2x2 array
    input_in_mtj[rand<.5] = np.array([0,0,1,1]) 
    input_in_mtj[rand>.5] = np.array([0,1,0,1])
    target = np.zeros((n_samples, 2)) 
    target[rand<.5] = np.array([0,1]) # horizontal
    target[rand>.5] = np.array([1,0]) # vertical
    Psw = np.zeros(n_samples)
    Psw[rand<.5] = rand[rand<.5] + 0.05*np.random.poisson(lam=lamb)
    Psw[rand>.5] = rand[rand>.5] - 0.05*np.random.poisson(lam=lamb)
    X, Xt, y, yt, p, pt = train_test_split(input_in_mtj, target, Psw, test_size=test_size)

    return X, Xt, y, yt, p, pt

def skdigits_model_38(n_samples, test_ratio): # classification
    """
    Function : same as horizontal_vertical but with 3 and 8 digits from sk-learn digits database
    """
    X, y = load_digits(return_X_y=True) # Loading data from sklearn digits
    X_new = X[(y==3)|(y==8)] # only the 3 and the 8
    y_new = y[(y==3)|(y==8)]
    n_prior = len(y_new)
    y_onehot = hf.one_hot(y_new) # # Transforming digit target (3, 8) into vector with 0 and 1 at the digit's position
    p_switch = np.zeros(n_prior)
    if n_samples > n_prior:
        train_size = int(n_prior*(1-test_ratio))
        test_size = int(n_prior*test_ratio)
    else:
        train_size = int(n_samples*(1-test_ratio))
        test_size = int(n_samples*test_ratio)
    X_train, X_test, y_train, y_test, p_train, p_test = train_test_split(X_new, y_onehot, p_switch, train_size=train_size, test_size=test_size, shuffle=False)

    return X_train, X_test, y_train, y_test, p_train, p_test

def skdigits_model_17(n_samples, test_ratio): # classification
    """
    Function : same as horizontal_vertical but with 1 and 7 digits from sk-learn digits database
    """
    X, y = load_digits(return_X_y=True) # Loading data from sklearn digits
    X_new = X[(y==1)|(y==7)] # only the 1 and the 7
    y_new = y[(y==1)|(y==7)]
    n_prior = len(y_new)
    y_onehot = hf.one_hot(y_new) # Transforming digit target (1, 7) into vector with 0 and 1 at the digit's position
    p_switch = np.random.rand(n_prior)
    if n_samples > n_prior:
        train_size = int(n_prior*(1-test_ratio))
        test_size = int(n_prior*test_ratio)
    else:
        train_size = int(n_samples*(1-test_ratio))
        test_size = int(n_samples*test_ratio)
    X_train, X_test, y_train, y_test, p_train, p_test = train_test_split(X_new, y_onehot, p_switch, train_size=train_size, test_size=test_size, shuffle=False)

    return X_train, X_test, y_train, y_test, p_train, p_test

def skdigits_model_1738(n_samples, test_ratio): # classification
    """
    Function : same as horizontal_vertical but with 1, 3, 8 and 7 digits from sk-learn digits database
                --> this is already multinomial classification as it is not binary anymore
    """
    X, y = load_digits(return_X_y=True) # Loading data from sklearn digits
    X_new = X[(y==3)|(y==8)|(y==1)|(y==7)]
    y_new = y[(y==3)|(y==8)|(y==1)|(y==7)]
    n_prior = len(y_new)
    y_onehot = hf.one_hot(y_new) # Transforming digit target (1, 7, 3, 8) into vector with 0 and 1 at the digit's position
    p_switch = np.zeros(n_prior)
    if n_samples > n_prior:
        train_size = int(n_prior*(1-test_ratio))
        test_size = int(n_prior*test_ratio)
    else:
        train_size = int(n_samples*(1-test_ratio))
        test_size = int(n_samples*test_ratio)
    X_train, X_test, y_train, y_test, p_train, p_test = train_test_split(X_new, y_onehot, p_switch, train_size=train_size, test_size=test_size, shuffle=False)

    return X_train, X_test, y_train, y_test, p_train, p_test

def skdigits_model(n_samples, test_ratio): # classification
    """
    Function : same as all above but now it is the 10 different digits from the sk-learn digits database
    """
    X, y = load_digits(return_X_y=True) # Loading data from sklearn digits
    n_prior = len(y)
    y_onehot = hf.one_hot(y) # Transforming digit target (e.g. 1, 7, 6, ...) into vector with 0 and 1 at the digit's position
    p_switch = np.zeros(n_prior)
    if n_samples > n_prior:
        train_size = int(n_prior*(1-test_ratio))
        test_size = int(n_prior*test_ratio)
    else:
        train_size = int(n_samples*(1-test_ratio))
        test_size = int(n_samples*test_ratio)
    X_train, X_test, y_train, y_test, p_train, p_test = train_test_split(X, y_onehot, p_switch, train_size=train_size, test_size=test_size, shuffle=False)

    return X_train, X_test, y_train, y_test, p_train, p_test

def periodic_classification_01_011_0011(n_samples, stream_length, test_ratio): # classification of periods
    """
    Function : Creates a dataset with periodic streams (01, 011, 0011) for classification.
               Each sample is randomly selected from one of the three stream types.
               The target is one-hot encoded based on which stream type was selected.
    Parameters :
                n_samples : int : number of total (training + testing) samples 
                test_ratio : float : between 0 and 1 : percentage of the total number 
                                    of samples used for testing the model
    Returns :
                X_train : np.array : array of all the training inputs (periodic streams)
                X_test : np.array : array of all the testing inputs
                y_train : np.array : array of all the training targets (one-hot encoded)
                y_test : np.array : array of all the testing targets
                p_train : np.array : array of all the set-to-zero training psw
                p_test : np.array : array of all the set-to-zero testing psw
    """
    # Define stream length (you can adjust this parameter)
    stream_length = stream_length  # Should be divisible by 2, 3, and 4 for clean patterns
    
    # Generate random choices for stream types (0: 01, 1: 011, 2: 0011)
    stream_choices = rng.integers(0, 3, size=n_samples)
    
    # Initialize arrays
    X = np.zeros((n_samples, stream_length))
    y = np.zeros(n_samples, dtype=int)
    
    # Generate streams based on random choices
    for i in range(n_samples):
        if stream_choices[i] == 0:  # stream_01
            X[i] = np.array(hf.stream_01(stream_length))
            y[i] = 0
        elif stream_choices[i] == 1:  # stream_011
            X[i] = np.array(hf.stream_011(stream_length))
            y[i] = 1
        else:  # stream_0011
            X[i] = np.array(hf.stream_0011(stream_length))
            y[i] = 2
    
    # Convert targets to one-hot encoding
    y_onehot = hf.one_hot(y)
    
    # Create dummy psw array
    p_switch = np.zeros(n_samples)
    
    # Split into training and testing sets
    train_size = int(n_samples * (1 - test_ratio))
    test_size = int(n_samples * test_ratio)
    
    X_train, X_test, y_train, y_test, p_train, p_test = train_test_split(
        X, y_onehot, p_switch, train_size=train_size, test_size=test_size, shuffle=False
    )
    
    return X_train, X_test, y_train, y_test, p_train, p_test



def periodic_classification_01_011(n_samples, stream_length, test_ratio): # classification of periods
    """
    Function : Creates a dataset with periodic streams (01, 011) for classification.
               Each sample is randomly selected from one of the two stream types.
               The target is one-hot encoded based on which stream type was selected.
    Parameters :
                n_samples : int : number of total (training + testing) samples 
                test_ratio : float : between 0 and 1 : percentage of the total number 
                                    of samples used for testing the model
    Returns :
                X_train : np.array : array of all the training inputs (periodic streams)
                X_test : np.array : array of all the testing inputs
                y_train : np.array : array of all the training targets (one-hot encoded)
                y_test : np.array : array of all the testing targets
                p_train : np.array : array of all the set-to-zero training psw
                p_test : np.array : array of all the set-to-zero testing psw
    """
    # Define stream length (you can adjust this parameter)
    stream_length = stream_length  # Should be divisible by 2, 3, and 4 for clean patterns
    
    # Generate random choices for stream types (0: 01, 1: 011, 2: 0011)
    stream_choices = rng.integers(0, 2, size=n_samples)
    
    # Initialize arrays
    X = np.zeros((n_samples, stream_length))
    y = np.zeros(n_samples, dtype=int)
    
    # Generate streams based on random choices
    for i in range(n_samples):
        if stream_choices[i] == 0:  # stream_01
            X[i] = np.array(hf.stream_01(stream_length))
            y[i] = 0
        elif stream_choices[i] == 1:  # stream_011
            X[i] = np.array(hf.stream_011(stream_length))
            y[i] = 1
        
    # Convert targets to one-hot encoding
    y_onehot = hf.one_hot(y)
    
    # Create dummy psw array
    p_switch = np.zeros(n_samples)
    
    # Split into training and testing sets
    train_size = int(n_samples * (1 - test_ratio))
    test_size = int(n_samples * test_ratio)
    
    X_train, X_test, y_train, y_test, p_train, p_test = train_test_split(
        X, y_onehot, p_switch, train_size=train_size, test_size=test_size, shuffle=False
    )
    
    return X_train, X_test, y_train, y_test, p_train, p_test

def predict_mackey_glass(datapoints: int = 100, crop_out_every_n_samples: int = 2): # prediction
    """
    Generate the Mackey-Glass time series data.
    The function generates a time series based on the Mackey-Glass equation.
    Each data point is a float representing the state of the system at that time.
    """
    mg_data = hf.mackey_glass()
    cropped_data = mg_data[::crop_out_every_n_samples]
    n_prior = len(cropped_data)
    if n_prior <= datapoints:
        final_data = cropped_data
    elif n_prior > datapoints:
        final_data = cropped_data[:datapoints]

    return final_data

def predict_lorenz_attractor(datapoints: int = 100, crop_out_every_n_samples: int = 2): # prediction
    """
    Generate the Lorenz attractor time series data.
    The function generates a time series based on the Lorenz attractor equations.
    Each data point is a list of three values representing the state of the system.
    """
    lorenz_data = hf.lorenz_attractor()
    cropped_data = lorenz_data[::crop_out_every_n_samples]
    n_prior = len(cropped_data)
    if n_prior <= datapoints:
        final_data = cropped_data
    elif n_prior > datapoints:
        final_data = cropped_data[:datapoints]

    return final_data