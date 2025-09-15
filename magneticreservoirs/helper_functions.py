"""
All the helper functions used in either prediction or classification tasks.
Next to each function is written for which task it is destined.
"""
import os
import numpy as np
from itertools import product
from sklearn.metrics import accuracy_score, confusion_matrix, mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
from typing import Union, List
import math

rng = np.random.default_rng(2025)

def one_hot(y): # classification
    """
    Function : one-hot a n-dimensional np.array(int) into a 
                n-dimensional np.array(n_class dimensional np.arrays(floats))
    Parameters :
                y : np.array(int)
    Returns :
                y_onehot : np.array(np.array(float))
    """
    unik = np.unique(y)
    for i,k in product(range(len(y)), range(len(unik))):
        if y[i] == unik[k]:
            y[i] = k
    n_class = len(np.unique(y))
    y_onehot = np.zeros((len(y), n_class))
    for i in range(len(y_onehot)):
        y_onehot[i,y[i]]= 1.

    return y_onehot


def inverse_one_hot(array): # classification
    """
    Function : input a n*d array of n d-dimensional vectors, 
               returns a n*1 array of each vector's inverse
               one-hot integer. 
    Parameters :
                array : np.array(np.array(floats))
    Returns :
                new_array : np.array(int)
    """
    new_array = np.zeros(len(array))
    for i in range(len(array)):
        new_array[i] = np.argmax(array[i])

    return new_array

def scores(target, prediction): #classification
    """
    Function : computes the accuracy and confusion matrix of a classification task
                accuracy : The accuracy is the percentage of truely predicted values.
                confusion matrix : A confusion matrix C is such that C_ij is equal 
                                    to the number of observations known to be 
                                    in group i and predicted to be in group j.
                                    For BINARY classification, the count of 
                                    true negatives is C_00
                                    false negatives is C_10
                                    true positives is C_11
                                    false positives is C_01
    Parameters :
                target : np.array(np.array(int)) : true values
                prediction : np.array(np.array(int)) : predicted values from our model
    Returns :
                accuracy_score : float : accuracy
                confusion_matrix : np.array(int): confusion matrix
    """
    target = inverse_one_hot(target)
    prediction = inverse_one_hot(prediction)

    return accuracy_score(target, prediction), confusion_matrix(target, prediction)


def add_features(X, n_features): # classification
    """
    Function : adds n_features to the input X, which is a m*n array,
                by separating the array into n_features vectors of n/n_features.
                The idea is to get one psw per feature, so that each vector of X 
                has n_features psw associated to it.
    """

    new_psw = np.zeros((len(X), n_features))
    list_from_X = []
    for x in X:
        list_from_x = []
        n_samples = len(x)
        if n_features > n_samples:
            raise ValueError("n_features cannot be greater than the length of the vector")
        number_of_elements_per_feature = n_samples // n_features
        for i in range(n_features):
            list_from_x.append(x[i * number_of_elements_per_feature : (i + 1) * number_of_elements_per_feature])
        list_from_X.append(list_from_x)

    return np.array(list_from_X), new_psw

def offsetter(data: Union[np.ndarray, List], offset: float) -> None: # classification AND prediction
    """
    Offset every float value in an array/list by a fixed offset value (in-place modification).
    
    Parameters:
    -----------
    data : np.ndarray or list
        The input data structure to modify in-place
    offset : float
        The fixed value to add to each numeric element

    Returns:
    None
        This function modifies the original data structure instead of creating a copy.
    
    """
    # Handle NumPy arrays
    if isinstance(data, np.ndarray):
        if np.issubdtype(data.dtype, np.number):
            data += offset
    
    # Handle Python lists
    elif isinstance(data, list):
        for i, item in enumerate(data):
            if isinstance(item, (int, float, np.number)):
                data[i] = item + offset
            elif isinstance(item, (list, np.ndarray)):
                offsetter(item, offset)


def stream_01(n_samples): # prediction
    pattern = [0, 1]
    result = pattern * (n_samples // 2)
    remainder = n_samples % 2
    if remainder > 0:
        result.extend(pattern[:remainder])
        
    return result

def stream_011(n_samples): # prediction
    pattern = [0, 1, 1]
    result = pattern * (n_samples // 3)
    remainder = n_samples % 3
    if remainder > 0:
        result.extend(pattern[:remainder])

    return result

def stream_0011(n_samples): # prediction
    pattern = [0, 0, 1, 1]
    result = pattern * (n_samples // 4)
    remainder = n_samples % 4
    if remainder > 0:
        result.extend(pattern[:remainder])

    return result

def prediction_scores(X_test, X_predicted): # prediction
    mse = mean_squared_error(X_test, X_predicted)
    mae = mean_absolute_error(X_test, X_predicted)
    rmse = np.sqrt(mse)
    mape = mean_absolute_percentage_error(X_test, X_predicted)

    return mse, mae, rmse, mape

def mackey_glass(): # prediction
    """
    Generate the Mackey-Glass time series data.
    The function generates a time series based on the Mackey-Glass equation.
    """
    t_min = 18
    t_max = 1100
    beta = 0.2
    gamma = 0.1
    tao = 17
    n = 10
    x = []
    for i in range(1, t_min) :
        x.append(0.0)
    x.append(1.2)
    for t in range(t_min, t_max):
        h = x[t-1] + (beta * x[t-tao-1] / (1 + math.pow(x[t-tao-1], n))) - (gamma * x[t-1])
        h = float("{:0.4f}".format(h))
        x.append(h)  
        
    return x[t_min-1:t_max]

def lorenz_helper(xyz, *, s=10, r=28, b=2.667): # prediction
    """
    Parameters
    ----------
    xyz : array-like, shape (3,)
       Point of interest in three-dimensional space.
    s, r, b : float
       Parameters defining the Lorenz attractor.

    Returns
    -------
    xyz_dot : array, shape (3,)
       Values of the Lorenz attractor's partial derivatives at *xyz*.
    """
    x, y, z = xyz
    x_dot = s*(y - x)
    y_dot = r*x - y - x*z
    z_dot = x*y - b*z

    return np.array([x_dot, y_dot, z_dot])

def lorenz(): # prediction
    
    dt = 0.01
    num_steps = 10000
    xyzs = np.empty((num_steps + 1, 3))  # Need one more for the initial values
    xyzs[0] = (0., 1., 1.05)  # Set initial values
    for i in range(num_steps):
        xyzs[i + 1] = xyzs[i] + lorenz_helper(xyzs[i]) * dt

    return xyzs.tolist()

def add_filter(file_path, X_train, X_test, filter_type='short', average_sequence_size=5, 
               sparse_density=0.6, random_0_1_sequence_size=8, random_minus_1_plus_1_sequence_size=100):

    filter_path = file_path + "\\input_layer"
    if not os.path.exists(filter_path):
        os.makedirs(filter_path)

    X_test_copy2 = X_test.copy()
    X_train_copy2 = X_train.copy()

    if filter_type == 'average':
        sequence_size = average_sequence_size
        new_size_train = int(X_train.shape[2] / sequence_size)
        new_size_test = int(X_test.shape[2] / sequence_size)
        X_train = np.zeros(shape=(X_train.shape[0], X_train.shape[1], new_size_train))
        X_test = np.zeros(shape=(X_test.shape[0], X_test.shape[1], new_size_test))
        for i in range(len(X_test)):
            X_test[i] = average_convolution(X_test_copy2[i], sequence_size)
        for i in range(len(X_train)):
            X_train[i] = average_convolution(X_train_copy2[i], sequence_size)
        file_directory = filter_path + "\\average.dat"
        file = open(file_directory, "w+")
        file.write("Sequence size: " + str(sequence_size) + "\n")
        file.close()

    elif filter_type == "sparse":
        density = sparse_density
        rc = rc_mapping(length=X_train.shape[2], density=density)
        rand_matrix_train = []
        rand_matrix_test = []
        for i in range(len(X_train)):
            rand_matrix_tr, X_train[i] = rc.sparse(X_train_copy2[i])
            rand_matrix_train.append(rand_matrix_tr)
        for i in range(len(X_test)):
            rand_matrix_te, X_test[i] = rc.sparse(X_test_copy2[i])
            rand_matrix_test.append(rand_matrix_te)

        file_directory = filter_path + "\\sparse.dat"
        file = open(file_directory, "w+")
        file.write("Density: " + str(density) + "\n")
        file.write("Input layer: \n")
        np.savetxt(file, rand_matrix_train[0], fmt='%.2f')
        file.close()

    elif filter_type == "short":
        sequence_size = random_0_1_sequence_size
        X_train = np.zeros(shape=(X_train.shape[0], X_train.shape[1], sequence_size))
        X_test = np.zeros(shape=(X_test.shape[0], X_test.shape[1], sequence_size))
        rc = rc_mapping(length=X_train_copy2.shape[2], size=sequence_size)
        rand_matrix_train = []
        rand_matrix_test = []
        for i in range(len(X_train)):
            rand_matrix_tr, X_train[i] = rc.small(X_train_copy2[i])
            rand_matrix_train.append(rand_matrix_tr)
        for i in range(len(X_test)):
            rand_matrix_te, X_test[i] = rc.small(X_test_copy2[i])
            rand_matrix_test.append(rand_matrix_te)
        file_directory = filter_path + "\\short.dat"
        file = open(file_directory, "w+")
        file.write("Sequence size: " + str(sequence_size) + "\n")
        file.write("Input layer: \n")
        np.savetxt(file, rand_matrix_train[0], fmt='%.2f')
        file.close()

    elif filter_type == "long":
        sequence_size = random_minus_1_plus_1_sequence_size
        X_train = np.zeros(shape=(X_train.shape[0], X_train.shape[1], sequence_size))
        X_test = np.zeros(shape=(X_test.shape[0], X_test.shape[1], sequence_size))
        rc = rc_mapping(length=X_train_copy2.shape[2], size=sequence_size)
        rand_matrix_train = []
        rand_matrix_test = []
        for i in range(len(X_train)):
            rand_matrix_tr, X_train[i] = rc.big(X_train_copy2[i])
            rand_matrix_train.append(rand_matrix_tr)
        for i in range(len(X_test)):
            rand_matrix_te, X_test[i] = rc.big(X_test_copy2[i])
            rand_matrix_test.append(rand_matrix_te)

        file_directory = filter_path + "\\long.dat"
        file = open(file_directory, "w+")
        file.write("Sequence size: " + str(sequence_size) + "\n")
        file.write("Input layer: \n")
        np.savetxt(file, rand_matrix_train[0], fmt='%.2f')
        file.close()

    elif filter_type == "none":
        pass

    return X_train, X_test


##################################### Initial filters for classification ################################################

def average_convolution(x, segment_length): # of a n*m array
    n_sample = len(x)
    new_length = int(x.shape[1]/segment_length)
    x_new = np.zeros((n_sample, new_length))
    for i in range(n_sample):
        for j in range(new_length):
            x_new[i][j] = np.sum(x[i][j*segment_length:(j+1)*segment_length]) / segment_length
            
    return x_new

class rc_mapping: 

    def __init__(self, length, density = 0.3, size = 8):
        self.length = length
        self.size = size
        self.density = density
        self.matrix_sparse = np.eye(length)
        self.matrix_small = rng.random((length, size))
        self.matrix_big = 2 * (rng.random((length, size)) - .5) # random matrix between - 1 and + 1
        for i in range(length):
            if rng.random() > density :
                self.matrix_sparse[i] = np.zeros(length)

    def sparse(self, x): # picks out just a few pixels of the 64 pixels image
        x_new = np.zeros((len(x), self.length))
        for i in range(len(x)):
            x_new[i] = np.matmul(x[i],self.matrix_sparse)

        return self.matrix_sparse, x_new

    def small(self, x): # multiplies the images by a fixed random matrix to bring it down to a few positive pulses (size positive pulses)
        x_new = np.zeros((len(x), self.size))
        for i in range(len(x)):
            x_new[i] = np.matmul(x[i], self.matrix_small)

        return self.matrix_small, x_new

    def big(self, x): # multiplies the images by a fixed random matrix to bring it to numerous relative pulses (positive and negative) 
        x_new = np.zeros((len(x), self.size))
        for i in range(len(x)):
            x_new[i] = np.matmul(x[i],self.matrix_big)

        return self.matrix_big, x_new

##################################### End of initial filters for classification ################################################

if __name__ == "__main__":
    pass
    
   