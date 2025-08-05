import os
import numpy as np
import helper_functions as hf

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
            X_test[i] = hf.average_convolution(X_test_copy2[i], sequence_size)
        for i in range(len(X_train)):
            X_train[i] = hf.average_convolution(X_train_copy2[i], sequence_size)
        file_directory = filter_path + "\\average.dat"
        file = open(file_directory, "w+")
        file.write("Sequence size: " + str(sequence_size) + "\n")
        file.close()

    elif filter_type == "sparse":
        density = sparse_density
        rc = hf.rc_mapping(length=X_train.shape[2], density=density)
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
        rc = hf.rc_mapping(length=X_train_copy2.shape[2], size=sequence_size)
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
        rc = hf.rc_mapping(length=X_train_copy2.shape[2], size=sequence_size)
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