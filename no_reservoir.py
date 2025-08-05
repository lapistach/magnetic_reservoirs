import datasets as ds
import helper_functions as hf
from output_layer import classifier_linear_regression, classifier_ridge_regression
import numpy as np
import os
from sklearn.model_selection import train_test_split

import time

def main():

    ######## In every code ########
    file_path = r"Y:\SOT_lab\People\Dashiell\fast_codes_folder\default" 
    samples_number = 500
    test_ratio = .2
    X_train_no_reservoir, X_test_no_reservoir, y_train, y_test, _, _ = ds.mnist_model(samples_number, test_ratio) 
    number_folds = 5
    alphas = np.linspace(0.1, 1)

    ######## No reservoir comparison ########
    start_time = time.time()
    file_no_reservoir = file_path + "\\no_reservoir"
    if not os.path.exists(file_no_reservoir):
        os.makedirs(file_no_reservoir)
    n_features_no_reservoir = X_train_no_reservoir[0].shape[0]
    y_total = np.concatenate((y_train, y_test))
    X_total = np.concatenate((X_train_no_reservoir, X_test_no_reservoir))
    cv_accu_no_reservoir_racies = []
    for k in range(number_folds):
        y_train_no_reservoir, y_test_no_reservoir, X_train_no_reservoir, X_test_no_reservoir = train_test_split(y_total, X_total, test_size=test_ratio, shuffle=True)  
        lin = classifier_linear_regression(n_features=n_features_no_reservoir)  
        lin.train(training_target=y_train_no_reservoir, training_Pswitch=X_train_no_reservoir) 
        y_from_lin = lin.test(test_Pswitch=X_test_no_reservoir) 
        lin_accu_no_reservoir, _ = hf.scores(target=y_test_no_reservoir, prediction=y_from_lin)
        ridge_accu_no_reservoir_racies = []
        for alpha in alphas:
            ridge = classifier_ridge_regression(beta=alpha, n_features=n_features_no_reservoir)  
            ridge.train(training_target=y_train_no_reservoir, training_Pswitch=X_train_no_reservoir)  
            y_from_ridge = ridge.test(test_Pswitch=X_test_no_reservoir) 
            rdg_accu_no_reservoir, _ = hf.scores(target=y_test_no_reservoir, prediction=y_from_ridge)
            ridge_accu_no_reservoir_racies.append(rdg_accu_no_reservoir)
        cv_accu_no_reservoir_racies.append(max(lin_accu_no_reservoir, np.max(ridge_accu_no_reservoir_racies)))
        file_directory = file_no_reservoir + "\\resultsCV" + str(k) + ".dat"
        file = open(file_directory , "w+")
        file.write("Number of training samples: " + str(len(X_train_no_reservoir)) + "\n")
        file.write("Number of testing samples: " + str(len(X_test_no_reservoir)) + "\n")
        file.write("Number of features: " + str(n_features_no_reservoir) + "\n")
        file.write("Linear Regression Accuracy: " + str(lin_accu_no_reservoir) + "\n")
        file.write("Ridge Regression Accuracies: " + str(ridge_accu_no_reservoir_racies) + "\n")
        file.close()
    file = open(file_no_reservoir + "\\highlights.dat", "w")
    file.write("Number of training samples: " + str(len(X_train_no_reservoir)) + "\n")
    file.write("Number of testing samples: " + str(len(X_test_no_reservoir)) + "\n")
    file.write("Number of features: " + str(n_features_no_reservoir) + "\n")
    file.write("Accuracies: " + str(cv_accu_no_reservoir_racies) + "\n")
    file.close()

    end_time = time.time()
    print(f"No reservoir comparison completed in {end_time - start_time:.2f} seconds.")

if __name__ == "__main__":
   main()