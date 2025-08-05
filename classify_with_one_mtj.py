import os
import numpy as np
import magnet_control
import experiment_control
import close_connections
import datasets as ds
import helper_functions as hf
from output_layer import classifier_linear_regression, classifier_ridge_regression
from sklearn.model_selection import train_test_split
from add_filter import add_filter

def main():

    file_path = r"Y:\SOT_lab\People\Dashiell\fast_codes_folder\first_try_several_psw_38" # you must create the folders before writing the path

    ######## Create a dataset ########
    samples_number = 100
    test_ratio = 0.2 # ratio of the dataset to be used for testing
    X_train_before, X_test_before, y_train, y_test, _, _ = ds.mnist_model_38(samples_number, test_ratio)
    X_train_copy = X_train_before.copy()
    X_test_copy = X_test_before.copy()

    ######## Add features, if no features to be added, set n_features=1 ########
    n_features = 1 # also good for output layer.
    filter_type = 'none' # 'none', 'average', 'short', 'long', 'sparse'
    X_train_before, psw_train = hf.add_features(X_train_before, n_features)
    X_test_before, psw_test = hf.add_features(X_test_before, n_features)
    X_train, X_test = add_filter(file_path, X_train_before, X_test_before, filter_type=filter_type)

    ######## Experiment parameters ########
    reference_resistance = 100 # reference resistance for the PXI
    resistance_measurement_technique =  'lock_in_fast' #'kreuken_accurate', 'kreuken_fast', 'lock_in_accurate', 'lock_in_fast'
    delimitation_resistance = 2000 # clear limit between P and AP state of the MTJ in the resistance measurement technique you choose
    low_to_high = True

    ######## Reset pulse parameters ########
    reset_V_amp = -1.0 # in V
    reset_tp = 100 # in nanoseconds
    reset_ratio = 1.

    ######## Forward pulse parameters ########
    V_max = 0.9 # in V
    pulse_duration = 11. # in nanoseconds
    forward_ratio = 0.85
    N_attempts = 15  # decide the number of switching attempts to get the switching probability

    ######## Apply magnetic field ########
    B_x = -3.0 # in A
    magnet = magnet_control.Danfysik7000()
    magnet.rampToCurrent(B_x)

    ######## Instantiate the experiment model ########
    controller = experiment_control.psw_control(low_to_high, resistance_measurement_technique)

    ######## Run the experiment : get all the training data first then all the testing data ########
    newpath = file_path + "\\raw_data"
    if not os.path.exists(newpath):
        os.makedirs(newpath)
    ######## Run the experiment : get all the training data first then all the testing data ########
    counter = 0 # counter for refreshing the awg
    for i in range(len(psw_train)):
        max_value = np.max(X_train[i])
        for j in range(n_features):
            file_number = "train" + str(i) + "_feature" + str(j)
            pulse = X_train[i][j]/max_value  # Normalize pulse to max value of 1
            pulse = pulse.tolist()
            psw_train[i][j] = controller.switching_probability(signal_array=pulse, pulse_duration=pulse_duration,
                                                            pulse_voltage=V_max, ref_resistance=reference_resistance,
                                                            r_low=delimitation_resistance, reset_duration=reset_tp,
                                                            reset_voltage=reset_V_amp, n_trials=N_attempts,
                                                            ch4_forward_amplitude_factor=forward_ratio,
                                                            ch4_reset_amplitude_factor=reset_ratio,
                                                            magnetic_field=B_x, filenumber = file_number, 
                                                            filepath = newpath
                                                            )
            counter += 1
            if counter % (10*n_features) == 0:  # Refresh AWG every 10 samples
                awg = experiment_control.AWG()
                awg.close()
    for i in range(len(psw_test)):
        max_value = np.max(X_test[i]) 
        for j in range(n_features):
            file_number = "test" + str(i) + "_feature" + str(j)
            pulse = X_test[i][j]/max_value  # Normalize pulse to max value of 1
            pulse = pulse.tolist()
            psw_test[i][j] = controller.switching_probability(signal_array=pulse, pulse_duration=pulse_duration,
                                                            pulse_voltage=V_max, ref_resistance=reference_resistance,
                                                            r_low=delimitation_resistance, reset_duration=reset_tp,
                                                            reset_voltage=reset_V_amp, n_trials=N_attempts, 
                                                            ch4_forward_amplitude_factor=forward_ratio,
                                                            ch4_reset_amplitude_factor=reset_ratio,
                                                            magnetic_field=B_x, filenumber = file_number, 
                                                            filepath = newpath
                                                            )
            counter += 1
            if counter % (10*n_features) == 0:  # Refresh AWG every 10 samples 
                awg = experiment_control.AWG()
                awg.close()
        
    ######## Close all connections ########
    close_connections.cleanup_setup()

    output_layer_path = newpath + "\\output_layer"
    if not os.path.exists(output_layer_path):
        os.makedirs(output_layer_path)

    newpath_wr = output_layer_path + "\\with_reservoir"
    if not os.path.exists(newpath_wr):
        os.makedirs(newpath_wr)

    ######## Cross validation ########
    number_folds = 5
    y_total = np.concatenate((y_train, y_test))
    psw_total = np.concatenate((psw_train, psw_test))
    cv_accuracies = []
    for k in range(number_folds):
        y_train_cv, y_test_cv, psw_train_cv, psw_test_cv = train_test_split(y_total, psw_total, test_size=test_ratio, shuffle=True)
        ######## Train and test the linear regression model ########
        lin = classifier_linear_regression(n_features=n_features)  # Instanciate
        lin.train(training_target=y_train_cv, training_Pswitch=psw_train_cv)  # Train
        y_from_lin = lin.test(test_Pswitch=psw_test_cv)  # Test
        lin_accu, lin_conf = hf.scores(target=y_test_cv, prediction=y_from_lin)
        lin_coef, lin_intercept = lin.matrix()

        ######## Train and test the ridge regression model for all alphas ########
        alphas = np.linspace(0.1, 1)
        ridge_accuracies = []
        ridge_conf_matrices = []
        ridge_coeffs = []
        ridge_intercepts = []
        ridge_predictions = []
        for alpha in alphas:
            ridge = classifier_ridge_regression(beta=alpha, n_features=n_features)  # Instanciate
            ridge.train(training_target=y_train_cv, training_Pswitch=psw_train_cv)  # Train
            y_from_ridge = ridge.test(test_Pswitch=psw_test_cv)  # Test
            rdg_accu, rdg_conf = hf.scores(target=y_test_cv, prediction=y_from_ridge)
            rdg_coef, rdg_intercept = ridge.matrix()
            ridge_conf_matrices.append(rdg_conf)
            ridge_accuracies.append(rdg_accu)
            ridge_coeffs.append(rdg_coef)
            ridge_intercepts.append(rdg_intercept)
            ridge_predictions.append(y_from_ridge)
        cv_accuracies.append(max(lin_accu, np.max(ridge_accuracies)))

        ######## Save the data from the current trial and current fold ########
        file_directory = newpath_wr + "\\resultsCV" + str(k) + ".dat"
        file = open(file_directory , "w+")
        file.write("Training targets : \n")
        np.savetxt(file, y_train_cv, fmt='%.2f')
        #file.write("\nTraining data : \n")
        #for i in range(len(X_train)):
        #    np.savetxt(file, X_train[i], fmt='%.2f')
        file.write("\nSwitching probabilities from training data : \n") 
        np.savetxt(file, psw_train_cv, fmt='%.4f')
        file.write("\nTesting targets : \n") 
        np.savetxt(file, y_test_cv, fmt='%.2f')
        #file.write("\nTesting data : \n")
        #for i in range(len(X_test)):
        #    np.savetxt(file, X_test[i], fmt='%.2f')
        file.write("\nSwitching probabilities from testing data : \n") 
        np.savetxt(file, psw_test_cv, fmt='%.4f')
        file.write("\nLinear model: \n" )
        file.write("\nAccuracy : " + str(lin_accu) + "\n") 
        file.write("\nConfusion matrix : \n") 
        np.savetxt(file, lin_conf)
        file.write("\nCoefficients : \n")
        np.savetxt(file, lin_coef)
        file.write("\nIntercept : \n")
        np.savetxt(file, lin_intercept)
        file.write("\nPrediction : \n") 
        np.savetxt(file, y_from_lin, fmt='%.2f')
        file.write("\n")
        file.write("\nRidge model: \n" )
        file.write("\nAccuracies : ")
        for i, accu in enumerate(ridge_accuracies):
            file.write(f"\nAccuracy for alpha {alphas[i]:<.2f}: {accu}\n")
        file.write("\nConfusion matrices : \n")
        for i, conf in enumerate(ridge_conf_matrices):
            file.write(f"\nConfusion matrix for alpha {alphas[i]:<.2f}:\n")
            np.savetxt(file, conf)
        file.write("\nCoefficients : \n")
        for i, coeffs in enumerate(ridge_coeffs):
            file.write(f"\nCoefficients for alpha {alphas[i]:<.2f}:\n")
            np.savetxt(file, coeffs)
        file.write("\nIntercepts : \n")
        for i, intercept in enumerate(ridge_intercepts):
            file.write(f"\nIntercept for alpha {alphas[i]:<.2f}:\n")
            np.savetxt(file, intercept)
        for i, preds in enumerate(ridge_predictions):
            file.write(f"\nPredictions for alpha {alphas[i]:<.2f}:\n")
            np.savetxt(file, preds, fmt='%.2f')
        file.close()
    
    # Save total file    
    file_directory = newpath_wr + "\\results_highlight.dat"
    file = open(file_directory , "w+")
    file.write("Number of training samples: " + str(len(X_train)) + "\n")
    file.write("Number of testing samples: " + str(len(X_test)) + "\n")
    file.write("Number of features: " + str(n_features) + "\n")
    file.write(f"Accuracies of {format(number_folds)}-fold cross-validation : \n")
    np.savetxt(file, cv_accuracies, fmt='%.2f')
    file.close()

    ######## No reservoir comparison ########
    file_no_reservoir = output_layer_path + "\\no_reservoir"
    if not os.path.exists(file_no_reservoir):
        os.makedirs(file_no_reservoir)
    n_features_no_reservoir = X_train_copy[0].shape[0]
    y_total = np.concatenate((y_train, y_test))
    X_total = np.concatenate((X_train_copy, X_test_copy))
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

    # Optional: Copy the current file to a new location
    with open(__file__, "r") as src:
        name = file_path + "\\example_classifying_task_copy.py"
        with open(name, "w") as tgt:
            tgt.write(src.read())
            
if __name__ == "__main__":
   main()