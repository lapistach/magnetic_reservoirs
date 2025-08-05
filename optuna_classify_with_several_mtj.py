import os
import optuna
import numpy as np
import run_for_one_mtj 
import datasets as ds
import helper_functions as hf
from output_layer import classifier_linear_regression, classifier_ridge_regression
from sklearn.model_selection import train_test_split

def main():

    file_path = r"Y:\SOT_lab\People\Dashiell\fast_codes_folder\optuna_periodic_class" # you must create the folders before writing the path
    study_name = 'periodic_class' # choose a study name
    n_trials = 1 # choose the number of trials of the study, can be interrupted and changed at any time

    samples_number = 40
    test_ratio = 0.2
    X_train, X_test, y_train, y_test, _, _ = ds.periodic_classification_01_011_0011(samples_number, 60, test_ratio)
    X_train_copy = X_train.copy()
    X_test_copy = X_test.copy()

    ######## Add features, if no features to be added, set n_features=1 ########
    n_features = 1  # Change this to the number of features you want to add 
    X_train, psw_train = hf.add_features(X_train, n_features)
    X_test, psw_test = hf.add_features(X_test, n_features)

    def objective(trial):
        ######## Instantiate the new folder ########
        current_trial_number = trial.number
        newpath = file_path + "\\trial{}".format({current_trial_number})
        if not os.path.exists(newpath):
            os.makedirs(newpath)
            
        ######## Parameters to change in trial ########
        Bx1 = trial.suggest_float("Bx1", -3.0, -2.0, step=0.5)  # example of grid search over a float value
        t_quantized1 = trial.suggest_float("t_quantized1", 0.5, 2.0, step=0.1) # example of grid search over a float value
        V_max1 = trial.suggest_float("V_max1", 0.70, 1.00, step = 0.01)
        forward_ratio1 = trial.suggest_float("forward_ratio1", 0.6, 1.0, step=0.05)

        Bx2 = trial.suggest_float("Bx2", -3.0, -2.0, step=0.5)  # example of grid search over a float value
        t_quantized2 = trial.suggest_float("t_quantized2", 0.5, 2.0, step=0.1) # example of grid search over a float value
        V_max2 = trial.suggest_float("V_max2", 0.70, 1.00, step = 0.01)
        forward_ratio2 = trial.suggest_float("forward_ratio2", 0.6, 1.0, step=0.05)

        list_of_mtjs = {
                "example neuron": ['Bx', 'tq', 'Vm', 'x', 'low_to_high', 'filter_type'],
                "neuron no1": [Bx1, t_quantized1, V_max1, forward_ratio1, True, 'none'],
                "neuron no2": [Bx2, t_quantized2, V_max2, forward_ratio2, True, 'none'],
                "neuron no3": [],
                "neuron no4": [],
                "neuron no5": [],
                "neuron no6": [],
                "neuron no7": [],
                "neuron no8": [],
                "neuron no9": [],
                "neuron no10": [],
                }

        file_directory = newpath + "\\list_of_mtjs.dat"
        with open(file_directory, "w") as file:
            for key, value in list_of_mtjs.items():
                file.write(f"{key}: {value}\n")
        file.close()

        psw_train_of_neurons = []
        psw_test_of_neurons = []    

        n_neurons = 2 # Change this to the number of neurons you want to emulate
        for i in range(n_neurons):
            if not list_of_mtjs["neuron no{}".format(i+1)]:
                continue
            psw_training, psw_testing = run_for_one_mtj.runner_for_one_neuron(newpath, X_train, X_test, 
                                                                                            psw_train, psw_test, n_features, i, 
                                                                                            B_x = list_of_mtjs["neuron no{}".format(i+1)][0], 
                                                                                            t_quantized = list_of_mtjs["neuron no{}".format(i+1)][1], 
                                                                                            V_max = list_of_mtjs["neuron no{}".format(i+1)][2], 
                                                                                            forward_ratio = list_of_mtjs["neuron no{}".format(i+1)][3], 
                                                                                            low_to_high = list_of_mtjs["neuron no{}".format(i+1)][4], 
                                                                                            filter_type = list_of_mtjs["neuron no{}".format(i+1)][5]
                                                                                            )
            psw_train_of_neurons.append(psw_training)
            psw_test_of_neurons.append(psw_testing)

        psw_train_of_neurons = np.asarray(psw_train_of_neurons)
        psw_test_of_neurons = np.asarray(psw_test_of_neurons)
        new_psw_train = np.zeros((psw_train_of_neurons.shape[1], n_features * n_neurons))
        new_psw_test = np.zeros((psw_test_of_neurons.shape[1], n_features * n_neurons))

        for i in range(len(new_psw_train)):
            for j in range(n_neurons):
                new_psw_train[i][j*n_features:(j+1)*n_features] = psw_train_of_neurons[j][i]
        for i in range(len(new_psw_test)):
            for j in range(n_neurons):
                new_psw_test[i][j*n_features:(j+1)*n_features] = psw_test_of_neurons[j][i]

        total_features = n_features * n_neurons  # Update the number of features after adding them for each neuron

        output_layer_path = newpath + "\\output_layer"
        if not os.path.exists(output_layer_path):
            os.makedirs(output_layer_path)

        newpath_wr = output_layer_path + "\\with_reservoir"
        if not os.path.exists(newpath_wr):
            os.makedirs(newpath_wr)

        ######## Cross validation ########
        number_folds = 5
        y_total = np.concatenate((y_train, y_test))
        psw_total = np.concatenate((new_psw_train, new_psw_test))
        cv_accuracies = []
        for k in range(number_folds):
            y_train_cv, y_test_cv, psw_train_cv, psw_test_cv = train_test_split(y_total, psw_total, test_size=test_ratio, shuffle=True)
            ######## Train and test the linear regression model ########
            lin = classifier_linear_regression(n_features=total_features)  # Instanciate
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
                ridge = classifier_ridge_regression(beta=alpha, n_features=total_features)  # Instanciate
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
            file.write("\nSwitching probabilities from training data : \n") 
            np.savetxt(file, psw_train_cv, fmt='%.4f')
            file.write("\nTesting targets : \n") 
            np.savetxt(file, y_test_cv, fmt='%.2f')
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
        file.write("Number of neurons: " + str(n_neurons) + "\n")
        file.write("Number of total features: " + str(total_features) + "\n")
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

        return np.mean(cv_accuracies)  # Return the mean accuracy of the cross-validation
    
    sampler = optuna.samplers.TPESampler() # choose your sampler, this one is the best one if you have categorical parameters
    storage_name = "sqlite:///" + file_path + "\\{}.db".format(study_name)
    study = optuna.create_study(study_name=study_name, storage=storage_name, load_if_exists=True, sampler=sampler, direction="maximize")
    study.optimize(objective, n_trials=n_trials)
    trial = study.best_trial
    best_trial_number = trial.number 

    # Saving the study results
    newpath = file_path + "\\study_results"
    if not os.path.exists(newpath):
        os.makedirs(newpath)
    filedirectory = newpath + "\\"
    df = study.trials_dataframe()
    datafile = filedirectory + "dataframe.csv"
    file = open(datafile, "w+")
    file.close()
    df.to_csv(datafile)
    file = open(filedirectory + "bestresult.dat", "w+")
    file.write(f"The best study showed an accuracy of {format(trial.value)} with parameters {format(trial.params)}\n")
    file.write(f"The best trial number is {best_trial_number}") 
    file.close()

    # Saving plots
    newpath = newpath + "\\plots"
    if not os.path.exists(newpath):
        os.makedirs(newpath)

    fig1 = optuna.visualization.plot_slice(study)
    fig1.write_image(newpath + "\parameter.png") 

    fig2 = optuna.visualization.plot_contour(study, params=["t_quantized1", "V_max1"])
    fig2.write_image(newpath + "\Vmax1_tq1.png")

    fig3 = optuna.visualization.plot_contour(study, params=["t_quantized2", "V_max2"])
    fig3.write_image(newpath + "\Vmax2_tq2.png")

    fig4 = optuna.visualization.plot_contour(study, params=["t_quantized1", "t_quantized2"])
    fig4.write_image(newpath + "\Vtq1_tq2.png")

    fig5 = optuna.visualization.plot_contour(study, params=["V_max1", "V_max2"])
    fig5.write_image(newpath + "\Vmax1_Vmax2.png")

    fig6 = optuna.visualization.plot_contour(study, params=["Bx1", "Bx2"])
    fig6.write_image(newpath + "\Bx2_Bx1.png")

    fig7 = optuna.visualization.plot_contour(study, params=["V_max1", "Bx1"])
    fig7.write_image(newpath + "\Vmax1_Bx1.png")

    fig8 = optuna.visualization.plot_contour(study, params=["V_max2", "Bx2"])
    fig8.write_image(newpath + "\Vmax2_Bx2.png")

    fig9 = optuna.visualization.plot_contour(study, params=["forward_ratio1", "forward_ratio2"])
    fig9.write_image(newpath + "\Bforw1_forw2.png")

    # Optional: Copy the current file to the data folder
    with open(__file__, "r") as src:
        name = file_path + "\\code_copy.py"
        with open(name, "w") as tgt:
            tgt.write(src.read())

if __name__ == "__main__":
   main()