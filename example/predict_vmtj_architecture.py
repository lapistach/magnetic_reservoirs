"""
This module runs the experiment detailed in the report, using the magneticreservoirs package, for prediction of data with a single MTJ.
"""
import os
import numpy as np
import datasets as ds
import magneticreservoirs as mr
import magneticreservoirs.helper_functions as hf
from magneticreservoirs.output_layer import prediction_linear_regression 

def main(dir_path):

    ######### Create a dataset ########
    training_datapoints = 100
    testing_datapoints = 1 # MH: if you put more then 1 then it will automatically inductively add the last predicted value at the end of the new training set.
    datapoints = hf.stream_0011(testing_datapoints + training_datapoints)  

    ######### Experiment architecture ########
    mem_array = np.array([[1., 0.], [0., 1.]])  # Memory coefficients. The two entries are two parameters A and B. 
    memstack = mr.memory_array(memory_coefficients=mem_array)  # Instantiate the memory array
    psw = memstack.psw_init()
    psw_list_train = [psw]  # Initialize the list of training switching probabilities
    pseudo_inverse = prediction_linear_regression(n_features=len(psw))  # Instantiate the prediction model

    ######## Experiment parameters ########
    reference_resistance = 100 # reference resistance for the PXI
    resistance_measurement_technique =  'lock_in_fast' #'kreuken_accurate', 'kreuken_fast', 'lock_in_accurate', 'lock_in_fast'
    delimitation_resistance = 0 # clear limit between P and AP state of the MTJ in the resistance measurement technique you choose
    low_to_high = False

    ######## Reset pulse parameters ########
    reset_V_amp = 1.0 # in V
    reset_tp = 100 # in nanoseconds
    reset_ratio = 0.

    ######## Forward pulse parameters ########
    V_max = -1.0 # in V
    pulse_duration = 100.0 # in nanoseconds
    forward_ratio = 0.  # ratio of the forward pulse amplitude to the reset pulse amplitude
    N_attempts = 15  # decide the number of switching attempts to get the switching probability

    ######## Apply magnetic field ########
    B_x = 1.5 # in A
    magnet = mr.Danfysik7000()
    magnet.rampToCurrent(B_x)

    ######## Instantiate the experiment model ########
    controller = mr.psw_control(low_to_high, resistance_measurement_technique)

    ######## Run the experiment : get all the training data first then all the testing data ########
    newpath = dir_path + "\\experiment_data"
    if not os.path.exists(newpath):
        os.makedirs(newpath)

    counter = 0 # counter for refreshing the awg
    for i in range(training_datapoints-1):
        file_number = "train" + str(i)
        pulse_before = np.array([0, datapoints[i], 0])
        pulse_in_architecture = memstack.pulses(pulse_before, psw)
        for j in range(len(pulse_in_architecture)):
            psw[j].tolist()  
            psw[j] = controller.switching_probability(signal_array=pulse_in_architecture[j], 
                                                            pulse_duration=pulse_duration,
                                                            pulse_voltage=V_max, 
                                                            ref_resistance=reference_resistance,
                                                            r_low=delimitation_resistance, 
                                                            reset_duration=reset_tp,
                                                            reset_voltage=reset_V_amp, 
                                                            n_trials=N_attempts,
                                                            ch4_forward_amplitude_factor=forward_ratio,
                                                            ch4_reset_amplitude_factor=reset_ratio,
                                                            magnetic_field=B_x, 
                                                            filenumber = file_number, 
                                                            filepath = newpath
                                                            )
            counter += 1
            if counter > 100:
                counter = 0
                awg = mr.AWG()
                awg.close()
        psw_list_train.append(psw)
    pseudo_inverse.train(datapoints[:training_datapoints], psw_list_train)  # Train the linear regression model
    pseudo_inverse_coefficients, pseudo_inverse_intercept = pseudo_inverse.matrix()  # Get the coefficients and intercept of the linear regression model

    psw_list_test = []  # Initialize the list of testing switching probabilities
    pred = datapoints[training_datapoints - 1 ] # initialize the prediction with the last training datapoint
    predicted_datapoints = []
    for i in range(testing_datapoints):
        file_number = "predict" + str(i)
        pulse_before = np.array([0, pred, 0])
        pulse_in_architecture = memstack.pulses(pulse_before, psw)
        for j in range(len(pulse_in_architecture)):
            psw[j].tolist()  
            psw[j] = controller.switching_probability(signal_array=pulse_in_architecture[j], 
                                                            pulse_duration=pulse_duration,
                                                            pulse_voltage=V_max, 
                                                            ref_resistance=reference_resistance,
                                                            r_low=delimitation_resistance, 
                                                            reset_duration=reset_tp,
                                                            reset_voltage=reset_V_amp, 
                                                            n_trials=N_attempts,
                                                            ch4_forward_amplitude_factor=forward_ratio,
                                                            ch4_reset_amplitude_factor=reset_ratio,
                                                            magnetic_field=B_x, 
                                                            filenumber = file_number, 
                                                            filepath = newpath
                                                            )
            counter += 1
            if counter > 100:
                counter = 0
                awg = mr.AWG()
                awg.close()
        psw_list_test.append(psw)
        pred = pseudo_inverse.predict([psw])  # Predict the next data point
        predicted_datapoints.append(pred)
    awg = mr.AWG()
    awg.close()
    # mr.cleanup_setup()

    mse, mae, rmse, mape = hf.prediction_scores(datapoints[training_datapoints:], predicted_datapoints)  # Get the prediction scores

    output_layer_path = dir_path + "\\output_layer"
    if not os.path.exists(output_layer_path):
        os.makedirs(output_layer_path)

    newpath_wr = output_layer_path + "\\with_reservoir"
    if not os.path.exists(newpath_wr):
        os.makedirs(newpath_wr)

    file_directory = newpath_wr + "\\results_highlights.dat"
    file = open(file_directory , "w+")
    file.write("Number of training samples: " + str(training_datapoints) + "\n")
    file.write("Number of predicted samples: " + str(testing_datapoints) + "\n")
    file.write("Architecture memories: \n")
    np.savetxt(file, mem_array, fmt='%.2f')
    file.write("Datapoints: \n")
    np.savetxt(file, datapoints, fmt='%.2f')
    file.write("Predicted datapoints: \n")
    np.savetxt(file, predicted_datapoints, fmt='%.2f')
    file.write("Training switching probabilities: \n")
    for i in range(len(psw_list_train)):
        file.write("Switching probabilities for training sample " + str(datapoints[i]) + ": \n")
        np.savetxt(file, psw_list_train[i], fmt='%.2f')
    for i in range(len(psw_list_test)):
        file.write("Switching probabilities for testing sample " + str(datapoints[training_datapoints + i]) + ": \n")
        np.savetxt(file, psw_list_test[i], fmt='%.2f')
    file.write("Mean Squared Error: " + str(mse) + "\n")
    file.write("Mean Absolute Error: " + str(mae) + "\n")
    file.write("Root Mean Squared Error: " + str(rmse) + "\n")
    file.write("Mean Absolute Percentage Error: " + str(mape) + "\n")
    file.write("Linear Regression Coefficients: \n")
    np.savetxt(file, pseudo_inverse_coefficients, fmt='%.2f')
    file.write("Linear Regression Intercept: \n")
    np.savetxt(file, pseudo_inverse_intercept.reshape(1, -1), fmt='%.2f')
    file.close()

    ######## No reservoir comparison ########
    file_no_reservoir = output_layer_path + "\\no_reservoir"
    if not os.path.exists(file_no_reservoir):
        os.makedirs(file_no_reservoir)
    pseudo_inverse = prediction_linear_regression(n_features=1)  # Instantiate the prediction model
    training_list = [0] + datapoints[:training_datapoints-1] 
    pseudo_inverse.train(datapoints[:training_datapoints], training_list)  # Train the linear regression model
    pseudo_inverse_coefficients, pseudo_inverse_intercept = pseudo_inverse.matrix()  # Get the coefficients and intercept of the linear regression model
    predicted_datapoints_no_reservoir = []
    for i in range(testing_datapoints):
        pred = pseudo_inverse.predict(datapoints[training_datapoints - 1 + i])
        predicted_datapoints_no_reservoir.append(pred)
    mse_no_reservoir, mae_no_reservoir, rmse_no_reservoir, mape_no_reservoir = hf.prediction_scores(datapoints[training_datapoints:], 
                                                                                                    predicted_datapoints_no_reservoir) 
    file_directory = file_no_reservoir + "\\results_highlights.dat"
    file = open(file_directory , "w+")
    file.write("Number of training samples: " + str(training_datapoints) + "\n")
    file.write("Number of predicted samples: " + str(testing_datapoints) + "\n")
    file.write("Datapoints: \n")
    np.savetxt(file, datapoints, fmt='%.2f')
    file.write("Predicted datapoints: \n")
    np.savetxt(file, predicted_datapoints_no_reservoir, fmt='%.2f')
    file.write("Mean Squared Error: " + str(mse_no_reservoir) + "\n")
    file.write("Mean Absolute Error: " + str(mae_no_reservoir) + "\n")
    file.write("Root Mean Squared Error: " + str(rmse_no_reservoir) + "\n")
    file.write("Mean Absolute Percentage Error: " + str(mape_no_reservoir) + "\n")
    file.write("Linear Regression Coefficients: \n")
    np.savetxt(file, pseudo_inverse_coefficients, fmt='%.2f')
    file.write("Linear Regression Intercept: \n")
    np.savetxt(file, pseudo_inverse_intercept.reshape(1, -1), fmt='%.2f')
    file.close()

    # Optional: Copy the current file to the data folder
    with open(__file__, "r") as src:
        name = dir_path + "\\code_copy.py"
        with open(name, "w") as tgt:
            tgt.write(src.read())

if __name__ == "__main__":
    dir_path = r"path/to/results/dir" # you must create the folders before writing the path
    main(dir_path)
