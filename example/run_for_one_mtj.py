"""
This module gets the switching probability for each data point of a whole dataset
and is reused in the experiment using several MTJs as a reservoir.
"""
import os
import numpy as np
import magneticreservoirs as mr
import magneticreservoirs.helper_functions as hf

def runner_for_one_neuron(file_path, X_train, X_test, psw_train, psw_test,
                          n_features, neuron_number, B_x, t_quantized, V_max, forward_ratio,
                          low_to_high, filter_type):

    ######## Experiment parameters ########
    reference_resistance = 100 # reference resistance for the PXI
    resistance_measurement_technique =  'lock_in_fast' #'kreuken_accurate', 'kreuken_fast', 'lock_in_accurate', 'lock_in_fast'
    delimitation_resistance = 2000 # clear limit between P and AP state of the MTJ in the resistance measurement technique you choose
    low_to_high = low_to_high
    
    ######## Reset pulse parameters ########
    reset_V_amp = -1.0 # in V
    reset_tp = 100 # in nanoseconds
    reset_ratio = 1.

    ######## Forward pulse parameters ########
    V_max = V_max # in V
    pulse_duration = t_quantized # in nanoseconds
    forward_ratio = forward_ratio  # ratio of the forward pulse amplitude to the reset pulse amplitude
    N_attempts = 20  # decide the number of switching attempts to get the switching probability

    ######## Apply magnetic field ########
    B_x = B_x # in A
    magnet = mr.Danfysik7000()
    magnet.rampToCurrent(B_x)

    ######## Instantiate the experiment model ########
    controller = mr.psw_control(low_to_high, resistance_measurement_technique)

    ######## Instantiate the new folder ########
    current_neuron_number = neuron_number
    newpath = file_path + "\\neuron{}".format({current_neuron_number})
    if not os.path.exists(newpath):
        os.makedirs(newpath)

    X_train, X_test = hf.add_filter(newpath, X_train, X_test, filter_type=filter_type)

    ######## Run the experiment : get all the training data first then all the testing data ########
    counter = 0 # counter for refreshing the awg
    for i in range(len(psw_train)):
        max_value = np.max(X_train[i])
        for j in range(n_features):
            file_number = "train" + str(i) + "_feature" + str(j)
            pulse = X_train[i][j]/max_value  # Normalize pulse to max value of 1
            pulse = pulse.tolist()
            psw_train[i][j] = controller.switching_probability(signal_array=pulse, 
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
            if counter % (10*n_features) == 0:  # Refresh AWG every 10 samples
                awg = mr.AWG()
                awg.close()
    for i in range(len(psw_test)):
        max_value = np.max(X_test[i])
        for j in range(n_features):
            file_number = "test" + str(i) + "_feature" + str(j)
            pulse = X_test[i][j]/max_value  # Normalize pulse to max value of 1
            pulse = pulse.tolist()
            psw_test[i][j] = controller.switching_probability(signal_array=pulse, 
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
            if counter % (10*n_features) == 0:  # Refresh AWG every 10 samples 
                awg = mr.AWG()
                awg.close()
        
    ######## Close all connections ########
    mr.cleanup_setup()   
    
    # Optional: Copy the current file to the data folder
    with open(__file__, "r") as src:
        name = file_path + "\\main_code_copy.py"
        with open(name, "w") as tgt:
            tgt.write(src.read())

    return psw_train, psw_test
