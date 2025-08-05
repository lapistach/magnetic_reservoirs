import time
import numpy as np
import nidaqmx
from nidaqmx.constants import AcquisitionType, TerminalConfiguration
import os
import numpy as np
import pyvisa

def super_fast(t_quantized, V_max, forward_ratio, X_train, n_features):
    #################### Initialize PXI ####################
    ai_channel= "4462/ai1"
    ao_channel= "4461/ao0"
    R_ref = 100
    frequency_test = 10.0  
    clock_time = 10240  
    meas_amp = 0.02
    outString = ao_channel
    inString1 = ai_channel
    inString2 = "4462/ai2"
    samples_per_cycle = int(clock_time / frequency_test) 
    total_samples = samples_per_cycle // 4  
    actual_time = total_samples / clock_time 
    t = np.linspace(0, actual_time, total_samples, endpoint=False)
    signal = meas_amp * np.sin(2 * np.pi * frequency_test * t)
    cos_ref = np.cos(2 * np.pi * frequency_test * t)
    sin_ref = np.sin(2 * np.pi * frequency_test * t)

    task = nidaqmx.Task()
    task.ao_channels.add_ao_voltage_chan(outString)
    task.timing.cfg_samp_clk_timing(
        clock_time, sample_mode=AcquisitionType.FINITE,
        samps_per_chan=total_samples)

    task2 = nidaqmx.Task()
    task2.ai_channels.add_ai_voltage_chan(
        inString1, terminal_config=TerminalConfiguration.BAL_DIFF)
    task2.ai_channels.add_ai_voltage_chan(
        inString2, terminal_config=TerminalConfiguration.BAL_DIFF)
    task2.timing.cfg_samp_clk_timing(
        clock_time, sample_mode=AcquisitionType.FINITE,
        samps_per_chan=total_samples)

    ############################## Initialize AWG ##############################

    address = "TCPIP0::localhost::inst0::INSTR"
    sample_rate = 64e9
    timeout = 60000
    rm = pyvisa.ResourceManager()
    awg = rm.open_resource(address)
    awg.timeout = timeout
    sample_rate = sample_rate
    awg.write("*RST")
    awg.write(f":SOUR:FREQ:RAST {int(sample_rate)}")
    awg.write(":INST:DACM DUAL")
    awg.write(":INST:MEM:EXT:RDIV DIV2")
    for ch in [1, 4]:
        awg.write(f":TRAC{ch}:MMOD EXT")
        awg.write(f":VOLT{ch} 1")
        awg.write(f":OUTP{ch} ON")
        awg.write(f":OUTP{ch}:COUP DC")
        awg.write(f":OUTP{ch}:POL NORM") 
        awg.write(f":INIT{ch}:CONT OFF")


    ############################## Global initialization ##############################
    low_to_high = True  
    pulse_duration = t_quantized*1e-9
    pulse_voltage = V_max
    r_low = 2000
    reset_duration = 100e-9
    reset_voltage = - 1.0
    n_trials = 15
    ch4_forward_amplitude_factor = forward_ratio
    ch4_reset_amplitude_factor = 1.0
    sw_prob_list = []

    for k in range(len(X_train)):
            max_value = np.max(X_train[k])
            for j in range(n_features):
                pulse = X_train[k][j]/max_value  # Normalize pulse to max value of 1
                signal_array = pulse.tolist()
                samples_per_pulse = max(2, int(pulse_duration * sample_rate))
                sequence_samples = samples_per_pulse * len(signal_array)
                padding_samples = max(int(sequence_samples * 0.2), 2)
                total_samples = sequence_samples + 2 * padding_samples
                waveform = np.zeros(total_samples)
                for i, amplitude_factor in enumerate(signal_array):
                    start_sample = padding_samples + i * samples_per_pulse
                    end_sample = padding_samples + (i + 1) * samples_per_pulse
                    voltage = amplitude_factor * pulse_voltage
                    waveform[start_sample:end_sample] = voltage

                ch1_amplitude=1.0
                ch4_amplitude=ch4_forward_amplitude_factor
                temp_dir = r"C:\Users\Sputnik2.0\Documents\Temp Data"
                os.makedirs(temp_dir, exist_ok=True)

                gran = 1280
                waveform_len = len(waveform)

                if waveform_len < gran:
                    padded_waveform = np.pad(waveform, (0, gran - waveform_len))
                else:
                    if waveform_len % gran != 0:
                        need_to_add = gran - (waveform_len % gran)
                        padded_waveform = np.pad(waveform, (0, need_to_add))
                    else:
                        padded_waveform = waveform

                for ch, amplitude in [(1, ch1_amplitude), (4, ch4_amplitude)]:
                    filename = f"{temp_dir}\\pulse_segment{1}_CH{ch}.csv"
                    
                    with open(filename, 'w') as f:
                        f.write("SampleRate = 64000000000\n")
                        f.write("SetConfig=true\n")
                        
                        for sample in padded_waveform:
                            scaled_sample = float(sample) * float(amplitude)
                            if abs(scaled_sample) < 1e-12:
                                scaled_sample = 0.0
                            f.write(f"{scaled_sample:.10f}\n")
                    
                    import_cmd = f":TRAC{ch}:IMP {1},'{filename}',CSV,IONLY,OFF,ALEN"
                    awg.write(import_cmd)
                    awg.write(f":TRAC{ch}:COUN 1")

                signal_array = [0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0]
                pulse_duration = reset_duration
                pulse_voltage = reset_voltage
                samples_per_pulse = max(2, int(pulse_duration * sample_rate))
                sequence_samples = samples_per_pulse * len(signal_array)
                padding_samples = max(int(sequence_samples * 0.2), 2)
                total_samples = sequence_samples + 2 * padding_samples
                waveform = np.zeros(total_samples)
                for i, amplitude_factor in enumerate(signal_array):
                    start_sample = padding_samples + i * samples_per_pulse
                    end_sample = padding_samples + (i + 1) * samples_per_pulse
                    voltage = amplitude_factor * pulse_voltage
                    waveform[start_sample:end_sample] = voltage

                ch1_amplitude=1.0
                ch4_amplitude=ch4_reset_amplitude_factor
                temp_dir = r"C:\Users\Sputnik2.0\Documents\Temp Data"
                os.makedirs(temp_dir, exist_ok=True)

                gran = 1280
                waveform_len = len(waveform)

                if waveform_len < gran:
                    padded_waveform = np.pad(waveform, (0, gran - waveform_len))
                else:
                    if waveform_len % gran != 0:
                        need_to_add = gran - (waveform_len % gran)
                        padded_waveform = np.pad(waveform, (0, need_to_add))
                    else:
                        padded_waveform = waveform

                for ch, amplitude in [(1, ch1_amplitude), (4, ch4_amplitude)]:
                    filename = f"{temp_dir}\\pulse_segment{2}_CH{ch}.csv"
                    
                    with open(filename, 'w') as f:
                        f.write("SampleRate = 64000000000\n")
                        f.write("SetConfig=true\n")
                        
                        for sample in padded_waveform:
                            scaled_sample = float(sample) * float(amplitude)
                            if abs(scaled_sample) < 1e-12:
                                scaled_sample = 0.0
                            f.write(f"{scaled_sample:.10f}\n")
                    
                    import_cmd = f":TRAC{ch}:IMP {2},'{filename}',CSV,IONLY,OFF,ALEN"
                    awg.write(import_cmd)
                    awg.write(f":TRAC{ch}:COUN 1")

                failures = 0
                switched = 0
                before_shots = []
                after_shots = []

                if low_to_high :
                    for i in range(n_trials):
                        task.write(signal, auto_start=True)
                        task2.start()
                        data2 = task2.read(number_of_samples_per_channel=total_samples)
                        task2.stop()
                        task.stop() 
                        data_array = np.array(data2)
                        channel1data = data_array[0][:total_samples]  # Direct slicing
                        channel2data = data_array[1][:total_samples]  # Direct slicing
                        ch1_normalized = channel1data * 1e-8 
                        x_component = 2 * np.mean(ch1_normalized * cos_ref)
                        y_component = 2 * np.mean(ch1_normalized * sin_ref)
                        amplitude = np.sqrt(x_component**2 + y_component**2)
                        if amplitude > 1e-12:  
                            v_meas = 2 * np.mean((channel1data / amplitude) * channel2data) * 1e-16
                            if v_meas != 0:  
                                rmtj = R_ref * amplitude / v_meas - R_ref  
                                resistance = float(rmtj)

                        if resistance > r_low:
                            awg.write(f":TRAC1:SEL {2};:TRAC4:SEL {2}")
                            awg.write(":INIT:IMM;:TRIG:BEG;:ABOR")

                            time.sleep(0.)

                            task.write(signal, auto_start=True)
                            task2.start()
                            data2 = task2.read(number_of_samples_per_channel=total_samples)
                            task2.stop()
                            task.stop() 
                            data_array = np.array(data2)
                            channel1data = data_array[0][:total_samples]  # Direct slicing
                            channel2data = data_array[1][:total_samples]  # Direct slicing
                            ch1_normalized = channel1data * 1e-8 
                            x_component = 2 * np.mean(ch1_normalized * cos_ref)
                            y_component = 2 * np.mean(ch1_normalized * sin_ref)
                            amplitude = np.sqrt(x_component**2 + y_component**2)
                            if amplitude > 1e-12:  
                                v_meas = 2 * np.mean((channel1data / amplitude) * channel2data) * 1e-16
                                if v_meas != 0:  
                                    rmtj = R_ref * amplitude / v_meas - R_ref  
                                    resistance = float(rmtj)

                            if resistance > r_low:
                                failures += 1
                                continue 
                        before_shots.append(resistance)

                        awg.write(f":TRAC1:SEL {1};:TRAC4:SEL {1}")
                        awg.write(":INIT:IMM;:TRIG:BEG;:ABOR")

                        time.sleep(0.)

                        task.write(signal, auto_start=True)
                        task2.start()
                        data2 = task2.read(number_of_samples_per_channel=total_samples)
                        task2.stop()
                        task.stop() 
                        data_array = np.array(data2)
                        channel1data = data_array[0][:total_samples]  # Direct slicing
                        channel2data = data_array[1][:total_samples]  # Direct slicing
                        ch1_normalized = channel1data * 1e-8 
                        x_component = 2 * np.mean(ch1_normalized * cos_ref)
                        y_component = 2 * np.mean(ch1_normalized * sin_ref)
                        amplitude = np.sqrt(x_component**2 + y_component**2)
                        if amplitude > 1e-12:  
                            v_meas = 2 * np.mean((channel1data / amplitude) * channel2data) * 1e-16
                            if v_meas != 0:  
                                rmtj = R_ref * amplitude / v_meas - R_ref  
                                resistance = float(rmtj)

                        after_shots.append(resistance)
                        if resistance > r_low:
                            switched += 1

                else :
                    for i in range(n_trials):
                        task.write(signal, auto_start=True)
                        task2.start()
                        data2 = task2.read(number_of_samples_per_channel=total_samples)
                        task2.stop()
                        task.stop() 
                        data_array = np.array(data2)
                        channel1data = data_array[0][:total_samples]  # Direct slicing
                        channel2data = data_array[1][:total_samples]  # Direct slicing
                        ch1_normalized = channel1data * 1e-8 
                        x_component = 2 * np.mean(ch1_normalized * cos_ref)
                        y_component = 2 * np.mean(ch1_normalized * sin_ref)
                        amplitude = np.sqrt(x_component**2 + y_component**2)
                        if amplitude > 1e-12:  
                            v_meas = 2 * np.mean((channel1data / amplitude) * channel2data) * 1e-16
                            if v_meas != 0:  
                                rmtj = R_ref * amplitude / v_meas - R_ref  
                                resistance = float(rmtj)

                        if resistance < r_low:
                            awg.write(f":TRAC1:SEL {2};:TRAC4:SEL {2}")
                            awg.write(":INIT:IMM;:TRIG:BEG;:ABOR")

                            time.sleep(0.)

                            task.write(signal, auto_start=True)
                            task2.start()
                            data2 = task2.read(number_of_samples_per_channel=total_samples)
                            task2.stop()
                            task.stop() 
                            data_array = np.array(data2)
                            channel1data = data_array[0][:total_samples]  # Direct slicing
                            channel2data = data_array[1][:total_samples]  # Direct slicing
                            ch1_normalized = channel1data * 1e-8 
                            x_component = 2 * np.mean(ch1_normalized * cos_ref)
                            y_component = 2 * np.mean(ch1_normalized * sin_ref)
                            amplitude = np.sqrt(x_component**2 + y_component**2)
                            if amplitude > 1e-12:  
                                v_meas = 2 * np.mean((channel1data / amplitude) * channel2data) * 1e-16
                                if v_meas != 0:  
                                    rmtj = R_ref * amplitude / v_meas - R_ref  
                                    resistance = float(rmtj)

                            if resistance < r_low:
                                failures += 1
                                continue  # Skip this trial
                        before_shots.append(resistance)



                        awg.write(f":TRAC1:SEL {1};:TRAC4:SEL {1}")
                        awg.write(":INIT:IMM;:TRIG:BEG;:ABOR")

                        time.sleep(0.)

                        task.write(signal, auto_start=True)
                        task2.start()
                        data2 = task2.read(number_of_samples_per_channel=total_samples)
                        task2.stop()
                        task.stop() 
                        data_array = np.array(data2)
                        channel1data = data_array[0][:total_samples]  # Direct slicing
                        channel2data = data_array[1][:total_samples]  # Direct slicing
                        ch1_normalized = channel1data * 1e-8 
                        x_component = 2 * np.mean(ch1_normalized * cos_ref)
                        y_component = 2 * np.mean(ch1_normalized * sin_ref)
                        amplitude = np.sqrt(x_component**2 + y_component**2)
                        if amplitude > 1e-12:  
                            v_meas = 2 * np.mean((channel1data / amplitude) * channel2data) * 1e-16
                            if v_meas != 0:  
                                rmtj = R_ref * amplitude / v_meas - R_ref  
                                resistance = float(rmtj)

                        after_shots.append(resistance)
                        if resistance < r_low:
                            switched += 1
                print("failures:", failures, "switched:", switched)
                switching_prob = (switched / (n_trials - failures)
                                if (n_trials - failures) > 0 else 0)
                sw_prob_list.append(switching_prob)

    ################################## Close all ###################################
    task.close()
    task2.close()
    awg.close()
    rm.close()

    return sw_prob_list



































































 
   
   