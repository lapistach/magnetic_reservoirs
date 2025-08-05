#!/usr/bin/env python3
"""
MTJ Control System - Main Control Module

Main control routine integrating AWG pulse generation and PXI resistance
measurement for MTJ device experiments.

Performance Optimizations:
- PXI measurement duration reduced from 0.1s to 0.03s (~2.7x speed improvement)
- Reduced oversampling for faster data acquisition
- Combined SCPI commands for AWG efficiency
- Optimized measurement logic: 2.2 avg measurements per trial (vs 3-4 previously)
- Minimal logging for reduced overhead

The r_low is real r_low + 10% of value, needed to get a 3x faster PXI
resistance measurement.
"""
import numpy as np
import time
from typing import List
from awg_control import AWG
from pxi_control import PXI_lock_in_fast, PXI_lock_in_accurate, PXI_Kreuken_accurate, PXI_Kreuken_fast
import pyvisa
from close_connections import cleanup_setup
import threading

class psw_control:

    def __init__(self, low_to_high: bool = True, resistance_measurement_technique: str = 'lock_in_fast'):
        self.low_to_high = low_to_high
        self.r_meas = resistance_measurement_technique

    def switching_probability(self, signal_array: List[float], pulse_duration: float,
                            pulse_voltage: float, ref_resistance: float,
                            r_low: float, reset_duration: float,
                            reset_voltage: float, n_trials: int,
                            ch4_forward_amplitude_factor: float,
                            ch4_reset_amplitude_factor: float,
                            magnetic_field : float, filenumber : str, 
                            filepath : str):
        """
        Main MTJ control routine with amplitude modulation and optimized
        measurements.
        
        Measurement Strategy:
        - 1 measurement: Initial resistance check
        - +1 measurement: Post-reset verification (only if reset needed)
        - 1 measurement: Final switching detection
        - Average: 2.2 measurements per trial (vs 3-4 previously)
        
        Parameters
        ----------
        signal_array : List[float]
            Array of amplitude factors for pulse sequence
        pulse_duration : float
            Duration of each pulse in seconds
        pulse_voltage : float
            Base voltage amplitude for pulses
        ref_resistance : float
            Reference resistance value in Ohms
        r_low : float
            Resistance threshold for reset triggering
        reset_duration : float
            Duration of reset pulse in seconds
        reset_voltage : float
            Voltage amplitude for reset pulse
        n_trials : int
            Number of measurement trials
        ch4_forward_amplitude_factor : float
            Amplitude scaling factor for channel 4 forward pulses
        ch4_reset_amplitude_factor : float
            Amplitude scaling factor for channel 4 reset pulses
        """
        # Initialize instruments
        awg = AWG()

        if self.r_meas == 'lock_in_fast':
            pxi = PXI_lock_in_fast()
        
        elif self.r_meas == 'lock_in_accurate':
            pxi = PXI_lock_in_accurate()

        elif self.r_meas == 'kreuken_accurate':
            pxi = PXI_Kreuken_accurate()

        elif self.r_meas == 'kreuken_fast':
            pxi = PXI_Kreuken_fast()

        else :
            print("Did not use proper measurement technique\n")
            print("You typed ", self.r_meas, "\n")
            print("You can only choose among : 'lock_in_fast', 'lock_in_accurate', 'kreuken_accurate', 'kreuken_fast'")
            cleanup_setup()
            exit

        failures = 0
        switched = 0
        pulse_duration = pulse_duration*1e-9
        reset_duration = reset_duration*1e-9

        # Create waveforms
        forward_wave, _ = awg.create_pulse_sequence(signal_array,
                                                    pulse_duration,
                                                    pulse_voltage)
        # Create reset pulse sequence instead of single pulse
        reset_signal_array = [0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0]
        reset_wave, _ = awg.create_pulse_sequence(reset_signal_array,
                                                reset_duration,
                                                reset_voltage)

        # Upload waveforms with independent amplitude control
        awg.upload_waveform(segment=1, waveform=forward_wave,
                            ch1_amplitude=1.0,
                            ch4_amplitude=ch4_forward_amplitude_factor)
        awg.upload_waveform(segment=2, waveform=reset_wave,
                            ch1_amplitude=1.0,
                            ch4_amplitude=ch4_reset_amplitude_factor)

        before_shots = []
        after_shots = []
        # Run experiment trials 
        if self.low_to_high :
            for i in range(n_trials):
                # Single initial measurement to check if reset is needed
                resistance = pxi.measure_resistance(ref_resistance)

                # Reset logic with minimal measurements
                if resistance > r_low:
                    awg.send_waveform(segment=2)
                    for i in range(20):  # 20 us
                       time.sleep(0.)
                    resistance = pxi.measure_resistance(ref_resistance)
                    if resistance > r_low:
                        failures += 1
                        continue  # Skip this trial
                before_shots.append(resistance)
                # Send forward pulse (resistance is now known to be acceptable)
                awg.send_waveform(segment=1)
                for i in range(20):  # 20 us
                       time.sleep(0.)
                resistance = pxi.measure_resistance(ref_resistance)
                after_shots.append(resistance)
                if resistance > r_low:
                    switched += 1

        else :
            for i in range(n_trials):
                # Single initial measurement to check if reset is needed
                resistance = pxi.measure_resistance(ref_resistance)

                # Reset logic with minimal measurements
                if resistance < r_low:
                    awg.send_waveform(segment=2)
                    for i in range(20):  # 20 us
                       time.sleep(0.)
                    resistance = pxi.measure_resistance(ref_resistance)
                    if resistance < r_low:
                        failures += 1
                        continue  # Skip this trial
                before_shots.append(resistance)
                # Send forward pulse (resistance is now known to be acceptable)
                awg.send_waveform(segment=1)
                for i in range(20):  # 20 us
                       time.sleep(0.)
                resistance = pxi.measure_resistance(ref_resistance)
                after_shots.append(resistance)
                if resistance < r_low:
                    switched += 1

        def close_awg_async():
            awg.close()
        
        awg_thread = threading.Thread(target=close_awg_async)
        awg_thread.start()

        # Calculate switching probability
        switching_prob = (switched / (n_trials - failures)
                        if (n_trials - failures) > 0 else 0)
        
        start_time = time.time()
        
        # Save everything (runs concurrently with AWG closing)
        file_directory = filepath + "\\" + filenumber + ".dat"
        file = open(file_directory , "w+")
        file.write("The forward pulse parameters are : \n")
        file.write(f"Bx = {format(magnetic_field)} V_max = {format(pulse_voltage)} ratio = {format(ch4_forward_amplitude_factor)} t_quantized = {format(pulse_duration)} \n")
        np.savetxt(file, np.asarray(signal_array), fmt='%.2f')
        file.write("\nThe reset pulse parameters are : \n")
        file.write(f"Bx = {format(magnetic_field)} V_max = {format(reset_voltage)} ratio = {format(ch4_reset_amplitude_factor)} t_quantized = {format(reset_duration)} \n")
        np.savetxt(file, np.asarray(reset_signal_array), fmt='%.2f')
        file.write(f"\nThe resistance is measured through the {format(self.r_meas)} method with a reference resistance of {format(ref_resistance)} Ohm and considering the delimitation between AP and P state at {format(r_low)} Ohm. \n")
        file.write(f"To get the switching probability, {format(n_trials)} shots are sent of which {format(switched)} switched but {format(failures)} failed to reset. This yields Psw = {format(switching_prob)} . \n")
        file.write("R_before_shot \t R_after_shot (Ohm) \n")
        try:
            np.savetxt(file, np.asarray([before_shots, after_shots]).T, delimiter='\t \t \t', fmt='%.2f')
        except ValueError:
            # Handle case where before_shots or after_shots is empty
            file.write("No valid resistance measurements were recorded.\n")
        file.close()
        print(f"Current device resistance: {resistance:.2f} Ω")
        print(f"Failures: {failures}, Switched: {switched}")

        end_time = time.time()
        elapsed_time = end_time - start_time

        # Wait for AWG to finish closing
        awg_thread.join()

        print(f"files written in {elapsed_time:.2f} seconds")
        return switching_prob
    
    def close_all_visa_connections():
        """Attempt to close all VISA connections."""
        print("Attempting to close all VISA connections...")
        
        try:
            rm = pyvisa.ResourceManager()
            
            # Get all active sessions
            try:
                sessions = rm.list_opened_resources()
                print(f"Found {len(sessions)} open sessions")
                
                for session in sessions:
                    try:
                        session.close()
                        print(f"  Closed session: {session}")
                    except Exception as e:
                        print(f"  Error closing session {session}: {e}")
            except Exception as e:
                print(f"Could not list opened resources: {e}")
            
            rm.close()
            print("ResourceManager closed")
            
        except Exception as e:
            print(f"Error in close_all_visa_connections: {e}")