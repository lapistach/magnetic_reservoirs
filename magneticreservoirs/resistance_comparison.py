#!/usr/bin/env python3
"""
PXI Measurement Precision Comparison Test

This script compares the precision (standard deviation) and speed of multiple resistance
measurements using 4 different PXI settings.

Usage:
- Takes 5 resistance measurements before pulse (all settings)
- Sends one pulse sequence to the MTJ device  
- Takes 5 resistance measurements after pulse (all settings)
- Compares standard deviation and speed between settings
"""
 
import time
import statistics
import numpy as np
from typing import List, Tuple
from awg_control import AWG
from pxi_control import PXI_lock_in_fast, PXI_lock_in_accurate, PXI_Kreuken_accurate, PXI_Kreuken_fast
from magnet_control import Danfysik7000
import close_connections

class PXIPrecisionComparator:
    """Class to compare precision and speed of 4 different PXI settings for resistance measurements."""
    
    def __init__(self):
        self.awg = AWG()
        self.magnet = Danfysik7000()
        

    def measure_resistance_multiple(self, method: str = "lock_in_fast", ref_resistance: float = 100,
                                   num_measurements: int = 5) -> Tuple[List[float], float, float, float]:
        """
        Take multiple resistance measurements and calculate statistics.

        Parameters
        ----------
        ref_resistance : float
            Reference resistance value
        num_measurements : int
            Number of measurements to take
        use_former : bool
            If True, use former high-precision method; 
            if False, use current fast method

        Returns
        -------
        Tuple[List[float], float, float]
            measurements, mean, standard_deviation
        """

        if method == 'lock_in_fast':
            pxi = PXI_lock_in_fast()
        elif method == 'lock_in_accurate':
            pxi = PXI_lock_in_accurate()
        elif method == 'kreuken_accurate':
            pxi = PXI_Kreuken_accurate()
        elif method == 'kreuken_fast':
            pxi = PXI_Kreuken_fast()        

        measurements = []
        start_time = time.time()
        for _ in range(num_measurements):
            resistance = pxi.measure_resistance(ref_resistance)
            measurements.append(resistance)
        end_time = time.time()

        # Calculate elapsed time
        elapsed_time = end_time - start_time

        # Filter out infinite values for statistics
        valid_measurements = [m for m in measurements
                             if not (np.isinf(m) or np.isnan(m))]

        if len(valid_measurements) == 0:
            return measurements, float('inf'), float('inf')
        elif len(valid_measurements) == 1:
            return measurements, valid_measurements[0], 0.0, elapsed_time
        else:
            mean_resistance = statistics.mean(valid_measurements)
            std_deviation = statistics.stdev(valid_measurements)
            return measurements, mean_resistance, std_deviation, elapsed_time

    def precision_comparison_test(self,
                                 signal_array: List[float],
                                 pulse_duration: float,
                                 pulse_voltage: float,
                                 magnetic_field: float = 1.0,
                                 ref_resistance: float = 100,
                                 ch4_amplitude_factor: float = 0.5,
                                 num_measurements: int = 5) -> dict:
        """
        Compare precision and speed of all 4 settings.

        Returns
        -------
        dict
            Precision comparison results with standard deviations and timings.
        """
        try:
            # Set magnetic field
            self.magnet.rampToCurrent(magnetic_field)
            self.magnetPS.set_current(magnetic_field)
            # Create and upload waveform
            waveform, _ = self.awg.create_pulse_sequence(signal_array,
                                                        pulse_duration,
                                                        pulse_voltage)
            self.awg.upload_waveform(segment=1, waveform=waveform,
                                   ch1_amplitude=1.0,
                                   ch4_amplitude=ch4_amplitude_factor)


            kreuken_accurate_measurements, kreuken_accurate_mean, kreuken_accurate_std, kreuken_accurate_time = \
                self.measure_resistance_multiple('kreuken_accurate', ref_resistance,
                                               num_measurements)
            print(f"Kreuken accurate settings - Before pulse:")
            print(f"  Measurements: "
                  f"{[f'{r:.1f}' for r in kreuken_accurate_measurements]} Ohm")
            print(f"  Mean: {kreuken_accurate_mean:.1f} Ohm")
            print(f"  Std Dev: {kreuken_accurate_std:.2f} Ohm")
            print(f"  Total time: {kreuken_accurate_time:.3f} s")

            kreuken_fast_measurements, kreuken_fast_mean, kreuken_fast_std, kreuken_fast_time = \
                self.measure_resistance_multiple('kreuken_fast', ref_resistance,
                                               num_measurements)
            print(f"Kreuken fast settings - Before pulse:")
            print(f"  Measurements: "
                  f"{[f'{r:.1f}' for r in kreuken_fast_measurements]} Ohm")
            print(f"  Mean: {kreuken_fast_mean:.1f} Ohm")
            print(f"  Std Dev: {kreuken_fast_std:.2f} Ohm")
            print(f"  Total time: {kreuken_fast_time:.3f} s")

            lock_in_accurate_measurements, lock_in_accurate_mean, lock_in_accurate_std, lock_in_accurate_time = \
                self.measure_resistance_multiple('lock_in_accurate', ref_resistance,
                                               num_measurements)
            print(f"Lock-in accurate settings - Before pulse:")
            print(f"  Measurements: "
                  f"{[f'{r:.1f}' for r in lock_in_accurate_measurements]} Ohm") 
            print(f"  Mean: {lock_in_accurate_mean:.1f} Ohm")
            print(f"  Std Dev: {lock_in_accurate_std:.2f} Ohm")
            print(f"  Total time: {lock_in_accurate_time:.3f} s")

            lock_in_fast_measurements, lock_in_fast_mean, lock_in_fast_std, lock_in_fast_time = \
                self.measure_resistance_multiple('lock_in_fast', ref_resistance,
                                               num_measurements)
            print(f"Lock-in fast settings - Before pulse:")
            print(f"  Measurements: "
                  f"{[f'{r:.1f}' for r in lock_in_fast_measurements]} Ohm") 
            print(f"  Mean: {lock_in_fast_mean:.1f} Ohm")
            print(f"  Std Dev: {lock_in_fast_std:.2f} Ohm")
            print(f"  Total time: {lock_in_fast_time:.3f} s")

            print("Sending pulse sequence...")
            self.awg.send_waveform(segment=1)
            time.sleep(3)

            kreuken_accurate_measurements, kreuken_accurate_mean, kreuken_accurate_std, kreuken_accurate_time = \
                self.measure_resistance_multiple('kreuken_accurate', ref_resistance,
                                               num_measurements)
            print(f"Kreuken accurate settings - After pulse:")
            print(f"  Measurements: "
                  f"{[f'{r:.1f}' for r in kreuken_accurate_measurements]} Ohm")
            print(f"  Mean: {kreuken_accurate_mean:.1f} Ohm")
            print(f"  Std Dev: {kreuken_accurate_std:.2f} Ohm")
            print(f"  Total time: {kreuken_accurate_time:.3f} s")

            
            kreuken_fast_measurements, kreuken_fast_mean, kreuken_fast_std, kreuken_fast_time = \
                self.measure_resistance_multiple('kreuken_fast', ref_resistance,
                                               num_measurements)
            print(f"Kreuken fast settings - After pulse:")
            print(f"  Measurements: "
                  f"{[f'{r:.1f}' for r in kreuken_fast_measurements]} Ohm")
            print(f"  Mean: {kreuken_fast_mean:.1f} Ohm")
            print(f"  Std Dev: {kreuken_fast_std:.2f} Ohm")
            print(f"  Total time: {kreuken_fast_time:.3f} s")

            lock_in_accurate_measurements, lock_in_accurate_mean, lock_in_accurate_std, lock_in_accurate_time = \
                self.measure_resistance_multiple('lock_in_accurate', ref_resistance,
                                               num_measurements)
            print(f"Lock-in accurate settings - After pulse:")
            print(f"  Measurements: "
                  f"{[f'{r:.1f}' for r in lock_in_accurate_measurements]} Ohm") 
            print(f"  Mean: {lock_in_accurate_mean:.1f} Ohm")
            print(f"  Std Dev: {lock_in_accurate_std:.2f} Ohm")
            print(f"  Total time: {lock_in_accurate_time:.3f} s")

            lock_in_fast_measurements, lock_in_fast_mean, lock_in_fast_std, lock_in_fast_time = \
                self.measure_resistance_multiple('lock_in_fast', ref_resistance,
                                               num_measurements)
            print(f"Lock-in fast settings - After pulse:")
            print(f"  Measurements: "
                  f"{[f'{r:.1f}' for r in lock_in_fast_measurements]} Ohm") 
            print(f"  Mean: {lock_in_fast_mean:.1f} Ohm")
            print(f"  Std Dev: {lock_in_fast_std:.2f} Ohm")
            print(f"  Total time: {lock_in_fast_time:.3f} s")

        finally:
            close_connections.cleanup_setup()


