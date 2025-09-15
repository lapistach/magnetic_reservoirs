"""
This module runs the resistance comparator from the package with a taylored pulse that can switch the MTJ.
This allows for the user to evaluate the speed and precision of each available resistance measurement in P and AP state.
No need to have a precise resistance if the AP ad P state are clearly distinguishable.

This uses the PXI Measurement Precision Comparison from the package. 
It compares the precision (standard deviation) and speed of multiple resistance
measurements using 4 different PXI settings.

Usage:
- Takes 5 resistance measurements before pulse (all settings)
- Sends one pulse sequence to the MTJ device  
- Takes 5 resistance measurements after pulse (all settings)
- Compares standard deviation and speed between settings

"""
from magneticreservoirs import PXIPrecisionComparator

def main():
    # Create comparison instance and run test
    comparison = PXIPrecisionComparator()
    comparison.precision_comparison_test(
        signal_array=[0, 1, 0],
        pulse_duration=100e-9,
        pulse_voltage=1.0,
        magnetic_field=0.,
        ref_resistance=100,
        ch4_amplitude_factor=0.5,
        num_measurements=5)
                
if __name__ == '__main__':
    main()