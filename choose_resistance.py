from resistance_comparison import PXIPrecisionComparator

def main():
    # Create comparison instance and run test
    comparison = PXIPrecisionComparator()
    comparison.precision_comparison_test(
        signal_array=[0, 1, 0],
        pulse_duration=100e-9,
        pulse_voltage=1.0,
        magnetic_field=0.,
        ref_resistance=100,
        ch4_amplitude_factor=0.0,
        num_measurements=5)
                
if __name__ == '__main__':
    main()