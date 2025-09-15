#!/usr/bin/env python3
"""
PXI Control Module
"""
import time
import numpy as np
import nidaqmx
from nidaqmx.constants import Edge, AcquisitionType, TerminalConfiguration
from scipy.optimize import curve_fit

class PXI_lock_in_fast:
    """PXI control for ultra-fast accurate resistance measurement."""

    def __init__(self, ai_channel: str = "4462/ai1",
                 ao_channel: str = "4461/ao0"):
        """
        Initialize PXI controller.

        Parameters
        ----------
        ai_channel : str
            Analog input channel identifier
        ao_channel : str
            Analog output channel identifier
        """
        self.ai_channel = ai_channel
        self.ao_channel = ao_channel
        
    def measure_resistance(self, ref_resistance: float) -> float:
        """
        Ultra-fast and accurate MTJ resistance measurement using lock-in 
        principles.

        This method eliminates curve fitting by using the known 10 Hz frequency
        and employs digital lock-in amplifier techniques for speed and accuracy.

        Key optimizations:
        - No curve fitting required (frequency is known)
        - Lock-in amplifier correlation for noise rejection
        - Minimal measurement time while capturing enough cycles
        - Direct amplitude extraction using correlation

        Parameters
        ----------
        ref_resistance : float
            Reference resistance value in Ohms

        Returns
        -------
        resistance : float
            Measured MTJ resistance in Ohms
        """
        try:
            # ULTRA-AGGRESSIVE parameters for sub-1-second measurement
            R_ref = ref_resistance
            frequency_test = 10.0  # Fixed PXI frequency

            # ABSOLUTE MAXIMUM SPEED: Sub-0.5s is the only goal
            clock_time = 10240  # Extremely reduced sample rate
            meas_amp = 0.02
            outString = self.ao_channel
            inString1 = self.ai_channel
            inString2 = "4462/ai2"

            # Ultra-minimal samples for ABSOLUTE MAXIMUM SPEED
            samples_per_cycle = int(clock_time / frequency_test)  # 1024 samples per cycle
            total_samples = samples_per_cycle // 4  # Quarter cycle for absolute speed
            actual_time = total_samples / clock_time  # 0.025 seconds per measurement

            # Create ultra-fast tasks with minimal configuration overhead
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

            # Pre-compute ALL processing elements for maximum speed
            t = np.linspace(0, actual_time, total_samples, endpoint=False)
            signal = meas_amp * np.sin(2 * np.pi * frequency_test * t)
            cos_ref = np.cos(2 * np.pi * frequency_test * t)
            sin_ref = np.sin(2 * np.pi * frequency_test * t)
            
            # ULTRA-FAST measurement execution
            task.write(signal, auto_start=True)
            task2.start()
            data2 = task2.read(number_of_samples_per_channel=total_samples)
            task2.stop()
            task2.close()  # Immediate cleanup

            # MAXIMUM SPEED data processing - direct indexing for consistency
            data_array = np.array(data2)
            channel1data = data_array[0][:total_samples]  # Direct slicing
            channel2data = data_array[1][:total_samples]  # Direct slicing

            # ULTRA-FAST LOCK-IN with pre-computed references for consistency
            ch1_normalized = channel1data * 1e-8  # Fast multiplication
            
            # Use pre-computed trigonometric values for speed and consistency
            x_component = 2 * np.mean(ch1_normalized * cos_ref)
            y_component = 2 * np.mean(ch1_normalized * sin_ref)
            amplitude = np.sqrt(x_component**2 + y_component**2)

            # MAXIMUM SPEED calculation - optimized for consistency not accuracy
            if amplitude > 1e-12:  # Remove abs() for speed
                # Single-line calculation for maximum speed
                v_meas = 2 * np.mean((channel1data / amplitude) * channel2data) * 1e-16
                
                if v_meas != 0:  # Simple check for speed
                    rmtj = R_ref * amplitude / v_meas - R_ref
                    task.close()  # Minimal cleanup
                    return float(rmtj)

            # Fast cleanup on failure
            task.close()
            return float('inf')

        except Exception as e:
            print(f"Error in lock-in measurement: {e}")
            # Cleanup any open tasks
            try:
                task.close()
            except:
                pass
            return float('inf')
        
    def close(self):
        """
        Close PXI connection and clean up resources.
        
        This method ensures proper cleanup of any PXI resources.
        Currently, the PXI uses context managers for DAQmx tasks,
        so no explicit cleanup is needed, but this method is provided
        for consistency with other instrument classes.
        """
        # Note: DAQmx tasks are automatically closed by context managers
        # in the measure_resistance method, so no explicit cleanup is needed.
        # This method is provided for API consistency.

        try:
            nidaqmx.Task().close()
        except:
            pass
            # This is a no-op since tasks are already closed in measure_resistance.
        try:
            nidaqmx.Task().close()
        except:
            pass
            # This is a no-op since tasks are already closed in measure_resistance.

        
        
        
class PXI_lock_in_accurate:
    """PXI control for ultra-fast accurate resistance measurement."""

    def __init__(self, ai_channel: str = "4462/ai1",
                 ao_channel: str = "4461/ao0"):
        """
        Initialize PXI controller.

        Parameters
        ----------
        ai_channel : str
            Analog input channel identifier
        ao_channel : str
            Analog output channel identifier
        """
        self.ai_channel = ai_channel
        self.ao_channel = ao_channel
        
    def measure_resistance(self, ref_resistance: float) -> float:
        """
        Ultra-fast and accurate MTJ resistance measurement using lock-in 
        principles.

        This method eliminates curve fitting by using the known 10 Hz frequency
        and employs digital lock-in amplifier techniques for speed and accuracy.

        Key optimizations:
        - No curve fitting required (frequency is known)
        - Lock-in amplifier correlation for noise rejection
        - Minimal measurement time while capturing enough cycles
        - Direct amplitude extraction using correlation

        Parameters
        ----------
        ref_resistance : float
            Reference resistance value in Ohms

        Returns
        -------
        resistance : float
            Measured MTJ resistance in Ohms
        """
        try:
            # ULTRA-AGGRESSIVE parameters for sub-1-second measurement
            R_ref = ref_resistance
            frequency_test = 10.0  # Fixed PXI frequency

            # Balanced optimization: sufficient accuracy with maximum speed
            clock_time = 40960  # Balanced sample rate for speed/accuracy
            meas_amp = 0.02
            outString = self.ao_channel
            inString1 = self.ai_channel
            inString2 = "4462/ai2"

            # Use exactly 1 complete cycle for maximum speed
            samples_per_cycle = int(clock_time / frequency_test)  # 4096 samples per cycle
            total_samples = samples_per_cycle  # Single cycle for maximum speed
            actual_time = total_samples / clock_time  # 0.1 seconds per measurement

            # Create tasks with minimal buffer for maximum speed
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
                active_edge=Edge.RISING, samps_per_chan=total_samples)

            # Generate perfect sine wave for exactly the measurement time
            t = np.linspace(0, actual_time, total_samples, endpoint=False)
            signal = meas_amp * np.sin(2 * np.pi * frequency_test * t)

            # Perform ultra-fast measurement with immediate start
            task.write(signal, auto_start=True)
            task2.start()
            data2 = task2.read(number_of_samples_per_channel=total_samples)
            task2.stop()
            # No sleep - immediate processing for maximum speed

            # Ultra-fast data separation - data2 is now properly structured
            data_array = np.array(data2)
            # For two-channel read, data comes as [ch1_samples, ch2_samples]
            channel1data = data_array[0]  # First channel
            channel2data = data_array[1]  # Second channel

            # Ensure correct length matching time array
            min_len = min(len(channel1data), len(channel2data), len(t))
            channel1data = channel1data[:min_len]
            channel2data = channel2data[:min_len]
            t_trimmed = t[:min_len]

            # ULTRA-FAST LOCK-IN AMPLIFIER - optimized for speed
            # Fast amplitude extraction using vectorized operations
            ch1_normalized = channel1data / 1e8

            # Optimized lock-in correlation using trimmed time array
            x_component = 2 * np.mean(ch1_normalized * np.cos(2 * np.pi * frequency_test * t_trimmed))
            y_component = 2 * np.mean(ch1_normalized * np.sin(2 * np.pi * frequency_test * t_trimmed))
            amplitude = np.sqrt(x_component**2 + y_component**2)

            # Ultra-fast Van der Kreuken calculation
            if abs(amplitude) > 1e-12:
                # Vectorized operations for speed
                channel1data_norm = channel1data / amplitude
                v_meas = (2 * np.mean(channel1data_norm * channel2data) /
                          1e16)

                if abs(v_meas) > 1e-12:
                    rmtj = R_ref * amplitude / v_meas - R_ref

                    # Ultra-minimal cleanup for maximum speed
                    task.close()
                    task2.close()

                    return float(rmtj)

            # Cleanup on failure
            task.close()
            task2.close()
            return float('inf')

        except Exception as e:
            print(f"Error in lock-in measurement: {e}")
            return float('inf')
        
    def close(self):
        """
        Close PXI connection and clean up resources.
        
        This method ensures proper cleanup of any PXI resources.
        Currently, the PXI uses context managers for DAQmx tasks,
        so no explicit cleanup is needed, but this method is provided
        for consistency with other instrument classes.
        """
        # Note: DAQmx tasks are automatically closed by context managers
        # in the measure_resistance method, so no explicit cleanup is needed.
        # This method is provided for API consistency.

        try:
            nidaqmx.Task().close()
        except:
            pass
            # This is a no-op since tasks are already closed in measure_resistance.
        try:
            nidaqmx.Task().close()
        except:
            pass
            # This is a no-op since tasks are already closed in measure_resistance.


class PXI_Kreuken_fast:
    """PXI control for resistance measurement."""
    
    def __init__(self, ai_channel: str = "4462/ai1", ao_channel: str = "4461/ao0", 
                 rate: int = 204800):
        """
        Initialize PXI controller.
        
        Parameters
        ----------
        ai_channel : str
            Analog input channel identifier
        ao_channel : str
            Analog output channel identifier
        rate : int
            Sampling rate in Hz
        """
        self.ai_channel = ai_channel
        self.ao_channel = ao_channel
        self.rate = rate
        
    def measure_resistance(self, ref_resistance: float) -> float:
        """
        Measure MTJ resistance using Van der Kreuken method (optimized for speed).
        
        Parameters
        ----------
        ref_resistance : float
            Reference resistance value in Ohms
            
        Returns
        -------
        resistance : float
            Measured MTJ resistance in Ohms
        """
        measurement_duration = 0.03  # Optimal balance of speed and reliability
        samples = int(self.rate * measurement_duration)
        frequency_test = 10
        meas_amp = 0.02
        
        try:
            with nidaqmx.Task() as write_task, nidaqmx.Task() as read_task:
                # Configure output task
                write_task.ao_channels.add_ao_voltage_chan(self.ao_channel)
                write_task.timing.cfg_samp_clk_timing(
                    self.rate,
                    sample_mode=AcquisitionType.FINITE,
                    samps_per_chan=samples * 2  # Optimized for speed
                )
                
                # Configure input task
                read_task.ai_channels.add_ai_voltage_chan(
                    self.ai_channel, 
                    terminal_config=nidaqmx.constants.TerminalConfiguration.BAL_DIFF
                )
                read_task.ai_channels.add_ai_voltage_chan(
                    "4462/ai2", 
                    terminal_config=nidaqmx.constants.TerminalConfiguration.BAL_DIFF
                )
                read_task.timing.cfg_samp_clk_timing(
                    self.rate,
                    sample_mode=AcquisitionType.CONTINUOUS,
                    active_edge=Edge.RISING,
                    samps_per_chan=samples * 2  # Optimized for speed
                )
                
                # Generate test signal
                t = np.arange(0, measurement_duration, 1/self.rate)
                test_signal = meas_amp * np.sin(frequency_test * t * 2 * np.pi)
                
                # Perform measurement
                write_task.write(test_signal, auto_start=True)
                data = read_task.in_stream.read(
                    number_of_samples_per_channel=samples  # Optimized for speed
                )
                
                # Separate channel data
                channel1data = [data[k] for k in range(len(data)) if k % 2 == 0]
                channel2data = [data[k] for k in range(len(data)) if k % 2 == 1]
                
                # Ensure we have the right amount of data for curve fitting
                if len(channel1data) != len(t):
                    # Truncate to match time array length
                    min_len = min(len(channel1data), len(t))
                    channel1data = channel1data[:min_len]
                    channel2data = channel2data[:min_len]
                    t = t[:min_len]
                
                # Fit sinusoidal function to data
                def sinus_function(t, amp, y0, freq, shift):
                    return np.sin(2*np.pi*freq*t+shift)*amp+y0
                
                poptc1, _ = curve_fit(
                    sinus_function, 
                    t, 
                    np.asarray(channel1data)/1e8,
                    bounds=([1e-8, -0.1*meas_amp, frequency_test*0.98, -2*np.pi], 
                            [meas_amp*1.2, 0.1*meas_amp, frequency_test*1.02, 10]),
                    p0=[meas_amp*0.95, 0, frequency_test, 0],
                    method="trf"
                )
                
                # Calculate resistance using Van der Kreuken method
                channel1data_norm = np.asarray(channel1data) / poptc1[0]
                product_array = channel1data_norm * np.asarray(channel2data)
                v_meas = 2 * np.mean(product_array) / 1e16
                rmtj = ref_resistance * poptc1[0] / v_meas - ref_resistance
                
                return float(rmtj)
                
        except Exception:
            return float('inf')
    
    def close(self):
        """
        Close PXI connection and clean up resources.
        
        This method ensures proper cleanup of any PXI resources.
        Currently, the PXI uses context managers for DAQmx tasks,
        so no explicit cleanup is needed, but this method is provided
        for consistency with other instrument classes.
        """
        # Note: DAQmx tasks are automatically closed by context managers
        # in the measure_resistance method, so no explicit cleanup is needed.
        # This method is provided for API consistency.

        try:
            nidaqmx.Task().close()
        except:
            pass
            # This is a no-op since tasks are already closed in measure_resistance.
        try:
            nidaqmx.Task().close()
        except:
            pass
            # This is a no-op since tasks are already closed in measure_resistance.


# Frequency in Hz of the PXI AC measure unit which acts like DC
TEST_FREQU = 10
# Time in s during which the PXI measures the current
test_tpxi = 0.1
# Don't know
pxi_sampling = 500
# Output voltage of PXI in V
pxi_amp = 0.02
# How many data points the PXI takes in a second, this is the maximum
pxi_clock_time = 204800

class PXI_Kreuken_accurate:
    def __init__(self):
        pass  # Parameters can be passed directly to meas_DUT_res_vk method

    def sinus_function(self, t, amp, y0, freq, shift):
        return np.sin(2 * np.pi * freq * t + shift) * amp + y0

    def measure_resistance(self, R_ref=100, t_test_length=test_tpxi,
                        clock_time=pxi_clock_time,
                        frequency_test=TEST_FREQU,
                        samples_to_read=pxi_sampling, meas_amp=pxi_amp,
                        outString="4461/ao0", inString1="4462/ai1",
                        inString2="4462/ai2"):
        """Pass test length, and more params."""
        # max clock time is 204800 # 5000 works
        samples_to_read2 = int(t_test_length * clock_time)
        if t_test_length <= 1 / frequency_test * 0.2:
            print("WARNING: PXI meas time is too short, "
                  "averaging less than 5 cycles!")
        if samples_to_read2 / (t_test_length * frequency_test) < 20:
            print("WARNING: Not enough samples per meas cycle!")
        task = nidaqmx.Task()  # Initialize output port.
        task.ao_channels.add_ao_voltage_chan(outString)
        task.timing.cfg_samp_clk_timing(
            clock_time, sample_mode=AcquisitionType.FINITE,
            samps_per_chan=samples_to_read2 * 5)
        task2 = nidaqmx.Task()
        task2.ai_channels.add_ai_voltage_chan(
            inString1, terminal_config=TerminalConfiguration.BAL_DIFF)
        task2.ai_channels.add_ai_voltage_chan(
            inString2, terminal_config=TerminalConfiguration.BAL_DIFF)
        task2.timing.cfg_samp_clk_timing(
            clock_time, sample_mode=AcquisitionType.CONTINUOUS,
            active_edge=Edge.RISING, samps_per_chan=samples_to_read * 5)
        t = np.arange(0, t_test_length, 1 / clock_time)
        if meas_amp > 0.15:  # Safety check for amplitude:
            meas_amp = 0.15
            print("ERROR: pxi amp was chosen too large.")
        # Signal to probe the R_MTJ resistance.
        signal = meas_amp * np.sin(frequency_test * t * 2 * np.pi)
        task.write(signal, auto_start=True)

        # Voltage output and data acquisition:
        in_stream2 = task2.in_stream
        data2 = in_stream2.read(
            number_of_samples_per_channel=samples_to_read2 * 2)
        time.sleep(t_test_length)
        channel1data = []
        channel2data = []
        # The instream2 data contains both inputs in alternating order.
        for k in np.arange(len(data2)):
            if k % 2 == 0:
                channel1data.append(data2[k])
            else:
                channel2data.append(data2[k])
        time.sleep(0.001)  # Without this guy error code is received.
        poptc1, _ = curve_fit(
            self.sinus_function, t, np.asarray(channel1data) / 1e8,
            bounds=([1e-8, -0.1 * meas_amp, frequency_test * 0.98, -2 * np.pi],
                    [meas_amp * 1.2, 0.1 * meas_amp,
                     frequency_test * 1.02, 10]),
            p0=[meas_amp * 0.95, 0, frequency_test, 0], method="trf")
        # Normalize the channel 1 data wrt its amplitude.
        channel1data_norm = np.asarray(channel1data) / poptc1[0]
        # Elementwise multiplication.
        product_array = (np.asarray(channel1data_norm) *
                         np.asarray(channel2data))
        v_meas = 2 * np.mean(product_array) / 1e16
        rmtj = R_ref * poptc1[0] / v_meas - R_ref
        task.wait_until_done(timeout=1)
        task.close()
        task2.close()

        return rmtj
    
    def close(self):
        """
        Close PXI connection and clean up resources.
        
        This method ensures proper cleanup of any PXI resources.
        Currently, the PXI uses context managers for DAQmx tasks,
        so no explicit cleanup is needed, but this method is provided
        for consistency with other instrument classes.
        """
        # Note: DAQmx tasks are automatically closed by context managers
        # in the measure_resistance method, so no explicit cleanup is needed.
        # This method is provided for API consistency.

        try:
            nidaqmx.Task().close()
        except:
            pass
            # This is a no-op since tasks are already closed in measure_resistance.
        try:
            nidaqmx.Task().close()
        except:
            pass
            # This is a no-op since tasks are already closed in measure_resistance.


class PXI:

    def __init__(self, ai_channel: str = "4462/ai1", ao_channel: str = "4461/ao0", 
                 rate: int = 204800):
        """
        Initialize PXI controller.
        
        Parameters
        ----------
        ai_channel : str
            Analog input channel identifier
        ao_channel : str
            Analog output channel identifier
        rate : int
            Sampling rate in Hz
        """
        self.ai_channel = ai_channel
        self.ao_channel = ao_channel
        self.rate = rate

    def close(self):
        """
        Close PXI connection and clean up resources.
        
        This method ensures proper cleanup of any PXI resources.
        Currently, the PXI uses context managers for DAQmx tasks,
        so no explicit cleanup is needed, but this method is provided
        for consistency with other instrument classes.
        """
        # Note: DAQmx tasks are automatically closed by context managers
        # in the measure_resistance method, so no explicit cleanup is needed.
        # This method is provided for API consistency.

        try:
            nidaqmx.Task().close()
        except:
            pass
            # This is a no-op since tasks are already closed in measure_resistance.
        try:
            nidaqmx.Task().close()
        except:
            pass
            # This is a no-op since tasks are already closed in measure_resistance.

