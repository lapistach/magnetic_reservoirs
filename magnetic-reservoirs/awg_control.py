#!/usr/bin/env python3
"""
AWG Control Module

Handles pulse generation and waveform upload for dual-channel AWG operation.
"""

import os
from typing import List, Tuple
import numpy as np
import pyvisa


class AWG:
    """AWG control for pulse generation using file-based approach."""
    
    def __init__(self, address: str = "TCPIP0::localhost::inst0::INSTR", 
                 sample_rate: float = 64e9, timeout: int = 60000): # one minute timeout for timeout errors
        """
        Initialize AWG controller.
        
        Parameters
        ----------
        address : str
            VISA address of the AWG
        sample_rate : float
            Sample rate in Hz
        """
        self.rm = pyvisa.ResourceManager()
        self.awg = self.rm.open_resource(address)
        self.awg.timeout = timeout
        self.sample_rate = sample_rate
        self._setup_awg()
        
        
    def _setup_awg(self):
        """Configure AWG for dual-channel operation."""
        self.awg.write("*RST")
        #self.awg.write("*CLS")  # Clear any errors but don't reset
        self.awg.write(f":SOUR:FREQ:RAST {int(self.sample_rate)}")
        self.awg.write(":INST:DACM DUAL")
        self.awg.write(":INST:MEM:EXT:RDIV DIV2")
        
        for ch in [1, 4]:
            self.awg.write(f":TRAC{ch}:MMOD EXT")
            self.awg.write(f":VOLT{ch} 1")
            self.awg.write(f":OUTP{ch} ON")
            self.awg.write(f":OUTP{ch}:COUP DC")
            self.awg.write(f":OUTP{ch}:POL NORM") 
            self.awg.write(f":INIT{ch}:CONT OFF")
            
    def create_pulse_sequence(self, signal_array: List[float], 
                            pulse_duration: float, pulse_voltage: float) -> Tuple[np.ndarray, float]:
        """
        Create pulse sequence from signal array.
        
        Parameters
        ----------
        signal_array : List[float]
            Array of amplitude factors for each pulse
        pulse_duration : float
            Duration of each pulse in seconds
        pulse_voltage : float
            Base voltage amplitude
            
        Returns
        -------
        waveform : np.ndarray
            Generated waveform data
        total_duration : float
            Total duration of the waveform in seconds
        """
        samples_per_pulse = max(2, int(pulse_duration * self.sample_rate))
        sequence_samples = samples_per_pulse * len(signal_array)
        padding_samples = max(int(sequence_samples * 0.2), 2)
        total_samples = sequence_samples + 2 * padding_samples
        
        waveform = np.zeros(total_samples)
        
        for i, amplitude_factor in enumerate(signal_array):
            start_sample = padding_samples + i * samples_per_pulse
            end_sample = padding_samples + (i + 1) * samples_per_pulse
            voltage = amplitude_factor * pulse_voltage
            waveform[start_sample:end_sample] = voltage
        
        return waveform, total_samples / self.sample_rate
    
    def upload_waveform(self, segment: int, waveform: np.ndarray, 
                       ch1_amplitude: float, ch4_amplitude: float):
        """
        Upload waveform to both channels with independent amplitude scaling.
        
        Parameters
        ----------
        segment : int
            Segment number for waveform storage
        waveform : np.ndarray
            Waveform data to upload
        ch1_amplitude : float
            Amplitude scaling factor for channel 1
        ch4_amplitude : float
            Amplitude scaling factor for channel 4
        """
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
            filename = f"{temp_dir}\\pulse_segment{segment}_CH{ch}.csv"
            
            with open(filename, 'w') as f:
                f.write("SampleRate = 64000000000\n")
                f.write("SetConfig=true\n")
                
                for sample in padded_waveform:
                    scaled_sample = float(sample) * float(amplitude)
                    if abs(scaled_sample) < 1e-12:
                        scaled_sample = 0.0
                    f.write(f"{scaled_sample:.10f}\n")
            
            import_cmd = f":TRAC{ch}:IMP {segment},'{filename}',CSV,IONLY,OFF,ALEN"
            self.awg.write(import_cmd)
            self.awg.write(f":TRAC{ch}:COUN 1")
            
    def send_waveform(self, segment: int):
        """
        Send waveform with optimized triggering.
        
        Parameters
        ----------
        segment : int
            Segment number to trigger
        """
        # Select segment for both channels simultaneously
        self.awg.write(f":TRAC1:SEL {segment};:TRAC4:SEL {segment}")
        
        # Trigger both channels
        self.awg.write(":INIT:IMM;:TRIG:BEG;:ABOR")
        
    def close(self):
        """Close the AWG connection."""
        try:
            self.awg.close()
            self.rm.close()
        except Exception:
            pass

    

if __name__ == "__main__":
    # Example usage
    awg = AWG()
    signal_array = [0, 1, 0]
    pulse_duration = 10e-9
    pulse_voltage = .5
    
    waveform, total_duration = awg.create_pulse_sequence(signal_array, pulse_duration, pulse_voltage)
    print(f"Generated waveform with total duration: {total_duration:.6f} seconds")
    
    awg.upload_waveform(segment=1, waveform=waveform, ch1_amplitude=1.0, ch4_amplitude=0.0)
    awg.send_waveform(segment=1)
    
    awg.close()