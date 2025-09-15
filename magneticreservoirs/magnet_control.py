#!/usr/bin/env python3
"""
Simplified Danfysik 7000 Magnet Control

Clean implementation for basic magnet control functionality:
- Ramp to specific current
- Ramp to zero
"""

import time
import pyvisa

DANFYSIK_VISA_ADDRESS = "ASRL4::INSTR"


class Danfysik7000:
    """Danfysik 7000 power supply control for magnet ramping."""
    
    def __init__(self, **kwargs):
        """Initialize magnet controller with default parameters."""
        # Set default communication parameters
        self.config = {
            "write_termination": "\r",
            "read_termination": "\n\r", 
            "baud_rate": 9600,
            "timeout": 500
        }
        self.config.update(kwargs)
        
        # Initialize VISA connection
        self.rm = pyvisa.ResourceManager()
        self.device = self.rm.open_resource(DANFYSIK_VISA_ADDRESS)
        
        # Apply configuration
        for param, value in self.config.items():
            if hasattr(self.device, param):
                setattr(self.device, param, value)
    
    def _write(self, command: str):
        """Write command to device."""
        self.device.write(command)
    
    def _read(self):
        """Read response from device."""
        return self.device.read()
    
    def _set_current(self, current_ma: int):
        """Set output current in mA."""
        self._write(f"DA 0,{current_ma:d}")
    
    def _set_output(self, enable: bool):
        """Enable or disable output."""
        command = "N" if enable else "F"
        self._write(command)
    
    def rampToCurrent(self, target_current: float):
        """
        Ramp magnet to target current.
        
        Parameters
        ----------
        target_current : float
            Target current in Amperes (-19.5 to 19.5 A)
        """
        if abs(target_current) > 19.5:
            raise ValueError("Target current must be between -19.5 and 19.5 A")
        
        # Convert A to mA for device communication
        target_ma = int(target_current * 1000)
        
        # Set up magnet for ramping
        print(f"Ramping magnet to {target_current} A ...")
        self._write("REM")  # Set to remote mode
        self._set_output(True)  # Enable output
        self._set_current(target_ma)  # Set target current
        self._write("LOC")  # Apply the current setting
        
        # Wait for ramping to complete
        time.sleep(3.0)
    
    def rampToZero(self):
        """Ramp magnet current to zero."""
        print("Ramping magnet to zero ...")
        self._write("REM")  # Set to remote mode
        self._set_output(True)  # Enable output
        self._set_current(0)  # Set current to zero
        self._write("LOC")  # Apply the setting
        
        # Wait for ramping to complete
        time.sleep(3.0)
    
    def close(self):
        """Close the device connection."""
        try:
            self.device.close()
            self.rm.close()
        except Exception:
            pass