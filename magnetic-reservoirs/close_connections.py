#!/usr/bin/env python3
"""
MTJ Setup Cleanup - Main Close Module

Safe shutdown and cleanup routine for MTJ experimental setup.
This script ensures all instruments are properly closed and the magnetic field is ramped to zero.

Use this script whenever you want to safely shut down the setup:
- Ramps magnet to zero field
- Closes magnet connection
- Closes AWG connection
- Closes PXI connection
- Handles errors gracefully

Usage:
    python main_close.py
"""

from awg_control import AWG
from pxi_control import PXI
from magnet_control import Danfysik7000
import pyvisa


def cleanup_setup():
    """
    Safely clean up and close all instruments in the MTJ setup.
    
    This function:
    1. Ramps the magnet to zero field
    2. Closes the magnet connection
    3. Closes the AWG connection
    4. Closes the PXI connection which is done automatically in nidaqmx
    """
    magnet = Danfysik7000()
    try:
        magnet.rampToZero()    
        magnet.close()
    except:  
        magnet.close()
    print("Magnet connection closed")
    
    try:
        rm = pyvisa.ResourceManager()
        try:
            sessions = rm.list_opened_resources()
            for session in sessions:
                try:
                    session.close()
                except Exception as e:
                    print(f"  Error closing session {session}: {e} in ressource manager")
        except Exception as e:
            print(f"Could not list opened resources: {e}")
        
        rm.close()
        print("AWG and ResourceManager closed")
        
    except Exception as e:
        print(f"Error in close_all_visa_connections: {e}")
    
    pxi = PXI()
    pxi.close()
    print("PXI connection closed.")
 
    print("The setup is now ready for safe power-down or next experiment.")

if __name__ == "__main__":
    cleanup_setup()
        
