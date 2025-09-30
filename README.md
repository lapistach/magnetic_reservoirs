# Magnetic reservoirs

A package for reservoir computing with spin-orbit torque magnetic tunnel juntions (MTJ). 
This code is used as the interface between an AWG, a PXI and a magnet to send pulses to the MTJ and retrieve its reservoir state. You can also use it to train and test your own magnetic reservoir for classification and prediction tasks.

This work was done during my master's project at the *Interface Physics and Magnetism* laboratory at ETH Zürich between April 2025 and August 2025.

For more information, you can check my [master's report](masters_report.pdf) and my [technical report](technical_report.pdf).


## Installation

To install the package, run the usual commands:
```
git clone https://github.com/lapistach/magnetic_reservoirs.git
cd magnetic_reservoirs
pip install .
```

For an editable install:
```
git clone https://github.com/lapistach/magnetic_reservoirs.git
cd magnetic_reservoirs
pip install -e .
```


## Example

>[!WARNING]
>This code is designed for the specific setup described in both reports. It is composed of a Keysight AWG model M8195A, a National Instruments PXI with ports 4462 and 4461, and a Danfysik Magnet. The reservoir itself is composed of spin-orbit torque magnetic tunnel junctions supplied by IMEC in Belgium. For the complete setup, go check the [technical report](technical_report.pdf).


Once you have setup your MTJ correctly, you can test it quickly by running 'classify_with_one_mtj.py'. Don't forget to create an empty folder to store the results that you give to the main function.

You can play around with the experiment by tuning :
- the resistance measurement method, you can run 'choose_resistance.py' for an educated guess (see script's header for more information)
- the filter applied to the input data
- the discretisation of the data (number of features)

Important functions all have descriptions. 
You can also pick from multiple datasets described in 'datasets.py'.
If you want to do prediction with several virtual mtjs, run 'predict_vmtj_architecture.py'. 
If you want to classify with several virtual mtjs, run 'classifiy_with_several_mtj.py'.
Instead of handpicking the hyperparameters, you can run optuna studies with 'optuna_classify_with_one_mtj.py' and 'optuna_classifiy_with_several_mtj.py'.

The playground directory contains a bunch of loose scripts used for side quests.

>[!NOTE]
>If you want to use 'datasets.py' or the optuna scripts, you need to install keras and optuna.
