# CapacitancenBn2022
## Description of Files:
1. CVData.xlsx - File containing the capacitance-voltage data of an InAs/InAsSb device nBn. The first row is header information and a note on the data. The second row contains column information of the file. Column 1 is the applied voltage in Volts, the second column is the measured capacitance file in Farads, and the third column is the measured conductance.
2. CapacitanceVoltageFit.py - File containing the functions to model the capacitance-voltage sweep of a device nBn. It also contains the function to fit to a measured capacitance-voltage curve. Import this file to use the functions within it to model and fit nBn capacitance-voltage curves.
3. Figure1Data.xlsx - File containing simulated data from Silvaco of a sample nBn. Dedicated Excel sheets are: "Electron-holeConcentration"; "ElectricField"; "Potential"; and "BandDiagram"
4. AbsorberDopingSweep; BarrierDopingSweep; BarrierThicknessSweep; ContactThicknessSweep - Contains the capacitance voltage sweeps of four different parameter sweeps simulated by Silvaco. The fitCapacitance_nBn function is called to extract best fit nBn parameters for the different sweeps and the code that implements this can be found in ParameterSweeps.py
5. fitCapacitance.py - Script that imports CapacitanceVoltageFit.py and demonstrates the model.
