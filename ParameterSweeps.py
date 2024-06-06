# -*- coding: utf-8 -*-
"""
Created on Wed Jan 18 15:25:44 2023

@author: rigoc
"""
#%%
import sys
sys.path.append('C:\\Users\\rigoc\\Desktop\\KirtlandDocs\\AFRL_DATA\\APPS\\CustomLibraries')
#%%
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
import os
from os.path import join

import CapacitanceVoltageFit as cvf



## Create results for figure 5



i = 0

# Preliminary inputs for absorber and barrier layer epsilons
epsAL = 15.3
epsBL = 12.75
epsCL = 15.3


#Folder paths for the different parameter sweeps
paths = ["AbsorberDopingSweep", "BarrierDopingSweep", "BarrierThicknessSweep","ContactThicknessSweep"]

#which index in the inital guess array should we change? 
guessIndices = {"AbsorberDopingSweep": 3,\
                "BarrierDopingSweep": 1,\
                "BarrierThicknessSweep":2,\
                "ContactThicknessSweep":np.nan}
    
# normalize the title inputs to t
factors = [1/1e15, 1/1e15, 1000, 1000]


# Headernames
headernamesSilvaco = ["Voltage", "Capacitance"]
skipLines = 4

dataSets = {}
bestFits = {}
parameters = {}
targets = {}

outputNames  = ["Contact Doping (x 1e17 cm-3)", "Barrier Doping (x 1e15 cm-3)",\
                "Barrier Thickness (nm)", "Absorber Doping (x 1e15)", "ec (meV)",\
                "Cpar (pF)", "epsBL (e0)", "epsAL (e0)", "epsBL (e0)", 'tCL(um)']
    
targetNames =  ["Contact Doping (x 1e17 cm-3)", "Barrier Doping (x 1e15 cm-3)",\
                "Barrier Thickness (nm)", "Absorber Doping (x 1e15)","tcl(nm)"]

    
#Loop through all of the parameter sweeps
for nameIndex, path in enumerate(paths):
    
    #initialize the guess: [contact doping, barrier doping, barrier thickness, absorber doping]
    guess = [0.03, 3, 200, 3]
    
    #initialize the target parameters
    target = [0.03, 3, 200, 3]
    # Crawl through the specified path
    for root, dirs, files in os.walk(path):
        ## Begin crawl through a parameter sweep
        
        # Take in the number of files
        numFiles = len(files)
        
        fig, ax = plt.subplots()
        
        for index, file in enumerate(files):
            #join the path and file
            pathAndFile = join(path, file)
            
            #Read the CV file
            df  = pd.read_table(pathAndFile, sep = ' ', names = headernamesSilvaco, skiprows = skipLines,\
                                index_col = False)
                
            #simulation had a reverse polarity, negative bias should deplete the absorber
            x = -np.array(df["Voltage"])
            
            #Convert capacitance density from F/um to units of nF/cm^(2)
            y = np.array(df["Capacitance"]*1e6**2*1e9/100**2)
            
            #determine size of the capacitance array
            if index == 0:
                # Include the appplied voltage array
                dataSets[path] = np.zeros([len(x), len(files) + 1])
                
                # First column of the matrix is the applied voltage
                dataSets[path][:,0] = x
                
                # You don't need the applied voltage since it's the same as the simulated
                # array
                bestFits[path] = np.zeros([len(x), len(files)])
                
                
                outputs = np.zeros([len(files), 10])
                targetout = np.zeros([len(files), 5])
                
                
     
            dataSets[path][:,index  + 1] = y
            
            
            ax.plot(-df["Voltage"], df["Capacitance"]*1e6**2*1e9/100**2,'.')
            inputs = cvf.inputs()
            
            target = guess
            xfit = x
            yfit = y
            
            """Is the parameter that is being sweeped an input parameter in the model?"""
            if ~np.isnan(guessIndices[path]): 
                """Yes? then provide a guess that  is 60% of the target parameter"""
                if file.split("_")[0].__contains__("p"):
                    guess[guessIndices[path]] =  -np.float(file.split("_")[nameIndex + 1])*factors[nameIndex]*0.6
                    target[guessIndices[path]] = -np.float(file.split("_")[nameIndex + 1])*factors[nameIndex]                              
                else:
                    guess[guessIndices[path]] =  np.float(file.split("_")[nameIndex + 1])*factors[nameIndex]*0.6
                    target[guessIndices[path]] = np.float(file.split("_")[nameIndex + 1])*factors[nameIndex]

            else:
                xfitIndices = x<=0.35
                xfit = x[xfitIndices]
                yfit = y[xfitIndices]
                
                
                
            try: 
                bestfit, cov, finalOutput = cvf.fitCapacitance_nBn(inputs, xfit, yfit, 130, *guess, eC = 0, parasiticCap = 0, epsBarr = epsBL,\
                                   epsAbs = epsAL, epsCon = epsCL, fitParasitic = False)
            except ValueError:
                finalOutput = np.nan*np.ones(9)
                
            
        
            if np.isnan(finalOutput[0]):
                capacitance = np.nan*np.ones(len(x))
            else:    
                capacitance = cvf.determineCapacitance_nBn(inputs, x, 130, *finalOutput)
            
            bestFits[path][:,index] = capacitance
            outputs[index,:-1] = finalOutput
            outputs[index,-1] = 2
            
            targetout[index,:-1] = target
            targetout[index,-1] = np.float(file.split("_")[-1][:-4])
            
            
            ax.plot(x, capacitance,'-')
        parameters[path] = np.transpose(pd.DataFrame(data = outputs, columns  = outputNames))
        targets[path] = np.transpose(pd.DataFrame(data = targetout, columns = targetNames))
    
#%% Calculate the depletion width in an absorber with 1e14 cm^-3 doping  by first calculating potential drop across the absorber
import numpy as np
import CapacitanceVoltageFit as cvf
import matplotlib.pyplot as plt

e0   = 8.854187812813e-12 # Farads/m
epsAL = 15.3  #Units of e0
epsBL = 12.75 #Units of e0
epsCL = 15.3  #Units of e0


NdAL = 0.5 #Units of 1e15 cm^(-3)
NdAL_SI  = NdAL*1e15*(100**3) #Units of m^(-3)

NdCL = 0.03  #Units of 1e17 cm^(-3)
NdBL = 5     #Units of 1e15 cm^(-3)
tBL = 200

inputs = cvf.inputs()


voltage  = np.linspace(-3, 3,601 )



cvInputs = [NdCL, NdBL, tBL, NdAL, 0, 0, epsBL, epsAL, epsCL]


capnFCm2, val, phiBarrier, phiContact, vbi  = cvf.capacitance_nBnReturnPotential(inputs, voltage, 130, *cvInputs)

appliedVoltageInterest = -1.7
indexInterest = appliedVoltageInterest == voltage
valInterest = val[indexInterest][0]
depletionDepth = np.sqrt(2*e0*epsAL/inputs.e/NdAL_SI*(-valInterest))*1e6

string = 'Depletion width in absorber at {:1.1f} V is {:1.2f} micron with'.format(appliedVoltageInterest,depletionDepth)
string2 = 'an absorber potential drop of {:1.2f} V'.format(valInterest)
print(string + string2)




