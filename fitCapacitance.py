# -*- coding: utf-8 -*-
"""
Created on Thu Aug 25 15:55:12 2022

@author: rigoc
"""

#Import necessary libraries
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import CapacitanceVoltageFit as cvf #Make sure "CapacitanceVoltageFit.py" is in the working directory
#%% CREATE DATA FOR FIGURE 2

"""Define inputs to function"""
absorberEps = 15.3056 #relative permittivity, units of e0
contactEps = 15.3056 #relative permittivity, units of e0
barrierEps = 12.795 #relative permittivity, units of e0
absorberDoping =  0.5 #in units of 1e15 cm^-3
barrierThick = 200 #in units of nm
barrierDope = -2 #in units of 1e15 cm^-3 (negative means a p-type barrier and positive, n-type barrier)
constants = cvf.inputs()
contactDoping = 0.08 #in units of 1e17 cm^-3
temperature = 80 #Kelvin
conductionOffset = 0 #conduction band offset between absorber and contact in units of meV
par = 0 #parasitic capacitance in units of picoFarads
mSize = 1000 #mesa side length necessary for considering parasitic capacitance, units of um

vApplied = np.linspace(-3,3, 300) #voltage sweep, Volts




#Calculate the capacitance, and corresponding potential drops across each layer in the nBn
capnFCm2, val, phiBarrier, phiContact, vbi = cvf.capacitance_nBnReturnPotential(constants, vApplied, temperature,\
                                            contactDoping, barrierDope, barrierThick, absorberDoping, eC = conductionOffset,\
                                            parasitic = par,\
                                            epsBar = barrierEps, epsAbs = absorberEps, epsCon = contactEps, size = mSize)

plt.figure()
plt.plot(vApplied, val, 'k-', label = "$\\Phi_{AL}$")
plt.plot(vApplied, phiBarrier, color = 'gray', label = "$\\Phi_{BL}$")
plt.plot(vApplied, phiContact, 'k--', label = "$\\Phi_{CL}$")
plt.plot(vApplied, val + phiContact + phiBarrier,'--', color ="gray", label ="$\\Phi_{AL} +\\Phi_{BL} + \\Phi_{CL}$" )
plt.plot(vApplied, vApplied + vbi,'k.', markersize = 0.98, label = "$V_{applied} + V_{bi}$")
plt.legend(loc = "lower right")
plt.xlabel("Applied Voltage (V)")
plt.ylabel("Potential Drop (V)")
plt.xlim([-3,3])
plt.ylim([-3,3])

"""Plot the inset to figure 2"""
plt.figure()
plt.plot(vApplied, val, 'k-', label = "$\\Phi_{AL}$")
plt.plot(vApplied, phiBarrier, color = 'gray', label = "$\\Phi_{BL}$")
plt.plot(vApplied, phiContact, 'k--', label = "$\\Phi_{CL}$")
plt.plot(vApplied, val + phiContact + phiBarrier,'--', color ="gray", label ="$\\Phi_{AL} +\\Phi_{BL} + \\Phi_{CL}$" )
plt.plot(vApplied, vApplied + vbi,'k.', markersize = 0.98, label = "$V_{applied} + V_{bi}$")
plt.legend(loc = "lower right")
plt.xlabel("Applied Voltage (V)")
plt.ylabel("Potential Drop (V)")
plt.xlim([-0.31,0.04])
plt.ylim([-0.06,0.1])


#%% CREATE DATA FOR FIGURE 3

#Calculate the surface charge density for the contact layer
qCL = cvf.chargeAbs(constants, temperature, contactDoping*100, -phiContact, epsin = contactEps)/constants.e/1e11/100**2

#Calculate the surface charge density for the absorber layer
qAL = cvf.chargeAbs(constants, temperature, absorberDoping, val, epsin = absorberEps)/constants.e/1e11/100**2

#Calculate the surface charge density for the barrier layer
qBL = barrierThick*1e-9*barrierDope*1e15*100**3/1e11/100**2

#calculate the net surface charge density
qNet = qCL + qAL + qBL


#Plot the charge components
plt.figure()
plt.plot(vApplied, qCL,'k--',label = "$Q_{CL}$")
plt.plot(vApplied, qAL, 'k-', label = "$Q_{AL}$")
plt.plot(vApplied, qBL*np.ones(len(vApplied)), '-', color = "gray", label = "$Q_{BL}$")
plt.plot(vApplied, qNet, "--", color = "gray", label = "$Q_{Net}$")
plt.legend(loc = "lower left")
plt.xlabel("Applied Votlage (V)")
plt.ylabel("Surface Charge ($10^{11}$ electrons/$cm^2$)")
plt.xlim([-3,3])
plt.ylim([-4.5,5])


#Plot the inset
plt.figure()

#put voltage in mV and charge in 1e9 electrons/cm^2
plt.plot(vApplied*1000, qCL*100,'k--',label = "$Q_{CL}$")
plt.plot(vApplied*1000, qAL*100, 'k-', label = "$Q_{AL}$")
plt.plot(vApplied*1000, qBL*100*np.ones(len(vApplied)), '-', color = "gray", label = "$Q_{BL}$")
plt.plot(vApplied*1000, qNet*100, "--", color = "gray", label = "$Q_{Net}$")
plt.legend(loc = "lower left")
plt.xlabel("Applied Votlage (mV)")
plt.ylabel("Surface Charge ($10^{9}$ electrons/$cm^2$)")
plt.xlim([-340, 80])
plt.ylim([-45,60])






#%% CREATE DATA FOR FIGURE 4
plt.figure()
plt.plot(vApplied, capnFCm2, 'k')
plt.xlabel("Applied Voltage (V)")
plt.ylabel("Capacitance (nF/cm$^2$)")
plt.xlim([-3, 3])
plt.ylim([3, 39])

barrierCapacitance = barrierEps*constants.e0/(barrierThick*1e-9)*1e9/100**2
capSubtracted = 1/(1/capnFCm2 - 1/barrierCapacitance)

invcap = 1/capnFCm2**2
incapBarr = 1/capSubtracted**2
plt.figure()
plt.plot(vApplied,invcap,'k', label = "1/C$^2$ no barrier subtraction")
plt.plot(vApplied, incapBarr,'k--', label = "barrier subtracted")
plt.xlabel("Applied Voltage (V)")


#Fit range to perform Schottky approximation
fitRange = [-0.5,-3]
lessThan = vApplied <= fitRange[0]
greaterThan = vApplied >= fitRange[1]
indices = np.logical_and(greaterThan, lessThan)
xin = vApplied[indices]
yin = invcap[indices]
yin2 = incapBarr[indices]


#Now perform a voltage subtraction
voltageSubtracted = vApplied - phiBarrier
lessThan2 = voltageSubtracted <=-0.5
greaterThan2 = voltageSubtracted >=-2.86

indices2 = np.logical_and(greaterThan2, lessThan2)
xin3 = voltageSubtracted[indices2]
yin3 = incapBarr[indices2]



#Schottky approximation with no barrier subtraction
coeffs = np.polyfit(xin, yin, 1)
polynoSub = np.poly1d(coeffs)
nd1 = -2/(constants.e*1e9)/absorberEps/(constants.e0*1e9/100)/(coeffs[0])/1e14 # doping in units of 1e14 cm^-3

#Schottky approximation with just barrier capacitance subtraction
coeffs2 = np.polyfit(xin, yin2, 1)
polysubtration = np.poly1d(coeffs2)
nd2 = -2/(constants.e*1e9)/absorberEps/(constants.e0*1e9/100)/(coeffs2[0])/1e14 #doping in units of 1e 14 cm^-3


#Schottky approximation with both barrier capacitance subtraction and barrier potential drop substraction
coeffs3 = np.polyfit(xin3, yin3, 1)
polysubtration2 = np.poly1d(coeffs3)
nd3 = -2/(constants.e*1e9)/absorberEps/(constants.e0*1e9/100)/(coeffs3[0])/1e14 #doping in units of 1e 14 cm^-3




plt.plot(xin, polynoSub(xin), color ="gray", label = "N$_D$ = %0.2f x 10$^{14}$ cm$^{-3}$"%(nd1))
plt.plot(xin, polysubtration(xin), "--", color ="gray", label = "N$_D$ = %0.2f x 10$^{14}$ cm$^{-3}$"%(nd2))

plt.legend(loc = "upper right")


#%% CREATE DATA FOR FIGURE 5
files = ["Mystery1.dat","Mystery_Device_2.dat","Mystery_Device_3.dat"]
names = ["Mystery1Silvaco","Mystery2Silvaco","Mystery3Silvaco"]
epsAL = 15.3
epsBL = 12.75


# samples = [cvf.SampleCV(epsAL, epsBL, 200, name = singleName) for singleName in names]
headernamesSilvaco = ["Voltage", "Capacitance"]
skipLines = 4

"Read all of the files"
dataframes = [pd.read_table(files[index],sep = ' ', names = headernamesSilvaco, skiprows = skipLines, index_col = False) for index in range(len(files))]
voltagesArrays = [np.array(dataframe["Voltage"]) for dataframe in dataframes]
capacitanceArrays = [np.array(dataframe["Capacitance"])*1e6**2*1e9/100**2 for dataframe in dataframes]
#%% Show the loaded files
fig, axes = plt.subplots(3,1)


[axes[ind].plot(voltagesArrays[ind], capacitanceArrays[ind],'.', label = names[ind]) for ind in range(len(axes))]
[axes[ind].legend(loc = "upper left") for ind in range(len(axes))]



#%% Now fit

result = [cvf.fitCapacitance_nBn(constants, voltagesArrays[ind], capacitanceArrays[ind], 130, 1, 1, 200, 4,\
          epsBarr = epsBL, epsAbs = epsAL, epsCon = epsAL, fitParasitic = False) for ind in range(len(axes))]



#%% Simulate the capacitance
bestFitCapacitancesAndDrops = [cvf.capacitance_nBnReturnPotential(constants,voltagesArrays[ind],\
                               130, *result[ind][2]) for ind in range(len(axes))]
#%% Show the fits
[axes[ind].plot(voltagesArrays[ind], bestFitCapacitancesAndDrops[ind][0], label = "fit") for ind in range(len(axes))]

#Show in the plot title the best fit results
[axes[ind].set_title(r"$N_{DC}$ = %0.2f $\times 10^{15} cm^{-3}$; $N_{DB}$ = %0.2f $\times 10^{15} cm^{-3}$; t$_{BL}$ = %0.0f nm; N$_{DAL}$ = %0.2f $\times 10^{15} cm^{-3}$"\
%(result[ind][0][0]*100,result[ind][0][1],result[ind][0][2],result[ind][0][3])) for ind in range(len(axes))]


[axes[ind].legend(loc = "upper left") for ind in range(len(axes))]




#%% CREATE DATA FOR FIGURE 6
#Perform a fit on an InAs/InAsSb superlattice

#easy way to read the file is to have it in the same path as this .py file
pathAndFile = "CVData.xlsx"

#device size
mesasize = 1000 #um
sheet = "GN2119G-S30"#Exel sheet
skiplines = 1 #header lines to skip

# header of the excel file
headerNames = ["Voltage (V)", "Capacitance (F)","Conductance (S)"]

#device parasitic capacitance
parasiticCap =  7.5

#temperature 
temperature = 80 # Kelvin


#include an initial guess on the parasitic capacitance
initialGuessCap = 7

#initial guess for the contact doping, barrier doping, barrier thickness, and absorber doping
initialGuess = [0.1,-2,200,4]


# The relative dielectric permittivity
epsAL =  15.3056
epsBL =  12.795
epsCL =  15.3056



# Read the data for this device
dataframe = pd.read_excel(pathAndFile, sheet_name = sheet, names = headerNames, skiprows = skiplines, usecols = "A:C") 

#have a forward and reverse bias cutoff
forwardBiasCutoff = 0.3 # volts
reverseBiasCutoff = -2.5 # volts

#Barrier thickness low
thickLow = 190 #nm

#Barrier thickness high
thickhigh = np.inf #nmf

#relative permittivity of barrier absorber and contact
epsbarlo = 10
epsbarhi = 13


epsAbslo = 15
epsAbshi = 16

epsConlo = 15
epsConhi = 16

#conduction band offset between contact and absorber
offsetUpper = 200
offsetLower = -100
"""
Currently fit the contact doping, barrier doping, barrier thickness,  absorber doping, conduction band offset,
parasitic capacitance, barrier epsilon, absorber epsilon, contact epsilon

NDCl -> 0
NDBL - NABL -> 1
tbl -> 2
NdAL -> 3
deltaEc -> 4
Cpar -> 5
epsBl -> 6
epsAL -> 7
epsCL -> 8
"""
inputParameters = ["NdCL","NdBL - NaAL", "tBL", "NdAL", "deltaEC", "Cpar", "epsBL","epsAL", "epsCL"]
units = ["x10^(17) cm^(-3)", "x10^(15) cm^(-3)", "nm", "x10^(15) cm^(-3)", "meV", "pF", "","", ""]

conFit = True
thickFit = True
parasiticFit = True
offsetFit = False
epsBLFit = False
epsALFit = False
epsCLFit = False

constants =  cvf.inputs()

#Set the parameter limits
parmBounds = [-np.inf, -np.inf, thickLow, -np.inf, offsetLower, 0, epsbarlo, epsAbslo, epsConlo],\
             [np.inf, np.inf, thickhigh, np.inf, offsetUpper, parasiticCap, epsbarhi, epsAbshi, epsConhi]

#Take the voltage sweep from the dataframe
voltage = dataframe[headerNames[0]]
    
#Take the capacitance from the dataframe
capacitance = dataframe[headerNames[1]]
forwardBiasCutoff = 0.3


#provide the logical arrays for the forwar and reverse bias cutoffs
greater = np.array(voltage) >= -2
lessThan = np.array(voltage)<forwardBiasCutoff

#Take the capacitance data and turn it to capacitance density
cap = capacitance/(mesasize*1e-6)**2*1e9/100**2


#take only the capacitance range that we will fit
indicesInterest = np.logical_and(greater, lessThan)
x = np.array(voltage[indicesInterest])
y = np.array(cap[indicesInterest])


sampleBestFit = cvf.fitCapacitance_nBn(constants, x, y, temperature,\
                         *initialGuess, 0, parasiticCap = initialGuessCap, epsBarr = epsBL,\
                         epsAbs = epsAL, epsCon = epsCL,fitConDope = conFit, fitParasitic = parasiticFit, fitOffset = offsetFit, \
                         fitThick = thickFit, fitEpsBarr = epsBLFit, fitEpsAbs = epsALFit, fitEpsCon = epsCLFit,\
                         mesasize = mesasize, parameterBounds = parmBounds)
    
print("Input parameters:")
for index, parameter in enumerate(sampleBestFit[2]):
    stringprint = r"%s: %0.3f %s"%(inputParameters[index], parameter, units[index])
    print(stringprint)
    
#%% Plot the results
lessThan2 = np.array(voltage)<1.3
indicesPlot = np.logical_and(greater, lessThan2)
voltagePlot = np.array(voltage)
capacitancePlot = np.array(cap)


capacitanceModel = cvf.determineCapacitance_nBn(constants, voltagePlot, temperature,\
                       *sampleBestFit[2], size = mesasize)
plt.figure()
plt.plot(voltagePlot, capacitanceModel,'b', label = "Model")
plt.plot(voltagePlot, capacitancePlot , 'r.',label = "Data")
plt.xlim([-2,1.3])
plt.legend(loc = "upper left")
plt.xlabel("Applied Voltage (V)")
plt.ylabel("Capacitance (nF/cm$^2$)")


# Plot the 1/C^2 inset
plt.figure()
plt.plot(voltagePlot, 1/capacitanceModel**2, 'b', label = "Model")
plt.plot(voltagePlot, 1/capacitancePlot**2, 'r.', label = "Data")
plt.xlim([-2, 0.3])
plt.legend(loc = "upper right")
plt.xlabel("Applied Voltage (V)")
plt.ylabel("1/C$^2$ (nF$^2$/cm$^4$)")
    
