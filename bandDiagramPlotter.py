# -*- coding: utf-8 -*-
"""
Created on Thu Jun  6 21:10:46 2024

@author: rigoc
"""


import CapacitanceVoltageFit as cvf
import numpy as np 
import matplotlib.pyplot as plt



devicenBn = cvf.device(0.13, 8, 100, 9.7, 0, 0, epsBar= 10)
voltageApplied  = np.array([-1, -1.3])
devTemp = 80
   
cvConst = cvf.inputs()

cap, vAl, phiBar, phiCon, vbi = cvf.capacitance_nBnReturnPotential(cvConst,\
    voltageApplied, devTemp, *devicenBn.deviceInfo)
wdAL = np.sqrt(2*cvConst.e0*(-vAl)*devicenBn.epsAbs/cvConst.e/(devicenBn.dopeAbsorber*1e15*100**3))

totalThickness = np.linspace(-wdAL[1]*1e9, 100, 10000)
potential = np.zeros(len(totalThickness))

absorberThickInd = totalThickness<=0

potential[absorberThickInd] = -cvConst.e*devicenBn.dopeAbsorber*1e15*100**3/(2*cvConst.e0*devicenBn.epsAbs) \
                *((totalThickness[absorberThickInd]+wdAL[1]*1e9)/1e9)**2

barrBoundary = cvf.chargeAbs(cvConst, devTemp, devicenBn.dopeAbsorber, vAl[1])/ \
                (cvConst.e0*devicenBn.epsBar)



## calculate the potential in the barrier
barrThickInd = totalThickness >= 0 


barrRegion = totalThickness[barrThickInd]

# barrConBoundary = cvf.chargeAbs(cvConst, devTemp, devicenBn.contactDoping*100, -phiCon[1], devicenBn.epsCon) /\
#     (cvConst.e0*devicenBn.epsBar) +cvConst.e*barrRegion[-1]*1e-9/(devicenBn.epsBar*cvConst.e0)*devicenBn.dopeBarrier*1e15*100**3


# potential[barrThickInd] = vAl[1]  + \
# barrConBoundary*barrRegion*1e-9 - \
# -cvConst.e*(barrRegion*1e-9)**2/(2*devicenBn.epsBar*cvConst.e0) * devicenBn.dopeBarrier*1e15*100**3

# plt.figure()
plt.plot(totalThickness, potential,'.')

#%%
madeUppotential = 2 *  cvf.chargeAbs(cvConst, devTemp, devicenBn.contactDoping*100, -phiCon[1], devicenBn.epsCon)/\
    (cvConst.e0*devicenBn.epsBar)/(100*1e-9) - phiBar[1]/(100*1e-9)**2