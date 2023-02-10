# -*- coding: utf-8 -*-
"""
Created on Wed Dec  1 10:09:52 2021

@author: rigoc
"""
import numpy as np #numerical py, so exponents, sine, cosine etc. functions
from scipy import optimize #fitting and root finding function
# import matplotlib.pyplot as plt #plotting library

class inputs:
    def __init__(self):

        
        """initialize constants"""
        self.h    = 6.62607015e-34 # J.s
        self.hbar = self.h/(2*np.pi) # J.s
        self.e    = 1.602176634e-19 # Coulomb
        self.e0   = 8.854187812813e-12 # Farads/m
        self.m0   = 9.109383701528e-31  # kg
        self.c    = 299792458# m/s
        self.kb   = 1.380649e-23 # J/K
         
    


def chargeAbs(const: inputs,temp, doping, phi, epsin = 15.3):
    
    eps = epsin*const.e0
    dope = doping*1e15*100**3
    charge = -np.sign(phi) * \
         np.sqrt(2*const.kb*temp*dope*eps*(-1-const.e*phi/const.kb/temp + np.exp(const.e*phi/const.kb/temp)))
    return charge
    

def builtInVoltageAbsorberBarrier(const:inputs, temp, barrierDoping, absorberDoping, \
                                  barrierAbsorbervbo, absorberGap,\
                                  absorberme = 0.026, barriermv = 0.3):
    """Convert to units of m-3"""
    nDAL = absorberDoping * 1e15  * 100 ** 3 
    nABL = barrierDoping * 1e15 * 100 ** 3

    """Calculate the effective density of states in the conduction and valence band in units of
    m-3"""
    nCAbsorber = 1/4*(2*absorberme*inputs.m0*inputs.kb*temp/(np.pi*inputs.hbar))**(3/2)
    
    nVBarrier = 1/4*(2*barriermv*inputs.m0*inputs.kb*temp/(np.pi*inputs.hbar))**(3/2)
    
    
    vBi = barrierAbsorbervbo/1000 + inputs.kb*temp/inputs.e*np.log(nCAbsorber*nVBarrier/(nDAL*nABL))
    
    return vBi
    





def builtInVoltageAbsorberContact(const:inputs, temp, contactDoping, absorberDoping, bandOffset):
    """
    Calculate the built-in voltage between the absorber and contact by using just the absorber doping 
    and the contact doping in the device. This assumes that the absorber and contact have the same
    effective mass.

    Parameters
    ----------
    const : inputs object
           an object that contains physical constants in fundamental SI units
    temp : float
        The temperature of the device 
    contactDoping : float
        contact doping conentration in units of 1e15 cm-3
    absorberDoping : float
        absorber doping concentration in units of 1e15 cm-3
    bandOffset : float 
        conduction band offset between the absorber and contact
        (positive means contact conduction band is higher than absorber conduction band
         negative means contact conduction band is lower than absorber conduction band)
        
        
    Returns
    -------
    vBi : float
        built in potential of the device in units of volts (eV)
    """
    
    vBi = const.kb*temp/const.e*np.log(contactDoping/absorberDoping)  + bandOffset #(eV or Volts)
    
    
    
    return vBi



def centralDifference(y,x):
    """Calculate the derivative of data by the central difference, tying the endpoints
    with forward difference and back difference methods"""
    
    dydx = np.nan*np.ones(len(y))
    for index, yval in enumerate(y):
        if(index == 0):
            #Do a forward difference for first index
            dydx[index] = (yval - y[index + 1])/(x[index] - x[index + 1])
        elif (index == len(y)-1):
            #do a backward difference for last index
            dydx[index] = (yval - y[index - 1])/(x[index] - x[index - 1])
        else:
            forward = (y[index + 1] - yval)/(x[index + 1] - x[index])
            backward = (yval - y[index-1])/(x[index] - x[index-1])
            dydx[index] = 0.5*(forward + backward)
            
    return dydx




def determineCapacitance_nBn(const: inputs, vA, temp, dopeContact,  dopeBarrier, thicBarrier,\
                               dopeAbsorber, eC = 0, parasitic = 0, \
                               epsBar = 12.795, epsAbs = 15.3056, epsCon = 15.3056, size = 800):
    """
    Determine the capacitance of an nBn with arbitrary doping densities on the barrier (n or p-type) as 
    a function of applied voltage vA at a specified temperature temp. Inputs include doping in the Contact dopeContact,
    barrier dopeBarrier, barrier thickness thicBarrier and doping in the absorber dopeAbsorber. The method is an extension
    of Glasman's paper where the contact and absorber have similar electric field formalisms. A. Glasmann, I. Prigozhin, 
    and E. Bellotti, IEEE J. electron Dev. Soc. vol. 7, 534 (2019)
    Understanding the C-V characteristics of InAsSb-based nBn infrared detectors with n and p-type barrier layers through
    numerical modeling.  https://doi.org/10.1109/JEDS.2019.2913157

    Future paper title and authors: 
    Capacitance-voltage modeling of mid-wavelength barrier infrared detectors
    R. A. Carrasco, A. Newell, Z. M. Alsaad, J. V. Logan, G. Ariyawansa, C. P. Morath, and P. T. Webster

    Parameters
    ----------
    const: inputs object
           an object that contains physical constants in fundamental SI units
    vA : float Array size m x 1
        Applied voltage (V)
    temp : float
        temperature (K)
    eC : float    
        conduction band offset (meV, or millivolts)
    dopeContact : float
        contact doping (1e17 cm-3)
    dopeBarrier : float
        barrier doping (1e15 cm-3)
    thicBarrier : float
        barrier thickness (nm)
    dopeAbsorber : float
        absorber doping (1e15 cm-3)
    parasitic : float, optional, default 0
        parasitic capacitance of the device (pF)
    size : float, optional, default 800

    Returns
    -------
    capacitance : float Array size m x 1
        The calculated capacitance density of the device (F/m^2)

    """

    """Convert inputs to SI units"""
    
    nDAL = dopeAbsorber*1e15*100**3 #convert from 1e15 cm^(-3) to m^(-3)
    
    """For a p-type barrier, the barrier will be a negative value"""
    nDBL = dopeBarrier*1e15*100**3 #convert from 1e15 cm^(-3) to m^(-3) 
    
    
    nDCL = dopeContact*1e17*100**3 #convert from 1e17 cm^(-3) to m^(-3)
    
    thic = thicBarrier*1e-9 #convert from nm to m
    
  
    
    
    bandOffeV = eC/1000 # change the band offset from meV to eV
    
    """Determine the built-in voltage between the absorber and contact"""
    vbi = builtInVoltageAbsorberContact(const,temp, dopeContact*100, dopeAbsorber, bandOffeV)
    

    #initialize the absorber-barrier surface potential
    val = np.nan*np.ones([np.size(vA)])
    
    """Calculate the electric permittivities from the relative permittivities"""
    epsAL = epsAbs*const.e0
    epsBL = epsBar*const.e0
    epsCL = epsCon*const.e0
        

    
    
    """Define the electric field and charge in terms of doping and potential drops"""
    
    #Define an electric field at absorber barrier interface as a function of absorber potential
    eFieldAbs = lambda dope, phi, eps:-np.sign(phi) * \
        np.sqrt(2*const.kb*temp*dope/eps*(-1-const.e*phi/const.kb/temp + np.exp(const.e*phi/const.kb/temp)))
        
    #Calculate the charge on the contact as a function of contact potential  
    charge = lambda dope, phi, eps: -np.sign(phi) * \
         np.sqrt(2*const.kb*temp*dope*eps*(-1-const.e*phi/const.kb/temp + np.exp(const.e*phi/const.kb/temp)))
    

    
    """Determine potential drop in absorber at each applied voltage"""
    for index, vapplied in enumerate(vA):  
        
        # Select a wide enough range to determine the correct potential drop across the 
        # absorber
        negativeGuess = -np.sign(vapplied)*vapplied - 1
        positiveEndpoint = np.sign(vapplied)*vapplied + 1
        upperEndpoint = positiveEndpoint
        
        # Define a barrier potential based on the absorber potential
        phiBL = lambda phAL: -thic**2 * const.e *nDBL / (2*epsBL) - thic * epsAL/epsBL*eFieldAbs(nDAL,phAL, epsAL)
        
        # Define a contact potential in terms of the absorber potential
        phiCL = lambda apo: vapplied + vbi - phiBL(apo) - apo
            
        # Net charge that needs to be 0 at each applied voltage
        funcToSolve = lambda phiabs: charge(nDCL, -phiCL(phiabs), epsCL) + const.e*nDBL*thic + epsAL*eFieldAbs(nDAL,phiabs, epsAL)
            
        
        #Determine proper potential drop in the absorber for 0 net charge
        val[index] = optimize.brenth(funcToSolve, negativeGuess, upperEndpoint)
        

    # Determine the charge for the absorber at each applied voltage
    chargeAbs = charge(nDAL,val, epsAL)
    
    """Now determine the capacitance"""
    capacitance = -centralDifference(chargeAbs,vA) + parasitic*1e-12/(size*1e-6)**2
    capnFCm2 = capacitance*1e9/100**2
    return capnFCm2


def capacitance_nBnReturnPotential(const:inputs,vA, temp, dopeContact,  dopeBarrier, thicBarrier, \
                                     dopeAbsorber, eC=0, parasitic = 0, epsBar = 12.795, epsAbs = 15.3056,\
                                     epsCon = 15.3056, size = 800):
    """
    Determine the capacitance of an nBn with arbitrary doping densities on the barrier (n or p-type) as 
    a function of applied voltage vA at a specified temperature temp. Inputs include doping in the Contact dopeContact,
    barrier dopeBarrier, barrier thickness thicBarrier and doping in the absorber dopeAbsorber. The method is an extension
    of Glasman's paper where the contact and absorber have similar electric field formalisms:
    A. Glasmann, I. Prigozhin, and E. Bellotti, IEEE J. electron Dev. Soc. vol. 7, 534 (2019)
    Understanding the C-V characteristics of InAsSb-based nBn infrared detectors with n and p-type barrier layers through
    numerical modeling.  https://doi.org/10.1109/JEDS.2019.2913157
    
    
    Future paper title and authors: 
    Capacitance-voltage modeling of mid-wavelength barrier infrared detectors
    R. A. Carrasco, A. Newell, Z. M. Alsaad, J. V. Logan, G. Ariyawansa, C. P. Morath, and P. T. Webster
    

    Parameters
    ----------
    vA : float Array size m x 1
        Applied voltage (V)
    temp : float
        temperature (K)
    dopeContact : float
        contact doping (1e17 cm-3)
    dopeBarrier : float
        barrier doping (1e15 cm-3)
    thicBarrier : float
        barrier thickness (nm)
    dopeAbsorber : float
        absorber doping (1e15 cm-3)
    eC : float    
        conduction band offset (meV, or millivolts)
    parasitic : float, optional, default 0
        parasitic capacitance of the device (pF)
    size : float, optional, default 800
        side length of the device (um)

    Returns
    -------
    capacitance : float Array size m x 1
        The calculated capacitance density of the device (nF/cm^2)
    val:   float Array size m x 1
        The calculated absorber potential drop of the device (V)
    phiBarrier:     float Array size m x1
        The calculated barrier potential drop of the device (V)
    phiContact:     float Array size m x1
        the calculated contact potential drop of the device (V)
    vBi:    float single
        The calculated built-in potential between the contact and absorber
    """


    """Convert to SI units"""
    nDAL = dopeAbsorber*1e15*100**3 #convert from 1e15 cm^(-3) to m^(-3)
    nDBL = dopeBarrier*1e15*100**3 #convert from 1e15 cm^(-3) to m^(-3) (Assuming a p-type barrier)
    
    # for ptype, the barrier will be a negative value
    nDCL = dopeContact*1e17*100**3 #convert from 1e17 cm^(-3) to m^(-3)
    
    
    
    bandOffeV = eC/1000 # change the band offset from meV to eV
    
    vbi = builtInVoltageAbsorberContact(const, temp, dopeContact*100, dopeAbsorber, bandOffeV)
    # print("vbi is",vbi)
    
    thic = thicBarrier*1e-9 #convert from nm to m
    
  
    #initialize the absorber-barrier surface potential
    val = np.nan*np.ones([np.size(vA)])


 
    epsAL = epsAbs*const.e0
    epsBL = epsBar*const.e0
    epsCL = epsCon*const.e0
        

        
        
    """Define electric field and charge in terms of doping and potential drops"""
    #Define an electric field at absorber barrier interface as a function of absorber potential
    eFieldAbs = lambda dope, phi, eps:-np.sign(phi) * \
        np.sqrt(2*const.kb*temp*dope/eps*(-1-const.e*phi/const.kb/temp + np.exp(const.e*phi/const.kb/temp)))
        
    #Calculate a charge on the contact as a function of contact potential  
    charge = lambda dope, phi, eps: -np.sign(phi) * \
         np.sqrt(2*const.kb*temp*dope*eps*(-1-const.e*phi/const.kb/temp + np.exp(const.e*phi/const.kb/temp)))
    

    
    """Determine potential drop in absorber at each applied voltage"""
    for index, vapplied in enumerate(vA):  
        
        # Select a wide enough range to determine the correct potential drop across the 
        # absorber
        negativeGuess = -np.sign(vapplied)*vapplied - 1
        positiveEndpoint = np.sign(vapplied)*vapplied + 1
        upperEndpoint = positiveEndpoint
        
        # Define a barrier potential based on the absorber potential
        phiBL = lambda phAL: -thic**2 * const.e *nDBL / (2*epsBL) - thic * epsAL/epsBL*eFieldAbs(nDAL,phAL, epsAL)
        
        # Define a contact potential in terms of the absorber potential
        phiCL = lambda apo: vapplied + vbi - phiBL(apo) - apo
            
        #Net charge that needs to be 0 at each applied voltage
        funcToSolve = lambda phiabs: charge(nDCL, -phiCL(phiabs), epsCL) + const.e*nDBL*thic + epsAL*eFieldAbs(nDAL,phiabs, epsAL)
            
        
        #Determine proper potential drop in the absorber for 0 net charge
        val[index] = optimize.brenth(funcToSolve, negativeGuess, upperEndpoint)
        

    # Determine the charge for the absorber at each applied voltage
    chargeAbs = charge(nDAL,val, epsAL)
    phiBarrier = -thic**2 * const.e *nDBL / (2*epsBL) - thic * epsAL/epsBL*eFieldAbs(nDAL,val, epsAL)
    phiContact = vA + vbi - phiBarrier - val
    
    #Differentiate the charge in the absorber and add any parasitic capacitance 
    capacitance = -centralDifference(chargeAbs,vA) + parasitic*1e-12/(size*1e-6)**2
    
    #Put the parasitic capacitance in units of nanofarads/cm^2
    capnFCm2 = capacitance*1e9/100**2

    
    return capnFCm2, val, phiBarrier, phiContact, vbi


def fitCapacitance_nBn(const:inputs, xdata, ydata, temp, dopeContact, dopeBarrier, thicBarrier, dopeAbsorber,\
                         eC = 0, parasiticCap = 0, epsBarr = 12.795, epsAbs = 15.3056, epsCon = 15.3056, \
                         fitConDope = True, fitOffset = False, fitParasitic = True, \
                         fitThick = True, fitEpsBarr = False, fitEpsAbs = False, fitEpsCon = False, \
                         mesasize = 800, parameterBounds = (- np.inf, np.inf)):
    """
    Fit a provided capacitance (ydata) at given voltages (xdata) and temperature (temp) 
    with initial guesses in the contact doping (dopeContact), barrier doping (dopeBarrier),
    thickness in the barrier (thicBarrier) and doping in the absorber (dopeAbsorber)

    Parameters
    ----------
    xdata : float Array size m x 1
        Applied voltage (V)
    ydata : float Array size m x 1
        Capacitance density in nF/cm^2
    temp :  float
        temperature (K)
    dopeContact : float
        contact doping (1e17 cm-3)
    dopeBarrier : float
        barrier doping (1e15 cm-3)
    thicBarrier : float
        barrier thickness (nm)
    dopeAbsorber : float
        absorber doping (1e15 cm-3)
    eC : float (default: 0)
        conduction band offset in meV (millivolts)
    parasiticCap : float (default: 0)
        parasitic capacitnace in pF
    fitParasiticCap : boolean (default: True)
        does the user want to fit the parasitic capacitance?
    fitOffset : boolean (default: False)
        does the user want to fit the conduction band offset
    fitThick : float (default: True)
        does the user want to fit the thickness?
    mesasize: float (default: 800)
        sidelength of the mesa (in um)
    parameterbounds: tuple (default: (-inf, inf))
        the fit parameter bounds

    Returns
    -------
    bestFit : TYPE
        DESCRIPTION.
    covariance : TYPE
        DESCRIPTION.

    """
    
    # First assume we are fitting the  contact doping, barrier doping, barrier thickness, absorber doping, 
    # band offset, and parasitic capacitance
    initialGuess = [dopeContact, dopeBarrier, thicBarrier, dopeAbsorber, eC, parasiticCap, \
                    epsBarr, epsAbs, epsCon]
    # booleanChoices = [fitThick, fitOffset, fitParasitic, fitEpsBarr, fitEpsAbs]
        
    finalOutput = np.nan*np.ones(len(initialGuess))
    
    
    printString = "Not fitting:"
    stringLambda = "lambda x" # initialize the anonymous function 
    stringFunction = "determineCapacitance_nBn(const, x, temp"
    dictionaryVariables = {'const': const,\
                           'temp': temp,\
                           'determineCapacitance_nBn':determineCapacitance_nBn,\
                            }
    
    
    
    
    """
    Go through each fit parameter and create a function handle that fits
    appropriate desired device parameters
    """
    
    
    """Are we fitting the contact doping?"""
    if(not(fitConDope)):
        """No? add the contact doping as a determined value and not 
        a variable parameter"""
        
        stringFunction = stringFunction + ",dopeContact" #add as a determined value
        
        printString = printString + " contact doping" #add as a value that is not being fitted
        
        finalOutput[0] = dopeContact #add the contact doping to the list of final outputs to create the final model
        
        dictionaryVariables['dopeContact'] = dopeContact #include the contact doping to the dictionary of strings

    else:
        """We are fitting the contact doping, so add as a variable parameter"""
        
        stringLambda = stringLambda + ",dc" #Add as a variable parameter in the anonymous function
        
        stringFunction = stringFunction + ",dc"
        
    
    
    """"Always fit barrier doping"""
    stringLambda = stringLambda + ",db"
    stringFunction = stringFunction + ",db" 
    
    
    
    """Fix the  thickness"""
    if(not(fitThick)):
        stringFunction = stringFunction + ",thicBarrier"
        printString = printString + " barrier thickness"
        dictionaryVariables['thicBarrier'] = thicBarrier
        finalOutput[2] = thicBarrier

    else:
        """Fit the barrier thickness"""
        stringLambda = stringLambda +",tb"
        stringFunction = stringFunction + ",tb"
        
    #we are now at the absorber which we will always fit
    stringLambda = stringLambda +",da"
    stringFunction = stringFunction +",da"
    
    #Fix the band offset
    if(not(fitOffset)):
        stringFunction = stringFunction + ",eC"
        printString = printString + " conduction band offset"
        finalOutput[4] = eC
        dictionaryVariables['eC'] = eC
    else:
        stringLambda = stringLambda + ",bOff"
        stringFunction = stringFunction +",bOff"
        
    #Fix the parasitic capacitance
    if(not(fitParasitic)):
        stringFunction = stringFunction + ", parasiticCap"
        printString = printString + " parasitic capacitance"
        finalOutput[5] = parasiticCap
        dictionaryVariables['parasiticCap'] = parasiticCap
        
    else:
        stringLambda = stringLambda + ",pc"
        stringFunction = stringFunction + ",pc"
    
    #Fix the epsilon barrier
    if(not(fitEpsBarr)):
        stringFunction = stringFunction + ",epsBarr"
        printString = printString + "; barrier epsilon"
        finalOutput[6] = epsBarr
        dictionaryVariables['epsBarr'] = epsBarr
    else:
        stringLambda = stringLambda + ",epbarr"
        stringFunction = stringFunction + ",epbarr"
        
    #Fix the epsilon absorber
    if(not(fitEpsAbs)):
        stringFunction = stringFunction + ",epsAbs"
        printString = printString + "; absorber epsilon"
        finalOutput[7] = epsAbs
        dictionaryVariables['epsAbs'] = epsAbs
      
    #Fit the absorber epsilon
    else:
        stringLambda = stringLambda + ", epabs"
        stringFunction = stringFunction + ",epabs"
        
    #Fix the epsilon contact
    if(not(fitEpsCon)):
        stringFunction = stringFunction + ", epsCon, mesasize)"      
        stringLambda = stringLambda + ":"
        printString = printString + "; contact epsilon"
        finalOutput[8] = epsCon
        dictionaryVariables["epsCon"] = epsCon 
    else:
        stringLambda = stringLambda + ",epCon:"
        stringFunction = stringFunction + ", epCon, mesasize)"

    dictionaryVariables['mesasize'] = mesasize 
    
    """now delete the initial Guesses that are fixed"""
   
    
    if(not(fitEpsCon)):
       initialGuess.pop(8)
       if(np.size(parameterBounds[0]) > 1):
           parameterBounds[0].pop(8)
           parameterBounds[1].pop(8)
             
    if(not(fitEpsAbs)):
        initialGuess.pop(7)  
        if(np.size(parameterBounds[0]) > 1):
            parameterBounds[0].pop(7)
            parameterBounds[1].pop(7)

    if(not(fitEpsBarr)):
        initialGuess.pop(6)
        if(np.size(parameterBounds[0]) > 1):
            parameterBounds[0].pop(6)
            parameterBounds[1].pop(6)
    
    if(not(fitParasitic)):
        initialGuess.pop(5)
        if(np.size(parameterBounds[0]) > 1):
            parameterBounds[0].pop(5)
            parameterBounds[1].pop(5)
    
    if(not(fitOffset)):
        initialGuess.pop(4)
        if(np.size(parameterBounds[0]) > 1):
            parameterBounds[0].pop(4)
            parameterBounds[1].pop(4)
    
    if(not(fitThick)):
        initialGuess.pop(2)
        if(np.size(parameterBounds[0]) > 1):
            parameterBounds[0].pop(2)
            parameterBounds[1].pop(2)
    
    
    if(not(fitConDope)):
        initialGuess.pop(0)
        if(np.size(parameterBounds[0]) > 1):
            parameterBounds[0].pop(0)
            parameterBounds[1].pop(0)
        
    
    
    """Print what is not being fitted"""
    # print(printString)
    
    """Print the function that will be fit"""
    # print(stringLambda + stringFunction)
    
    # print(parameterBounds)
    """The model that will be fed into the curve fit"""
    model = eval(stringLambda + stringFunction, dictionaryVariables)
    bestFit, covariance = optimize.curve_fit(model, xdata, ydata, initialGuess, bounds = parameterBounds)
    
    """SetTheOutputs"""
    fitIndex = 0

    """Create an array where all of the inputs are provided, including conduction band offset and parasitic capacitance"""
    for index, value in enumerate(finalOutput):
        if(np.isnan(value)):
            finalOutput[index] = bestFit[fitIndex]
            fitIndex = fitIndex + 1
            
    """provide the best fit, covariance, and all of the model parameter inputs to determine the capacitance"""
    return bestFit, covariance, finalOutput

    
    
    


def calculateAlGaAsSbBandGap(comp, temp):
  """
  calculate the bandgap of quaternary AlGaAsSb lattice matched to GaSb at a given composition
  and temperature using Vurgaftman's paper

  Parameters
  ----------
  composition : float
  the aluminum compostion of the barrier and it will assume a lattice match to GaSb 
  temp : float
  temperature to shift the bandgap, it will assume an AlAsSb temperature dependent bandgap
  OR we can do AlAsSb (GaSb) band gap shift.....

  Returns
  -------
  AlGaAsSbBandGap : TYPE
  DESCRIPTION.

  """
  """Temperature dependent bandgaps of the binary consituents from
  Vurgaftman et al. JAP vol. 89, p. 5815 (2001)"""
  GaSbBandGap = 0.812 - 0.417/1000*temp**2/(temp + 140) #in eV
  AlAsBandGap = 3.099 - 0.885/1000*temp**2/(temp + 530) #in eV
  AlSbBandGap = 2.386 - 0.420/1000*temp**2/(temp + 140) #in eV
  
  
  # Lattice match AlAsSb to GaSb
  y = 0.92
  
  # Lattice match AlAsSb to GaSb with a bowing
  AlAsSbBandGap = (1 - y)*AlAsBandGap + y*AlSbBandGap - 0.8*y*(1 - y) # eV
  
      
  AlGaAsSbBandGap = GaSbBandGap*(1-comp) + comp*AlAsSbBandGap - 0.48*comp*(1-comp) #eV
  
  
  return AlGaAsSbBandGap
    
    
        