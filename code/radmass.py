
import numpy as np


datapath = "../resources/massradius"  # assumed relative to code within stellarprop ## replace with something smarter...


def posterior(massin, Mode='Frac',N=5000,Single=True):
    """
    TO Be filled in...
    
    """
    
    if Mode not in ['Frac','Line']:
        print("`Mode' must be either `Frac' or `Line', check inputs.")
        return None
    
    ## Load Resources ...
    trfile = {'Frac':datapath+"/fractional_01/chains.csv",
              'Line':datapath+"/linear_01/chains.csv"}[Mode]

    ## scatter modes taken from paper Pineda et al. 2021b
    scatter = {'Frac': 0.031,
                'Line': 0.0143}[Mode]

    trace = np.genfromtxt(trfile,delimiter=',',names=True) ## b, coef__1, scat

    b = trace['b']
    slp = trace['coef__1']
    numsamples = len(b)

    ## Consider inputs...
    inputshp = np.shape(massin)

    if N > numsamples:
        print("Warning: requested posterior samples per input mass exceeds length of mcmc in dataset, setting to max: {}".format(numsamples))
        N = numsamples
    


    if len(inputshp) == 0:
        mass = massin
        index = np.random.randint(0,numsamples,size=N)
        radii = b[index] + slp[index]*massin

        if Mode=='Frac':
            radii = radii + np.random.normal(0,1,size=N)*radii*scatter
        elif Mode == 'Line':
            radii = radii + np.random.normal(0,1,size=N)*scatter
        
    else:
        mass = np.array(massin).flatten()
        nm =len(mass)
        mass2d = np.transpose(np.tile(mass,(N,1)))

        index = np.random.randint(0,numsamples,size=N*nm)
        b2d = np.reshape(b[index],(nm,N))
        slp2d = np.reshape(slp[index],(nm,N))

        radii = b2d + slp2d*mass2d
    
        if Mode=='Frac':
            radii = radii + np.random.normal(0,1,size=(nm,N))*radii*scatter
        elif Mode == 'Line':
            radii = radii + np.random.normal(0,1,size=(nm,N))*scatter

        if Single:
            radii = radii.flatten()
        else:
            pass

    return radii




    
