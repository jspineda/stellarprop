
import numpy as np


datapath = "../resources/massradius"  # assumed relative to code within stellarprop ## replace with something smarter?...


def posterior(massin, Mode='Frac',N=5000,Single=True):
    """
    Used for determining radius from mass for low-mass stars on 0.09-0.70 Msun following Pineda et al. 2021b, includes correlated uncertainties on fit coefficients plus scatter term.
    
    >----
    
    Parameters
    ----------
    
    massin: array-like or float
        Input stellar mass (solar units) for which you desire an estimate radius distribution
    
    Mode: str - either 'Frac' or 'Const'
        Mode sets the relationship to use for the mass-radius relation, either fractional scatter 'Frac' (default) or constant scatter 'Const' , entry must match either of those two strings
        
    N: int default to 5000
        N Sets the length of sampling per input mass, so if massin is a single value, the function returns an array of length N that provides the right posterior distribution on the radius accounting for the uncertainty in the relation. Should not be a small number to average out stochastic nature of uncertainty distributions. For input arrays of length M, the output will be of total length M times N. This allows for the user to input a mass distribution of length M, then the routine, for each entry of massin, samples the relation N times to produce the output array of total length M * N. For large values of M, decrease N to maintain manageable arrays.
    
    Single: boolean default to true
        Option to make sure to flatten the output radius distribution. Otherwise the routine will take array-like inputs and create and M x N array where each row corresponds to the radius distribution of each entry mass, where M is the number of elements in massin. Setting Single=True flattens that array. Useful for when getting radii distribution for multple distinct masses.
    
    Returns
    -------
    
    radii: array_like
        radius distribution for input mass(es)


    Examples
    -------
    
    >> import radmass
    
    ## get radii for single mass
    >> r1 = radmass.posterior(0.4,N=2000)
    >> r1
    array([0.41178628, 0.41333893, 0.38339227, ..., 0.39793819, 0.41142308,
       0.41203498])
    >> r1.shape
    (2000,)

    ## get radii for mass distribution 0.3 +- 0.01
    >> mass = np.random.normal(0,1,size=100)*0.01 + 0.3
    >> r2 = radmass.posterior(mass,N=200)
    >> r2
    array([0.31035685, 0.29291541, 0.3198167 , ..., 0.32789612, 0.32924407,
       0.31066338])
    >> r2.shape
    (20000,)
    
    
    ## get radii distribution for 5 different masses
    >> r3 = radmass.posterior([0.1,0.2,0.4,0.5,0.6],N=1000,Single=False)
    >> r3.shape
    (5,1000)

    ## distribution for mass entry 0 in row 0 etc.



    """
    
    if Mode not in ['Frac','Const']:
        print("`Mode' must be either `Frac' or `Const', check inputs.")
        return None
    
    ## Load Resources ...
    trfile = {'Frac':datapath+"/fractional_01/chains.csv",
              'Const':datapath+"/const_01/chains.csv"}[Mode]

    ## scatter modes taken from paper Pineda et al. 2021b
    scatter = {'Frac': 0.031,
                'Const': 0.0143}[Mode]

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
        elif Mode == 'Const':
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
        elif Mode == 'Const':
            radii = radii + np.random.normal(0,1,size=(nm,N))*scatter

        if Single:
            radii = radii.flatten()
        else:
            pass

    return radii




    
