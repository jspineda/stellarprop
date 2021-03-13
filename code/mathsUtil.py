
import numpy as np

from scipy.stats.mstats import mquantiles
from scipy.optimize import fsolve
import scipy.stats as STS



def confidenceInterval(values,interval=0.682689,ThSig=False):
    "For determining the median and central confidence interval of input values array"
    if ThSig:
        interval = 0.997300203937 # 3 sigma
    qL = (1. - interval)/2.
    qU = 1. - qL
    out = mquantiles(values,prob=[qL,0.5,qU])
    return out




def estSkewNorm(xin,conf=(0.16,0.5,0.84),Guess=None,Mode='Med',Check=True):
    """
    Used for estimating the parameters of a Skew Normal distribution that matches input mode/median and given confidence interval.
    
    Numerically solves system of equations to give the 3 output parameters of corresponding skew normal. May require multiple guess to result in right solution. Output should give practically exact values, see `Check'.
    
    Returns the skew normal parameters (mu, sigma, alpha) --- see wikipedia on skew normal
    
    WARNING: Code is provided with minimal explanation, and verfication --- use with that in mind.
    
    >---
    
    Parameters
    ----------
    xin : array_like, tuple, or list, with 3 entries
        entries correspond to the data value matching the corresponding index in the `conf' key word. By default first entry indicates the lower bound for the central 68% confidence interval, second entry corresponds to median/mode value depending on keyword `Mode', and the third entry is the upper bound for the central 68% confidence interval.
        
    conf : array_like, tuple, or list, with 3 entries
        indicates the values that the skew normal cumulative probability distribution should give with input `xin'. By default, set to median and central 68% confidence interval. If Mode is `Peak' the median equation is replaced by one corresponding to peak of distribution.
    
    Guess : array_like, tuple or list, with 3 entries
        allows for user to input starting point for numerical equation solve. Default values are a general guess. If output does not converge, use Guess to change starting point for computation. May require iteration for adequete solution. Use `Check' to verify. If there is difficult, input parameters may not be well suited for approximation with skew normal.
    
    Mode : str one of ['Peak','Med2','Med','SF']
        Defines to set of equations used in defining the skew normal distribution. If 'Peak' system sets second entry to be the mode of skew normal instead of median. All others are for setting the median, but with slightly different numerical implementations. 'Peak' and 'Med' are the recommended modes.
        
    Check : boolean
        By default it is True. Used as verification on output solution, gives printed diagnostics as check. Outputs for converged solutions should be exact if fully successful.
    
    
    Returns
    -------
    
    out : array_like with 3 entries
        gives the (mu, sigma, alpha) parameters that define the skew normal distribution matching inputs
    
    Notes
    -----
    Printed warnings also given from scipy from fsolve to diagnose progress of numerical solution to system of equations
    
    Examples
    --------
    
    ## Note that here we use data from https://github.com/jspineda/stellarprop for illustration ; see also Pineda et al. 2021b
    
    >> trace = np.genfromtxt('../resources/massradius/fractional_01/chains.csv',delimiter=',',names=True)
    >> scatlb, scatmid, scatub = confidenceInterval(trace['scatter'])  # the scatter psoterior distribution is asymetric, these are typically the reported values in literature
    >> print([scatlb,scatmid,scatub])
    [0.02787424918238516, 0.0320051813038165, 0.03692976181631807]
    >> params = estSkewNorm( [scatlb, scatmid, scatub])
    Mode at [0.03121118]
    Median at 0.032005181304171265
    Result gives centeral 68.0% confidence interval: (0.027874249182851436, 0.03692976181636316)
    Peak at [0.03121118] - [0.00333693]  +  [0.00571858]
    >> print(params)
    [0.02771848 0.0065575  1.95731243]
    
    ## Note that Check outputs reported numerically match nearly exactly to inputs, these would be kinda off if iteration needed
    ## In this example alpha ~2, indicating positive skewness, peak (mode) is at 0.031, a little less than median at 0.032   -- see appendix of Pineda et al. 2021b
    
    
    """
    
    xl, x0, xu = xin
    cl,c0,cu = conf
    if Guess is not None:
        p0 = Guess
    else:
        p0 = (x0, (xu-xl)/2., ((xu-x0) - (x0-xl))/ ((xu-xl)/2.)  )
    
    ## if block used to toggle set of equations to solve using scipy fsolve
    if Mode=='Peak':
        print("Setting Peak of Distribution")
        def eq_sys(p):
            mu,sigma,alpha = p
            t = (x0 - mu)/sigma
            return STS.skewnorm.cdf(xl,alpha,mu,sigma) - conf[0],STS.skewnorm.cdf(xu,alpha,mu,sigma) - conf[2],alpha*STS.norm.pdf(alpha*t) - STS.norm.cdf(alpha*t)*t
    elif Mode=='Med2':
        print("Setting Median of Distribution")
        def eq_sys(p):
            mu,sigma,alpha = p
            return np.power(STS.skewnorm.cdf(xin,alpha,mu,sigma) - np.array(conf),2)
    elif Mode == 'SF':
        print("Setting Median of Distribution")
        def eq_sys(p):
            mu,sigma,alpha = p
            return STS.skewnorm.isf(1-np.array(conf),alpha,mu,sigma) - np.array(xin)
    elif Mode == 'Med':
        print("Setting Median of Distribution")
        def eq_sys(p):
            mu,sigma,alpha = p
            return (STS.skewnorm.cdf(xl,alpha,mu,sigma)-cl,STS.skewnorm.cdf(x0,alpha,mu,sigma)-c0,STS.skewnorm.cdf(xu,alpha,mu,sigma)-cu)

    out = fsolve(eq_sys,p0)

    if Check:
        ff = lambda a: STS.norm.pdf(out[2]*a)*out[2] - a*STS.norm.cdf(a*out[2])
        tm = fsolve(ff,0.2*out[2])
        xm = tm*out[1] + out[0]
        print("Mode at {}".format(xm))
        print("Median at {}".format(STS.skewnorm.median(out[2],out[0],out[1])))
        print("Result gives centeral {0}% confidence interval:".format((conf[2]-conf[0])*100),STS.skewnorm.interval(conf[2]-conf[0],out[2],out[0],out[1]))
        print("Peak at {0} - {1}  +  {2} ".format(xm, xm - xl, xu  - xm))

    return out  # out is mu, sigma, alpha

