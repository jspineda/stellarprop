## Notes on usage of MCMC output chains and files

The fractional and linear folders here contain the mcmc samples for the fractional and linear scatter regression fits to the mass-radius data as discussed in Pineda et al. 2021b. The chains files are comma separated and contain as columns the three variables of interest: 

b - the linear offset
coef__1 - the slope
scatter - the scatter on the relationship either relative (`fractional') or absolute ('linear')


Therefore in the linear model the scatter is in units of solar radii, in fractional the scatter is dimensionless and relative to the predicted radius at fixed mass.

the 'stats' file in each folder records the median (first row), lower 68% confidence bound (second row), and upper 68% confidence bound (third row) for each variable.
