#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on Sun Dec  3 12:55:07 2022

@author: mbeasley
Code snippet to determine the slope and zeropoint, or zeropoint only and 
distance based on observed magnitudes and velocity dispersions of globular clusters.
See:  Beasley, Fahrion & Gvozdenko MNRAS, 2023 ; 10.1093/mnras/stad3541

Inputs :

x = log10 of gc velocity dispersion
y = V-band magnitude
x_err = uncertainty on x (define as : np.log10(x) * (xe / x))
y_err uncertainty on y

"""
import numpy as np 
import emcee
import matplotlib.pyplot as plt
import corner 

# Define the log-likelihood functions.
# log_likelihood leaves the the slope and zeropoint of GCVD as free parameters.
# log_likelihood2 fixes the slope to that of the fiducial relation.

def log_likelihood(theta, x, y, x_err, y_err):
    slope, intercept = theta
    model = slope * x + intercept
    sigma = np.sqrt(y_err**2 + (slope * x_err)**2)
    return -0.5 * np.sum((y - model)**2 / sigma**2 + np.log(sigma**2))


def log_likelihood2(theta, x, y, x_err, y_err):
    slope, intercept = theta
    model = -4.73 * x + intercept #fix slope to fiducial
    sigma = np.sqrt(y_err**2 + (slope * x_err)**2)
    return -0.5 * np.sum((y - model)**2 / sigma**2 + np.log(sigma**2))

def log_prior(theta):
    slope, intercept = theta
    if -100.0 < slope < 100.0 and -30 < intercept < 30.0:
        return 0.0
    return -np.inf

# Define the log-posterior function
def log_posterior(theta, x, y, x_err, y_err):
    lp = log_prior(theta)
    if not np.isfinite(lp):
        return -np.inf  
# ******************************************************************************************************************
    return lp + log_likelihood(theta, x, y, x_err, y_err) # fits for slope + zeropoint
    # return lp + log_likelihood2(theta, x, y, x_err, y_err) # fits for zeropoint only
# ******************************************************************************************************************

# Set up the initial conditions for the walkers

n_walkers = 25
ndim = 2
pos = np.random.normal([true_slope, true_intercept], [0.5, 5.0], size=(n_walkers, ndim))

# Set up the emcee sampler
sampler = emcee.EnsembleSampler(n_walkers, ndim, log_posterior, args=(x, y, x_err, y_err))

# Run the MCMC sampler
n_steps = 10000
sampler.run_mcmc(pos, n_steps, progress=True)

# Get the chain of samples and discard the burn-   # mM = intercept[1] - -4.53  #MWin period
burn_in = 100
samples = sampler.chain[:, burn_in:, :].reshape((-1, ndim))
flat_samples = sampler.get_chain(discard=100, thin=1, flat=True)

#  Extract parameter estimates and uncertainties

slope = np.percentile(flat_samples[:,0],[16,50,84]) # 1 SIGMA
slope_err =np.diff(np.percentile(flat_samples[:,0],[16,50,84]))
intercept= np.percentile(flat_samples[:,1],[16,50,84])
intercept_err=np.diff(np.percentile(flat_samples[:,1],[16,50,84]))


# print out parameters and percentiles
print()
print(" intercept beta0 : {0:.3f} + {1:.3f} - {2:.3f} ".format(intercept[1], intercept_err[0], intercept_err[1]))
print(" slope beta1 : {0:.3f} + {1:.3f} - {2:.3f} ".format(slope[1], slope_err[0], slope_err[1]))
 

# Here we calcuate the distance modulus and uncertainty based on the fiducial relation
   
fiducial_err = 0.064 
mM = intercept[1] - -4.49

mMe = np.max(intercept_err)
mMtot= np.sqrt(mMe**2+fiducial_err**2)

D = 10**(mM / 5) / 1e5
D_max = 10**((mM + mMe)/ 5) / 1e5
DD_max = 10**((mM + mMtot)/ 5) / 1e5

De = D_max - D
Dtot = DD_max -D

print ()
print (' m-M : {0:.3f} ±  {1:.3f} {2:.3f} '.format(mM, mMe, mMtot))
print (' D : {0:.3f} ±  {1:.3f} {2:.3f} '.format(D, De, Dtot))

# Plot the regression line and data

plt.figure(99)
plt.errorbar(x, y, xerr=x_err, yerr=y_err, fmt='.', label='Data')

plt.plot(x, slope[1] * x + intercept[1], 'g', label='Estimated regression line')
plt.plot(x, slope[0] * x + intercept[2], 'g')
plt.plot(x, slope[2] * x + intercept[0], 'g')
# plt.legend()
plt.ylim(-2,-12)
plt.xlabel('X')
plt.ylabel('Y')
plt.show()

# show corner plots for 2-parameter case

figure = corner.corner(
samples,
labels=[
    r"$Slope~\beta_1$",
    r"$Intercept~\beta_0$",
],
quantiles=[0.16, 0.5, 0.84],
show_titles=True,
title_kwargs={"fontsize": 12},
)
