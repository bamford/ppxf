#!/usr/bin/env python
#----------------------------------------------------------------------------
#
# Monte Carlo simulation of the kinematic extraction with pPXF. It is useful
# to determine the desired value for the BIAS keyword of the pPXF procedure.
# This procedure generates a plot similar (but not identical) to Figure 6 in
# Cappellari & Emsellem, 2004, PASP, 116, 138.
#
# A rough guideline to determine the BIAS value is the following: choose the *largest*
# value which make sure that in the range sigma>3*velScale and for (S/N)>30 the true values
# for the Gauss-Hermite parameters are well within the rms scatter of the measured values.
# See the documentation in the file ppxf.pro for a more accurate description.
#
# V1.0.0: By Michele Cappellari, Leiden, 28 March 2003
# V1.1.0: Included in the standard PPXF distribution. After feedback
#   from Alejandro Garcia Bedregal. MC, Leiden, 13 April 2005
# V1.1.1: Adjust GOODPIXELS according to the size of the convolution kernel.
#   MC, Oxford, 13 April 2010
# V1.1.2: Use Coyote Graphics (http://www.idlcoyote.com/) by David W. Fanning.
#   The required routines are now included in NASA IDL Astronomy Library.
#   MC, Oxford, 29 July 2011
# V2.0.0: Translated from IDL into Python. MC, Oxford, 9 December 2013
# V2.0.1: Support both Python 2.6/2.7 and Python 3.x. MC, Oxford, 25 May 2014
# V2.0.2: Support both Pyfits and Astropy to read FITS files.
#   MC, Oxford, 22 October 2015
#
##############################################################################

from __future__ import print_function

try:
    import pyfits
except:
    from astropy.io import fits as pyfits
from scipy import ndimage, signal
import numpy as np
import matplotlib.pyplot as plt
from time import clock

from ppxf import ppxf
import ppxf_util as util

#----------------------------------------------------------------------------

def rebin(x,factor):
    """
    Rebin a one-dimensional vector by averaging
    in groups of "factor" adjacent values

    """
    return np.mean(x.reshape(-1,factor),axis=1)

#----------------------------------------------------------------------------

def ppxf_simulation_example():

    dir = 'spectra/'
    file = dir + 'Rbi1.30z+0.00t12.59.fits'
    hdu = pyfits.open(file)
    ssp = hdu[0].data
    h = hdu[0].header

    lamRange = h['CRVAL1'] + np.array([0.,h['CDELT1']*(h['NAXIS1']-1)])
    star, logLam, velscale = util.log_rebin(lamRange, ssp)

    # The finite sampling of the observed spectrum is modeled in detail:
    # the galaxy spectrum is obtained by oversampling the actual observed spectrum
    # to a high resolution. This represent the true spectrum, which is later resampled
    # to lower resolution to simulate the observations on the CCD. Similarly, the
    # convolution with a well-sampled LOSVD is done on the high-resolution spectrum,
    # and later resampled to the observed resolution before fitting with PPXF.

    factor = 10                    # Oversampling integer factor for an accurate convolution
    starNew = ndimage.interpolation.zoom(star,factor,order=1) # This is the underlying spectrum, known at high resolution
    star = rebin(starNew,factor)        # Make sure that the observed spectrum is the integral over the pixels

    vel = 0.3      # velocity in *pixels* [=V(km/s)/velScale]
    h3 = 0.1       # Adopted G-H parameters of the LOSVD
    h4 = -0.1
    sn = 60.        # Adopted S/N of the Monte Carlo simulation
    m = 300        # Number of realizations of the simulation
    sigmaV = np.linspace(0.8,4,m) # Range of sigma in *pixels* [=sigma(km/s)/velScale]

    result = np.zeros((m,4)) # This will store the results
    t = clock()
    np.random.seed(123) # for reproducible results

    for j in range(m):

        sigma = sigmaV[j]
        dx = int(abs(vel)+4.0*sigma)   # Sample the Gaussian and GH at least to vel+4*sigma
        x = np.linspace(-dx,dx,2*dx*factor+1) # Evaluate the Gaussian using steps of 1/factor pixels.
        w = (x - vel)/sigma
        w2 = w**2
        gauss = np.exp(-0.5*w2)/(np.sqrt(2.*np.pi)*sigma*factor) # Normalized total(gauss)=1
        h3poly = w*(2.*w2 - 3.)/np.sqrt(3.)           # H3(y)
        h4poly = (w2*(4.*w2 - 12.) + 3.)/np.sqrt(24.) # H4(y)
        losvd = gauss *(1. + h3*h3poly + h4*h4poly)

        galaxy = signal.fftconvolve(starNew,losvd,mode="same") # Convolve the oversampled spectrum
        galaxy = rebin(galaxy,factor) # Integrate spectrum into original spectral pixels
        noise = galaxy/sn        # 1sigma error spectrum
        galaxy = np.random.normal(galaxy, noise) # Add noise to the galaxy spectrum
        start = np.array([vel+np.random.random(), sigma*np.random.uniform(0.85,1.15)])*velscale # Convert to km/s

        pp = ppxf(star, galaxy, noise, velscale, start,
                  goodpixels=np.arange(dx,galaxy.size-dx),
                  plot=False, moments=4, bias=0.5)
        result[j,:] = pp.sol

    print('Calculation time: %.2f s' % (clock()-t))

    plt.clf()
    plt.subplot(221)
    plt.plot(sigmaV*velscale, result[:,0]-vel*velscale, '+k')
    plt.plot(sigmaV*velscale, sigmaV*velscale*0, '-r')
    plt.ylim(-40, 40)
    plt.xlabel('$\sigma_{in}\ (km\ s^{-1})$')
    plt.ylabel('$V - V_{in}\ (km\ s^{-1}$)')

    plt.subplot(222)
    plt.plot(sigmaV*velscale, result[:,1]-sigmaV*velscale, '+k')
    plt.plot(sigmaV*velscale, sigmaV*velscale*0, '-r')
    plt.ylim(-40, 40)
    plt.xlabel('$\sigma_{in}\ (km\ s^{-1})$')
    plt.ylabel('$\sigma - \sigma_{in}\ (km\ s^{-1}$)')

    plt.subplot(223)
    plt.plot(sigmaV*velscale, result[:,2], '+k')
    plt.plot(sigmaV*velscale, sigmaV*velscale*0+h3, '-r')
    plt.ylim(-0.2+h3, 0.2+h3)
    plt.xlabel('$\sigma_{in}\ (km\ s^{-1})$')
    plt.ylabel('$h_3$')

    plt.subplot(224)
    plt.plot(sigmaV*velscale, result[:,3], '+k')
    plt.plot(sigmaV*velscale, sigmaV*velscale*0+h4, '-r')
    plt.ylim(-0.2+h4, 0.2+h4)
    plt.xlabel('$\sigma_{in}\ (km\ s^{-1})$')
    plt.ylabel('$h_4$')

    plt.tight_layout()
    plt.pause(0.01)

#----------------------------------------------------------------------------

if __name__ == '__main__':
    ppxf_simulation_example()
