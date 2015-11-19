#!/usr/bin/env python
##############################################################################
#
# Usage example for the procedure PPXF, which
# implements the Penalized Pixel-Fitting (pPXF) method by
# Cappellari M., & Emsellem E., 2004, PASP, 116, 138.
#
# This example shows how to fit multiple stellar components with different
# stellar population and kinematics.
#
# MODIFICATION HISTORY:
#   V1.0.0: Early test version. Michele Cappellari, Oxford, 20 July 2009
#   V1.1.0: Cleaned up for the paper by Johnston et al. (MNRAS, 2013).
#       MC, Oxford, 26 January 2012
#   V2.0.0: Converted to Python and adapted to the changes in the new public
#       PPXF version, Oxford 8 January 2014
#   V2.0.1: Support both Python 2.6/2.7 and Python 3.x. MC, Oxford, 25 May 2014
#   V2.0.2: Support both Pyfits and Astropy to read FITS files.
#       MC, Oxford, 22 October 2015
#
##############################################################################

from __future__ import print_function

try:
    import pyfits
except:
    from astropy.io import fits as pyfits
from scipy import signal
import numpy as np
from time import clock
import matplotlib.pyplot as plt

from ppxf import ppxf
import ppxf_util as util

def ppxf_two_components_example():

    velscale = 30.

    hdu = pyfits.open('spectra/Rbi1.30z+0.00t12.59.fits')  # Solar metallicitly, Age=12.59 Gyr
    gal_lin = hdu[0].data
    h1 = hdu[0].header
    lamRange1 = h1['CRVAL1'] + np.array([0., h1['CDELT1']*(h1['NAXIS1']-1)])
    model1, logLam1, velscale = util.log_rebin(lamRange1, gal_lin, velscale=velscale)
    model1 /= np.median(model1)

    hdu = pyfits.open('spectra/Rbi1.30z+0.00t01.00.fits')  # Solar metallicitly, Age=1.00 Gyr
    gal_lin = hdu[0].data
    model2, logLam1, velscale = util.log_rebin(lamRange1, gal_lin, velscale=velscale)
    model2 /= np.median(model2)

    model = np.column_stack([model1, model2])
    galaxy = np.empty_like(model)

    # These are the input values in spectral pixels
    # for the (V,sigma) of the two kinematic components
    #
    vel = np.array([0., 250.])/velscale
    sigma = np.array([200., 100.])/velscale

    # The synthetic galaxy model consists of the sum of two
    # SSP spectra with age of 1Gyr and 13Gyr respectively
    # with different velocity and dispersion
    #
    for j in range(len(vel)):
        dx = int(abs(vel[j]) + 4.*sigma[j])   # Sample the Gaussian at least to vel+4*sigma
        v = np.linspace(-dx, dx, 2*dx + 1)
        losvd = np.exp(-0.5*((v - vel[j])/sigma[j])**2) # Gaussian LOSVD
        losvd /= np.sum(losvd) # normaize LOSVD
        galaxy[:, j] = signal.fftconvolve(model[:, j], losvd, mode="same")
        galaxy[:, j] /= np.median(model[:, j])
    galaxy = np.sum(galaxy, axis=1)
    sn = 200.
    np.random.seed(2) # Ensure reproducible results
    galaxy = np.random.normal(galaxy, galaxy/sn) # add noise to galaxy

    # Adopts two templates per kinematic component
    #
    templates = np.column_stack([model1, model2, model1, model2])

    # Start both kinematic components from the same guess.
    # With multiple stellar kinematic components
    # a good starting guess is essential
    #
    start = [np.mean(vel)*velscale, np.mean(sigma)*velscale]
    start = [start, start]
    goodPixels = np.arange(20, 1280)

    t = clock()

    plt.clf()
    plt.subplot(211)
    plt.title("Two components pPXF fit")
    print("+++++++++++++++++++++++++++++++++++++++++++++")

    pp = ppxf(templates, galaxy, galaxy*0+1, velscale, start,
              goodpixels=goodPixels, plot=True, degree=4,
              moments=[4, 4], component=[0, 0, 1, 1])

    plt.subplot(212)
    plt.title("Single component pPXF fit")
    print("---------------------------------------------")

    start = start[0]
    pp = ppxf(templates, galaxy, galaxy*0+1, velscale, start,
              goodpixels=goodPixels, plot=True, degree=4, moments=4)

    plt.tight_layout()
    plt.pause(0.01)

    print("=============================================")
    print("Total elapsed time %.2f s" % (clock() - t))

#------------------------------------------------------------------------------

if __name__ == '__main__':
    ppxf_two_components_example()
