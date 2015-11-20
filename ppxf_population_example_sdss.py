#!/usr/bin/env python
##############################################################################
#
# This PPXF_POPULATION_EXAMPLE routine shows how to study stellar population with
# the procedure PPXF, which implements the Penalized Pixel-Fitting (pPXF) method by
# Cappellari M., & Emsellem E., 2004, PASP, 116, 138.
#
# MODIFICATION HISTORY:
#   V1.0.0: Adapted from PPXF_KINEMATICS_EXAMPLE.
#       Michele Cappellari, Oxford, 12 October 2011
#   V1.1.0: Made a separate routine for the construction of the templates
#       spectral library. MC, Vicenza, 11 October 2012
#   V1.1.1: Includes regul_error definition. MC, Oxford, 15 November 2012
#   V2.0.0: Translated from IDL into Python. MC, Oxford, 6 December 2013
#   V2.0.1: Fit SDSS rather than SAURON spectrum. MC, Oxford, 11 December 2013
#   V2.0.2: Support both Python 2.6/2.7 and Python 3.x. MC, Oxford, 25 May 2014
#   V2.0.3: Explicitly sort template files as glob() output may not be sorted.
#       Thanks to Marina Trevisan for reporting problems under Linux.
#       MC, Sydney, 4 February 2015
#   V2.0.4: Included origin='upper' in imshow(). Thanks to Richard McDermid
#       for reporting a different default value with older Matplotlib versions.
#       MC, Oxford, 17 February 2015
#   V2.1.0: Illustrates how to compute mass-weighted quantities from the fit.
#       After feedback from Tatiana Moura. MC, Oxford, 25 March 2015
#   V2.1.1: Use redshift in determine_goodpixels. MC, Oxford, 3 April 2015
#   V2.1.2: Support both Pyfits and Astropy to read FITS files.
#       MC, Oxford, 22 October 2015
#
##############################################################################

from __future__ import print_function

try:
    import pyfits
except:
    from astropy.io import fits as pyfits
from scipy import ndimage
import numpy as np
import glob
import matplotlib.pyplot as plt
from time import clock

from ppxf import ppxf
import ppxf_util as util

def setup_spectral_library(velscale, FWHM_gal):

    # Read the list of filenames from the Single Stellar Population library
    # by Vazdekis et al. (2010, MNRAS, 404, 1639) http://miles.iac.es/.
    #
    # For this example I downloaded from the above website a set of
    # model spectra with default linear sampling of 0.9A/pix and default
    # spectral resolution of FWHM=2.51A. I selected a Salpeter IMF
    # (slope 1.30) and a range of population parameters:
    #
    #     [M/H] = [-1.71, -1.31, -0.71, -0.40, 0.00, 0.22]
    #     Age = np.linspace(np.log10(1), np.log10(17.7828), 26)
    #
    # This leads to a set of 156 model spectra with the file names like
    #
    #     Mun1.30Zm0.40T03.9811.fits
    #
    # IMPORTANT: the selected models form a rectangular grid in [M/H]
    # and Age: for each Age the spectra sample the same set of [M/H].
    #
    # We assume below that the model spectra have been placed in the
    # directory "miles_models" under the current directory.
    #
    vazdekis = glob.glob('miles_models/Mun1.30*.fits')
    vazdekis.sort()
    FWHM_tem = 2.51 # Vazdekis+10 spectra have a resolution FWHM of 2.51A.

    # Extract the wavelength range and logarithmically rebin one spectrum
    # to the same velocity scale of the SDSS galaxy spectrum, to determine
    # the size needed for the array which will contain the template spectra.
    #
    hdu = pyfits.open(vazdekis[0])
    ssp = hdu[0].data
    h2 = hdu[0].header
    lamRange_temp = h2['CRVAL1'] + np.array([0., h2['CDELT1']*(h2['NAXIS1']-1)])
    sspNew, logLam2, velscale = util.log_rebin(lamRange_temp, ssp, velscale=velscale)

    # Create a three dimensional array to store the
    # two dimensional grid of model spectra
    #
    nAges = 26
    nMetal = 6
    templates = np.empty((sspNew.size, nAges, nMetal))

    # Convolve the whole Vazdekis library of spectral templates
    # with the quadratic difference between the SDSS and the
    # Vazdekis instrumental resolution. Logarithmically rebin
    # and store each template as a column in the array TEMPLATES.

    # Quadratic sigma difference in pixels Vazdekis --> SDSS
    # The formula below is rigorously valid if the shapes of the
    # instrumental spectral profiles are well approximated by Gaussians.
    #
    FWHM_dif = np.sqrt(FWHM_gal**2 - FWHM_tem**2)
    sigma = FWHM_dif/2.355/h2['CDELT1'] # Sigma difference in pixels

    # These are the array where we want to store
    # the characteristics of each SSP model
    #
    logAge_grid = np.empty((nAges, nMetal))
    metal_grid = np.empty((nAges, nMetal))

    # These are the characteristics of the adopted rectangular grid of SSP models
    #
    logAge = np.linspace(np.log10(1), np.log10(17.7828), nAges)
    metal = [-1.71, -1.31, -0.71, -0.40, 0.00, 0.22]

    # Here we make sure the spectra are sorted in both [M/H]
    # and Age along the two axes of the rectangular grid of templates.
    # A simple alphabetical ordering of Vazdekis's naming convention
    # does not sort the files by [M/H], so we do it explicitly below
    #
    metal_str = ['m1.71', 'm1.31', 'm0.71', 'm0.40', 'p0.00', 'p0.22']
    for k, mh in enumerate(metal_str):
        files = [s for s in vazdekis if mh in s]
        for j, filename in enumerate(files):
            hdu = pyfits.open(filename)
            ssp = hdu[0].data
            ssp = ndimage.gaussian_filter1d(ssp,sigma)
            sspNew, logLam2, velscale = util.log_rebin(lamRange_temp, ssp, velscale=velscale)
            templates[:, j, k] = sspNew  # Templates are *not* normalized here
            logAge_grid[j, k] = logAge[j]
            metal_grid[j, k] = metal[k]

    return templates, lamRange_temp, logAge_grid, metal_grid

#------------------------------------------------------------------------------

def ppxf_population_example_sdss():

    # Read SDSS DR8 galaxy spectrum taken from here http://www.sdss3.org/dr8/
    # The spectrum is *already* log rebinned by the SDSS DR8
    # pipeline and log_rebin should not be used in this case.
    #
    file = 'spectra/NGC3522_SDSS_DR8.fits'
    hdu = pyfits.open(file)
    t = hdu[1].data
    z = float(hdu[1].header["Z"]) # SDSS redshift estimate

    # Only use the wavelength range in common between galaxy and stellar library.
    #
    mask = (t['wavelength'] > 3540) & (t['wavelength'] < 7409)
    flux = t['flux'][mask]
    galaxy = flux/np.median(flux)   # Normalize spectrum to avoid numerical issues
    wave = t['wavelength'][mask]

    # The noise level is chosen to give Chi^2/DOF=1 without regularization (REGUL=0)
    #
    noise = galaxy*0 + 0.01528           # Assume constant noise per pixel here

    # The velocity step was already chosen by the SDSS pipeline
    # and we convert it below to km/s
    #
    c = 299792.458 # speed of light in km/s
    velscale = np.log(wave[1]/wave[0])*c
    FWHM_gal = 2.76 # SDSS has an instrumental resolution FWHM of 2.76A.

    templates, lamRange_temp, logAge_grid, metal_grid = \
        setup_spectral_library(velscale, FWHM_gal)

    # The galaxy and the template spectra do not have the same starting wavelength.
    # For this reason an extra velocity shift DV has to be applied to the template
    # to fit the galaxy spectrum. We remove this artificial shift by using the
    # keyword VSYST in the call to PPXF below, so that all velocities are
    # measured with respect to DV. This assume the redshift is negligible.
    # In the case of a high-redshift galaxy one should de-redshift its
    # wavelength to the rest frame before using the line below as described
    # in PPXF_KINEMATICS_EXAMPLE_SAURON.
    #
    c = 299792.458
    dv = c*np.log(lamRange_temp[0]/wave[0])  # km/s
    goodpixels = util.determine_goodpixels(np.log(wave), lamRange_temp, z)

    # Here the actual fit starts. The best fit is plotted on the screen.
    #
    # IMPORTANT: Ideally one would like not to use any polynomial in the fit
    # as the continuum shape contains important information on the population.
    # Unfortunately this is often not feasible, due to small calibration
    # uncertainties in the spectral shape. To avoid affecting the line strength of
    # the spectral features, we exclude additive polynomials (DEGREE=-1) and only use
    # multiplicative ones (MDEGREE=10). This is only recommended for population, not
    # for kinematic extraction, where additive polynomials are always recommended.
    #
    vel = c*np.log(1 + z)   # Initial estimate of the galaxy velocity in km/s
    start = [vel, 180.]  # (km/s), starting guess for [V,sigma]

    # See the pPXF documentation for the keyword REGUL,
    # for an explanation of the following two lines
    #
    templates /= np.median(templates) # Normalizes templates by a scalar
    regul_err = 0.004  # Desired regularization error

    t = clock()

    plt.clf()
    plt.subplot(211)

    pp = ppxf(templates, galaxy, noise, velscale, start,
              goodpixels=goodpixels, plot=True, moments=4, degree=-1,
              vsyst=dv, clean=False, mdegree=10, regul=1./regul_err)

    # When the two numbers below are the same, the solution is the smoothest
    # consistent with the observed spectrum.
    #
    print('Desired Delta Chi^2: %.4g' % np.sqrt(2*goodpixels.size))
    print('Current Delta Chi^2: %.4g' % ((pp.chi2 - 1)*goodpixels.size))
    print('Elapsed time in PPXF: %.2f s' % (clock() - t))

    print('Mass-weighted <logAge> [Gyr]: %.3g' %
          (np.sum(pp.weights*logAge_grid.ravel())/np.sum(pp.weights)))
    print('Mass-weighted <[M/H]>: %.3g' %
          (np.sum(pp.weights*metal_grid.ravel())/np.sum(pp.weights)))

    plt.subplot(212)
    s = templates.shape
    weights = pp.weights.reshape(s[1:])/pp.weights.sum()
    plt.imshow(np.rot90(weights), interpolation='nearest', 
               cmap='gist_heat', aspect='auto', origin='upper',
               extent=[np.log10(1), np.log10(17.7828), -1.9, 0.45])
    plt.colorbar()
    plt.title("Mass Fraction")
    plt.xlabel("log$_{10}$ Age (Gyr)")
    plt.ylabel("[M/H]")
    plt.tight_layout()
    plt.show()

#------------------------------------------------------------------------------

if __name__ == '__main__':
    ppxf_population_example_sdss()
    plt.pause(0.01)
