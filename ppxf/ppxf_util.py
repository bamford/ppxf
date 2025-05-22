"""
###############################################################################

    Copyright (C) 2001-2024, Michele Cappellari
    E-mail: michele.cappellari_at_physics.ox.ac.uk

    This software is provided as is without any warranty whatsoever.
    Permission to use, for non-commercial purposes is granted.
    Permission to modify for personal or internal use is granted,
    provided this copyright and disclaimer are included unchanged
    at the beginning of the file. All other rights are reserved.

###############################################################################

    This file contains the following independent programs:

    1) log_rebin() to rebin a spectrum logarithmically
    2) determine_goodpixels() to mask gas emission lines for pPXF
    3) determine_mask() to mask gas emission lines for pPXF
    4) vac_to_air() to convert vacuum to air wavelengths
    5) air_to_vac() to convert air to vacuum wavelengths
    6) emission_lines() to create gas emission line templates for pPXF
    7) gaussian_filter1d() **deprecated** to convolve a spectrum with a variable sigma
    8) plot_weights_2d() to plot an image of the 2-dim weights
    9) convolve_gauss_hermite() to accurately convolve a spectrum with a LOSVD
    10) synthetic_photometry() to compute photometry from spectra and filters
    11) mag_sun() to compute the Sun absolute magnitude in any band
    12) mag_spectrum() to compute the apparent magnitude from a spectrum in any band
    13) varsmooth() to convolve a spectrum with a variable sigma using FFT

"""
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from ppxf.ppxf import losvd_rfft, rebin

###############################################################################
#
# NAME:
#   log_rebin
#
# MODIFICATION HISTORY:
#   V1.0.0: Using interpolation. Michele Cappellari, Leiden, 22 October 2001
#   V2.0.0: Analytic flux conservation. MC, Potsdam, 15 June 2003
#   V2.1.0: Allow a velocity scale to be specified by the user.
#       MC, Leiden, 2 August 2003
#   V2.2.0: Output the optional logarithmically spaced wavelength at the
#       geometric mean of the wavelength at the border of each pixel.
#       Thanks to Jesus Falcon-Barroso. MC, Leiden, 5 November 2003
#   V2.2.1: Verify that lam_range[0] < lam_range[1].
#       MC, Vicenza, 29 December 2004
#   V2.2.2: Modified the documentation after feedback from James Price.
#       MC, Oxford, 21 October 2010
#   V2.3.0: By default, now preserve the shape of the spectrum, not the
#       total flux. This seems what most users expect from the procedure.
#       Set the keyword /FLUX to preserve flux like in previous versions.
#       MC, Oxford, 30 November 2011
#   V3.0.0: Translated from IDL into Python. MC, Santiago, 23 November 2013
#   V3.1.0: Fully vectorized log_rebin. Typical speed up by two orders of magnitude.
#       MC, Oxford, 4 March 2014
#   V3.1.1: Updated documentation. MC, Oxford, 16 August 2016
#   V3.2.0: Support log-rebinning of arrays of spectra. MC, Oxford, 27 May 2021
#   V4.0.0: Support log-rebinning with non-uniform wavelength sampling.
#       MC, Oxford, 13 April 2022
#   V4.0.1: Fixed program stop with non-uniform log-rebinning of arrays.
#       MC, Oxford, 16 June 2022
#   V4.0.2: Make `velscale` a scalar. MC, Oxford, 4 January 2023


def log_rebin(lam, spec, velscale=None, oversample=1, flux=False):
    """
    Logarithmically rebin a spectrum, or the first dimension of an array of
    spectra arranged as columns, while rigorously conserving the flux. The
    photons in the spectrum are simply redistributed according to a new grid of
    pixels, with logarithmic sampling in the spectral direction.

    When `flux=True` keyword is set, this program performs an exact integration
    of the original spectrum, assumed to be a step function constant within
    each pixel, onto the new logarithmically-spaced pixels. When `flux=False`
    (default) the result of the integration is divided by the size of each
    pixel to return a flux density (e.g. in erg/(s cm^2 A)). The output was
    tested to agree with the analytic solution.

    Parameters
    ----------
    lam: either [lam_min, lam_max] or wavelength `lam` per spectral pixel.
        * If this has two elements, they are assumed to represent the central
          wavelength of the first and last pixels in the spectrum, which is
          assumed to have constant wavelength scale.
          log_rebin is faster with regular sampling.
        * Alternatively one can input the central wavelength of every spectral
          pixel and this allows for arbitrary irregular sampling in wavelength.
          In this case the program assumes the pixels edges are the midpoints
          of the input pixels wavelengths.

        EXAMPLE: For uniform wavelength sampling, using the values in the
        standard FITS keywords (but note that the format can be different)::

            lam = CRVAL1 + CDELT1*np.arange(NAXIS1)

    spec: array_like with shape (npixels,) or (npixels, nspec)
        Input spectrum or array of spectra to rebin logarithmically.
        This can be a vector `spec[npixels]` or array `spec[npixels, nspec]`.
    oversample: int
        Can be used, not to degrade spectral resolution, especially for
        extended wavelength ranges and to avoid aliasing. Default:
        `oversample=1` implies same number of output pixels as input.
    velscale: float
        Velocity scale in km/s per pixels. If this variable is not defined, it
        will be computed to produce the same number of output pixels as the
        input. If this variable is defined by the user it will be used to set
        the output number of pixels and wavelength scale.
    flux: bool
        `True` to preserve total flux, `False` to preserve the flux density.
        When `flux=True` the log rebinning changes the pixels flux in
        proportion to their dlam and the following command will show large
        differences between the spectral shape before and after `log_rebin`::

           plt.plot(exp(ln_lam), specNew)  # Plot log-rebinned spectrum
           plt.plot(np.linspace(lam[0], lam[1], spec.size), spec)

        By default `flux=`False` and `log_rebin` returns a flux density and the
        above two lines produce two spectra that almost perfectly overlap each
        other.

    Returns
    -------
    spec_new: 
        Logarithmically-rebinned spectrum flux.

    ln_lam: 
        Natural logarithm of the wavelength.

    velscale:
        Velocity scale per pixel in km/s.

    """
    lam, spec = np.asarray(lam, dtype=float), np.asarray(spec, dtype=float)
    assert np.all(np.diff(lam) > 0), '`lam` must be monotonically increasing'
    n = len(spec)
    assert lam.size in [2, n], "`lam` must be either a 2-elements range or a vector with the length of `spec`"

    if lam.size == 2:
        dlam = np.diff(lam)/(n - 1)             # Assume constant dlam
        lim = lam + [-0.5, 0.5]*dlam
        borders = np.linspace(*lim, n + 1)
    else:
        lim = 1.5*lam[[0, -1]] - 0.5*lam[[1, -2]]
        borders = np.hstack([lim[0], (lam[1:] + lam[:-1])/2, lim[1]])
        dlam = np.diff(borders)

    ln_lim = np.log(lim)
    c = 299792.458                          # Speed of light in km/s

    if velscale is None:
        m = int(n*oversample)               # Number of output elements
        velscale = c*np.diff(ln_lim)/m      # Only for output (eq. 8 of Cappellari 2017, MNRAS)
        velscale = velscale.item()          # Make velscale a scalar
    else:
        ln_scale = velscale/c
        m = int(np.diff(ln_lim)/ln_scale)   # Number of output pixels

    newBorders = np.exp(ln_lim[0] + velscale/c*np.arange(m + 1))

    if lam.size == 2:
        k = ((newBorders - lim[0])/dlam).clip(0, n-1).astype(int)
    else:
        k = (np.searchsorted(borders, newBorders) - 1).clip(0, n-1)

    specNew = np.add.reduceat((spec.T*dlam).T, k)[:-1]    # Do analytic integral of step function
    specNew.T[...] *= np.diff(k) > 0                      # fix for design flaw of reduceat()
    specNew.T[...] += np.diff(((newBorders - borders[k]))*spec[k].T)    # Add to 1st dimension

    if not flux:
        specNew.T[...] /= np.diff(newBorders)   # Divide 1st dimension

    # Output np.log(wavelength): natural log of geometric mean
    ln_lam = 0.5*np.log(newBorders[1:]*newBorders[:-1])

    return specNew, ln_lam, velscale

###############################################################################
#
# NAME:
#   DETERMINE_GOODPIXELS
#
# MODIFICATION HISTORY:
#   V1.0.0: Michele Cappellari, Leiden, 9 September 2005
#   V1.0.1: Made a separate routine and included additional common emission lines.
#       MC, Oxford 12 January 2012
#   V2.0.0: Translated from IDL into Python. MC, Oxford, 10 December 2013
#   V2.0.1: Updated line list. MC, Oxford, 8 January 2014
#   V2.0.2: Use redshift instead of velocity as input for higher accuracy at large z.
#       MC, Lexington, 31 March 2015
#   V2.0.3: Includes `width` keyword after suggestion by George Privon (Univ. Florida).
#       MC, Oxford, 2 July 2018
#   V2.0.4: More exact determination of limits. MC, Oxford, 28 March 2022


def determine_goodpixels(ln_lam, lam_range_temp, redshift=0, width=800):
    """
    Generates a list of goodpixels to mask a given set of gas emission
    lines. This is meant to be used as input for pPXF.

    :param ln_lam: Natural logarithm np.log(wave) of the wavelength in
        Angstrom of each pixel of the log rebinned *galaxy* spectrum.
    :param lam_range_temp: Two elements vectors [lam_min_temp, lam_max_temp]
        with the minimum and maximum wavelength in Angstrom in the stellar
        *template* used in pPXF.
    :param z: Estimate of the galaxy redshift.
    :return: vector of goodpixels to be used as input for pPXF

    """
    return np.flatnonzero(determine_mask(ln_lam, lam_range_temp, redshift, width))

###############################################################################


def determine_mask(ln_lam, lam_range_temp, redshift=0, width=800):
    """
    Generates a mask to avoid fitting the region possibly contaminated by a
    given set of gas emission lines. This is meant to be used as input for pPXF.
    Here ``mask = True`` for the pixels to include in the fit.

    :param ln_lam: Natural logarithm np.log(wave) of the wavelength in
        Angstrom of each pixel of the log rebinned *galaxy* spectrum.
    :param lam_range_temp: Two elements vectors [lam_min_temp, lam_max_temp]
        with the minimum and maximum wavelength in Angstrom in the stellar
        *template* used in pPXF.
    :param z: Estimate of the galaxy redshift.
    :return: boolean vector mask to be used as input for pPXF

    """
#                     -----[OII]-----    Hdelta   Hgamma   Hbeta   -----[OIII]-----   [OI]    -----[NII]-----   Halpha   -----[SII]-----
    lines = np.array([3726.03, 3728.82, 4101.76, 4340.47, 4861.33, 4958.92, 5006.84, 6300.30, 6548.03, 6583.41, 6562.80, 6716.47, 6730.85])
    # width/2 of masked gas emission region in km/s
    dv = np.full_like(lines, width)
    c = 299792.458  # speed of light in km/s

    flag = False
    for line, dvj in zip(lines, dv):
        flag |= (ln_lam > np.log(line*(1 + redshift)) - dvj/c) \
            & (ln_lam < np.log(line*(1 + redshift)) + dvj/c)

    # Mask edges of stellar library
    flag |= ln_lam > np.log(lam_range_temp[1]*(1 + redshift)) - 900/c
    flag |= ln_lam < np.log(lam_range_temp[0]*(1 + redshift)) + 900/c

    return ~flag

###############################################################################


def _wave_convert(lam):
    """
    Convert between vacuum and air wavelengths using
    equation (1) of Ciddor 1996, Applied Optics 35, 1566
        http://doi.org/10.1364/AO.35.001566

    :param lam - Wavelength in Angstroms
    :return: conversion factor

    """
    lam = np.asarray(lam)
    sigma2 = (1e4/lam)**2
    fact = 1 + 5.792105e-2/(238.0185 - sigma2) + 1.67917e-3/(57.362 - sigma2)

    return fact

###############################################################################


def vac_to_air(lam_vac):
    """
    Convert vacuum to air wavelengths

    :param lam_vac - Wavelength in Angstroms
    :return: lam_air - Wavelength in Angstroms

    """
    return lam_vac/_wave_convert(lam_vac)

###############################################################################


def air_to_vac(lam_air):
    """
    Convert air to vacuum wavelengths

    :param lam_air - Wavelength in Angstroms
    :return: lam_vac - Wavelength in Angstroms

    """
    return lam_air*_wave_convert(lam_air)

###############################################################################
# NAME:
#   GAUSSIAN
#
# MODIFICATION HISTORY:
#   V1.0.0: Written using analytic pixel integration.
#       Michele Cappellari, Oxford, 10 August 2016
#   V2.0.0: Define lines in frequency domain for a rigorous
#       convolution within pPXF at any sigma, including sigma=0.
#       Introduced `pixel` keyword for optional pixel convolution.
#       MC, Oxford, 26 May 2017
#   V2.0.1: Removed Scipy next_fast_len usage. MC, Oxford, 25 January 2019


def gaussian(ln_lam_temp, line_wave, FWHM_gal, pixel=True):
    """
    Instrumental Gaussian line spread function (LSF), optionally analytically
    integrated within the pixels. When used as template for pPXF it is
    rigorously insensitive to undersampling.

    It implements equations (14) and (15) `of Westfall et al. (2019)
    <https://ui.adsabs.harvard.edu/abs/2019AJ....158..231W>`_

    The function is normalized in such a way that::

            line.sum(0) = 1

    When the LSF is not severely undersampled, and when pixel=False, the output
    of this function is nearly indistinguishable from a normalized Gaussian::

      x = (ln_lam_temp[:, None] - np.log(line_wave))/dx
      gauss = np.exp(-0.5*(x/xsig)**2)
      gauss /= np.sqrt(2*np.pi)*xsig

    However, to deal rigorously with the possibility of severe undersampling,
    this Gaussian is defined analytically in frequency domain and transformed
    numerically to time domain. This makes the convolution within pPXF exact
    to machine precision regardless of sigma (including sigma=0).

    :param ln_lam_temp: np.log(wavelength) in Angstrom
    :param line_wave: Vector of lines wavelength in Angstrom
    :param FWHM_gal: FWHM in Angstrom. This can be a scalar or the name of
        a function which returns the instrumental FWHM for given wavelength.
        In this case the sigma returned by pPXF will be the intrinsic one,
        namely the one corrected for instrumental dispersion, in the same
        way as the stellar kinematics is returned.
      - To measure the *observed* dispersion, ignoring the instrumental
        dispersion, one can set FWHM_gal=0. In this case the Gaussian
        line templates reduce to Dirac delta functions. The sigma returned
        by pPXF will be the same one would measure by fitting a Gaussian
        to the observed spectrum (except for the fact that this function
        accurately deals with pixel integration).
    :param pixel: set to True to perform analytic integration over the pixels.
    :return: LSF computed for every ln_lam_temp

    """
    line_wave = np.asarray(line_wave)

    if callable(FWHM_gal):
        FWHM_gal = FWHM_gal(line_wave)

    n = ln_lam_temp.size
    npad = 2**int(np.ceil(np.log2(n)))
    nl = npad//2 + 1  # Expected length of rfft

    dx = (ln_lam_temp[-1] - ln_lam_temp[0])/(n - 1)   # Delta\ln\lam
    x0 = (np.log(line_wave) - ln_lam_temp[0])/dx      # line center
    w = np.linspace(0, np.pi, nl)[:, None]            # Angular frequency

    # Gaussian with sigma=xsig and center=x0,
    # optionally convolved with a unitary pixel UnitBox[]
    # analytically defined in frequency domain
    # and numerically transformed to time domain
    xsig = FWHM_gal/2.355/line_wave/dx    # sigma in pixels units
    rfft = np.exp(-0.5*(w*xsig)**2 - 1j*w*x0)
    if pixel:
        rfft *= np.sinc(w/(2*np.pi))

    line = np.fft.irfft(rfft, n=npad, axis=0)

    return line[:n, :]

###############################################################################
# NAME:
#   EMISSION_LINES
#
# MODIFICATION HISTORY:
#   V1.0.0: Michele Cappellari, Oxford, 7 January 2014
#   V1.1.0: Fixes [OIII] and [NII] doublets to the theoretical flux ratio.
#       Returns line names together with emission lines templates.
#       MC, Oxford, 3 August 2014
#   V1.1.1: Only returns lines included within the estimated fitted wavelength range.
#       This avoids identically zero gas templates being included in the pPXF fit
#       which can cause numerical instabilities in the solution of the system.
#       MC, Oxford, 3 September 2014
#   V1.2.0: Perform integration over the pixels of the Gaussian line spread function
#       using the new function gaussian(). Thanks to Eric Emsellem for the suggestion.
#       MC, Oxford, 10 August 2016
#   V1.2.1: Allow FWHM_gal to be a function of wavelength. MC, Oxford, 16 August 2016
#   V1.2.2: Introduced `pixel` keyword for optional pixel convolution.
#       MC, Oxford, 3 August 2017
#   V1.3.0: New `tie_balmer` keyword to assume intrinsic Balmer decrement.
#       New `limit_doublets` keyword to limit ratios of [OII] & [SII] doublets.
#       New `vacuum` keyword to return wavelengths in vacuum.
#       MC, Oxford, 31 October 2017
#   V1.3.1: Account for the varying pixel size in Angstrom, when specifying the
#       weights for the Balmer series with tie_balmer=True. Many thanks to
#       Kyle Westfall (Santa Cruz) for reporting this bug. MC, Oxford, 10 April 2018
#   V1.3.2: Include more Balmer lines when fitting with tie_balmer=False.
#       MC, Oxford, 8 April 2022
#   V1.4.0: Include extra emission lines from Table 1 of Belfiore+19.
#       MC, Oxford, 28 June 2022


def emission_lines(ln_lam_temp, lam_range_gal, FWHM_gal, pixel=True,
                   tie_balmer=False, limit_doublets=False, vacuum=False):
    """
    Generates an array of Gaussian emission lines to be used as gas templates in pPXF.

    ****************************************************************************

    **ADDITIONAL LINES CAN BE ADDED BY EDITING THE CODE OF THIS PROCEDURE, WHICH 
    IS MEANT AS A TEMPLATE TO BE COPIED AND MODIFIED BY THE USERS AS NEEDED**

    ****************************************************************************

    Generally, these templates represent the instrumental line spread function
    (LSF) at the set of wavelengths of each emission line. In this case, pPXF
    will return the intrinsic (i.e. astrophysical) dispersion of the gas lines.

    Alternatively, one can input FWHM_gal=0, in which case the emission lines
    are delta-functions and pPXF will return a dispersion which includes both
    the instrumental and the intrinsic dispersion.

    For accuracy the Gaussians are integrated over the pixels boundaries.
    This can be changed by setting `pixel`=False.

    The [OI], [OIII] and [NII] doublets are fixed at theoretical flux ratio~3.

    The [OII] and [SII] doublets can be restricted to physical range of ratios.

    The Balmer Series can be fixed to the theoretically predicted decrement.

    Parameters
    ----------
    ln_lam_temp: array_like
        is the natural log of the wavelength of the templates in Angstrom.
        ``ln_lam_temp`` should be the same as that of the stellar templates.
    lam_range_gal: array_like
        is the estimated rest-frame fitted wavelength range. Typically::

            lam_range_gal = np.array([np.min(wave), np.max(wave)])/(1 + z),

        where wave is the observed wavelength of the fitted galaxy pixels and
        z is an initial rough estimate of the galaxy redshift.
    FWHM_gal: float, func or dict
        Instrumental resolution FWHM of the galaxy spectrum under study in
        Angstrom. One can pass either:
            * A scalar; 
            * The name "func" of a function ``func(wave)`` which returns the
              FWHM for a given vector of input wavelengths in Angstrom; 
            * A dictionary ``{"lam":lam, "fwhm":fwhm}`` with the wavelength and
              corresponding instrumental resolution of every pixel of the
              galaxy spectrum in Angstroms.
    pixel: bool, optional
        Set this to ``False`` to ignore pixels integration (default ``True``).
    tie_balmer: bool, optional
        Set this to ``True`` to tie the Balmer lines according to a theoretical
        decrement (case B recombination T=1e4 K, n=100 cm^-3).

        IMPORTANT: The relative fluxes of the Balmer components assumes the
        input spectrum has units proportional to ``erg/(cm**2 s A)``.
    limit_doublets: bool, optional
        Set this to True to limit the ratio of the [OII] and [SII] doublets to
        the ranges allowed by atomic physics.

        An alternative to this keyword is to use the ``constr_templ`` keyword
        of pPXF to constrain the ratio of two templates weights.

        IMPORTANT: when using this keyword, the two output fluxes (flux_1 and
        flux_2) provided by pPXF for the two lines of the doublet, do *not*
        represent the actual fluxes of the two lines, but the fluxes of the two
        input *doublets* of which the fit is a linear combination.
        If the two doublets templates have line ratios rat_1 and rat_2, and
        pPXF prints fluxes flux_1 and flux_2, the actual ratio and flux of the
        fitted doublet will be::

            flux_total = flux_1 + flux_1
            ratio_fit = (rat_1*flux_1 + rat_2*flux_2)/flux_total

        EXAMPLE: For the [SII] doublet, the adopted ratios for the templates are::

            ratio_d1 = flux([SII]6716/6731) = 0.44
            ratio_d2 = flux([SII]6716/6731) = 1.43.

        When pPXF prints (and returns in pp.gas_flux)::

            flux([SII]6731_d1) = flux_1
            flux([SII]6731_d2) = flux_2

        the total flux and true lines ratio of the [SII] doublet are::

            flux_total = flux_1 + flux_2
            ratio_fit([SII]6716/6731) = (0.44*flux_1 + 1.43*flux_2)/flux_total

        Similarly, for [OII], the adopted ratios for the templates are::

            ratio_d1 = flux([OII]3729/3726) = 0.28
            ratio_d2 = flux([OII]3729/3726) = 1.47.

        When pPXF prints (and returns in pp.gas_flux)::

            flux([OII]3726_d1) = flux_1
            flux([OII]3726_d2) = flux_2

        the total flux and true lines ratio of the [OII] doublet are::

            flux_total = flux_1 + flux_2
            ratio_fit([OII]3729/3726) = (0.28*flux_1 + 1.47*flux_2)/flux_total

    vacuum:  bool, optional
        set to ``True`` to assume wavelengths are given in vacuum.
        By default the wavelengths are assumed to be measured in air.

    Returns
    -------
    emission_lines: ndarray
        Array of dimensions ``[ln_lam_temp.size, line_wave.size]`` containing
        the gas templates, one per array column.

    line_names: ndarray
        Array of strings with the name of each line, or group of lines'

    line_wave: ndarray
        Central wavelength of the lines, one for each gas template'

    """

    if isinstance(FWHM_gal, dict):
        FWHM_gal1 = lambda lam: np.interp(lam, FWHM_gal["lam"], FWHM_gal["fwhm"])
    else:
        FWHM_gal1 = FWHM_gal

    #        Balmer:     H10       H9         H8        Heps    Hdelta    Hgamma    Hbeta     Halpha
    balmer = np.array([3798.983, 3836.479, 3890.158, 3971.202, 4102.899, 4341.691, 4862.691, 6564.632])  # vacuum wavelengths

    if tie_balmer:

        # Balmer decrement for Case B recombination (T=1e4 K, ne=100 cm^-3)
        # from Storey & Hummer (1995) https://ui.adsabs.harvard.edu/abs/1995MNRAS.272...41S
        # In electronic form https://cdsarc.u-strasbg.fr/viz-bin/Cat?VI/64
        # See Table B.7 of Dopita & Sutherland (2003) https://www.amazon.com/dp/3540433627
        # Also see Table 4.2 of Osterbrock & Ferland (2006) https://www.amazon.co.uk/dp/1891389343/
        wave = balmer
        if not vacuum:
            wave = vac_to_air(wave)
        gauss = gaussian(ln_lam_temp, wave, FWHM_gal1, pixel)
        ratios = np.array([0.0530, 0.0731, 0.105, 0.159, 0.259, 0.468, 1, 2.86])
        # Account for varying log-sampled pixel size in Angstrom
        ratios *= wave[-2]/wave
        emission_lines = gauss @ ratios
        line_names = ['Balmer']
        w = (lam_range_gal[0] < wave) & (wave < lam_range_gal[1])
        line_wave = np.mean(wave[w]) if np.any(w) else np.mean(wave)

    else:

        line_wave = balmer
        if not vacuum:
            line_wave = vac_to_air(line_wave)
        line_names = ['H10', 'H9', 'H8', 'Heps', 'Hdelta', 'Hgamma', 'Hbeta', 'Halpha']
        emission_lines = gaussian(ln_lam_temp, line_wave, FWHM_gal1, pixel)

    if limit_doublets:

        # The line ratio of this doublet lam3727/lam3729 is constrained by
        # atomic physics to lie in the range 0.28--1.47 (e.g. fig.5.8 of
        # Osterbrock & Ferland (2006) https://www.amazon.co.uk/dp/1891389343/).
        # We model this doublet as a linear combination of two doublets with the
        # maximum and minimum ratios, to limit the ratio to the desired range.
        #       -----[OII]-----
        wave = [3727.092, 3729.875]    # vacuum wavelengths
        if not vacuum:
            wave = vac_to_air(wave)
        names = ['[OII]3726_d1', '[OII]3726_d2']
        gauss = gaussian(ln_lam_temp, wave, FWHM_gal1, pixel)
        doublets = gauss @ [[1, 1], [0.28, 1.47]]  # produces *two* doublets
        emission_lines = np.column_stack([emission_lines, doublets])
        line_names = np.append(line_names, names)
        line_wave = np.append(line_wave, wave)

        # The line ratio of this doublet lam6717/lam6731 is constrained by
        # atomic physics to lie in the range 0.44--1.43 (e.g. fig.5.8 of
        # Osterbrock & Ferland (2006) https://www.amazon.co.uk/dp/1891389343/).
        # We model this doublet as a linear combination of two doublets with the
        # maximum and minimum ratios, to limit the ratio to the desired range.
        #        -----[SII]-----
        wave = [6718.294, 6732.674]    # vacuum wavelengths
        if not vacuum:
            wave = vac_to_air(wave)
        names = ['[SII]6731_d1', '[SII]6731_d2']
        gauss = gaussian(ln_lam_temp, wave, FWHM_gal1, pixel)
        doublets = gauss @ [[0.44, 1.43], [1, 1]]  # produces *two* doublets
        emission_lines = np.column_stack([emission_lines, doublets])
        line_names = np.append(line_names, names)
        line_wave = np.append(line_wave, wave)

    else:

        # Here the two doublets are free to have any ratio
        #         -----[OII]-----     -----[SII]-----
        wave = [3727.092, 3729.875, 6718.294, 6732.674]  # vacuum wavelengths
        if not vacuum:
            wave = vac_to_air(wave)
        names = ['[OII]3726', '[OII]3729', '[SII]6716', '[SII]6731']
        gauss = gaussian(ln_lam_temp, wave, FWHM_gal1, pixel)
        emission_lines = np.column_stack([emission_lines, gauss])
        line_names = np.append(line_names, names)
        line_wave = np.append(line_wave, wave)

    # Here the lines are free to have any ratio
    #       -----[NeIII]-----    HeII      HeI
    wave = [3968.59, 3869.86, 4687.015, 5877.243]  # vacuum wavelengths
    if not vacuum:
        wave = vac_to_air(wave)
    names = ['[NeIII]3968', '[NeIII]3869', 'HeII4687', 'HeI5876']
    gauss = gaussian(ln_lam_temp, wave, FWHM_gal1, pixel)
    emission_lines = np.column_stack([emission_lines, gauss])
    line_names = np.append(line_names, names)
    line_wave = np.append(line_wave, wave)

    ######### Doublets with fixed ratios #########

    # To keep the flux ratio of a doublet fixed, we place the two lines in a single template
    #        -----[OIII]-----
    wave = [4960.295, 5008.240]    # vacuum wavelengths
    if not vacuum:
        wave = vac_to_air(wave)
    doublet = gaussian(ln_lam_temp, wave, FWHM_gal1, pixel) @ [0.33, 1]
    emission_lines = np.column_stack([emission_lines, doublet])
    # single template for this doublet
    line_names = np.append(line_names, '[OIII]5007_d')
    line_wave = np.append(line_wave, wave[1])

    # To keep the flux ratio of a doublet fixed, we place the two lines in a single template
    #        -----[OI]-----
    wave = [6302.040, 6365.535]    # vacuum wavelengths
    if not vacuum:
        wave = vac_to_air(wave)
    doublet = gaussian(ln_lam_temp, wave, FWHM_gal1, pixel) @ [1, 0.33]
    emission_lines = np.column_stack([emission_lines, doublet])
    # single template for this doublet
    line_names = np.append(line_names, '[OI]6300_d')
    line_wave = np.append(line_wave, wave[0])

    # To keep the flux ratio of a doublet fixed, we place the two lines in a single template
    #       -----[NII]-----
    wave = [6549.860, 6585.271]    # air wavelengths
    if not vacuum:
        wave = vac_to_air(wave)
    doublet = gaussian(ln_lam_temp, wave, FWHM_gal1, pixel) @ [0.33, 1]
    emission_lines = np.column_stack([emission_lines, doublet])

    # single template for this doublet
    line_names = np.append(line_names, '[NII]6583_d')
    line_wave = np.append(line_wave, wave[1])

    # Only include lines falling within the estimated fitted wavelength range.
    #
    w = (lam_range_gal[0] < line_wave) & (line_wave < lam_range_gal[1])
    emission_lines = emission_lines[:, w]
    line_names = line_names[w]
    line_wave = line_wave[w]

    print('Emission lines included in gas templates:')
    print(line_names)

    return emission_lines, line_names, line_wave

###############################################################################
# NAME:
#   GAUSSIAN_FILTER1D
#
# MODIFICATION HISTORY:
#   V1.0.0: Written as a replacement for the Scipy routine with the same name,
#       to be used with variable sigma per pixel. MC, Oxford, 10 October 2015
#   V1.1.0: Introduced `mode` keyword. MC, Oxford, 22 April 2022
#   V1.2.0: Removed `mode` keyword and changed edge treatment to agree with
#       ndimage.gaussian_filter1d(mode='constant'). MC, Oxford, 28 June 2024

def gaussian_filter1d(spec, sig):
    """
    **************************************************************************

    **This function is deprecated: one should use `ppxf_util.varsmooth`
    instead, which is superior in every situation.**

    **************************************************************************

    Convolve a spectrum by a Gaussian with different sigma for every pixel.
    If all sigma are the same this routine produces the same output as
    `scipy.ndimage.gaussian_filter1d(mode='constant')
    <https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.gaussian_filter1d.html>`_
    When creating a template library for SDSS data (4000 pixels long),
    this implementation is 60x faster than a naive for-loop over pixels.

    :param spec: vector with the spectrum to convolve
    :param sig: vector of sigma values (in pixels) for every pixel
    :return: spec convolved with a Gaussian with dispersion sig
    """
    sig = sig.clip(0.01)  # forces zero sigmas to have 0.01 pixels
    p = int(1 + 4*np.max(sig))
    m = 2*p + 1  # kernel size
    x2 = np.linspace(-p, p, m)**2

    gau = np.exp(-x2[:, None]/(2*sig**2))
    gau /= gau.sum(0)  # Normalize kernel

    n = spec.size
    a = np.zeros((m, n + 2*p))
    for j in range(m):   # Loop over the small size of the kernel
        a[j, j: j + n] = spec

    conv_spectrum = (a[:, p: -p]*gau).sum(0)

    return conv_spectrum

###############################################################################
# MODIFICATION HISTORY:
#   V1.0.0: Written. Michele Cappellari, Oxford, 25 November 2016
#   V1.0.1: Set `edgecolors` keyword in pcolormesh.
#       MC, Oxford, 14 March 2017
#   V1.1.0: Allow for 1D xgrid, ygrid. Set pcolormesh(..., lw=0.3)
#       MC, Oxford, 27 August 2022

def plot_weights_2d(xgrid, ygrid, weights, xlabel="lg Age (yr)",
                    ylabel="[M/H]", title="Weights Fraction", nodots=False,
                    colorbar=True, colorbar_label="", **kwargs):
    """
    Plot an image of the 2-dim weights, as a function of xgrid and ygrid.
    This function allows for non-uniform spacing in x or y.

    """
    if (xgrid.ndim == ygrid.ndim == 1) and (xgrid.size, ygrid.size) == weights.shape:
        ygrid, xgrid = np.meshgrid(ygrid, xgrid)  # note swapped x -- y

    assert weights.ndim == 2, "`weights` must be 2-dim"
    assert (xgrid.shape == ygrid.shape == weights.shape), \
        'Input arrays (xgrid, ygrid, weights) must have the same shape'

    x = xgrid[:, 0]  # Grid centers
    y = ygrid[0, :]
    xb = (x[1:] + x[:-1])/2  # internal grid borders
    yb = (y[1:] + y[:-1])/2
    xb = np.hstack([1.5*x[0] - x[1]/2, xb, 1.5*x[-1] - x[-2]/2])   # 1st/last border
    yb = np.hstack([1.5*y[0] - y[1]/2, yb, 1.5*y[-1] - y[-2]/2])

    # pcolormesh() is used below to allow for irregular spacing in the
    # sampling of the stellar population parameters (e.g. metallicity)

    ax = plt.gca()
    pc = plt.pcolormesh(xb, yb, weights.T, edgecolors='face', lw=0.3, **kwargs)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    if not nodots:
        plt.plot(xgrid, ygrid, 'w,')
    if colorbar:
        cb = plt.colorbar(pc)
        cb.set_label(colorbar_label)
        plt.sca(ax)  # Activate main plot before returning

    return pc

###############################################################################
# MODIFICATION HISTORY:
#   V1.0.0: Written. Michele Cappellari, Oxford, 8 February 2018
#   V1.0.1: Changed imports for pPXF as a package. MC, Oxford, 16 April 2018
#   V1.0.2: Removed Scipy next_fast_len usage. MC, Oxford, 25 January 2019


def convolve_gauss_hermite(templates, velscale, start, npix,
                           velscale_ratio=1, sigma_diff=0, vsyst=0):
    """
    Convolve a spectrum, or a set of spectra, arranged into columns of an array,
    with a LOSVD parametrized by the Gauss-Hermite series.

    This is intended to reproduce what pPXF does for the convolution and it
    uses the analytic Fourier Transform of the LOSVD introduced in
    Cappellari (2017) http://adsabs.harvard.edu/abs/2017MNRAS.466..798C

    EXAMPLE::

        ...
        pp = ppxf(templates, galaxy, noise, velscale, start,
                  degree=4, mdegree=4, velscale_ratio=ratio, vsyst=dv)

        spec = convolve_gauss_hermite(templates, velscale, pp.sol, galaxy.size,
                                      velscale_ratio=ratio, vsyst=dv)

        The spectrum below is equal to pp.bestfit to machine precision

        spectrum = (spec @ pp.weights)*pp.mpoly + pp.apoly

    :param templates: array[npix_temp, ntemp] (or vector[npix_temp]) of log rebinned spectra
    :param velscale: velocity scale c*Delta(ln_lam) in km/s
    :param start: parameters of the LOSVD [vel, sig, h3, h4,...] in km/s
    :param npix: number of desired output pixels (must be npix <= npix_temp)
    :return: array[npix_temp, ntemp] (or vector[npix_temp]) with the convolved templates

    """
    npix_temp = templates.shape[0]
    templates = templates.reshape(npix_temp, -1)
    start = np.array(start)  # make copy
    start[:2] /= velscale    # convert velocities to pixels
    vsyst /= velscale

    npad = 2**int(np.ceil(np.log2(npix_temp)))
    templates_rfft = np.fft.rfft(templates, npad, axis=0)
    lvd_rfft = losvd_rfft(start, 1, start.shape, templates_rfft.shape[0],
                          1, vsyst, velscale_ratio, sigma_diff)

    conv_temp = np.fft.irfft(templates_rfft*lvd_rfft[:, 0], npad, axis=0)
    conv_temp = rebin(conv_temp[:npix*velscale_ratio, :], velscale_ratio)

    return conv_temp.squeeze()

###############################################################################
# MODIFICATION HISTORY:
#   V1.0.0: Written. Michele Cappellari, Oxford, 15 March 2022
#   V1.1.0: Moved from, sps_util to ppxf_util and request spectra as input.
#       MC, Oxford, 4 April 2022

class synthetic_photometry:
    """
    Purpose
    -------

    Returns the flux in the given band for a single spectrum or a set of
    spectra arranged as columns of an array.

    The fluxes are in units of `ergs/(s cm^2 A)` and are consistent with the
    input spectra for pPXF. The function also returns the effective and pivot
    wavelengths of each band, for each spectrum.

    The filter wavelength is shifted by the input redshift, but the flux
    density is not adjusted for cosmological effects. The user is expected to
    explicitly account for the factor `(1 + z)` when needed.

    Parameters
    ----------

    lam_spec : array_like of shape (n_pixels,)
        Restframe wavelength in Angstrom for every pixel of `spectra`. This can
        be non-uniformly sampled.
    spectra : array_like of shape (n_pixels, n_spectra) or (n_pixels, ...)
        Single spectrum vector or multiple spectra arranged as columns of an
        array. Fluxes must be proportional to `ergs/(s cm^2 A)` units.
    redshift : float
        Approximate redshift of the galaxy under study.
    bands : string array_like of shape (n_bands,)
        String identifying the filter. The matching is done using partial
        string matching with the file FILTER.RES.txt.
        For example, if the filter in the file is "Johnson B-band", one can
        use "B-band" as long as it does not match other filters in the file.
        Additional filters can be added to the file under
        "ppxf/examples/FILTER.RES.txt".

    Returns
    -------

    Attributes of the ``synthetic_photometry`` class.

    .flux : array_like of shape (n_bands, n_spectra,) or (n_bands, ...)
        Fluxes for all spectra in all bands, normalized as in eq.(15) of
        `Cappellari (2023) <https://ui.adsabs.harvard.edu/abs/2023MNRAS.526.3273C>`_.
    .lam_eff : array_like of shape (n_bands, n_spectra,) or (n_bands, ...)
        Effective wavelength for each band, weighted by each spectrum,
        as in eq.(22) of `Cappellari (2023)`_.
    .lam_piv : array_like of shape (n_bands)
        Source-independent pivot wavelength for each band, as in eq.(17) of 
        `Cappellari (2023)`_. This is used to convert exactly 
        ``<f(lam)> = <f(nu)>c/lam_piv^2``, using the observed 
        ``lam_piv*(1 + z)`` pivot wavelength.
    .ok : array_like of shape (n_bands)
        Boolean array indicating whether the input `bands` are within the
        wavelength range of the spectra at the input redshift.
        Fluxes and wavelengths with corresponding `ok` values of False are
        returned as np.nan and should not be used.
    """

    def __init__(self, lam_spec, spectra, bands,
                 redshift=0, filters_file=None, quiet=False):

        assert len(lam_spec) == len(spectra), \
            "`lam_spec`  must have the same number of elements as `spectra.shape[0]`"

        if filters_file is None:
            ppxf_dir = Path(__file__).parent  # path of current file
            filters_file = ppxf_dir / 'examples' / 'FILTER.RES.txt'

        bands = np.atleast_1d(bands)
        lam_eff, flux = np.empty((2, len(bands), *spectra.shape[1:]))
        lam_piv = np.empty(len(bands))
        ok = np.empty(len(bands), dtype=bool)
        for j, band in enumerate(bands):
            lam_eff[j], lam_piv[j], flux[j], ok[j] = synthetic_photometry_one_band(
                lam_spec, spectra, band, redshift, filters_file)
            if not quiet:
                if ok[j]:
                    print(f"{j + 1:3d}: {band}")
                else:
                    print(f"{j + 1:3d} --- Outside template: {band}")

        self.lam_piv = lam_piv
        self.lam_eff = lam_eff
        self.flux = flux
        self.ok = ok

###############################################################################


def synthetic_photometry_one_band(lam_spec, spectra, band, redshift, filters_file):
    """
    Compute the average fluxes inside the `band` for a single spectrum or a set
    of `spectra` arranged as columns of an array, all with common vector of
    wavelengths `lam_spec`.

    `flux` uses eq.(15), `lam_piv` eq.(17) and `lam_eff` eq.(22)
    of Cappellari (2023, MNRAS)

    """
    lam_resp, response = read_filter(band, filters_file)
    lam_resp /= 1 + redshift    
    lam_in_fwhm = lam_resp[response > 0.5*np.max(response)]     # I want FWHM fully covered
    ok = np.all((lam_in_fwhm >= lam_spec[0]) & (lam_in_fwhm <= lam_spec[-1]))

    if ok:
        fil = np.interp(lam_spec, lam_resp, response, left=0, right=0)
        fdlam = fil*np.gradient(lam_spec)
        filam2 = fdlam*lam_spec
        filam3 = filam2*lam_spec
        int1 = filam2.sum()                 # Integrate[S[lam]*lam, lam]
        int2 = filam3.sum()                 # Integrate[S[lam]*lam^2, lam]
        int3 = (spectra.T @ filam2).T       # Integrate[g[lam]*S[lam]*lam, lam]        
        int4 = (spectra.T @ filam3).T       # Integrate[g[lam]*S[lam]*lam^2, lam]
        int5 = (fdlam/lam_spec).sum()       # Integrate[S[lam]/lam, lam]
        flux = int3/int1
        with np.errstate(invalid='ignore'):
            lam_eff = np.where(int3 > 0, int4/int3, int2/int1)
        lam_piv = np.sqrt(int1/int5)
    else:
        lam_eff = flux = np.full(spectra.shape[1:], np.nan)
        lam_piv = np.nan

    return lam_eff, lam_piv, flux, ok

###############################################################################


def read_filter(band, filters_file):
    """
    Read a filter response function from a text file. The file is expected to
    contain a set of filters where the first line includes the filter name and
    starts with an integer number giving the number of rows with the filter
    response values. Subsequent rows contain the response function wavelength
    in Angstrom and the response function value, in the second and third column
    respectively. The absolute normalization of the response function is
    irrelevant.

    FILE FORMAT EXAMPLE:

          4 Johnson B-band ...
        1   3000   0
        2   3500   0.5
        3   4000   0.3
        4   4500   0
          3 Johnson V-band ...
        1   4500   0
        2   5000   0.5
        3   5500   0
    """
    lines = []
    with open(filters_file) as f:
        for ln in f:
            if band in ln:
                for j in range(int(ln.split()[0])):
                    lines.append(next(f))
                break

    if not lines:
        raise ValueError(f"Filter {band} not included in file {filters_file}")

    j, lam, resp = np.array([ln.split() for ln in lines], float).T

    return lam, resp

###############################################################################


def mag_sun(bands, redshift=0, system='AB', quiet=True):
    """
    Computes the absolute magnitude of the Sun in the AB or Vega magnitude
    system in the given set of bands, optionally shifted to the given redshift.

    Parameters
    ----------

    bands : string or array_like of strings with shape (n_bands,)
        Bands for which one wants to compute the magnitude.
    redshift : float, optional
        The wavelenggth is multiplied by ``(1 + redshift)`` before the
        computation of the flux. No dimming is applied to the spectrum flux.
    system : {'AB', 'Vega'}
        Assumed photometric system.
    """
    ppxf_dir = Path(__file__).parent        # path of current file
    filename = ppxf_dir / 'sps_models' / 'spectra_sun_vega.npz'
    a = np.load(filename)                       # Spectra in erg/s/cm2/A at 10pc
    flux_sun, lam = a["flux_sun"], a["lam"]     # lam in Angstrom

    if system == 'AB':
        c_as = 299792458e+10                # speed of light in A/s
        flux_ref = 3631e-23*c_as/lam**2     # AB flux in erg/s/cm2/A
    elif system == 'Vega':
        flux_ref = a["flux_vega"]

    flux = np.column_stack([flux_sun, flux_ref])
    p = synthetic_photometry(lam, flux, bands, redshift=redshift, quiet=quiet)
    sun_mag = -2.5*np.log10(np.divide(*p.flux.T))

    return sun_mag

###############################################################################


def mag_spectrum(lam, spectrum, bands='SDSS/r', redshift=0, system='AB', quiet=True):
    """
    Computes the apparent magnitude from a spectrum in erg/s/cm2/A, in the AB
    or Vega magnitude system in the given set of bands, optionally shifted to
    the given redshift.

    Parameters
    ----------

    lam : array_like with shape (n_pixels,)
        Wavelength in Angstrom of every pixel in the spectrum.
    spectrum : array_like with shape (n_pixels,)
        Spectrum flux density in erg/s/cm2/A.
    bands : string or array_like of strings with shape (n_bands,)
        Bands for which one wants to compute the magnitude.
    redshift : float, optional
        The wavelenggth is multiplied by ``(1 + redshift)`` before the
        computation of the flux. No dimming is applied to the spectrum flux.
    system : {'AB', 'Vega'}
        Assumed photometric system.

    Returns
    -------

    mag : array_like with shape (n_bands,)
        Apparent magnitude in each of the input bands. If the redshift is
        nonzero, the spectrum is red-shifted to the desired redshift before
        computing the magnitude. However, no cosmological dimming of the
        spectrum is performed.
    """
    p2 = synthetic_photometry(lam, spectrum, bands, redshift=redshift, quiet=quiet)

    if system == 'AB':
        c_as = 299792458e+10                # speed of light in A/s
        flux = 3631e-23*c_as/lam**2         # AB flux in erg/s/cm2/A
    elif system == 'Vega':
        ppxf_dir = Path(__file__).parent    # path of current file
        filename = ppxf_dir / 'sps_models' / 'spectra_sun_vega.npz'
        a = np.load(filename)               # Spectra in erg/s/cm2/A at 10pc
        flux, lam = a["flux_vega"], a["lam"]

    p1 = synthetic_photometry(lam, flux, bands, redshift=redshift, quiet=quiet)
    mag = -2.5*np.log10(p2.flux/p1.flux)

    return mag

##############################################################################


def interp(xout, xin, yin):
    """Applies `numpy.interp` to the last dimension of `yin`"""

    yout = [np.interp(xout, xin, y) for y in yin.reshape(-1, xin.size)]

    return np.reshape(yout, (*yin.shape[:-1], -1))

##############################################################################
# NAME:
#   varsmooth
#
# MODIFICATION HISTORY:
#   V1.0.0: Michele Cappellari, Oxford, 3 September 2022
#   V1.1.0: Faster convolution for a scalar sigma. MC, Oxford, 12 November 2023
#   V1.1.1: Removed dependency on legacy scipy.interpolate.interp1d using
#       faster loop over np.interp. MC, Oxford, 26 April 2024

def varsmooth(x, y, sig_x, xout=None, oversample=1):
    """    
    Fast and accurate convolution with a Gaussian of variable width.

    This function performs an accurate Fourier convolution of a vector, or the
    columns of an array, with a Gaussian kernel that has a varying or constant
    standard deviation (sigma) per pixel. The convolution is done using fast
    Fourier transform (FFT) and the analytic expression of the Fourier
    transform of the Gaussian function, like in the pPXF method. This allows
    for an accurate convolution even when the Gaussian is severely
    undersampled.

    This function is recommended over standard convolution even when dealing
    with a constant Gaussian width, due to its more accurate handling of
    undersampling issues.

    This function implements Algorithm 1 in `Cappellari (2023)
    <https://ui.adsabs.harvard.edu/abs/2023MNRAS.526.3273C>`_

    Parameters
    ----------
    x : array_like
        Coordinate of every pixel in `y`.
    y : array_like
        Input vector or array of column-spectra.
    sig_x : float or array_like
        Gaussian sigma of every pixel in units of `x`.
        If sigma is constant, `sig_x` can be a scalar. 
        In this case, `x` must be uniformly sampled.
    oversample : float, optional
        Oversampling factor before convolution (default: 1).
    xout : array_like, optional
        Output `x` coordinate used to compute the convolved `y`.

    Returns
    -------
    yout : array_like
        Convolved vector or columns of the array `y`.

    """
    assert len(x) == len(y), "`x` and `y` must have the same length"

    if np.isscalar(sig_x):
        dx = np.diff(x)
        assert np.all(np.isclose(dx[0], dx)), "`x` must be uniformly spaced, when `sig_x` is a scalar"
        n = len(x)
        sig_max = sig_x*(n - 1)/(x[-1] - x[0])
        y_new = y.T
    else:
        assert len(x) == len(sig_x), "`x` and `sig_x` must have the same length"
        
        # Stretches spectrum to have equal sigma in the new coordinate
        sig = sig_x/np.gradient(x)
        sig = sig.clip(0.1)   # Clip to >=0.1 pixels
        sig_max = np.max(sig)*oversample
        xs = np.cumsum(sig_max/sig)
        n = int(np.ceil(xs[-1] - xs[0]))
        x_new = np.linspace(xs[0], xs[-1], n)
        y_new = interp(x_new, xs, y.T)

    # Convolve spectrum with a Gaussian using analytic FT like pPXF
    npad = 2**int(np.ceil(np.log2(n)))
    ft = np.fft.rfft(y_new, npad)
    w = np.linspace(0, np.pi*sig_max, ft.shape[-1])
    ft_gau = np.exp(-0.5*w**2)
    yout = np.fft.irfft(ft*ft_gau, npad)[..., :n]

    if not np.isscalar(sig_x):
        if xout is not None:
            xs = np.interp(xout, x, xs)  # xs is 1-dim
        yout = interp(xs, x_new, yout)

    return yout.T

###############################################################################
