# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     formats: py:percent,ipynb
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.4
# ---

# %% [markdown]
#
# # pPXF: Stellar population and gas emission lines
#
# ![pPXF](https://www-astro.physics.ox.ac.uk/~cappellari/images/ppxf-logo.svg)
#
# ## Description
#
# Usage example for the procedure pPXF originally described in 
# [Cappellari & Emsellem (2004)](http://adsabs.harvard.edu/abs/2004PASP..116..138C),
# substantially upgraded in 
# [Cappellari (2017)](http://adsabs.harvard.edu/abs/2017MNRAS.466..798C) 
# and with the inclusion of photometry and linear constraints in 
# [Cappellari (2023)](https://ui.adsabs.harvard.edu/abs/2023MNRAS.526.3273C).
#
# This example shows how to study stellar population and include gas emission
# lines as templates instead of masking them using the `goodpiixels` keyword.
#
# ### MODIFICATION HISTORY
#
# - V1.0.0: Adapted from PPXF_KINEMATICS_EXAMPLE.
#       Michele Cappellari, Oxford, 12 October 2011
# - V1.1.0: Made a separate routine for the construction of the templates
#       spectral library. MC, Vicenza, 11 October 2012
# - V1.1.1: Includes regul_error definition. MC, Oxford, 15 November 2012
# - V2.0.0: Translated from IDL into Python. MC, Oxford, 6 December 2013
# - V2.0.1: Fit SDSS rather than SAURON spectrum. MC, Oxford, 11 December 2013
# - V2.1.0: Includes gas emissions as templates instead of masking the spectrum.
#       MC, Oxford, 7 January 2014
# - V2.1.1: Support both Python 2.6/2.7 and Python 3.x. MC, Oxford, 25 May 2014
# - V2.1.2: Illustrates how to print and plot emission lines.
#       MC, Oxford, 5 August 2014
# - V2.1.3: Only includes emission lines falling within the fitted wavelength
#       range. MC, Oxford, 3 September 2014
# - V2.1.4: Explicitly sort template files as glob() output may not be sorted.
#       Thanks to Marina Trevisan for reporting problems under Linux.
#       MC, Sydney, 4 February 2015
# - V2.1.5: Included origin='upper' in imshow(). Thanks to Richard McDermid
#       for reporting a different default value with older Matplotlib versions.
#       MC, Oxford, 17 February 2015
# - V2.1.6: Use color= instead of c= to avoid new Matplotlib bug.
#       MC, Oxford, 25 February 2015
# - V2.1.7: Uses Pyfits from Astropy to read FITS files.
#       MC, Oxford, 22 October 2015
# - V2.1.8: Included treatment of the SDSS/MILES vacuum/air wavelength difference.
#       MC, Oxford, 12 August 2016
# - V2.1.9: Automate and test computation of nAge and nMetals.
#       MC, Oxford 1 November 2016
# - V3.0.0: Major upgrade. Compute mass-weighted population parameters and M/L
#       using the new `miles` class which leaves no room for user mistakes.
#       MC, Oxford, 2 December 2016
# - V3.0.1: Make files paths relative to this file, to run the example from
#       any directory. MC, Oxford, 18 January 2017
# - V3.1.0: Use ppxf method pp.plot(gas_component=...) to produce gas emission
#       lines plot. MC, Oxford, 13 March 2017
# - V3.2.0: Uses new ppxf keywords `gas_component` and `gas_names` to print the
#       fluxes and formal errors for the gas emission lines.
#       Uses different kinematic components for the Balmer lines and the rest.
#       MC, Oxford, 28 June 2017
# - V3.3.0: Illustrate how to tie the Balmer emission lines and fit for the
#       gas reddening using the `tie_balmer` keyword. Also limit doublets.
#       MC, Oxford, 31 October 2017
# - V3.3.1: Changed imports for pPXF as a package.
#       Make file paths relative to the pPXF package to be able to run the
#       example unchanged from any directory. MC, Oxford, 17 April 2018
# - V3.3.2: Dropped legacy Python 2.7 support. MC, Oxford, 10 May 2018
# - V4.0.3: Fixed clock DeprecationWarning in Python 3.7.
#       MC, Oxford, 27 September 2018
# - V4.1.0: Produce light-weighted instead of mass-weighted quantities and show
#       how to convert between the two. MC, Oxford, 16 July 2021
# - V4.2.0: Use E-Miles spectral library. MC, Oxford, 16 March 2022
# - V4.3.0: Use the new `sps_util` instead of `miles_util`. 
#       MC, Oxford, 12 November 2023
# - V4.4.0: Included XSL SPS models. MC, Oxford. 29 may 2024
#
# ___

# %%
from time import perf_counter as clock
from pathlib import Path
from urllib import request

import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits

from ppxf.ppxf import ppxf
import ppxf.ppxf_util as util
import ppxf.sps_util as lib

# %% [markdown]
# The two variables below define whether one wants to fit the Balmer lines as
# individual lines with free fluxes, or as a single component, with relative
# fluxes dictated by atomic physics. Similarly, the [SII] doublet can be fit
# individually, or they can have their flux ratios limited (biu not fixed) by
# atomic physics (see details in the docstring of `ppxf.util.emission_lines`).

# %%
tie_balmer = True
limit_doublets = True

# %% [markdown]
# Read SDSS galaxy spectrum taken from here https://www.sdss4.org/.
# The spectrum is *already* log rebinned by the SDSS
# pipeline and `log_rebin` should not be used in this case.

# %%
ppxf_dir = Path(lib.__file__).parent
file = ppxf_dir / 'spectra' / 'NGC3522_SDSS_DR18.fits'
hdu = fits.open(file)
t = hdu['COADD'].data
redshift = hdu['SPECOBJ'].data['z'].item()       # SDSS redshift estimate

flux = t['flux']
galaxy = flux/np.median(flux)             # Normalize spectrum to avoid numerical issues
ln_lam_gal = t['loglam']*np.log(10)       # Convert lg --> ln
lam_gal = np.exp(ln_lam_gal)              # Wavelength in Angstroms (log sampled)

# %% [markdown]
# The SDSS wavelengths are in vacuum, while the MILES ones are in air.
# For a rigorous treatment, the SDSS vacuum wavelengths should be
# converted into air wavelengths and the spectra should be resampled.
# To avoid resampling, given that the wavelength dependence of the
# correction is very weak, I approximate it with a constant factor.

# %%
lam_gal *= np.median(util.vac_to_air(lam_gal)/lam_gal)

# %% [markdown]
# The noise level is chosen to give `Chi^2/DOF=1` for E-MILES without
# regularization (`regul=0`). A constant noise is not a bad approximation in
# the fitted wavelength range and reduces the noise in the fit.

# %%
rms = 0.0158
noise = np.full_like(galaxy, rms)  # Assume constant noise per pixel here

# %% [markdown]
# The velocity step was already chosen by the SDSS pipeline and I convert it below to km/s.
# I use eq.(8) of [Cappellari (2017)](http://adsabs.harvard.edu/abs/2017MNRAS.466..798C)

# %%
c = 299792.458  # speed of light in km/s
d_ln_lam_gal = (ln_lam_gal[-1] - ln_lam_gal[0])/(ln_lam_gal.size - 1)  # Use full lam range for accuracy
velscale = c*d_ln_lam_gal                   # Velocity scale in km/s per pixel (eq.8 of Cappellari 2017)

# %% [markdown]
# I convert the instrumental dispersion to Angstroms

# %%
dlam_gal = np.gradient(lam_gal)             # Size of every pixel in Angstroms
wdisp = t['wdisp']                          # Instrumental dispersion of every pixel, in pixels units
fwhm_gal = 2.355*wdisp*dlam_gal             # Resolution FWHM of every pixel, in Angstroms

# %% [markdown]
# ## Setup stellar templates 
#
# pPXF can be used with any set of SPS population templates. However, I am
# currently providing (with permission) ready-to-use template files for four
# SPS. One can just uncomment one of the four models below. The included files
# are only a subset of the SPS that can be produced with the models, and one
# should use the relevant software/website to produce different sets of SPS
# templates if needed.
#
# 1. If you use the [fsps v3.2](https://github.com/cconroy20/fsps) SPS model
#    templates, please also cite in your paper 
#    [Conroy et al. (2009)](https://ui.adsabs.harvard.edu/abs/2009ApJ...699..486C) and
#    [Conroy et al. (2010)](https://ui.adsabs.harvard.edu/abs/2010ApJ...712..833C).
#
# 2. If you use the [GALAXEV v2020](http://www.bruzual.org/bc03/) SPS model 
#    templates, please also cite in your paper 
#    [Bruzual & Charlot (2003)](https://ui.adsabs.harvard.edu/abs/2003MNRAS.344.1000B).
#
# 3. If you use the [E-MILES](http://miles.iac.es/) SPS model templates,
#    please also cite  in your paper 
#    [Vazdekis et al. (2016)](https://ui.adsabs.harvard.edu/abs/2016MNRAS.463.3409V).
#    <font color="red">WARNING: The E-MILES models only include SPS with age > 63 Myr and
#    are not recommended for highly star forming galaxies.</font>
#
# 4. If you use the [X-Shooter Spectral Library (XSL)](http://xsl.u-strasbg.fr/) 
#    SPS model templates, please also cite in your paper 
#    [Verro et al. (2022)](https://ui.adsabs.harvard.edu/abs/2022A%26A...661A..50V). 
#    <font color="red">WARNING: The XSL models only include SPS with age > 50 Myr and
#    are not recommended for highly star forming galaxies.</font>

# %%
# sps_name = 'fsps'
# sps_name = 'galaxev'
sps_name = 'emiles'
# sps_name = 'xsl'

# %% [markdown]
# Read SPS models file from my GitHub if not already in the `pPXF` package dir.
# The SPS model files are also available on my [GitHub Page](https://github.com/micappe/ppxf_data).

# %%
basename = f"spectra_{sps_name}_9.0.npz"
filename = ppxf_dir / 'sps_models' / basename
if not filename.is_file():
    url = "https://raw.githubusercontent.com/micappe/ppxf_data/main/" + basename
    request.urlretrieve(url, filename)

# %% [markdown]
# Only the E-MILES and XLS SPS libraries have sufficient resolution beyond 7400 A

# %%
if sps_name in ['fsps', 'galaxev']:
    w = lam_gal < 7400
    galaxy = galaxy[w]
    noise = noise[w]
    lam_gal = lam_gal[w]
    fwhm_gal = fwhm_gal[w]

# %% [markdown]
# The templates are normalized to the V-band using norm_range. In this way
# the weights returned by pPXF represent V-band light fractions of each SSP.

# %%
fwhm_gal_dic = {"lam": lam_gal, "fwhm": fwhm_gal}
sps = lib.sps_lib(filename, velscale, fwhm_gal_dic, norm_range=[5070, 5950])

# %% [markdown]
# ## Setup gas templates
#
# The stellar templates are reshaped below into a 2-dim array with each
# spectrum as a column; however, I save the original array dimensions,
# which are needed to specify the regularization dimensions

# %%
reg_dim = sps.templates.shape[1:]
stars_templates = sps.templates.reshape(sps.templates.shape[0], -1)

# %% [markdown]
# Given that both the galaxy spectrum and the templates were normalized to a
# median value around unity, a regularization error of about one percent is a
# good start. See the pPXF documentation for the keyword `regul`.

# %%
regul_err = 0.01

# %% [markdown]
# Estimate the wavelength's fitted range in the rest frame.

# %%
lam_range_gal = np.array([np.min(lam_gal), np.max(lam_gal)])/(1 + redshift)

# %% [markdown]
# Construct a set of Gaussian emission line templates. The `emission_lines`
# function defines the most common lines, but additional lines can be
# included by editing the function in the file ppxf_util.py.

# %%
gas_templates, gas_names, line_wave = util.emission_lines(
    sps.ln_lam_temp, lam_range_gal, fwhm_gal_dic, tie_balmer=tie_balmer,
    limit_doublets=limit_doublets)

# %% [markdown]
# Combines the stellar and gaseous templates into a single array. During the
# pPXF fit they will be assigned a different kinematic `component` value

# %%
templates = np.column_stack([stars_templates, gas_templates])

# %% [markdown]
# Compute the velocity from the redshift estimate using eq. (8) of 
# [Cappellari (2017)](http://adsabs.harvard.edu/abs/2017MNRAS.466..798C).
# If the spectrum was at higher redshift, it would be more convenient to
# de-redshift it before the `pPXF` fit and set `vel=0` below.

# %%
vel = c*np.log(1 + redshift)   # eq.(8) of Cappellari (2017)
start = [vel, 180.]     # (km/s), starting guess for [V, sigma]

# %% [markdown]
# Extract the number of Balmer and forbidden lines from their list of names in `gas_names`.

# %%
n_temps = stars_templates.shape[1]
n_forbidden = np.sum(["[" in a for a in gas_names])  # forbidden lines contain "[*]"
n_balmer = len(gas_names) - n_forbidden

# %% [markdown]
# Assign `component=0` to the stellar templates, `component=1` to the Balmer
# gas emission lines templates and `component=2` to the forbidden lines.

# %%
component = [0]*n_temps + [1]*n_balmer + [2]*n_forbidden
gas_component = np.array(component) > 0  # gas_component=True for gas templates

# %% [markdown]
# Fit $(V, \sigma, h_3, h_4)$ `moments=4` for the stars
# and $(V, \sigma)$ `moments=2` for the two gas kinematic components

# %%
moments = [4, 2, 2]

# %% [markdown]
# Adopt the same starting value for the stars and the two gas components

# %%
start = [start, start, start]

# %% [markdown]
# If the Balmer lines are tied, one should allow for gas reddening.
# The `gas_reddening` can be different from the stellar one, if both are fitted.

# %%
gas_reddening = 0 if tie_balmer else None

# %% [markdown]
# ## pPXF fitting
#
# IMPORTANT: Ideally one would like not to use any polynomial in the fit as
# the continuum shape contains important information on the population.
# Unfortunately, this is often not feasible, due to small calibration
# uncertainties in the spectral shape. To avoid affecting the line strength
# of the spectral features, I exclude additive polynomials (`degree=-1`) and
# only use multiplicative ones (`mdegree=10`). This is only recommended for
# population, not for kinematic extraction, where additive polynomials are
# always recommended.

# %%
t = clock()
pp = ppxf(templates, galaxy, noise, velscale, start, moments=moments,
          degree=-1, mdegree=10, lam=lam_gal, lam_temp=sps.lam_temp,
          regul=1/regul_err, reg_dim=reg_dim, component=component,
          gas_component=gas_component, gas_names=gas_names,
          gas_reddening=gas_reddening)
print(f"Elapsed time in pPXF: {(clock() - t):.2f}")

# %% [markdown]
# When the two $\Delta\chi^2$ below are approximately the same, the solution
# is the smoothest consistent with the observed spectrum.

# %%
print(f"Desired Delta Chi^2: {np.sqrt(2*galaxy.size):#.4g}")
print(f"Current Delta Chi^2: {(pp.chi2 - 1)*galaxy.size:#.4g}")

light_weights = pp.weights[~gas_component]            # Exclude weights of the gas templates
light_weights = light_weights.reshape(reg_dim)        # Reshape to (n_ages, n_metal)

# %% [markdown]
# Given that the templates are normalized to the V-band, the pPXF weights
# represent v-band light fractions, and the computed ages and metallicities
# below are also light weighted in the V-band.

# %%
sps.mean_age_metal(light_weights);

# %% [markdown]
# The `M*/L` is independent on whether one inputs light or mass weights
# and the overall normalization is also irrelevant

# %%
sps.mass_to_light(light_weights, band="SDSS/r", redshift=redshift);

# %% [markdown]
# ## Plot results
#
# Plot fit results for stars and gas. Plot stellar population mass-fraction
# distribution

# %%
plt.clf()
plt.subplot(211)
pp.plot()
plt.title(f"pPXF fit with {sps_name} SPS templates");

plt.subplot(212)
sps.plot(light_weights/light_weights.sum())  # Normalize to light fractions
txt1 = "tied" if tie_balmer else "free"
txt2 = "limited" if limit_doublets else "free"
plt.title(f"Fit with {txt1} Balmer lines and {txt2} [SII] doublet")
plt.tight_layout()
plt.pause(5);
