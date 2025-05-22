# ---
# jupyter:
#   jupytext:
#     formats: py:percent,ipynb
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.4
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
#
# # pPXF: Bootstrapping non-parametric stellar population
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
# ### MODIFICATION HISTORY
#
# * V1.0.0: Michele Cappellari, Oxford, 29 March 2022
# * V1.1.0: MC, Oxford, 28 November 2023: Updated for pPXF 9.0 using the new `sps_util`.
# * V1.1.1: MC, Oxford, 7 February 2024: Updated for pPXF 9.1 using the improved `sps_util.mass_to_light`.

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
# ## Function to boostrap the spectrum residuals

# %%
def bootstrap_residuals(model, resid, wild=True):
    """
    https://en.wikipedia.org/wiki/Bootstrapping_(statistics)#Resampling_residuals
    https://en.wikipedia.org/wiki/Bootstrapping_(statistics)#Wild_bootstrap

    Davidson & Flachaire (2008) eq.(7) gives the recommended Rademacher
    distribution form of the wild bootstrapping probability used here.

    https://doi.org/10.1016/j.jeconom.2008.08.003

    :param spec: model (e.g. best fitting spectrum)
    :param res: residuals (best_fit - observed)
    :param wild: use wild bootstrap to allow for variable errors
    :return: new model with bootstrapped residuals

    """
    if wild:    # Wild Bootstrapping: generates -resid or resid with prob=1/2
        eps = resid*(2*np.random.randint(2, size=resid.size) - 1)
    else:       # Standard Bootstrapping: random selection with repetition
        eps = np.random.choice(resid, size=resid.size)

    return model + eps


# %% [markdown]
# ## Read the spectrum
#
# Read SDSS galaxy spectrum taken from here https://www.sdss4.org/.
# The spectrum is *already* log rebinned by the SDSS
# pipeline and `log_rebin` should not be used in this case.

# %%
ppxf_dir = Path(lib.__file__).parent
file = ppxf_dir / 'spectra' / 'NGC3522_SDSS_DR18.fits'
hdu = fits.open(file)
t = hdu['COADD'].data
redshift = hdu['SPECOBJ'].data['z'].item()       # SDSS redshift estimate

galaxy = t['flux']/np.median(t['flux'])     # Normalize spectrum to avoid numerical issues
ln_lam_gal = t['loglam']*np.log(10)         # Convert lg --> ln
lam_gal = np.exp(ln_lam_gal)                # Wavelength in Angstroms (log sampled)

# %% [markdown]
# The SDSS wavelengths are in vacuum, while the MILES ones are in air. For a
# rigorous treatment, the SDSS vacuum wavelengths should be converted into air
# wavelengths and the spectra should be resampled. To avoid resampling, given
# that the wavelength dependence of the correction is very weak, I approximate
# it with a constant factor.

# %%
lam_gal *= np.median(util.vac_to_air(lam_gal)/lam_gal)

# %% [markdown]
# I choose the noise to give $\chi^2/{\rm DOF}=1$ without regularization
# (`regul=0`). A constant noise is not a bad approximation in the fitted
# wavelength range and reduces the noise in the fit.
#

# %%
noise = np.full_like(galaxy, 0.01635)  # Assume constant noise per pixel here

# %% [markdown]
# The velocity step per spectral pixel was already chosen by the SDSS pipeline
# and I convert it below to km/s.

# %%
c = 299792.458                          # speed of light in km/s
d_ln_lam = np.log(lam_gal[-1]/lam_gal[0])/(lam_gal.size - 1)  # Average ln_lam step
velscale = c*d_ln_lam                   # eq. (8) of Cappellari (2017)
FWHM_gal = 2.76                         # SDSS has an approximate instrumental resolution FWHM of 2.76A.

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
# Below, I read the SPS models file from my GitHub if not already in the pPXF
# package dir. I am not distributing the templates with pPXF anymore. The SPS
# model files are also available [this GitHub
# page](https://github.com/micappe/ppxf_data).

# %%
ppxf_dir = Path(lib.__file__).parent
basename = f"spectra_{sps_name}_9.0.npz"
filename = ppxf_dir / 'sps_models' / basename
if not filename.is_file():
    url = "https://raw.githubusercontent.com/micappe/ppxf_data/main/" + basename
    request.urlretrieve(url, filename)

# %% [markdown]
# I normalize the templates to `mean=1` within the FWHM of the V-band. In this
# way the weights returned by pPXF and mean values are light-weighted
# quantities

# %%
sps = lib.sps_lib(filename, velscale, FWHM_gal, norm_range=[5070, 5950])

# %% [markdown]
# I reshape the stellar templates into a 2-dim array with each spectrum as a
# column, however we save the original array dimensions, which are needed to
# specify the regularization dimensions

# %%
reg_dim = sps.templates.shape[1:]
stars_templates = sps.templates.reshape(sps.templates.shape[0], -1)

# %% [markdown]
# I want the first `pPXF` fit to be regularized because this suppresses the
# noise makes it more representative of the underlying galaxy spectrum. See the
# `pPXF` documentation for the keyword `regul`

# %%
regul_err = 0.1  # Large regularization error

# %% [markdown]
# ## Setup the gas emission lines templates
#
# This is the estimated wavelength fitted range in the rest frame. It is needed
# to find which gas emission lines fall within the observed range.

# %%
lam_range_gal = np.array([np.min(lam_gal), np.max(lam_gal)])/(1 + redshift)

# %% [markdown]
# I construct a set of Gaussian emission line templates. The
# `ppxf_util.emission_lines` function defines the most common lines, but `pPXF`
# allows complete freedom to add additional lines by simply editing the
# function in the file `ppxf_util.py`. I fix the ratios of the Balmer lines
# using the `tie_balmer=True` keyword.

# %%
gas_templates, gas_names, line_wave = util.emission_lines(sps.ln_lam_temp, lam_range_gal, FWHM_gal, tie_balmer=1)

# %% [markdown]
# I combines the stellar and gaseous templates into a single array of
# templates. During the PPXF fit they will be assigned a different kinematic
# `component` value.

# %%
templates = np.column_stack([stars_templates, gas_templates])

# %% [markdown]
# ## Input parameters for pPXF
#
# The expression below is the **definition** of velocities in `pPXF`. One
# should always use this precise expression rather than other approximations.
# E.g. **never** use $z\approx V/c$. See Sec. 2.3 of 
# [Cappellari (2017)](https://ui.adsabs.harvard.edu/abs/2017MNRAS.466..798C/abstract) 
# for details

# %%
c = 299792.458
vel = c*np.log(1 + redshift)   # eq. (8) of Cappellari (2017)
start = [vel, 180.]     # (km/s), starting guess for [V, sigma]

# %% [markdown]
# In this example I consider two gas components, one for the Balmer and another for the forbidden lines.

# %%
n_temps = stars_templates.shape[1]
n_forbidden = np.sum(["[" in a for a in gas_names])  # forbidden lines contain "[*]"
n_balmer = len(gas_names) - n_forbidden

# %% [markdown]
# Assign `component=0` to the stellar templates, `component=1` to the Balmer
# gas emission lines templates and `component=2` to the gas forbidden lines.

# %%
component = [0]*n_temps + [1]*n_balmer + [2]*n_forbidden
gas_component = np.array(component) > 0  # gas_component=True for gas templates

# %% [markdown]
# Fit two moments (V, sig) `moments=2` for the stars and for the two gas
# kinematic components

# %%
moments = [2, 2, 2]

# %% [markdown]
# Adopt the same starting value for the stars and the two gas components

# %%
start = [start, start, start]

# %% [markdown]
# ## The first pPXF fit with regularization
#
# I enforce some regularization on the solution with `regul` and I fit for a
# different reddening for the stars and the gas.

# %%
t = clock()
pp = ppxf(templates, galaxy, noise, velscale, start,
          moments=moments, degree=-1, mdegree=-1, lam=lam_gal, lam_temp=sps.lam_temp,
          regul=1/regul_err, reg_dim=reg_dim, component=component, gas_component=gas_component,
          gas_names=gas_names, reddening=0, gas_reddening=0)
print('Elapsed time in PPXF: %.2f s' % (clock() - t))

# %% [markdown]
# Plot fit results for stars and gas.

# %%
plt.figure(figsize=(15,5))
pp.plot()
plt.title(f"pPXF fit with {sps_name} SPS templates");

# %% [markdown]
# ## Plot stellar population for first regularized fit

# %%
weights = pp.weights[~gas_component]                # Exclude weights of the gas templates
weights = weights.reshape(reg_dim)/weights.sum()    # Normalized

# %% [markdown]
# Compute the light-weighted stellar population parameters and the stellar M/L.

# %%
sps.mean_age_metal(weights)
sps.mass_to_light(weights, band="SDSS/r", redshift=redshift);

# %% [markdown]
# Plot stellar population mass fraction distribution

# %%
plt.figure(figsize=(9,3))
sps.plot(weights)

# %% [markdown]
# ## Start bootstrapping
#
# IMPORTANT: I store the residuals from the first fit. I will use them to bootstrap the spectrum by 
# [reasampling residuals](https://en.wikipedia.org/wiki/Bootstrapping_(statistics)#Resampling_residuals)

# %%
bestfit0 = pp.bestfit.copy()
resid = galaxy - bestfit0
start = pp.sol.copy()

# %% [markdown]
# Here below I start the bootstrapping loop. All `pPXF` parameters are the same
# as before, with the exception that now I don't include regularization.

# %%
np.random.seed(123)

plt.subplots(3, 3, sharex=True, sharey=True, figsize=(5, 2))
plt.subplots_adjust(hspace=0.04)
plt.subplots_adjust(wspace=0.02)

nrand = 9
weights_array = np.empty((nrand, pp.weights.size))
for j in range(nrand):

    galaxy1 = bootstrap_residuals(bestfit0, resid)

    t = clock()
    pp = ppxf(templates, galaxy1, noise, velscale, start,
              moments=moments, degree=-1, mdegree=-1, lam=lam_gal, lam_temp=sps.lam_temp,
              component=component, gas_component=gas_component, gas_names=gas_names,
              reddening=0, gas_reddening=0, quiet=1)
    print(f"{j + 1}/{nrand}: Elapsed time in pPXF: {clock() - t:.2f} s")

    weights_array[j] = pp.weights

    # Plot stellar population mass fraction distribution
    weights = pp.weights[~gas_component]                # Exclude weights of the gas templates
    weights = weights.reshape(reg_dim)/weights.sum()    # Normalized

    plt.subplot(3, 3, j + 1)
    ylabel="[M/H]" if j in [0, 3] else ""
    sps.plot(weights, colorbar=False, title="", ylabel=ylabel)

pp.weights = weights_array.sum(0)
weights_err = weights_array.std(0)

# %% [markdown]
# Plot the average stellar population mass fraction distribution and uncertainties.
# See an example of application of this bootstrapping approach with pPXF in 
# [Kacharov et al. (2018)](https://ui.adsabs.harvard.edu/abs/2018MNRAS.480.1973K)

# %%
weights = pp.weights[~gas_component]  # Exclude weights of the gas templates
weights = weights.reshape(reg_dim)/weights.sum()  # Normalized

plt.figure(figsize=(9,3))
sps.plot(weights)
plt.title("Averaged Weights Fraction");
plt.tight_layout()

# %%
weights_err = weights_err[~gas_component]  # Exclude weights of the gas templates
weights_err = weights_err.reshape(reg_dim)/weights_err.sum()  # Normalized

plt.figure(figsize=(9,3))
sps.plot(weights_err)
plt.title("Weights Standard Deviation")
plt.pause(5);
