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
# # pPXF: NIRSpec/JWST mock spectrum at redshift $z\approx3$
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
# * V1.0.0: Michele Cappellari, Oxford, 29 March 2022: Created
# * V1.1.0: MC, Oxford, 10 June 2022: Updated for `pPXF` 8.1 using the new `ppxf_util.synthetic_photometry`
# * V1.2.0: MC, Oxford, 7 September 2022: Updated for pPXF 8.2
# * V1.3.0: MC, Oxford, 28 November 2023: Updated for pPXF 9.0 using the new `sps_util`
# * V1.4.0: MC, Oxford, 28 June 2024: Updated for pPXF 9.3.0
#
# ___

# %%
from pathlib import Path
from urllib import request

import matplotlib.pyplot as plt
import numpy as np

from ppxf.ppxf import ppxf
import ppxf.ppxf_util as util
import ppxf.sps_util as lib

# %% [markdown]
# ## Read the galaxy spectrum and de-redshift it
#
# Read the input data. If the data file does not exists, download it from my GitHub page.

# %%
objfile = Path('ppxf_high_redshift_mock_spectrum.npz')
if not objfile.is_file():
    url = "https://raw.githubusercontent.com/micappe/ppxf_examples/main/" + objfile.name
    request.urlretrieve(url, objfile)

# %%
data = np.load(objfile)
lam, galaxy = data["lam"], data["galaxy"]

# %% [markdown]
# JWST/NIRSpec G235H/F170LP covers the observed wavelength range
# $\lambda=1.66–3.17$ μm with a quoted resolving power $R \approx 2700$.
#
# We can compute the spectral resolution in wavelength units assuming it is
# approximately constant. The spectral resolution FWHM is, by definition of
# resolving power, and using as reference the geometric mean of the wavelength
#
# $\Delta\lambda=\frac{\lambda}{R}\approx\frac{\sqrt{\lambda_{\rm min}\lambda_{\rm max}}}{R}$

# %%
R = 2700
FWHM_gal = 1e4*np.sqrt(1.66*3.17)/R  
print( f"FWHM_gal: {FWHM_gal:.1f} Å")   # 8.5 Angstrom  

# %% [markdown]
# It is also useful to know the instrumental dispersion in km/s
#
# $\sigma_{\rm inst}\approx\frac{c}{R\sqrt{4\ln{4}}}$

# %%
c = 299792.458                      # speed of light in km/s
sigma_inst = c/(R*2.355)
print( f"sigma_inst: {sigma_inst:.0f} km/s")   # 47 km/s

# %% [markdown]
# It is generally simpler to de-redshift the spectrum before performing the pPXF fit.
# Crucially, one has to correct the instrumental resolution in wavelength units too.

# %%
z = 3.000                       # Initial estimate of the galaxy redshift
lam /= (1 + z)               # Compute approximate restframe wavelength
FWHM_gal /= (1 + z)     # Adjust resolution in Angstrom
print(f"de-redshifted NIRSpec G235H/F170LP resolution FWHM in Å: {FWHM_gal:.1f}")

# %% [markdown]
# I assume a constant error spectrum `noise` per spectral pixel. This is often
# a good approximation and I can correct later for the scaling, after obtaining
# the residuials from the fit.

# %%
galaxy = galaxy/np.median(galaxy)       # Normalize spectrum to avoid numerical issues
noise = np.full_like(galaxy, 0.05)      # Assume constant noise per pixel here. I adopt a noise that gives chi2/DOF~1

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
sps_name = 'fsps'
# sps_name = 'galaxev'
# sps_name = 'emiles'
# sps_name = 'xsl'

# %% [markdown]
# Read SPS models file from my GitHub if not already in the pPXF package dir. I
# am not distributing the templates with pPXF anymore. The SPS model files are
# also available [this GitHub page](https://github.com/micappe/ppxf_data).

# %%
ppxf_dir = Path(lib.__file__).parent
basename = f"spectra_{sps_name}_9.0.npz"
filename = ppxf_dir / 'sps_models' / basename
if not filename.is_file():
    url = "https://raw.githubusercontent.com/micappe/ppxf_data/main/" + basename
    request.urlretrieve(url, filename)

# %% [markdown]
# The templates have a larger dispersion than the galaxy: I do **not** broaden
# the templates (i.e. I do not set the `fwhm_gal` keyword in `lib.sps_lib`).
# But I need to correct the measured velocity dispersion later.
#
# The galaxy spectrum was already log-rebinned. I measure the adopted velocity
# scale for the data.

# %%
d_ln_lam = np.log(lam[-1]/lam[0])/(lam.size - 1)  # Average ln_lam step
velscale = c*d_ln_lam                   # eq. (8) of Cappellari (2017)
print(f"Velocity scale per pixel: {velscale:.2f} km/s")

# %%
FWHM_temp = 2.51   # Resolution of E-MILES templates in the fitted range

# %% [markdown]
# The templates are normalized to the V-band using norm_range. In this way the
# weights returned by pPXF represent V-band light fractions of each SSP.
# I limit the age of the templates to the age $T\approx2.2$ Gyr of the Universe at $z=3$.

# %%
sps = lib.sps_lib(filename, velscale, norm_range=[5070, 5950], age_range=[0, 2.2])

# %% [markdown]
# The stellar templates are reshaped below into a 2-dim array with each
# spectrum as a column, however we save the original array dimensions
# ``reg_dim``, which are needed to specify the regularization dimensions

# %%
reg_dim = sps.templates.shape[1:]
stars_templates = sps.templates.reshape(sps.templates.shape[0], -1)

# %% [markdown]
# Construct a set of Gaussian emission line templates.
#
# The `emission_lines` function defines the most common lines, but additional
# lines can be included by editing the function in the file `ppxf_util.py`.

# %%
lam_range_gal = [np.min(lam), np.max(lam)]
gas_templates, gas_names, line_wave = util.emission_lines(sps.ln_lam_temp, lam_range_gal, FWHM_gal, tie_balmer=1)

# %% [markdown]
# Combines the stellar and gaseous templates into a single array. During the
# pPXF fit they will be assigned a different kinematic `component` value

# %%
templates = np.column_stack([stars_templates, gas_templates])

# %% [markdown]
# ## Setup pPXF parameters
#
# As the spectrum was deredshifted, the starting guess for the velocity becomes
# close to zero

# %%
c = 299792.458
start = [1200, 200.]     # (km/s), starting guess for [V, sigma]

# %% [markdown]
# I fit two kinematics components, one for the stars and one for the gas.
# Assign `component=0` to the stellar templates, `component=1` to the gas.

# %%
n_stars = stars_templates.shape[1]
n_gas = len(gas_names)
component = [0]*n_stars + [1]*n_gas
gas_component = np.array(component) > 0  # gas_component=True for gas templates

# %% [markdown]
# Fit (V, sig) moments=2 for both the stars and the gas

# %%
moments = [2, 2]

# %% [markdown]
# Adopt the same starting value for both the stars and the gas components

# %%
start = [start, start]

# %% [markdown]
# ## Start pPXF fit

# %%
pp = ppxf(templates, galaxy, noise, velscale, start,
          moments=moments, degree=-1, mdegree=-1, lam=lam, lam_temp=sps.lam_temp,
          reg_dim=reg_dim, component=component, gas_component=gas_component,
          reddening=0, gas_reddening=0, gas_names=gas_names)
plt.figure(figsize=(15, 5))
pp.plot()
plt.title(f"pPXF fit with {sps_name} SPS templates");

# %% [markdown]
# Zoom into the $H\gamma$, $H\beta$ and $[OIII]$ spectral region

# %%
plt.figure(figsize=(15, 5))
pp.plot(gas_clip=1)
plt.title(f"pPXF fit with {sps_name} SPS templates")
plt.xlim([0.42, 0.52]);

# %% [markdown]
# IMPORTANT: As the templates have larger instrumental dispersion than the
# galaxy spectrum, and for this reason it was not possible to match the
# resolutions of the templates before the fit, I now need to correct the fitted
# sigma by the quadratic differences in instrumental resolutions. In this case
# the correction is negligible, but in general it cannot be ignored.
#
# $\sigma_{\rm obs}^2=\sigma_\star^2 + \sigma_{\rm inst}^2$

# %%
lam_med = np.median(lam)  # Angstrom
sigma_gal = c*FWHM_gal/lam_med/2.355  # in km/s
sigma_temp = c*FWHM_temp/lam_med/2.355
sigma_obs = pp.sol[0][1]   # sigma is second element of first kinematic component
sigma_diff2 = sigma_gal**2 - sigma_temp**2   # eq. (5) of Cappellari (2017)
sigma = np.sqrt(sigma_obs**2 - sigma_diff2)
print(f"sigma stars corrected: {sigma:.0f} km/s")

# %% [markdown]
# Uncertainties on stellar kinematics.
# More accurate ones can be obtained with bootstrapping.

# %%
errors = pp.error[0]*np.sqrt(pp.chi2)      # assume the fit is good
print("Formal errors:")
print("   dV   dsigma")
print("".join("%6.2g" % f for f in errors))

# %% [markdown]
# An improved estimate of the best-fitting redshift is given by the following
# lines using equation 5 of [Cappellari
# (2023)](https://ui.adsabs.harvard.edu/abs/2023MNRAS.526.3273C).

# %%
vpec = pp.sol[0][0]                         # This is the fitted residual velocity in km/s
znew = (1 + z)*np.exp(vpec/c) - 1           # eq.(5c) Cappellari (2023)
dznew = (1 + znew)*errors[0]/c              # eq.(5d) Cappellari (2023)
print(f"Best-fitting redshift z = {znew:#.6f} +/- {dznew:#.2g}")

# %% [markdown]
# # Include the photometric measurements in the fit
#
# ## Observed galaxy photometric fluxes
#
# Mean galaxy fluxes in the photometric bands `[i, z, Y, J, H, K]`. Bluer bands
# fall outside the range of the E-MILES templates. In a realistic application,
# the fluxes will come from HST or JWST photometry. They are normalized like
# the galaxy spectrum. See
# `ppxf.ppxf_examples.ppxf_example_population_photometry` for an example of how
# to match the overall normalization of the photometric fluxes to the spectrum.

# %%
phot_galaxy = np.array([0.54,0.48, 0.44, 0.56, 1.08, 1.03])   # fluxes
phot_noise = np.full_like(phot_galaxy, np.max(phot_galaxy)*0.03)   # 1sigma uncertainties (3% of max flux)

# %% [markdown]
# ## Setup photometric templates
#
# To compute the photometric prediction I need to give a redshift estimate. In
# this way the predictions are computed on the redshifted templates.

# %%
bands = ['SDSS/i', 'SDSS/z', 'VISTA/Y', 'VISTA/J', 'VISTA/H', 'VISTA/Ks']
p2 = util.synthetic_photometry(sps.lam_temp, templates, bands, redshift=3.01666, quiet=1)
phot = {"templates": p2.flux[p2.ok], "lam": p2.lam_eff[p2.ok], "galaxy": phot_galaxy[p2.ok], "noise": phot_noise[p2.ok]}

# %% [markdown]
# ## Start pPXF fit

# %%
pp = ppxf(templates, galaxy, noise, velscale, start,
          moments=moments, degree=-1, mdegree=-1, lam=lam, lam_temp=sps.lam_temp, regul=10,
          reg_dim=reg_dim, component=component, gas_component=gas_component, gas_names=gas_names,
          reddening=0.1, gas_reddening=0.1, phot=phot)

# %%
plt.figure(figsize=(15, 5))
pp.plot()
plt.title(f"pPXF fit with {sps_name} SPS templates");

# %% [markdown]
# ## Plot of the stellar population distribution

# %%
light_weights = pp.weights[~gas_component]      # Exclude weights of the gas templates
light_weights = light_weights.reshape(reg_dim)  # Reshape to (n_ages, n_metal)
light_weights /= light_weights.sum()            # Normalize to light fractions

plt.figure(figsize=(9, 3))
sps.plot(light_weights)
plt.title("Light Weights Fractions")
plt.tight_layout()
plt.pause(5);
