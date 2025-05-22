# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: title,-all
#     formats: py:percent,ipynb
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.4
# ---

# %% [markdown]
#
# # pPXF: Fitting multiple stellar kinematic components
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
# This example shows how to fit multiple stellar components with different
# stellar population and kinematics.
#
# ### MODIFICATION HISTORY
#
# - V1.0.0: Early test version. Michele Cappellari, Oxford, 20 July 2009
# - V1.1.0: Cleaned up for the paper by Johnston et al. (MNRAS, 2013).
#       MC, Oxford, 26 January 2012
# - V2.0.0: Converted to Python and adapted to the changes in the new public
#       PPXF version, Oxford 8 January 2014
# - V2.0.1: Support both Python 2.6/2.7 and Python 3.x. MC, Oxford, 25 May 2014
# - V2.0.2: Support both Pyfits and Astropy to read FITS files.
#       MC, Oxford, 22 October 2015
# - V2.0.3: Use proper noise in input. MC, Oxford, 8 March 2016
# - V2.1.0: Replaced the Vazdekis-99 SSP models with the Vazdekis+10 ones.
#       MC, Oxford, 3 May 2016
# - V2.1.1: Make files paths relative to this file, to run the example from
#       any directory. MC, Oxford, 18 January 2017
# - V2.1.2: Updated MILES file names. MC, Oxford, 29 November 2017
# - V2.1.3: Changed imports for pPXF as a package.
#       Make file paths relative to the pPXF package to be able to run the
#       example unchanged from any directory. MC, Oxford, 17 April 2018
# - V2.1.4: Dropped legacy Python 2.7 support. MC, Oxford, 10 May 2018
# - V2.1.5: Fixed clock DeprecationWarning in Python 3.7.
#       MC, Oxford, 27 September 2018
# - V2.2.0: Illustrates the usage of the `constr_kinem` keyword.
#       MC, Oxford, 5 February 2020
# - V2.3.0: Modified usage example of the `constr_kinem` keyword.
#       MC, Oxford, 21 December 2020
# - V2.4.0: Use E-Miles spectral library. MC, Oxford, 16 March 2022
# - V2.5.0: Use the new `sps_util` instead of `miles_util`. 
#       MC, Oxford, 12 November 2023
# ___

# %%
from pathlib import Path
from time import perf_counter as clock
from urllib import request

from scipy import signal
import numpy as np
import matplotlib.pyplot as plt

from ppxf.ppxf import ppxf
import ppxf.ppxf_util as util
import ppxf.sps_util as lib

# %% [markdown]
# ## Creation of the mock spectrum

# %%
lam_range_temp = [3500, 7500]
sps_name = 'emiles'
velscale = 35   # km/s

# %% [markdown]
# Read SPS models file from my GitHub if not already in the `pPXF` package dir.
# The SPS model files are also available on my [GitHub Page](https://github.com/micappe/ppxf_data).

# %%
ppxf_dir = Path(util.__file__).parent
basename = f"spectra_{sps_name}_9.0.npz"
filename = ppxf_dir / 'sps_models' / basename
if not filename.is_file():
    url = "https://raw.githubusercontent.com/micappe/ppxf_data/main/" + basename
    request.urlretrieve(url, filename)

# %%
sps = lib.sps_lib(filename, velscale, lam_range=lam_range_temp)

# %% [markdown]
# Extract two SPS templates, one is young and the other is old

# %%
model1 = sps.templates[:, -2, -2]  # age = 12.59 Gyr [M/H] = 0
model2 = sps.templates[:, 12, -2]  # age = 1.0 Gyr [M/H] = 0
model1 /= np.median(model1)
model2 /= np.median(model2)

# %%
model = np.column_stack([model1, model2])
galaxy = np.empty_like(model)

# %% [markdown]
# These are the input values in spectral pixels for the `(V, sigma)` of the two
# kinematic components

# %%
vel = np.array([0., 300.])/velscale
sigma = np.array([200., 100.])/velscale

# %% [markdown]
# The synthetic galaxy model consists of the sum of two SSP spectra with age of
# 1Gyr and 13Gyr respectively with different velocity and dispersion

# %%
for j in range(len(vel)):
    dx = int(abs(vel[j]) + 4.*sigma[j])   # Sample the Gaussian at least to vel+4*sigma
    v = np.linspace(-dx, dx, 2*dx + 1)
    losvd = np.exp(-0.5*((v - vel[j])/sigma[j])**2) # Gaussian LOSVD
    losvd /= np.sum(losvd)      # normalize LOSVD
    galaxy[:, j] = signal.fftconvolve(model[:, j], losvd, mode="same")
    galaxy[:, j] /= np.median(model[:, j])
galaxy = np.sum(galaxy, axis=1)
sn = 100.
noise = np.full_like(galaxy, np.median(galaxy)/sn)
prng = np.random.default_rng(123)  # For reproducible results
galaxy = prng.normal(galaxy, noise) # add noise to galaxy

# %% [markdown]
# ## pPXF fitting
#
# For simplicity of illustration, I adopt two templates per kinematic
# component. In a real situation, I may determine the templates of the two
# kinematics components by first fitting the galaxy spectra in regions where I
# expect one of the two components to provide negligible contribution to the
# surface brightness.

# %%
templates = np.column_stack([model1, model2, model1, model2])
goodpixels = np.arange(100, model1.size - 100)  # Remove edge effects

# %% [markdown]
# ### Two unconstrained kinematic components
#
# With multiple stellar kinematic components a good starting velocity is
# essential. Starting too far from the solution pPXF may *not* converge to the
# global minimum. One should give different starting velocities for the two
# stellar components. In general one should explore a grid of starting
# velocities as illustrated e.g. in Sec.3.3 of 
# [Mitzkus et al. (2017)](https://ui.adsabs.harvard.edu/abs/2017MNRAS.464.4789M)
# or use the `global_search` keyword to perform global optimization and ensure
# convergence to the global minimum.

# %%
start = [[100, 200], [200, 200]]

# %%
print(f"\n{'#'*60}\n{'No constraints on the kinematics'.center(60)}\n{'-'*60}")
t = clock()
pp = ppxf(templates, galaxy, noise, velscale, start, goodpixels=goodpixels,
          degree=4, moments=[2, 2], component=[0, 0, 1, 1], 
          lam=sps.lam_temp, lam_temp=sps.lam_temp)
print(f"{'='*60}\nElapsed time in pPXF {clock() - t:.2f} s")

# %%
plt.clf()
plt.subplot(211)
pp.plot()
plt.title("Two components pPXF fit")

# %% [markdown]
# ### Two linearly-constrained kinematic components
#
# Just to illustrate how to use `constr_kinem`, here I constrain the velocity
# dispersion of the two stellar kinematic components to be within 50% of each
# other: $ 1/1.5 < \sigma_1/\sigma_0 < 1.5 $. In other words, I want to have:
#
# $$
# \begin{cases}
# \sigma_0/1.5 - \sigma_1 < 0\\
# -1.5\,\sigma_0 + \sigma_1 < 0\\
# \end{cases}
# $$
#
# Following standard practice in numerical optimization (e.g.
# [HERE](https://uk.mathworks.com/help/optim/ug/linear-constraints.html)), I
# express these two inequalities as linear constraints in matrix form as this
# leads to an efficient optimization algorithm, with guaranteed local
# convergence:
#
# $$\mathbf{A}_{\rm ineq}\cdot \mathbf{p} < \mathbf{b}_{\rm ineq}.$$
#
# In this example, the vector of nonlinear kinematic parameters is
# $\mathbf{p}=[V_0, \sigma_0, V_1, \sigma_1]$
# (following the same order of the `start` parameter of `pPXF`) and I can write
# the above matrix equation explicitly as
#
# $$
# \begin{bmatrix}
# 0 & 1/1.5 & 0 & -1\\
# 0 & -1.5 & 0 & 1
# \end{bmatrix}
# \cdot
# \begin{bmatrix}
# V_0 \\
# \sigma_0 \\
# V_1 \\
# \sigma_1
# \end{bmatrix}
# <
# \begin{bmatrix}
# 0 \\
# 0 
# \end{bmatrix}.
# $$
#
# This translates into the following Python code

# %%
A_ineq = [[0, 1/1.5, 0, -1],    # sigma0/1.5 - sigma1 <= 0
          [0,  -1.5, 0,  1]]    # -sigma0*1.5 + sigma1 <= 0
b_ineq = [0, 0]
constr_kinem = {"A_ineq": A_ineq, "b_ineq": b_ineq}

# %% [markdown]
# Perform the fit. The best fit is at the boundary of the feasible region.

# %%
print(f"\n{'#'*60}\n{'Constraint: sigma[0]/1.5 <= sigma[1] <= sigma[0]*1.5'.center(60)}\n{'-'*60}")
t = clock()
pp = ppxf(templates, galaxy, noise, velscale, start,
          goodpixels=goodpixels, degree=4, moments=[2, 2],
          component=[0, 0, 1, 1], constr_kinem=constr_kinem,
          lam=sps.lam_temp, lam_temp=sps.lam_temp)
print(f"{'='*60}\nElapsed time in pPXF {clock() - t:.2f} s")

# %% [markdown]
# ### Single kinematic component
#
# This is a fit with a single kinematic component, for reference.

# %%
start = [100, 200]
print(f"\n{'#'*60}\n{'Single component pPXF fit'.center(60)}\n{'-'*60}")
t = clock()
pp = ppxf(templates, galaxy, noise, velscale, start,
          goodpixels=goodpixels, degree=4, moments=2,
          lam=sps.lam_temp, lam_temp=sps.lam_temp)
print(f"{'='*60}\nElapsed time in pPXF {clock() - t:.2f} s")

# %%
plt.subplot(212)
pp.plot()
plt.title("Single component pPXF fit")
plt.tight_layout()
plt.pause(5);
