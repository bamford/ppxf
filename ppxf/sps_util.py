###############################################################################
#
# Copyright (C) 2016-2024, Michele Cappellari
# E-mail: michele.cappellari_at_physics.ox.ac.uk
#
# This software is provided as is without any warranty whatsoever.
# Permission to use, for non-commercial purposes is granted.
# Permission to modify for personal or internal use is granted,
# provided this copyright and disclaimer are included unchanged
# at the beginning of the file. All other rights are reserved.
#
###############################################################################

from pathlib import Path
import numpy as np

from ppxf import ppxf_util as util

###############################################################################

class sps_lib:
    """
    This class is meant as an example that can be easily adapted by the users
    to deal with other spectral templates libraries, different IMFs or
    different chemical abundances.

    Parameters
    ----------
    filename : string
        Name of a Numpy np.savez() file containing the following arrays for a
        given SPS models library, like FSPS, Miles, GALEXEV, BPASS, XSL,...

        1. templates[npixels, n_ages, n_metals] SPS spectra in units of L_Sun/A
           (solar luminosities per Angstrom)
        2. lam[npixels] Wavelength in Angstroms in common to all the spectra
           (can be non-uniform)
        3. fwhm[npixels] vector or scalar in Angstroms, for the instrumental
           line-spread function at every wavelength
        4. ages[n_ages] for the SPS spectra along the 2nd dimension. These are
           typically logarithmically spaced, but other choices are possible.
        5. metals[n_metals] for the SPS spectra along the 3nd dimension
        6. masses[n_ages, n_metals] mass in solar masses of living stars +
           remnants for each SPS

        This file can be created with a command like::

            np.savez_compressed(filename, templates=templates, masses=masses, 
                                lam=lam, ages=ages, metals=metals, fwhm=fwhm)

    velscale : float
        desired velocity scale for the output templates library in km/s 
        (e.g. 60). This is generally the same or an integer fraction of the 
        ``velscale`` of the galaxy spectrum used as input to ``ppxf``.
    FWHM_gal : float or dictionary
        scalar with the constant resolution of the galaxy in Angstroms, or
        dictionary with the wavelength and instrumental resolution of every
        pixel of the galaxy spectrum in Angstroms ``{"lam":lam, "fwhm":fwhm}``.
        
        If ``FWHM_gal=None`` (default), no convolution is performed.

    Other Parameters
    ----------------
    age_range : array_like with shape (2,), optional
        ``[age_min, age_max]`` optional age range (inclusive) in Gyr for the 
        SPS models. This can be useful e.g. to limit the age of the templates 
        to be younger than the age of the Universe at a given redshift.
    metal_range : array_like with shape (2,), optional
        ``[metal_min, metal_max]`` optional metallicity [M/H] range (inclusive) 
        for the SPS models (e.g.`` metal_range = [0, np.inf]`` to select only
        the spectra with Solar metallicity and above).
    norm_range : array_like with shape (2,), optional
        A two-elements vector specifying the wavelength range in Angstroms 
        within which to compute the templates normalization
        (e.g. ``norm_range=[5070, 5950]`` for the FWHM of the V-band). In this
        case, the output weights will represent light weights.

        If ``norm_range=None`` (default), the templates are not normalized
        individually, but instead are all normalized by the same scalar, given
        by the median of all templates. In this case, the output weights will
        represent mass weights.

    norm_type : {'mean', 'max', 'lbol'}, optional
        * 'mean': the templates are normalized to ``np.mean(template[band]) = 1``
          in the given ``norm_range`` wavelength range. When this keyword is
          used, ``ppxf`` will output light weights, and ``mean_age_metal()``
          will provide light-weighted stellar population quantities.
        * 'max':  the templates are normalized to ``np.max(template[band]) = 1``.
        * 'lbol':  the templates are normalized to ``lbol(template[band]) = 1``,
          where ``lbol`` is the integrated luminosity in the given wavelength
          range. If ``norm_range=[-np.inf, np.inf]`` and the templates extend
          over a wide wavelength range, the normalization approximates the
          true bolometric luminosity.

        One can use the output attribute ``.flux`` to convert light-normalized
        weights into mass weights, without repeating the ``ppxf`` fit.
        However, when using regularization in ``ppxf`` the results will not
        be identical. In fact, enforcing smoothness to the light-weights is
        not quite the same as enforcing it to the mass-weights.
    lam_range : array_like with shape (2,), optional
        A two-elements vector specifying the wavelength range in Angstroms for
        which to extract the stellar templates. Restricting the wavelength
        range of the templates to the range of the galaxy data is useful to
        save some computational time. By default ``lam_range=None``

    Returns
    -------
    Stored as attributes of the ``sps_lib`` class:

    .ages_grid : array_like with shape (n_ages, n_metals)
        Age in Gyr of every template.
    .flux : array_like with shape (n_ages, n_metals)
        If ``norm_range is not None`` then ``.flux`` contains the mean flux
        in each template spectrum within ``norm_range`` before normalization.

        When using the ``norm_range`` keyword, the weights returned by 
        ``ppxf`` represent light contributed by each SPS population template.
        One can then use this ``.flux`` attribute to convert the light weights
        into fractional masses as follows::

            from ppxf.ppxf import ppxf
            import ppxf.sps_util as lib

            sps = lib.sps_lib(...)
            pp = ppxf(...)                                  # Perform the ppxf fit
            light_weights = pp.weights[~gas_component]      # Exclude gas templates weights
            light_weights = light_weights.reshape(reg_dim)  # Reshape to a 2D matrix
            mass_weights = light_weights/sps.flux           # Divide by .flux attribute
            mass_weights /= mass_weights.sum()              # Normalize to sum=1

    .templates : array_like with shape (npixels, n_ages, n_metals)
        Logarithmically sampled array with the spectral templates in Lsun/A.
    .lam_temp : array_like with shape (npixels,)
        Wavelength in Angstroms of every pixel of the output templates.
    .ln_lam_temp : array_like with shape (npixels,)
        Natural logarithm of `.lam_temp`.
    .metals_grid : array_like with shape (n_ages, n_metals)
        Metallicity [M/H] of every template.
    .n_ages : 
        Number of different ages.
    .n_metal : 
        Number of different metallicities.

    """

    def __init__(self, filename, velscale, fwhm_gal=None, age_range=None, lam_range=None,
                 metal_range=None, norm_range=None, norm_type='mean'):

        assert norm_type in ['max', 'lbol', 'mean'], "`norm_type` must be in ['max', 'lbol', 'mean']"

        a = np.load(filename)
        spectra, masses, ages, metals, lam, fwhm_tem = \
            a["templates"], a["masses"], a["ages"], a["metals"], a["lam"], a["fwhm"]

        assert len(lam) == len(fwhm_tem) == len(spectra), \
            "`lam`, `fwhm` and `templates` must have the same length"
        assert masses.shape == spectra.shape[1:] == (ages.size, metals.size), \
            "must be masses.shape == spectra.shape[1:] == (ages.size, metals.size)"

        metal_grid, age_grid = np.meshgrid(metals, ages)

        if fwhm_gal is not None:

            if isinstance(fwhm_gal, dict):
                # Computes the spectral resolution of the galaxy at each pixel of
                # the templates. The resolution is interpolated from the galaxy
                # spectrum within its range, and constant outside its range.
                fwhm_gal = np.interp(lam, fwhm_gal["lam"], fwhm_gal["fwhm"])

            fwhm_diff2 = (fwhm_gal**2 - fwhm_tem**2).clip(0)  # NB: clip if fwhm_tem too large!
            sigma = np.sqrt(fwhm_diff2)/np.sqrt(4*np.log(4))
            spectra = util.varsmooth(lam, spectra, sigma)

        templates, ln_lam_temp = util.log_rebin(lam, spectra, velscale=velscale)[:2]
        lam_temp = np.exp(ln_lam_temp)

        if norm_range is None:
            flux = np.median(templates[templates > 0])  # Single factor for all templates
            flux = np.full(templates.shape[1:], flux)
        else:
            assert len(norm_range) == 2, 'norm_range must have two elements [lam_min, lam_max]'
            band = (norm_range[0] <= lam_temp) & (lam_temp <= norm_range[1])
            if norm_type == 'mean':
                flux = templates[band].mean(0)          # Different factor for every template
            elif norm_type == 'max':
                flux = templates[band].max(0)           # Different factor for every template
            elif norm_type == 'lbol':
                lbol = (templates[band].T @ np.gradient(lam_temp[band])).T  # Bolometric luminosity in Lsun
                flux = lbol*(templates[band].mean()/lbol.mean())            # Make overall mean level ~1
        templates /= flux

        if age_range is not None:
            w = (age_range[0] <= ages) & (ages <= age_range[1])
            templates = templates[:, w, :]
            age_grid = age_grid[w, :]
            metal_grid = metal_grid[w, :]
            flux = flux[w, :]
            masses = masses[w, :]

        if metal_range is not None:
            w = (metal_range[0] <= metals) & (metals <= metal_range[1])
            templates = templates[:, :, w]
            age_grid = age_grid[:, w]
            metal_grid = metal_grid[:, w]
            flux = flux[:, w]
            masses = masses[:, w]

        self.templates_full = templates
        self.ln_lam_temp_full = ln_lam_temp
        self.lam_temp_full = lam_temp
        if lam_range is not None:
            good_lam = (lam_temp >= lam_range[0]) & (lam_temp <= lam_range[1])
            lam_temp = lam_temp[good_lam]
            ln_lam_temp = ln_lam_temp[good_lam]
            templates = templates[good_lam]

        self.templates = templates
        self.ln_lam_temp = ln_lam_temp
        self.lam_temp = lam_temp
        self.age_grid = age_grid    # in Gyr
        self.metal_grid = metal_grid
        self.n_ages, self.n_metals = age_grid.shape
        self.flux = flux            # factor by which each template was divided
        self.mass_no_gas_grid = masses


###############################################################################

    def plot(self, weights, nodots=False, colorbar=True, **kwargs):

        assert weights.ndim == 2, "`weights` must be 2-dim"
        assert self.age_grid.shape == self.metal_grid.shape == weights.shape, \
            "Input weight dimensions do not match"

        xgrid = np.log10(self.age_grid) + 9
        ygrid = self.metal_grid
        util.plot_weights_2d(xgrid, ygrid, weights,
                             nodots=nodots, colorbar=colorbar, **kwargs)


##############################################################################

    def mean_age_metal(self, weights, quiet=False):
        """
        Compute the weighted ages and metallicities, given the weights returned
        by pPXF. The output population will be light or mass-weighted,
        depending on whether the input is light or mass weights.
        The normalization of the weights is irrelevant as it cancels out.
        """
        assert weights.ndim == 2, "`weights` must be 2-dim"
        assert self.age_grid.shape == self.metal_grid.shape == weights.shape, \
            "Input weight dimensions do not match"

        lg_age_grid = np.log10(self.age_grid) + 9
        metal_grid = self.metal_grid

        # These are eq.(39) of Cappellari (2023)
        mean_lg_age = np.sum(weights*lg_age_grid)/np.sum(weights)
        mean_metal = np.sum(weights*metal_grid)/np.sum(weights)

        if not quiet:
            print(f'Weighted <lg_age> [yr]: {mean_lg_age:#.3g}')
            print(f'Weighted <[M/H]>: {mean_metal:#.3g}')

        return mean_lg_age, mean_metal
 

##############################################################################
#
# MODIFICATION HISTORY:
#   V1.0.0: Written. Michele Cappellari, Vicenza, 20 December 2023

    def mass_to_light(self, weights, band='SDSS/r', redshift=0, quiet=False):
        """
        This function calculates the stellar mass-to-light ratio (M*/L), in
        solar units, in a given band at a specified redshift, using the pPXF
        output weights. The M*/L accounts for both living and stellar remnants,
        but not the gas ejected during stellar evolution.

        The function accepts either light or mass weights as returned from
        pPXF. If the weights are light weights, they are automatically
        converted to mass weights using the .flux attribute of the class. The
        weights overall normalization does not affect the M*/L calculation.
        """
        assert self.templates_full.shape[1:] == weights.shape, "Input weight dimensions do not match"

        p1 = util.synthetic_photometry(self.lam_temp_full, self.templates_full, band, 
                                       redshift=redshift, quiet=True)
        dist = 3.085677581491367e+19    # 10pc in cm by definition
        p1.flux /= 4*np.pi*dist**2      # convert luminosity to observed flux/cm^2 at 10pc
        p1.flux *= 3.828e+33            # spectra are in units of Lsun (erg/s IAU 2015)

        ppxf_dir = Path(__file__).parent  # path of current file
        filename = ppxf_dir / 'sps_models' / 'spectra_sun_vega.npz'
        a = np.load(filename)           # Spectrum in cgs/A at 10pc
        p2 = util.synthetic_photometry(a["lam"], a["flux_sun"], band, 
                                       redshift=redshift, quiet=True) 

        mass_weights = weights/self.flux    # Revert possible templates normalization
        lum = p1.flux/p2.flux               # Lum in solar luminosities
        mlpop = np.sum(mass_weights*self.mass_no_gas_grid)/np.sum(weights*lum)

        if not quiet:
            print(f'(M*/L)={mlpop:#.4g} ({band} at z={redshift:#.4f})')

        return mlpop


###############################################################################
