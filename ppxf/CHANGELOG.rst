
Changelog
---------

V9.4.2: MC, Oxford, 12 March 2025
+++++++++++++++++++++++++++++++++

- ``ppxf``: Added a new boolean keyword ``fit`` to skip the fit and only
  initialize the class. When ``fit=False``, the ``pp.linear_fit`` method also
  skips the linear fit, only setting the ``pp.bestfit`` attribute and returning
  the residuals. In this scenario, the user must define all class parameters
  and attributes manually.

V9.4.1: MC, Oxford, 05 August 2024
++++++++++++++++++++++++++++++++++

- Removed ``capfit`` from the ``pPXF`` package and made it a separate package.
- ``sps_util.sps_lib``: Renamed ``wave_range`` to ``lam_range`` for consistency
  with other procedures in the ``pPXF`` package.
- ``ppxf_util.emission_lines``: Allow for ``FWHM_gal`` to be a dictionary, to
  be consistent with ``sps_util.sps_lib``.
- Removed the NGC4636 spectrum from the examples and updated the other spectra
  to the latest SDSS data release, which includes the instrumental dispersion
  per pixel. Illustrated how to deal with a variable instrumental dispersion in
  the examples.

V9.3.0: MC, Oxford, 28 June 2024
++++++++++++++++++++++++++++++++

- Converted all ``pPXF`` Python examples to ``py:percent`` format, which allows
  one to open them as Jupyter Notebooks in most Python IDEs. Also added
  several of my public Jupyter Notebook examples as ``py:percent`` Python files
  in the ``ppxf/examples`` folder.
- Updated examples and associated documentation to demonstrate the use of the
  X-Shooter Stellar Library (XSL) SPS models (Verro+2022) with ``pPXF``.
  The XSL template spectra can now be automatically retrieved from GitHub by
  specifying ``sps_name = 'xsl'`` within the examples code.
- ``sps_util.synthetic_photometry``: Swapped order of input parameters from
  ``(spectrum, lam)`` to ``(lam, spectrum)`` for consistency with other
  ``pPXF`` procedures. Updated corresponding examples.
- ``sps_util.sps_lib``: Removed the warning regarding the template having a
  lower resolution than the galaxy. This is because it could lead to false
  positives, considering that the galaxy's fitted spectral range is not
  generally known during template generation.

V9.2.2: MC, Oxford, 20 May 2024
+++++++++++++++++++++++++++++++

- ``ppxf``: Fixed an issue where the absence of the ``reg_dim`` keyword could
  halt the program (bug from V9.2.1). Thanks to Jesse van de Sande
  (unsw.edu.au) and Luiz A. Silva-Lima (Univ. Sao Paulo, Brazil) for the
  report.

- ``ppxf``: Avoid possible program stop when ``templates`` are shorter than
  ``galaxy`` (bug from V9.2.1).

- ``ppxf``: Avoid truncating ``templates`` to ``goodpixels`` in the final plot,
  when both ``lam`` and ``lam_temp`` parameters are supplied. Instead, show the
  ``templates`` over the full ``galaxy`` spectral range.
  
- ``ppxf``: Corrected behaviour to only use the range between the first and
  last ``goodpixels`` for determining the vertical plotting range.  Thanks to
  David Gooding (chch.ox.ac.uk) for illustrating the problem.

- ``ppxf_example_kinematics_sdss.py``: Modified the example to de-redshift the
  spectrum by default, minimizing potential confusion.

V9.2.1: MC, Oxford, 18 April 2024
+++++++++++++++++++++++++++++++++

- ``sps_util.synthetic_photometry``: Accepts input wavelengths that are not
  evenly spaced.
- ``ppxf_util.mag_spectrum``: A new function to get the apparent magnitude from
  a spectrum in any photometric band, at any redshift, in either the AB or Vega
  magnitude system.
- ``ppxf``: Now considers the given ``goodpixels`` when checking that the
  ``templates`` cover the full ``galaxy`` spectrum, if both ``lam`` and
  ``lam_temp`` are provided.
- ``ppxf``: Allow for variable step size across intervals and dimensions in the
  numerical derivatives used for regularization, with the new keyword
  ``reg_step``.
- ``ppxf``: Avoids new ``SyntaxWarning`` in Matplotlib LaTeX string in the
  latest Python 3.12.
- ``ppxf``: Avoid program stop when passing an input covariance matrix while
  fitting for gas emission lines. Thanks to Jackson ODonnell (ucsc.edu) for the
  report.

V9.1.1: MC, Oxford, 18 January 2024
+++++++++++++++++++++++++++++++++++

- ``sps_util.mass_to_light``: Now it can calculate the stellar mass-to-light
  ratio (``M*/L``) for any stellar population synthesis (SPS) model, any
  filter, and any redshift, using the output weights from ``pPXF``. No need for
  pre-computed tables anymore.
- ``ppxf_util.mag_sun``: A new function to get the absolute solar magnitude in
  any photometric band, at any redshift, in either the AB or Vega magnitude
  system.

V9.0.2: MC, Oxford, 30 November 2023
++++++++++++++++++++++++++++++++++++

- ``ppxf``: Fixed bug in the automatic truncation of the template wavelength,
  when using ``bounds`` together with ``lam`` and ``lam_temp``.

V9.0.1: MC, Oxford, 12 November 2023
++++++++++++++++++++++++++++++++++++

- Replaced ``miles_util`` with ``sps_util``, a versatile and flexible function
  that supports various stellar population synthesis (SPS) models. Currently, I
  distribute (with permission) the ``E-MILES`` (Vazdekis+2016), ``GALAXEV``
  (Bruzual & Charlot 2003) and ``fsps`` (Conroy+2009,2010) stellar population
  templates. One can now change SPS by just modifying the filename.
  
  IMPORTANT: I no longer include the SPS models in the ``pPXF`` package, due to
  file size constraints. One must download the SPS templates separately after
  the ``pPXF`` installation, as demonstrated in all the ``pPXF`` examples.
- I adapted all examples included in the package to use the new ``sps_util``. 
- ``ppxf.plot``: New keyword ``lam_flam`` to plot ``lam*f(lam) = nu*f(nu)``.
- ``ppxf_util.synthetic_photometry``: Converted from a function to a class.
  Added ``.lam_piv`` attribute to compute the pivot wavelength of each filter.
- ``ppxf_util.varsmooth``: Specialized faster convolution if sigma is a scalar
  while using the analytic Fourier transform to deal with undersampling.
- ``ppxf``: Raised an error if ``phot`` input is not finite.

V8.2.6: MC, Oxford, 6 July 2023
+++++++++++++++++++++++++++++++

- ``capfit``: Relaxed tolerance when checking initial guess feasibility.
- ``ppxf_util``: Fixed program stop in ``gaussian_filter1d`` when ``sig=0``.
  Thanks to Jesse Van de Sande (sydney.edu.au) for the report.

V8.2.4: MC, Oxford, 12 May 2023
+++++++++++++++++++++++++++++++

- ``ppxf``: Fixed incorrectly switching to ``method='linear'`` when requesting
  to fit stellar reddening alone, while all other parameters are fixed. Thanks
  to Jong Chul Lee (kasi.re.kr) for a clear example of reproducing the bug.

V8.2.3: MC, Oxford, 5 January 2023
++++++++++++++++++++++++++++++++++

- ``ppxf``: Fixed program stop when fitting both sky spectra and gas emission
  lines. Many thanks to Adarsh Ranjan (kasi.re.kr) for a clear example
  of reproducing the problem and for the bug fix.
- ``ppxf``: Raise an error if ``velscale`` is not a scalar. Updated example
  procedures. This fixes a program stop due to a change in NumPy 1.24.
- ``ppxf_util``: Changed ``log_rebin`` to return ``velscale`` as a scalar.
- New example procedure ``ppxf_example_gas_sdss_tied.py`` to illustrate the
  use of the ``tied`` and ``constr_kinem`` keywords.

V8.2.2: MC, Oxford, 11 October 2022
+++++++++++++++++++++++++++++++++++

- ``ppxf``: Fixed program stop with ``tied`` or ``fixed`` keywords and
  nonlinear variables in addition to the kinematics. Thanks to Tobias Looser
  (cam.ac.uk) for the report and fix.

V8.2.1: MC, Oxford, 3 September 2022
++++++++++++++++++++++++++++++++++++

- ``ppxf``: New keyword ``dust`` which allows one to associate different
  general attenuation functions to different sets of templates. This is useful
  for example to apply three different attenuation functions to the young and
  old stellar templates and to the gas emission lines respectively.
- ``ppxf``: Uses ``A_V`` instead of ``E(B-V)`` to parametrize attenuation, when
  using the now-obsolete keywords ``reddening`` and  ``gas_reddening``.
- ``ppxf``: New default function ``attenuation()``. However, one can now use
  general attenuation functions with an arbitrary number of bound or fixed
  parameters.
- ``ppxf``: New internal functions ``set_lam_input``, ``set_gas_input``,
  ``set_dust_input``, ``set_phot_input`` to organize the code.
- ``ppxf``: Improved vertical scaling of default plots.
- ``ppxf``: New keywords ``pp.plot(spec=True, phot=True)`` to plot only the
  photometric or spectroscopic best fits respectively.
- ``ppxf_util``: New function ``varsmooth`` for Gaussian convolution with a
  variable sigma. Unlike the similar ``gaussian_filter1d``, this new function
  uses FFT and the analytic Fourier Transform of a Gaussian, like ``ppxf``.
- ``ppxf_util``: Included additional gas emission lines in ``emission_lines()``.
- ``capfit``: Use ``scipy.optimize.linprog(method='highs')`` to find feasible
  starting point in ``lsq_lin``. This eliminates possible program stops in
  certain situations with linearly dependent constraints.
- ``capfit``: Set default ``linear_method='lsq_lin'``. This removes the need
  to install ``cvxopt`` when using ``constr_kinem`` in ``pPXF``.

V8.1.0: MC, Oxford, 10 June 2022
++++++++++++++++++++++++++++++++

- ``ppxf``: More accurate determination of the range for truncating the
  templates when passing both ``lam_temp`` and ``lam``.
- ``ppxf``: Check for ``lam`` or ``lam_temp`` consistency with ``velscale`` and
  return an error if they do not match.
- ``ppxf``: Use micrometre units and denser tick labels for the logarithmic
  wavelength axis.
- ``ppxf_util.synthetic_photometry``: moved from ``miles_util`` and made it
  independent of the stellar library. Adopted the same filter file format as
  EAZY, FAST, HyperZ... for interoperability.
  Allow passing a file with user-defined filter response functions.
- ``ppxf_util.log_rebin``: Support irregularly sampled input wavelength.
- ``ppxf_util.gaussian_filter1d``: New keyword ``mode='constant'`` or
  ``mode='wrap'``.
- Updated ``ppxf_example_population_photometry.py``

V8.0.2: MC, Oxford, 28 March 2022
+++++++++++++++++++++++++++++++++

- ``ppxf``: Allow fitting photometric measurements (SED fitting) together with
  a spectrum. This is implemented via the new keyword ``phot`` passing a
  dictionary of parameters.
- ``ppxf``: plot photometric fit together with spectrum when fitting
  photometry.
- ``ppxf``: New keyword ``lam_temp`` to input the templates wavelength. When
  this is given, together with the galaxy wavelength ``lam``, the templates are
  automatically truncated to an optimal wavelength range, and it becomes
  unnecessary to use the keyword ``vsyst``.
- ``ppxf``: Warning if ``templates`` are ``> 2x`` longer than ``galaxy``.
- ``ppxf``: When fitting photometry one can input extended template spectra to
  overplot the extrapolated best-fit spectrum together with the photometry.
- New demo file on photometric fitting
  ``ppxf_example_population_photometry.py``.
- ``miles_util.photometry_from_table``: New example function to illustrate the
  generation of the input photometric templates for the ``phot`` keyword, using
  tabulated SSP model magnitudes.
- ``miles_util.photometry_from_spectra``: New example function to illustrate
  the generation of photometric templates from the spectra using filter
  responses.
- Replaced MILES spectral models of Vazdekis et al. (2010) with E-MILES models
  of Vazdekis et al. (2016). Thanks to Alexandre Vazdekis (iac.es) for the
  permission.
- Adapted all ``pPXF`` examples to use the E-MILES templates.
- ``miles_util.miles``: changed names of output wavelength ``.ln_lam_temp`` to
  make clear they represent natural logarithms.
- ``miles_util.miles``: set ``FWHM_gal=None`` to skip templates convolution.
- ``ppxf``: Optionally performs global optimization of the non-linear
  parameters. This is implemented via the new keyword ``global_search``.
- ``ppxf``: Allow the use of multiplicative polynomials together with
  reddening.
- ``ppxf``: Plot individual gas emission components in addition to their sum.
- ``ppxf``: Updated docstring documentation for the new features.
- ``capfit``: Completely removed tied/fixed variables from the optimization and
  constraints. This improves the conditioning of the Jacobian and further
  strengthens the robustness of the optimization.
- ``miles_util``: fixed ``flux`` array mismatch when using ``age_range`` or
  ``metal_range``. Thanks to Davide Bevacqua (inaf.it) for the report.
- ``ppxf``: Fixed program stop when fitting gas with a template length that is
  not a multiple of ``velscale_ratio``.

V7.4.5: MC, Oxford, 16 July 2021
++++++++++++++++++++++++++++++++

- ``ppxf``: New keyword ``pp.plot(clip_gas=True)`` to ignore the gas emission
  lines while determining the plotting ranges for the best-fitting model.
- ``miles_util``: New attribute ``.flux`` to convert between light-weighted
  and mass-weighted stellar population quantities. Updated the corresponding
  documentation in the docstring.
- ``ppxf_example_population_gas_sdss``: Show how to convert between light-weighted
  and mass-weighted stellar population using the new ``miles.flux`` attribute.
- ``ppxf_util.log_rebin``: support fast log rebinning of all columns of 2-dim arrays.

V7.4.4: MC, Oxford, 10 February 2021
++++++++++++++++++++++++++++++++++++

- ``ppxf``: More robust matrix scaling when using linear equality constraints
  in ``constr_templ`` with ``linear_method='lsq_box'``. Thanks to Shravan Shetty
  (pku.edu.cn) for a detailed report and for testing my fix.

V7.4.3: MC, Oxford, 21 December 2020
++++++++++++++++++++++++++++++++++++

- ``capfit``: New ``linear_method`` keyword to select between ``cvxopt`` or
  ``lsq_lin``, when using linear constraints, for cases where the latter stops.
  The ``cvxopt`` package must be installed when setting that option.
- ``ppxf``: Adapted to use ``capfit`` with ``linear_method='cvxopt'`` when
  enforcing linear constraints on the kinematics with ``constr_kinem``.
- ``ppxf``: Included NOTE in the documentation of ``constr_kinem``.
  All changes above were after detailed reports by Kyle Westfall (ucolick.org).

V7.4.2: MC, Oxford, 9 October 2020
++++++++++++++++++++++++++++++++++

- ``ppxf``: Corrected typo in example in the documentation of ``constr_templ``.
- ``ppxf``: Check that ``constr_templ`` and ``constr_kinem`` are dictionaries.
  Thanks to Davide Bevacqua (unibo.it) for the feedback.

V7.4.1: MC, Oxford, 11 September 2020
+++++++++++++++++++++++++++++++++++++

- ``capfit``: Fixed possible infinite loop in ``lsq_box`` and ``lsq_lin``.
  Thanks to Shravan Shetty (pku.edu.cn) for the detailed report and to both
  him and Kyle Westfall (ucolick.org) for testing the fix.
- ``capfit``: Use NumPy rather than the SciPy version of ``linalg.lstsq`` to
  avoid a current SciPy bug in the default criterion for rank deficiency.
- ``capfit``: Renamed ``cond`` keyword to ``rcond`` for consistency with NumPy.
- ``capfit``: Passed ``rcond`` keyword to ``cov_err`` function.
- ``ppxf``: removed ``rcond`` keyword in ``capfit`` call. Use default instead.

V7.4.0: MC, Oxford, 20 August 2020
++++++++++++++++++++++++++++++++++

- ``capfit``: New function ``lsq_lin`` implementing a linear least-squares
  linearly constrained algorithm supporting rank-deficient matrices and allowing
  for a starting guess.
- ``capfit``: Removed the ``lsqlin`` procedure which is superseded by ``lsq_lin``.
- ``capfit``: Renamed ``lsqbox`` to ``lsq_box`` and revised its interface.
- ``ppxf``: Modified to use the new ``lsq_lin`` and the updated ``lsq_box`` functions.
- ``ppxf``: More examples for the ``constr_templ`` and ``constr_kinem`` keywords.
- Set redshift ``z = 0`` when one uncomments the lines to bring the spectrum to
  the rest-frame in ``ppxf_example_kinematics_sdss.py``. Thanks to
  Vaidehi S. Paliya (desy.de) for pointing out the inconsistency in my example.

V7.3.0: MC, Oxford, 10 July 2020
++++++++++++++++++++++++++++++++

- ``capfit``: New function ``lsqbox`` implementing a fast linear least-squares
  box-constrained (bounds) algorithm which allows for a starting guess.
  While testing I also discovered a major mistake in the current implementation
  of ``scipy.optimize.lsq_linear`` (my fix was later included in Scipy 1.6).
- ``ppxf``: The new ``linear_method='lsqbox'`` and ``linear_method='cvxopt'``
  now use an initial guess for the solution, which significantly speeds up the
  kinematic fit with multiple templates. As an example, my procedure
  ``ppxf_example_population_gas_sdss`` is now about 4 times faster with the new
  ``linear_method='lsqbox'`` than with the legacy ``linear_method='nnls'``.
- ``ppxf``: Added support for linear equality constraints on the templates
  ``constr_templ`` and for using the keyword ``fraction`` with both
  ``linear_method='lsqbox'`` and ``linear_method='nnls'``.
- Print ``degree`` and ``mdegree`` with the final results.
- Set ``linear=True`` automatically if the fit has no free non-linear parameters,
  to avoid a program stop. Thanks to Shravan Shetty (pku.edu.cn) for the report.

V7.2.1: MC, Oxford, 12 June 2020
++++++++++++++++++++++++++++++++

- ``capfit``: New input keyword ``cond`` for Jacobian rank tolerance.
- ``capfit``: Use ``bvls`` to solve quadratic subproblem with only ``bounds``.
- ``ppxf``: Set ``cond=1e-7`` in ``capfit`` call, when using linear constraints.
  The ``capfit`` related changes were due to detailed feedback by Kyle Westfall
  (ucolick.org), to deal with situations with degenerate Jacobians, like when
  there is no stellar continuum and one uses multiplicative polynomials.
- ``ppxf``: Clarified documentation for ``.gas_zero_template`` and the
  corresponding warning message, after feedback by Laura Salo (umn.edu).

V7.2.0: MC, Oxford, 4 May 2020
++++++++++++++++++++++++++++++

- Allow for ``linear_method='cvxopt'`` when the optional ``cvxopt`` package
  is installed.

V7.1.0: MC, Oxford, 30 April 2020
+++++++++++++++++++++++++++++++++

- Introduced new ``ppxf`` keyword ``linear_method``, and corresponding changes
  in the code, to select between the old ('nnls') and the new ('lsqlin')
  approach to the solution of the linear least-squares subproblem in ``ppxf``.
  Thanks to Sam Vaughan (sydney.edu.au) for a convincing minimal example
  illustrating the usefulness of this keyword.

V7.0.1: MC, Oxford, 8 April 2020
++++++++++++++++++++++++++++++++

- Support ``.gas_zero_template`` and ``fraction`` together with other
  equality constraints.
- Included ``np.pad(...mode='constant')`` for backward compatibility with
  Numpy 1.16. Thanks to Shravan Shetty (KIAA-PKU) for the suggestion.
- Fix ``rebin()`` not retaining the dimensionality of an input column-vector.
  This resulted in a program stop with a single gas template and
  ``velscale_ratio > 1``. Thanks to Zhiyuan Ji (astro.umass.edu) for a clear
  example reproducing the bug.
- ``capfit``: New keyword ``cond`` for ``lsqlin``.
- ``capfit``: Relaxed assertion for inconsistent inequalities in ``lsqlin``
  to avoid false positives. Thanks to Kyle Westfall (UCO Lick) for a detailed
  bug report.

V7.0.0: MC, Oxford, 10 January 2020
+++++++++++++++++++++++++++++++++++

- ``capfit``: New general linear least-squares optimization function
  ``lsqlin`` which is now used to solve the quadratic subproblem.
- ``capfit``: Allow for linear inequality/equality constraints
  ``A_ineq``, ``b_ineq`` and  ``A_eq``, ``b_eq``.
- ``ppxf``: Use (faster) ``capfit.lsqlin`` for the linear fit.
- ``ppxf``: Use updated ``capfit.capfit`` for the non-linear optimization.
- ``ppxf``: Allow for linear equalities/inequalities for both the template
  weights and the kinematic parameters with the ``constr_templ`` and
  ``constr_kinem`` optional keywords.
- ``ppxf``: New ``set_linear_constraints`` function.
- ``ppxf``: Updated documentation.

V6.7.17: MC, Oxford, 14 November 2019
+++++++++++++++++++++++++++++++++++++

- ``capfit``: Written complete documentation.
- ``capfit``: Improved print formatting.
- ``capfit``: Return ``.message`` attribute.
- ``capfit``: Improved ``xtol`` convergence test.
- ``capfit``: Only accept final move if ``chi2`` decreased.
- ``capfit``: Strictly satisfy bounds during Jacobian computation.

V6.7.16: MC, Oxford, 12 June 2019
+++++++++++++++++++++++++++++++++

- ``capfit``: Use only free parameters for ``xtol`` convergence test.
- ``capfit``: Describe in words convergence status with nonzero ``verbose``.
- ``capfit``: Fixed program stop when ``abs_step`` is undefined.
- ``capfit``: Fixed ignoring optional ``max_nfev``.

V6.7.15: MC, Oxford, 7 February 2019
++++++++++++++++++++++++++++++++++++
- Removed unused ``re`` import.
- Removed Scipy's ``next_fast_len`` usage due to an issue with odd padding size.
  Thanks to Eric Emsellem (ESO) for a clear example illustrating this rare and
  subtle bug.

V6.7.14: MC, Oxford, 27 November 2018
++++++++++++++++++++++++++++++++++++++
- Print the used ``tied`` parameters equalities, if any.
- Return ``.ndof`` attribute.
- Do not remove ``fixed`` or ``tied`` parameters from the DOF calculation.
  Thanks to Joanna Woo (Univ. of Victoria) for the correction.
- Replaced ``normalize``, ``min_age``, ``max_age`` and ``metal`` keywords with
  ``norm_range``, ``age_range`` and ``metal_range`` in ``ppxf.miles_util.miles``.
- Fixed ``clock`` ``DeprecationWarning`` in Python 3.7.

V6.7.13: MC, Oxford, 20 September 2018
++++++++++++++++++++++++++++++++++++++
- Expanded documentation of ``reddening`` and ``gas_reddening``.
  Thanks to Nick Boardman (Univ. Utah) for the feedback.
- ``capfit`` now raises an error if one tries to tie parameters to themselves.
  Thanks to Kyle Westfall (Univ. Santa Cruz) for the suggestion.
- ``capfit`` uses Python 3.6 f-strings.

V6.7.12: MC, Oxford, 9 July 2018
++++++++++++++++++++++++++++++++
- Allow for ``velscale`` and ``vsyst`` to be Numpy arrays rather than scalars.
- Improved criterion for when the Balmer series is within the fitted wavelength
  range in ``ppxf.ppxf_util.emission_lines``. Thanks to Sam Vaughan
  (Univ. of Oxford) for the feedback.
- Included ``width`` keyword in ``ppxf.ppxf_util.determine_goodpixels``.
  Thanks to George Privon (Univ. of Florida) for the suggestion.
- Expanded ``.gas_flux`` documentation.

V6.7.11: MC, Oxford, 5 June 2018
++++++++++++++++++++++++++++++++

- Formatted ``ppxf.py`` docstring in reStructuredText.
- Removed CHANGELOG from the code and placed it in a separate file.
- Modified ``setup.py`` to show help and CHANGELOG on PyPi page.
- Included ``ppxf.__version__``.

V6.7.8: MC, Oxford, 21 May 2018
+++++++++++++++++++++++++++++++

- Moved package to the Python Package Index (PyPi).
- Dropped legacy Python 2.7 support.

V6.7.6: MC, Oxford, 16 April 2018
+++++++++++++++++++++++++++++++++

- Changed imports for the conversion of pPXF to a package.
  Thanks to Joe Burchett (Santa Cruz) for the suggestion.

V6.7.5: MC, Oxford, 10 April 2018
+++++++++++++++++++++++++++++++++

- Fixed syntax error under Python 2.7.

V6.7.4: MC, Oxford, 16 February 2018
++++++++++++++++++++++++++++++++++++

- Fixed bug in ``reddening_cal00()``. It only affected NIR lam > 1000 nm.

V6.7.3: MC, Oxford, 8 February 2018
+++++++++++++++++++++++++++++++++++

- Plot wavelength in nm instead of Angstrom, following IAU rules.
- Ensures each element of ``start`` is not longer than its ``moments``.
- Removed underscore from internal function names.
- Included ``ftol`` keyword.

V6.7.2: MC, Oxford, 30 January 2018
+++++++++++++++++++++++++++++++++++

- Included dunder names as suggested by Peter Weilbacher (Potsdam).
- Fixed wrong ``.gas_reddening`` when ``mdegree > 0``.
- Improved formatting of the documentation.

V6.7.1: MC, Oxford, 29 November 2017
++++++++++++++++++++++++++++++++++++

- Removed import of ``misc.factorial``, deprecated in Scipy 1.0.

V6.7.0: MC, Oxford, 6 November 2017
+++++++++++++++++++++++++++++++++++

- Allow users to input identically zero gas templates while still
  producing a stable NNLS solution. In this case, warn the user and set
  the .gas_zero_template attribute. This situation can indicate an input
  bug or a gas line that entirely falls within a masked region.
- Corrected ``gas_flux_error`` normalization, when input not normalized.
- Return ``.gas_bestfit``, ``.gas_mpoly``, ``.mpoly`` and ``.apoly`` attributes.
- Do not multiply gas emission lines by polynomials, instead allow for
  ``gas_reddening`` (useful with tied Balmer emission lines).
- Use ``axvspan`` to visualize masked regions in the plot.
- Fixed program stop with ``linear`` keyword.
- Introduced ``reddening_func`` keyword.

V6.6.4: MC, Oxford, 5 October 2017
++++++++++++++++++++++++++++++++++

- Check for NaN in ``galaxy`` and check all ``bounds`` have two elements.
- Allow ``start`` to be either a list or an array or vectors.

V6.6.3: MC, Oxford, 25 September 2017
+++++++++++++++++++++++++++++++++++++

- Reduced bounds on multiplicative polynomials and clipped to positive
  values. Thanks to Xihan Ji (Tsinghua University) for providing an
  example of slightly negative gas emission lines, when the spectrum
  contains essentially just noise.
- Improved visualization of masked pixels.

V6.6.2: MC, Oxford, 15 September 2017
+++++++++++++++++++++++++++++++++++++

- Fixed program stop with a 2-dim template array and regularization.
  Thanks to Adriano Poci (Macquarie University) for the clear report and
  the fix.

V6.6.1: MC, Oxford, 4 August 2017
+++++++++++++++++++++++++++++++++

- Included note on ``.gas_flux`` output units. Thanks to Xihan Ji
  (Tsinghua University) for the feedback.

V6.6.0: MC, Oxford, 27 June 2017
++++++++++++++++++++++++++++++++

- Print and return gas fluxes and errors, if requested, with the new
  ``gas_component`` and ``gas_names`` keywords.

V6.5.0: MC, Oxford, 23 June 2017
++++++++++++++++++++++++++++++++

- Replaced ``MPFIT`` with ``capfit`` for a Levenberg-Marquardt method with
  fixed or tied variables, which rigorously accounts for box constraints.

V6.4.2: MC, Oxford, 2 June 2017
+++++++++++++++++++++++++++++++

- Fixed removal of bounds in solution, introduced in V6.4.1.
  Thanks to Kyle Westfall (Univ. Santa Cruz) for reporting this.
- Included ``method`` keyword to use Scipy's ``least_squares()``
  as an alternative to MPFIT.
- Force float division in pixel conversion of ``start`` and ``bounds``.

V6.4.1: MC, Oxford, 25 May 2017
+++++++++++++++++++++++++++++++

- ``linear_fit()`` does not return unused status anymore, for
  consistency with the corresponding change to ``cap_mpfit``.

V6.4.0: MC, Oxford, 12 May 2017
+++++++++++++++++++++++++++++++

- Introduced ``tied`` keyword to tie parameters during fitting.
- Included discussion of formal errors of ``.weights``.

V6.3.2: MC, Oxford, 4 May 2017
++++++++++++++++++++++++++++++

- Fixed possible program stop introduced in V6.0.7 and consequently
  removed unnecessary function ``_templates_rfft()``. Many thanks to
  Jesus Falcon-Barroso for a very clear and useful bug report!

V6.3.1: MC, Oxford, 13 April 2017
+++++++++++++++++++++++++++++++++

- Fixed program stop when fitting two galaxy spectra with
  reflection-symmetric LOSVD.

V6.3.0: MC, Oxford, 30 March 2017
+++++++++++++++++++++++++++++++++

- Included ``reg_ord`` keyword to allow for both first and second-order
  regularization.

V6.2.0: MC, Oxford, 27 March 2017
+++++++++++++++++++++++++++++++++

- Improved curvature criterion for regularization when ``dim > 1``.

V6.1.0: MC, Oxford, 15 March 2017
+++++++++++++++++++++++++++++++++

- Introduced ``trig`` keyword to use a trigonometric series as
  alternative to Legendre polynomials.

V6.0.7: MC, Oxford, 13 March 2017
+++++++++++++++++++++++++++++++++

- Use ``next_fast_len()`` for optimal ``rfft()`` zero padding.
- Included keyword ``gas_component`` in the ``.plot()`` method, to
  distinguish gas emission lines in best-fitting plots.
- Improved plot of residuals for noisy spectra.
- Simplified regularization implementation.

V6.0.6: MC, Oxford, 23 February 2017
++++++++++++++++++++++++++++++++++++

- Added ``linear_fit()`` and ``nonlinear_fit()`` functions to better
  clarify the code structure. Included ``templates_rfft`` keyword.
- Updated documentation. Some code simplifications.

V6.0.5: MC, Oxford, 21 February 2017
++++++++++++++++++++++++++++++++++++

- Consistently use new format_output() function both with/without
  the ``linear`` keyword. Added ``.status`` attribute. Changes suggested by
  Kyle Westfall (Univ. Santa Cruz).

V6.0.4: MC, Oxford, 30 January 2017
+++++++++++++++++++++++++++++++++++

- Re-introduced ``linear`` keyword to only perform a linear fit and
  skip the non-linear optimization.

V6.0.3: MC, Oxford, 1 December 2016
+++++++++++++++++++++++++++++++++++

- Return usual ``Chi**2/DOF`` instead of Biweight estimate.

V6.0.2: MC, Oxford, 15 August 2016
++++++++++++++++++++++++++++++++++

- Improved formatting of printed output.

V6.0.1: MC, Oxford, 10 August 2016
++++++++++++++++++++++++++++++++++

- Allow ``moments`` to be an arbitrary integer.
- Allow for scalar ``moments`` with multiple kinematic components.

V6.0.0: MC, Oxford, 28 July 2016
++++++++++++++++++++++++++++++++

- Compute the Fourier Transform of the LOSVD analytically:
- Major improvement in velocity accuracy when ``sigma < velscale``.
- Removed ``oversample`` keyword, which is now unnecessary.
- Removed limit on velocity shift of templates.
- Simplified FFT zero padding. Updated documentation.

V5.3.3: MC, Oxford 24 May 2016
++++++++++++++++++++++++++++++

- Fixed Python 2 compatibility. Thanks to Masato Onodera (NAOJ).

V5.3.2: MC, Oxford, 22 May 2016
+++++++++++++++++++++++++++++++

- Backward compatibility change: allow ``start`` to be smaller than
  ``moments``. After feedback by Masato Onodera (NAOJ).
- Updated documentation of ``bounds`` and ``fixed``.

V5.3.1: MC, Oxford, 18 May 2016
+++++++++++++++++++++++++++++++

- Use wavelength in the plot when available. Make ``plot()`` a class function.
  Changes suggested and provided by Johann Cohen-Tanugi (LUPM).

V5.3.0: MC, Oxford, 9 May 2016
++++++++++++++++++++++++++++++

- Included ``velscale_ratio`` keyword to pass a set of templates with
  higher resolution than the galaxy spectrum.
- Changed ``oversample`` keyword to require integers, not Booleans.

V5.2.0: MC, Baltimore, 26 April 2016
++++++++++++++++++++++++++++++++++++

- Included ``bounds``, ``fixed`` and ``fraction`` keywords.

V5.1.18: MC, Oxford, 20 April 2016
++++++++++++++++++++++++++++++++++

- Fixed deprecation warning in Numpy 1.11. Changed order from 1 to 3
  during oversampling. Warn if sigma is under-sampled.

V5.1.17: MC, Oxford, 21 January 2016
++++++++++++++++++++++++++++++++++++

- Expanded explanation of the relationship between output velocity and redshift.

V5.1.16: MC, Oxford, 9 November 2015
++++++++++++++++++++++++++++++++++++

- Fixed potentially misleading typo in documentation of ``moments``.

V5.1.15: MC, Oxford, 22 October 2015
++++++++++++++++++++++++++++++++++++

- Updated documentation. Thanks to Peter Weilbacher (Potsdam) for
  corrections.

V5.1.14: MC, Oxford, 19 October 2015
++++++++++++++++++++++++++++++++++++

- Fixed deprecation warning in Numpy 1.10.

V5.1.13: MC, Oxford, 24 April 2015
++++++++++++++++++++++++++++++++++

- Updated documentation.

V5.1.12: MC, Oxford, 25 February 2015
+++++++++++++++++++++++++++++++++++++

- Use ``color=`` instead of ``c=`` to avoid a new Matplotlib 1.4 bug.

V5.1.11: MC, Sydney, 5 February 2015
++++++++++++++++++++++++++++++++++++

- Reverted change introduced in V5.1.2. Thanks to Nora Lu"tzgendorf
  for reporting problems with ``oversample``.

V5.1.10: MC, Oxford, 14 October 2014
++++++++++++++++++++++++++++++++++++

- Fixed bug in saving output introduced in the previous version.

V5.1.9: MC, Las Vegas Airport, 13 September 2014
++++++++++++++++++++++++++++++++++++++++++++++++

- Pre-compute FFT and oversampling of templates. This speeds up the
  calculation for very long or highly oversampled spectra. Thanks to
  Remco van den Bosch for reporting situations where this optimization
  may be useful.

V5.1.8: MC, Utah, 10 September 2014
+++++++++++++++++++++++++++++++++++

- Fixed program stop with ``reddening`` keyword. Thanks to Masatao
  Onodera for reporting the problem.

V5.1.7: MC, Oxford, 3 September 2014
++++++++++++++++++++++++++++++++++++

- Relaxed requirement on input maximum velocity shift.
- Minor reorganization of the code structure.

V5.1.6: MC, Oxford, 6 August 2014
+++++++++++++++++++++++++++++++++

- Catch an additional input error. Updated documentation for Python.
  Included templates ``matrix`` in output. Modified plotting colours.

V5.1.5: MC, Oxford, 21 June 2014
++++++++++++++++++++++++++++++++

- Fixed deprecation warning.

V5.1.4: MC, Oxford, 25 May 2014
+++++++++++++++++++++++++++++++

- Support both Python 2.7 and Python 3.

V5.1.3: MC, Oxford, 7 May 2014
++++++++++++++++++++++++++++++

- Allow for an input covariance matrix instead of an error spectrum.

V5.1.2: MC, Oxford, 6 May 2014
++++++++++++++++++++++++++++++

- Replaced REBIN with INTERPOLATE + /OVERSAMPLE keyword. This is
  to account for the fact that the Line Spread Function of the observed
  galaxy spectrum already includes pixel convolution. Thanks to Mike
  Blanton for the suggestion.

V5.1.1: MC, Dallas Airport, 9 February 2014
+++++++++++++++++++++++++++++++++++++++++++

- Fixed typo in the documentation of ``nnls_flags``.

V5.1.0: MC, Oxford, 9 January 2014
++++++++++++++++++++++++++++++++++

- Allow for a different LOSVD for each template. Templates can be stellar or
  can be gas emission lines. A pPXF version adapted for multiple kinematic
  components existed for years. It was updated in JAN/2012 for the paper by
  Johnston et al. (2013, MNRAS). This version merges those changes with the
  public pPXF version, making sure that all previous pPXF options are still
  supported.

V5.0.1: MC, Oxford, 12 December 2013
++++++++++++++++++++++++++++++++++++

- Minor cleaning and corrections.

V5.0.0: MC, Oxford, 6 December 2013
+++++++++++++++++++++++++++++++++++

- Translated from IDL into Python and tested against the original version.

V4.6.6: MC, Paranal, 8 November 2013
++++++++++++++++++++++++++++++++++++

- Uses CAP_RANGE to avoid potential naming conflicts.

V4.6.5: MC, Oxford, 15 November 2012
++++++++++++++++++++++++++++++++++++

- Expanded documentation of REGUL keyword.

V4.6.4: MC, Oxford, 9 December 2011
+++++++++++++++++++++++++++++++++++

- Increased oversampling factor to 30x, when the /OVERSAMPLE keyword
  is used. Updated corresponding documentation. Thanks to Nora
  Lu"tzgendorf for test cases illustrating errors in the recovered
  velocity when the sigma is severely undersampled.

V4.6.3: MC, Oxford 25 October 2011
++++++++++++++++++++++++++++++++++

- Do not change TEMPLATES array in output when REGUL is nonzero.
  From the feedback of Richard McDermid.

V4.6.2: MC, Oxford, 17 October 2011
+++++++++++++++++++++++++++++++++++

- Included option for 3D regularization and updated documentation of
  REGUL keyword.

V4.6.1: MC, Oxford, 29 July 2011
++++++++++++++++++++++++++++++++

- Use Coyote Graphics (http://www.idlcoyote.com/) by David W. Fanning.
  The required routines are now included in NASA IDL Astronomy Library.

V4.6.0: MC, Oxford, 12 April 2011
+++++++++++++++++++++++++++++++++

- Important fix to /CLEAN procedure: bad pixels are now properly
  updated during the 3sigma iterations.

V4.5.0: MC, Oxford, 13 April 2010
+++++++++++++++++++++++++++++++++

- Dramatic speed up in the convolution of long spectra.

V4.4.0: MC, Oxford, 18 September 2009
+++++++++++++++++++++++++++++++++++++

- Introduced Calzetti et al. (2000) ppxf_REDDENING_CURVE function to
  estimate the reddening from the fit.

V4.3.0: MC, Oxford, 4 Mach 2009
+++++++++++++++++++++++++++++++

- Introduced REGUL keyword to perform linear regularization of WEIGHTS
  in one or two dimensions.

V4.2.3: MC, Oxford, 27 November 2008
++++++++++++++++++++++++++++++++++++

- Corrected error message for too big velocity shift.

V4.2.2: MC, Windhoek, 3 July 2008
+++++++++++++++++++++++++++++++++

- Added keyword POLYWEIGHTS.

V4.2.1: MC, Oxford, 17 May 2008
+++++++++++++++++++++++++++++++

- Use LA_LEAST_SQUARES (IDL 5.6) instead of SVDC when fitting a single
  template. Please let me know if you need to use pPXF with an older IDL
  version.

V4.2.0: MC, Oxford, 15 March 2008
+++++++++++++++++++++++++++++++++

- Introduced optional fitting of SKY spectrum. Many thanks to
  Anne-Marie Weijmans for testing.

V4.1.7: MC, Oxford, 6 October 2007
++++++++++++++++++++++++++++++++++

- Updated documentation with an important note on penalty determination.

V4.1.6: MC, Leiden, 20 January 2006
+++++++++++++++++++++++++++++++++++

- Print the number of nonzero templates. Do not print outliers in /QUIET mode.

V4.1.5: MC, Leiden, 10 February 2005
++++++++++++++++++++++++++++++++++++

- Verify that GOODPIXELS is monotonic and does not contain duplicated
  values. After feedback from Richard McDermid.

V4.1.4: MC, Leiden, 12 January 2005
+++++++++++++++++++++++++++++++++++

- Make sure input NOISE is a positive vector.

V4.1.3: MC, Vicenza, 30 December 2004
+++++++++++++++++++++++++++++++++++++

- Updated documentation.

V4.1.2: MC, Leiden, 11 November 2004
++++++++++++++++++++++++++++++++++++

- Handle special case where a single template without additive
  polynomials is fitted to the galaxy.

V4.1.1: MC, Leiden, 21 September 2004
+++++++++++++++++++++++++++++++++++++

- Increased maximum number of iterations ITMAX in BVLS. Thanks to
  Jesus Falcon-Barroso for reporting problems.
- Introduced error message when velocity shift is too big.
- Corrected output when MOMENTS=0.

V4.1.0: MC, Leiden, 3 September 2004
++++++++++++++++++++++++++++++++++++

- Corrected implementation of two-sided fitting of the LOSVD. Thanks
  to Stefan van Dongen for reporting problems.

V4.0.0: MC, Vicenza, 16 August 2004
+++++++++++++++++++++++++++++++++++

- Introduced optional two-sided fitting assuming a reflection
  symmetric LOSVD for two input spectra.

V3.7.3: MC, Leiden, 7 August 2004
+++++++++++++++++++++++++++++++++

- Corrected bug: keyword ERROR was returned in pixels instead of km/s.
- Decreased lower limit on fitted dispersion. Thanks to Igor V. Chilingarian.

V3.7.2: MC, Leiden, 28 April 2004
+++++++++++++++++++++++++++++++++

- Corrected program stop after fit when MOMENTS=2. The bug was introduced in V3.7.0.

V3.7.1: MC, Leiden, 31 March 2004
+++++++++++++++++++++++++++++++++

- Updated documentation.

V3.7.0: MC, Leiden, 23 March 2004
+++++++++++++++++++++++++++++++++

- Revised implementation of MDEGREE option. Nonlinear implementation:
  straightforward, robust, but slower.

V3.6.0: MC, Leiden, 19 March 2004
+++++++++++++++++++++++++++++++++

- Added MDEGREE option for multiplicative polynomials. Linear implementation:
  fast, works well in most cases, but can fail in certain cases.

V3.5.0: MC, Leiden, 11 December 2003
++++++++++++++++++++++++++++++++++++

- Included /OVERSAMPLE option.

V3.4.7: MC, Leiden, 8 December 2003
+++++++++++++++++++++++++++++++++++

- First released version.

V1.0.0: Leiden, 10 October 2001
+++++++++++++++++++++++++++++++

- Created by Michele Cappellari.

