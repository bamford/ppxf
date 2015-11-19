------------------------------
The pPXF software distribution
------------------------------

This distribution contains a Python (http://www.python.org/)
implementation of the Penalized Pixel-Fitting (pPXF) method developed by
Cappellari M., & Emsellem E., (2004, PASP, 116, 138).

The following files are included in the software distribution:

cap_mpfit.py                ---> Craig Markwardt MPFIT adapted for Python 2/3
ppxf.py                     ---> The main pPXF routine
ppxf_kinematics_example_sauron.py ---> pPXF kinematics usage example
ppxf_kinematics_example_sdss.py   ---> pPXF kinematics usage example
ppxf_population_example_sdss.py   ---> Example for stellar population
ppxf_population_gas_example_sdss.py ---> Extracting stellar population and gas emission
ppxf_simulation_example.py     ---> Example Monte Carlo simulation
ppxf_two_components_example.py ---> Extracting two stellar kinematic components
ppxf_utils.py               ---> Contains log_rebin and determine_good_pixels
ppxf_python_reference_output.txt ---> The output one should get from the examples
readme.txt                  ---> This README file
/spectra                    ---> directory of FITS spectra for example
/Miles_Models               ---> directory of FITS SSP models

------------
Installation
------------

To be able to import the module from any directory, one can add the directory with the Python code to a "path configuration file" (.pth) placed in the .../site-packages/ directory (see https://docs.python.org/install/#modifying-python-s-search-path for details).

------------------
pPXF usage example
------------------

To learn how to use the main program PPXF run the example programs
and read the detailed documentation at the top of the file ppxf.py

The procedure PPXF uses the following Python procedure, which is already 
included in the distribution, adapted to support both Python 2.6/2.7 and Python 3:

- MPFIT: by C.B. Markwardt -> Python version https://code.google.com/p/astrolibpy/

The core scientific library Numpy/Scipy/Matplotlib http://scipy.org/ are assumed 
to be installed and Pyfits or Astropy for reading FITS files 
http://www.stsci.edu/institute/software_hardware/pyfits

The program was tested on:

    Python 2.7 and Python 3.4 with NumPy 1.10, SciPy 0.15, Matplotlib 1.4

No attempt was made to make PPXF work on outdated versions of those packages!

-------------------------------
IMPORTANT: Proper usage of pPXF
-------------------------------

The PPXF routine can give sensible quick results with the default BIAS
parameter, however, like in any penalized/filtered/regularized method, the
optimal amount of penalization generally depends on the problem under study.

The general rule here is that the penalty should leave the line-of-sight
velocity-distribution (LOSVD) virtually unaffected, when it is well
sampled and the signal-to-noise ratio (S/N) is sufficiently high.

EXAMPLE: If you expect an LOSVD with up to a high h4~0.2 and your
adopted penalty biases the solution towards a much lower h4~0.1 even
when the measured sigma > 3*velScale and the S/N is high, then you
are *misusing* the pPXF method!


THE RECIPE: The following is a simple practical recipe for a sensible
determination of the penalty in pPXF:

1. Choose a minimum (S/N)_min level for your kinematics extraction and
   spatially bin your data so that there are no spectra below (S/N)_min;

2. Perform a fit of your kinematics *without* penalty (PPXF keyword BIAS=0).
   The solution will be noisy and may be affected by spurious solutions,
   however this step will allow you to check the expected mean ranges in
   the Gauss-Hermite parameters [h3,h4] for the galaxy under study;

3. Perform a Monte Carlo simulation of your spectra, following e.g. the
   included ppxf_simulation_example.pro routine. Adopt as S/N in the simulation
   the chosen value (S/N)_min and as input [h3,h4] the maximum representative
   values measured in the non-penalized pPXF fit of the previous step;

4. Choose as penalty (BIAS) the *largest* value such that, for sigma > 3*velScale,
   the mean difference between the output [h3,h4] and the input [h3,h4]
   is well within the rms scatter of the simulated values
   (see e.g. Fig.2 of Emsellem et al. 2004, MNRAS, 352, 721).

-----------------------------------
Problems with your first pPXF fit ?
-----------------------------------

Common problems with your first PPXF fit are caused by incorrect wavelength
ranges or different velocity scales between galaxy and templates. To quickly
detect these problems try to overplot the (log rebinned) galaxy and the
template just before calling the PPXF procedure.

You can use something like the following Python lines while adjusting the
smoothing window and the pixels shift. If you cannot get a rough match
by eye it means something is wrong and it is unlikely that PPXF
(or any other program) will find a good match.

  import numpy as np
  import matplotlib.pyplot as plt
  import scipy.ndimage
  
  plt.plot(galaxy, 'k')
  tmp = np.roll(ndimage.gaussian_filter1d(template, 2), -20)
  plt.plot(tmp/np.median(tmp)*np.median(galaxy), 'r')

################

Written: Michele Cappellari, Leiden, 6 November 2003
Last updated: MC, Oxford, 22 October 2015
