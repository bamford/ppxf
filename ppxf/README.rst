The pPXF Package
================

**pPXF: Full Spectrum Fitting with Photometry for Stars and Galaxies**

.. image:: http://www-astro.physics.ox.ac.uk/~cappellari/images/ppxf-logo.svg
    :target: https://www-astro.physics.ox.ac.uk/~cappellari/software
    :width: 100
.. image:: https://img.shields.io/pypi/v/ppxf.svg
    :target: https://pypi.org/project/ppxf/
.. image:: https://img.shields.io/badge/arXiv-2208.14974-orange.svg
    :target: https://arxiv.org/abs/2208.14974
.. image:: https://img.shields.io/badge/DOI-10.1093/mnras/stad2597-green.svg
    :target: https://doi.org/10.1093/mnras/stad2597

This ``pPXF`` package contains a Python implementation of the Penalized
PiXel-Fitting (``pPXF``) method. It uses full-spectrum fitting with photometry
(SED) to extract the stellar and gas kinematics, as well as the stellar population of
stars and galaxies. The method was originally described in `Cappellari & Emsellem (2004)
<https://ui.adsabs.harvard.edu/abs/2004PASP..116..138C>`_
and was substantially upgraded in subsequent years and particularly in
`Cappellari (2017) <https://ui.adsabs.harvard.edu/abs/2017MNRAS.466..798C>`_
and with the inclusion of photometry and linear constraints in
`Cappellari (2023) <https://ui.adsabs.harvard.edu/abs/2023MNRAS.526.3273C>`_.

.. contents:: :depth: 2

Attribution
-----------

If you use this software for your research, please cite at least `Cappellari (2023)`_,
or additionally some ``pPXF`` papers above. The BibTeX entry for the paper is::

    @ARTICLE{Cappellari2023,
        author = {{Cappellari}, M.},
        title = "{Full spectrum fitting with photometry in PPXF: stellar population
            versus dynamical masses, non-parametric star formation history and
            metallicity for 3200 LEGA-C galaxies at redshift $z\approx0.8$}",
        journal = {MNRAS},
        eprint = {2208.14974},
        year = 2023,
        volume = 526,
        pages = {3273-3300},
        doi = {10.1093/mnras/stad2597}
    }

Installation
------------

install with::

    pip install ppxf

Without write access to the global ``site-packages`` directory, use::

    pip install --user ppxf

To upgrade ``pPXF`` to the latest version use::

    pip install --upgrade ppxf

Usage Examples
--------------

To learn how to use the ``pPXF`` package, copy, modify and run
the example programs in the ``ppxf/examples`` directory. 
It can be found within the main ``ppxf`` package installation folder 
inside `site-packages <https://stackoverflow.com/a/46071447>`_. 
The detailed documentation is contained in the docstring of the file 
``ppxf.py``, or on `PyPi <https://pypi.org/project/ppxf/>`_ or as PDF 
from `<https://purl.org/cappellari/software>`_.

.. image:: http://www-astro.physics.ox.ac.uk/~cappellari/images/jupyter-logo.svg
    :target: https://github.com/micappe/ppxf_examples
    :width: 100
    :alt: Jupyter Notebook

Examples as Jupyter Notebooks are also available on my
`GitHub repository <https://github.com/micappe/ppxf_examples>`_.

###########################################################################
