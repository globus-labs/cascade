Installation
============

Explaining the particularities for running experiments with Cascade at different HPC centers.

Quick Install
-------------

Build Cascade on a generic Linux computer by building the environment from Anaconda

.. code-block:: shell

    conda env create --file environment.yml

The environment will include packages for all `machine learning packages <./learning.html>`_
and a Conda-Forge-provided version of CP2K.

Custom Python Environment
-------------------------

You need not install Cascade via the Anaconda environment.
All packages needed by Cascade are available via PyPI, so
feel free to create an environment using desired versions of packages (e.g., PyTorch)
and with our without the optional requirements (e.g., MACE).

Link Cascade to atomic simulation codes via the Atomic Simulation Environment's
`configuration options <https://wiki.fysik.dtu.dk/ase/ase/calculators/calculators.html#calculator-configuration>`_.

System-Specific Instructions
----------------------------

Notes for clusters used by our team

Improv (LCRC)
~~~~~~~~~~~~~

`Improv <https://www.lcrc.anl.gov/systems/improv>`_ is a cluster at Argonne with two 64-core AMD processors per node.

Install your own copy of Miniconda on the system (see `Linux instructions <https://docs.anaconda.com/free/miniconda/miniconda-install/>`_)
then build the environment:

.. code-block:: shell

    conda env create --file environment.yml -p ./env

Use the CP2K installation at `/lcrc/project/Athena/cp2k-mpich`.
The build file for CP2K is in `../cp2k-build <../cp2k-build>`_.
