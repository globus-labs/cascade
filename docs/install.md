# Installation Notes

Explaining the particularities for running experiments with Cascade at different HPC centers.

## Improv (LCRC)

[Improv](https://www.lcrc.anl.gov/systems/improv) is a cluster at Argonne with two 64-core AMD processors per node.

Install your own copy of Miniconda on the system (see [Linux instructions here](https://docs.anaconda.com/free/miniconda/miniconda-install/))
then build the environment:

```commandline
conda env create --file enviornment -p ./env
```

We need a version of CP2K and ASE that are not relased yet:

1. Use the CP2K installation at `/lcrc/project/Athena/cp2k`
2. Install ASE from [this branch](https://gitlab.com/WardLT/ase/-/tree/cp2k-set-pos-file): `pip install git+https://gitlab.com/wardlt/ase.git@cp2k-set-pos-file`

The build file for CP2K is in [./cp2k-build](./cp2k-build)
