# CP2K Presents

This directory contains a set of presets for running CP2K via ASE. 
Each preset is defined by a short string, such as "blyp".

[`presets.yml`](./presets.yml) maps the preset to keyword arguments used when
establishing the calculator.

The remaining input file settings are defined in a file named `cp2k-[preset]-template.inp`.
