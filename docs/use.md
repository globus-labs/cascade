# Using Cascade

Cascade is very much an experimental code, so it is only a series of 
ad hoc notebooks.

Run them on a cluster using `papermill`. We provide run scripts for different
clusters (./queue-files)[./queue-files] which you can edit to run a specific notebook

## Configuring CP2K 

Cascade uses a factory method, `cascade.calculator.make_calculator`, to make an ASE calculator
ready to use for the CP2K calculations employed in our various workflows.

Configure the CP2K executable used by ASE by passing it as an argument to the factory method,
the [ASE_CP2K_COMMAND environment variable](https://wiki.fysik.dtu.dk/ase/ase/calculators/cp2k.html),
setting [the ASE configuration file](https://wiki.fysik.dtu.dk/ase/ase/calculators/calculators.html#calculator-configuration),

The factory method uses a set of predefined configurations, which are described in the 
[`cascade/files/` directory](../cascade/files).
Alter the set of prefixes through the `template_dir` argument of `make_calculator`
or setting the `CASCADE_CP2K_TEMPLATE` environment variable.
