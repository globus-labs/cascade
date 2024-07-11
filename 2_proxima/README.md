# Evaluate On-the-Fly Learning with Proxima

[Proxima](https://dl.acm.org/doi/10.1145/3447818.3460370) implements a serial form of on-the-fly learning.
We run a single trajectory of a dynamics simulation and evaluate whether to use the target function
or a surrogate model at each timestep.
The surrogate model is improved by expanding the training set with data produced each time the target function is run.
As the model improves, a control system will gradually increase likelihood a surrogate model will be used.

The notebooks in this directory explore the effects of changing different aspects of the learning algorithm or the 
dynamic system being studied.
