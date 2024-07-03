# Learning on the Fly with Controlled Model Error

These notebooks demonstrate an ASE Calculator which gradually transitions from training a surrogate model to using it to accelerate molecular dynamics.
It employs a controller to decide when the model is accurate enough to replace the original calcualtor, following the [Proxima strategy developed by Zamora et al.](https://dl.acm.org/doi/10.1145/3447818.3460370).
