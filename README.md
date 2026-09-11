# libRC

A reservoir-computing and autoencoder pipeline research framework.

The importable package lives in [`src/`](src/): the reservoir-computing core
(`reservoir.py`, `connectivity.py`, `integrators.py`) and the convolutional
autoencoder (`autoencoder.py`) used to compress simulation frames into the
latent trajectories the reservoir is trained on. Research, sweep, and
data-pipeline scripts that use `src` live under
[`HPHall-data-Summer24/scripts/`](HPHall-data-Summer24/scripts/).
