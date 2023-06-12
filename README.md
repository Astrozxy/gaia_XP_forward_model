Gaia XP Forward Model
=====================

This is the tidied-up version of the Gaia XP forward model.

Workflow
========

1. Download all the Gaia continuous mean spectra in HDF5 format (available [here](https://sdsc-users.flatironinstitute.org/~gaia/dr3/hdf5/XpContinuousMeanSpectrum/)) to the folder `data/xp_continuous_mean_spectrum/`.
2. Create the folder `data/xp_continuous_metadata/` and run `query_xp_metadata.py`, which queries the Gaia Archive for additional information about each XP source.
3. ...
