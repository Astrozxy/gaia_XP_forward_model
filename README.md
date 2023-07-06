Gaia XP Forward Model
=====================

A data-driven forward model of Gaia XP stellar spectra, which maps from atmospheric parameters, distance and extinction to predicted XP spectra. The model is learned from a subset of XP stars with observed higher-resolution spectra (e.g., from LAMOST). After the model has been learned, it can be applied to *all* XP spectra to obtain inferred stellar parameters.

Workflow
========

The following workflow should bring you from zero to having inferred stellar parameters for the entire Gaia XP dataset:

1. Download all the Gaia XP continuous mean spectra in HDF5 format (available [here](https://sdsc-users.flatironinstitute.org/~gaia/dr3/hdf5/XpContinuousMeanSpectrum/)) to the folder `data/xp_continuous_mean_spectrum/`.
2. Create the folder `data/xp_metadata/` and run `query_xp_metadata.py`, which queries the [Gaia Archive](https://gea.esac.esa.int/archive/) for additional information about each XP source.
3. Create the folder `data/xp_tmass_match/` and run `query_2mass.py`, which queries the Gaia Archive for 2MASS photometry for each XP source.
4. Download the unWISE catalog (for example, from [here](https://portal.nersc.gov/project/cosmo/data/unwise/neo6/unwise-catalog/cat/)). Set the environment variable `UNWISE_DIR` to the directory into which you downloaded the catalog files. Create the folder `data/xp_unwise_match/` and run `match_unwise_gaia.py`.
5. Create the folder `data/xp_dustmap_match/` and run `query_reddening_gaia.py`, which queries various dust maps at the location of each XP source.
6. Prepare a training dataset with three columns: `gdr3_source_id` (the Gaia DR3 `source_id`), `params_est` (estimated intrinsic stellar parameters) and `params_err` (the corresponding uncertainties). These could come from LAMOST, APOGEE, GALAH or other spectroscopic surveys, for example. Save this dataset as an [Astropy Table](https://docs.astropy.org/en/stable/table/index.html). Let's call this file `training_params.h5`.
7. Run `compile_training_data.py` (for example, as follows: `python3 compile_training_data.py --stellar-params training_params.h5 --output training_data.h5`) to pull all the data (stellar parameters, XP spectra, photometry, astrometry, reddenings, etc.) for the training set into one file. A log file will be generated (e.g., `training_data.log`), which should be filled with harmless (but informative) `INFO` entries. Make sure that there are no `WARNING` or `ERROR` entries, which indicate that there's something wrong with the training set.
8. Run `train.py` (for example, `python3 train.py -i training_data.h5 -o output_folder/`) with the training dataset generated in the previous step. This results in a model of the XP spectra, which maps from intrinsic stellar parameters, distance and extinction to predicted XP spectrum and photometry.
9. Generate input files for all the XP sources, containing the necessary information to infer stellar parameters: `python3 gen_xp_input_allsources.py`.
10. Infer stellar parameters for all the XP sources: `python3 xp_opt_allsources.py`.
