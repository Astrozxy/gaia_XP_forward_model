#!/usr/bin/env python

from __future__ import print_function, division

import numpy as np

from astropy.table import Table
from astroquery.gaia import Gaia

import h5py
import os
from glob import glob
from tqdm import tqdm

import logging


def query_metadata(source_id_0, source_id_1, out_fname):
    job = Gaia.launch_job_async(f"""
        SELECT
            g.random_index, g.source_id, g.ref_epoch,

            g.ra, g.ra_error, g.dec, g.dec_error, g.parallax, g.parallax_error,
            g.pmra, g.pmra_error, g.pmdec, g.pmdec_error, g.radial_velocity, g.radial_velocity_error,

            g.ruwe, g.astrometric_excess_noise, g.astrometric_sigma5d_max,
            g.rv_renormalised_gof, g.rv_chisq_pvalue, g.rv_expected_sig_to_noise, g.rv_nb_transits,

            g.nu_eff_used_in_astrometry, g.pseudocolour, g.ecl_lat, g.astrometric_params_solved,

            spur.fidelity_v2, spur.norm_dg, spur.theta_arcsec_worst_source,
            spur.dist_nearest_neighbor_at_least_m2_brighter,
            spur.dist_nearest_neighbor_at_least_0_brighter,
            spur.dist_nearest_neighbor_at_least_2_brighter,

            g.phot_g_mean_mag, g.phot_g_mean_flux, g.phot_g_mean_flux_error,
            g.phot_bp_mean_mag, g.phot_bp_mean_flux, g.phot_rp_mean_flux_error,
            g.phot_rp_mean_mag, g.phot_rp_mean_flux, g.phot_bp_mean_flux_error,

            g.has_xp_continuous, g.has_rvs, g.non_single_star, g.phot_variable_flag,
            g.phot_bp_n_obs, g.phot_rp_n_obs, g.phot_bp_rp_excess_factor, g.visibility_periods_used

        FROM gaiadr3.gaia_source as g
            LEFT OUTER JOIN external.gaiaedr3_spurious as spur
                ON g.source_id = spur.source_id

        WHERE
            (g.has_xp_continuous = 'True')
            AND (g.source_id >= {source_id_0}) AND (g.source_id <= {source_id_1})

        ORDER BY g.source_id
        """,
        dump_to_file=True, output_format='fits',
        output_file=out_fname
    )
    #print(job)
    #print(job.__dict__.keys())
    #print(job.responseStatus)

    return (not job.failed)


def main():
    # Output directory for metadata on spectra
    out_dir = 'data/xp_metadata'

    # Directory containing BP/RP spectra
    spec_dir = 'data/xp_continuous_mean_spectrum'
    spec_fnames = glob(os.path.join(
        spec_dir,
        'XpContinuousMeanSpectrum_??????-??????.h5'
    ))
    spec_fnames.sort()

    # Set up warning/error logging
    query_log = os.path.join(out_dir, 'query_log.txt')
    logging.basicConfig(
        filename=query_log,
        level=logging.INFO,
        format='%(asctime)s : %(levelname)s : %(message)s'
    )

    # Loop over BP/RP spectral files, and query corresponding metadata
    for fn in tqdm(spec_fnames):
        # Check if the metadata file already exists
        fid = fn.split('_')[-1].split('.')[0]
        out_fname = os.path.join(
            out_dir,
            f'xp_metadata_{fid}.fits.gz'
        )
        if os.path.exists(out_fname):
            continue

        # Look up the starting and ending source_ids in the spectral file
        print(f'Spectral file: {fn}')
        with h5py.File(fn, 'r') as f:
            sid = f['source_id'][:]
        source_id_0, source_id_1 = sid[0], sid[-1]
        print(f'Querying from {source_id_0} through {source_id_1} ...')

        # Query this range of source_ids in the Gaia Archive
        success = query_metadata(source_id_0, source_id_1, out_fname)

        if not success:
            logging.error(f'Query failed at file: "{fn}"')
            continue

        # Check that the length of the output matches the # of spectra
        t = Table.read(out_fname, format='fits')
        n_out = len(t)
        n_in = len(sid)
        if n_out != n_in:
            logging.warning(
                f'Mismatch: {n_in} (# of spectra) vs {n_out} (result rows)'
            )
            continue

        logging.info(f'Finished querying metadata for file {fn}.')

    return 0

if __name__ == '__main__':
    main()

