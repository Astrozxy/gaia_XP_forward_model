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


def query_2mass(source_id_0, source_id_1, out_fname):
    job = Gaia.launch_job_async(f"""
        SELECT
          g.random_index, g.source_id,
          g.ra, g.dec,

          tmass.designation,
          tmass.ra as tm_ra, tmass.dec as tm_dec,
          tmass.j_m, tmass.j_msigcom,
          tmass.h_m, tmass.h_msigcom,
          tmass.ks_m, tmass.ks_msigcom,
          tmass.ph_qual,

          xmatch.angular_distance as gaia_tmass_angular_distance

        FROM gaiadr3.gaia_source AS g
        JOIN gaiadr3.tmass_psc_xsc_best_neighbour AS xmatch USING (source_id)
        JOIN gaiadr3.tmass_psc_xsc_join AS xjoin USING (clean_tmass_psc_xsc_oid)
        JOIN gaiadr1.tmass_original_valid AS tmass ON
           xjoin.original_psc_source_id = tmass.designation

        WHERE
              (g.has_xp_continuous = 'true')
          AND (
              (tmass.ph_qual LIKE 'A__')
           OR (tmass.ph_qual LIKE '_A_')
           OR (tmass.ph_qual LIKE '__A')
          )
          AND (tmass.ext_key is NULL)
          AND (g.source_id BETWEEN {source_id_0} AND {source_id_1})

        ORDER BY g.source_id
        """,
        dump_to_file=True, output_format='fits',
        output_file=out_fname
    )

    return (not job.failed)


def main():
    # Output directory for 2MASS match
    out_dir = 'data/xp_tmass_match'

    # Directory containing BP/RP spectra
    spec_dir = 'data/xp_continuous_mean_spectrum'
    spec_fnames = glob(os.path.join(
        spec_dir,
        'XpContinuousMeanSpectrum_??????-??????.h5'
    ))
    spec_fnames.sort()

    # Set up warning/error logging
    query_log = os.path.join(out_dir, 'tmass_query_log.txt')
    logging.basicConfig(
        filename=query_log,
        level=logging.INFO,
        format='%(asctime)s : %(levelname)s : %(message)s'
    )

    # Loop over BP/RP spectral files, and query 2MASS matches
    for fn in tqdm(spec_fnames):
        # Check if the match file already exists
        fid = fn.split('_')[-1].split('.')[0]
        out_fname = os.path.join(
            out_dir,
            f'xp_tmass_match_{fid}.fits.gz'
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
        success = query_2mass(source_id_0, source_id_1, out_fname)

        if not success:
            logging.error(f'Query failed at file: "{fn}"')
            continue

        logging.info(f'Finished querying 2MASS matches for file {fn}.')

        # Compare the length of the output matches and the # of spectra
        # (in general, not all Gaia sources have 2MASS matches)
        t = Table.read(out_fname, format='fits')
        n_out = len(t)
        n_in = len(sid)
        f_match = n_out/n_in*100
        logging.info(f'{fn}: {n_out} of {n_in} ({f_match:.3f}%) have matches.')

    return 0

if __name__ == '__main__':
    main()

