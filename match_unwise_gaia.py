#!/usr/bin/env python

from __future__ import print_function, division

import numpy as np
import matplotlib.pyplot as plt

import astropy.io.fits as fits
from astropy.table import Table, vstack as table_vstack
from astropy.coordinates import SkyCoord, match_coordinates_sky
import astropy.units as units
from astropy.time import Time

import h5py

import os
from glob import glob
from tqdm import tqdm

from time import perf_counter


import crossmatch


def main():
    match_radius = 0.5*units.arcsec
    unwise_nside = 1024
    gaia_nside = 512

    # Load unWISE catalog
    unwise_dir = os.environ.get('UNWISE_DIR', None)
    if unwise_dir is None:
        print('Set UNWISE_DIR environment variable.')
        return 1

    unwise_fnames = glob(os.path.join(unwise_dir, '*.cat.fits'))
    unwise_fnames.sort()
    #unwise_fnames = unwise_fnames[:1000]

    print('Loading unWISE files ...')
    unwise_tables = []
    for i,fn in enumerate(tqdm(unwise_fnames)):
        t = Table.read(fn, format='fits')
        t.keep_columns([
            'unwise_objid',
            'ra', 'dec',
            'flux', 'dflux',
            'qf', 'rchi2', 'fracflux',
            #'nm',
            #'flags_unwise', 'flags_info'
        ])
        unwise_tables.append(t.copy()) # copy() forces deletion of dropped cols
    print('Done loading individual unWISE files ...')

    unwise = table_vstack(unwise_tables, join_type='exact')
    print('Done stacking unWISE files ...')
    print(f'  --> {len(unwise)} sources.')

    print(f'Partitioning unWISE into HEALPix pixels (nside={unwise_nside})...')
    t0 = perf_counter()
    unwise_hpxcat = crossmatch.HEALPixCatalog(
        unwise['ra'].data*units.deg,
        unwise['dec'].data*units.deg,
        unwise_nside,
        show_progress=True
    )
    t1 = perf_counter()
    print(f'  --> {t1-t0:.5f} s')

    # Loop over Gaia BP/RP spectra metadata files, and match each to unWISE
    out_dir = 'data/xp_unwise_match/'
    fnames = glob('data/xp_metadata/xp_metadata_*-*.h5')
    fnames.sort()

    for fn in tqdm(fnames):
        # Skip file if matches have already been stored
        fid = fn.split('_')[-1].split('.')[0]
        out_fname = os.path.join(
            out_dir,
            f'xp_unwise_match_{fid}.h5'
        )
        if os.path.exists(out_fname):
            continue

        print(f'Finding unWISE matches for {fn} ...')

        # Load metadata on Gaia BP/RP spectra
        gaia_meta = Table.read(fn)

        # Gaia coordinates
        print(f'Partitioning Gaia into HEALPix pixels (nside={gaia_nside})...')
        t0 = perf_counter()
        gaia_hpxcat = crossmatch.HEALPixCatalog(
            gaia_meta['ra'].data*units.deg,
            gaia_meta['dec'].data*units.deg,
            gaia_nside
        )
        t1 = perf_counter()
        print(f'  --> {t1-t0:.5f} s')

        # Match to unWISE
        print('Calculating crossmatch ...')
        t0 = perf_counter()
        idx_unwise, idx_gaia, sep2d = crossmatch.match_catalogs(
            unwise_hpxcat,
            gaia_hpxcat,
            match_radius
        )
        t1 = perf_counter()
        print(f'  --> {t1-t0:.5f} s')
        source_id = gaia_meta['source_id'][idx_gaia]

        print(
            f'{len(idx_gaia)} of {len(gaia_meta)} '
            'Gaia sources have unWISE match.'
        )

        unwise_matches = unwise[idx_unwise]

        # Sort by Gaia DR3 source ID
        idx_sort = np.argsort(source_id)
        source_id = source_id[idx_sort]
        idx_gaia = idx_gaia[idx_sort]
        sep2d = sep2d[idx_sort]
        unwise_matches = unwise_matches[idx_sort]

        # Save matches
        kw = dict(compression='lzf', chunks=True)
        with h5py.File(out_fname, 'w') as f:
            f.create_dataset('gdr3_source_id', data=source_id, **kw)
            f.create_dataset('gaia_index', data=idx_gaia, **kw)
            f.create_dataset('sep_arcsec', data=sep2d.to('arcsec').value, **kw)
            unwise_matches.write(
                f, path='unwise_data',
                append=True,
                compression='lzf'
            )

    return 0


if __name__ == '__main__':
    main()

