#!/usr/bin/env python

from __future__ import print_function, division

import numpy as np
import matplotlib.pyplot as plt

from astropy.table import Table
import astropy.io.fits as fits

import h5py
import os
from glob import glob
from tqdm import tqdm

from model import load_h5

def gather_data(thin=None):
    lamost_dir = 'data/xp_lamost_dr8_match'
    meta_dir = 'data/xp_continuous_metadata'
    xp_summary_dir = 'data/xp_summary_h5'
    xp_out_dir = 'data/xp_opt_output_files'
    xp_in_dir = 'data/xp_opt_input_files'

    lamost_fname_pattern = 'match_xp_continuous_lamost_*-*.h5'
    lamost_fnames = glob(os.path.join(lamost_dir, lamost_fname_pattern))
    # Deterministically shuffle the data filenames, by first ordering
    # and then shuffling with a fixed seed. This ensures that the data
    # is always ordered in the exact same way, and that the
    # training/validation split is consistent.
    lamost_fnames.sort()
    import random
    random.seed(17)
    random.shuffle(lamost_fnames)
    if thin is not None:
        print(f'Only using one of every {thin} XP spectrum files.')
        lamost_fnames = lamost_fnames[::thin]

    # Load LAMOST catalog
    fname = 'data/lamost/dr8_v2.0_LRS_stellar.csv'
    print(f'Loading LAMOST: {fname} ...')
    lamost_cat = Table.read(fname, format='ascii.csv')
    print(f' -> {len(lamost_cat)} sources.')

    # Load the HotPayne catalog
    fname = 'data/lamost/hot_payne.fits.gz'
    print(f'Loading HotPayne LAMOST catalog: {fname} ...')
    hotpayne_cat = Table.read(fname, format='fits')
    print(f' -> {len(hotpayne_cat)} sources.')

    # Join the LAMOST and HotPayne catalogs
    print('Joining LAMOST and HotPayne catalogs ...')
    lamost_specid = np.array([
        '{: <36s}'.format(
            '-'.join([
                l['obsdate'].replace('-',''),
                l['planid'],
                f'{l["spid"]:02d}',
                f'{l["fiberid"]:03d}'
            ])
        )
        for l in lamost_cat
    ], dtype='S36')
    idx_lamost_sort = np.argsort(lamost_specid)
    idx_hotpayne_in_lamost = idx_lamost_sort[np.searchsorted(
        lamost_specid,
        hotpayne_cat['SpecID'],
        sorter=idx_lamost_sort
    )]
    idx_match = (
        lamost_specid[idx_hotpayne_in_lamost] == hotpayne_cat['SpecID']
    )
    idx_hotpayne = np.where(idx_match)[0]
    idx_lamost = idx_hotpayne_in_lamost[idx_match]

    # Prefer HotPayne catalog values, where available
    print('Replacing LAMOST cat values with HotPayne values, where avail...')
    print(f' -> Replacing {idx_hotpayne.size} rows ...')
    key_replace = [
        ('teff', 'Teff'),
        ('teff_err', 'e_Teff'),
        ('logg', 'logg'),
        ('logg_err', 'e_logg'),
        ('feh', '[Fe/H]'),
        ('feh_err', 'e_[Fe/H]')
    ]
    for k1,k2 in key_replace:
        print(f' * {k2} -> {k1}')
        lamost_cat[k1][idx_lamost] = hotpayne_cat[k2][idx_hotpayne]

    # Open up output file, to collect results
    fname_out = 'data/xp_results_lamost_match.h5'

    with h5py.File(fname_out, 'w') as fout:
        # Loop over XP files
        for i,fn in enumerate(tqdm(lamost_fnames, smoothing=1/100)):
            fid = fn.split('_')[-1].split('.')[0]

            d = {} # Empty dict for data to store

            #
            # Load information on LAMOST - Gaia XP matches
            #
            with h5py.File(fn, 'r') as f:
                n = f['gdr3_source_id'].size
                if n == 0:
                    continue

                # Load Gaia source id
                d['gdr3_source_id'] = f['gdr3_source_id'][:]

                # LAMOST - Gaia XP match indices
                gaia_idx = f['gaia_index'][:]
                lamost_idx = f['lamost_index'][:]
                sep_arcsec = f['sep_arcsec'][:]

            #
            # Load stellar type from LAMOST
            #
            type_fields = [
                'teff',
                'feh',
                'logg'
            ]
            type_err_fields = [
                'teff_err',
                'feh_err',
                'logg_err'
            ]
            n_params = len(type_fields)
            d['lamost_stellar_type'] = np.empty((n,n_params), dtype='f4')
            d['lamost_stellar_type_err'] = np.empty((n,n_params), dtype='f4')
            for k,(key0,key1) in enumerate(zip(type_fields,type_err_fields)):
                d['lamost_stellar_type'][:,k] = lamost_cat[key0][lamost_idx]
                d['lamost_stellar_type_err'][:,k] = lamost_cat[key1][lamost_idx]

            # Convert T_eff to kiloKelvins
            d['lamost_stellar_type'][:,0] *= 0.001
            d['lamost_stellar_type_err'][:,0] *= 0.001

            # LAMOST S/N
            for b in 'gri':
                d[f'lamost_snr_{b}'] = lamost_cat[f'snr{b}'][lamost_idx]

            #
            # Load extra Gaia information
            #
            meta_fn = os.path.join(
                meta_dir,
                f'xp_metadata_{fid}.h5'
            )
            
            m = Table.read(meta_fn)
            #with fits.open(meta_fn, mode='readonly') as f:
            #    m = f[1].data[:]

            # Check that GDR3 source IDs match
            assert np.all(m['source_id'][gaia_idx] == d['gdr3_source_id'])

            # Load Gaia photometry (not used in model)
            for band in ('g', 'bp', 'rp'):
                for suffix in ('', '_error'):
                    key = f'phot_{band}_mean_flux{suffix}'
                    d[key] = m[key][gaia_idx].astype('f4')

            # Additional Gaia keys to load
            extra_keys = (
                'ra', 'dec',
                'fidelity_v2',
                'phot_bp_rp_excess_factor',
                'astrometric_excess_noise',
                'ruwe',
                'norm_dg',
                'phot_g_mean_mag',
                'parallax',
                'parallax_error',
                'phot_bp_n_obs',
                'phot_rp_n_obs',
            )
            for key in extra_keys:
                d[key] = m[key][gaia_idx]

            #
            # Load extra BP/RP spectroscopic information
            #
            xp_summary_fn = os.path.join(
                xp_summary_dir,
                f'XpSummary_{fid}.hdf5'
            )
            summary_keys = (
                'bp_chi_squared',
                'rp_chi_squared',
                'bp_n_transits',
                'rp_n_transits',
                'bp_n_blended_transits',
                'rp_n_blended_transits',
                'bp_n_contaminated_transits',
                'rp_n_contaminated_transits'
            )
            with h5py.File(xp_summary_fn, 'r') as f:
                for key in summary_keys:
                    d[key] = f[key][:][gaia_idx]

            #
            # Load XP inferred parameters
            #
            xp_out_fn = os.path.join(
                xp_out_dir,
                f'xp_opt_output_{fid}.h5'
            )
            xp_out_keys = (
                'stellar_params_est',
                'stellar_params_err',
                'rchi2_opt'
            )
            with h5py.File(xp_out_fn, 'r') as f:
                sid = f['gdr3_source_id'][:][gaia_idx]
                assert np.all(sid == d['gdr3_source_id'])
                for k in xp_out_keys:
                    d[k] = f[k][:][gaia_idx]

            #
            # Load XP input
            #
            xp_in_fn = os.path.join(
                xp_in_dir,
                f'xp_opt_input_{fid}.h5'
            )
            xp_in_keys = (
                'flux',
                'flux_err',
                'flux_cov_eival_max',
                'flux_cov_eival_min',
                'tmass_flux',
                'tmass_flux_var',
                'unwise_flux',
                'unwise_flux_var'
            )
            with h5py.File(xp_in_fn, 'r') as f:
                sid = f['gdr3_source_id'][:][gaia_idx]
                assert np.all(sid == d['gdr3_source_id'])
                for k in xp_in_keys:
                    d[k] = f[k][:][gaia_idx]

            #
            # Append data to combined output file
            #
            n = len(d['gdr3_source_id'])
            chunk_size = 4096
            for key in d:
                if key in fout:
                    dset = fout[key]
                    dset_len = len(fout[key])
                    s = dset.shape
                    dset.resize((s[0]+n,)+s[1:])
                    dset[s[0]:] = d[key][:]
                else:
                    s = d[key].shape
                    fout.create_dataset(
                        key, data=d[key],
                        maxshape=(None,)+s[1:],
                        chunks=(chunk_size,)+s[1:],
                        compression='lzf'
                    )


def main():
    gather_data()

    return 0

if __name__ == '__main__':
    main()

