#!/usr/bin/env python

from __future__ import print_function, division

import numpy as np
import matplotlib.pyplot as plt

from astropy.table import Table
from astropy.io import fits

import h5py

import os
from glob import glob
from tqdm import tqdm
import logging
import hashlib

from xp_utils import XPSampler, sqrt_icov_eigen


def main():
    from argparse import ArgumentParser
    parser = ArgumentParser(
        description='Generate input files for all BP/RP sources.',
        add_help=True
    )
    parser.add_argument(
        '--partition',
        type=int,
        nargs=2,
        help='Only handle fraction input files: (# of partitions, partition).'
    )
    args = parser.parse_args()

    # Set up warning/error logging
    if args.partition is None:
        log_suffix = ''
    else:
        n_parts,part = args.partition
        log_suffix = f'_{part:03d}of{n_parts:03d}'
    query_log = os.path.join(
        'data/xp_opt_input_files',
        f'gen_xp_input_allsources{log_suffix}.log'
    )
    logging.basicConfig(
        filename=query_log,
        level=logging.INFO,
        format='%(asctime)s : %(levelname)s : %(message)s'
    )

    # Output file pattern
    out_fname_pattern = os.path.join(
        'data/xp_opt_input_files',
        'xp_opt_input_{}.h5'
    )

    # Gaia BP/RP spectra
    xp_spec_fnames = glob(os.path.join(
        'data/xp_continuous_mean_spectrum',
        'XpContinuousMeanSpectrum_*-*.h5'
    ))
    xp_spec_fnames.sort()
    n_files = len(xp_spec_fnames)
    print(f'{n_files} XP spectral files found.')

    # Metadata for Gaia BP/RP spectra
    xp_meta_fnames = glob(os.path.join(
        'data/xp_continuous_metadata',
        'xp_metadata_*-*.fits.gz'
    ))
    xp_meta_fnames.sort()
    n = len(xp_meta_fnames)
    if n != n_files:
        print(f'  # of XP metadata files ({n}) does not match!')
        return 1

    # 2MASS photometry
    tmass_fnames = glob(os.path.join(
        'data/xp_tmass_match',
        'xp_tmass_match_*-*.fits.gz'
    ))
    tmass_fnames.sort()
    n = len(tmass_fnames)
    if n != n_files:
        print(f'  # of 2MASS files ({n}) does not match!')
        return 1

    # unWISE photometry
    wise_fnames = glob(os.path.join(
        'data/xp_unwise_match',
        'match_xp_continuous_unwise_*-*.h5'
    ))
    wise_fnames.sort()
    n = len(wise_fnames)
    if n != n_files:
        print(f'  # of WISE files ({n}) does not match!')
        return 1

    # Bayestar reddening
    # TODO

    # Choose a subset of input files to work with
    if args.partition is not None:
        n_parts,part = args.partition
        # Use an MD5 hash of the filename to deterministically assign
        # filenames to subsets
        fn_hashes = [
            int(hashlib.md5(fn.encode('utf-8')).hexdigest(), base=16)
            for fn in xp_spec_fnames
        ]
        # Filter files
        idx_keep = [h%n_parts==part for h in fn_hashes]
        xp_spec_fnames = [fn for fn,b in zip(xp_spec_fnames,idx_keep) if b]
        xp_meta_fnames = [fn for fn,b in zip(xp_meta_fnames,idx_keep) if b]
        tmass_fnames = [fn for fn,b in zip(tmass_fnames,idx_keep) if b]
        wise_fnames = [fn for fn,b in zip(wise_fnames,idx_keep) if b]
        n_files = len(xp_spec_fnames)
        logging.info(f'{n_files} files in partition {part+1} of {n_parts}.')

    # Object to sample XP spectra
    sample_wavelengths = np.arange(392., 993., 10.) # in nm
    flux_scale = 1e-18 # in W/m^2/nm
    xp_sampler = XPSampler(sample_wavelengths, flux_scale=flux_scale)

    # Define which fields to extract from input files
    # Gaia astrometry
    plx_field = 'parallax'
    plx_err_field = 'parallax_error'
    # Gaia BP/RP spectra
    bprp_fields = [ # key, bad definition, bad filler
        ('bp_coefficients', lambda x: np.isnan(x), 1.),
        ('rp_coefficients', lambda x: np.isnan(x), 1.),
        ('bp_n_parameters', lambda x: x != 55, 55),
        ('rp_n_parameters', lambda x: x != 55, 55),
        ('bp_coefficient_errors', lambda x: np.isnan(x), 1e4),
        ('rp_coefficient_errors', lambda x: np.isnan(x), 1e4),
        ('bp_coefficient_correlations', lambda x: np.isnan(x), 0.),
        ('rp_coefficient_correlations', lambda x: np.isnan(x), 0.),
        ('bp_standard_deviation', lambda x: np.isnan(x), 1.),
        ('rp_standard_deviation', lambda x: np.isnan(x), 1.)
    ]
    # Stellar extinction
    stellar_ext_field = 'E_mean_bayestar'
    stellar_ext_err_field = 'E_sigma_bayestar'
    stellar_ext_err_floor_abs = 0.03
    stellar_ext_err_floor_pct = 0.10

    # Conversion from unWISE reported fluxes to f_lambda,
    # in units of (flux_scale) * W/m^2/nm
    unwise_dm = np.array([2.699, 3.339])
    unwise_wl = np.array([3.3526e-6, 4.6028e-6])
    C = 3631e-26 * 2.99792458e8 / unwise_wl**2 * 1e-9
    unwise_f0 = C / flux_scale * 10**(-0.4*(22.5+unwise_dm))
    print(f'unwise_f0 = {unwise_f0}')
    unwise_f0.shape = (1,2)

    # Conversion from 2MASS Vega magnitudes to f_lambda,
    # in units of (flux_scale) * W/m^2/nm
    tmass_dm = np.array([0.91, 1.39, 1.85])
    tmass_wl = np.array([1.235e-6, 1.662e-6, 2.159e-6])
    C = 3631e-26 * 2.99792458e8 / tmass_wl**2 * 1e-9
    tmass_f0 = C / flux_scale * 10**(-0.4*tmass_dm) # Flux of m_Vega=0 source
    print(f'tmass_f0 = {tmass_f0}')
    tmass_f0.shape = (1,3)

    n_b = 5 # Number of photometric bands to be appended to spectrum

    for i,fn in enumerate(tqdm(xp_spec_fnames)):
        fid = fn.split('_')[-1].split('.')[0]
        out_fname = out_fname_pattern.format(fid)

        # Skip file if matches have already been stored
        if os.path.exists(out_fname):
            logging.info(f'Skipping existing file: "{out_fname}"')
            continue

        # Ignore the one defective file that Gaia released:
        #if fid == '614517-614573':
        #    logging.warning(f'Skipping known defective GDR3 file: "{out_fname}"')
        #    continue

        d = {}

        # Gaia XP spectra
        fn_spec = xp_spec_fnames[i]
        with h5py.File(fn_spec, 'r') as f:
            # Load Gaia source id
            d['gdr3_source_id'] = f['source_id'][:]

            n_sources = d['gdr3_source_id'].size

            # Load BP/RP spectral data
            try:
                bprp_data = []
                for key,f_bad,fill_bad in bprp_fields:
                    x = f[key][:]
                    # Fix any problems in XP spectral data
                    idx_bad = f_bad(x)
                    if np.any(idx_bad):
                        logging.warning(f'{fn_spec} has bad {key} values.')
                        x[idx_bad] = fill_bad
                    bprp_data.append(x)
                #bprp_data = [f[key][:] for key in bprp_fields]
            except KeyError as err:
                logging.error(f'Missing fields in {fn_spec}. Skipping file.')
                print(err)
                continue

        logging.info(f'{n_sources} sources in {fn_spec}.')

        # Gaia metadata
        fn_meta = xp_meta_fnames[i]
        assert fn_meta.split('_')[-1].split('.')[0] == fid
        with fits.open(fn_meta, mode='readonly') as f:
            # Load Gaia photometry (not used in model)
            for band in ('g', 'bp', 'rp'):
                for suffix in ('', '_error'):
                    key = f'phot_{band}_mean_flux{suffix}'
                    d[key] = f[1].data[key][:].astype('f4')

            # Load parallax
            d['plx'] = f[1].data[plx_field][:].astype('f4')
            d['plx_err'] = f[1].data[plx_err_field][:].astype('f4')

            # Load sky coordinates
            d['ra'] = f[1].data['ra'][:].astype('f4')
            d['dec'] = f[1].data['dec'][:].astype('f4')

            # Load norm_dg (a measure of nearby bad neighbors)
            norm_dg = f[1].data['norm_dg'][:]
            idx = np.isfinite(norm_dg)
            norm_dg[~idx] = -45.

        # Sample BP/RP fluxes and calculate their inverse covariances
        n_wl = sample_wavelengths.size
        flux = np.zeros((n_sources,n_wl+n_b), dtype='f4')
        flux_err = np.full((n_sources,n_wl+n_b), np.nan, dtype='f4')
        flux_sqrticov = np.zeros((n_sources,n_wl+n_b,n_wl+n_b), dtype='f4')
        flux_cov_eival_min = np.empty(n_sources, dtype='f4')
        flux_cov_eival_max = np.empty(n_sources, dtype='f4')
        for j in range(n_sources):
            # Get sampled spectrum and covariance matrix
            fl,fl_err,fl_cov,_,_ = xp_sampler.sample(
                *[col[j] for col in bprp_data],
                bp_zp_errfrac=0.001, # BP zero-point uncertainty
                rp_zp_errfrac=0.001, # RP zero-point uncertainty
                zp_errfrac=0.005, # Joint BP/RP zero-point uncertainty
                diag_errfrac=0.005 # Indep. uncertainty at each wavelength
            )
            flux[j,:n_wl] = fl
            flux_err[j,:n_wl] = fl_err
            U,(eival0,eival1) = sqrt_icov_eigen(fl_cov, eival_floor=1e-9)
            flux_sqrticov[j,:n_wl,:n_wl] = U.T
            flux_cov_eival_min[j] = eival0
            flux_cov_eival_max[j] = eival1

        n_neg_eival = np.count_nonzero(eival0<=0)
        pct_neg_eival = n_neg_eival/eival0.size * 100
        logging.info(
            f'{n_neg_eival} negative eigenvalues ({pct_neg_eival:.4f}%).'
        )

        ## Load stellar extinctions
        #d['stellar_ext'] = f[stellar_ext_field][:].astype('f4')
        #d['stellar_ext_err'] = f[stellar_ext_err_field][:].astype('f4')
        #d['stellar_ext_err'] = np.sqrt(
        #    d['stellar_ext_err']**2
        #  + (stellar_ext_err_floor_pct*d['stellar_ext'])**2
        #  + stellar_ext_err_floor_abs**2
        #)

        # Load unWISE data
        fn_wise = wise_fnames[i]
        assert fn_wise.split('_')[-1].split('.')[0] == fid
        with h5py.File(fn_wise, 'r') as f:
            d_unwise = {k: f[k][:] for k in f}

        # Join unWISE to Gaia
        unwise_sort_idx = np.argsort(d_unwise['gdr3_source_id'])
        insert_idx = np.searchsorted(
            d_unwise['gdr3_source_id'],
            d['gdr3_source_id'],
            sorter=unwise_sort_idx
        )
        insert_idx[insert_idx >= d_unwise['gdr3_source_id'].size] = -1
        unwise_idx = unwise_sort_idx[insert_idx]
        is_match = (
            d_unwise['gdr3_source_id'][unwise_idx] == d['gdr3_source_id']
        )
        unwise_idx = unwise_idx[is_match]
        d_idx = np.where(is_match)[0]

        for k in d_unwise:
            d_unwise[k] = d_unwise[k][unwise_idx]

        # Calculate unWISE fluxes, in units of the flux scale
        # (which is 10^{-18} W/m^2/nm, by default)
        unwise_flux = d_unwise['unwise_data']['flux']*unwise_f0
        unwise_flux_err = d_unwise['unwise_data']['dflux']*unwise_f0
        unwise_err_over_flux = (
            d_unwise['unwise_data']['dflux']
          / d_unwise['unwise_data']['flux']
        )

        # Apply quality cuts to unWISE
        idx_unwise_good = (
            # Minimal contamination from neighboring sources
            (d_unwise['unwise_data']['fracflux'] > 0.8)
            # Few flags set in PSF of source
          & (d_unwise['unwise_data']['qf'] > 0.8)
            # Distance of cross-match < 0.25"
          & (d_unwise['sep_arcsec'][:,None] < 0.25)
            # Flux uncertainty < 20%
          & (unwise_err_over_flux < 0.2)
            # Positive flux
          & (d_unwise['unwise_data']['flux'] > 0.)
            # Neighbor contamination
          & (norm_dg[d_idx] < -10.)[:,None]
        )

        # Insert unWISE columns with default values (for non-observations)
        wise_flux_fill = np.nanmedian(unwise_flux, axis=0)
        d['unwise_flux'] = np.full((n_sources,2), wise_flux_fill, dtype='f4')
        d['unwise_flux_var'] = np.full((n_sources,2), np.inf, dtype='f4')

        # Copy in good unWISE fluxes
        for b in (0,1):
            idx0 = d_idx[idx_unwise_good[:,b]]
            idx1 = idx_unwise_good[:,b]
            d['unwise_flux'][idx0,b] = unwise_flux[idx1,b]
            d['unwise_flux_var'][idx0,b] = unwise_flux_err[idx1,b]**2
            # Check that matching still correct (this is really a bug check)
            assert np.all(
                 d['gdr3_source_id'][idx0]
              == d_unwise['gdr3_source_id'][idx1]
            )

        # Add in 1% uncertainty floor to unWISE flux errors
        d['unwise_flux_var'] += (0.01*d['unwise_flux'])**2
        unwise_flux_err = np.sqrt(d['unwise_flux_var'])

        # Paste WISE photometric bands into flux arrays
        flux[:,-2:] = d['unwise_flux']
        flux_err[:,-2:] = unwise_flux_err
        flux_sqrticov[:,-2,-2] = 1/unwise_flux_err[:,0]
        flux_sqrticov[:,-1,-1] = 1/unwise_flux_err[:,1]

        # Load 2MASS photometry
        fn_tmass = tmass_fnames[i]
        assert fn_tmass.split('_')[-1].split('.')[0] == fid
        d_tmass = Table.read(fn_tmass, format='fits')

        # Join 2MASS to already selected data
        tmass_idx = np.searchsorted(
            d_tmass['source_id'],
            d['gdr3_source_id']
        )
        tmass_idx[tmass_idx >= d_tmass['source_id'].size] = -1
        is_match = (
            d_tmass['source_id'][tmass_idx] == d['gdr3_source_id']
        )
        tmass_idx = tmass_idx[is_match]
        d_idx = np.where(is_match)[0]

        d_tmass = d_tmass[tmass_idx]

        # Extract high-quality 2MASS photometry
        tmass_mag = np.full((len(d_tmass),3), np.nan)
        tmass_mag_err = np.full((len(d_tmass),3), np.nan)
        for k,b in enumerate(['j','h','ks']):
            # Only use high-quality photometry (phot_qual == 'A')
            idx_phot = np.array([s[k]=='A' for s in d_tmass['ph_qual']])
            #print(f'{b}: {np.count_nonzero(idx_phot)/idx_phot.size*100:.5f}% A')
            tmass_mag[idx_phot,k] = d_tmass[f'{b}_m'][idx_phot]
            tmass_mag_err[idx_phot,k] = d_tmass[f'{b}_msigcom'][idx_phot]

        # Additionally require 2MASS magnitude uncertainty < 0.2
        idx_unphot = tmass_mag_err > 0.2
        tmass_mag[idx_unphot] = np.nan
        tmass_mag_err[idx_unphot] = np.nan

        # Ignore 2MASS when close neighbor detected
        idx_bad_neighbor = (norm_dg[d_idx] > -5.)
        tmass_mag[idx_bad_neighbor] = np.nan
        tmass_mag_err[idx_bad_neighbor] = np.nan

        # Calculate 2MASS fluxes, in units of the flux scale
        # (which is 10^{-18} W/m^2/nm, by default)
        tmass_flux = 10**(-0.4*tmass_mag) * tmass_f0
        tmass_flux_err = 2.5/np.log(10) * tmass_mag_err * tmass_flux

        # Insert 2MASS columns with default values (for non-observations)
        tmass_flux_fill = np.nanmedian(tmass_flux, axis=0)
        d['tmass_flux'] = np.full((n_sources,3), tmass_flux_fill, dtype='f4')
        d['tmass_flux_var'] = np.full((n_sources,3), np.inf, dtype='f4')

        # Copy in good 2MASS fluxes
        for b in range(3):
            idx_tmass_good = np.isfinite(tmass_flux[:,b])
            idx0 = d_idx[idx_tmass_good]
            idx1 = idx_tmass_good
            d['tmass_flux'][idx0,b] = tmass_flux[idx1,b]
            d['tmass_flux_var'][idx0,b] = tmass_flux_err[idx1,b]**2
            # Check that matching still correct (this is really a bug check)
            assert np.all(
                 d['gdr3_source_id'][idx0]
              == d_tmass['source_id'][idx1]
            )

        # Add in 1% uncertainty floor to 2MASS flux errors
        d['tmass_flux_var'] += (0.01*d['tmass_flux'])**2
        tmass_flux_err = np.sqrt(d['tmass_flux_var'])

        # Paste 2MASS photometry into flux arrays
        flux[:,-5:-2] = d['tmass_flux']
        flux_err[:,-5:-2] = tmass_flux_err
        for b in range(3):
            flux_sqrticov[:,-5+b,-5+b] = 1/tmass_flux_err[:,b]

        # Add flux data into output dictionary
        d['flux'] = flux
        d['flux_err'] = flux_err
        d['flux_sqrticov'] = flux_sqrticov
        d['flux_cov_eival_min'] = flux_cov_eival_min
        d['flux_cov_eival_max'] = flux_cov_eival_max

        # Write data into output file
        with h5py.File(out_fname, 'w') as fout:
            for key in d:
                fout.create_dataset(
                    key, data=d[key],
                    chunks=True,
                    compression='lzf'
                )

            # Store the wavelengths at which the spectrum is sampled, in nm
            sample_wavelengths_p = np.hstack([
                sample_wavelengths,
                1e9*tmass_wl,
                1e9*unwise_wl
            ])
            fout['flux'].attrs['sample_wavelengths'] = sample_wavelengths_p

    return 0

if __name__ == '__main__':
    main()

