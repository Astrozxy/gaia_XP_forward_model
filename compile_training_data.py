#!/usr/bin/env python

from __future__ import print_function, division

import numpy as np
import matplotlib.pyplot as plt

from astropy.table import Table
import h5py

import os.path
from tqdm import tqdm
import logging
from argparse import ArgumentParser
from glob import glob

from xp_utils import XPSampler, sqrt_icov_eigen


def get_match_indices(a, b):
    idx_a = np.searchsorted(a, b)
    idx_a[idx_a == len(a)] = len(a) - 1
    is_match = (a[idx_a] == b)
    idx_a = idx_a[is_match]
    idx_b = np.where(is_match)[0]
    assert np.all(a[idx_a] == b[idx_b])
    return idx_a, idx_b


def append_to_h5_file(h5_file, data, chunk_size=512):
    for key in data:
        if key in h5_file:
            # Append to existing dataset
            dset = h5_file[key]
            n_add = len(data[key])
            s = dset.shape
            dset.resize((s[0]+n_add,)+s[1:])
            dset[s[0]:] = data[key][:]
        else:
            # Create new dataset
            s = data[key].shape
            h5_file.create_dataset(
                key, data=data[key],
                maxshape=(None,)+s[1:],
                chunks=(chunk_size,)+s[1:],
                compression='lzf'
            )


def extract_fluxes(fid, match_source_ids=None, thin=1):
    # Fields needed to sample XP spectra
    bprp_fields = [
        'bp_coefficients',
        'rp_coefficients',
        'bp_n_parameters',
        'rp_n_parameters',
        'bp_coefficient_errors',
        'rp_coefficient_errors',
        'bp_coefficient_correlations',
        'rp_coefficient_correlations',
        'bp_standard_deviation',
        'rp_standard_deviation'
    ]
    
    # Wavelength sampling (in nm)
    sample_wavelengths = np.arange(392., 993., 10.)
    flux_scale = 1e-18 # W/m^2/nm
    xp_sampler = XPSampler(sample_wavelengths, flux_scale=flux_scale)

    # Number of XP wavelengths and photometric bands
    n_wl = len(sample_wavelengths)
    n_b = 5

    # Conversion from unWISE reported fluxes to f_lambda,
    # in units of (flux_scale) * W/m^2/nm
    unwise_dm = np.array([2.699, 3.339])
    unwise_wl = np.array([3.3526e-6, 4.6028e-6])
    C = 3631e-26 * 2.99792458e8 / unwise_wl**2 * 1e-9
    unwise_f0 = C / flux_scale * 10**(-0.4*(22.5+unwise_dm))
    unwise_f0.shape = (1,2)

    # Conversion from 2MASS Vega magnitudes to f_lambda,
    # in units of (flux_scale) * W/m^2/nm
    tmass_dm = np.array([0.91, 1.39, 1.85])
    tmass_wl = np.array([1.235e-6, 1.662e-6, 2.159e-6])
    C = 3631e-26 * 2.99792458e8 / tmass_wl**2 * 1e-9
    tmass_f0 = C / flux_scale * 10**(-0.4*tmass_dm) # Flux of m_Vega=0 source
    tmass_f0.shape = (1,3)

    # Load Gaia metadata
    meta_fn = f'data/xp_metadata/xp_metadata_{fid}.h5'
    d_meta = Table.read(meta_fn)[::thin]

    # Match XP and stellar parameters
    if match_source_ids is None:
        idx_xp = slice(None)
    else:
        idx_xp, idx_matches = get_match_indices(
            d_meta['source_id'],
            match_source_ids
        )
        logging.info(f'{fid}: {len(idx_xp)} matches.')

        if len(idx_xp) == 0:
            return None, None # No matches

        # Cut down d_meta
        d_meta = d_meta[idx_xp]
        n = len(d_meta) # Number of XP sources selected

        # Determine which XP sources to include
        idx_good = np.ones(n, dtype='bool')
        # Parallax S/N: TODO: Make this cut later?
        idx_good &= (d_meta['parallax']/d_meta['parallax_error'] > 3.)
        # Gaia astrometric fidelity
        idx_good &= (d_meta['fidelity_v2'] > 0.5)
        # Gaia BP/RP excess
        idx_good &= (d_meta['phot_bp_rp_excess_factor'] < 1.3)
        #idx_xp = idx_xp[good_idx]
        #idx_params = idx_params[good_idx]

        n_good = np.count_nonzero(idx_good)
        pct_good = np.mean(idx_good) * 100
        logging.info(
            f'{fid}: {n_good} matches ({pct_good:.3g}%) '
            'pass Gaia quality cuts.'
        )

        # Apply cut
        idx_xp = idx_xp[idx_good]
        idx_matches = idx_matches[idx_good]
        d_meta = d_meta[idx_good]

    n = len(d_meta) # Number of XP sources selected

    # The GDR3 source IDs of the sources that will be in the output
    gdr3_source_id = d_meta['source_id']

    # Load XP information
    xp_fn = (
        'data/xp_continuous_mean_spectrum/'
       f'XpContinuousMeanSpectrum_{fid}.h5'
    )
    with h5py.File(xp_fn, 'r') as f:
        d_xp = {k:f[k][:][::thin][idx_xp] for k in f.keys()}

    # Sample XP spectra
    flux = np.zeros((n,n_wl+n_b), dtype='f4')
    flux_err = np.full((n,n_wl+n_b), np.nan, dtype='f4')
    flux_sqrticov = np.zeros((n,n_wl+n_b,n_wl+n_b), dtype='f4')
    flux_cov_eival_min = np.empty(n, dtype='f4')
    flux_cov_eival_max = np.empty(n, dtype='f4')
    for j in range(n):
        # Get sampled spectrum and covariance matrix
        fl,fl_err,fl_cov,_,_ = xp_sampler.sample(
            *[d_xp[k][j] for k in bprp_fields],
            bp_zp_errfrac=0.001, # BP zero-point uncertainty
            rp_zp_errfrac=0.001, # RP zero-point uncertainty
            zp_errfrac=0.005, # Joint BP/RP zero-point uncertainty
            diag_errfrac=0.005 # Indep. uncertainty at each wavelength
        )
        flux[j,:n_wl] = fl
        flux_err[j,:n_wl] = fl_err
        U,(eival0,eival1) = sqrt_icov_eigen(fl_cov, eival_floor=1e-9)#condition_max=1e6)
        flux_sqrticov[j,:n_wl,:n_wl] = U.T
        flux_cov_eival_min[j] = eival0
        flux_cov_eival_max[j] = eival1
    
    idx_neg = (flux_cov_eival_min <= 0)
    n_neg = np.count_nonzero(idx_neg)
    pct_neg = 100 * np.mean(idx_neg)
    logging.info(f'{fid}: {n_neg} have negative eivals ({pct_neg:.3g}%).')

    # Load stellar extinctions

    # Fix norm_dg (if NaN, then no close neighbor)
    idx = np.isfinite(d_meta['norm_dg'])
    d_meta['norm_dg'][~idx] = -50. # TODO: Is this correct?

    # Load unWISE photometry
    unwise_fn = os.path.join(
        'data', 'xp_unwise_match',
        f'match_xp_continuous_unwise_{fid}.h5'
    )
    with h5py.File(unwise_fn, 'r') as f:
        sid = f['gdr3_source_id'][:]
        d_unwise = f['unwise_data'][:]
        sep_unwise = f['sep_arcsec'][:]
    idx_insert_unwise,idx = get_match_indices(gdr3_source_id, sid)
    d_unwise = d_unwise[idx]
    sep_unwise = sep_unwise[idx]
    n_unwise = len(d_unwise)
    pct_unwise = 100 * n_unwise / n
    logging.info(f'{fid}: {n_unwise} have unWISE ({pct_unwise:.3g}%).')

    # Calculate unWISE fluxes, in units of the flux scale
    # (which is 10^{-18} W/m^2/nm, by default)
    unwise_flux = d_unwise['flux']*unwise_f0
    unwise_flux_err = d_unwise['dflux']*unwise_f0
    unwise_err_over_flux = (
        d_unwise['dflux']
      / d_unwise['flux']
    )

    # Add 1% uncertainties to unWISE (in quadrature)
    unwise_flux_var = unwise_flux_err**2 + (0.1*unwise_flux)**2
    unwise_flux_err = np.sqrt(unwise_flux_err)

    # Apply quality cuts to unWISE
    idx_unwise_good = (
        # Minimal contamination from neighboring sources
        (d_unwise['fracflux'] > 0.8)
        # Few flags set in PSF of source
      & (d_unwise['qf'] > 0.8)
        # Distance of cross-match < 0.25"
      & (sep_unwise[:,None] < 0.25)
        # Flux uncertainty < 20%
      & (unwise_err_over_flux < 0.2)
        # Positive flux
      & (d_unwise['flux'] > 0.)
        # Close neighbors
      & (d_meta['norm_dg'][idx_insert_unwise][:,None] < -10.)
    )
    pct_unwise = 100 * np.mean(idx_unwise_good, axis=0)
    logging.info(
        f'{fid}: unWISE: '
        f'{pct_unwise}% pass quality cuts.'
    )

    for b,idx in enumerate(idx_unwise_good.T):
        unwise_flux[~idx,b] = np.nanmedian(unwise_flux[:,b])
        unwise_flux_err[~idx,b] = np.inf
        unwise_flux_var[~idx,b] = np.inf

    # Paste in WISE photometric bands
    flux[idx_insert_unwise,-2:] = unwise_flux
    flux_err[idx_insert_unwise,-2:] = unwise_flux_err
    flux_sqrticov[idx_insert_unwise,-2,-2] = 1/unwise_flux_err[:,0]
    flux_sqrticov[idx_insert_unwise,-1,-1] = 1/unwise_flux_err[:,1]
    
    # Load 2MASS photometry
    tmass_fname = os.path.join(
        'data',
        'xp_tmass_match',
        f'xp_tmass_match_{fid}.fits.gz'
    )
    d_tmass = Table.read(tmass_fname)
    idx_insert_tmass,idx = get_match_indices(
        gdr3_source_id,
        d_tmass['source_id']
    )
    d_tmass = d_tmass[idx]
    n_tmass = len(d_tmass)
    pct_tmass = 100 * n_tmass / n
    logging.info(f'{fid}: {n_tmass} have 2MASS ({pct_tmass:.3g}%).')

    # Extract high-quality 2MASS photometry
    tmass_mag = np.full((len(d_tmass),3), np.nan)
    tmass_mag_err = np.full((len(d_tmass),3), np.nan)
    for k,b in enumerate(['j','h','ks']):
        if not len(d_tmass):
            continue
        # Only use high-quality photometry (phot_qual == 'A')
        idx_phot = np.array(
            [s[k]=='A' for s in d_tmass['ph_qual']]
        )
        tmass_mag[idx_phot,k] = d_tmass[f'{b}_m'][idx_phot]
        tmass_mag_err[idx_phot,k] = d_tmass[f'{b}_msigcom'][idx_phot]

    # Additionally require:
    #   * 2MASS magnitude uncertainty < 0.2
    #   * No close, bright neighbors (norm_dg > -5)
    idx_unphot = (
        (tmass_mag_err > 0.2)
      | (d_meta['norm_dg'][idx_insert_tmass][:,None] > -5.)
    )
    tmass_mag[idx_unphot] = np.nan
    tmass_mag_err[idx_unphot] = np.nan

    pct_tmass = 100 * np.mean(np.isfinite(tmass_mag), axis=0)
    logging.info(f'{fid}: 2MASS: {pct_tmass}% pass cuts.')

    # Calculate 2MASS fluxes, in units of the flux scale
    # (which is 10^{-18} W/m^2/nm, by default)
    tmass_flux = 10**(-0.4*tmass_mag) * tmass_f0
    tmass_flux_err = 2.5/np.log(10) * tmass_mag_err * tmass_flux

    # Replace NaNs with default values
    for b in range(tmass_flux.shape[1]):
        idx_nan = ~np.isfinite(tmass_flux[:,b])
        tmass_flux[idx_nan,b] = np.nanmedian(tmass_flux[:,b])
        tmass_flux_err[idx_nan,b] = np.inf

    # Paste in 2MASS photometric bands
    flux[idx_insert_tmass,-5:-2] = tmass_flux
    flux_err[idx_insert_tmass,-5:-2] = tmass_flux_err
    for b in range(3):
        flux_sqrticov[idx_insert_tmass,-5+b,-5+b] = 1/tmass_flux_err[:,b]

    # Check that there are no NaNs in the fluxes
    assert np.all(np.isfinite(flux))
    assert np.all(np.isfinite(flux_sqrticov))

    # Compile all the information
    d = {
        'gdr3_source_id': gdr3_source_id,
        'flux': flux,
        'flux_err': flux_err,
        'flux_sqrticov': flux_sqrticov,
        'flux_cov_eival_min': flux_cov_eival_min,
        'flux_cov_eival_max': flux_cov_eival_max,
    }

    # Gaia photometry (not used in model)
    for band in ('g', 'bp', 'rp'):
        for suffix in ('', '_error'):
            key = f'phot_{band}_mean_flux{suffix}'
            d[key] = d_meta[key].astype('f4')

    # Sky coordinates
    for key in ('ra', 'dec'):
        d[key] = d_meta[key]

    # Parallax
    d['plx'] = d_meta['parallax'].astype('f4')
    d['plx_err'] = d_meta['parallax_error'].astype('f4')

    # Load Gaia quality flags
    for key in ('fidelity_v2', 'phot_bp_rp_excess_factor', 'norm_dg'):
        d[key] = d_meta[key]

    # Check that all arrays have same length
    for key in d:
        assert len(d[key]) == len(d['gdr3_source_id'])

    if match_source_ids is None:
        return d
    else:
        return d, idx_matches


def main():
    parser = ArgumentParser(
        description='Compile training data for a set of training sources.',
        add_help=True
    )
    parser.add_argument(
        '--stellar-params',
        '-sp',
        type=str,
        required=True,
        help='Table containing gdr3_source_id, params_est and params_err.'
    )
    parser.add_argument(
        '--output',
        '-o',
        type=str,
        required=True,
        help='Output filename.'
    )
    parser.add_argument(
        '--thin',
        '-th',
        type=int,
        nargs=2,
        default=(1,1),
        help='Only use every n^th XP file and every m^th star in each file.'
    )
    args = parser.parse_args()

    # Set up warning/error logging
    query_log = os.path.splitext(args.output)[0] + '.log'
    logging.basicConfig(
        filename=query_log,
        level=logging.INFO,
        format='%(asctime)s : %(levelname)s : %(message)s'
    )

    #unwise_dir = 'data/xp_unwise_match'
    #tmass_dir = 'data/xp_tmass_match'
    #reddening_dir = 'data/xp_bayestar_match'
    #meta_dir = 'data/xp_metadata'

    # Load the stellar parameters linked to Gaia source_ids
    d_params = Table.read(args.stellar_params)
    d_params.sort('gdr3_source_id')
    print(f'Loaded {len(d_params)} source_ids for training dataset.')

    # Compile required data (Gaia,2MASS,WISE,Bayestar) for each training star
    xp_fnames = glob(os.path.join(
        'data',
        'xp_continuous_mean_spectrum',
        'XpContinuousMeanSpectrum_*-*.h5'
    ))
    xp_fnames.sort()
    xp_fnames = xp_fnames[::args.thin[0]]

    with h5py.File(args.output, 'w') as fout:
        for xp_fn in tqdm(xp_fnames):
            fid = xp_fn.split('_')[-1].split('.')[0]

            d,idx_params = extract_fluxes(fid, d_params['gdr3_source_id'])
            if d is None:
                continue

            d['stellar_type'] = d_params['params_est'][idx_params]
            d['stellar_type_err'] = d_params['params_err'][idx_params]

            append_to_h5_file(fout, d)

    return 0

if __name__ == '__main__':
    main()

