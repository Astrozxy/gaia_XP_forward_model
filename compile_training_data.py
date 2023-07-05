#!/usr/bin/env python

from __future__ import print_function, division

import numpy as np
import matplotlib.pyplot as plt

from astropy.table import Table
from astropy.coordinates import SkyCoord
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
    
    # Wavelength sampling (in nm)
    sample_wavelengths = np.arange(392., 993., 10.)
    flux_scale = 1e-18 # W/m^2/nm
    xp_sampler = XPSampler(sample_wavelengths, flux_scale=flux_scale)

    # Number of XP wavelengths and photometric bands
    n_wl = len(sample_wavelengths)
    n_b = 5

    # Conversion from 2MASS Vega magnitudes to f_lambda,
    # in units of (flux_scale) * W/m^2/nm
    tmass_dm = np.array([0.91, 1.39, 1.85])
    tmass_wl = np.array([1.235e-6, 1.662e-6, 2.159e-6])
    C = 3631e-26 * 2.99792458e8 / tmass_wl**2 * 1e-9
    tmass_f0 = C / flux_scale * 10**(-0.4*tmass_dm) # Flux of m_Vega=0 source
    tmass_f0.shape = (1,3)

    # Conversion from unWISE reported fluxes to f_lambda,
    # in units of (flux_scale) * W/m^2/nm
    unwise_dm = np.array([2.699, 3.339])
    unwise_wl = np.array([3.3526e-6, 4.6028e-6])
    C = 3631e-26 * 2.99792458e8 / unwise_wl**2 * 1e-9
    unwise_f0 = C / flux_scale * 10**(-0.4*(22.5+unwise_dm))
    unwise_f0.shape = (1,2)

    sample_wavelengths = np.hstack([
        sample_wavelengths,
        tmass_wl*1e9,
        unwise_wl*1e9
    ])

    # Load Gaia metadata
    meta_fn = f'data/xp_metadata/xp_metadata_{fid}.h5'
    d_meta = Table.read(meta_fn)[::thin]

    # Match XP and stellar parameters
    if match_source_ids is not None:
        idx_xp, idx_matches = get_match_indices(
            d_meta['source_id'],
            match_source_ids
        )
        logging.info(f'{fid}: {len(idx_xp)} matches.')

        if len(idx_xp) == 0:
            return None, None, sample_wavelengths # No matches

        # Cut down d_meta
        d_meta = d_meta[idx_xp]
        n = len(d_meta) # Number of XP sources selected

        # Determine which XP sources to include
        idx_good = np.ones(n, dtype='bool')
        # Parallax S/N: TODO: Make this cut later?
        idx_good &= (d_meta['parallax']/d_meta['parallax_error'] > 3.)
        # Gaia astrometric fidelity
        idx_good &= (d_meta['fidelity_v2'] > 0.5)
        # Gaia BP/RP excess (cut weakened, so as not to eliminate M-dwarfs)
        #idx_good &= (d_meta['phot_bp_rp_excess_factor'] < 2.0)
        #idx_xp = idx_xp[good_idx]
        #idx_params = idx_params[good_idx]

        n_good = np.count_nonzero(idx_good)
        pct_good = np.mean(idx_good) * 100
        logging.info(
            f'{fid}: {n_good} matches ({pct_good:.3g}%) '
            'pass Gaia quality cuts.'
        )

        # Apply cut
        idx_matches = idx_matches[idx_good]
        d_meta = d_meta[idx_good]

    n = len(d_meta) # Number of XP sources selected
    if n == 0:
        return None, None, sample_wavelengths # No matches

    # The GDR3 source IDs of the sources that will be in the output
    gdr3_source_id = d_meta['source_id']

    # Load XP information
    xp_fn = (
        'data/xp_continuous_mean_spectrum/'
       f'XpContinuousMeanSpectrum_{fid}.h5'
    )
    d_xp = {}
    with h5py.File(xp_fn, 'r') as f:
        for key,f_bad,fill_bad in bprp_fields:
            x = f[key][:]
            # Fix any problems in XP spectral data
            idx_bad = f_bad(x)
            if np.any(idx_bad):
                logging.warning(f'{fid}: Fixing bad {key} values.')
                x[idx_bad] = fill_bad
            d_xp[key] = x

    # Load stellar extinctions
    E_fname = f'data/xp_dustmap_match/xp_reddening_match_{fid}.h5'
    with h5py.File(E_fname, 'r') as f:
        d_E = {k:f[k][:] for k in f.keys()}
    idx,idx_insert_E = get_match_indices(d_E['gdr3_source_id'], gdr3_source_id)
    d_E = {k:d_E[k][idx] for k in d_E}
    coords_E = SkyCoord(
        d_meta['ra'][idx_insert_E],
        d_meta['dec'][idx_insert_E],
        unit='deg', frame='icrs'
    )
    n_noE = n - len(idx_insert_E)
    pct_noE = 100 * n_noE / n
    logging.info(f'{fid}: {n_noE} have no reddening ({pct_noE:.3g}%).')
    stellar_ext = np.full(n, 0.)
    stellar_ext_err = np.full(n, np.inf)
    stellar_ext_source = np.full(n, '', dtype='S12')
    # Order or priority: Bayestar, SFD (only above plane), nothing
    idx = np.isfinite(d_E['E_mean_bayestar'])
    stellar_ext[idx_insert_E[idx]] = d_E['E_mean_bayestar'][idx]
    stellar_ext_err[idx_insert_E[idx]] = d_E['E_sigma_bayestar'][idx]
    stellar_ext_source[idx_insert_E[idx]] = 'bayestar'
    # Only use SFD when:
    #   1. Bayestar not available
    #   2. >400 pc above Galactic midplane
    z_min = 0.4 # kpc
    z_gal = coords_E.transform_to('galactic').represent_as('cartesian').z.value
    plx_upper = (
        d_meta['parallax'][idx_insert_E]
      + 2*d_meta['parallax_error'][idx_insert_E]
    )
    plx_upper = np.clip(plx_upper, 0., np.inf)
    z_gal_lower = z_gal / plx_upper # kpc
    idx_aboveplane = (np.abs(z_gal_lower) > z_min)
    idx = idx_aboveplane & ~np.isfinite(d_E['E_mean_bayestar'])
    stellar_ext[idx_insert_E[idx]] = d_E['E_mean_sfd'][idx]
    stellar_ext_err[idx_insert_E[idx]] = 0.
    stellar_ext_source[idx_insert_E[idx]] = 'sfd'
    # Inflate all reddening uncertainties
    stellar_ext_err_floor_abs = 0.03
    stellar_ext_err_floor_pct = 0.10
    stellar_ext_err = np.sqrt(
        stellar_ext_err**2
      + (stellar_ext_err_floor_pct*stellar_ext)**2
      + stellar_ext_err_floor_abs**2
    )
    # Log statistics
    source_name,n_source = np.unique(stellar_ext_source, return_counts=True)
    for s,n_s in zip(source_name, n_source):
        pct = 100 * n_s / n
        s = s.decode()
        if not len(s):
            s = '-'
        logging.info(f'{fid}: reddening: {n_s} sources use {s} ({pct:.3g}%)')

    # Sample XP spectra
    flux = np.zeros((n,n_wl+n_b), dtype='f4')
    flux_err = np.full((n,n_wl+n_b), np.nan, dtype='f4')
    flux_sqrticov = np.zeros((n,n_wl+n_b,n_wl+n_b), dtype='f4')
    flux_cov_eival_min = np.empty(n, dtype='f4')
    flux_cov_eival_max = np.empty(n, dtype='f4')
    for j in range(n):
        # Get sampled spectrum and covariance matrix
        fl,fl_err,fl_cov,_,_ = xp_sampler.sample(
            *[d_xp[k][j] for k,_,_ in bprp_fields],
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

    # Fix norm_dg (if NaN, then no close neighbor)
    idx = np.isfinite(d_meta['norm_dg'])
    d_meta['norm_dg'][~idx] = -50. # TODO: Is this correct?

    # Load unWISE photometry
    unwise_fn = os.path.join(
        'data', 'xp_unwise_match',
        f'xp_unwise_match_{fid}.h5'
    )
    with h5py.File(unwise_fn, 'r') as f:
        sid = f['gdr3_source_id'][:]
        d_unwise = f['unwise_data'][:]
        sep_unwise = f['sep_arcsec'][:]
    idx,idx_insert_unwise = get_match_indices(sid, gdr3_source_id)
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

    # Replace bad fluxes with something +- infinity
    for b,idx in enumerate(idx_unwise_good.T):
        unwise_flux[~idx,b] = 1e4 * unwise_f0[0,b] # A typical WISE flux value
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
    idx,idx_insert_tmass = get_match_indices(
        d_tmass['source_id'],
        gdr3_source_id
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

    # Replace NaN 2MASS mags or mag_errs with something +- infinity
    for b in range(tmass_mag.shape[1]):
        idx_nan = ~(
            np.isfinite(tmass_mag[:,b])
          & np.isfinite(tmass_mag_err[:,b])
        )
        tmass_mag[idx_nan,b] = 12. # A typical 2MASS magnitude
        tmass_mag_err[idx_nan,b] = np.inf

    # Calculate 2MASS fluxes, in units of the flux scale
    # (which is 10^{-18} W/m^2/nm, by default)
    tmass_flux = 10**(-0.4*tmass_mag) * tmass_f0
    tmass_flux_err = 2.5/np.log(10) * tmass_mag_err * tmass_flux

    ## Replace NaNs with default values
    #for b in range(tmass_flux.shape[1]):
    #    idx_nan = ~np.isfinite(tmass_flux[:,b])
    #    tmass_flux[idx_nan,b] = np.nanmedian(tmass_flux[:,b])
    #    tmass_flux_err[idx_nan,b] = np.inf

    # Paste in 2MASS photometric bands
    flux[idx_insert_tmass,-5:-2] = tmass_flux
    flux_err[idx_insert_tmass,-5:-2] = tmass_flux_err
    for b in range(3):
        flux_sqrticov[idx_insert_tmass,-5+b,-5+b] = 1/tmass_flux_err[:,b]

    # Compile all the information
    d = {
        'gdr3_source_id': gdr3_source_id,
        'flux': flux,
        'flux_err': flux_err,
        'flux_sqrticov': flux_sqrticov,
        'flux_cov_eival_min': flux_cov_eival_min,
        'flux_cov_eival_max': flux_cov_eival_max,
        'stellar_ext': stellar_ext,
        'stellar_ext_err': stellar_ext_err,
        'stellar_ext_source': stellar_ext_source
    }

    # Check for NaN in fluxes and extinctions
    for key in ('flux','flux_err','stellar_ext','stellar_ext_err'):
        try:
            assert np.all(~np.isnan(d[key]))
        except AssertionError as err:
            print(f'NaN {key} encountered here:')
            idx = np.where(np.isnan(d[key]))
            print(idx)
            raise err

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

    # Ensure that certain columns are in float32 format
    f4_keys = [
        'stellar_ext', 'stellar_ext_err',
        'plx', 'plx_err',
        'flux', 'flux_err', 'flux_sqrticov'
    ]
    for key in f4_keys:
        d[key] = f[key].astype('f4')

    if match_source_ids is None:
        return d, sample_wavelengths
    else:
        return d, idx_matches, sample_wavelengths


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

            d,idx_params,wl = extract_fluxes(
                fid,
                match_source_ids=d_params['gdr3_source_id'],
                thin=args.thin[1]
            )
            if d is None:
                continue

            d['stellar_type'] = d_params['params_est'][idx_params]
            d['stellar_type_err'] = d_params['params_err'][idx_params]
            d['stellar_type'] = d['stellar_type'].astype('f4')
            d['stellar_type_err'] = d['stellar_type_err'].astype('f4')

            # Append data from this XP file to the output file
            append_to_h5_file(fout, d)

        fout['flux'].attrs['sample_wavelengths'] = wl

    return 0

if __name__ == '__main__':
    main()

