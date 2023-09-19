 #!/usr/bin/env python

from __future__ import print_function, division

import numpy as np
import matplotlib.pyplot as plt

import h5py

import os
from glob import glob
from tqdm import tqdm
import hashlib
import time

import tensorflow as tf
# Disable usage of TensorFloat32, which is a reduced-precision format.
# This is very important for proper convergence of the fits.
tf.config.experimental.enable_tensor_float_32_execution(False)

from xp_utils import XPSampler, sqrt_icov_eigen, calc_invs_eigen
from model import FluxModel, GaussianMixtureModel, \
                              optimize_stellar_params, \
                              grid_search_stellar_params, \
                              calc_stellar_fisher_hessian


def load_data(fname):
    d = {}

    with h5py.File(fname, 'r') as f:
        for key in f:
            d[key] = f[key][:]
            
        sample_wavelengths = np.load('wl.npy')
        #f['flux'].attrs['sample_wavelengths'][:]

    req_keys = (
        'gdr3_source_id',
        'flux', 'flux_sqrticov', 'flux_err',
        'plx', 'plx_err'
    )
    for key in req_keys:
        try:
            assert key in d
        except AssertionError as err:
            print(f'{key} not in {fname} !')
            raise err

    # Identify sources with NaN fluxes
    idx_goodflux = np.all(np.isfinite(d['flux']), axis=1)
    if not np.all(idx_goodflux):
        n_bad = np.count_nonzero(~idx_goodflux)
        print(f'{n_bad} sources with NaN fluxes!')

    # Identify sources with high condition number
    #idx_wellcond = (d['flux_cov_eival_max']/d['flux_cov_eival_min'] < 1e5)
    #n_illcond = np.count_nonzero(~idx_wellcond)
    #print(f'{n_illcond} sources with ill-conditioned flux covariances!')

    # Remove bad sources
    idx = idx_goodflux #& idx_wellcond
    if not np.all(idx):
        n_bad = np.count_nonzero(~idx)
        print(f'  -> Removing {n_bad} sources.')
        for key in d:
            d[key] = d[key][idx]

    # Include extinction prior
    d['stellar_ext'] = np.zeros_like(d['plx'])
    d['stellar_ext_err'] = np.full_like(d['plx'], np.inf)

    # Replace non-finite parallaxes with flat prior
    idx_bad_plx = ~np.isfinite(d['plx']) | ~np.isfinite(d['plx_err'])
    d['plx'][idx_bad_plx] = 1.
    d['plx_err'][idx_bad_plx] = np.inf

    ## Check for non-finite parallaxes
    #n_bad_plx = np.count_nonzero(~np.isfinite(d['plx']))
    #n_bad_plx_err = np.count_nonzero(~np.isfinite(d['plx_err']))
    #if n_bad_plx or n_bad_plx_err:
    #    print(f'{n_bad_plx} NaN/Inf parallaxes!')
    #    print(f'{n_bad_plx_err} NaN/Inf parallax errors!')

    return d, sample_wavelengths


def calc_param_cov(d):
    n_stars = d['gdr3_source_id'].size

    # Calculate covariance matrix of stellar parameters
    icov = d['fisher'][:].copy()
    for k,ivar in enumerate(d['ivar_priors']):
        icov[k] += np.diag(ivar)
    cov,eival0 = calc_invs_eigen(
        icov,
        eival_floor=1e-6,
        return_min_eivals=True
    )

    err = np.empty((n_stars, 6), dtype='f4')
    for k,c in enumerate(cov):
        err[k] = np.sqrt(np.diag(c))

    d['stellar_params_icov'] = icov
    d['stellar_params_cov'] = cov
    d['stellar_params_icov_eival_min'] = eival0
    d['stellar_params_err'] = err

    # Combine all stellar parameters into one array
    d['stellar_params_est'] = np.concatenate([
        d['stellar_type_est'],
        np.reshape(d['xi_est'], (-1,1)),
        np.reshape(d['stellar_ext_est'], (-1,1)),
        np.reshape(d['plx_est'], (-1,1))
    ], axis=1)

    # Calculate chi^2/dof
    d['rchi2_opt'] = d['chi2_opt'] / 61.

    return d


def save_opt_stellar_params(data, fname, overwrite=False):
    keys = ('gdr3_source_id', 'stellar_type_est', 'xi_est', 'stellar_ext_est', 'plx_est')
    keys_opt = (
        'ra', 'dec',
        'phot_g_mean_flux', 'phot_g_mean_flux_error',
        'phot_bp_mean_flux', 'phot_bp_mean_flux_error',
        'phot_rp_mean_flux', 'phot_rp_mean_flux_error',
        'fisher', 'hessian',
        'ivar_priors', 'hessian_gmm',
        'chi2_opt',
        'stellar_params_est',
        'stellar_params_err',
        'stellar_params_icov',
        'stellar_params_cov',
        'stellar_params_icov_eival_min',
        'rchi2_opt',
        'plx','plx_err',
        'ln_prob'
    )
    kw = dict(compression='lzf', chunks=True)
    mode = 'w' if overwrite else 'a'
    with h5py.File(fname, mode) as f:
        for k in keys:
            print(f'Saving {k} ...')
            f.create_dataset(k, data=data[k], **kw)
        for k in keys_opt:
            if k not in data:
                print(f'Skipping {k} ...')
                continue
            print(f'Saving {k} ...')
            f.create_dataset(k, data=data[k], **kw)


def calc_icov_eigenvalues(data):
    d = data

    icov = np.stack([
        d['hessian'][i] + d['hessian_gmm'][i] + np.diag(d['ivar_priors'][i])
        for i in range(len(d['hessian']))
    ], axis=0)

    eival_min = []
    eival_max = []
    for ic in icov:
        val,vec = np.linalg.eigh(ic)
        eival_min.append(val[0])
        eival_max.append(val[-1])
    eival_min = np.array(eival_min)
    eival_max = np.array(eival_max)

    return eival_min, eival_max


def main():
    from argparse import ArgumentParser
    parser = ArgumentParser(
        description='Optimize stellar parameters.',
        add_help=True
    )
    parser.add_argument(
        '-i', '--input-dir',
        type=str,
        default='data/xp_opt_input_files',
        help='Directory with input files.'
    )
    parser.add_argument(
        '-o', '--output-dir',
        type=str,
        default='data/xp_opt_output_files',
        help='Directory for output files.'
    )
    parser.add_argument(
        '--partition',
        type=int,
        nargs=2,
        help='Only handle fraction input files: (# of partitions, partition).'
    )
    args = parser.parse_args()

    # Input files
    in_fnames = glob(os.path.join(
        args.input_dir,
        '*.h5'
    ))
    in_fnames.sort()
    n_files = len(in_fnames)
    print(f'{n_files} input files found.')

    # Choose a subset of input files to work with
    if args.partition is not None:
        n_parts,part = args.partition
        # Use an MD5 hash of the filename to deterministically assign
        # input filenames to subsets
        in_hash = [
            int(hashlib.md5(fn.encode('utf-8')).hexdigest(), base=16)
            for fn in in_fnames
        ]
        in_fnames = [fn for fn,h in zip(in_fnames,in_hash) if h%n_parts==part]
        n_files = len(in_fnames)
        print(f'{n_files} input files in partition {part+1} of {n_parts}.')

    # Output file pattern
    out_fname_pattern = os.path.join(
        args.output_dir,
        'xp_opt_output_{}.h5'
    )

    # Load stellar flux model and stellar type priors
    print('Loading trained flux model ...')
    stellar_model = FluxModel.load('data/output01/models/flux/xp_spectrum_model_final_Rv-1')
    print('Loading Gaussian Mixture Model prior on stellar type ...')
    stellar_type_prior = GaussianMixtureModel.load('models/prior/gmm_prior-1')
    
    # Calculate the clipping limit of ln_prior
    samples = stellar_type_prior.sample(1024*1024)
    sample_ln_prior = stellar_type_prior.ln_prob(samples)
    ln_prior_clip = np.percentile(sample_ln_prior, 0.1).astype('f4')
    samples = 0

    # Loop over input files, optimizing all stars in each file
    for i,ifn in enumerate(tqdm(in_fnames)):
        fid = ifn.split('_')[-1].split('.')[0]
        ofn = out_fname_pattern.format(fid)

        # Skip file if output already exists
        if os.path.exists(ofn):
            continue

        print(f'Loading {ifn} ...')
        d, sample_wl = load_data(ifn)
        n_stars = len(d['plx'])

        print('Performing rough grid search on stellar parameters ...')
        grid_search_stellar_params(
            stellar_model, d,
            gmm_prior=stellar_type_prior,
            batch_size=128
        )
        print(
            'parallax estimate / observation:',
            np.percentile(d['plx_est']/d['plx'], [1., 16., 50., 84., 99.])
        )

        kw = dict(
            lr_init=0.01,
            optimizer='adam',
            batch_size=min([n_stars, 1024*32]),
            n_steps=1024*12,
            reduce_lr_every=128*6,
            ln_prior_clip=ln_prior_clip,
            use_prior=stellar_type_prior
        )

        t = [time.monotonic()]

        for k in range(8): # Max. 8 rounds of optimization
            # Use SGD to refine parameters
            if k > 0:
                print(f'Stellar parameter refinement round {k} ...')
                kw['optimizer'] = 'sgd'
                kw['optimize_subset'] = np.where(idx_neg_eival)[0]
                kw['batch_size'] = min([n_neg_eival, 1024*32])
                kw['n_steps'] = 1024*12
                kw['reduce_lr_every'] = 1024*4
                kw['lr_init'] = 1e-5 / 2**(2*(k-1))

            print(f'Optimizing stellar parameters with {kw["optimizer"]} ...')
            optimize_stellar_params(stellar_model, d, **kw)

            print('Estimating uncertainties on stellar parameters ...')
            calc_stellar_fisher_hessian(
                stellar_model, d,
                ln_prior_clip=ln_prior_clip,
                gmm=stellar_type_prior,
                batch_size=1024
            )

            print('Calculating eigenvalues of inverse covariance matrices ...')
            eival_min, eival_max = calc_icov_eigenvalues(d)

            # Identify stars that have negative icov eigenvalues
            idx_neg_eival = (eival_min <= 0.)
            n_neg_eival = np.count_nonzero(idx_neg_eival)
            pct_neg = n_neg_eival / n_stars * 100.
            print(f'{n_neg_eival} sources ({pct_neg:.3f}%) have '
                   'negative eigenvalues in their icov.')
            if n_neg_eival > 0:
                print('  - indices:', np.where(idx_neg_eival)[0])
                cond_med = np.nanmedian(
                    d['flux_cov_eival_max'] / d['flux_cov_eival_min']
                )
                flux_eival_min = d['flux_cov_eival_min'][idx_neg_eival]
                flux_eival_max = d['flux_cov_eival_max'][idx_neg_eival]
                print('  - min flux eivals:', flux_eival_min)
                print('  - max flux eivals:', flux_eival_max)
                print(
                    f'  - flux cov cond # (med={cond_med:.3g}):',
                    flux_eival_max/flux_eival_min
                )
                print(
                    r'  - min icov eivals {0,16,50,84,100}%:',
                    np.percentile(eival_min[idx_neg_eival], [0,16,50,84,100])
                )
                print(
                    r' - sum(min icov eivals):',
                    np.sum(eival_min[idx_neg_eival])
                )

            t.append(time.monotonic())
            dt = t[-1] - t[-2]
            print(f'Elapsed time for optimization round: {dt:.4f} s')

            # Target <0.1% of sources having negative eigvals in cov matrix
            if n_neg_eival < 0.001 * n_stars:
                break

        # Calculate standard covariance matrices
        print('Calculate standard covariance matrices ...')
        calc_param_cov(d)

        # Calculate prior
        print('Calculating prior')
        d['ln_prob']=stellar_type_prior.ln_prob(d['stellar_params_est'][:,:3])

        # Save updated stellar parameters, as well an uncertainty estimates
        print('Saving results ...')
        save_opt_stellar_params(d, ofn, overwrite=True)

        dt = t[-1] - t[0]
        print(f'Time elapsed for file: {dt:.4f} s')

    return 0

if __name__ == '__main__':
    main()

