import numpy as np

import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, AutoMinorLocator
from matplotlib.colors import Normalize, LogNorm

import tensorflow as tf

from argparse import ArgumentParser
from pathlib import Path
from tqdm import tqdm
import h5py
import os
import os.path

from xp_utils import XPSampler, sqrt_icov_eigen, calc_invs_eigen
from utils import batch_apply_tf
from model import GaussianMixtureModel, FluxModel, chi_band, gaussian_prior, \
                  grads_stellar_model, grads_stellar_params, \
                  get_batch_iterator,  identify_outlier_stars, \
                  identify_flux_outliers, train_stellar_model, \
                  plot_gmm_prior, assign_variable_padded, corr_matrix, \
                  calc_stellar_fisher_hessian, load_data, save_as_h5, load_h5
import model
import plot_utils

nscale = 1.

def load_training_data(fname, validation_frac=0.2, seed=1, thin=1):
    # Load training data
    with h5py.File(fname, 'r') as f:
        d = {key:f[key][:][::thin] for key in f.keys()}
        sample_wavelengths = f['flux'].attrs['sample_wavelengths'][:]
    #sample_wavelengths = np.load('wl.npy').astype('f4')

    # Ensure that certain fields are in float32 format
    f4_keys = [
        'flux', 'flux_err', 'flux_sqrticov',
        'flux_cov_eival_max', 'flux_cov_eival_min',
        'plx', 'plx_err',
        'stellar_ext', 'stellar_ext_err',
        'stellar_type', 'stellar_type_err'
    ]
    for k in f4_keys:
        d[k] = d[k].astype('f4')

    # Warn about NaNs
    check_nan_keys = [
        'flux', 'flux_err', 'flux_sqrticov',
        'flux_cov_eival_max', 'flux_cov_eival_min',
        'plx', 'plx_err',
        'stellar_type', 'stellar_type_err'
    ]
    for k in check_nan_keys:
        if np.any(np.isnan(d[k])):
            raise ValueError(f'NaNs detected in {k}!')

    # Replace NaN extinctions with 0 +- infinity
    idx = np.isnan(d['stellar_ext']) | np.isnan(d['stellar_ext_err'])
    print(f'Replacing {np.count_nonzero(idx)} NaN extinctions with 0+-inf.')
    d['stellar_ext'][idx] = 0.
    d['stellar_ext_err'][idx] = np.inf

    # Shuffle data, and put last X% into validation set
    rng = np.random.default_rng(seed=seed)
    n = len(d['gdr3_source_id'])
    shuffle_idx = np.arange(n, dtype='i8')
    rng.shuffle(shuffle_idx)

    n_val = int(np.ceil(n * validation_frac))
    d_train = {'shuffle_idx': shuffle_idx[:-n_val]}
    d_val = {'shuffle_idx': shuffle_idx[-n_val:]}
    
    # Copy data into training and validation sets
    for key in d:
        d_train[key] = d[key][d_train['shuffle_idx']]
        d_val[key] = d[key][d_val['shuffle_idx']]

    return d_train, d_val, sample_wavelengths


def down_sample_weighing(x_ini, all_x, bin_edges,
                         n_bins=100, max_upsampling=5.):
    # Use high-Extinction stars for empirical distribution of xi
    bin_edges = np.hstack([[-np.inf], bin_edges, [np.inf]])
    
    # Calculate the emperical distribution of x_ini under the given bins
    bin_indices = np.digitize(x_ini, bin_edges)
    counts = np.bincount(bin_indices, minlength=n_bins+3)
    weights = counts / counts.sum()  # convert counts to probabilities
    weights_per_bin = 1./(weights+0.05)

    weights_per_bin[0] = 0
    weights_per_bin[-1] = 0

    # Weigh all stars by the inverse of density of the ini sample
    bin_indices_all = np.digitize(all_x, bin_edges)
    weights_per_star = weights_per_bin[bin_indices_all]

    # Normalize the weights per star by median value
    weights_per_star /= (1e-5 + np.median(weights_per_star))

    weights_per_star = soft_clip_weights(weights_per_star, max_upsampling)
    
    return weights_per_star.astype('f4')


def plot_param_histograms_1d(stellar_type, weights, title, fname):
    n_params = stellar_type.shape[1]
    figsize = (3*n_params,3)

    fig,ax_arr = plt.subplots(
        1, n_params,
        figsize=figsize,
        layout='constrained'
    )

    for i,(ax,p) in enumerate(zip(ax_arr.flat,stellar_type.T)):
        p_range = (np.min(p), np.max(p))

        kw = dict(
            range=p_range, bins=100,
            log=True, density=True,
            histtype='step',
            alpha=0.8
        )
        ax.hist(p, label=r'$\mathrm{raw}$', **kw)
        ax.hist(p, label=r'$\mathrm{weighted}$', weights=weights, **kw)

        ax.xaxis.set_minor_locator(AutoMinorLocator())
        ax.grid(True, which='major', alpha=0.2)
        ax.grid(True, which='minor', alpha=0.05)

        ax.set_xlabel(rf'$\mathrm{{parameter\ {i}}}$')
        if i == 0:
            ax.legend()

    fig.suptitle(title)

    fig.savefig(fname)
    plt.close(fig)
    

def soft_clip_weights(weights, max_upsampling):
    return weights / (1 + weights/max_upsampling)

    
def weigh_prior(stellar_type_prior, d_train, scale=1., max_upsampling=5.):
    all_ln_prior = batch_apply_tf(
        stellar_type_prior.ln_prob,
        1024,
        d_train['stellar_type'],
        function=True,
        progress=True,
        numpy=True
    )
    all_prior = np.exp(scale*all_ln_prior)
    all_prior /= np.max(all_prior)
    weight_per_star = 1 / (all_prior + 1e-9)
        
    # Weigh stars for better representation
    weight_per_star = soft_clip_weights(weight_per_star, max_upsampling)
    
    return weight_per_star.astype('f4')


def train(data_fname, output_dir, stage=0, thin=1, E_low=0.1, n_epochs=256):
    '''
    Stage 0: Train the stellar model, using universal extinction curve.
    Stage 1: Train the extinction model with hqlE stars, using initial guess of 
                    the slope of extinction curve.
    Stage 2: Train both extinction model and stellar model with hqlE stars
    Stage 3: Self-cleaning & Further optimize the model
    '''

    # General training parameters
    batch_size = 512
    n_bins = 100
    
    loss_hist = []

    # Ensure that output directories exist
    path_list = [
        'data', 'plots',
        'models/prior', 'models/flux',
        'index', 'hist_loss'
    ]
    for p in path_list:
        Path(os.path.join(output_dir,p)).mkdir(exist_ok=True, parents=True)

    # Helper function: save an h5 file, appending a base directory to filename
    def full_fn(fn):
        return os.path.join(output_dir, fn)
    
    if stage == 0:        
        
        # Stage 0, begin without initial stellar model
        print(f'Loading training data from {data_fname} ...')
        d_train, d_val, sample_wavelengths = load_training_data(
            data_fname,
            thin=thin,
            seed=0,
        )
        n_train, n_val = [len(d['plx']) for d in (d_train,d_val)]
        print(f'Loaded {n_train} ({n_val}) training (validation) sources.')

        plot_training_data(d_train, output_dir=output_dir)
        
        # Initial guess of xi
        d_train['xi'] = np.zeros(n_train, dtype='f4')
        d_val['xi'] = np.zeros(n_val, dtype='f4')

        save_as_h5(d_val, full_fn('data/d_val.h5'))
        save_as_h5(d_train, full_fn('data/d_train.h5'))
        
        # Generate GMM prior
        fn_prior = full_fn('models/prior/gmm_prior')
        if os.path.exists(fn_prior+'-1.index'):
            print('Loading existing Gaussian Mixture Model prior ...')
            stellar_type_prior = GaussianMixtureModel.load(fn_prior+'-1')
        else:
            print('Generating Gaussian Mixture Model prior on stellar type ...')
            stellar_type_prior = GaussianMixtureModel(3, n_components=16)
            # Use at most this many stars to learn GMM prior
            n_prior_max = 1024*256 
            stellar_type_prior.fit(d_train['stellar_type'][:n_prior_max])
            stellar_type_prior.save(fn_prior)
            print('  -> Plotting prior ...')
            plot_gmm_prior(stellar_type_prior, base_path=output_dir)
        
        weights_per_star = weigh_prior(stellar_type_prior, d_train)
        
        print('Plotting stellar-type histograms ...')
        plot_param_histograms_1d(
            d_train['stellar_type'],
            weights_per_star,
            r'$\mathrm{Training\ distribution:\ stellar\ type}$',
            os.path.join(output_dir, 'plots/training_stellar_type_hist1d')
        )
        
        # Generate tracks through stellar parameter space
        tracks_fn = full_fn('models/prior/tracks.h5')
        if os.path.exists(tracks_fn):
            print('Loading tracks through stellar parameter space ...')
            atm_tracks = model.load_stellar_type_tracks(tracks_fn)
        else:
            print('Generating tracks through stellar parameter space ...')
            atm_tracks = model.calculate_stellar_type_tracks(stellar_type_prior)
            model.save_stellar_type_tracks(atm_tracks, tracks_fn)
        print('Plotting tracks through stellar parameter space ...')
        for track in atm_tracks:
            plot_gmm_prior(
                stellar_type_prior,
                base_path=output_dir,
                overlay_track=track
            )
        
        # Initialize the parameter estimates at their measured (input) values
        for key in ('stellar_type', 'xi', 'stellar_ext', 'plx'):
            d_train[f'{key}_est'] = d_train[key].copy()
            d_val[f'{key}_est'] = d_train[key].copy()
        
        print('Creating flux model ...')
        n_stellar_params = d_train['stellar_type'].shape[1]
        p_low,p_high = np.percentile(
            d_train['stellar_type'],
            [16.,84.],
            axis=0
        )
        
        print('Training flux model on high-quality data ...')
        # Select a subset of "high-quality" stars with good measurements
        idx_hq = np.where(
            (d_train['plx']/d_train['plx_err'] > 10.)
          & np.all(d_train['stellar_type_err']<0.2, axis=1) # Type uncertainty
          & (d_train['stellar_ext_err'] < 0.1)
        )[0]
        n_hq = len(idx_hq)
        pct_hq = n_hq / d_train['plx'].shape[0]
        print(f'Training on {n_hq} ({pct_hq:.3%}) high-quality stars ...')
        
        stellar_model = FluxModel(
            sample_wavelengths, n_input=n_stellar_params,
            input_zp=np.median(d_train['stellar_type'],axis=0),
            input_scale=0.5*(p_high-p_low),
            hidden_size=32,
            l2=5., l2_ext_curve=0.5
        )
        
        # First, train the model with stars with good measurements,
        # with fixed slope of ext_curve
        title = (
            r'$\mathrm{Training\ distribution'
            r'\ (step\ 0a:\ HQ):'
            r'\ stellar\ type}$'
        )
        plot_param_histograms_1d(
            d_train['stellar_type'][idx_hq],
            weights_per_star[idx_hq],
            title,
            full_fn('plots/training_stellar_type_hist1d_step0a')
        )
        ret = train_stellar_model(
            stellar_model,
            d_train,
            weights_per_star,
            idx_train=idx_hq,
            optimize_stellar_model=True,
            optimize_stellar_params=False,
            batch_size=batch_size,
            n_epochs=n_epochs,
            model_update=['stellar_model','ext_curve_b'],
        )
        save_as_h5(ret, full_fn('hist_loss/hist_step0a.h5'))
        loss_hist.append(ret)
        stellar_model.save(full_fn('models/flux/xp_spectrum_model_initial'))
        
        # Plot stellar model along tracks through parameter space
        print('Plotting stellar model ...')
        for i,track in enumerate(atm_tracks):
            fig,ax = model.plot_stellar_model(stellar_model, track)
            fig.savefig(full_fn(f'plots/stellar_model_step0a_track{i}'))
            plt.close(fig)

        fig,_ = plot_roughness(d_train['stellar_type'], stellar_model)
        fig.savefig(full_fn('plots/roughness_step0a'))
        plt.close(fig)
        
        fig,_ = model.plot_extinction_curve(stellar_model, show_variation=False)
        fig.savefig(full_fn('plots/extinction_curve_step0a'))
        plt.close(fig)

        # Next, simultaneously train the stellar model and update stellar
        # parameters, using only the HQ data
        print('Training flux model and optimizing high-quality stars ...')
        ret = train_stellar_model(
            stellar_model,
            d_train,
            weights_per_star,
            idx_train=idx_hq,
            optimize_stellar_model=True,
            optimize_stellar_params=True,
            lr_model_init=1e-7*nscale,
            lr_stars_init=1e-5*nscale,
            batch_size=batch_size,
            n_epochs=n_epochs,
            model_update=['stellar_model','ext_curve_b'],
            var_update = ['atm','E','plx'],
        )
        save_as_h5(ret, full_fn('hist_loss/hist_step0b.h5'))
        loss_hist.append(ret)
        #plot_loss(ret, suffix='_intermediate')
        stellar_model.save(
            full_fn('models/flux/xp_spectrum_model_intermediate')
        )
        
        print('Plotting stellar model ...')
        for i,track in enumerate(atm_tracks):
            fig,ax = model.plot_stellar_model(stellar_model, track)
            fig.savefig(full_fn(f'plots/stellar_model_step0b_track{i}'))
            plt.close(fig)
        
        fig,_ = plot_roughness(d_train['stellar_type'], stellar_model)
        fig.savefig(full_fn('plots/roughness_step0b'))
        plt.close(fig)

        fig,_ = model.plot_extinction_curve(stellar_model, show_variation=False)
        fig.savefig(full_fn('plots/extinction_curve_step0b'))
        plt.close(fig)

        stellar_model = FluxModel.load(
            full_fn('models/flux/xp_spectrum_model_intermediate-1')
        )

        # Update E for stars with large E uncertainties (often, +-inf)
        print('Optimizing uncertain E estimates ...')
        idx_large_ext_err = np.where(d_train['stellar_ext_err'] > 0.1)[0]
        ret = train_stellar_model(
            stellar_model,
            d_train,
            weights_per_star,
            idx_train=idx_large_ext_err,
            optimize_stellar_model=False,
            optimize_stellar_params=True,
            batch_size=batch_size,
            lr_stars_init=1e-2,
            n_epochs=32*n_epochs,
            var_update=['E'],
        )

        # Next, update parameters of all the stars, holding the model fixed
        print('Optimizing all stars ...')
        ret = train_stellar_model(
            stellar_model,
            d_train,
            weights_per_star,
            optimize_stellar_model=False,
            optimize_stellar_params=True,
            batch_size=batch_size,
            n_epochs=n_epochs,
            var_update=['atm','E','plx'],
        )
        loss_hist.append(ret)

        # Self-cleaning: Identify outlier stars to exclude from further training,
        # using distance from priors
        idx_params_good = identify_outlier_stars(d_train)
        pct_good = np.mean(idx_params_good)
        print(f'Parameter outliers: {1-pct_good:.3%} of sources.')

        idx_flux_good = identify_flux_outliers(
            d_train, stellar_model,
            chi2_dof_clip=10.,
            #chi_indiv_clip=20.
        )
        
        pct_good = np.mean(idx_flux_good)
        print(f'Flux outliers: {1-pct_good:.3%} of sources.')

        idx_good = idx_params_good & idx_flux_good
        pct_good = np.mean(idx_good)
        print(f'Combined outliers: {1-pct_good:.3%} of sources.')
        
        title = (
            r'$\mathrm{Training\ distribution'
            r'\ (step\ 0c:\ cut\ flux/param\ outliers):'
            r'\ stellar\ type}$'
        )
        plot_param_histograms_1d(
            d_train['stellar_type'][idx_good],
            weights_per_star[idx_good],
            title,
            full_fn('plots/training_stellar_type_hist1d_step0c')
        )
        
        # Finally, simultaneously train the stellar model and update stellar
        # parameters, using all the (non-outlier) data
        print('Training flux model and optimizing all non-outlier stars ...')
        ret = train_stellar_model(
            stellar_model,
            d_train,
            weights_per_star,
            idx_train=np.where(idx_good)[0],
            optimize_stellar_model=True,
            optimize_stellar_params=True,
            lr_model_init=1e-7*nscale,
            lr_stars_init=1e-5*nscale,
            batch_size=batch_size,
            n_epochs=n_epochs,
            model_update=['stellar_model','ext_curve_b'],
            var_update = ['atm','E','plx'],
        )
        save_as_h5(ret, full_fn('hist_loss/hist_step0c.h5'))
        loss_hist.append(ret)
        
        # Plot stellar model
        print('Plotting stellar model ...')
        for i,track in enumerate(atm_tracks):
            fig,ax = model.plot_stellar_model(stellar_model, track)
            fig.savefig(full_fn(f'plots/stellar_model_step0c_track{i}'))
            plt.close(fig)
        
        fig,_ = plot_roughness(d_train['stellar_type'], stellar_model)
        fig.savefig(full_fn('plots/roughness_step0c'))
        plt.close(fig)

        fig,_ = model.plot_extinction_curve(stellar_model, show_variation=False)
        fig.savefig(full_fn('plots/extinction_curve_step0c'))
        plt.close(fig)
        
        np.save(full_fn('index/idx_good_wo_Rv.npy'), idx_good)
        stellar_model.save(full_fn('models/flux/xp_spectrum_model_final'))
        save_as_h5(d_train, full_fn('data/dtrain_final_wo_Rv.h5'))
        save_as_h5(ret, full_fn('hist_loss/final_wo_Rv.h5'))

    if stage<2:
         
        d_val = load_h5(full_fn('data/d_val.h5'))
        d_train = load_h5(full_fn('data/dtrain_final_wo_Rv.h5'))
        atm_tracks = model.load_stellar_type_tracks(
            full_fn('models/prior/tracks.h5')
        )
        
        stellar_model = FluxModel.load(
            full_fn('models/flux/xp_spectrum_model_final-1')
        )
        
        print('Loading Gaussian Mixture Model prior on stellar type ...')
        stellar_type_prior = GaussianMixtureModel.load(
            full_fn('models/prior/gmm_prior-1')
        )    
        
        # Initial weight of stars: equal
        weights_per_star = np.ones(len(d_train['plx']), dtype='f4')
        idx_hq = np.load(full_fn('index/idx_good_wo_Rv.npy'))
        
        # Run more steps in this stage
        n_epochs_1 = 256

        # Optimize the params of high-quality stars 
        print('Optimizing params of hq stars ...')
        ret = train_stellar_model(
            stellar_model,
            d_train,
            weights_per_star,
            idx_train=np.where(idx_hq)[0],
            optimize_stellar_model=False,
            optimize_stellar_params=True,
            #lr_model_init=1e-7,
            #lr_stars_init=1e-5,        
            batch_size=batch_size,
            n_epochs=n_epochs,
            var_update=['atm','E','plx'],
        )
        save_as_h5(ret, full_fn('hist_loss/hist_step1a.h5'))
        
        save_as_h5(d_train, full_fn('data/dtrain_final_wo_Rv_optimized.h5'))
        
        d_train = load_h5(full_fn('data/dtrain_final_wo_Rv_optimized.h5'))
        
        idx_large_E = (d_train['stellar_ext_est']>E_low)
        idx_hq_large_E = idx_hq & idx_large_E
        pct_use = np.mean(idx_hq_large_E)

        title = (
            r'$\mathrm{Training\ distribution'
            r'\ (step\ 1a:\ HQ,\ large\ E):'
            r'\ stellar\ type}$'
        )
        plot_param_histograms_1d(
            d_train['stellar_type'][idx_hq_large_E],
            weights_per_star[idx_hq_large_E],
            title,
            full_fn('plots/training_stellar_type_hist1d_step1a')
        )
        
        print(f'Learning (xi,E,plx) for {pct_use:.3%} of sources (high E) ...')
        ret = train_stellar_model(
            stellar_model,
            d_train,
            weights_per_star,
            idx_train=np.where(idx_hq_large_E)[0],
            optimize_stellar_model=False,
            optimize_stellar_params=True,
            #lr_model_init=1e-7,
            #lr_stars_init=1e-5,        
            batch_size=batch_size,
            n_epochs=n_epochs,
            var_update=['E','plx','xi'],
        )
        
        save_as_h5(d_train, full_fn('data/dtrain_initial_guess_xi.h5'))
        
        d_train = load_h5(full_fn('data/dtrain_initial_guess_xi.h5'))

        weights_per_star = down_sample_weighing( 
            d_train['xi_est'][idx_hq_large_E],
            d_train['xi_est'], 
            bin_edges = np.linspace(-0.8, 0.8, n_bins+1)
        )
        weights_per_star *= weigh_prior(stellar_type_prior, d_train)
        weights_per_star = soft_clip_weights(weights_per_star, 10.)

        title = (
            r'$\mathrm{Training\ distribution'
            r'\ (step\ 1b:\ HQ,\ large\ E):'
            r'\ stellar\ type}$'
        )
        plot_param_histograms_1d(
            d_train['stellar_type'][idx_hq_large_E],
            weights_per_star[idx_hq_large_E],
            title,
            full_fn('plots/training_stellar_type_hist1d_step1b')
        )
        
        print(f'Learning (xi,E) for {pct_use:.3%} of sources (high E)')
        print('... in addition to mean extinction curve ...')
        ret = train_stellar_model(
            stellar_model,
            d_train,
            weights_per_star,
            idx_train=np.where(idx_hq_large_E)[0],
            optimize_stellar_model=True,
            optimize_stellar_params=True,
            lr_model_init=1e-5,
            lr_stars_init=1e-5,
            batch_size=batch_size,
            n_epochs=n_epochs,
            var_update=['E','xi'],
            model_update=['ext_curve_b'],
        ) 
        save_as_h5(ret, full_fn('hist_loss/hist_step1b.h5'))

        fig,_ = model.plot_extinction_curve(stellar_model, show_variation=True)
        fig.savefig(full_fn('plots/extinction_curve_step1b'))
        plt.close(fig)
        
        stellar_model.save(full_fn('models/flux/xp_spectrum_model_initial_update_b'))
        
        stellar_model = FluxModel.load(
            full_fn('models/flux/xp_spectrum_model_initial_update_b-1')
        )
        
        print(f'Learning (xi,E) for {pct_use:.3%} of sources (high E)')
        print('... in addition to extinction curve variation ...')
        ret = train_stellar_model(
            stellar_model,
            d_train,
            weights_per_star,
            idx_train=np.where(idx_hq_large_E)[0],
            optimize_stellar_model=True,
            optimize_stellar_params=True,
            #lr_model_init=5e-6,
            lr_stars_init=1e-5,
            batch_size=batch_size,
            n_epochs=n_epochs_1,
            var_update=['E','xi'],
            model_update=['ext_curve_w'],
        ) 
        save_as_h5(ret, full_fn('hist_loss/hist_step1c.h5'))
        
        fig,_ = model.plot_extinction_curve(stellar_model, show_variation=True)
        fig.savefig(full_fn('plots/extinction_curve_step1c'))
        plt.close(fig)

        np.save(full_fn('index/idx_hq.npy'), idx_hq)
        # Save initial guess of xi
        save_as_h5(d_train, full_fn('data/dtrain_Rv_initial.h5'))
        save_as_h5(ret, full_fn('hist_loss/Rv_initial.h5'))
        stellar_model.save(full_fn('models/flux/xp_spectrum_model_initial_Rv'))
        
    if stage<3:
        
        #n_epochs = 512

        stellar_model = FluxModel.load(
            full_fn('models/flux/xp_spectrum_model_initial_Rv-1')
        )
        d_train = load_h5(full_fn('data/dtrain_Rv_initial.h5'))
        d_val = load_h5(full_fn('data/d_val.h5'))

        idx_hq = np.load(full_fn('index/idx_hq.npy')) 
        idx_large_E = (d_train['stellar_ext_est']>E_low)
        idx_hq_large_E = idx_hq & idx_large_E
        pct_use = np.mean(idx_hq_large_E)
        
        stellar_type_prior = GaussianMixtureModel.load(
            full_fn('models/prior/gmm_prior-1')
        )   
        atm_tracks = model.calculate_stellar_type_tracks(stellar_type_prior)
        
        weights_per_star = down_sample_weighing( 
            d_train['xi_est'][idx_hq_large_E],
            d_train['xi_est'], 
            bin_edges = np.linspace(-0.8, 0.8, n_bins+1)
        )
        weights_per_star *= weigh_prior(stellar_type_prior, d_train)
        weights_per_star = soft_clip_weights(weights_per_star, 10.)

        print(f'Learning everything for {pct_use:.3%} of sources (high E) ...')
        ret = train_stellar_model(
            stellar_model,
            d_train,
            weights_per_star,
            idx_train=np.where(idx_hq_large_E)[0],
            optimize_stellar_model=True,
            optimize_stellar_params=True,
            lr_model_init=1e-5,
            lr_stars_init=1e-4,
            batch_size=batch_size,
            n_epochs=2*n_epochs,
            var_update=['atm','E','plx','xi'],
            model_update=['stellar_model','ext_curve_w','ext_curve_b'],
        )
        save_as_h5(ret, full_fn('hist_loss/hist_step2.h5'))

        print('Plotting stellar model ...')
        for i,track in enumerate(atm_tracks):
            fig,ax = model.plot_stellar_model(stellar_model, track)
            fig.savefig(full_fn(f'plots/stellar_model_step2_track{i}'))
            plt.close(fig)
        
        fig,_ = plot_roughness(d_train['stellar_type'], stellar_model)
        fig.savefig(full_fn('plots/roughness_step2'))
        plt.close(fig)

        fig,_ = model.plot_extinction_curve(stellar_model, show_variation=True)
        fig.savefig(full_fn('plots/extinction_curve_step2'))
        plt.close(fig)

        fig,_ = model.plot_RV_histogram(stellar_model, d_train)
        fig.savefig(full_fn('plots/RV_histogram_step2'))
        plt.close(fig)
        
        #fig,_ = model.plot_RV_skymap(stellar_model, d_train)
        #fig.savefig(full_fn('plots/RV_skymap_step2'))
        #plt.close(fig)
        
        save_as_h5(d_train, full_fn('data/dtrain_Rv_intermediate_0.h5'))
        save_as_h5(ret, full_fn('hist_loss/Rv_intermediate_0.h5'))
        stellar_model.save(
            full_fn('models/flux/xp_spectrum_model_intermediate_Rv')
        )
                
    if stage<4:
        
        stellar_model = FluxModel.load(
            full_fn('models/flux/xp_spectrum_model_intermediate_Rv-1')
        )
        stellar_type_prior = GaussianMixtureModel.load(
            full_fn('models/prior/gmm_prior-1')
        )
        
        d_train = load_h5(full_fn('data/dtrain_Rv_intermediate_0.h5'))
        d_val = load_h5(full_fn('data/d_val.h5'))
        atm_tracks = model.load_stellar_type_tracks(
            full_fn('models/prior/tracks.h5')
        )
        #idx_hq = np.load(full_fn('index/idx_hq.npy')) 
     
        # Optimize all stellar params, in order to pick up 
        # stars that were rejected due to extinction variation law
        
        weights_per_star = np.ones(d_train['stellar_type'].shape[0]).astype('float32')
        
        print('Learning stellar parameters for all sources ...')
        ret = train_stellar_model(
            stellar_model,
            d_train,
            weights_per_star,
            #idx_train=idx_hq_large_E,
            optimize_stellar_model=False,
            optimize_stellar_params=True,
            batch_size=batch_size,
            n_epochs=n_epochs//2,
            var_update = ['atm','E','plx','xi'],
        )
       
        save_as_h5(d_train, full_fn('data/dtrain_Rv_intermediate_1.h5'))
        save_as_h5(ret, full_fn('hist_loss/Rv_intermediate_1.h5'))
        
        d_train = load_h5(full_fn('data/dtrain_Rv_intermediate_1.h5'))
        d_val = load_h5(full_fn('data/d_val.h5'))
        
        # Remove outliers
        idx_params_good = identify_outlier_stars( # TODO: rename to "identify_param_outliers"
            d_train,
            sigma_clip_teff=4., # TODO: Make this clipping be agnostic about the parameter names
            sigma_clip_logg=4.,
            sigma_clip_feh=4.,
        )
        pct_good = np.mean(idx_params_good)
        print(f'Parameter outliers: {1-pct_good:.3%} of sources.')

        idx_flux_good = identify_flux_outliers(
            d_train, stellar_model,
            chi2_dof_clip=10.,
            #chi_indiv_clip=20.
        )
        pct_good = np.mean(idx_flux_good)
        print(f'Flux outliers: {1-pct_good:.3%} of sources.')

        idx_good = idx_params_good & idx_flux_good
        pct_good = np.mean(idx_good)
        print(f'Combined outliers: {1-pct_good:.3%} of sources.')        
                
        idx_final_train = idx_good #& (d_train['stellar_ext_est']>0.1)
        pct_use = np.mean(idx_final_train)
        print(f'Training on {pct_use:.3%} of sources.')

        weights_per_star = down_sample_weighing( 
            d_train['xi_est'][idx_hq_large_E],
            d_train['xi_est'], 
            bin_edges = np.linspace(-0.8, 0.8, n_bins+1)
        )
        weights_per_star *= weigh_prior(stellar_type_prior, d_train)
        weights_per_star = soft_clip_weights(weights_per_star, 10.)

        title = (
            r'$\mathrm{Training\ distribution'
            r'\ (step\ 3a:\ cut\ flux/param\ outliers):'
            r'\ stellar\ type}$'
        )
        plot_param_histograms_1d(
            d_train['stellar_type'][idx_final_train],
            weights_per_star[idx_final_train],
            title,
            full_fn('plots/training_stellar_type_hist1d_step3a')
        )

        pct_use = np.mean(idx_final_train)
        print('Learning everything except extinction curve variation')
        print(f'using {pct_use:.3%} of sources (cut flux/param outliers) ...')
        ret = train_stellar_model(
            stellar_model,
            d_train,
            np.ones_like(weights_per_star, dtype='f4'),
            idx_train=np.where(idx_final_train)[0],
            optimize_stellar_model=True,
            optimize_stellar_params=True,
            lr_model_init=1e-5,
            lr_stars_init=1e-4,
            batch_size=batch_size,
            n_epochs=n_epochs//2,
            # Freeze extinction curve variation when training with
            # all stars (not necessarily having high E)
            var_update=['atm','E','plx','xi'],
            model_update=['stellar_model','ext_curve_b'], 
        )

        stellar_model.save(full_fn('models/flux/xp_spectrum_model_final_Rv'))
        save_as_h5(ret, full_fn('hist_loss/final_Rv.h5'))

        fig,ax = model.plot_stellar_model(stellar_model, track)
        fig.savefig(full_fn(f'plots/stellar_model_step3_track{i}'))
        plt.close(fig)
        
        fig,_ = plot_roughness(d_train['stellar_type'], stellar_model)
        fig.savefig(full_fn('plots/roughness_step3'))
        plt.close(fig)

        fig,_ = model.plot_extinction_curve(stellar_model, show_variation=True)
        fig.savefig(full_fn('plots/extinction_curve_step3'))
        plt.close(fig)

        fig,_ = model.plot_RV_histogram(stellar_model, d_train)
        fig.savefig(full_fn('plots/RV_histogram_step3'))
        plt.close(fig)
        
        fig,_ = model.plot_RV_skymap(stellar_model, d_train)
        fig.savefig(full_fn('plots/RV_skymap_step3'))
        plt.close(fig)
        
        stellar_model.save(full_fn('models/flux/xp_spectrum_model_final_Rv'))
        save_as_h5(ret, full_fn('hist_loss/final_Rv.h5'))

        print('Optimizing stellar parameters in training set ...')
        ret = train_stellar_model(
            stellar_model,
            d_train,
            weights_per_star,
            #idx_train=np.where(idx_final_train)[0],
            optimize_stellar_model=False,
            optimize_stellar_params=True,
            batch_size=batch_size,
            n_epochs=n_epochs,
            var_update = ['atm','E','plx','xi'],
        )

        fig,_ = model.plot_RV_histogram(stellar_model, d_train)
        fig.savefig(full_fn('plots/RV_histogram_final'))
        plt.close(fig)
        
        fig,_ = model.plot_RV_skymap(stellar_model, d_train)
        fig.savefig(full_fn('plots/RV_skymap_final'))
        plt.close(fig)
        
        for key in ('stellar_type', 'xi', 'stellar_ext', 'plx'):
            d_val[f'{key}_est'] = d_val[key].copy()
        
        print('Optimizing stellar parameters in validation set ...')
        ret = train_stellar_model(
            stellar_model,
            d_val,
            weights_per_star,
            #idx_train=np.where(idx_final_train)[0],
            optimize_stellar_model=False,
            optimize_stellar_params=True,
            batch_size=batch_size,
            n_epochs=n_epochs,
            var_update = ['atm','E','plx','xi'],
        )                   
        
        save_as_h5(d_train, full_fn('data/dtrain_Rv_final.h5'))
        save_as_h5(d_val, full_fn('data/dval_Rv_final.h5'))
        
        return 0


def plot_roughness(theta, stellar_model):
    roughness_plot = batch_apply_tf(
        lambda x: stellar_model.roughness(tf.convert_to_tensor(x),
                                          reduce_stars=False),
        1024, theta,
        function=True,
        numpy=True,
        progress=False
    )
    fig,ax_arr = plot_utils.projection_grid(
        theta,
        c=roughness_plot,
        labels=(r'$T_{\mathrm{eff}} / (1000\,\mathrm{K})$',
                r'$\left[\mathrm{Fe/H}\right] / \mathrm{dex}$',
                r'$\log g / \mathrm{dex}$'),
        extents=[(13.,3.),(-2.7,1.0),(5.2,-0.3)],
        scatter=False,
        clabel=(r'$\langle\left| '
                r'\nabla_{\!\theta}\, f_{\lambda}\, '
                r'\right|^2\rangle$'),
        hist_kw=dict(bins=100),
        #fig_kw=dict(dpi=200),
        norm=Normalize(0,5)
    )
    for ax in ax_arr.flat:
        ax.xaxis.set_minor_locator(AutoMinorLocator())
        ax.yaxis.set_minor_locator(AutoMinorLocator())
    rough = np.std(roughness_plot)
    fig.suptitle(rf'$\mathrm{{roughness}} = {rough:.4f}$')
    fig.subplots_adjust(
        left=0.14, right=0.97,
        bottom=0.12, top=0.92,
        hspace=0.07, wspace=0.07
    )
    return fig,ax_arr


def plot_camd(d, color_key=None, color_dim=None, low_extinction=False):
    fig,ax = plt.subplots(figsize=(6,6), layout='constrained')
    
    # Calculate M_G and BP-RP
    zp_g,zp_bp,zp_rp = (25.6884, 25.3514, 24.7619)
    m_g = -2.5*np.log10(d['phot_g_mean_flux']) + zp_g
    m_bp = -2.5*np.log10(d['phot_bp_mean_flux']) + zp_bp
    m_rp = -2.5*np.log10(d['phot_rp_mean_flux']) + zp_rp
    bprp = m_bp - m_rp
    g_absmag = m_g + 5*np.log10(d['plx']) - 10.

    # Only plot stars with well determined Gaia parallaxes
    idx = (d['plx'] / d['plx_err'] > 3.)

    # Additionally require decent photometry
    for b in ('g','bp','rp'):
        idx &= (d[f'phot_{b}_mean_flux']/d[f'phot_{b}_mean_flux_error']>5)

    # Select low-extinction stars?
    if low_extinction:
        idx &= (d['stellar_ext'] + d['stellar_ext_err'] < 0.3)

    bprp = bprp[idx]
    g_absmag = g_absmag[idx]

    bprp_lim = (-1.0, 5.0)
    g_absmag_lim = (15., -5.5)

    if color_key is None:
        _,_,_,im = ax.hist2d(
            bprp, g_absmag,
            range=[sorted(bprp_lim),sorted(g_absmag_lim)],
            bins=50,
            norm=LogNorm()
        )
        ax.set_xlim(bprp_lim)
        ax.set_ylim(g_absmag_lim)
        clabel = r'$\mathrm{density}$'
    else:
        c = d[color_key][idx]

        if color_dim is not None:
            c = c[:,color_dim]
        clim = plot_utils.choose_lim(c)

        im = plot_utils.hist2d_reduce(
            bprp, g_absmag, c,
            ax=ax,
            xlim=bprp_lim,
            ylim=g_absmag_lim,
            hist_kw=dict(bins=50),
            imshow_kw=dict(vmin=clim[0],vmax=clim[1])
        )

        clabel = color_key.replace(r'_', r'\_')
        if color_dim is not None:
            clabel = rf'{clabel}\_{color_dim}'
        clabel = rf'$\mathtt{{{clabel}}}$'

    cb = fig.colorbar(im, ax=ax, label=clabel)
    ax.set_xlabel(r'$BP-RP$')
    ax.set_ylabel(r'$m_G + 5\log_{10}\hat{\varpi} - 10$')

    return fig,ax
        

def plot_training_data(d, output_dir=''):
    fig,ax = plot_camd(d)
    fn = os.path.join(output_dir, 'plots', f'camd_density')
    fig.savefig(fn)
    plt.close(fig)

    n_params = d['stellar_type'].shape[1]
    for key in ('stellar_type','stellar_type_err'):
        for dim in range(n_params):
            print(f'Plotting {key}_{dim} ...')
            fig,ax = plot_camd(
                d, color_key=key, color_dim=dim,
                low_extinction=True
            )
            fn = os.path.join(output_dir, 'plots', f'camd_{key}_{dim}')
            fig.savefig(fn)
            plt.close(fig)

    print(f'Plotting extinctions ...')
    fig,ax = plot_camd(d, color_key='stellar_ext')
    fn = os.path.join(output_dir, 'plots', 'camd_extinction')
    fig.savefig(fn)
    plt.close(fig)


def main():
    parser = ArgumentParser(
        description='Train forward model of XP spectra.',
        add_help=True
    )
    parser.add_argument(
        '-i', '--input',
        type=str,
        required=True,
        help='Filename of training data.'
    )
    parser.add_argument(
        '-o', '--output-dir',
        type=str,
        required=True,
        help='Directory in which to store output.'
    )
    parser.add_argument(
        '--stage',
        type=int,
        default=0,
        help='Stage at which to begin training.'
    )
    parser.add_argument(
        '--thin',
        type=int,
        default=1,
        help='Only use every Nth star in the training set.'
    )
    parser.add_argument(
        '--E_low',
        type=float,
        default=0.1,
        help='Lower limit of extinction for ext curve variation.'
    )
    
    args = parser.parse_args()
    train(args.input, args.output_dir, stage=args.stage, thin=args.thin, E_low=args.E_low)

    return 0


if __name__=='__main__':
    main()
