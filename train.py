import numpy as np

import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, AutoMinorLocator

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


def load_training_data(fname, validation_frac=0.2, seed=1, thin=1):
    # Load training data
    with h5py.File(fname, 'r') as f:
        d = {key:f[key][:][::thin] for key in f.keys()}
        sample_wavelengths = f['flux'].attrs['sample_wavelengths'][:]

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


def prepare_training_data():
    print('Gathering training data into one file ...')
    sample_wavelengths = np.arange(392., 993., 10.)
    data_out_fname = 'data/training/xp_nn_training_data.h5'
    gather_data_into_file(
        data_out_fname,
        sample_wavelengths
    )
    

def down_sample_weighing(x_ini, all_x, bin_edges, n_bins=100):
    # Use high-Extinction stars for empirical distribution of xi
    bin_edges = np.hstack([[-np.inf], bin_edges, [np.inf]])
    
    # Calculate the emperical distribution of x_ini under the given bins
    bin_indices = np.digitize(x_ini, bin_edges)
    counts = np.bincount(bin_indices, minlength=n_bins+3)
    weights = counts / counts.sum()  # convert counts to probabilities
    weights_per_bin = 1./(weights+0.001)
    
    # Weigh all stars by the inverse of density of the ini sample
    bin_indices_all = np.digitize(all_x, bin_edges)
    weights_per_star = weights_per_bin[bin_indices_all]
    
    # Normalize the weights per star  by median value
    weights_per_star /= np.median(weights_per_star)
    
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


def train(data_fname, output_dir, stage=0, thin=1):
    '''
    Stage 0: Train the stellar model, using universal extinction curve.
    Stage 1: Train the extinction model with hqlE stars, using initial guess of 
                    the slope of extinction curve.
    Stage 2: Train both extinction model and stellar model with hqlE stars
    Stage 3: Self-cleaning & Further optimize the model
    '''

    # General training parameters
    n_epochs = 128
    batch_size = 1024
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
            thin=thin
        )
        n_train, n_val = [len(d['plx']) for d in (d_train,d_val)]
        print(f'Loaded {n_train} ({n_val}) training (validation) sources.')
        
        # Initial guess of xi
        d_train['xi'] = np.zeros(n_train, dtype='f4')
        d_val['xi'] = np.zeros(n_val, dtype='f4')

        save_as_h5(d_val, full_fn('data/d_val.h5'))
        save_as_h5(d_train, full_fn('data/d_train.h5'))
        
        # Generate GMM prior
        print('Generating Gaussian Mixture Model prior on stellar type ...')
        stellar_type_prior = GaussianMixtureModel(3, n_components=16)
        stellar_type_prior.fit(d_train['stellar_type'])
        stellar_type_prior.save(full_fn('models/prior/gmm_prior'))
        print('  -> Plotting prior ...')
        plot_gmm_prior(stellar_type_prior, base_path=output_dir)
                
        all_ln_prior = batch_apply_tf(
            stellar_type_prior.ln_prob,
            1024,
            d_train['stellar_type'],
            function=True,
            progress=True,
            numpy=True
        )
        all_prior = np.exp(all_ln_prior)
        all_prior /= np.max(all_prior)
        
        # Weigh stars for better representation
        #weights_per_star = np.exp(-d_train['stellar_type'][:,1]/2.)
        max_upsampling = 100.
        weights_per_star = (1./(all_prior+1/max_upsampling)).astype('f4')

        print('Plotting stellar-type histograms ...')
        plot_param_histograms_1d(
            d_train['stellar_type'],
            weights_per_star,
            r'$\mathrm{Training\ distribution:\ stellar\ type}$',
            os.path.join(output_dir, 'plots/training_stellar_type_hist1d')
        )
        
        # Generate tracks through stellar parameter space
        print('Generating tracks through stellar parameter space ...')
        atm_tracks = model.calculate_stellar_type_tracks(stellar_type_prior)
        model.save_stellar_type_tracks(
            atm_tracks,
            full_fn('models/prior/tracks.h5')
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
            l2=1., l2_ext_curve=1.
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
            d_train, d_val,
            weights_per_star,
            idx_train=idx_hq,
            optimize_stellar_model=True,
            optimize_stellar_params=False,
            batch_size=batch_size,
            n_epochs=n_epochs,
            model_update=['stellar_model','ext_curve_b'],
        )
        loss_hist.append(ret)
        stellar_model.save(full_fn('models/flux/xp_spectrum_model_initial'))
        
        # Plot stellar model along tracks through parameter space
        print('Plotting stellar model ...')
        for i,track in enumerate(atm_tracks):
            fig,ax = model.plot_stellar_model(stellar_model, track)
            fig.savefig(full_fn(f'plots/stellar_model_step0a_track{i}'))
            plt.close(fig)
        
        fig,ax = plot_extinction_curve(flux_model, show_variation=False)
        fig.savefig(full_fn('plots/extinction_curve_step0a'))
        plt.close(fig)

        # Next, simultaneously train the stellar model and update stellar
        # parameters, using only the HQ data
        print('Training flux model and optimizing high-quality stars ...')
        ret = train_stellar_model(
            stellar_model,
            d_train, d_val,
            weights_per_star,
            idx_train=idx_hq,
            optimize_stellar_model=True,
            optimize_stellar_params=True,
            lr_model_init=1e-5,
            batch_size=batch_size,
            n_epochs=n_epochs,
            model_update=['stellar_model','ext_curve_b'],
            var_update = ['atm','E','plx'],
        )
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
        
        fig,ax = plot_extinction_curve(flux_model, show_variation=False)
        fig.savefig(full_fn('plots/extinction_curve_step0b'))
        plt.close(fig)

        stellar_model = FluxModel.load(
            full_fn('models/flux/xp_spectrum_model_intermediate-1')
        )
        # Next, update parameters of all the stars, holding the model fixed
        print('Optimizing all stars ...')
        ret = train_stellar_model(
            stellar_model,
            d_train, d_val,
            weights_per_star,
            optimize_stellar_model=False,
            optimize_stellar_params=True,
            batch_size=batch_size,
            n_epochs=n_epochs,
            var_update = ['atm','E','plx'],
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
            d_train, d_val,
            weights_per_star,
            idx_train=np.where(idx_good)[0],
            optimize_stellar_model=True,
            optimize_stellar_params=True,
            lr_model_init=1e-7,
            lr_stars_init=1e-5,
            batch_size=batch_size,
            n_epochs=n_epochs,
            model_update=['stellar_model','ext_curve_b'],
            var_update = ['atm','E','plx'],
        )
        loss_hist.append(ret)
        
        # Plot stellar model
        print('Plotting stellar model ...')
        for i,track in enumerate(atm_tracks):
            fig,ax = model.plot_stellar_model(stellar_model, track)
            fig.savefig(full_fn(f'plots/stellar_model_step0c_track{i}'))
            plt.close(fig)
        
        fig,ax = plot_extinction_curve(flux_model, show_variation=False)
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
        
        #print('Calculating prior weight of stars in the training set')
        #all_ln_prior = []
        #teff_ini, feh_ini, logg_ini= d_train['stellar_type'].T
        #for i in tqdm(range(int(len(teff_ini)/10000)+1)):
        #    ln_prior = stellar_type_prior.ln_prob(
        #        np.vstack([
        #                teff_ini[i*10000: (i+1)*10000], 
        #                feh_ini[i*10000: (i+1)*10000], 
        #                logg_ini[i*10000: (i+1)*10000]]).T
        #    ).numpy()
        #    all_ln_prior.append(ln_prior)
        #teff_ini, feh_ini, logg_ini = 0,0,0
        #all_ln_prior = np.hstack(all_ln_prior)

        #print('Removing outliers in stellar types')
        #for key in tqdm(d_train.keys()):
        #    d_train[key] = d_train[key][all_ln_prior>-7.43]            
        
        # Initial weight of stars: equal
        weights_per_star = np.ones(len(d_train["plx"]), dtype='f4')
        
        idx_hq = np.load(full_fn('index/idx_good_wo_Rv.npy'))

        # Optimize the params of high-quality stars 
        n_epochs = 128
        print('Optimizing params of hq stars')
        ret = train_stellar_model(
            stellar_model,
            d_train, d_val,
            weights_per_star,
            idx_train=np.where(idx_hq)[0],
            optimize_stellar_model=False,
            optimize_stellar_params=True,
            #lr_model_init=1e-7,
            #lr_stars_init=1e-5,        
            batch_size=batch_size,
            n_epochs=n_epochs,
            var_update=['atm', 'E', 'plx'],
        )
        save_as_h5(d_train, full_fn('data/dtrain_final_wo_Rv_optimized.h5'))
        d_train = load_h5(full_fn('data/dtrain_final_wo_Rv_optimized.h5'))
        
        idx_hq_large_E = idx_hq & (d_train['stellar_ext_est']>0.1)
        pct_use = np.mean(idx_hq_large_E)
        print(f'Learning (xi, E, plx) for {pct_use:.3%} of sources.')

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
        
        ret = train_stellar_model(
            stellar_model,
            d_train, d_val,
            weights_per_star,
            idx_train=np.where(idx_hq_large_E)[0],
            optimize_stellar_model=False,
            optimize_stellar_params=True,
            #lr_model_init=1e-7,
            #lr_stars_init=1e-5,        
            batch_size=batch_size,
            n_epochs=n_epochs,
            var_update=['E', 'plx', 'xi'],
        )
        
        weights_per_star = down_sample_weighing( 
            d_train['xi_est'][idx_hq_large_E],
            d_train['xi_est'], 
            bin_edges = np.linspace(-1, 1, n_bins + 1)
        )
        
        weights_per_star /= (0.001+np.median(weights_per_star))

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
        
        ret = train_stellar_model(
            stellar_model,
            d_train, d_val,
            weights_per_star,
            idx_train=np.where(idx_hq_large_E)[0],
            optimize_stellar_model=True,
            optimize_stellar_params=True,
            #lr_model_init=1e-7,
            #lr_stars_init=1e-5,
            batch_size=batch_size,
            n_epochs=n_epochs,
            var_update=['E', 'plx', 'xi'],
            model_update=['ext_curve_w'],
        ) 
        
        ret = train_stellar_model(
            stellar_model,
            d_train, d_val,
            weights_per_star,
            idx_train=np.where(idx_hq_large_E)[0],
            optimize_stellar_model=True,
            optimize_stellar_params=True,
            lr_model_init=1e-7,
            lr_stars_init=1e-5,
            batch_size=batch_size,
            n_epochs=n_epochs,
            var_update=['E', 'plx', 'xi'],
            model_update=['ext_curve_w', 'ext_curve_b'],
        ) 
        
        # TODO: Plot extinction curve here

        np.save(full_fn('index/idx_with_Rv_good.npy'), idx_hq_large_E)
        # Save initial guess of xi
        save_as_h5(d_train, full_fn('data/dtrain_Rv_initial.h5'))
        save_as_h5(ret, full_fn('hist_loss/Rv_initial.h5'))
        stellar_model.save(full_fn('models/flux/xp_spectrum_model_initial_Rv'))
        
    if stage<3:
        
        n_epochs = 128
        
        stellar_model = FluxModel.load(
            full_fn('models/flux/xp_spectrum_model_initial_Rv-1')
        )
        d_train = load_h5(full_fn('data/dtrain_Rv_initial.h5'))
        d_val = load_h5(full_fn('data/d_val.h5'))

        idx_hq = np.load(full_fn('index/idx_good_wo_Rv.npy')) 
        
        weights_per_star = down_sample_weighing( 
            d_train['xi_est'][idx_hq],
            d_train['xi_est'], 
            bin_edges = np.linspace(-1, 1, n_bins + 1)
        )
        
        weights_per_star /= (0.001+np.median(weights_per_star))    
        # TODO: Multiply in prior-based weights
        
        ret = train_stellar_model(
            stellar_model,
            d_train, d_val,
            weights_per_star,
            idx_train=np.where(idx_hq)[0],
            optimize_stellar_model=True,
            optimize_stellar_params=True,
            lr_model_init=1e-7,
            lr_stars_init=1e-5,
            batch_size=batch_size,
            n_epochs=n_epochs,
            var_update = ['atm','E','plx','xi'],
            model_update = ['stellar_model', 'ext_curve_w', 'ext_curve_b'],
        )

        print('Plotting stellar model ...')
        for i,track in enumerate(atm_tracks):
            fig,ax = model.plot_stellar_model(stellar_model, track)
            fig.savefig(full_fn(f'plots/stellar_model_step2_track{i}'))
            plt.close(fig)
        
        fig,ax = plot_extinction_curve(flux_model, show_variation=True)
        fig.savefig(full_fn('plots/extinction_curve_step2'))
        plt.close(fig)
        
        save_as_h5(d_train, full_fn('data/dtrain_Rv_intermediate_0.h5'))
        save_as_h5(ret, full_fn('hist_loss/Rv_intermediate_0.h5'))
        stellar_model.save(
            full_fn('models/flux/xp_spectrum_model_intermediate_Rv')
        )
                
    if stage<4:
        
        stellar_model = FluxModel.load(
            full_fn('models/flux/xp_spectrum_model_intermediate_Rv-1')
        )
        
        d_train = load_h5(full_fn('data/dtrain_Rv_intermediate_0.h5'))
        d_val = load_h5(full_fn('data/d_val.h5'))
        atm_tracks = model.load_stellar_type_tracks(
            full_fn('models/prior/tracks.h5')
        )
     
        # Optimize all stellar params, in order to pick up 
        # stars that were rejected due to extinction variation law
        
        n_epochs = 128
        
        weights_per_star = np.ones(len(d_train['plx']),dtype='f4')
        ret = train_stellar_model(
            stellar_model,
            d_train, d_val,
            weights_per_star,
            #idx_train=idx_hq_large_E,
            optimize_stellar_model=False,
            optimize_stellar_params=True,
            batch_size=batch_size,
            n_epochs=n_epochs,
            var_update = ['atm','E','plx','xi'],
        )
       
        save_as_h5(d_train, full_fn('data/dtrain_Rv_intermediate_1.h5'))
        save_as_h5(ret, full_fn('hist_loss/Rv_intermediate_1.h5'))
        
        d_train = load_h5(full_fn('data/dtrain_Rv_intermediate_1.h5'))
        d_val = load_h5(full_fn('data/d_val.h5'))
        
        # Remove outliers
        idx_params_good = identify_outlier_stars( # TODO: rename to "identify_param_outliers"
            d_train,
            sigma_clip_teff=10., # TODO: Make this clipping be agnostic about the parameter names
            sigma_clip_logg=10.,
            sigma_clip_feh=10.,
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
            d_train['xi_est'][idx_final_train],
            d_train['xi_est'], 
            bin_edges = np.linspace(-1, 1, n_bins + 1)
        )
        weights_per_star /= (0.001+np.median(weights_per_star))    

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

        ret = train_stellar_model(
            stellar_model,
            d_train, d_val,
            weights_per_star,
            idx_train=np.where(idx_final_train)[0],
            optimize_stellar_model=True,
            optimize_stellar_params=True,
            lr_model_init=1e-7,
            lr_stars_init=1e-5,
            batch_size=batch_size,
            n_epochs=n_epochs,
            var_update = ['atm','E','plx','xi'],
            model_update = ['stellar_model', 'ext_curve_w', 'ext_curve_b'],
        )
        
        print('Plotting stellar model ...')
        for i,track in enumerate(atm_tracks):
            fig,ax = model.plot_stellar_model(stellar_model, track)
            fig.savefig(full_fn(f'plots/stellar_model_step3_track{i}'))
            plt.close(fig)
        
        fig,ax = plot_extinction_curve(flux_model, show_variation=True)
        fig.savefig(full_fn('plots/extinction_curve_step3'))
        plt.close(fig)
        
        stellar_model.save(full_fn('models/flux/xp_spectrum_model_final_Rv'))
        save_as_h5(ret, full_fn('hist_loss/final_Rv.h5'))

        ret = train_stellar_model(
            stellar_model,
            d_train, d_val,
            weights_per_star,
            #idx_train=np.where(idx_final_train)[0],
            optimize_stellar_model=False,
            optimize_stellar_params=True,
            batch_size=batch_size,
            n_epochs=n_epochs,
            var_update = ['atm','E','plx','xi'],
        )         
        
        for key in ('stellar_type', 'xi', 'stellar_ext', 'plx'):
            d_val[f'{key}_est'] = d_val[key].copy()
        
        ret = train_stellar_model(
            stellar_model,
            d_val, d_val,
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
    args = parser.parse_args()

    train(args.input, args.output_dir, stage=args.stage, thin=args.thin)

    return 0


if __name__=='__main__':
    main()
