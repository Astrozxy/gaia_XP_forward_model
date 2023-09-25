#!/usr/bin/env python

from __future__ import print_function, division

import numpy as np

import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, Normalize
import matplotlib.ticker as ticker

import h5py

import os
from glob import glob
from tqdm import tqdm

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.regularizers import L2
from tensorflow.keras.layers.experimental import preprocessing
import tensorflow_addons as tfa

import astropy.units as units
import astropy.constants as const
from astropy.table import Table

from plot_utils import plot_corr, choose_lim, hist2d_reduce
from xp_neural_network import FluxModel, GaussianMixtureModel
from xp_utils import calc_invs_eigen


feature_keys = [
    'ln_rchi2_opt',
    'ln_dplx2',
    'ln_prior',
    'teff_est',
    'logg_est',
    'feh_est',
    'teff_est_err',
    'logg_est_err',
    'feh_est_err',
    'asinh_plx_snr',
    'asinh_g_snr',
    'asinh_bp_snr',
    'asinh_rp_snr',
    'ln_phot_bp_rp_excess_factor',
    'phot_g_mean_mag',
    'ln_ruwe',
    'fidelity_v2',
    'norm_dg',
    'ln_bp_chi_squared',
    'ln_rp_chi_squared',
]
feature_keys += [f'asinh_flux_chi_{i:02d}' for i in range(66)]

# Wavelengths at which the spectrum is sampled
xp_wl = np.arange(392., 993., 10.)
tmass_wl = np.array([1.235e3, 1.662e3, 2.159e3])
unwise_wl = np.array([3.3526e3, 4.6028e3])
sample_wl = np.hstack([xp_wl, tmass_wl, unwise_wl])

print('Loading trained flux model ...')
stellar_model = FluxModel.load('models/flux/xp_spectrum_model_final-1')

print('Loading stellar type prior ...')
stellar_type_prior = GaussianMixtureModel.load('models/prior/gmm_prior-1')


def permutation_feature_importance(model, features, labels, batch_size=4096):
    loss = []

    # Evaluate the baseline model performance
    l = model.evaluate(features, labels, batch_size=batch_size)
    loss.append(l)

    rng = np.random.default_rng()
    n_dim = features.shape[1]

    for d in range(n_dim):
        # Permute feature d
        features_scrambled = features.copy()
        rng.shuffle(features_scrambled[:,d])
        # Evaluate the model performance, with feature d scrambled
        l = model.evaluate(features_scrambled, labels, batch_size=batch_size)
        loss.append(l)

    loss = np.array(loss)
    print(loss)

    metric_names = model.metrics_names

    header = f'{"feature": >45s} :'
    for mn in metric_names:
        header += f'{"d"+mn: >10s}'
    print(header)

    delta = loss[1:] - loss[0]
    sort_idx = np.argsort(delta[:,0])[::-1]

    #for i,fn in enumerate(feature_keys):
    for i in sort_idx:
        fn = feature_keys[i]
        txt = f'{fn: >45s} :'
        for k,mn in enumerate(metric_names):
            delta = loss[i+1,k] - loss[0,k]
            txt += f'   {delta: >+7.3f}'
        print(txt)

    return np.array(loss)


def load_model(fname):
    model = tf.keras.models.load_model(fname)
    #model.compile(
    #    loss='binary_crossentropy',
    #    metrics=['accuracy']
    #)
    return model


def train_model(features, labels,
                n_hidden_layers=0,
                hidden_size=32,
                dropout=0.1,
                batch_size=1024*4,
                n_epochs=4096,
                validation_frac=0.2,
                learn_rate_init=1e-3
               ):
    n_features = features.shape[1]

    model = keras.models.Sequential(name='flag_model')

    model.add(
        layers.InputLayer(input_shape=(n_features,), name='features')
    )
    normalizer = preprocessing.Normalization(name='feature_normalizer')
    normalizer.adapt(features[:2**14]) # Hangs when feeding in more data
    model.add(normalizer)
    for i in range(n_hidden_layers):
        model.add(layers.Dropout(dropout))
        model.add(layers.Dense(
            hidden_size,
            activation='relu',
            kernel_regularizer=L2(l2=1e-3),
            name=f'hidden_{i}'
        ))
    model.add(
        layers.Dense(1, activation='sigmoid', name='prediction')
    )

    #opt = keras.optimizers.SGD(learning_rate=1e-2, momentum=0.5)
    opt = keras.optimizers.Adam(learning_rate=1e-3)
    #opt = tfa.optimizers.AdamW(learning_rate=1e-3, weight_decay=1e-4)

    model.compile(
        loss='binary_crossentropy',
        optimizer=opt,
        metrics=['accuracy']
    )

    model.summary()

    callbacks = [
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.1,
            patience=256,
            min_delta=0.0001,
            verbose=1,
        ),
        keras.callbacks.EarlyStopping(
            monitor='val_loss',
            min_delta=0.0001,
            patience=1024
        )
    ]
    #lr_schedule = np.full(n_epochs, learn_rate_init)
    #lr_schedule[n_epochs//3:] *= 0.1 # Reduce learning rate 1/3 through
    #lr_schedule[2*n_epochs//3:] *= 0.1 # Reduce learning rate 2/3 through
    #lr_fn = lambda k: lr_schedule[k]
    #lr_callback = tf.keras.callbacks.LearningRateScheduler(lr_fn)

    fit_history = model.fit(
        features, labels,
        batch_size=batch_size,
        epochs=n_epochs,
        validation_split=validation_frac,
        callbacks=callbacks
    )

    if n_hidden_layers == 0:
        w_flat = model.layers[1].weights[0].numpy().flat
        idx = np.argsort(np.abs(w_flat))[::-1]
        for i,w in zip(idx,w_flat[idx]):
            fn = feature_keys[i]
            print(f'{fn: >45s} : {w: >+7.3f}')

    return model, fit_history


def extract_features(d):
    n = len(d['gdr3_source_id'])

    features = np.empty((n,len(feature_keys)), dtype='f4')
    for i,k in enumerate(feature_keys):
        features[:,i] = d[k]
        idx_nonfinite = ~np.isfinite(features[:,i])
        if np.any(idx_nonfinite):
            print(f'key = {k}:')
            print(d[k][idx_nonfinite])

    return features


def extract_training_validation(d, training=True):
    n_stars = len(d['gdr3_source_id'])

    n_val = int(0.2 * n_stars)

    if training:
        d_sel = {k:d[k][:n_val] for k in d}
    else:
        d_sel = {k:d[k][-n_val:] for k in d}

    return d_sel


def prepare_training_data(d, param='feh', source='lamost'):
    #d = extract_training_validation(d, training=True)

    n_stars = len(d['gdr3_source_id'])

    features = extract_features(d)

    param_ref = d[f'{param}_{source}']
    dparam = d[f'{param}_est'] - param_ref
    sigma = np.sqrt(d[f'{param}_est_err']**2 + d[f'{param}_{source}_err']**2)
    chi = dparam / sigma
    chi_floored = dparam / np.sqrt(sigma**2 + 0.1**2)

    #idx_bad = (
    #    (np.abs(dfeh) > 0.5)
    #)
    #idx_good = (
    #    (np.abs(dfeh) < 0.3)
    #)

    if param == 'teff':
        param_ref = np.log10(param_ref)

    bins = np.linspace(
        np.nanmin(param_ref),
        np.nanmax(param_ref),
        31
    )

    param_hist,bin_edges = np.histogram(param_ref, bins=bins)
    param_hist = param_hist / np.percentile(param_hist, 70.)
    print(f'{param} histogram:')
    for i,n in enumerate(param_hist):
        print(f' ({bins[i]:.2f},{bins[i+1]:.2f}) : {n:.5f}')

    bin_idx = np.searchsorted(bin_edges, param_ref) - 1
    bin_idx[bin_idx<0] = 0
    bin_idx[bin_idx==len(bin_edges)-1] = -1
    p_select = 1 / param_hist[bin_idx]
    for v,p in zip(param_ref[:10], p_select[:10]):
        print(f'  {v:+6.2f}  {p:6.4f}')

    rng = np.random.default_rng(1)

    idx_bad = (
        (np.abs(chi_floored) > 3.)
      & (rng.uniform(size=p_select.size) < p_select)
    )
    idx_good = (
        (np.abs(chi_floored) < 2.)
      & (rng.uniform(size=p_select.size) < p_select)
    )

    n_good = np.count_nonzero(idx_good)
    n_bad = np.count_nonzero(idx_bad)
    
    idx_good &= (
      (rng.uniform(size=idx_good.size) < 2*n_bad/n_good)
    )

    print(f'{np.count_nonzero(idx_bad)} bad sources.')
    print(f'{np.count_nonzero(idx_good)} good sources.')

    labels = np.full(n_stars, -1, dtype='f4')
    labels[idx_bad] = 0
    labels[idx_good] = 1

    idx_sel = idx_bad | idx_good
    features = features[idx_sel]
    labels = labels[idx_sel]

    return features, labels


def plot_data(d, idx_good, idx_bad):
    teff_lim = (2., 10.)
    logg_lim = (-1., 5.5)
    bins = (40, 30)

    params = d['stellar_params_est']
    teff = params[:,0]
    logg = params[:,2]

    idx_finite = np.isfinite(teff) & np.isfinite(logg)

    idx_good = idx_good & idx_finite
    idx_bad = idx_bad & idx_finite

    img_good,_,_ = np.histogram2d(
        teff[idx_good], logg[idx_good],
        range=[teff_lim,logg_lim],
        bins=bins
    )
    img_bad,_,_ = np.histogram2d(
        teff[idx_bad], logg[idx_bad],
        range=[teff_lim,logg_lim],
        bins=bins
    )
    img_tot = img_good + img_bad
    img_badfrac = img_bad / img_tot

    fig,ax_arr = plt.subplots(
        2,2,
        figsize=(6,4),
        gridspec_kw=dict(height_ratios=[1,0.1])
    )
    ((ax_tot,ax_ratio),(cax_tot,cax_ratio)) = ax_arr

    im = ax_tot.imshow(
        img_tot.T,
        extent=teff_lim+logg_lim,
        origin='lower',
        aspect='auto',
        interpolation='nearest',
        norm=LogNorm()
    )
    cb = fig.colorbar(
        im,
        cax=cax_tot,
        orientation='horizontal',
        label=r'$N$'
    )
    im = ax_ratio.imshow(
        img_badfrac.T,
        extent=teff_lim+logg_lim,
        origin='lower',
        aspect='auto',
        interpolation='nearest',
        cmap='coolwarm',
        vmin=0., vmax=1.
        #norm=LogNorm(vmin=1e-3,vmax=1)
    )
    cb = fig.colorbar(
        im,
        cax=cax_ratio,
        orientation='horizontal',
        label=r'$\mathrm{bad\ fraction}$'
    )

    for ax in (ax_tot,ax_ratio):
        ax.invert_xaxis()
        ax.invert_yaxis()
        ax.set_xlabel(r'$T_{\mathrm{eff}}$')
        ax.xaxis.set_minor_locator(ticker.AutoMinorLocator())
        ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())
        ax.grid(True, which='major', alpha=0.2, c='gray')
        ax.grid(True, which='minor', alpha=0.05, c='gray')

    ax_ratio.set_yticklabels([])
    ax_tot.set_ylabel(r'$\log g$')

    fig.subplots_adjust(
        bottom=0.12,
        top=0.96,
        left=0.10,
        right=0.97,
        hspace=0.35,
        wspace=0.1
    )

    fig.savefig(f'plots/kiel_goodbad')
    plt.close(fig)


def plot_training_data(features, labels,
                       name='training', param='feh', source='lamost'):
    plot_dir = f'plots/outliers_val/{param}'

    idx_good = (labels == 1)

    for i,key in enumerate(feature_keys):
        print(f'Plotting {key} ...')
        fig,ax = plt.subplots(1,1, figsize=(6,4))

        x = features[:,i]

        xlim = np.nanpercentile(x, [0.1, 99.9])
        w = xlim[1] - xlim[0]
        xlim = (xlim[0]-0.1*w, xlim[1]+0.1*w)

        ax.hist(
            x[idx_good],
            range=xlim,
            bins=50,
            density=True,
            label='good',
            histtype='step'
        )
        ax.hist(
            x[~idx_good],
            range=xlim,
            bins=50,
            density=True,
            label='bad',
            histtype='step'
        )

        ax.legend()
        key_clean = key.replace('_',r'\_')
        ax.set_xlabel(f'$\mathtt{{ {key_clean} }}$')

        fig.savefig(os.path.join(
            plot_dir, fr'xp_vs_{source}_{name}_feature_{key}'
        ))
        plt.close(fig)


def plot_results(d, model, param='feh'):
    plot_dir = f'plots/outliers_val/{param}'

    if 'teff_lamost' in d:
        source = 'lamost'
    elif 'teff_apogee' in d:
        source = 'apogee'
    else:
        raise KeyError('Could not find teff_lamost or teff_apogee!')

    dvar = {}
    sigma2_var = {}
    chi_var = {}
    for i,t in enumerate(['teff', 'feh', 'logg']):
        dvar[t] = d[f'{t}_est'] - d[f'{t}_{source}']
        sigma2_var[t] = d[f'{t}_est_err']**2 + d[f'{t}_{source}_err']**2
        chi_var[t] = dvar[t] / np.sqrt(sigma2_var[t])

    features = extract_features(d)
    pred = model.predict(features, batch_size=4096)[:,0]
    pred_good = (pred > 0.5)

    #plot_training_data(features, pred_good.astype('i1'), name='pred')

    print(rf'{np.mean(pred_good)*100:.2f}% of sources labeled "good"')

    #xlim = (-3., 1.)
    #ylim = (-3., 3.)
    ylim = (-10., 10.)

    var_label = {
        'teff': r'T_{\mathrm{eff}}',
        'feh': r'\left[\mathrm{Fe/H}\right]',
        'logg': r'\log g',
    }

    def finalize_fig(fig, ax, title, var='feh', chi=True):
        ax.set_xlabel(
            fr'${var_label[var]}\ \left(\mathrm{{{source.upper()}}}\right)$'
        )
        ylabel = (
            fr'$\Delta {var_label[var]} '
          + (r'/ \sigma \ ' if chi else r'\ ')
          + fr'(\mathrm{{XP}}-\mathrm{{ {source.upper()} }})$'
        )
        print(ylabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        fig.subplots_adjust(
            left=0.12,
            right=0.96,
            bottom=0.14,
            top=0.90
        )
        ax.grid(True, which='major', alpha=0.1, c='k')
        ax.xaxis.set_minor_locator(ticker.AutoMinorLocator())
        ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())
        title_sub = title.replace(" ","")
        prefix = 'chi' if chi else 'diff'
        fig.savefig(os.path.join(
            plot_dir, f'xp_vs_{source}_{prefix}_{var}_{title_sub}'
        ))
        plt.close(fig)

    for var in ('teff', 'feh', 'logg'):
        x = d[f'{var}_{source}']
        y = chi_var[var]
        xlim = choose_lim(x, pct=[0.1,99.9])

        x1 = x + dvar[var]

        for n,idx in [('all',slice(None)),('good',pred_good),('bad',~pred_good)]:
            fig,ax = plt.subplots(1,1, figsize=(6,4))
            plot_corr(
                ax, x[idx], y[idx],
                diff=False,
                x_lim=xlim,
                d_max=ylim[1]
            )
            finalize_fig(fig, ax, n, var=var, chi=True)

            fig,ax = plt.subplots(1,1, figsize=(6,4))
            plot_corr(
                ax, x[idx], dvar[var][idx],
                diff=False,
                x_lim=xlim,
                d_max=3.
            )
            finalize_fig(fig, ax, n, var=var, chi=False)

            fig,ax = plt.subplots(1,1, figsize=(6,6))
            ax.hist2d(
                x[idx], x1[idx],
                range=[xlim,xlim],
                bins=50,
                density=True,
                norm=LogNorm()
            )
            ax.set_xlabel(
                rf'${var_label[var]}\ \left(\mathrm{{{source.upper()}}}\right)$'
            )
            ax.set_ylabel(rf'${var_label[var]}\ \left(\mathrm{{XP}}\right)$')
            fig.savefig(os.path.join(
                plot_dir, f'xp_vs_{source}_hist2d_{var}_{n}'
            ))
            plt.close(fig)

        fig,ax = plt.subplots(1,1, figsize=(6,4))
        plot_corr(
            ax, x[pred_good], y[pred_good],
            diff=False,
            x_lim=xlim,
            d_max=4.#0.8
        )
        finalize_fig(fig, ax, 'good (zoomed)', var=var)

        for sel in (None,'bp_snr','rp_snr','plx_snr','fit','hq'):
            idx = np.isfinite(x) & np.isfinite(y)
            if sel == 'bp_snr':
                suffix = ' (high BP SNR)'
                idx &= (d['asinh_bp_snr'] > 7.)
            elif sel == 'rp_snr':
                suffix = ' (high RP SNR)'
                idx &= (d['asinh_rp_snr'] > 7.)
            elif sel == 'plx_snr':
                suffix = ' (high plx SNR)'
                idx &= (d['asinh_plx_snr'] > 3.)
            elif sel == 'fit':
                suffix = ' (low rchi2)'
                idx &= (d['ln_rchi2_opt'] < np.log(1.5))
            elif sel == 'hq':
                suffix = ' (HQ subset)'
                idx &= (
                    (d['asinh_bp_snr'] > 7.)
                  & (d['asinh_rp_snr'] > 7.)
                  & (d['asinh_plx_snr'] > 3.)
                  & (d['ln_rchi2_opt'] < np.log(1.5))
                )
            else:
                suffix = ''

            pct = 100 * np.mean(idx)
            txt = fr'$\mathrm{{subset}} = {pct:.1f}\%$'

            # Reduce c values in each bin
            from scipy.stats import binned_statistic_2d
            img,_,_,_ = binned_statistic_2d(
                x[idx], y[idx], pred[idx],
                range=[xlim,ylim],
                bins=(50,31)
            )
            fig,ax = plt.subplots(1,1, figsize=(6,4))
            im = ax.imshow(
                img.T,
                extent=xlim+ylim,
                origin='lower',
                aspect='auto',
                interpolation='nearest',
                cmap='coolwarm_r',
                vmin=0., vmax=1.
            )
            cax = fig.add_axes([0.90,0.14,0.03,0.86-0.14])
            cb = fig.colorbar(im, cax=cax)
            ax.text(0.97, 0.95, txt, ha='right', va='top', transform=ax.transAxes)
            finalize_fig(fig, ax, 'good fraction'+suffix, var=var)

            fig,ax = plt.subplots(1,1, figsize=(6,4))
            im = ax.imshow(
                1-img.T,
                extent=xlim+ylim,
                origin='lower',
                aspect='auto',
                interpolation='nearest',
                norm=LogNorm(vmin=1e-2, vmax=1)
            )
            cax = fig.add_axes([0.90,0.14,0.03,0.86-0.14])
            cb = fig.colorbar(im, cax=cax)
            ax.text(0.97, 0.95, txt, ha='right', va='top', transform=ax.transAxes)
            finalize_fig(fig, ax, 'bad fraction'+suffix, var=var)


def paper_figures(d_lamost, d_apogee, model, param):
    d_lamost = extract_training_validation(d_lamost, training=False)

    var_label = {
        'teff': r'T_{\mathrm{eff}} / 10^3 \, \mathrm{K}',
        'feh': r'\left[\mathrm{Fe/H}\right]',
        'logg': r'\log g',
    }[param]

    chi_label = {
        'teff': r'T_{\mathrm{eff}}',
        'feh': r'\left[\mathrm{Fe/H}\right]',
        'logg': r'\log g',
    }[param]

    xlim = {
        'teff': [3.7, 13.],
        'feh': [-2.5, 1.0],
        'logg': [0., 5.]
    }[param]

    # LAMOST good vs. bad comparison
    fig,(ax_g,ax_b) = plt.subplots(1,2, figsize=(3,1.8))

    lamost = d_lamost[f'{param}_lamost']
    delta_lamost = d_lamost[f'{param}_est'] - d_lamost[f'{param}_lamost']
    features = extract_features(d_lamost)
    pred = model.predict(features, batch_size=4096)[:,0]
    pred_good = (pred > 0.5)

    #pred_good &= (d_lamost['reliability_flag'] == 0)
    #reliable = (d_lamost['reliability_flag'] == 0)

    if param == 'teff':
        lamost = np.log10(lamost)
        xlim = np.log10(xlim)

    kw = dict(
        diff=False,
        x_lim=xlim,
        d_max=1.2,
        bins=(20,21),
        cmap='Purples',
        envelope_kw=dict(c='gold')
    )
    plot_corr(
        ax_g,
        lamost[pred_good],
        delta_lamost[pred_good],
        **kw
    )
    plot_corr(
        ax_b,
        lamost[~pred_good],
        delta_lamost[~pred_good],
        **kw
    )

    kw = dict(fontsize=7, labelpad=2)
    xlabel = fr'${var_label}\ \left(\mathrm{{LAMOST}}\right)$'
    ylabel = (
        fr'$\Delta {var_label}\ \left(\mathrm{{XP}}-\mathrm{{LAMOST}}\right)$'
    )
    ax_g.set_xlabel(xlabel, **kw)
    ax_b.set_xlabel(xlabel, **kw)
    ax_g.set_ylabel(ylabel, **kw)
    ax_b.set_yticklabels([])

    ax_g.set_title(r'$\mathrm{Flagged\ as}\ \mathtt{good}$', fontsize=9)
    ax_b.set_title(r'$\mathrm{Flagged\ as}\ \mathtt{bad}$', fontsize=9)

    for ax in (ax_g,ax_b):
        if param == 'teff':
            x_major = [5., 10.]
            x_minor = np.arange(
                np.ceil(10**xlim[0]),
                np.floor(10**xlim[1])+0.001,
                1.
            )
            ax.set_xticks(np.log10(x_major))
            ax.set_xticklabels([f'${xt:.0f}$' for xt in x_major])
            ax.xaxis.set_minor_locator(ticker.FixedLocator(np.log10(x_minor)))
        else:
            ax.xaxis.set_minor_locator(ticker.AutoMinorLocator())

        ax.grid(True, which='major', alpha=0.1, c='k')
        ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())
        ax.tick_params(axis='both', which='major', labelsize=7)
    
    #fig.subplots_adjust(
    #    left=0.10,
    #    right=0.97,
    #    bottom=0.20,
    #    top=0.87,
    #    wspace=0.04
    #)
    fig.subplots_adjust(
        left=0.17,
        right=0.97,
        bottom=0.22,
        top=0.85,
        wspace=0.06
    )

    fig.savefig(f'plots/validation_flag_lamost_{param}.pdf')
    plt.close(fig)

    # LAMOST uncertainty-normalized comparison (only good)
    fig,ax = plt.subplots(1,1, figsize=(3,1.8))

    sigma = np.sqrt(
        d_lamost[f'{param}_est_err']**2
      + d_lamost[f'{param}_lamost_err']**2
    )
    chi_lamost = delta_lamost / sigma

    plot_corr(
        ax,
        lamost[pred_good],
        chi_lamost[pred_good],
        diff=False,
        x_lim=xlim,
        d_max=6.,
        bins=(40,31),
        cmap='Purples',
        envelope_kw=dict(c='gold')
    )

    kw = dict(fontsize=7, labelpad=2)
    xlabel = fr'${var_label}\ \left(\mathrm{{LAMOST}}\right)$'
    ylabel = (
        fr'$\Delta {chi_label} / \sigma \ '
        r'\left(\mathrm{XP} \! - \! \mathrm{LAMOST}\right)$'
    )
    ax.set_xlabel(xlabel, **kw)
    ax.set_ylabel(ylabel, **kw)

    if param == 'teff':
        x_major = [5., 10.]
        x_minor = np.arange(
            np.ceil(10**xlim[0]),
            np.floor(10**xlim[1])+0.001,
            1.
        )
        ax.set_xticks(np.log10(x_major))
        ax.set_xticklabels([f'${xt:.0f}$' for xt in x_major])
        ax.xaxis.set_minor_locator(ticker.FixedLocator(np.log10(x_minor)))
    else:
        ax.xaxis.set_minor_locator(ticker.AutoMinorLocator())

    ax.grid(True, which='major', alpha=0.1, c='k')
    ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())
    ax.tick_params(axis='both', which='major', labelsize=7)

    fig.subplots_adjust(
        left=0.18,
        right=0.96,
        bottom=0.22,
        top=0.94
    )

    fig.savefig(f'plots/validation_lamost_{param}_chi.pdf')
    plt.close(fig)

    # APOGEE comparison (only good)
    fig,ax = plt.subplots(1,1, figsize=(3,1.8))

    apogee = d_apogee[f'{param}_apogee']
    delta_apogee = d_apogee[f'{param}_est'] - d_apogee[f'{param}_apogee']
    features = extract_features(d_apogee)
    pred = model.predict(features, batch_size=4096)[:,0]
    pred_good = (pred > 0.5)

    #pred_good &= (d_apogee['reliability_flag'] == 0)
    #reliable = (d_apogee['reliability_flag'] == 0)

    xlim = {
        'teff': [3.7, 12.],
        'feh': [-2.5, 0.6],
        'logg': [0., 5.]
    }[param]

    d_max = {
        'teff': 1.4,
        'feh': 0.5,
        'logg': 1.0
    }[param]

    if param == 'teff':
        apogee = np.log10(apogee)
        xlim = np.log10(xlim)

    plot_corr(
        ax,
        apogee[pred_good],
        delta_apogee[pred_good],
        diff=False,
        x_lim=xlim,
        d_max=d_max,
        bins=(40,31),
        cmap='Purples',
        envelope_kw=dict(c='gold')
    )

    kw = dict(fontsize=7, labelpad=2)
    xlabel = fr'${var_label}\ \left(\mathrm{{APOGEE}}\right)$'
    ylabel = (
        fr'$\Delta {var_label}\ '
        r'\left(\mathrm{XP} \! - \! \mathrm{APOGEE}\right)$'
    )
    ax.set_xlabel(xlabel, **kw)
    ax.set_ylabel(ylabel, **kw)

    if param == 'teff':
        ax.set_xticks(np.log10(x_major))
        ax.set_xticklabels([f'${xt:.0f}$' for xt in x_major])
        ax.xaxis.set_minor_locator(ticker.FixedLocator(np.log10(x_minor)))
    else:
        ax.xaxis.set_minor_locator(ticker.AutoMinorLocator())

    ax.grid(True, which='major', alpha=0.1, c='k')
    ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())
    ax.tick_params(axis='both', which='major', labelsize=7)

    fig.subplots_adjust(
        left=0.18,
        right=0.96,
        bottom=0.22,
        top=0.94
    )

    fig.savefig(f'plots/validation_apogee_{param}.pdf')
    plt.close(fig)

    # APOGEE uncertainty-normalized comparison (only good)
    chi_label = {
        'teff': r'T_{\mathrm{eff}}',
        'feh': r'\left[\mathrm{Fe/H}\right]',
        'logg': r'\log g',
    }[param]

    fig,ax = plt.subplots(1,1, figsize=(3,1.8))

    sigma = np.sqrt(
        d_apogee[f'{param}_est_err']**2
      + d_apogee[f'{param}_apogee_err']**2
    )
    chi_apogee = delta_apogee / sigma

    plot_corr(
        ax,
        apogee[pred_good],
        chi_apogee[pred_good],
        diff=False,
        x_lim=xlim,
        d_max=6.,
        bins=(40,31),
        cmap='Purples',
        envelope_kw=dict(c='gold')
    )

    kw = dict(fontsize=7, labelpad=2)
    xlabel = fr'${var_label}\ \left(\mathrm{{APOGEE}}\right)$'
    ylabel = (
        fr'$\Delta {chi_label} / \sigma \ '
        r'\left(\mathrm{XP} \! - \! \mathrm{APOGEE}\right)$'
    )
    ax.set_xlabel(xlabel, **kw)
    ax.set_ylabel(ylabel, **kw)

    if param == 'teff':
        ax.set_xticks(np.log10(x_major))
        ax.set_xticklabels([f'${xt:.0f}$' for xt in x_major])
        ax.xaxis.set_minor_locator(ticker.FixedLocator(np.log10(x_minor)))
    else:
        ax.xaxis.set_minor_locator(ticker.AutoMinorLocator())

    ax.grid(True, which='major', alpha=0.1, c='k')
    ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())
    ax.tick_params(axis='both', which='major', labelsize=7)

    fig.subplots_adjust(
        left=0.18,
        right=0.96,
        bottom=0.22,
        top=0.94
    )

    fig.savefig(f'plots/validation_apogee_{param}_chi.pdf')
    plt.close(fig)

    # Kiel diagram (APOGEE)
    fig,ax = plt.subplots(1,1, figsize=(3,3.5), layout='constrained')

    teff_lim = (3., 14.)
    logg_lim = (0., 5.5)

    def f_reduce(x):
        if len(x) < 4:
            return np.nan
        return np.mean(x)

    im = hist2d_reduce(
        np.log10(d_apogee['teff_apogee']),
        d_apogee['logg_apogee'],
        c=pred,
        xlim=np.log10(teff_lim),
        ylim=logg_lim,
        ax=ax,
        hist_kw=dict(bins=40, statistic=f_reduce),
        imshow_kw=dict(vmin=0., vmax=1.)
    )

    ax.invert_xaxis()
    ax.invert_yaxis()

    x_major = [5., 10.]
    x_minor = np.arange(
        np.ceil(teff_lim[0]),
        np.floor(teff_lim[1])+0.001,
        1.
    )
    ax.set_xticks(np.log10(x_major))
    ax.set_xticklabels([f'${xt:.0f}$' for xt in x_major])
    ax.xaxis.set_minor_locator(ticker.FixedLocator(np.log10(x_minor)))

    fig.colorbar(
        im, ax=ax,
        location='bottom',
        label=r'$\mathrm{predicted\ good\ fraction}$'
    )

    ax.set_xlabel(
        r'$T_{\mathrm{eff}}/10^3 \, \mathrm{K} \left(\mathrm{APOGEE}\right)$'
    )
    ax.set_ylabel(r'$\log g \ \left(\mathrm{APOGEE}\right)$')

    fig.savefig(f'plots/validation_apogee_{param}_kiel.pdf')
    plt.close(fig)


def load_apogee_data():
    d = {}

    # Load raw data
    print('Loading APOGEE raw data ...')
    fname = 'data/apogee_compare_updated.h5'
    with h5py.File(fname, 'r') as f:
        for key in f['gaia'].keys():
            d[key] = f['gaia'][key][:]

        d['feh_apogee'] = f['apogee']['FE_H'][:]
        d['feh_apogee_err'] = f['apogee']['FE_H_ERR'][:]
        d['aspcap_chi2'] = f['apogee']['ASPCAP_CHI2'][:]
        d['aspcap_flag'] = f['apogee']['ASPCAPFLAG'][:]

        d['teff_apogee'] = 1e-3 * f['apogee']['TEFF'][:]
        d['teff_apogee_err'] = 1e-3 * f['apogee']['TEFF_ERR'][:]
        d['logg_apogee'] = f['apogee']['LOGG'][:]
        d['logg_apogee_err'] = f['apogee']['LOGG_ERR'][:]

    # Rename some fields
    d['flux'] = d.pop('spectra_obs')
    d['flux_err'] = d.pop('spectra_obs_err')

    # Remove bad APOGEE measurements, defined as any of the following:
    #   19: METALS_BAD
    #   24: SNR_BAD
    #   27: CHI2_BAD
    idx_apogee_good = ~((
        (2**19 | 2**24 | 2**27) & d['aspcap_flag']
    ).astype('bool'))
    frac_bad = 1 - np.mean(idx_apogee_good)
    print(f'{100*frac_bad:.2f}% of APOGEE spectra fail cut.')
    for key in d:
        d[key] = d[key][idx_apogee_good]

    compute_additional_features(d)

    return d


def load_lamost_data():
    d = {}

    # Load raw data
    print('Loading LAMOST raw data ...')
    fname = 'data/xp_results_lamost_match.h5'
    with h5py.File(fname, 'r') as f:
        for k in f.keys():
            d[k] = f[k][:]

    # Filter out bad data
    idx_lamost_good = np.all(
        (d['lamost_stellar_type_err'] < np.array([0.5, 0.5, 0.5])[None,:])
      & (d['lamost_stellar_type_err'] > 0.)
      & (d['lamost_stellar_type'] > np.array([2.5, -3.5, -2.0])[None,:])
      & (d['lamost_stellar_type'] < np.array([30., 1.5, 6.5])[None,:]),
        axis=1
    )
    for b in 'gri':
        idx_lamost_good &= (d[f'lamost_snr_{b}'] > 10.)

    n_all = idx_lamost_good.size
    n_stars = np.count_nonzero(idx_lamost_good)
    pct_good = 100*n_stars/n_all
    print(f'{n_stars} of {n_all} ({pct_good:.2f}%) LAMOST sources pass cuts.')

    for k in d:
        d[k] = d[k][idx_lamost_good]

    for i,t in enumerate(['teff', 'feh', 'logg']):
        d[f'{t}_lamost'] = d['lamost_stellar_type'][:,i]
        d[f'{t}_lamost_err'] = d['lamost_stellar_type_err'][:,i]

    compute_additional_features(d)

    return d


def load_xp_data(fid):
    meta_fn = f'data/xp_continuous_metadata/xp_continuous_metadata_{fid}.fits.gz'
    out_fn = f'data/xp_opt_output_files/xp_opt_output_{fid}.h5'
    in_fn = f'data/xp_opt_input_files/xp_opt_input_{fid}.h5'
    summary_fn = f'data/xp_summary_h5/XpSummary_{fid}.hdf5'
    #print(meta_fn)
    #print(out_fn)
    #print(in_fn)
    #print(summary_fn)

    d = {}
    
    t_meta = Table.read(meta_fn, format='fits')
    for k in t_meta.colnames:
        d[k] = t_meta[k].data
    
    with h5py.File(out_fn, 'r') as f:
        for k in f.keys():
            s = f[k].shape
            if len(s) > 2:
                continue
            d[k] = f[k][:]

    with h5py.File(in_fn, 'r') as f:
        for k in f.keys():
            s = f[k].shape
            if len(s) > 2:
                continue
            d[k] = f[k][:]

    with h5py.File(summary_fn, 'r') as f:
        for k in ('bp_chi_squared', 'rp_chi_squared'):
            d[k] = f[k][:]
    
    compute_additional_features(d, shuffle=False, filter_unreliable_fits=False)

    return d




def compute_additional_features(d, shuffle=True, filter_unreliable_fits=True):
    n_stars = len(d['gdr3_source_id'])

    # Shuffle data
    if shuffle:
        print('Shuffling data ...')
        rng = np.random.default_rng(7)
        shuffle_idx = np.arange(n_stars)
        rng.shuffle(shuffle_idx)
        for key in d:
            d[key] = d[key][shuffle_idx]

    d['flux_pred'] = stellar_model.predict_obs_flux(
        d['stellar_params_est'][:,:3],
        d['stellar_params_est'][:,3],
        d['stellar_params_est'][:,4]
    ).numpy()

    d['ln_prior'] = stellar_type_prior.ln_prob(
        d['stellar_params_est'][:,:3]
    ).numpy()

    # Computed features
    print('Computing additional features ...')
    idx = ~np.isfinite(d['norm_dg'])
    d['norm_dg'][idx] = -45.

    idx = ~np.isfinite(d['fidelity_v2'])
    d['fidelity_v2'][idx] = 0.5

    idx = ~np.isfinite(d['parallax'])
    d['parallax'][idx] = 0.
    d['parallax_error'][idx] = 9999.

    ln_clipped = lambda x: np.log(np.clip(x, 1e-7, np.inf))

    delta_plx = d['stellar_params_est'][:,4] - d['parallax']
    d['dplx'] = delta_plx / d['parallax_error']
    d['ln_dplx2'] = ln_clipped(d['dplx']**2)

    d['ln_rchi2_opt'] = ln_clipped(d['rchi2_opt'])
    d['ln_phot_bp_rp_excess_factor'] = ln_clipped(d['phot_bp_rp_excess_factor'])
    d['ln_ruwe'] = ln_clipped(d['ruwe'])

    for b in ('bp','rp'):
        key = f'{b}_chi_squared'
        d[f'ln_{key}'] = ln_clipped(d[key])
        idx = ~np.isfinite(d[f'ln_{key}'])
        d[f'ln_{key}'] = np.nanpercentile(d[f'ln_{key}'], 99.9)

    d['reliable_fit'] = (
        (d['rchi2_opt'] < 2.)
      & (d['ln_prior'] > -7.43)
      & (np.abs(d['dplx']) < 10.)
    )

    for b in ('g', 'bp', 'rp'):
        key = f'asinh_{b}_snr'
        d[key] = np.arcsinh(
            d[f'phot_{b}_mean_flux']/d[f'phot_{b}_mean_flux_error']
        )
        idx = ~np.isfinite(d[key])
        d[key][idx] = 0.

    idx = ~np.isfinite(d['phot_g_mean_mag'])
    d['phot_g_mean_mag'][idx] = 20.7

    #for i,b in enumerate(['J', 'H', 'Ks']):
    #    d[f'asinh_{b}_snr'] = np.arcsinh(
    #        d['tmass_flux'][:,i]/np.sqrt(d['tmass_flux_var'][:,i])
    #    )

    #for i,b in enumerate(['W1', 'W2']):
    #    d[f'asinh_{b}_snr'] = np.arcsinh(
    #        d['unwise_flux'][:,i]/np.sqrt(d['unwise_flux_var'][:,i])
    #    )

    d['asinh_plx_snr'] = np.arcsinh(d['parallax'] / d['parallax_error'])

    for i,t in enumerate(['teff', 'feh', 'logg']):
        d[f'{t}_est'] = d['stellar_params_est'][:,i]
        d[f'{t}_est_err'] = d['stellar_params_err'][:,i]

    flux_chi = (d['flux_pred'] - d['flux']) / d['flux_err']
    n_wl = flux_chi.shape[1]
    for i in range(n_wl):
        d[f'asinh_flux_chi_{i:02d}'] = np.arcsinh(flux_chi[:,i])
        d[f'ln_flux_chi2_{i:02d}'] = ln_clipped(flux_chi[:,i]**2)

    if filter_unreliable_fits:
        idx = d['reliable_fit']
        for k in d:
            d[k] = d[k][idx]

    return d


def apply_flags(batch_size=4096):
    model_fn = 'models/outlier_model_lamost_v2_{}'
    models = {k:load_model(model_fn.format(k)) for k in ('teff','feh','logg')}

    out_names = glob('data/xp_opt_output_files/xp_opt_output_*-*.h5')
    out_names.sort()
    #out_names = out_names[::100]

    kw = dict(chunks=True, compression='lzf')

    for ofn in tqdm(out_names):
        fid = ofn.split('_')[-1].split('.')[0]
        print(fid)
        fn = f'data/xp_flags_v5/reliability_{fid}.h5'
        if os.path.exists(fn):
            print(f'Skipping {fn} ...')
            continue

        print('Loading data ...')
        d = load_xp_data(fid)

        print('Extracting features ...')
        features = extract_features(d)

        print('Calculating flags ...')
        pred = {}
        for k in models:
            pred[k] = models[k].predict(features, batch_size=batch_size)[:,0]
            pct_good = np.mean(pred[k] > 0.5) * 100
            print(f'  {k: >4s} : {pct_good:.2f}% good')
            # Abort script if any non-finite predictions are encountered
            idx_nonfinite = ~np.isfinite(pred[k])
            if np.any(idx_nonfinite):
                print('features (non-finite):')
                print(features[idx_nonfinite][:3])
                print('features (finite):')
                print(features[~idx_nonfinite][:3])
                raise ValueError('Non-finite values in predictions!')

        print('Writing flags to disk ...')
        with h5py.File(fn ,'w') as f:
            f.create_dataset('gdr3_source_id', data=d['gdr3_source_id'], **kw)
            for k in pred:
                f.create_dataset(f'{k}_reliability', data=pred[k], **kw)


def main():
    #d_lamost = load_lamost_data()
    #d_apogee = load_apogee_data()

    #for param in ('teff', 'feh', 'logg'):
    #    #features,labels = prepare_training_data(d_lamost, param=param)
    #    #plot_training_data(features, labels, param=param)

    #    #model, fit_history = train_model(
    #    #    features, labels,
    #    #    n_hidden_layers=3,
    #    #    hidden_size=64,
    #    #    n_epochs=1024*32
    #    #)
    #    #model.save(f'models/outlier_model_lamost_v2_{param}')

    #    model = load_model(f'models/outlier_model_lamost_v2_{param}')
    #    #feature_imp = permutation_feature_importance(model, features, labels)

    #    #plot_results(d_lamost, model, param=param)
    #    #plot_results(d_apogee, model, param=param)

    #    paper_figures(d_lamost, d_apogee, model, param=param)

    apply_flags()

    return 0


if __name__ == '__main__':
    main()

