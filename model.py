#!/usr/bin/env python

from __future__ import print_function, division

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.ticker as ticker
from matplotlib.colors import Normalize, LogNorm
from matplotlib.cm import ScalarMappable
from matplotlib.gridspec import GridSpec

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
import tensorflow.keras as keras
import tensorflow_probability as tfp
import sonnet as snt
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
  tf.config.experimental.set_memory_growth(gpu, True)

from scipy.stats import binned_statistic_2d
from scipy.special import logsumexp
from scipy.interpolate import UnivariateSpline
import scipy.ndimage
import scipy.signal

from astropy.table import Table
import astropy.io.fits as fits
import astropy.units as units
import astropy.constants as const

import h5py
import os
import json
from glob import glob
from tqdm.auto import tqdm

from xp_utils import XPSampler, sqrt_icov_eigen
import plot_utils
#from plot_utils import plot_corr, plot_mollweide, \
#                      projection_grid, healpix_mean_map
from utils import batch_apply_tf

from astropy_healpix import HEALPix
from astropy.coordinates import SkyCoord
import astropy.units as units
import astropy.constants as const



def corr_matrix(cov):
    sigma = np.sqrt(np.diag(cov))
    corr = cov / np.outer(sigma, sigma)
    corr[np.diag_indices(corr.shape[0])] = sigma
    return corr


class FluxModel(snt.Module):
    def __init__(self, sample_wavelengths, n_input=3, 
                       n_hidden=3, hidden_size=32,
                       input_zp=None, input_scale=None,
                       l2=1., l2_ext_curve=1.):
        """
        Constructor for FluxModel.
        """
        super(FluxModel, self).__init__(name='flux_model')

        self._sample_wavelengths = tf.Variable(
            sample_wavelengths,
            name='sample_wavelengths',
            dtype=tf.float32,
            trainable=False
        )
        self._n_output = len(sample_wavelengths)
        self._n_input = n_input
        self._n_hidden = n_hidden
        self._hidden_size = hidden_size

        # L2 penalty on neural network weights
        self._l2 = tf.Variable(
            l2, name='l2',
            dtype=tf.float32,
            trainable=False
        )
        self._l2_ext_curve = tf.Variable(
            l2_ext_curve, name='l2_ext_curve',
            dtype=tf.float32,
            trainable=False
        )

        # Scaling of input
        if input_zp is None:
            zp = np.zeros((1, n_input), dtype='f4')
        else:
            zp = np.reshape(input_zp, (1, n_input)).astype('f4')

        if input_scale is None:
            iscale = np.ones((1, n_input), dtype='f4')
        else:
            iscale = 1 / np.reshape(input_scale, (1, n_input)).astype('f4')
        self._input_zp = tf.Variable(
            zp,
            name='input_zp',
            trainable=False
        )
        self._input_iscale = tf.Variable(
            iscale,
            name='input_iscale',
            trainable=False
        )
        
        # Neural network of the stellar model
        self._layers = [
            snt.Linear(hidden_size, name=f'hidden_{i}')
            for i in range(n_hidden)
        ]
        self._layers.append(snt.Linear(self._n_output, name='flux'))        
        self._activation = tf.math.tanh
        
        # Initialize extinction curve
        #self._ext_slope = tf.Variable(
        #    tf.zeros((1, self._n_output)),
        #    name='ext_slope'
        #)

        slope_unwise = np.zeros((1,2), dtype='f4')
        self._ext_slope_unwise = tf.Variable(
            slope_unwise,
            name='ext_slope_unwise'
                )

        self._ext_slope = tf.Variable(
            tf.zeros((1, self._n_output-2)),
            name='ext_slope'
        )


        self._ext_bias = tf.Variable(
            tf.zeros((1, self._n_output)),
            name='ext_bias'
        )   

        # Initialize neural network        
        self.predict_intrinsic_ln_flux(tf.zeros([1,n_input]))
        self.predict_ln_ext_curve(tf.zeros([1, 1]))
        
        # Initial guess of the extinction curve
        R0_guess = np.log(2 * (sample_wavelengths/550.)**(-1.5))
        R0_guess = R0_guess.astype('float32')
        R1_guess = np.clip((sample_wavelengths-550.)/(992.-392.), -0.5, 0.5)
        R1_guess = R1_guess.astype('float32')
        
        self._ext_bias.assign(R0_guess.reshape(1,-1))
        self._ext_slope.assign(R1_guess[:-2].reshape(1,-1))
        
        # Count the total number of weights
        self._n_flux_weights = sum([int(tf.size(l.w)) for l in self._layers])   
        self._n_ext_weights = int(tf.size(self._ext_slope))

    def predict_intrinsic_ln_flux(self, stellar_type):
        """
        Returns the predicted ln(flux) for the input stellar types,
        at zero extinction and a standard distance (corresponding
        to parallax = 1, in whatever units are being used).
        """
        # Normalize the stellar parameters
        x = (stellar_type - self._input_zp) * self._input_iscale
        # Run the normalized parameters through the neural net
        for layer in self._layers[:-1]:
            x = layer(x)
            x = self._activation(x)
        # No activation on the final layer
        flux = self._layers[-1](x)
        return flux

    def predict_ln_ext_curve(self, xi):
        """
        Returns the predicted ln(ext_curve) for the input xi.
        """
        # Run xi through the neural net
        x = tf.expand_dims(xi, 1) # shape = (star, 1)
        ext_slope_full = tf.concat([self._ext_slope, 
                                    self._ext_slope_unwise], axis=1)
        ln_ext = ext_slope_full*tf.math.tanh(x) + self._ext_bias # shape = (star, wavelength)
        return ln_ext

    def predict_obs_flux(self, stellar_type, xi,
                         stellar_extinction, stellar_parallax):
        """
        Returns predicted flux for the given stellar types,
        at  the given parallaxes and extinctions.
        """
        # Convert extinction curve from log to linear scale
        ext_curve = tf.math.exp(self.predict_ln_ext_curve(xi))
        # All tensors should end up with shape (star, wavelength)
        optical_depth = (
            ext_curve
          * tf.expand_dims(stellar_extinction, axis=1) # Add in wavelength axis
        )
        # Add in wavelength axis. Reference distance = 1/(units of parallax).
        distance_factor = tf.expand_dims(stellar_parallax**2, axis=1)
        # ln(flux) at reference distance
        ln_flux_pred_r0 = self.predict_intrinsic_ln_flux(stellar_type)
        # Flux at parallax distance
        flux_pred = (
            tf.math.exp(ln_flux_pred_r0 - optical_depth)
          * distance_factor
        )
        return flux_pred

    def calc_chi2(self, stellar_type, xi,
                  stellar_extinction, stellar_parallax,
                  flux_obs, flux_sqrticov):
        # Calculate predicted flux
        flux_pred = self.predict_obs_flux(
            stellar_type, xi, stellar_extinction, stellar_parallax
        )
        # chi^2
        dflux = flux_pred - flux_obs
        #print('flux_icov.shape:', flux_icov.shape)
        #print('dflux.shape:', dflux.shape)
        sqrticov_dflux = tf.linalg.matvec(flux_sqrticov, dflux)
        #print('icov_dflux.shape:', icov_dflux.shape)
        chi2 = tf.reduce_sum(sqrticov_dflux*sqrticov_dflux, axis=1)
        #print('chi2.shape:', chi2.shape)
        return chi2

    def roughness(self, stellar_type):
        with tf.GradientTape(watch_accessed_variables=False) as g:
            g.watch(stellar_type)
            ln_flux = self.predict_intrinsic_ln_flux(stellar_type)
        df_dx = g.batch_jacobian(ln_flux, stellar_type) # (star,wl,param)
        rough_i = tf.reduce_mean(df_dx**2, axis=(1,2)) # (star,)
        rough = tf.math.reduce_std(rough_i)
        return rough

    def penalty(self):
        l2_sum = 0.
        # Penalty on stellar model nn weights
        for layer in self._layers:
            l2_sum = l2_sum + tf.reduce_sum(layer.w**2)
        l2_sum = self._l2 * l2_sum / float(self._n_flux_weights)
        # Penalty on variation of the extinction curve
        l2_sum = l2_sum + self._l2_ext_curve*tf.reduce_mean(self._ext_slope**2)
        # Delta wavelength (in units of 10 nm)
        _, wl0 = tf.split(
            self._sample_wavelengths,
            [1,self._n_output-1],
            axis=0
        )
        wl1, _ = tf.split(
            self._sample_wavelengths,
            [self._n_output-1,1],
            axis=0
        )
        dwl = (1./wl0 - 1./wl1) * 30250 # 30250 == 550**2 /10. /10.
        dwl = tf.expand_dims(dwl, axis=0)
        # Penalty on roughness of extinction curve zero point
        _, bias0 = tf.split(self._ext_bias, [1,self._n_output-1], axis=1)
        bias1, _ = tf.split(self._ext_bias, [self._n_output-1,1], axis=1)
        l2_sum = l2_sum + self._l2_ext_curve*tf.reduce_mean(((bias0-bias1)/dwl)**2)
        # Penalty on roughness of extinction curve slope
        dwl_wo_unwise,_ = tf.split(dwl, [self._n_output-3,2], axis=1)
        _, slope0 = tf.split(self._ext_slope, [1,self._n_output-3], axis=1)
        slope1, _ = tf.split(self._ext_slope, [self._n_output-3,1], axis=1)
        l2_sum = l2_sum + self._l2_ext_curve*tf.reduce_mean(((slope0-slope1)/dwl_wo_unwise)**2)
        return  l2_sum

    def get_sample_wavelengths(self):
        return self._sample_wavelengths.numpy()

    def save(self, checkpoint_name):
        # Checkpoint Tensorflow Variables
        checkpoint = tf.train.Checkpoint(flux_model=self)
        checkpoint.save(checkpoint_name)

    @classmethod
    def load(cls, checkpoint_name, latest=False):
        if latest:
            checkpoint_name = tf.train.latest_checkpoint(checkpoint_name)
            print(f'Latest checkpoint: {checkpoint_name}')

        # Inspect checkpoint to find input arguments
        spec = {'n_hidden':-1}
        print('Checkpoint contents:')
        for vname,vshape in tf.train.list_variables(checkpoint_name):
            print(f' * {vname} has shape {vshape}.')
            if vname.startswith('flux_model/_sample_wavelengths/'):
                spec['n_output'], = vshape
            if vname.startswith('flux_model/_layers/0/w/'):
                spec['n_input'], spec['hidden_size'] = vshape
            if vname.startswith('flux_model/_layers/') and '/w/' in vname:
                spec['n_hidden'] += 1
                
        print('Found checkpoint with following specifications:')
        print(json.dumps(spec, indent=2))

        # Create flux model
        flux_model = cls(
            np.linspace(0., 1., spec['n_output']).astype('f4'), # dummy array
            n_input=spec['n_input'],
            n_hidden=spec['n_hidden'],
            hidden_size=spec['hidden_size'],
        )

        # Restore variables
        checkpoint = tf.train.Checkpoint(flux_model=flux_model)
        checkpoint.restore(checkpoint_name)

        print('Loaded the following properties from checkpoint:')
        for var in (flux_model._l2, flux_model._input_zp,
                    flux_model._input_iscale,
                    ):
            print(var)
        print('... in addition to weights and biases.')

        return flux_model


def chi_band(stellar_model, 
             type_est_batch, xi_est_batch, 
             ext_est_batch, plx_est_batch,
             flux_batch, flux_err_batch,
            ):
    '''
    Record the chi of given band.
    '''
    flux_pred = stellar_model.predict_obs_flux(
        type_est_batch, xi_est_batch, ext_est_batch, plx_est_batch
    )
    # chi^2
    dflux = flux_pred - flux_batch
    chi_flux = (dflux/flux_err_batch)**2
    return chi_flux[:, 64:].numpy()


def grads_stellar_model(stellar_model,
                        stellar_type, xi,
                        stellar_extinction, stellar_parallax,
                        flux_obs, flux_sqrticov,
                        extra_weight,
                        chi2_turnover=tf.constant(20.),        
                        roughness_l2=tf.constant(1.),
                        model_update = ['stellar_model', 'ext_curve_w', 'ext_curve_b'],
                        ):
    """
    Calculates the gradients of the loss (i.e., chi^2 + regularization)
    w.r.t. the stellar model parameters, for a given batch of stars.
    """
    # Scale at which chi^2 growth is suppressed
    chi2_factor = chi2_turnover * stellar_model._n_output

    # Only want gradients of stellar model parameters and extinction curve
    trainable_var = []
    if 'stellar_model' in model_update:
            trainable_var += [ f'flux_model/hidden_0/w:{i}'
                for i in range(stellar_model._n_hidden)
            ]
            trainable_var += [ f'flux_model/hidden_0/b:{i}'
                for i in range(stellar_model._n_hidden)
            ]
            trainable_var += [
                 'flux_model/flux/b:0',
                 'flux_model/flux/w:0'
            ]
            
    if 'ext_curve_b' in model_update:
        trainable_var += [
             'flux_model/ext_bias:0',
        ]
    if 'ext_curve_w' in model_update:
        trainable_var += [
            'flux_model/ext_slope:0',
        ]
        
    variables = [i for i in stellar_model.trainable_variables 
                        if i.name in trainable_var]

    with tf.GradientTape(watch_accessed_variables=False) as g:
        g.watch(variables)
        
        # Calculate chi^2 of each star
        chi2 = stellar_model.calc_chi2(
            stellar_type, xi,
            stellar_extinction, stellar_parallax,
            flux_obs, flux_sqrticov
        )

        # Suppress large chi^2 values
        chi2 = chi2_factor * tf.math.asinh(chi2/chi2_factor)

        # Average individual chi^2 values
        chi2 = tf.reduce_sum(extra_weight*chi2) / tf.reduce_sum(extra_weight)
        # Add in penalty (generally, for model complexity)
        loss = chi2 + stellar_model.penalty()

        # Penalty on gradients of model (similar to model discontinuity)
        loss = loss + roughness_l2 * stellar_model.roughness(stellar_type)

    # Calculate d(loss)/d(variables)
    grads = g.gradient(loss, variables)

    return loss, variables, grads


def gaussian_prior(x, mu, sigma):
    return ((x - mu) / sigma)**2


def grads_stellar_params(stellar_model,
                         stellar_type, stellar_type_obs, stellar_type_sigma,
                         xi, 
                         ln_stellar_ext, stellar_ext_obs, stellar_ext_sigma,
                         ln_stellar_plx, stellar_plx_obs, stellar_plx_sigma,
                         flux_obs, flux_sqrticov,
                         extra_weight,
                         chi2_turnover=tf.constant(20.),
                         var_update = ['atm','E','plx','xi'],
                        ):
    # Scale at which chi^2 growth is suppressed
    chi2_factor = chi2_turnover * stellar_model._n_output

    # Only want gradients of stellar model parameters and extinction curve
    variables = ()
    if 'atm' in var_update:
        variables += (stellar_type,)
    if 'E' in var_update:
        variables += (ln_stellar_ext,)
    if 'plx' in var_update:
        variables += (ln_stellar_plx,)
    if 'xi' in var_update:
        variables += (xi,)
        
    with tf.GradientTape(watch_accessed_variables=False) as g:
        g.watch(variables)
        # Convert quantities from log to linear scale
        stellar_plx = tf.math.exp(ln_stellar_plx)
        stellar_ext = tf.math.exp(ln_stellar_ext)
        # Calculate chi^2 of each star
        chi2 = stellar_model.calc_chi2(
            stellar_type, xi, 
            stellar_ext, stellar_plx,
            flux_obs, flux_sqrticov
        )
        # Priors
        prior_type = tf.math.reduce_sum(
            gaussian_prior(
                stellar_type,
                stellar_type_obs,
                stellar_type_sigma
            ),
            axis=1
        )
        prior_plx = (
            gaussian_prior(stellar_plx, stellar_plx_obs, stellar_plx_sigma)
          - 2*ln_stellar_plx # Jacobian of transformation from plx to ln(plx)
        )
        prior_ext = (
            gaussian_prior(stellar_ext, stellar_ext_obs, stellar_ext_sigma)
          - 2*ln_stellar_ext # Jacobian of transformation from E to ln(E)
        )
        prior_xi = gaussian_prior(xi, 0, 1)
        prior = prior_type + prior_plx + prior_ext + prior_xi
        # Posterior (one value per star, as they are independent)
        # Caution: Do not multiply extra weight here, which slows down the update of atm
        posterior = (chi2 + prior)
    # Calculate d(posterior)/d(variables). Separate gradient per star.
    grads = g.gradient(posterior, variables)
    return posterior, variables, grads


def get_batch_iterator(indices, batch_size, n_epochs):
    n = len(indices)
    batches = tf.data.Dataset.from_tensor_slices(indices)
    batches = batches.shuffle(n, reshuffle_each_iteration=True)
    batches = batches.repeat(n_epochs)
    batches = batches.batch(batch_size, drop_remainder=True)
    n_batches = int(np.floor(n * n_epochs / batch_size))
    return batches, n_batches


def identify_outlier_stars(d_train,
                           sigma_clip_teff=4.,
                           sigma_clip_logg=4.,
                           sigma_clip_feh=4.,
                           sigma_clip_plx=4.,
                           sigma_clip_E=np.inf):
    n_stars = d_train['stellar_type_est'].shape[0]
    idx_good = np.ones(n_stars, dtype='bool')

    dparam = (
        (d_train['stellar_type_est']-d_train['stellar_type'])
      / d_train['stellar_type_err']
    )
    dplx = (d_train['plx_est']-d_train['plx']) / d_train['plx_err']
    dE = (
        (d_train['stellar_ext_est']-d_train['stellar_ext'])
      / d_train['stellar_ext_err']
    )

    sigma_clip_param = np.array([
        sigma_clip_teff,sigma_clip_logg,sigma_clip_feh
    ])
    idx_good &= np.all(np.abs(dparam) < sigma_clip_param[None,:], axis=1)
    idx_good &= (np.abs(dplx) < sigma_clip_plx)
    idx_good &= (np.abs(dE) < sigma_clip_E)

    return idx_good


def identify_flux_outliers(data,
                           stellar_model,
                           chi2_dof_clip=4.,
                           chi_indiv_clip=None):
    n_stars = data['stellar_type_est'].shape[0]
    idx_good = np.ones(n_stars, dtype='bool')
    
    length = len(data[f'xi_est'])
    flux_chi = np.zeros(data['flux'].shape)
    for i in tqdm(range(int(length/10000))):
        flux_pred = stellar_model.predict_obs_flux(
            data[f'stellar_type_est'][i*10000: (i+1)*10000],
            data[f'xi_est'][i*10000: (i+1)*10000],
            data[f'stellar_ext_est'][i*10000: (i+1)*10000],
            data[f'plx_est'][i*10000: (i+1)*10000]
        ).numpy()
        flux_err = data['flux_err'][i*10000 : (i+1)*10000]
        flux = data['flux'][i*10000 : (i+1)*10000]
        flux_chi[i*10000 : (i+1)*10000] = (flux_pred - flux) / flux_err

    chi2_dof = np.nanmean(flux_chi**2, axis=1)
    idx_good &= (chi2_dof < chi2_dof_clip)

    if chi_indiv_clip is not None:
        if hasattr(chi_indiv_clip, '__len__'):
            idx_good &= np.all((np.abs(flux_chi)<chi_indiv_clip[None]),axis=1)
        else:
            idx_good &= np.all((np.abs(flux_chi)<chi_indiv_clip),axis=1)

    return idx_good


def train_stellar_model(stellar_model,
                        d_train,
                        extra_weight,
                        idx_train=None,
                        optimize_stellar_model=True,
                        optimize_stellar_params=False,
                        batch_size=128, n_epochs=32,
                        lr_stars_init=1e-3,
                        lr_model_init=1e-4,
                        model_update=['stellar_model','ext_curve_w','ext_curve_b'],
                        var_update=['atm','E','plx','xi'],
                       ):
    
    # Make arrays to hold estimated stellar types
    type_est = d_train['stellar_type_est'].copy()
    ln_ext_est = np.log(np.clip(d_train['stellar_ext_est'], 1.e-4, np.inf))
    ln_plx_est = np.log(np.clip(d_train['plx_est'], 1.e-6, np.inf))
    xi_est = d_train['xi_est'].copy()
    flux_err = d_train['flux_err'].copy()

    # Get training data iterator and determine # of batches
    n_train = len(d_train['plx'])
    if idx_train is None:
        idx_train = np.arange(n_train)
    n_sel = len(idx_train)
    print(f'Training on {n_sel} ({n_sel/n_train*100:.2f}%) sources.')
    batches, n_batches = get_batch_iterator(idx_train, batch_size, n_epochs)

    # Optimizer for stellar model and extinction curve
    n_drops = 4
    lr_model = keras.optimizers.schedules.PiecewiseConstantDecay(
        [int(n_batches*k/n_drops) for k in range(1,n_drops)],
        [lr_model_init*(0.1**k) for k in range(n_drops)]
    )
    opt_model = keras.optimizers.SGD(learning_rate=lr_model, momentum=0.9)
    #opt_model = keras.optimizers.Adam(learning_rate=1e-4)

    @tf.function
    def grad_step_stellar_model(type_b, xi_b, ext_b, 
                                plx_b, flux_b, flux_sqrticov_b, 
                                extra_weight_b,
                                model_update=model_update):
        print('Tracing grad_step_stellar_model ...')
        loss, variables, grads = grads_stellar_model(
            stellar_model,
            type_b, xi_b, 
            ext_b, plx_b,
            flux_b, flux_sqrticov_b,
            extra_weight_b,
            model_update=model_update, 
        )
        grads_clipped, grads_norm = tf.clip_by_global_norm(
            grads, 1000.
        )
        opt_model.apply_gradients(zip(grads_clipped, variables))

        return loss, grads_norm

    # Optimizer for stellar parameters.
    # No momentum, because different stellar parameters fit each batch,
    # so direction of last step is irrelevant.
    n_drops = 4
    lr_stars = keras.optimizers.schedules.PiecewiseConstantDecay(
        [int(n_batches*k/n_drops) for k in range(1,n_drops)],
        [lr_stars_init*(0.1**k) for k in range(n_drops)]
    )
    opt_st_params = keras.optimizers.SGD(learning_rate=lr_stars, momentum=0.)

    st_type_dim = d_train['stellar_type'].shape[1]
    type_est_batch = tf.Variable(
        tf.zeros([batch_size,st_type_dim], dtype=tf.float32),
        name='type_est_batch'
    )
    xi_est_batch = tf.Variable(
        tf.zeros([batch_size], dtype=tf.float32),
        name='xi_est_batch'
    )
    ln_ext_est_batch = tf.Variable(
        tf.zeros([batch_size], dtype=tf.float32),
        name='ln_ext_est_batch'
    )
    ln_plx_est_batch = tf.Variable(
        tf.zeros([batch_size], dtype=tf.float32),
        name='ln_plx_est_batch'
    )

    @tf.function
    def grad_step_stellar_params(type_b, type_obs_b, type_sigma_b,
                                 xi_b, 
                                 ln_ext_b, ext_obs_b, ext_sigma_b,
                                 ln_plx_b, plx_obs_b, plx_sigma_b,
                                 flux_b, flux_sqrticov_b,
                                 extra_weight_b,
                                 var_update=var_update,
                                ):
        print('Tracing grad_step_stellar_params ...')
        loss, variables, grads = grads_stellar_params(
            stellar_model,
            type_b, type_obs_b, type_sigma_b,
            xi_b, 
            ln_ext_b, ext_obs_b, ext_sigma_b,
            ln_plx_b, plx_obs_b, plx_sigma_b,
            flux_b, flux_sqrticov_b,
            extra_weight_b,
            var_update=var_update,
        )
        grads_clipped = [tf.clip_by_norm(g, 1000.) for g in grads]
        opt_st_params.apply_gradients(zip(grads_clipped, variables))
        return loss

    model_loss_hist = []
    stellar_loss_hist = []
    chi_w1_hist = []
    chi_w2_hist = []
    slope_record = []
    bias_record = []

    pbar = tqdm(batches, total=n_batches)
    for b in pbar:
        # Load in data for this batch
        idx = b.numpy()

        type_est_batch.assign(type_est[idx])
        ext_est_batch = tf.constant(np.exp(ln_ext_est[idx]))
        plx_est_batch = tf.constant(np.exp(ln_plx_est[idx]))
        xi_est_batch.assign(xi_est[idx])
        flux_err_batch = flux_err[idx]
        extra_weight_batch = tf.constant(extra_weight[idx])

        if optimize_stellar_params:
            type_obs_batch = tf.constant(d_train['stellar_type'][idx])
            type_err_batch = tf.constant(d_train['stellar_type_err'][idx])

            ln_ext_est_batch.assign(ln_ext_est[idx])
            ext_obs_batch = tf.constant(d_train['stellar_ext'][idx])
            ext_err_batch = tf.constant(d_train['stellar_ext_err'][idx])

            ln_plx_est_batch.assign(ln_plx_est[idx])
            plx_obs_batch = tf.constant(d_train['plx'][idx])
            plx_err_batch = tf.constant(d_train['plx_err'][idx])
            

        flux_batch = tf.constant(d_train['flux'][idx])
        flux_sqrticov_batch = tf.constant(d_train['flux_sqrticov'][idx])

        pbar_disp = {}

        # Take a step in the model parameters
        if optimize_stellar_model:
            loss, grads_norm = grad_step_stellar_model(
                type_est_batch, xi_est_batch, 
                ext_est_batch, plx_est_batch,
                flux_batch, flux_sqrticov_batch,
                extra_weight_batch,
                model_update=model_update,
            )
            mod_loss = float(loss.numpy())
            model_loss_hist.append(mod_loss)
            pbar_disp['mod_loss'] = mod_loss

            if np.isnan(mod_loss):
                check_dict = {
                    'type_est': type_est_batch,
                    'xi_est': xi_est_batch,
                    'ext_est': ext_est_batch,
                    'plx_est': plx_est_batch,
                    'flux': flux_batch,
                    'flux_sqrticov': flux_sqrticov_batch,
                    'extra_weight': extra_weight_batch
                }
                print('NaN model loss!')
                for key in check_dict:
                    v = check_dict[key].numpy()
                    print('  * NaN {key}? {np.any(np.isnan(v))}')
                    raise ValueError('NaN model loss')

            chi_w1, chi_w2 = chi_band(stellar_model, 
                            #tf.constant(np.array([64, 65], dtype='int')), 
                            type_est_batch, xi_est_batch, 
                            ext_est_batch, plx_est_batch,
                            flux_batch, flux_err_batch,
            ).T
            chi_w1_hist.append(np.mean(chi_w1))
            chi_w2_hist.append(np.mean(chi_w2))
            slope_record.append(stellar_model._ext_slope.numpy()[0])
            bias_record.append(stellar_model._ext_bias.numpy()[0])
            #pbar_disp['mod_lr'] = opt_model._decayed_lr(tf.float32).numpy()

        # Take a step in the stellar paramters (for this batch)
        if optimize_stellar_params:
            loss = grad_step_stellar_params(
                type_est_batch, type_obs_batch, type_err_batch,
                xi_est_batch, 
                ln_ext_est_batch, ext_obs_batch, ext_err_batch,
                ln_plx_est_batch, plx_obs_batch, plx_err_batch,
                flux_batch, flux_sqrticov_batch,
                extra_weight_batch,
                var_update=var_update,
            )
            st_loss = float(np.median(loss.numpy()))
            stellar_loss_hist.append(st_loss)
            pbar_disp['st_loss'] = st_loss
            #pbar_disp['st_lr'] = opt_st_params._decayed_lr(tf.float32).numpy()

            if np.isnan(st_loss):
                check_dict = {
                    'type_est (before update)': type_est[idx],
                    'type_est (after update)': type_est_batch,
                    'type_err': type_err_batch,
                    'type_obs': type_obs_batch,
                    'xi_est (before update)': xi_est[idx],
                    'xi_est (after update)': xi_est_batch,
                    'ln_ext_est (before update)': ln_ext_est[idx],
                    'ln_ext_est (after update)': ln_ext_est_batch,
                    'ext_obs': ext_obs_batch,
                    'ext_err': ext_err_batch,
                    'ln_plx_est (before update)': ln_plx_est[idx],
                    'ln_plx_est (after update)': ln_plx_est_batch,
                    'plx_obs': plx_obs_batch,
                    'plx_err': plx_err_batch,
                    'flux': flux_batch,
                    'flux_sqrticov': flux_sqrticov_batch,
                    'extra_weight': extra_weight_batch
                }
                print('NaN stellar loss!')
                for key in check_dict:
                    v = check_dict[key]
                    if isinstance(v, tf.Tensor):
                        v = v.numpy()
                    print(f'  * NaN {key}? {np.any(np.isnan(v))}')
                raise ValueError('NaN stellar loss')

            # Update estimated stellar parameters (for this batch)
            type_est[idx] = type_est_batch.numpy()
            xi_est[idx] = xi_est_batch.numpy()
            ln_ext_est[idx] = ln_ext_est_batch.numpy()
            ln_plx_est[idx] = ln_plx_est_batch.numpy()


        # Display losses in progress bar
        pbar.set_postfix(pbar_disp)

    ret = {}

    if optimize_stellar_model:
        ret['model_loss'] = model_loss_hist
        ret['chi_w1'] = chi_w1_hist
        ret['chi_w2'] = chi_w2_hist
        ret['ext_slope'] = np.vstack(slope_record)
        ret['ext_bias'] = np.vstack(bias_record)

    if optimize_stellar_params:
        ret['stellar_loss'] = stellar_loss_hist
        d_train['stellar_type_est'] = type_est
        d_train['xi_est'] = xi_est
        d_train['stellar_ext_est'] = np.exp(ln_ext_est)
        d_train['plx_est'] = np.exp(ln_plx_est)
        ret['chi_w1'] = chi_w1_hist
        ret['chi_w2'] = chi_w2_hist

    return ret


class GaussianMixtureModel(snt.Module):
    def __init__(self, n_dim, n_components=5):
        super(GaussianMixtureModel, self).__init__(name='gauss_mix_model')

        self._n_dim = n_dim
        self._n_components = n_components

        #self._n_components = tf.Variable(
        #    n_components,
        #    name='n_components',
        #    dtype=tf.int32,
        #    trainable=False
        #)
        self._weight = tf.Variable(
            tf.zeros([n_components]),
            name='weight',
            dtype=tf.float32,
            trainable=False
        )
        self._ln_norm = tf.Variable(
            tf.zeros([1,n_components]),
            name='ln_norm',
            dtype=tf.float32,
            trainable=False
        )
        self._mean = tf.Variable(
            tf.zeros([1,n_components,n_dim]),
            name='mean',
            dtype=tf.float32,
            trainable=False
        )
        self._sqrt_icov = tf.Variable(
            tf.zeros([1,n_components,n_dim,n_dim]),
            name='sqrt_icov',
            dtype=tf.float32,
            trainable=False
        )
        self._sqrt_cov = tf.Variable(
            tf.zeros([n_components,n_dim,n_dim]),
            name='sqrt_cov',
            dtype=tf.float32,
            trainable=False
        )

    def save(self, checkpoint_name):
        # Checkpoint Tensorflow Variables
        checkpoint = tf.train.Checkpoint(gauss_mix_model=self)
        checkpoint.save(checkpoint_name)

    @classmethod
    def load(cls, checkpoint_name, latest=False):
        if latest:
            checkpoint_name = tf.train.latest_checkpoint(checkpoint_name)
            print(f'Latest checkpoint: {checkpoint_name}')

        # Inspect checkpoint to find input arguments
        spec = {}
        for vname,vshape in tf.train.list_variables(checkpoint_name):
            if vname.startswith('gauss_mix_model/_mean/'):
                print(vname)
                print(vshape)
                _,spec['n_components'],spec['n_dim'] = vshape

        print('Found checkpoint with following specifications:')
        print(json.dumps(spec, indent=2))

        # Create model
        gauss_mix_model = cls(spec['n_dim'], n_components=spec['n_components'])

        # Restore variables
        checkpoint = tf.train.Checkpoint(gauss_mix_model=gauss_mix_model)
        checkpoint.restore(checkpoint_name)

        return gauss_mix_model

    def fit(self, x):
        from sklearn.mixture import BayesianGaussianMixture
        gmm = BayesianGaussianMixture(
            n_components=self._n_components,
            weight_concentration_prior=0.25/self._n_components,
            reg_covar=1e-3,
            max_iter=2048,
            n_init=1,
            verbose=2
        )
        gmm.fit(x)

        ln_cov_det = [np.linalg.slogdet(c)[1] for c in gmm.covariances_]
        ln_norm = np.stack([
            np.log(w)-0.5*d for w,d in zip(gmm.weights_,ln_cov_det)
        ])
        ln_norm.shape = (1,) + ln_norm.shape
        self._ln_norm.assign(ln_norm)
        self._weight.assign(gmm.weights_)
        print(self._weight)

        mu = np.reshape(gmm.means_, (1,)+gmm.means_.shape)
        self._mean.assign(mu)

        sqrt_icov = []
        sqrt_cov = []
        for c in gmm.covariances_:
            U,sqrt_c,(eival0,_) = sqrt_icov_eigen(c, return_sqrtcov=True)
            if eival0 <= 0.:
                print(f'Non-positive eigenvalue(s) in GMM cov! {eival0} <= 0.')
            sqrt_icov.append(U.T)
            sqrt_cov.append(sqrt_c)
        sqrt_icov = np.stack(sqrt_icov)
        sqrt_icov.shape = (1,) + sqrt_icov.shape
        sqrt_cov = np.stack(sqrt_cov)
        self._sqrt_icov.assign(sqrt_icov)
        self._sqrt_cov.assign(sqrt_cov)

    def ln_prob(self, x):
        # Expand dimensions of x to be consistent with (star, component, dim)
        x = tf.expand_dims(x, axis=1) # Dummy component axis

        # Calculate ln(p) of each component
        dx = x - self._mean # shape = (star, comp, dim)
        # shape = (star, comp, dim):
        sqrt_icov_dx = tf.linalg.matvec(self._sqrt_icov, dx)
        # shape = (star, comp):
        chi2 = tf.reduce_sum(sqrt_icov_dx*sqrt_icov_dx, axis=2)
        lnp_comp = self._ln_norm - 0.5*chi2 # shape = (star, comp)

        # Sum the probabilities of the components and take the log
        lnp = tf.reduce_logsumexp(lnp_comp, axis=1) # shape = (star,)

        return lnp

    def sample(self, n, rng=None):
        if rng is None:
            rng = np.random.default_rng()
        # Draw component indices. Shape = (sample,)
        comp_idx = rng.choice(
            np.arange(self._n_components),
            p=self._weight.numpy(),
            size=n
        )
        # Draw unit normal vectors. Shape = (sample, dim).
        v = rng.normal(size=(n,self._n_dim)).astype('f4')
        # Transform unit normal vectors. Shape = (sample, dim).
        Av = np.einsum(
            'sij,sj->si',
            self._sqrt_cov.numpy()[comp_idx,:,:], v,
            optimize=True
        )
        # Add in means. Shape = (sample, dim).
        z = self._mean.numpy()[0,comp_idx,:] + Av
        return z


def plot_gmm_prior(stellar_type_prior, base_path='.', overlay_track=None):
    # Plot samples from prior
    import corner

    n_comp = stellar_type_prior._n_components
    params = stellar_type_prior.sample(1024**2*10)
    n_params = params.shape[1]

    #labels = [
    #    r'$T_{\mathrm{eff}}$',
    #    r'$\left[\mathrm{Fe/H}\right]$',
    #    r'$\log g$',
    #]
    labels = [rf'$\mathrm{{param\ {i}}}$' for i in range(n_params)]

    ranges = [
        plot_utils.choose_lim(p, pct=[0.1,99.9], expand=0.4)
        for p in params.T
    ]
    
    fig = corner.corner(
        params,
        labels=labels,
        range=ranges,
        bins=100,
        quantiles=[0.16, 0.5, 0.84],
        levels=[0.68,0.95,0.99],
        show_titles=True,
        color='midnightblue'
        #pcolor_kwargs={'cmap':cm.Blues}
    )

    fn = os.path.join(
        base_path, 'plots',
        f'stellar_prior_samples_{n_comp:02d}components'
    )
    fig.savefig(fn)
    plt.close(fig)

    # Plot 2D marginal distributions of prior
    param_ranges = [np.linspace(r0,r1,100) for r0,r1 in ranges]
    param_grid = np.meshgrid(*param_ranges, indexing='ij')
    param_grid = np.stack(param_grid, axis=-1)
    s = param_grid.shape[:-1]
    param_grid.shape = (-1,n_params)
    param_grid = tf.constant(param_grid.astype('f4'))

    lnp = batch_apply_tf(
        stellar_type_prior.ln_prob,
        1024,
        param_grid,
        function=True,
        progress=True,
        numpy=True
    )

    lnp.shape = s

    for suffix in ('linear', 'log'):
        fig = plt.figure(figsize=(2*(n_params-1),)*2, layout='constrained')
        gs = GridSpec(n_params-1, n_params-1, figure=fig)

        for i in range(0,n_params-1):
            for j in range(i,n_params-1):
                dims = i, j+1
                ax = fig.add_subplot(gs[j,i])

                # Marginalize over all but the selected dimensions
                sum_dims = [k for k in range(n_params) if k not in dims]
                lnp_marginal = lnp
                for k in sum_dims:
                    lnp_marginal = logsumexp(
                        lnp_marginal,
                        axis=k,
                        keepdims=True
                    )
                lnp_marginal = np.squeeze(lnp_marginal, axis=tuple(sum_dims))
                lnp_marginal -= np.max(lnp_marginal)
                p_marginal = np.exp(lnp_marginal)

                if suffix == 'linear':
                    img = p_marginal
                else:
                    img = lnp_marginal

                im = ax.imshow(
                    img.T,
                    origin='lower',
                    extent=ranges[dims[0]]+ranges[dims[1]],
                    interpolation='nearest',
                    aspect='auto'
                )

                p_ordered = np.sort(p_marginal.flat)
                P_cumulative = np.cumsum(p_ordered)
                P_cumulative /= P_cumulative[-1]
                pct_levels = [0.001, 0.01]
                idx_p = np.searchsorted(P_cumulative, pct_levels) + 1
                p_levels = p_ordered[idx_p]
                cs = ax.contour(
                    param_ranges[dims[0]],
                    param_ranges[dims[1]],
                    p_marginal.T,
                    levels=p_levels,
                    colors='w',
                    alpha=0.4,
                    linewidths=0.75
                )
                p_fmt = {p:f'{l:.1%}' for p,l in zip(cs.levels,pct_levels)}
                ax.clabel(cs, cs.levels, fmt=p_fmt, inline=True, fontsize=7)

                # Overlay track?
                if overlay_track is not None:
                    curves = [overlay_track['curve_smooth'][:,d] for d in dims]
                    ax.plot(curves[0], curves[1], c='gray')

                if dims[1] == n_params-1:
                    ax.set_xlabel(labels[dims[0]])
                else:
                    ax.set_xticklabels([])

                if dims[0] == 0:
                    ax.set_ylabel(labels[dims[1]])
                else:
                    ax.set_yticklabels([])

                ax.set_xlim(ranges[dims[0]])
                ax.set_ylim(ranges[dims[1]])

        fn = f'stellar_prior_marginals_{n_comp:02d}components'
        if overlay_track is not None:
            fn += f'_track_{overlay_track["title"]}'
        fn += f'_{suffix}'
        fn = os.path.join(base_path, 'plots', fn)
        fig.savefig(fn)
        plt.close(fig)

        
def calculate_stellar_type_tracks(stellar_type_prior):
    # Draw samples from the prior, to determine parameter ranges
    # and point of maximum prior density
    prior_samples = stellar_type_prior.sample(1024*32)
    d = prior_samples.shape[1] # Dimensionality of stellar-type space
    param_lims = [
        plot_utils.choose_lim(prior_samples[:,i], pct=[0.1,99.9])
        for i in range(d)
    ]
    
    prior_values = stellar_type_prior.ln_prob(prior_samples)
    idx_maxprior = np.argmax(prior_values)
    param_maxprior = prior_samples[idx_maxprior]
    
    lnp_low = np.percentile(prior_values, 1.)
    
    # Function to maximize the prior, holding one variable fixed
    def conditional_optimize_type(type_init, hold_dim, n_steps=128, reg=0.):
        p0 = tf.Variable(np.reshape(type_init,(1,-1)), dtype=tf.float32)

        param_list = [tf.Variable(p) for p in tf.unstack(p0, axis=1)]
        watch_params = [p for i,p in enumerate(param_list) if i!=hold_dim]

        opt = tf.keras.optimizers.experimental.SGD(
            learning_rate=3e-3,
            momentum=0.5
        )

        for i in range(n_steps):
            with tf.GradientTape(watch_accessed_variables=False) as g:
                g.watch(watch_params)
                p = tf.stack(param_list, axis=1)
                loss = -stellar_type_prior.ln_prob(p)
                loss = loss + reg*tf.reduce_sum((p-p0)**2, axis=1)
                loss = tf.reduce_sum(loss)
            grads = g.gradient(loss, watch_params)
            opt.apply_gradients(zip(grads, watch_params))

        p1 = tf.stack(param_list, axis=1)
        return p1
    
    # Extend a curve from starting point outwards, changing one dimension
    # independently, and setting the other dimensions to follow the ridge
    # of highest prior density
    def extend_type_curve(indep_dim, dp):
        p = param_maxprior.copy()
        p_list = [p.copy()]
        
        p_indep_range = np.arange(
            param_maxprior[indep_dim],
            param_lims[indep_dim][0 if dp<0 else 1],
            dp
        )

        for p_i in p_indep_range:
            p[indep_dim] += dp
            p = conditional_optimize_type(p, indep_dim).numpy()[0]
            p_list.append(p.copy())
            
            lnp = stellar_type_prior.ln_prob(np.reshape(p,(1,-1)))[0]
            if lnp < lnp_low:
                break

        p_list = np.stack(p_list, axis=0)
        return p_list
    
    # Extend a curve out in both directions, and then stitch them together
    def construct_type_curve(indep_dim):
        p_high = extend_type_curve(indep_dim, 0.2)
        p_low = extend_type_curve(indep_dim, -0.2)
        p_curve = np.concatenate([p_low[::-1][:-1], p_high], axis=0)
        return p_curve
    
    def extend_straight(indep_dim, dp):
        p_indep_range = np.arange(
            param_maxprior[indep_dim],
            param_lims[indep_dim][0 if dp<0 else 1],
            dp
        )
        p = np.repeat(
            np.reshape(param_maxprior,(1,-1)),
            len(p_indep_range),
            axis=0
        )
        p[:,indep_dim] = p_indep_range
        lnp = stellar_type_prior.ln_prob(p).numpy()
        i1 = np.where(lnp < lnp_low)[0]
        if len(i1):
            p = p[:i1[0]]
        return p
    
    # Construct a simple curve, in which all but one dimension are held
    # constant
    def construct_straight_curve(indep_dim):
        p_high = extend_straight(indep_dim, 0.2)
        p_low = extend_straight(indep_dim, -0.2)
        p_curve = np.concatenate([p_low[::-1][:-1], p_high], axis=0)
        return p_curve
    
    def smoothed_curve(p, indep_dim, s=0.1):
        curve = []
        for i in range(d):
            if i == indep_dim:
                curve.append(p[:,i])
            else:
                spl = UnivariateSpline(p[:,indep_dim], p[:,i], s=s)
                curve.append(spl(p[:,indep_dim]))
        curve = np.stack(curve, axis=1)
        return curve
    
    curves = [
        {'curve':construct_type_curve(i), 'indep_dim':i, 'title':f'curve_{i}'}
        for i in range(d)
    ]
    curves += [
        {'curve':construct_straight_curve(i), 'indep_dim':i, 'title':f'ray_{i}'}
        for i in range(d)
    ]
    
    for c in curves:
        c['curve_smooth'] = smoothed_curve(c['curve'], c['indep_dim'])
    
    return curves


def save_stellar_type_tracks(tracks, fname):
    with h5py.File(fname, 'w') as f:
        for i,t in enumerate(tracks):
            gp = f'track_{i}'
            for k in ('curve', 'curve_smooth'):
                f[f'{gp}/{k}'] = t[k]
            for k in ('indep_dim', 'title'):
                f[gp].attrs[k] = t[k]


def load_stellar_type_tracks(fname):
    tracks = []
    with h5py.File(fname, 'r') as f:
        n_groups = len(f.keys())
        for i in range(n_groups):
            gp = f'track_{i}'
            t = {}
            for k in ('curve', 'curve_smooth'):
                t[k] = f[f'{gp}/{k}'][:]
            for k in ('indep_dim', 'title'):
                t[k] = f[gp].attrs[k]
            tracks.append(t)
    return tracks


def plot_stellar_model(flux_model, track,
                       show_lines=('hydrogen','metals','molecules')):
    sample_wavelengths = flux_model.get_sample_wavelengths()

    def Ryd_wl(n0, n1):
        return (1 / (const.Ryd * (1/n0**2 - 1/n1**2))).to('nm').value

    lines = {
        'hydrogen': [
            (r'$\mathrm{Balmer\ series}$', [Ryd_wl(2,n) for n in (3,4,5,6)],
                'g', ':'),
            (r'$\mathrm{Paschen\ series}$', [Ryd_wl(3,n) for n in (8,9,10)],
                'b', ':')
        ],
        'metals': [
            (r'$\mathrm{Mg}$', [518.362], 'orange', '-'),
            (r'$\mathrm{Ca}$', [422.6727, 430.774, 854.2], 'violet', '-'),
            (r'$\mathrm{Fe}$', [431.,438.,527.], 'purple', '-')
        ],
        'molecules': [
            #(r'$\mathrm{TiO}$', [632.2,656.9,665.1,705.3,766.6,820.6,843.2],
            #    'gray', '--'),
            (r'$\mathrm{TiO}$', [675.,715.,775.], 'gray', '--'),
            (r'$\mathrm{CaH}$', [685.0], 'salmon', '--')
        ]
    }
    
    p = track['curve_smooth']
    c = p[:,track['indep_dim']]
    flux = np.exp(flux_model.predict_intrinsic_ln_flux(p))

    fig = plt.figure(figsize=(6,4), layout='constrained')
    ax = fig.add_subplot(1,1,1)

    norm = Normalize(np.min(c), np.max(c))
    cmap = plt.get_cmap('coolwarm')
    c = cmap(norm(c))

    for fl,cc in zip(flux,c):
        ax.loglog(sample_wavelengths, fl, c=cc)

    cb = fig.colorbar(
        ScalarMappable(norm=norm, cmap=cmap),
        ax=ax,
        label=fr'$\mathrm{{parameter\ {track["indep_dim"]}}}$'
    )

    ax.set_xlabel(r'$\lambda\ \left(\mathrm{nm}\right)$')

    ax.set_ylabel(
        r'$f_{\lambda}\ '
        r'\left(10^{-18}\,\mathrm{W\,m^{-2}\,nm^{-1}}\right)$'
    )
    
    ax.grid(True, which='major', alpha=0.2)
    ax.grid(True, which='minor', alpha=0.05)

    for key in show_lines:
        for line_label,line_wl,c,ls in lines[key]:
            for i,wl in enumerate(line_wl):
                ax.axvline(
                    wl,
                    color=c, ls=ls, lw=1., alpha=0.4,
                    label=line_label if i==0 else None
                )

    legend = ax.legend(fontsize=5,loc=4)
    frame = legend.get_frame() 
    frame.set_alpha(1) 
    frame.set_facecolor('white')
    
    return fig, ax


def quick_RV_estimate(flux_model, xi, return_curve=False):
    """
    Returns a quick-and-dirty estimate of R(V) for the given values of
    xi. A correct calculation of R(V) requires a source spectrum and
    a full bandpass response function. This function simply uses the
    extinction curve at the effective wavelengths of the B and V filters.
    This is a useful estimate for making quick plots, but should be
    avoided when a precise value of R(V) is required for a particular source.

    Inputs:
      flux_model (FluxModel): The stellar/extinction model to use.
      xi (float or array of floats): The values of xi at which to calculate
          R(V).

      return_curve (Optional[bool]): If `True`, the full extinction curve will
          also be returned.

    Returns:
      Approximate R(V) values at the given values of xi. If `return_true` is
      `True`, then the full extinction curve will also be returned.
    """
    # Calculate extinction curve at given values of xi
    xi = np.array(xi).astype('f4')
    R = np.exp(flux_model.predict_ln_ext_curve(xi).numpy())

    # Interpolate extinction curve to effective wavelengths of B and V
    lam_BV = np.array([431.8, 533.5]) # in nm
    sample_wavelengths = flux_model.get_sample_wavelengths()
    idx_BV = np.searchsorted(sample_wavelengths, lam_BV)
    lam1 = sample_wavelengths[idx_BV]
    lam0 = sample_wavelengths[idx_BV-1]
    a1 = (lam_BV - lam0) / (lam1 - lam0) # Linear interpolation coefficient

    R_BV = (1-a1)*R[:,idx_BV-1] + a1*R[:,idx_BV]
    R_V = R_BV[:,1] / (R_BV[:,0] - R_BV[:,1])

    if return_curve:
        return R_V, R

    return R_V


def plot_extinction_curve(flux_model, show_variation=True):
    sample_wavelengths = flux_model.get_sample_wavelengths()
    if show_variation:
        xi = np.linspace(-0.75, 0.75, 11, dtype='f4')
    else:
        xi = np.array([0.], dtype='f4')

    R_V,R_lam = quick_RV_estimate(flux_model, xi, return_curve=True)

    fig = plt.figure(figsize=(6,4), layout='constrained')
    ax = fig.add_subplot(1,1,1)

    norm = Normalize(np.min(R_V), np.max(R_V))
    cmap = plt.get_cmap('coolwarm_r')
    c = cmap(norm(R_V))

    for xi_i,R_i,RV_i,cc in zip(xi,R_lam,R_V,c):
        ax.semilogx(sample_wavelengths, R_i, c=cc)

    cb = fig.colorbar(
        ScalarMappable(norm=norm, cmap=cmap),
        ax=ax,
        label=r'$R_V$'
    )

    ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())
    ax.grid(True, which='major', alpha=0.2)
    ax.grid(True, which='minor', alpha=0.05)

    ax.set_xlabel(r'$\lambda\ \left(\mathrm{nm}\right)$')
    ax.set_ylabel(r'$R_{\lambda}\ \left(\mathrm{mag}\right)$')
    
    return fig, ax


def plot_RV_histogram(flux_model, data):
    RV = quick_RV_estimate(flux_model, data['xi_est'])

    idx = (data['stellar_ext_est'] > 0.1)

    fig,ax = plt.subplots(1, 1, figsize=(6,5))

    kw = dict(histtype='step', range=(1.,9.), bins=100, log=True, alpha=0.8)
    ax.hist(RV[idx], label=r'$E > 0.1$', **kw)
    ax.hist(RV[~idx], label=r'$E < 0.1$', **kw)

    ax.legend(loc='upper right')
    ax.set_xlabel(r'$R(V)$')

    ax.xaxis.set_minor_locator(ticker.AutoMinorLocator())
    ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())
    ax.grid(True, which='major', alpha=0.2)
    ax.grid(True, which='minor', alpha=0.05)

    return fig, ax


def plot_RV_skymap(flux_model, data, nside=16):
    RV = quick_RV_estimate(flux_model, data['xi_est'])

    idx = (RV > 0.) & (RV < 10.) & (data['stellar_ext_est'] < 10.)

    RV_healpix_map = plot_utils.healpix_mean_map(
        data['ra'][idx]*units.deg,
        data['dec'][idx]*units.deg,
        RV[idx],
        nside,
        weights=data['stellar_ext_est'][idx]
    )
    
    fig = plt.figure(figsize=(6,3.5), layout='constrained')

    _,ax,im = plot_utils.plot_mollweide(
        RV_healpix_map,
        w=2000,
        fig=fig,
        input_frame='icrs',
        plot_frame='galactic',
        cmap='coolwarm_r',
        vmin=2.1, vmax=4.1
    )

    cb = fig.colorbar(
        im, ax=ax,
        label=r'$R(V)$',
        location='bottom',
        shrink=0.75,
        pad=0.02,
        aspect=40
    )

    return fig, ax


def grid_search_stellar_params(flux_model, data,
                               gmm_prior=None, rng=None,
                               batch_size=128,
                               sample_by_prior=False
                               ):
    # Set up stellar type grid
    if sample_by_prior:
        type_grid = gmm_prior.sample(256, rng=rng)
    else:
        teff_range = np.arange(3., 9.01, 0.3, dtype='f4') # 21 grids
        feh_range = np.arange(-2.0, 0.701, 0.3, dtype='f4') # 10 grids
        logg_range = np.arange(-1.0, 5.01, 0.3, dtype='f4') # 21 grids
        
        teff_grid,feh_grid,logg_grid = np.meshgrid(
            teff_range, feh_range, logg_range
        )
        teff_grid.shape = (-1,)
        feh_grid.shape = (-1,)
        logg_grid.shape = (-1,)
        
        # Reject grid points that are rare in the training set
        type_grid = np.stack([teff_grid,feh_grid,logg_grid], axis=1)
        ln_prior_grid = gmm_prior.ln_prob(type_grid)
        select_by_prior = (ln_prior_grid>-5.41)
        teff_grid = teff_grid[select_by_prior]
        feh_grid = feh_grid[select_by_prior]
        logg_grid = logg_grid[select_by_prior]
        type_grid = np.stack([teff_grid,feh_grid,logg_grid], axis=1)

    # Cartesian product with extinction grid
    xi_range = np.arange(-0.6, 0.601, 0.3, dtype='f4') # 5 grids
    ext_range = 10**np.arange(-2.0, 1.01, 0.05, dtype='f4') # 61 grids
    idx_ext, idx_type, idx_xi = np.indices((ext_range.size, type_grid.shape[0], xi_range.size))
    ext_grid = ext_range[idx_ext]
    type_grid = type_grid[idx_type]
    xi_grid = xi_range[idx_xi]

    type_grid.shape = (-1,3)
    ext_grid.shape = (-1,)
    xi_grid.shape = (-1,)
    plx_grid = np.ones_like(ext_grid)
    
    
    # Calculate model flux over parameter grid, assuming plx = 1 mas
    flux_grid = flux_model.predict_obs_flux(type_grid, xi_grid, ext_grid, plx_grid)
    flux_grid_exp = tf.expand_dims(flux_grid, 1) # Shape = (param, 1, wavelength)

    n_stars = len(data['plx'])

    # Empty Tensors to hold batches of observed data
    n_wl = data['flux'].shape[1]
    flux_obs_b = tf.Variable(tf.zeros((batch_size,n_wl)))
    flux_sqrticov_b = tf.Variable(tf.zeros((batch_size,n_wl,n_wl)))
    plx_obs_b = tf.Variable(tf.zeros((batch_size,)))
    plx_sigma_b = tf.Variable(tf.zeros((batch_size,)))

    @tf.function
    def determine_best_param_idx():
        print('Tracing determine_best_param_idx() ...')
        
        # Insert dummy stellar parameter axis into observations
        # Shape = (1, star, wavelength)
        flux_obs_b_exp = tf.expand_dims(flux_obs_b, 0)
        # Shape = (1, star, wavelength, wavelength)
        flux_sqrticov_b_exp = tf.expand_dims(flux_sqrticov_b, 0)
        # Shape = (1, star)
        plx_obs_b_exp = tf.expand_dims(plx_obs_b, 0)
        # Shape = (1, star)
        plx_sigma_b_exp = tf.expand_dims(plx_sigma_b, 0)

        # Easy to guess decent parallax, by looking at flux ratio.
        # Only use middle third of spectrum, as the edges are noisier.
        # Would like to use median (which is available through
        # tensorflow_probability), but it is significantly slower than
        # mean.
        _,flux_obs_b_core,_ = tf.split(flux_obs_b_exp, 3, axis=2)
        _,flux_grid_core,_ = tf.split(flux_grid_exp, 3, axis=2)
        # ReLU ensures non-negativity inside sqrt:
        plx_b = tf.math.sqrt(tf.reduce_mean(
            tf.nn.relu(flux_obs_b_core / flux_grid_core),
            axis=2,
            keepdims=True
        )) # Shape = (param, star, 1)

        # Calculate chi^2
        # Shape = (param, star, wavelength)
        dflux = flux_grid_exp * plx_b**2 - flux_obs_b_exp
        # Shape = (param, star, wavelength)
        sqrticov_dflux = tf.linalg.matvec(flux_sqrticov_b_exp, dflux)
        # Shape = (param, star)
        chi2 = tf.reduce_sum(sqrticov_dflux*sqrticov_dflux, axis=2)

        #tf.print('NaN chi2:', tf.math.count_nonzero(tf.math.is_nan(chi2)))
        #tf.print('neg flux_obs_core:', tf.math.count_nonzero(tf.math.is_nan(flux_obs_b_core/flux_grid_core)))
        #tf.print('NaN flux ratio:', tf.math.count_nonzero(tf.math.is_nan(flux_obs_b_core/flux_grid_core)))
        #tf.print('NaN plx:', tf.math.count_nonzero(tf.math.is_nan(plx_b)))
        #tf.print('NaN flux_obs:', tf.math.count_nonzero(tf.math.is_nan(flux_obs_b_exp)))
        #tf.print('NaN flux_grid:', tf.math.count_nonzero(tf.math.is_nan(flux_grid_exp)))
        #tf.print('NaN dflux:', tf.math.count_nonzero(tf.math.is_nan(dflux)))
        #tf.print('NaN flux_sqrticov:', tf.math.count_nonzero(tf.math.is_nan(flux_sqrticov_b_exp)))
        #tf.print('NaN sqrticov_dflux:', tf.math.count_nonzero(tf.math.is_nan(sqrticov_dflux)))

        # Add in prior term for parallax
        prior = gaussian_prior(
            tf.squeeze(plx_b, axis=2),
            plx_obs_b_exp,
            plx_sigma_b_exp
        )
        #tf.print('NaN prior:', tf.math.count_nonzero(tf.math.is_nan(prior)))
        chi2 = chi2 + prior

        # Determine best chi^2 (+prior) per star
        pidx_best_b = tf.math.argmin(chi2, axis=0) # Shape = (star,)
        chi2_best_b = tf.math.reduce_min(chi2, axis=0) # Shape = (star,)

        return pidx_best_b, chi2_best_b, plx_b

    param_idx_best = []
    chi2_best = []
    plx_best = []

    pbar = tqdm(range(0, n_stars, batch_size))
    for i0 in pbar:
        # Load in data for this batch
        i1 = min(i0+batch_size, n_stars) # Actual final idx of batch
        idx = slice(i0, i1)
        idx0 = slice(0, i1-i0) # Goes from 0 to actual batch size

        assign_variable_padded(flux_obs_b, data['flux'][idx])
        assign_variable_padded(flux_sqrticov_b, data['flux_sqrticov'][idx])
        assign_variable_padded(plx_obs_b, data['plx'][idx])
        assign_variable_padded(plx_sigma_b, data['plx_err'][idx], fill=1)

        # Locate best fit
        pidx_best, chi2, plx = determine_best_param_idx()

        pidx_best = pidx_best.numpy()
        chi2 = chi2.numpy()

        # Extract best parallax (array w/ 1 plx per param value was returned)
        plx = plx.numpy()
        plx = plx[pidx_best, np.arange(plx.shape[1]), 0]

        param_idx_best.append(pidx_best[idx0])
        chi2_best.append(chi2[idx0])
        plx_best.append(plx[idx0])

        chi2_pct = np.percentile(chi2[idx0],[5.,50.,95., 99.])
        chi2_str = '('+','.join([f'{c:.1f}' for c in chi2_pct])+')'
        pbar.set_postfix({
            'chi2_{5,50,95,99}': chi2_str
        })

        idx_bad = np.where(~np.isfinite(chi2[idx0]))[0]
        if len(idx_bad):
            print('plx_opt:', plx[idx_bad])
            print('plx_obs:', data['plx'][idx][idx_bad])
            print('plx_err:', data['plx_err'][idx][idx_bad])
            print('mean flux:', np.mean(data['flux'][idx][idx_bad], axis=1))
            n_flux_bad = np.count_nonzero(
                ~np.isfinite(data['flux'][idx][idx_bad]),
                axis=1
            )
            print('NaN flux:', n_flux_bad)
            n_sqrticov_bad = np.sum(
                np.count_nonzero(
                    ~np.isfinite(data['flux_sqrticov'][idx][idx_bad]),
                    axis=1
                ),
                axis=1
            )
            print('NaN flux_sqrticov:', n_sqrticov_bad)

    param_idx_best = np.hstack(param_idx_best)
    chi2_best = np.hstack(chi2_best)
    plx_best = np.hstack(plx_best)

    type_best = type_grid[param_idx_best]
    ext_best = ext_grid[param_idx_best]
    xi_best = xi_grid[param_idx_best]
    
    data['stellar_type_est'] = type_best
    data['xi_est'] = xi_best
    data['stellar_ext_est'] = ext_best
    data['plx_est'] = plx_best #data['plx'].copy()
    

def ln_prior_clipped(use_prior, st_type_b, ln_prior_clip):
    ln_prior_by_gmm = use_prior.ln_prob(st_type_b)
    #prior_type = (
    #    ln_prior_by_gmm + ln_prior_clip 
    #  - tf.math.log(
    #        tf.math.exp(ln_prior_clip)
    #      + tf.math.exp(ln_prior_by_gmm)
    #    )
    #)
    return ln_prior_by_gmm

    
def optimize_stellar_params(flux_model, data,
                            n_steps=64*1024,
                            lr_init=0.01,
                            reduce_lr_every=16*1024,
                            batch_size=4096,
                            optimizer='adam',
                            use_prior=None,
                            ln_prior_clip=-15.,
                            optimize_subset=None):
    if 'stellar_type' in data:
        n_type = data['stellar_type'].shape[1]
    else:
        n_type = 3
    n_wl = data['flux'].shape[1]

    # Initial values of ln(extinction) and ln(parallax)
    ln_stellar_ext_all = np.log(np.clip(data['stellar_ext_est'], 1e-9, np.inf))
    ln_stellar_plx_all = np.log(np.clip(data['plx_est'], 1e-9, np.inf))
    xi_all = data['xi_est'].copy()

    # Empty Tensors to hold batches of stellar parameters
    st_type_b = tf.Variable(tf.zeros((batch_size,n_type)))
    ln_ext_b = tf.Variable(tf.zeros((batch_size,)))
    ln_plx_b = tf.Variable(tf.zeros((batch_size,)))
    xi_b = tf.Variable(tf.zeros((batch_size,)))

    # Empty Tensors to hold batches of observed data
    flux_obs_b = tf.Variable(tf.zeros((batch_size,n_wl)))
    flux_sqrticov_b = tf.Variable(tf.zeros((batch_size,n_wl,n_wl)))
    plx_obs_b = tf.Variable(tf.zeros((batch_size,)))
    plx_sigma_b = tf.Variable(tf.zeros((batch_size,)))
    ext_obs_b = tf.Variable(tf.zeros((batch_size,)))
    ext_sigma_b = tf.Variable(tf.zeros((batch_size,)))
    if use_prior == 'observation':
        st_type_obs_b = tf.Variable(tf.zeros((batch_size,n_type)))
        st_type_sigma_b = tf.Variable(tf.zeros((batch_size,n_type)))

    # Optimizers
    if optimizer == 'adam':
        opt = tf.keras.optimizers.Adam(learning_rate=lr_init)
    else:
        opt = tf.keras.optimizers.SGD(learning_rate=lr_init, momentum=0.)

    def stellar_loss(use_pr='observation'):
        # Convert quantities from log to linear scale
        ext_b = tf.math.exp(ln_ext_b)
        plx_b = tf.math.exp(ln_plx_b)
        # Calculate chi^2 of each star
        chi2 = flux_model.calc_chi2(
            st_type_b, xi_b, ext_b, plx_b,
            flux_obs_b, flux_sqrticov_b
        )
        # We always have a parallax observation
        prior_plx = (
            gaussian_prior(plx_b, plx_obs_b, plx_sigma_b)
          - 2*ln_plx_b # Jacobian of transformation from plx to ln(plx)
        )
        # Can always set a Gaussian prior on extinction (even if sigma -> inf)
        prior_ext = (
            gaussian_prior(ext_b, ext_obs_b, ext_sigma_b)
          - 2*ln_ext_b # Jacobian of transformation from E to ln(E)
        )
        prior_xi = gaussian_prior(xi_b, 0, 1)
        prior = prior_plx + prior_ext + prior_xi
        # Stellar type prior: multiple options
        prior_type = 0.
        if use_pr == 'observation':
            print('Using prior: observation')
            prior_type = tf.math.reduce_sum(
                gaussian_prior(st_type_b, st_type_obs_b, st_type_sigma_b),
                axis=1
            )
        elif isinstance(use_pr, GaussianMixtureModel):
            print('Using prior: GMM')
            prior_type = -2.*ln_prior_clipped(
                use_prior,
                st_type_b,
                ln_prior_clip
            )
        elif use_pr is None:
            print('Using prior: None (chi^2 only)')
            prior_type = 0.
        else:
            raise ValueError(
                f'Invalid <use_prior> value ("{use_prior}")'
                ' - can be None, "observation" or a GaussianMixtureModel.'
            )
        #tf.print(prior_type)
        prior = prior + prior_type
        # Combine chi^2 and prior
        loss = chi2 + prior
        return loss

    @tf.function
    def step(use_pr='observation'):
        print('Tracing step() ...')
        variables = [st_type_b, xi_b, ln_plx_b, ln_ext_b]
        with tf.GradientTape(watch_accessed_variables=False) as g:
            g.watch(variables)
            loss = stellar_loss(use_pr=use_pr)
        grads = g.gradient(loss, variables)
        
        norm = tf.zeros_like(loss)
        for g in grads:
            if len(g.shape) == 2:
                norm += tf.reduce_sum(g**2, axis=1)
            else:
                norm += g**2
        norm = tf.math.sqrt(norm)
        #tf.print('norm:')
        #tf.print(norm)
        inorm = tf.clip_by_value(100/norm, 0, 1)
        processed_grads = []
        for g in grads:
            if len(g.shape) == 2:
                processed_grads.append(g*tf.expand_dims(inorm, 1))
            else:
                processed_grads.append(g*inorm)

        opt.apply_gradients(zip(processed_grads, variables))
              
        #processed_grads = [tf.clip_by_norm(g, 100.) for g in grads]
        #opt.apply_gradients(zip(grads, variables))
        return loss

    if optimize_subset is None:
        n_stars = len(data['plx'])
    else:
        n_stars = optimize_subset.size

    for i0 in tqdm(range(0, n_stars, batch_size)):
        # Load in data for this batch
        if optimize_subset is None:
            i1 = min(i0+batch_size, n_stars) # Actual final idx of batch
            idx = slice(i0, i1)
            batch_size_actual = i1 - i0
        else:
            idx = optimize_subset[i0:i0+batch_size]
            batch_size_actual = idx.size
        idx0 = slice(0, batch_size_actual) # Goes from 0 to actual batch size

        # Initial stellar parameter values
        assign_variable_padded(st_type_b, data['stellar_type_est'][idx])
        assign_variable_padded(ln_ext_b, ln_stellar_ext_all[idx])
        assign_variable_padded(ln_plx_b, ln_stellar_plx_all[idx])
        assign_variable_padded(xi_b, xi_all[idx])

        # Observed data
        assign_variable_padded(flux_obs_b, data['flux'][idx])
        assign_variable_padded(flux_sqrticov_b, data['flux_sqrticov'][idx])
        assign_variable_padded(plx_obs_b, data['plx'][idx])
        assign_variable_padded(plx_sigma_b, data['plx_err'][idx], fill=1)
        assign_variable_padded(ext_obs_b, data['stellar_ext'][idx])
        assign_variable_padded(
            ext_sigma_b,
            data['stellar_ext_err'][idx],
            fill=1
        )
        if use_prior == 'observation':
            assign_variable_padded(st_type_obs_b, data['stellar_type'][idx])
            assign_variable_padded(
                st_type_sigma_b,
                data['stellar_type_err'][idx],
                fill=1
            )

        lr = lr_init
        opt.learning_rate.assign(lr)

        # Clear memory of optimizer, which is defined globally
        for optvar in opt.variables():
            optvar.assign(tf.zeros_like(optvar))

        pbar = tqdm(range(n_steps))
        for j in pbar:
            loss = step(use_pr=use_prior)

            # Decrease learning rate every set # of steps
            if j % reduce_lr_every == reduce_lr_every-1:
                lr = lr * 0.5 # Careful not to change lr_init!
                opt.learning_rate.assign(lr)

            # Show the loss and learning rate in the progress bar
            loss_pct = np.percentile(loss[idx0], [5., 50., 95., 99.])
            loss_str = '('+','.join([f'{c:.1f}' for c in loss_pct])+')'
            pbar.set_postfix({
                'loss_{5,50,95,99}': loss_str,
                'lr': lr
            })

        idx_bad = ~np.isfinite(loss[idx0])
        if np.any(idx_bad):
            idx_bad = np.where(idx_bad)[0]
            print('Non-finite loss!')
            print('Initial guesses:')
            type_guess = data['stellar_type_est'][idx][idx_bad]
            ext_guess = data['stellar_ext_est'][idx][idx_bad]
            plx_guess = data['plx_est'][idx][idx_bad]
            print('  * stellar type:', type_guess)
            print('  * extinction:', ext_guess)
            print('  * parallax:', plx_guess)
            type_bounds = np.min(type_guess,axis=0), np.max(type_guess,axis=0)
            ext_bounds = np.min(ext_guess), np.max(ext_guess)
            plx_bounds = np.min(plx_guess), np.max(plx_guess)
            print('  - stellar type bounds:', type_bounds)
            print('  - extinction bounds:', ext_bounds)
            print('  - parallax bounds:', plx_bounds)
            print('Optimized values:')
            type_opt = st_type_b.numpy()[idx_bad]
            ext_opt = ln_ext_b.numpy()[idx_bad]
            plx_opt = ln_plx_b.numpy()[idx_bad]
            print('  * stellar type:', type_opt)
            print('  * extinction:', ext_opt)
            print('  * parallax:', plx_opt)
            type_bounds = np.min(type_opt,axis=0), np.max(type_opt,axis=0)
            ext_bounds = np.min(ext_opt), np.max(ext_opt)
            plx_bounds = np.min(plx_opt), np.max(plx_opt)
            print('  - stellar type bounds:', type_bounds)
            print('  - extinction bounds:', ext_bounds)
            print('  - parallax bounds:', plx_bounds)
            print('Data:')
            ext_bounds = np.min(ext_obs_b.numpy()), np.max(ext_obs_b.numpy())
            plx_bounds = np.min(plx_obs_b.numpy()), np.max(plx_obs_b.numpy())
            flux_nonfinite = np.count_nonzero(~np.isfinite(flux_obs_b.numpy()))
            cov_nonfinite = np.count_nonzero(~np.isfinite(flux_sqrticov_b.numpy()))
            print(' - parallax bounds:', plx_bounds)
            print(' - extinction bounds:', ext_bounds)
            print(' - flux non-finite:', flux_nonfinite)
            print(' - flux_sqrticov non-finite:', cov_nonfinite)

        data['stellar_type_est'][idx] = st_type_b.numpy()[idx0]
        data['stellar_ext_est'][idx] = np.exp(ln_ext_b.numpy()[idx0])
        data['plx_est'][idx] = np.exp(ln_plx_b.numpy()[idx0])
        data['xi_est'][idx] = xi_b.numpy()[idx0]
        

def assign_variable_padded(var, value, fill=0):
    """
    Assigns values to a tf.Variable from a numpy.ndarray.
    If the numpy array is smaller along the batch axis (axis=0),
    it will be zero-padded at the end to the same shape as the
    Variable.
    """
    var_size = var.shape[0]
    value_size = value.shape[0]
    if var_size == value_size:
        var.assign(value)
    elif value_size < var_size:
        value_padded = np.full(
            var.shape, fill,
            dtype=var.dtype.as_numpy_dtype()
        )
        value_padded[:value_size] = value
        var.assign(value_padded)
    else:
        raise ValueError(
            'value.shape[0] > var.shape[0]: '
            f'({value.shape[0]} > {var.shape[0]})'
        )


def calc_stellar_fisher_hessian(stellar_model, data, gmm=None,
                                ln_prior_clip=-15.,
                                batch_size=1024, ignore_wl=None):
    n_sources,st_type_dim = data['stellar_type_est'].shape
    n_wl = len(stellar_model.get_sample_wavelengths())

    type_b = tf.Variable(
        tf.zeros([batch_size,st_type_dim], dtype=tf.float32),
        name='type_est_batch'
    )
    xi_b = tf.Variable(
        tf.zeros([batch_size], dtype=tf.float32),
        name='xi_est_batch'
    )
    ext_b = tf.Variable(
        tf.zeros([batch_size], dtype=tf.float32),
        name='ext_est_batch'
    )
    plx_b = tf.Variable(
        tf.zeros([batch_size], dtype=tf.float32),
        name='plx_est_batch'
    )
    flux_obs_b = tf.Variable(
        tf.zeros([batch_size,n_wl], dtype=tf.float32),
        name='flux_batch'
    )
    flux_sqrticov_b = tf.Variable(
        tf.zeros([batch_size,n_wl,n_wl], dtype=tf.float32),
        name='flux_sqrticov_batch'
    )

    wl_mask = np.ones(n_wl, dtype='f4')
    if ignore_wl is not None:
        wl_mask[ignore_wl] = 0.
    wl_mask.shape = (1,-1,1)
    wl_mask = tf.constant(wl_mask, name='wavelength_mask')

    @tf.function
    def fisher_batch():
        print('Tracing fisher_batch ...')

        theta = tf.concat([
            type_b,
            tf.expand_dims(xi_b, axis=1),
            tf.expand_dims(ext_b, axis=1),
            tf.expand_dims(plx_b, axis=1)
        ], axis=1)

        with tf.GradientTape(watch_accessed_variables=False) as g:
            g.watch(theta)

            t,xi,ext,plx = tf.split(theta, [3,1,1,1], axis=1)

            xi = tf.squeeze(xi, axis=1)
            ext = tf.squeeze(ext, axis=1)
            plx = tf.squeeze(plx, axis=1)

            # Calculate flux of each star
            flux_pred = stellar_model.predict_obs_flux(t, xi, ext, plx)

        dflux_dtheta = g.batch_jacobian(flux_pred, theta)

        dflux_dtheta = dflux_dtheta * wl_mask

        # Left-multiply by sqrt(flux_icov)
        A = tf.linalg.matmul(flux_sqrticov_b, dflux_dtheta)
        #print('A.shape:', tf.shape(A))

        fisher_I = tf.linalg.matmul(A, A, transpose_a=True)
        #print('fisher_I.shape:', tf.shape(fisher_I))

        return fisher_I

    @tf.function
    def hessian_batch():
        print('Tracing hessian_batch ...')
        theta = tf.concat([
            type_b,
            tf.expand_dims(xi_b, axis=1),
            tf.expand_dims(ext_b, axis=1),
            tf.expand_dims(plx_b, axis=1)
        ], axis=1)

        with tf.GradientTape(watch_accessed_variables=False,
                             persistent=True) as g2:
            g2.watch(theta)
            with tf.GradientTape(watch_accessed_variables=False,
                                 persistent=True) as g1:
                g1.watch(theta)
                t,xi,ext,plx = tf.split(theta, [3,1,1,1], axis=1)
                xi = tf.squeeze(xi, axis=1)
                ext = tf.squeeze(ext, axis=1)
                plx = tf.squeeze(plx, axis=1)

                # Calculate chi^2 of each star
                chi2 = stellar_model.calc_chi2(
                    t, xi, ext, plx,
                    flux_obs_b, flux_sqrticov_b
                )

                # Calculate GMM prior for each star
                if gmm is not None:
                    #ln_prior_gmm = gmm.ln_prob(t)
                    ln_prior_gmm = ln_prior_clipped(
                            gmm,
                            t,
                            ln_prior_clip
                        )

            # Calculate d(chi^2)/d(variables). Separate gradient per star.
            grads = g1.gradient(chi2, theta)
            if gmm is not None:
                grads_gmm = g1.gradient(ln_prior_gmm, theta)

        hess = 0.5 * g2.batch_jacobian(grads, theta)
        if gmm is not None:
            hess_gmm = -1. * g2.batch_jacobian(grads_gmm, theta)
        else:
            hess_gmm = None

        return grads, hess, hess_gmm, chi2

    fisher_out = []
    hessian_out = []
    hessian_gmm_out = []
    chi2_out = []
    ivar_priors = []

    for i0 in tqdm(range(0,n_sources,batch_size)):
        # Load in data for this batch
        i1 = min(i0+batch_size, n_sources)
        idx = slice(i0, i1)
        idx_zeroed = slice(0, i1-i0)

        assign_variable_padded(type_b, data['stellar_type_est'][idx])
        assign_variable_padded(xi_b, data['xi_est'][idx])
        assign_variable_padded(ext_b, data['stellar_ext_est'][idx])
        assign_variable_padded(plx_b, data['plx_est'][idx])
        assign_variable_padded(flux_obs_b, data['flux'][idx])
        assign_variable_padded(flux_sqrticov_b, data['flux_sqrticov'][idx])

        if 'stellar_type_err' in data:
            st_type_err = data['stellar_type_err'][idx]
        else:
            st_type_err = np.full((i1-i0,st_type_dim), np.inf, dtype='f4')

        prior_std = np.concatenate([
            st_type_err,
            np.ones((data['xi_est'][idx].shape[0], 1)),# prior of xi: standard normal distribution
            np.reshape(data['stellar_ext_err'][idx], (-1,1)),
            np.reshape(data['plx_err'][idx], (-1,1))
        ], axis=1)
        ivar_priors.append(1/prior_std**2)

        #if ignore_bands is not None:
        #    for b in ignore_bands:
        #        flux_sqrticov[:,ignore_bands,ignore_bands] = 0.

        # Calculate Fisher matrix
        ret = fisher_batch().numpy()
        fisher_out.append(ret[idx_zeroed])

        # Calculate Hessian matrix
        ret_grads, ret_hess, ret_hess_gmm, ret_chi2 = hessian_batch()
        hessian_out.append(ret_hess.numpy()[idx_zeroed])
        if gmm is not None:
            hessian_gmm_out.append(ret_hess_gmm.numpy()[idx_zeroed])
        chi2_out.append(ret_chi2.numpy()[idx_zeroed])

    fisher_out = np.concatenate(fisher_out, axis=0)
    hessian_out = np.concatenate(hessian_out, axis=0)
    if gmm is not None:
        hessian_gmm_out = np.concatenate(hessian_gmm_out, axis=0)
    chi2_out = np.concatenate(chi2_out)
    ivar_priors = np.concatenate(ivar_priors, axis=0)

    data['fisher'] = fisher_out
    data['hessian'] = hessian_out
    if gmm is not None:
        data['hessian_gmm'] = hessian_gmm_out
    data['ivar_priors'] = ivar_priors
    data['chi2_opt'] = chi2_out

def load_data(fname, type_err_floor=(0.02,0.02,0.02),
              validation_frac=0.2, seed=1, n_max=None):
    d_train, d_val = {}, {}
    shuffle_idx, n_val = None, None

    with h5py.File(fname, 'r') as f:
        for key in f:
            d = f[key][:n_max]

            # Shuffle and split into training/validation sets
            if shuffle_idx is None:
                rng = np.random.default_rng(seed=seed)
                shuffle_idx = np.arange(d.shape[0], dtype='i8')
                rng.shuffle(shuffle_idx)
                n_val = int(np.ceil(d.shape[0] * validation_frac))
                d_train['shuffle_idx'] = shuffle_idx[:-n_val]
                d_val['shuffle_idx'] = shuffle_idx[-n_val:]
            d = d[shuffle_idx]
            d_train[key] = d[:-n_val]
            d_val[key] = d[-n_val:]

        sample_wavelengths = f['flux'].attrs['sample_wavelengths'][:]

    for d in (d_train, d_val):
        # Reduce unWISE uncertainties. TODO: Remove this treatment.
        #WISE_flux = d['flux'][:,-2:]
        #WISE_flux_var = (d['flux_err'][:,-2:]**2)
        #WISE_flux_var -= (0.01*WISE_flux)**2
        #WISE_flux_err = np.sqrt(WISE_flux_var)
        #d['flux_err'][:,-2:] = WISE_flux_err
        #d['flux_sqrticov'][:,-2,-2] = 1/WISE_flux_err[:,0]
        #d['flux_sqrticov'][:,-1,-1] = 1/WISE_flux_err[:,1]

        # Remove sources with NaN fluxes or extinctions
        idx_goodflux = np.all(np.isfinite(d['flux']), axis=1)
        if not np.all(idx_goodflux):
            n_bad = np.count_nonzero(~idx_goodflux)
            print(f'{n_bad} sources with NaN fluxes.')

        idx_goodext = np.isfinite(d['stellar_ext'])
        if not np.all(idx_goodext):
            n_bad = np.count_nonzero(~idx_goodext)
            print(f'{n_bad} sources with NaN extinctions.')

        idx = idx_goodext & idx_goodflux
        if not np.all(idx):
            n_bad = np.count_nonzero(~idx)
            print(f'  -> Removing {n_bad} sources.')
            for key in d:
                d[key] = d[key][idx]

        # Replace negative uncertainties with default value
        # TODO: Re-evaluate this treatment
        for p,(l,err) in enumerate([('teff',0.5),('logg',0.5),('feh',0.3)]):
            idx = (d['stellar_type_err'][:,p] < 0.)
            n_bad = np.count_nonzero(idx)
            print(f'{n_bad} sources with negative uncertainties in {l}.')
            d['stellar_type_err'][idx,p] = err

        # Add in error floor to stellar types
        for k,err_floor in enumerate(type_err_floor):
            d['stellar_type_err'][:,k] = np.sqrt(
                d['stellar_type_err'][:,k]**2 + err_floor**2
            )

        # Fix some metadata fields
        idx = np.isfinite(d['norm_dg'])
        d['norm_dg'][~idx] = np.nanmin(d['norm_dg'])

        # Ignore WISE observations when very close neighbor detected
        idx = (d['norm_dg'] > -10.)
        n_bad = np.count_nonzero(idx)
        pct_bad = 100 * n_bad / idx.size
        print(f'{n_bad} sources ({pct_bad:.3f}%) with very close/bright '
              'neighbors.')
        print('  -> Ignore W1,2 bands.')
        d['flux_err'][idx,-2:] = np.inf

        # Ignore 2MASS observations when close neighbor detected
        idx = (d['norm_dg'] > -5.)
        n_bad = np.count_nonzero(idx)
        pct_bad = 100 * n_bad / idx.size
        print(f'{n_bad} sources ({pct_bad:.3f}%) with less close/bright '
              'neighbors.')
        print('  -> Ignore 2MASS bands.')
        d['flux_err'][idx,-5:-2] = np.inf

    return d_train, d_val, sample_wavelengths
    

def save_as_h5(d, name):
    print(f'Saving as {name}')
    with h5py.File(name, 'w') as f:
        for key in d.keys():
            f[key] = d[key]
    return 0


def load_h5(name):
    print(f'Loading {name}')
    d = {}
    with h5py.File(name, 'r') as f:
        for key in f.keys():
            d[key] = f[key][:]
    return d     

