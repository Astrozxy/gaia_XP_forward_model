import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import h5py 
from tqdm import tqdm
import dustmaps as dm
from glob import glob

import astropy.units as units
from astropy.coordinates import SkyCoord
from astropy.table import Table

from dustmaps.bayestar import BayestarQuery
bayestar = BayestarQuery(max_samples=100)
from dustmaps.sfd import SFDQuery
sfd = SFDQuery()
#from dustmaps.planck import PlanckQuery
#planck = PlanckQuery(component='tau')

import os
fnames = glob('data/xp_continuous_metadata/*.h5')
fnames.sort()

import scipy.stats
lower = -5
upper = 5
mu = 0.
sigma = 1.
N = 1000

gaussian_generator  = scipy.stats.truncnorm.rvs(
          (lower-mu)/sigma,(upper-mu)/sigma,loc=mu,scale=sigma,size=N)

def calculate_E(fn):
    idx = fn[-16:-3]    
    # This is consuming too much time.
    # TODO: change the form of metadata
    d = Table.read(fn)
    
    # Using corrected parallax
    parallax = np.array(d['parallax'])
    parallax_err = np.array(d['parallax_error'])
    err_over_parallax = parallax_err/parallax
    ra = np.array(d['ra'])
    dec = np.array(d['dec'])

    # Generate samples to represent uncertainty of distance
    all_distance = [ 1./np.clip(parallax+parallax_err*gen, 
                                    a_min=0.0001, a_max=np.inf)
                                    for gen in gaussian_generator ]
    all_distance = np.vstack(all_distance).T   
    all_position = [ SkyCoord(ra*units.deg,dec*units.deg,
                                  distance=all_distance[:, j]*units.kpc,
                                  frame='icrs') 
                                for j, gen in enumerate(gaussian_generator)]
    
    #L_scale = 1.35 # Scale length, kpc
    #prior = all_distance**2*np.exp(-all_distance/L_scale)
    
    prior = np.ones(all_distance.shape) # Flat prior
    prior_normalized = prior/np.sum(prior,axis=1).reshape(-1,1)

    # Query extinction values of bayestar
    E_individual_bayestar = np.array([bayestar(coords, mode='samples') for coords in all_position])
    E_mean_bayestar = np.einsum('jik,ij->i',E_individual_bayestar, prior_normalized)/5.
    res = (E_individual_bayestar-E_mean_bayestar.reshape(1,-1,1))
    E_sigma_bayestar = (np.einsum('jik,ij->i',res**2,prior_normalized)/5.)**0.5

    # SFD : 2D
    E_individual_sfd = np.array([sfd(coords) for coords in all_position])
    E_mean_sfd = np.mean(E_individual_sfd, axis=0)
        
    # Planck: 2D
    #E_individual_planck = np.array([planck(coords) for coords in all_position])
    #E_mean_planck = np.mean(E_individual_planck, axis=0)

    # Saving the output
    with h5py.File('data/xp_bayestar_match/'+f'xp_reddening_match_{idx}.h5', 'w') as f:
        f['E_mean_bayestar'] = E_mean_bayestar
        f['E_sigma_bayestar'] = E_sigma_bayestar
        f['E_mean_sfd'] = E_mean_sfd
        #f['E_mean_planck'] = E_mean_planck
        f['gdr3_source_id'] = np.array(d['gdr3_source_id'])

from p_tqdm import p_map

if __name__ =='__main__':
    nprocess = 16
    all_E_sigma = p_map(calculate_E , 
                        fnames, 
                        num_cpus = nprocess
                        )      
