from model import *
from xp_utils import *


def gather_data_into_file(fname_out,
                          sample_wavelengths,
                          data_dir='data',
                          flux_scale=1e-18,
                          chunk_size=1024,
                          thin=None):
    xp_sampler = XPSampler(sample_wavelengths, flux_scale=flux_scale)

    lamost_dir = os.path.join(data_dir, 'xp_lamost_dr8_match')
    unwise_dir = os.path.join(data_dir, 'xp_unwise_match')
    tmass_dir = os.path.join(data_dir, 'xp_tmass_match')
    reddening_dir = os.path.join(data_dir, 'xp_dustmap_match')
    meta_dir = os.path.join(data_dir, 'xp_continuous_metadata')
    xp_dir = os.path.join(data_dir, 'xp_continuous_mean_spectrum')

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
    fname = os.path.join('data', 'lamost/dr8_v2.0_LRS_stellar.csv')
    print(f'Loading LAMOST: {fname} ...')
    lamost_cat = Table.read(fname, format='ascii.csv')
    print(f' -> {len(lamost_cat)} sources.')

    # Load the HotPayne catalog
    fname = os.path.join('data', 'lamost/hot_payne.fits.gz')
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

    #lamost_cat.add_column(specid, name='SpecID')
    #lamost_cat = astropy.table.join(
    #    lamost_cat, hotpayne_cat,
    #    join_type='left', keys='SpecID'
    #)
    #for k in lamost_cat.colnames:
    #    if k.endswith('_1'):
    #        lamost_cat.rename_column(k, k[:-2])

    # Prefer HotPayne catalog values, where available
    print('Replacing LAMOST cat values with HotPayne values, where avail...')
    #idx_replace = ~lamost_cat['Teff'].mask
    #idx_replace = ~hotpayne_cat['Teff'].mask[idx_hotpayne]
    #idx_hotpayne = idx_hotpayne[idx_replace]
    #idx_lamost = idx_lamost[idx_replace]
    #print(f' -> Replacing {np.count_nonzero(idx_replace)} rows ...')
    print(f' -> Replacing {idx_hotpayne.size} rows ...')
    key_replace = [
        ('teff', 'Teff'),
        ('teff_err', 'e_Teff'),
        #('logg', 'logg_2'),
        ('logg', 'logg'),
        ('logg_err', 'e_logg'),
        ('feh', '[Fe/H]'),
        ('feh_err', 'e_[Fe/H]')
    ]
    for k1,k2 in key_replace:
        print(f' * {k2} -> {k1}')
        #lamost_cat[k1][idx_replace] = lamost_cat[k2][idx_replace]
        lamost_cat[k1][idx_lamost] = hotpayne_cat[k2][idx_hotpayne]

    # Stellar atmospheric parameters
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
    # Gaia astrometry
    plx_field = 'parallax'
    plx_err_field = 'parallax_error'
    # Gaia BP/RP spectra
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
        'rp_standard_deviation',
    ]
    # Stellar extinction
    stellar_ext_err_floor_abs = 0.03
    stellar_ext_err_floor_pct = 0.10

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
    print(f'tmass_f0 = {tmass_f0}')
    tmass_f0.shape = (1,3)

    with h5py.File(fname_out, 'w') as fout:
        for i,fn in enumerate(tqdm(lamost_fnames, smoothing=1/100)):
            fid = fn.split('_')[-1].split('.')[0]

            d = {} # Empty dict for data to store

            # Load information on LAMOST - Gaia XP matches
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

            # Load stellar type from LAMOST
            n_params = len(type_fields)
            d['stellar_type'] = np.empty((n,n_params), dtype='f4')
            d['stellar_type_err'] = np.empty((n,n_params), dtype='f4')
            for k,(key0,key1) in enumerate(zip(type_fields,type_err_fields)):
                d['stellar_type'][:,k] = lamost_cat[key0][lamost_idx]
                d['stellar_type_err'][:,k] = lamost_cat[key1][lamost_idx]
            # Convert T_eff to kiloKelvins
            d['stellar_type'][:,0] *= 0.001
            d['stellar_type_err'][:,0] *= 0.001

            # Gaia information
            meta_fn = os.path.join(
                meta_dir,
                f'xp_metadata_{fid}.h5'
            )
            m = {}
            f = Table.read(meta_fn)
            for key in f.keys():
                m[key] = np.array(f[key])
            f = []
            # Check that GDR3 source IDs match
            assert np.all(m['source_id'][gaia_idx] == d['gdr3_source_id'])

            # Load Gaia photometry (not used in model)
            for band in ('g', 'bp', 'rp'):
                for suffix in ('', '_error'):
                    key = f'phot_{band}_mean_flux{suffix}'
                    d[key] = m[key][gaia_idx].astype('f4')

            # Load sky coordinates
            for key in ('ra', 'dec'):
                d[key] = m[key][gaia_idx]

            # Load parallax
            d['plx'] = m[plx_field][gaia_idx].astype('f4')
            d['plx_err'] = m[plx_err_field][gaia_idx].astype('f4')

            # Load Gaia quality flags
            for key in ('fidelity_v2', 'phot_bp_rp_excess_factor', 'norm_dg'):
                d[key] = m[key][gaia_idx]

            # Filter out bad data
            idx_good = np.ones(n, dtype="bool")
            # LAMOST S/N
            for b in 'gri':
                idx_good &= (lamost_cat[f'snr{b}'][lamost_idx] > 20.)
            # LAMOST uncertainties (TODO: update these tolerances?)
            idx_good &= np.all(
                (d['stellar_type_err'] < np.array([0.5, 0.5, 0.5])[None,:])
              & (d['stellar_type_err'] > 0.),
                axis=1
            )
            # Parallax S/N
            idx_good &= (d['plx']/d['plx_err'] > 3.) # TODO: Make this cut later?
            # Gaia astrometric fidelity
            idx_good &= (d['fidelity_v2'] > 0.5)
            # Gaia BP/RP excess
            idx_good &= (d['phot_bp_rp_excess_factor'] < 1.3)

            n_good = np.count_nonzero(idx_good)
            if n_good == 0:
                print(f'  No good data in {fn}!')
                continue
            print(f'  Keeping {n_good} of {n} sources.')

            lamost_idx = lamost_idx[idx_good]
            gaia_idx = gaia_idx[idx_good]

            for key in d:
                d[key] = d[key][idx_good]

            xp_fn = os.path.join(xp_dir, f'XpContinuousMeanSpectrum_{fid}.h5')
            with h5py.File(xp_fn, 'r') as f:
                # Load BP/RP spectra
                bprp_data = [f[key][:][gaia_idx] for key in bprp_fields]

            # Load stellar extinctions
            ext_fname = os.path.join(reddening_dir, 'xp_reddening_match_'+fid+'.h5')
            with h5py.File(ext_fname, 'r') as f:
                d['stellar_ext'] = f['E_mean_bayestar'][:].astype('f4')[gaia_idx]
                d['stellar_ext_err'] = f['E_sigma_bayestar'][:].astype('f4')[gaia_idx]
                ext_gdr3_source_id = f['gdr3_source_id'][:][gaia_idx]
            d['stellar_ext_err'] = np.sqrt(
                d['stellar_ext_err']**2
              + (stellar_ext_err_floor_pct*d['stellar_ext'])**2
              + stellar_ext_err_floor_abs**2
            )
            # Check that the Gaia DR3 source IDs match
            assert np.all(d['gdr3_source_id'] == ext_gdr3_source_id)

            # Load unWISE photometry
            unwise_fname = os.path.join(
                unwise_dir,
                f'match_xp_continuous_unwise_{fid}.h5'
            )
            with h5py.File(unwise_fname, 'r') as f:
                d_unwise = {k: f[k][:] for k in f}

            # Join unWISE to already selected data
            unwise_sort_idx = np.argsort(d_unwise['gdr3_source_id'])
            insert_idx = np.searchsorted(
                d_unwise['gdr3_source_id'],
                d['gdr3_source_id'],
                sorter=unwise_sort_idx
            )
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
            )

            # Insert unWISE columns with default values (for non-observations)
            n = d['plx'].shape[0]
            d['unwise_flux'] = np.full((n,2), np.nanmedian(unwise_flux,axis=0))
            d['unwise_flux_var'] = np.full((n,2), np.inf)

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

            # Inflate unWISE errors
            wise_errmul = (1.0, 1.0) # Multiplicative factor
            wise_errfr = (0.01, 0.01) # Error floor, in % of flux
            for b,(errfrac,errmult) in enumerate(zip(wise_errfr,wise_errmul)):
                d['unwise_flux_var'][:,b] *= errmult**2
                d['unwise_flux_var'][:,b] += (errfrac*d['unwise_flux'][:,b])**2
            ## Add in 1% uncertainty floor to unWISE W1 flux errors
            #d['unwise_flux_var'][:,0] += (0.01*d['unwise_flux'][:,0])**2
            ## Add in 2% uncertainty floor to unWISE W2 flux errors
            #d['unwise_flux_var'][:,1] += (0.02*d['unwise_flux'][:,1])**2
            unwise_flux_err = np.sqrt(d['unwise_flux_var'])

            # Load 2MASS photometry
            tmass_fname = os.path.join(
                tmass_dir,
                f'xp_tmass_match_{fid}.fits.gz'
            )
            print(f'tmass_fname = {tmass_fname}')
            d_tmass = Table.read(tmass_fname, format='fits')

            # Join 2MASS to already selected data
            tmass_idx = np.searchsorted(
                d_tmass['source_id'],
                d['gdr3_source_id']
            )
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
                if not len(d_tmass):
                    print('No 2MASS sources.')
                    continue
                # Only use high-quality photometry (phot_qual == 'A')
                idx_phot = np.array(
                    [s[k]=='A' for s in d_tmass['ph_qual']]
                )
                pct_phot = np.count_nonzero(idx_phot)/(idx_phot.size+1e-10)*100
                print(f'{b}: {pct_phot:.5f}% A')
                tmass_mag[idx_phot,k] = d_tmass[f'{b}_m'][idx_phot]
                tmass_mag_err[idx_phot,k] = d_tmass[f'{b}_msigcom'][idx_phot]

            # Additionally require 2MASS magnitude uncertainty < 0.2
            idx_unphot = tmass_mag_err > 0.2
            tmass_mag[idx_unphot] = np.nan
            tmass_mag_err[idx_unphot] = np.nan

            # Calculate 2MASS fluxes, in units of the flux scale
            # (which is 10^{-18} W/m^2/nm, by default)
            tmass_flux = 10**(-0.4*tmass_mag) * tmass_f0
            tmass_flux_err = 2.5/np.log(10) * tmass_mag_err * tmass_flux

            # Insert 2MASS columns with default values (for non-observations)
            n = d['plx'].shape[0]
            d['tmass_flux'] = np.full((n,3), np.nanmedian(tmass_flux,axis=0))
            d['tmass_flux_var'] = np.full((n,3), np.inf)

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

            # Add in uncertainty floor to 2MASS flux errors
            tmass_errfr = (0.01, 0.01, 0.01)
            for b,errfrac in enumerate(tmass_errfr):
                d['tmass_flux_var'][:,b] += (errfrac*d['tmass_flux'][:,b])**2
            #d['tmass_flux_var'] += (0.005*d['tmass_flux'])**2
            tmass_flux_err = np.sqrt(d['tmass_flux_var'])

            # Get BP/RP fluxes and their inverse covariances
            n_b = 5 # Number of photometric bands to be appended to spectrum
            n_wl = sample_wavelengths.size
            flux = np.zeros((n_good,n_wl+n_b), dtype='f4')
            flux_err = np.full((n_good,n_wl+n_b), np.nan, dtype='f4')
            flux_sqrticov = np.zeros((n_good,n_wl+n_b,n_wl+n_b), dtype='f4')
            flux_cov_eival_min = np.empty(n_good, dtype='f4')
            flux_cov_eival_max = np.empty(n_good, dtype='f4')
            #for j,k in enumerate(np.where(idx_good)[0]):
            for j in range(len(lamost_idx)):
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
                U,(eival0,eival1) = sqrt_icov_eigen(fl_cov, eival_floor=1e-9)#condition_max=1e6)
                flux_sqrticov[j,:n_wl,:n_wl] = U.T
                flux_cov_eival_min[j] = eival0
                flux_cov_eival_max[j] = eival1

            n_neg_eival = np.count_nonzero(eival0<=0)
            pct_neg_eival = n_neg_eival/eival0.size * 100
            print(f'{n_neg_eival} negative eigenvalues ({pct_neg_eival:.4f}%).')

            # Paste in 2MASS photometric bands
            flux[:,-5:-2] = d['tmass_flux']
            flux_err[:,-5:-2] = tmass_flux_err
            for b in range(3):
                flux_sqrticov[:,-5+b,-5+b] = 1/tmass_flux_err[:,b]

            # Paste in WISE photometric bands
            flux[:,-2:] = d['unwise_flux']
            flux_err[:,-2:] = unwise_flux_err
            flux_sqrticov[:,-2,-2] = 1/unwise_flux_err[:,0]
            flux_sqrticov[:,-1,-1] = 1/unwise_flux_err[:,1]

            d['flux'] = flux
            d['flux_err'] = flux_err
            d['flux_sqrticov'] = flux_sqrticov
            d['flux_cov_eival_min'] = flux_cov_eival_min
            d['flux_cov_eival_max'] = flux_cov_eival_max

            # Write data to combined file
            for key in d:
                if key in fout:
                    dset = fout[key]
                    dset_len = len(fout[key])
                    s = dset.shape
                    dset.resize((s[0]+n_good,)+s[1:])
                    dset[s[0]:] = d[key][:]
                else:
                    s = d[key].shape
                    fout.create_dataset(
                        key, data=d[key],
                        maxshape=(None,)+s[1:],
                        chunks=(chunk_size,)+s[1:],
                        compression='lzf'
                    )

        # Store the wavelengths at which the spectrum is sampled, in nm
        sample_wavelengths_p = np.hstack([
            sample_wavelengths,
            1e9*tmass_wl,
            1e9*unwise_wl
        ])
        fout['flux'].attrs['sample_wavelengths'] = sample_wavelengths_p


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
        
        return weights_per_star.astype('float32')


def train(stage=0):
    '''
    Stage 0: Train the stellar model, using universal extinction curve.
    Stage 1: Train the extinction model with hqlE stars, using initial guess of 
                    the slope of extinction curve.
    Stage 2: Train both extinction model and stellar model with hqlE stars
    Stage 3: Self-cleaning & Further optimize the model
    '''
    # General training parameters
    n_epochs = 128
    batch_size = 512
    n_bins = 100
    
    loss_hist=[]    
    
    if stage==0:
        # Stage 0, begin without initial stellar model
        data_fname = 'data/training/xp_nn_training_data.h5'
        print(f'Loading training data from {data_fname} ...')
        d_train, d_val, sample_wavelengths = load_data(data_fname)
        print(f'Loaded {len(d_train["plx"])} sources.')

        # Initial guess of xi
        d_train['xi'] = np.zeros(len(d_train["plx"]), dtype='float32')
        d_val['xi'] = np.zeros(len(d_val["plx"]), dtype='float32')

        save_as_h5('d_val', d_val)
        save_as_h5('data/d_train', d_train)
        
        # Initial weight: equal for all stars
        weights_per_star = np.ones(len(d_train["plx"]), dtype='float32')
        
        # Initialize the parameter estimates at their measured (input) values
        for key in ('stellar_type', 'xi','stellar_ext', 'plx'):
            d_train[f'{key}_est'] = d_train[key].copy()
            d_val[f'{key}_est'] = d_train[key].copy()
        
        print('Creating flux model ...')
        n_stellar_params = d_train['stellar_type'].shape[1]
        p_low,p_high = np.percentile(d_train['stellar_type'], [16.,84.], axis=0)

        # Remove infrared constraints if norm_dg is bad
        good_norm_dg = d_train['norm_dg']<-10.
        d_train['flux_sqrticov'][~good_norm_dg, -5:, -5:] *= 0.
        
        print('Training flux model on high-quality data ...')
        # Select a subset of "high-quality" stars with good measurements
        idx_hq = np.where(
            (d_train['plx']/d_train['plx_err'] > 10.)
          & (d_train['stellar_type_err'][:,0] < 0.2) # 0.2 kiloKelvin = 200 K
          & (d_train['stellar_type_err'][:,1] < 0.2) # 0.2 dex in logg
          & (d_train['stellar_type_err'][:,2] < 0.2) # 0.2 dex in [Fe/H]
          & (d_train['stellar_ext_err'] < 0.1)
        )[0]
        pct_hq = len(idx_hq) / d_train['plx'].shape[0] * 100
        print(f'Training on {len(idx_hq)} ({pct_hq:.3f}%) high-quality stars ...')
        
        stellar_model = FluxModel(
            sample_wavelengths, n_input=n_stellar_params,
            input_zp=np.median(d_train['stellar_type'],axis=0),
            input_scale=0.5*(p_high-p_low),
            hidden_size=32,
            l2=1., l2_ext_curve=1.
         )   
    
        # First, train the model with stars with good measurements,
        # with fixed slope of ext_curve
        ret = train_stellar_model(
            stellar_model,
            d_train, d_val,
            weights_per_star,
            idx_train=idx_hq,
            optimize_stellar_model=True,
            optimize_stellar_params=False,
            batch_size=batch_size,
            n_epochs=n_epochs,
            model_update=['stellar_model',  'ext_curve_b'],
        )
        loss_hist.append(ret)
        stellar_model.save('models/flux/xp_spectrum_model_initial')

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
            model_update=['stellar_model',  'ext_curve_b'],
            var_update = ['atm','E','plx'],
        )
        loss_hist.append(ret)
        #plot_loss(ret, suffix='_intermediate')
        stellar_model.save('models/flux/xp_spectrum_model_intermediate')


        stellar_model = FluxModel.load('models/flux/xp_spectrum_model_intermediate-1')
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
        pct_good = np.count_nonzero(idx_params_good) / idx_params_good.size * 100
        print(f'Parameter outliers: {100-pct_good:.3f}% of sources.')

        idx_flux_good = identify_flux_outliers(
            d_train, stellar_model,
            chi2_dof_clip=5.,
            #chi_indiv_clip=20.
        )
        
        pct_good = np.count_nonzero(idx_flux_good) / idx_flux_good.size * 100
        print(f'Flux outliers: {100-pct_good:.3f}% of sources.')

        idx_good = idx_params_good & idx_flux_good
        pct_good = np.count_nonzero(idx_good) / idx_good.size * 100
        print(f'Combined outliers: {100-pct_good:.3f}% of sources.')

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
            model_update=['stellar_model',  'ext_curve_b'],
            var_update = ['atm','E','plx'],
        )
        loss_hist.append(ret)
        
        # Calculate hq standard again, with a higher limit on chi2
        idx_flux_good = identify_flux_outliers(
            d_train, stellar_model,
            chi2_dof_clip=10.,
            #chi_indiv_clip=20.
        )
        
        pct_good = np.count_nonzero(idx_flux_good) / idx_flux_good.size * 100
        print(f'Flux outliers: {100-pct_good:.3f}% of sources.')

        idx_good = idx_params_good & idx_flux_good
        pct_good = np.count_nonzero(idx_good) / idx_good.size * 100
        print(f'Combined outliers: {100-pct_good:.3f}% of sources.')
        
        np.save('index/idx_good_wo_Rv.npy', idx_good)
        stellar_model.save('models/flux/xp_spectrum_model_final')
        save_as_h5(d_train, 'data/dtrain_final_wo_Rv.h5')
        save_as_h5(ret, 'hist_loss/final_wo_Rv.h5')         

    if stage<2:
         
        d_val = load_h5('d_val.h5')
        d_train = load_h5('data/dtrain_final_wo_Rv.h5')      
        
        stellar_model = FluxModel.load('models/flux/xp_spectrum_model_final-1') 
        
        print('Loading Gaussian Mixture Model prior on stellar type ...')
        stellar_type_prior = GaussianMixtureModel.load('models/prior/gmm_prior-1')
        
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
        weights_per_star = np.ones(len(d_train["plx"]), dtype='float32')
        
        idx_hq= np.load('index/idx_good_wo_Rv.npy')

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
            var_update = [ 'atm', 'E', 'plx'],
        )
        save_as_h5(d_train, 'data/dtrain_final_wo_Rv_optimized.h5')
        d_train = load_h5('data/dtrain_final_wo_Rv_optimized.h5')
        
        idx_hq_large_E = idx_hq & (d_train['stellar_ext_est']>0.1)
        print(f'Training on {100*np.where(idx_hq_large_E)[0].shape[0]/len(idx_hq_large_E)}% of sources.')
        
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
            var_update = [ 'E', 'plx', 'xi'],
        )
        
        weights_per_star = down_sample_weighing( 
            d_train['xi_est'][idx_hq_large_E],
            d_train['xi_est'], 
            bin_edges = np.linspace(-1, 1, n_bins + 1)
        )
        
        weights_per_star /= (0.001+np.median(weights_per_star))        
        
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
            var_update = [ 'E', 'plx', 'xi'],
            model_update = ['ext_curve_w'],
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
            var_update = [ 'E', 'plx', 'xi'],
            model_update = ['ext_curve_w', 'ext_curve_b'],
        ) 
        
        
        np.save('index/idx_with_Rv_good.npy' ,idx_hq_large_E)
        # Save initial guess of xi
        save_as_h5(d_train, 'data/dtrain_Rv_initial.h5')
        save_as_h5(ret, 'hist_loss/Rv_initial.h5') 
        stellar_model.save('models/flux/xp_spectrum_model_initial_Rv')
        
    if stage<3:
        
        n_epochs = 256
        
        stellar_model = FluxModel.load('models/flux/xp_spectrum_model_initial_Rv-1')
        d_train = load_h5('data/dtrain_Rv_initial.h5') 
        d_val = load_h5('d_val.h5')        

        idx_hq_large_E = np.load('index/idx_with_Rv_good.npy')
        
        weights_per_star = down_sample_weighing( 
            d_train['xi_est'][idx_hq_large_E],
            d_train['xi_est'], 
            bin_edges = np.linspace(-1, 1, n_bins + 1)
        )
        
        weights_per_star /= (0.001+np.median(weights_per_star))    
        
        ret = train_stellar_model(
            stellar_model,
            d_train, d_val,
            weights_per_star,
            idx_train=np.where(idx_hq_large_E)[0],
            optimize_stellar_model=True,
            optimize_stellar_params=True,
            lr_model_init=1e-4,
            lr_stars_init=1e-4,
            batch_size=batch_size,
            n_epochs=n_epochs,
            var_update = ['atm','E','plx','xi'],
            model_update = ['stellar_model', 'ext_curve_w', 'ext_curve_b'],
        )    
        
        save_as_h5(d_train, 'data/dtrain_Rv_intermediate_0.h5')        
        save_as_h5(ret, 'hist_loss/Rv_intermediate_0.h5')   
        stellar_model.save('models/flux/xp_spectrum_model_intermediate_Rv') 
                
    if stage<4:
        
        stellar_model = FluxModel.load('models/flux/xp_spectrum_model_intermediate_Rv-1')
        
        d_train = load_h5('data/dtrain_Rv_intermediate_0.h5')
        d_val = load_h5('d_val.h5')
        idx_hq_large_E = np.load('index/idx_with_Rv_good.npy')
        
        # Optimize all stellar params, in order to pick up 
        # stars that were rejected due to extinction variation law
        
        n_epochs = 256
        
        weights_per_star = np.ones(len(d_train['plx']),dtype='float32')
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
       
        save_as_h5(d_train, 'data/dtrain_Rv_intermediate_1.h5')
        save_as_h5(ret, 'hist_loss/Rv_intermediate_1.h5')    
        
        d_train = load_h5('data/dtrain_Rv_intermediate_1.h5')
        d_val = load_h5('d_val.h5')  
        
        # remove outliers 
        idx_params_good = identify_outlier_stars(d_train)
        pct_good = np.count_nonzero(idx_params_good) / idx_params_good.size * 100
        print(f'Parameter outliers: {100-pct_good:.3f}% of sources.')

        idx_flux_good = identify_flux_outliers(
            d_train, stellar_model,
            chi2_dof_clip=5.,
            #chi_indiv_clip=20.
        )
        pct_good = np.count_nonzero(idx_flux_good) / idx_flux_good.size * 100
        print(f'Flux outliers: {100-pct_good:.3f}% of sources.')

        idx_good = idx_params_good & idx_flux_good
        pct_good = np.count_nonzero(idx_good) / idx_good.size * 100
        print(f'Combined outliers: {100-pct_good:.3f}% of sources.')        
                
        idx_final_train  =  (d_train['stellar_ext_est']>0.1)& idx_good
        print(f'Training on {100*np.where(idx_final_train)[0].shape[0]/len(idx_final_train)}% of sources.')
        
        n_epochs = 512
        weights_per_star = down_sample_weighing( 
            d_train['xi_est'][idx_final_train],
            d_train['xi_est'], 
            bin_edges = np.linspace(-1, 1, n_bins + 1)
        )
        weights_per_star /= (0.001+np.median(weights_per_star))    

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
        
        stellar_model.save('models/flux/xp_spectrum_model_final_Rv')
        save_as_h5(ret, 'hist_loss/final_Rv.h5')    

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
            d_val[f'{key}_est'] = d_train[key].copy()
        
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
        
        save_as_h5(d_train,'data/dtrain_Rv_final.h5')
        save_as_h5(d_val,'data/dval_Rv_final.h5')
        
        return 0
        
        
if __name__=='__main__':
    prepare_training_data()
    train(stage=0)    
