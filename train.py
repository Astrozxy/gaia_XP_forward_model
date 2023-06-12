from model import *


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
        data_fname = 'xp_nn_training_data.h5'
        print(f'Loading training data from {data_fname} ...')
        d_train, d_val, sample_wavelengths = load_data(data_fname)
        print(f'Loaded {len(d_train["plx"])} sources.')

        # Initial guess of xi
        d_train['xi'] = np.zeros(len(d_train["plx"]), dtype='float32')
        d_val['xi'] = np.zeros(len(d_val["plx"]), dtype='float32')

        # Initial weight: equal for all stars
        weights_per_star = np.ones(len(d_train["plx"]), dtype='float32')
        
        # Initialize the parameter estimates at their measured (input) values
        for key in ('stellar_type', 'xi','stellar_ext', 'plx'):
            d_train[f'{key}_est'] = d_train[key].copy()
        
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
        
        idx_hq_large_E = idx_hq & (d_train['stellar_ext_est']>0.3)
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
            lr_stars_init=1e-5,
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
            optimize_stellar_params=False,
            lr_model_init=1e-7,
            #lr_stars_init=1e-5,
            batch_size=batch_size,
            n_epochs=n_epochs,
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
            lr_model_init=1e-7,
            lr_stars_init=1e-5,
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
        '''
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
         '''
        
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
                
        idx_final_train  =  (d_train['stellar_ext_est']>0.3)& idx_good
        print(f'Training on {100*np.where(idx_final_train)[0].shape[0]/len(idx_final_train)}% of sources.')
        
        n_epochs = 256
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
    train(stage=0, data_fname = 'xp_nn_training_data.h5')