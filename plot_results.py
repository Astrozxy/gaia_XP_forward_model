from xp_neural_network_v3 import *
import matplotlib.colors as colors
from matplotlib.gridspec import GridSpec
from dust_extinction.parameter_averages import G23


def compare_with_Gordan23():
    # Load the latest version of model
    stellar_model = FluxModel.load('models/flux/xp_spectrum_model_final_Rv-1')
    wl = stellar_model.get_sample_wavelengths()
    cmap = cm.coolwarm_r
    norm = Normalize(vmin=-1., vmax=1.)

    ax1 = plt.subplot(121)
    for xi in np.linspace(-1.,1.,10):
        y_g = np.exp(stellar_model.predict_ln_ext_curve([xi.astype('float32')])[0])
        ax1.loglog(wl[:-5] ,y_g[:-5],linewidth=1,color=cmap(norm(xi)))
        ax1.scatter(wl[-5:] ,y_g[-5:],color=cmap(norm(xi)) , s=2)
    cmap = cm.coolwarm_r
    norm = Normalize(vmin=2.2, vmax=4.3)
    ax2 = plt.subplot(122)  
    for Rv in np.arange(2.3,4.3,0.25):
        y = G23(Rv)(1000/wl.astype('float64'))* 2.5/np.log(10)
        l = ax2.loglog(wl[:-5],y[:-5],color=cmap(norm(Rv)))
        ax2.scatter(wl[-5:] ,y[-5:],color=cmap(norm(Rv)) , s=2)

    ylim0 = np.min([ax1.get_ylim()[0], ax2.get_ylim()[0]])
    ylim1 = np.max([ax1.get_ylim()[1], ax2.get_ylim()[1]])

    ax1.set_ylim([ylim0, ylim1])
    ax2.set_ylim([ylim0, ylim1])
    plt.subplots_adjust(wspace=0)
    ax1.grid(True, which='minor',axis='y')
    ax2.grid(True, which='minor',axis='y')
    ax2.set_yticks([])
    ax2.set_yticklabels([], minor=True)
    ax1.tick_params(axis='both',  direction = 'in',width=1,length=4,which='both')
    ax2.tick_params(axis='both',  direction = 'in',width=1,length=4,which='both')
    ax1.set_xticks([400, 1000, 3000])
    ax1.set_xticklabels(['400', '1000', '3000'])

    ax2.set_xticks([400, 1000, 3000])
    ax2.set_xticklabels(['400', '1000', '3000'])

    ax1.set_title('Forward model v2.0')
    ax2.set_title('Gordon 23')

    ax1.set_ylabel(r'$A\left(\lambda\right) / A\left(392 \mathrm{nm}\right)$')
    plt.savefig('plots/ext_curve_vs_G23.pdf')
    plt.close()
    

def plot_stellar_model(n_bands=5, n_val=15, ndim=5, suffix=''):
    stellar_model = FluxModel.load('models/flux/xp_spectrum_model_final_Rv-1')
    sample_wavelengths = stellar_model.get_sample_wavelengths()

    def Ryd_wl(n0, n1):
        return (1 / (const.Ryd * (1/n0**2 - 1/n1**2))).to('nm').value
    
    #'-', '--', '-.', ':', 'None', ' ', '', 'solid', 'dashed', 'dashdot', 'dotted'
    lines = [
        (r'$\mathrm{Balmer\ series}$', [Ryd_wl(2,n) for n in (3,4,5,6)], 'g', ':'),
        (r'$\mathrm{Paschen\ series}$', [Ryd_wl(3,n) for n in (8,9,10)], 'b', '-.'),
        (r'$\mathrm{Mg}$', [518.362], 'orange', '-'),
        (r'$\mathrm{Ca}$', [422.6727, 430.774, 854.2], 'violet', '--'),
        (r'$\mathrm{Fe}$', [431.,438.,527.], 'purple', 'solid')
    ]

    for param in ('teff', 'logg', 'feh', 'dwarfs', 'giants'):
        teff = np.full(n_val, 5500.)
        logg = np.full(n_val, 4.6)
        feh = np.full(n_val, 0.)
        if param == 'dwarfs':
            teff = np.linspace(4000., 11000., n_val)
            logg = np.zeros_like(teff)
            logg[teff<5000] = 4.6
            idx = (teff>=5000) & (teff<6300)
            logg[idx] = 4.6-0.65*(teff[idx]-5000)/1300
            logg[teff>=6300] = 3.95
            label = r'$T_{\mathrm{eff}}$'
            title = (
                r'$'
                r'\mathrm{Main\ Sequence} ,\ '
                rf'\left[\mathrm{{Fe/H}}\right] = {feh[0]:.1f}'
                '$'
            )
            v = teff
        elif param == 'giants':
            logg = np.linspace(4.15, 1.5, n_val)
            teff = np.zeros_like(logg)
            idx = (logg >= 3.65)
            teff[idx] = 5900 - 700*(4.15-logg[idx])/(4.15-3.65)
            teff[~idx] = 5200 - 950*(3.65-logg[~idx])/(3.65-1.5)
            label = r'$\log g$'
            title = (
                r'$'
                r'\mathrm{Giant\ Branch} ,\ '
                rf'\left[\mathrm{{Fe/H}}\right] = {feh[0]:.1f}'
                '$'
            )
            v = logg
            print(teff)
            print(logg)
        elif param == 'teff':
            teff = np.linspace(4000., 8000., n_val)
            label = r'$T_{\mathrm{eff}}$'
            title = (
                r'$'
                rf'\left[\mathrm{{Fe/H}}\right] = {feh[0]:.1f} ,\,'
                rf'\log g = {logg[0]:.1f}'
                '$'
            )
            v = teff
        elif param == 'logg':
            logg = np.linspace(4.3, 4.9, n_val)
            label = r'$\log g$'
            title = (
                r'$'
                rf'T_{{\mathrm{{eff}}}} = {teff[0]:.0f} ,\,'
                rf'\left[\mathrm{{Fe/H}}\right] = {feh[0]:.1f}'
                '$'
            )
            v = logg
        elif param == 'feh':
            feh = np.linspace(-1.0, 0.4, n_val)
            label = r'$\left[\mathrm{Fe/H}\right]$'
            title = (
                r'$'
                rf'T_{{\mathrm{{eff}}}} = {teff[0]:.0f} ,\,'
                rf'\log g = {logg[0]:.1f}'
                '$'
            )
            v = feh

        stellar_params = np.stack([0.001*teff,feh,logg], axis=1)
        flux = np.exp(stellar_model.predict_intrinsic_ln_flux(stellar_params))
        
        import matplotlib.gridspec as gridspec
        fig = plt.figure(figsize=(6,3))
        outer = gridspec.GridSpec(1, 2, width_ratios = [1, 1]) 
        (gs,gs1) = gridspec.GridSpecFromSubplotSpec(1, 2, width_ratios = [3, 1], subplot_spec = outer[0], wspace = 0)
        (gs2,) = gridspec.GridSpecFromSubplotSpec(1, 1, subplot_spec = outer[1])
        ax = plt.subplot(gs)
        ax1 = plt.subplot(gs1)
        ax2 = plt.subplot(gs2)

        if param == 'feh':
            cmap = cm.coolwarm
        else:
            cmap = cm.coolwarm_r
        norm = Normalize(vmin=min(v), vmax=max(v))

        ax_min,ax1_min,ax2_min = 10**20,10**20,10**20
        ax_max,ax1_max,ax2_max = 0,0,0
        
        for vv,fl in zip(v, flux):
            x = (sample_wavelengths[:-n_bands])
            y = fl[:-n_bands]*sample_wavelengths[:-n_bands]**2
            
            ax.loglog(
                x,
                y,
                color=cmap(norm(vv)),
                linewidth=0.5
            )
            
            ax_min = np.min([np.min(y),ax_min])
            ax_max = np.max([np.max(y),ax_max])
            
            x = sample_wavelengths[:-n_bands]
            y = fl[:-n_bands]*sample_wavelengths[:-n_bands]**2
            
            
            coeffs = np.polyfit(x, y, ndim)
            y_fit = np.sum([i*x**(ndim-k) for k,i in enumerate(coeffs)],axis=0)
            
            ax2.loglog(
                x,
                y/y_fit,
                color=cmap(norm(vv)),
                linewidth=0.5
                #linestyle=':'
            )

            ax2_min = np.min([np.min(y/y_fit),ax2_min])
            ax2_max = np.max([np.max(y/y_fit),ax2_max])
            
        for vv,fl in zip(v,flux):
            x = sample_wavelengths[-n_bands:]
            y = fl[-n_bands:]*sample_wavelengths[-n_bands:]**2
            
            ax1.scatter(
                x,
                y,
                color=cmap(norm(vv)),
                edgecolors='none',
                s=4
            )
            ax1_min = np.min([np.min(y),ax1_min])
            ax1_max = np.max([np.max(y),ax1_max])
        

        fig.colorbar(
            cm.ScalarMappable(norm=norm, cmap=cmap),
            ax=ax2,
            label=label
        )

        ax.set_xlabel(r'$\lambda\ \left(\mathrm{nm}\right)$')
        #ax1.set_xlabel(r'$\lambda\ \left(\mathrm{nm}\right)$')
        ax2.set_xlabel(r'$\lambda\ \left(\mathrm{nm}\right)$')
            
        ylabel = (
            r'$\lambda^2 f_{\lambda}\ '
            r'\left(10^{-18}\,\mathrm{W\,m^{-2}\,nm}\right)$'
        )
        ax.set_ylabel(ylabel)

        for line_label,line_wl,c,ls in lines:
            for i,wl in enumerate(line_wl):
                ax.axvline(
                    wl,
                    color=c, ls=ls, lw=1., alpha=0.4,
                    label=line_label if i==0 else None
                )
                ax2.axvline(
                    wl,
                    color=c, ls=ls, lw=1., alpha=0.4,
                    label=line_label if i==0 else None
                )

        legend = ax.legend(fontsize=5,loc=4)
        frame = legend.get_frame() 
        frame.set_alpha(1) 
        frame.set_facecolor('white')

        plt.suptitle(title)

        plt.subplots_adjust(
            left=0.1,
            right=0.90,
            bottom=0.2,
            top=0.88,
            wspace=0.3
        )
        
        for a in (ax,ax1,ax2):
            a.tick_params(
                axis='both',
                direction='in',
                which='both'
            )
            
        ax1.set_xscale('log')
        
        share_min = np.min([ax_min,ax1_min])
        share_max = np.max([ax_max,ax1_max])
            
        ax.set_ylim([0.8*share_min,1.5*share_max])
        ax1.set_xlim([1000,6000])
        ax.set_xlim([380,1000])
        ax1.set_xticks([1000,4000],['1000','4000'])
        ax1.set_ylim([0.8*share_min,1.5*share_max])
        ax2.set_ylim([0.9*ax2_min,1.2*ax2_max])
        
        ax1.set_yscale('log')
        ax1.yaxis.set_ticklabels(ticklabels=[])
         
        ax2.set_ylabel(r'$f_{\lambda}/f_{\mathrm{baseline},\ \lambda}$')
        ax2.set_yticks(np.arange(0.5,2.0,0.2),['%.1f'%i for i in np.arange(0.5,2.0,0.2)])

        ax.set_xticks([400,500,700],['400','500','700'])
        ax2.set_xticks([400,500,600,800,1000],['400','500','600','800','1000'])
        
        for a in (ax,ax1,ax2):
            a.grid(True, which='major', c='k', alpha=0.1)
            #a.grid(True, which='minor', c='k', alpha=0.04)         
            a.minorticks_off()
        
        fig.savefig(f'plots/model_flux_vs_{param}{suffix}.pdf')
        plt.close(fig)
    

def plot_fisher():
    stellar_model = FluxModel.load('models/flux/xp_spectrum_model_final_Rv-1')
    d_train = load_h5('data/dtrain_Rv_final.h5')
    stellar_type_prior = GaussianMixtureModel.load('models/prior/gmm_prior-1')

    calc_stellar_fisher_hessian(
        stellar_model, d_train,
        gmm=stellar_type_prior,
        batch_size=1024
    )
    
    fisher = d_train['fisher'][...].copy()
    fisher_wp = d_train['fisher'][...].copy()
    for k in range(5):
        fisher_wp[:,k,k] = (fisher_wp[:,k,k]+d_train['ivar_priors'][...][:,k])

    fisher = np.array([np.linalg.inv(i) for i in fisher])
    fisher_wp = np.array([np.linalg.inv(i) for i in fisher_wp])

    corr_fisher = []
    corr_fisher_wp = []
    for i in tqdm(range(fisher.shape[0])):
        diag = np.diagonal(fisher[i])**0.5

        corr_fisher.append(fisher[i]/diag/diag.reshape(-1,1))

        diag = np.diagonal(fisher_wp[i])**0.5
        corr_fisher_wp.append(fisher_wp[i]/diag/diag.reshape(-1,1))
        
    cmap = cm.coolwarm_r
    norm = colors.Normalize(vmin=-1, vmax=1)

    fig = plt.figure(figsize=(3,2.6))
    gs = GridSpec(2,2,
                  height_ratios=[1, 0.05]
    )

    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])

    ax1.matshow(np.nanmedian(corr_fisher,axis=0),cmap=cmap,norm=norm)
    ax2.matshow(np.nanmedian(corr_fisher_wp,axis=0),cmap=cmap,norm=norm)

    ax1.set_title(r'$\mathrm{Without}' + "\n" +  r'\mathrm{obs.\ parallax}$', pad=4.0)
    ax2.set_title(r'$\mathrm{With}' + "\n" + r'\mathrm{obs.\ parallax}$', pad=4.0)

    labels = [r'$T_\mathrm{eff}$',r'$\mathrm{[Fe/H]}$',r'$\log g$', r'$\xi$', r'$E$',r'$\varpi$']
    for ax in (ax1,):
        ax.set_xticks([0,1,2,3,4,5], labels, rotation=90)
        ax.set_yticks([0,1,2,3,4,5], labels)
        ax.tick_params(axis='both', direction = 'in', which='both', length=2., width=0.5)

    for ax in (ax2,):
        ax.set_xticks([0,1,2,3,4,5], labels, rotation=90)
        ax.set_yticks([0,1,2,3,4,5], [])
        ax.tick_params(axis='both', direction = 'in', which='both', length=2., width=0.5)

    ax_colorbar = fig.add_subplot(gs[1, :])
    cbar = fig.colorbar(cm.ScalarMappable(norm=norm, cmap=cm.coolwarm_r),
                           orientation='horizontal', cax=ax_colorbar, use_gridspec=True,)
    cbar.ax.tick_params(axis='both',direction = 'in',which='major',length=2., width=0.5)
    cbar.set_label(r'$\mathrm{Correlation}$')
    plt.subplots_adjust(wspace=0.1, top=0.75, right=0.95, left=0.15, hspace=0.0, bottom=0.15)
    plt.savefig('./plots/fisher_vs_hessian.pdf')
    plt.close()
    
    
def make_plots():
    compare_with_Gordan23()
    plot_stellar_model()
    plot_fisher()
    
    
if __name__=='__main__':
    make_plots()
    