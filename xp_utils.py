#!/usr/bin/env python

from __future__ import print_function, division

import numpy as np
import gaiaxpy as gy


class XPSampler(object):
    def __init__(self, wavelengths, flux_scale=1.):
        self.wavelengths = wavelengths.copy()
        self._xp_dm, self._xp_merge = gy.calibrator.calibrator._generate_xp_matrices_and_merge(
            'calibrator', self.wavelengths, 'v375wi', 'v142r'
        )
        self._dm_bp = self._xp_dm['bp']._get_design_matrix() / flux_scale
        self._dm_rp = self._xp_dm['rp']._get_design_matrix() / flux_scale

    def sample(self, bp_coefficients, rp_coefficients,
                     bp_n_parameters, rp_n_parameters,
                     bp_coefficient_errors, rp_coefficient_errors,
                     bp_coefficient_correlations, rp_coefficient_correlations,
                     truncation=None, diag_coefcov=False,
                     bp_zp_errfrac=None, rp_zp_errfrac=None,
                     zp_errfrac=None, diag_errfrac=None,
                     sample_coeffs=False, rng=None):
        """
        Returns a sampled BP/RP spectrum, as well as a corresponding
        covariance matrix.

        Inputs:
            bp_coefficients
            rp_coefficients
            bp_n_parameters
            rp_n_parameters
            bp_coefficient_errors
            rp_coefficient_errors
            bp_coefficient_correlations
            rp_coefficient_correlations
            truncation

        Outputs:
            flux
            flux_error
            pixcov
            cov_bp
            cov_rp
        """
        # Truncate the coefficients at the requested order in the
        # basis expansion
        coef_bp = bp_coefficients[:truncation]
        coef_rp = rp_coefficients[:truncation]

        # Generate the BP and RP covariance matrices
        corr_bp = gy.core.generic_functions.array_to_symmetric_matrix(
            bp_coefficient_correlations,
            bp_n_parameters
        )
        cov_bp = gy.spectrum.utils._correlation_to_covariance_dr3int5(
            corr_bp,
            bp_coefficient_errors,
            1.
        )
        corr_rp = gy.core.generic_functions.array_to_symmetric_matrix(
            rp_coefficient_correlations,
            rp_n_parameters
        )
        cov_rp = gy.spectrum.utils._correlation_to_covariance_dr3int5(
            corr_rp,
            rp_coefficient_errors,
            1.
        )

        # Truncate the design and covariance matrices at the
        # requested order in the basis expansion
        dm_rp = self._dm_rp[:truncation]
        dm_bp = self._dm_bp[:truncation]

        cov_bp = cov_bp[:truncation, :truncation]
        cov_rp = cov_rp[:truncation, :truncation]

        # Select only diagonal part of the covariance matrix
        # (in the coefficient space)?
        if diag_coefcov:
            offdiag = ~np.eye(cov_bp.shape[0], dtype=bool)
            cov_bp[offdiag] = 0
            cov_rp[offdiag] = 0

        # Sample from BP/RP coefficients?
        if sample_coeffs:
            coef_bp = rng.multivariate_normal(coef_bp, cov_bp)
            coef_rp = rng.multivariate_normal(coef_rp, cov_rp)

        # Compute the BP and RP flux and covariance matrices in pixel space
        flux_bp = coef_bp.dot(dm_bp)
        flux_rp = coef_rp.dot(dm_rp)

        pixcov_bp = dm_bp.T.dot(cov_bp).dot(dm_bp)
        pixcov_rp = dm_rp.T.dot(cov_rp).dot(dm_rp)

        # Add in term for overall zero-point offset in BP (fraction of flux)
        if bp_zp_errfrac is not None:
            pixcov_bp += bp_zp_errfrac**2 * np.outer(flux_bp,flux_bp)

        # Add in term for overall zero-point offset in RP (fraction of flux)
        if rp_zp_errfrac is not None:
            pixcov_rp += rp_zp_errfrac**2 * np.outer(flux_rp,flux_rp)

        # Merge the BP and RP fluxes and covariance matrices
        flux = flux_bp * self._xp_merge['bp'] + flux_rp * self._xp_merge['rp']

        pixcov = (
            pixcov_bp*self._xp_merge['bp'][:,None]*self._xp_merge['bp'][None,:]
          + pixcov_rp*self._xp_merge['rp'][:,None]*self._xp_merge['rp'][:,None]
        )

        # Add in term for overall zero-point offset, in BP and RP jointly
        # (again, as a fraction of flux)
        if zp_errfrac is not None:
            pixcov += zp_errfrac**2 * np.outer(flux,flux)

        # Add in term to diagonal (fraction of flux)
        if diag_errfrac is not None:
            pixcov[np.diag_indices_from(pixcov)] += (diag_errfrac*flux)**2

        # Hack to ensure that pixcov is always symmetric - this shouldn't
        # be necessary!
        pixcov = 0.5 * (pixcov + pixcov.T)

        # Also return the uncertainty in each individual pixel
        flux_error = np.sqrt(np.diag(pixcov))

        return flux, flux_error, pixcov, cov_bp, cov_rp


def sqrt_icov_eigen(cov, eival_floor=None, condition_max=None,
                         return_sqrtcov=False):
    eival,eivec = np.linalg.eigh(cov)

    eival_min, eival_max = eival[0], eival[-1]

    if condition_max is not None:
        eival0 = eival_max / condition_max
        eival[eival < eival0] = eival0

    if eival_floor is not None:
        eival[eival < eival_floor] = eival_floor

    sqrt_icov = eivec / np.sqrt(eival)

    if return_sqrtcov:
        sqrt_cov = eivec * np.sqrt(eival)
        return sqrt_icov, sqrt_cov, (eival_min, eival_max)

    return sqrt_icov, (eival_min, eival_max)


def calc_invs_eigen(m, eival_floor=None, condition_max=None,
                       return_min_eivals=False):
    eival,eivec = np.linalg.eigh(m)

    if return_min_eivals:
        eival_min = eival[:,0].copy()

    if condition_max is not None:
        eival_max = eival[:,-1]
        eival0 = eival_max / condition_max
        eival = np.maximum(eival, eival0[:,None])

    if eival_floor is not None:
        eival[eival < eival_floor] = eival_floor

    m_inv = np.einsum('nik,nk,njk->nij', eivec, 1/eival, eivec, optimize=True)

    if return_min_eivals:
        return m_inv, eival_min

    return m_inv


def decompose_symm_matrix(A):
    '''
    Decomposes a symmetric matrix into two flat arrays, containing its
    diagonal and the upper triangle (without the diagonal).
    '''
    n = A.shape[0]
    diag = np.diag(A)
    upper_triangle_wo_diag = A[np.triu_indices(n,k=1)]
    return diag, upper_triangle_wo_diag


def reconstruct_symm_matrix(diag, upper_triangle_wo_diag):
    '''
    Returns a symmetric matrix, reconstructed from two flat arrays,
    containing its diagonal and the upper triangle (without the diagonal).
    '''
    n = diag.size
    A = np.zeros(A)
    A[np.triu_indices(n,k=1)] = upper_triangle_wo_diag
    A += A.T
    A[np.diag_indices(n)] = diag
    return A


def main():
    import matplotlib.pyplot as plt
    from tqdm import tqdm
    import h5py

    #
    # Test that chi^2 is correctly calculated for simple covariance matrices
    #
    n_ex = 1024
    n_vec_per_ex = 1024
    n_dim = 16
    rng = np.random.default_rng()
    for i in tqdm(range(n_ex)):
        A = rng.normal(size=(int(1.5*n_dim),n_dim,n_dim))
        A /= np.sqrt(A.shape[0])
        cov = np.einsum('ijk,ilk->jl', A, A)
        U,(eival_min,eival_max) = sqrt_icov_eigen(cov)

        dx = rng.normal(size=(n_vec_per_ex,n_dim))
        UTdx = np.einsum('ij,nj->ni', U.T, dx)
        chi2_U = np.sum(UTdx*UTdx, axis=1)

        icovdx = np.einsum('ij,nj->ni', np.linalg.inv(cov), dx)
        chi2_direct = np.sum(dx*icovdx, axis=1)

        np.testing.assert_allclose(chi2_U, chi2_direct, rtol=1e-7, atol=1e-7)

    #
    # Apply to real XP spectra
    #

    spec_dir = 'data/xp_continuous_mean_spectrum'
    fname = 'XpContinuousMeanSpectrum_000000-003111.h5'
    n_max = 1024*32

    xp_data = {}
    with h5py.File(os.path.join(spec_dir,fname), 'r') as f:
        for key in f.keys():
            xp_data[key] = f[key][:n_max]

    #wl = np.arange(324., 1049., 8.)
    #wl = np.linspace(400., 980., 70)
    #wl = np.arange(380., 985., 8.)
    wl = np.arange(392., 993., 10.)
    #wl = np.arange(392., 993., 12.)
    #wl = np.hstack([np.arange(392., 625., 8.), np.arange(656., 993., 8.)])
    print(f'{len(wl)} wavelengths.')
    xp_sampler = XPSampler(wl, flux_scale=1e-18)

    #
    # Properties of merge
    #

    fig,ax = plt.subplots(1,1, figsize=(6,4))
    ax.scatter(wl, xp_sampler._xp_merge['bp'], c='b', s=16, edgecolors='none')
    ax.scatter(wl, xp_sampler._xp_merge['rp'], c='r', s=16, edgecolors='none')
    ax.set_xlabel(r'$\lambda\ \left(\mathrm{nm}\right)$')
    ax.set_ylabel(r'$\texttt{xp\_merge}$')
    fig.savefig(f'plots/interpolated_merge.png', dpi=150)
    plt.close(fig)

    #
    # Sample from coefficients and check distribution of chi^2
    #
    # TODO

    #
    # Properties of eigenvalues
    #

    n_sources = xp_data['bp_coefficients'].shape[0]

    eival_min, eival_max = [], []
    flux_tot = []

    for i in tqdm(range(n_sources)):
        flux, flux_err, pixcov, cov_bp, cov_rp = xp_sampler.sample(
            xp_data['bp_coefficients'][i],
            xp_data['rp_coefficients'][i],
            xp_data['bp_n_parameters'][i],
            xp_data['rp_n_parameters'][i],
            xp_data['bp_coefficient_errors'][i],
            xp_data['rp_coefficient_errors'][i],
            xp_data['bp_coefficient_correlations'][i],
            xp_data['rp_coefficient_correlations'][i],
            #bp_zp_errfrac=0.01,
            #rp_zp_errfrac=0.01,
            zp_errfrac=0.01,
            diag_errfrac=0.01,
        )
        U,(val_min,val_max) = sqrt_icov_eigen(pixcov)
        eival_min.append(val_min)
        eival_max.append(val_max)
        flux_tot.append(np.sum(flux))

    eival_min = np.array(eival_min)
    eival_max = np.array(eival_max)
    flux_tot = np.array(flux_tot)

    eival_neg_frac = np.count_nonzero(eival_min<0) / eival_min.size
    print(f'{eival_neg_frac*100:.5f}% of sources have negative eigenvalues.')

    fig,(ax1,ax2,ax3) = plt.subplots(1,3, figsize=(9,3))
    ax1.hist(np.log10(eival_min), bins=50)
    ax1.set_xlabel(r'$\log_{10} \lambda_{\mathrm{min}}$')
    condition_number = eival_max / eival_min
    ax2.hist(np.log10(condition_number), bins=50)
    ax2.set_xlabel(
        r'$\log_{10}\left(\lambda_{\mathrm{max}}/\lambda_{\mathrm{min}}\right)$'
    )
    ax3.scatter(
        flux_tot,
        condition_number,
        s=4,
        edgecolors='none',
        alpha=0.05
    )
    ax3.set_xscale('log')
    ax3.set_yscale('log')
    ax3.set_xlabel(r'$\int f_{\lambda}\left(\lambda\right) \mathrm{d}\lambda$')
    ax3.set_ylabel(
        r'$\log_{10}\left(\lambda_{\mathrm{max}}/\lambda_{\mathrm{min}}\right)$'
    )
    fig.subplots_adjust(
        bottom=0.18,
        top=0.94,
        left=0.05,
        right=0.97,
        wspace=0.26
    )
    fig.savefig(f'plots/interpolated_eigenval_hist_{len(wl)}wl.png', dpi=150)
    plt.close(fig)

    #
    # More detailed plots on small subset of sources
    #

    n_plot = min(n_sources, 32)

    for i in tqdm(range(n_plot)):
        flux, flux_err, pixcov, cov_bp, cov_rp = xp_sampler.sample(
            xp_data['bp_coefficients'][i],
            xp_data['rp_coefficients'][i],
            xp_data['bp_n_parameters'][i],
            xp_data['rp_n_parameters'][i],
            xp_data['bp_coefficient_errors'][i],
            xp_data['rp_coefficient_errors'][i],
            xp_data['bp_coefficient_correlations'][i],
            xp_data['rp_coefficient_correlations'][i],
            #bp_zp_errfrac=0.01,
            #rp_zp_errfrac=0.01,
            zp_errfrac=0.01,
            diag_errfrac=0.01,
        )

        fig,ax = plt.subplots(1,1, figsize=(6,4))
        ax.fill_between(wl, flux-flux_err, flux+flux_err, alpha=0.2)
        ax.plot(wl, flux)
        ax.set_xlabel(r'$\lambda\ \left(\mathrm{nm}\right)$')
        ax.set_ylabel(r'$f \left(\lambda\right)$')
        fig.savefig(f'plots/interpolated_flux_{i:05d}.png', dpi=150)
        plt.close(fig)

        fig,ax = plt.subplots(1,1, figsize=(7,6))
        img = pixcov
        vmax = np.max(np.abs(img))
        im = ax.matshow(img, cmap='coolwarm_r', vmin=-vmax, vmax=vmax)
        fig.colorbar(im, ax=ax, label=r'$C$')
        fig.savefig(f'plots/interpolated_flux_cov_{i:05d}.png', dpi=150)
        plt.close(fig)

        fig,ax = plt.subplots(1,1, figsize=(7,6))
        pixcorr = pixcov[:,:] / (flux_err[:,None]*flux_err[None,:])
        im = ax.matshow(pixcorr, cmap='coolwarm_r', vmin=-1., vmax=1.)
        fig.colorbar(im, ax=ax, label='correlation')
        fig.savefig(f'plots/interpolated_flux_correlation_{i:05d}.png', dpi=150)
        plt.close(fig)

        fig,ax = plt.subplots(1,1, figsize=(7,6))
        icov = np.linalg.inv(pixcov)
        img = icov
        vmax = np.max(np.abs(img))
        im = ax.matshow(img, cmap='coolwarm_r', vmin=-vmax, vmax=vmax)
        fig.colorbar(im, ax=ax, label=r'$C^{-1}$')
        fig.savefig(f'plots/interpolated_flux_icov_{i:05d}.png', dpi=150)
        plt.close(fig)

        fig,ax = plt.subplots(1,1, figsize=(7,6))
        img = 2 * (pixcov-pixcov.T) / (pixcov+pixcov.T)
        vmax = np.nanmax(np.abs(img))
        im = ax.matshow(img, cmap='coolwarm_r', vmin=-vmax, vmax=vmax)
        fig.colorbar(im, ax=ax, label=r'$2 (C-C^T) / (C+C^T)$')
        fig.savefig(f'plots/interpolated_flux_cov_asymmetry_frac_{i:05d}.png', dpi=150)
        plt.close(fig)

        fig,ax = plt.subplots(1,1, figsize=(7,6))
        img = (pixcov-pixcov.T)
        vmax = np.nanmax(np.abs(img))
        im = ax.matshow(img, cmap='coolwarm_r', vmin=-vmax, vmax=vmax)
        fig.colorbar(im, ax=ax, label=r'$C-C^T$')
        fig.savefig(f'plots/interpolated_flux_cov_asymmetry_{i:05d}.png', dpi=150)
        plt.close(fig)

        fig,ax = plt.subplots(1,1, figsize=(7,6))
        eival,eivec = np.linalg.eigh(pixcov)
        #U = eivec / np.sqrt(eival)
        #print(eival)
        U, (eival_min, eival_max) = sqrt_icov_eigen(pixcov, eival_floor=1e-4)
        img = U
        vmax = np.nanmax(np.abs(img))
        im = ax.matshow(img, cmap='coolwarm_r', vmin=-vmax, vmax=vmax)
        fig.colorbar(im, ax=ax, label=r'$Q \Lambda^{-\frac{1}{2}}$')
        fig.savefig(f'plots/interpolated_flux_cov_decomp_{i:05d}.png', dpi=150)
        plt.close(fig)

        fig,ax = plt.subplots(1,1, figsize=(7,6))
        img = eivec
        vmax = np.nanmax(np.abs(img))
        im = ax.matshow(img, cmap='coolwarm_r', vmin=-vmax, vmax=vmax)
        fig.colorbar(im, ax=ax, label=r'$Q$')
        fig.savefig(f'plots/interpolated_flux_cov_eigenvecs_{i:05d}.png', dpi=150)
        plt.close(fig)

        n_neg_eivals = np.count_nonzero(eival<0)
        condition_number = eival[-1] / eival[0]
        print(f'Smallest eigenvalue: {eival[0]:.7g}')
        print(f'# of negative eigenvalues: {n_neg_eivals}')
        print(f'Condition number: {condition_number:.7g}')

        fig,ax = plt.subplots(1,1, figsize=(6,4))
        ax.semilogy(eival)
        #ax.set_yscale('log')
        ax.set_xlabel('eigenvalue order')
        ax.set_ylabel('eigenvalue')
        ax.set_title(f'condition number = {condition_number:.5g}')
        fig.savefig(f'plots/interpolated_eigenspectrum_{i:05d}.png', dpi=150)
        plt.close(fig)

    return 0


if __name__ == '__main__':
    main()

