#!/usr/bin/env python

from __future__ import print_function, division

import numpy as np

import matplotlib.pyplot as plt
import matplotlib.patheffects as patheffects
import matplotlib.colors as colors

from astropy.wcs import WCS
from astropy.visualization.wcsaxes.frame import EllipticalFrame
from reproject import reproject_from_healpix

from astropy_healpix import HEALPix


def mollweide_wcs(w, frame):
    coordsys,ctype0,ctype1 = {
        'galactic': ('GAL', 'GLON-MOL', 'GLAT-MOL'),
        'icrs':     ('EQU', 'RA---MOL', 'DEC--MOL')
    }[frame]
    target_header = dict(
        naxis=2,
        naxis1=w,
        naxis2=w//2,
        ctype1=ctype0,
        crpix1=w//2+0.5,
        crval1=0.0,
        cdelt1=-0.675*480/w,
        cunit1='deg',
        ctype2=ctype1,
        crpix2=w//4+0.5,
        crval2=0.0,
        cdelt2=0.675*480/w,
        cunit2='deg',
        coordsys=coordsys
    )
    wcs = WCS(target_header)
    return wcs


def plot_mollweide(healpix_data, *subplot_args, fig=None,
                   w=480, input_frame='galactic', plot_frame='galactic',
                   grid_kw=dict(), **kwargs):
    wcs = mollweide_wcs(w, plot_frame)
    print(wcs)

    array, footprint = reproject_from_healpix(
        (healpix_data, input_frame),
        wcs, nested=True,
        shape_out=(w//2,w),
        order='nearest-neighbor'
    )

    if fig is None:
        figsize = (6, 2.6)
        fig = plt.figure(figsize=figsize, layout='tight')
    
    ax = fig.add_subplot(
        *subplot_args, projection=wcs,
        frame_class=EllipticalFrame
    )
    im = ax.imshow(array, **kwargs)

    pe = [
        patheffects.Stroke(linewidth=1.0, foreground='white', alpha=0.7),
        patheffects.Normal()
    ]

    coord0,coord1 = {'galactic':('glon','glat'), 'icrs':('ra','dec')}[plot_frame]
    kw = dict(color='gray', alpha=0.2)
    kw.update(**grid_kw)
    ax.coords.grid(**kw)
    ax.coords[coord0].set_ticklabel(color='k', path_effects=pe, fontsize=7)
    ax.coords[coord1].set_ticklabel(color='k', path_effects=pe, fontsize=7)
    ax.coords[coord1].set_ticklabel_position('v')
    ax.coords[coord0].set_ticks_visible(False)
    ax.coords[coord1].set_ticks_visible(False)

    return fig, ax, im


def tangent_wcs(center, fov, w, frame, rotation=0):
    # Determine height (in pixels) of image, based on width and fov ratio
    tan_th0 = np.tan(np.radians(fov[0]/2))
    tan_th1 = np.tan(np.radians(fov[1]/2))
    h = int(np.round(w * tan_th1 / tan_th0))

    # Coordinate frame
    coordsys,ctype0,ctype1 = {
        'galactic': ('GAL', 'GLON-TAN', 'GLAT-TAN'),
        'icrs':     ('EQU', 'RA---TAN', 'DEC--TAN')
    }[frame]

    # Rotation matrix
    th = np.radians(rotation)
    pc = [[np.cos(th), np.sin(th)],
          [-np.sin(th), np.cos(th)]]
    
    target_header = dict(
        naxis=2,
        naxis1=w,
        naxis2=h,
        ctype1=ctype0,
        crpix1=w//2+0.5,
        crval1=center[0],
        cdelt1=-np.degrees(tan_th0)/(0.5*w),
        cunit1='deg',
        ctype2=ctype1,
        crpix2=h//2+0.5,
        crval2=center[1],
        cdelt2=np.degrees(tan_th1)/(0.5*h),
        cunit2='deg',
        coordsys=coordsys,
        pc1_1=pc[0][0],
        pc1_2=pc[0][1],
        pc2_1=pc[1][0],
        pc2_2=pc[1][1]
    )
    wcs = WCS(target_header)
    return wcs


def plot_tangent(healpix_data, *subplot_args, fig=None,
                 center=(0.,0.), fov=(15.,15.), w=480,
                 input_frame='galactic', plot_frame='galactic',
                 grid_kw=dict(), **kwargs):
    wcs = tangent_wcs(center, fov, w, plot_frame)
    print(wcs)

    array, footprint = reproject_from_healpix(
        (healpix_data, input_frame),
        wcs, nested=True,
        shape_out=wcs.pixel_shape,
        order='nearest-neighbor'
    )

    if fig is None:
        figsize = (6, 2.6)
        fig = plt.figure(figsize=figsize, layout='tight')
    
    ax = fig.add_subplot(*subplot_args, projection=wcs)
    im = ax.imshow(array, **kwargs)

    pe = [
        patheffects.Stroke(linewidth=1.0, foreground='white', alpha=0.7),
        patheffects.Normal()
    ]

    # coord0,coord1 = {'galactic':('glon','glat'), 'icrs':('ra','dec')}[plot_frame]
    kw = dict(color='gray', alpha=0.2)
    kw.update(**grid_kw)
    ax.coords.grid(**kw)
    # ax.coords[coord0].set_ticklabel(color='k', path_effects=pe, fontsize=7)
    # ax.coords[coord1].set_ticklabel(color='k', path_effects=pe, fontsize=7)
    # ax.coords[coord1].set_ticklabel_position('v')
    # ax.coords[coord0].set_ticks_visible(False)
    # ax.coords[coord1].set_ticks_visible(False)

    return fig, ax, im


def wcs_project_points(wcs, coord0, coord1):
    img_pix_idx = wcs.all_world2pix(coord0, coord1, 0)
    for idx in img_pix_idx:
        idx[~np.isfinite(idx)] = -1

    idx_oob = (
        (img_pix_idx[0] < 0)
      | (img_pix_idx[1] < 0)
      | (img_pix_idx[0] >= wcs.pixel_shape[0])
      | (img_pix_idx[1] >= wcs.pixel_shape[1])
    )

    idx_proj = np.where(~idx_oob)[0]
    xy_proj = [i[idx_proj] for i in img_pix_idx]

    return xy_proj, idx_proj


def get_wcs_coordimage(wcs):
    idx_y,idx_x = np.indices(wcs.array_shape)
    return wcs.pixel_to_world(idx_x, idx_y)


def healpix_mean_map(lon, lat, data, nside, weights=None):
    hp = HEALPix(nside=nside, order='nested')
    hpix_idx = hp.lonlat_to_healpix(lon, lat)

    sum_data = np.zeros(hp.npix, dtype='f8')

    if weights is None:
        np.add.at(sum_data, hpix_idx, data)
        sum_weight = np.zeros(hp.npix, dtype='u8')
        np.add.at(sum_weight, hpix_idx, 1)
    else:
        np.add.at(sum_data, hpix_idx, data*weights)
        sum_weight = np.zeros(hp.npix, dtype='f8')
        np.add.at(sum_weight, hpix_idx, weights)

    sum_data /= sum_weight

    return sum_data


def plot_corr(ax, x, y, x_lim=None, d_max=None,
                        diff=False, bins=(50,31),
                        pct=(16,50,84),
                        cmap='binary',
                        envelope_kw={}):
    idx = np.isfinite(x) & np.isfinite(y)
    x = x[idx]
    y = y[idx]

    if x_lim is None:
        x_min, x_max = np.min(x), np.max(x)
        # w = x_max - x_min
        xlim = (x_min, x_max)
    else:
        xlim = x_lim

    if diff:
        d = y - x
    else:
        d = y

    if d_max is None:
        dmax = 1.2 * np.percentile(np.abs(d), 99.9)
    else:
        dmax = d_max
    dlim = (-dmax, dmax)

    n,x_edges,y_edges = np.histogram2d(x, d, range=(xlim, dlim), bins=bins)

    norm = np.max(n, axis=1) + 1.e-10
    n /= norm[:,None]

    extent = (x_edges[0], x_edges[-1], y_edges[0], y_edges[-1])
    ax.imshow(
        n.T,
        origin='lower',
        interpolation='nearest',
        aspect='auto',
        extent=extent,#tuple(xlim)+tuple(dlim),
        cmap=cmap
    )
    ax.axhline(0., c='w', alpha=0.2, lw=1)

    if len(pct):
        x_pct = np.empty((3, len(x_edges)-1))
        for i,(x0,x1) in enumerate(zip(x_edges[:-1],x_edges[1:])):
            idx = (x > x0) & (x < x1)
            if np.any(idx):
                x_pct[:,i] = np.percentile(d[idx], pct)
            else:
                x_pct[:,i] = np.nan

        for i,x_env in enumerate(x_pct):
            step_kw = dict(c='cyan', alpha=0.5)
            step_kw.update(envelope_kw)
            ax.step(
                x_edges,
                np.hstack([x_env[0], x_env]),
                **step_kw
            )

    ax.set_xlim(x_edges[0], x_edges[-1])
    ax.set_ylim(y_edges[0], y_edges[-1])
    
    return x_pct

def choose_lim(x, pct=[1., 99.], expand=0.2):
    x0,x1 = np.percentile(x[np.isfinite(x)], pct)
    w = x1 - x0
    x0 -= expand * w
    x1 += expand * w
    return x0, x1


def projection_grid(data, c=None, labels=None,
                          extents=None, scatter=True, clabel=None,
                          hist_kw={}, fig_kw={}, **kwargs):
    _,n_dim = data.shape
    n_col = n_dim - 1

    if c is not None:
        assert len(c) == data.shape[0]

    if extents is None:
        # Automatically choose axis limits
        extents = [choose_lim(x) for x in data.T]

    kw = {'figsize': (2*n_col,2*n_col)}
    kw.update(**fig_kw)

    fig,ax_arr = plt.subplots(
        nrows=n_col,
        ncols=n_col,
        sharex='col',
        sharey='row',
        **kw
    )
    if not isinstance(ax_arr,np.ndarray):
        ax_arr = np.array([[ax_arr]])

    for col in range(n_col):
        j = col
        for row in range(n_col):
            k = row+1
            ax = ax_arr[row,col]

            if k <= j:
                # Hide unused axes (upper-right triangle)
                ax.axis('off')
                continue

            xlim = extents[j]
            ylim = extents[k]

            x = data[:,j]
            y = data[:,k]

            if scatter:
                # Scatterplot of points
                im = ax.scatter(x, y, c=c, **kwargs)
            else:
                xlim_s = np.sort(xlim).tolist()
                ylim_s = np.sort(ylim).tolist()
                if c is not None:
                    # Reduce c values in each bin
                    from scipy.stats import binned_statistic_2d
                    idx = np.isfinite(x) & np.isfinite(y)
                    img,_,_,_ = binned_statistic_2d(
                        x[idx], y[idx], c[idx],
                        range=[xlim_s,ylim_s],
                        **hist_kw
                    )
                    im = ax.imshow(
                        img.T,
                        extent=xlim_s+ylim_s,
                        origin='lower',
                        aspect='auto',
                        interpolation='nearest',
                        **kwargs
                    )
                else:
                    # Plot density of points
                    _,_,_,im = ax.hist2d(
                        x, y,
                        range=[xlim_s,ylim_s],
                        **hist_kw
                    )

            ax.set_xlim(xlim)
            ax.set_ylim(ylim)

            if labels is not None:
                if row == n_col-1:
                    ax.set_xlabel(labels[j])
                if col == 0:
                    ax.set_ylabel(labels[k])

    if c is not None:
        # Add colorbar on the top right
        cax = fig.add_subplot(2, 12, 12)
        cb = fig.colorbar(im, cax=cax, label=clabel)
        # Remove alpha from colorbar
        cb.set_alpha(1)
        cb.draw_all()
        # Move labels and ticks to the left side of the colorbar
        cax.yaxis.set_ticks_position('left')
        cax.yaxis.set_label_position('left')

    return fig, ax_arr


def hist2d_reduce(x, y, c, xlim=None, ylim=None, ax=None,
                           hist_kw={}, imshow_kw={}):
    from scipy.stats import binned_statistic_2d

    if ax is None:
        ax = plt.gca()

    idx = np.isfinite(x) & np.isfinite(y)

    if xlim is None:
        xlim = choose_lim(x)
    if ylim is None:
        ylim = choose_lim(y)

    xlim_s = np.sort(xlim).tolist()
    ylim_s = np.sort(ylim).tolist()

    kw = dict(range=[xlim_s,ylim_s])
    kw.update(hist_kw)
    img,_,_,_ = binned_statistic_2d(
        x[idx], y[idx], c[idx],
        **kw
    )

    kw = dict(
        extent=xlim_s+ylim_s,
        origin='lower',
        aspect='auto',
        interpolation='nearest'
    )
    kw.update(imshow_kw)
    im = ax.imshow(img.T, **kw)

    ax.set_xlim(xlim)
    ax.set_ylim(ylim)

    return im


def main():
    rng = np.random.default_rng()
    x = rng.normal(size=(1000,3))
    for i in range(x.shape[1]):
        x[:,i] += i
    c = np.arange(x.shape[0])

    labels = [chr(97+i) for i in range(x.shape[1])]

    fig,ax_arr = projection_grid(x, labels=labels, c=c, clabel='value', alpha=0.2)
    fig.savefig('plots/projection_grid_scatter.png', dpi=300)
    plt.close(fig)

    fig,ax_arr = projection_grid(
        x, c=c,
        labels=labels, clabel='value',
        scatter=False
    )
    fig.savefig('plots/projection_grid_hist2d.png', dpi=300)
    plt.close(fig)

    return 0

if __name__ == '__main__':
    main()

