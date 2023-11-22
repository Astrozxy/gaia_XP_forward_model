#!/usr/bin/env python

from __future__ import print_function, division

import numpy as np

import astropy.units as units
from astropy_healpix import HEALPix

import scipy.spatial.distance


def sph2cart(phi, theta):
    xyz = np.empty((phi.size,3), dtype=phi.dtype)
    xyz[:,0] = np.cos(phi) * np.sin(theta)
    xyz[:,1] = np.sin(phi) * np.sin(theta)
    xyz[:,2] = np.cos(theta)
    return xyz


def cart2sph(x, y, z):
    r = np.sqrt(x**2 + y**2 + z**2)
    theta = np.arccos(z/r)
    phi = np.arctan2(y, x)
    return phi, theta, r


def draw_from_sphere(n, rng=None):
    if rng is None:
        rng = np.random.default_rng()
    phi = rng.uniform(0, 2*np.pi, size=n)
    theta = np.arccos(2*rng.uniform(size=n)-1)
    return phi, theta


def offset_points_on_sphere(phi, theta, dist, rng=None):
    if rng is None:
        rng = np.random.default_rng()
    # Convert to Cartesian
    x,y,z = sph2cart(phi, theta).T
    # Add a small offset
    x += rng.normal(size=x.size) * dist / np.sqrt(3)
    y += rng.normal(size=x.size) * dist / np.sqrt(3)
    z += rng.normal(size=x.size) * dist / np.sqrt(3)
    # Project back down onto sphere
    phi, theta, r = cart2sph(x, y, z)
    # Return only position on unit sphere
    return phi, theta


def standardize_phi(phi):
    phi = np.mod(phi, 2*np.pi)
    idx = phi > np.pi
    if np.any(idx):
        phi[idx] -= 2*np.pi
    return phi


def gen_data(n1, n2, n_shared, offset, rng=None):
    if rng is None:
        rng = np.random.default_rng()

    # Draw points that are only in one catalog
    phi1,theta1 = draw_from_sphere(n1, rng=rng)
    phi2,theta2 = draw_from_sphere(n2, rng=rng)

    # Draw points shared between both catalogs
    phi_s,theta_s = draw_from_sphere(n_shared, rng=rng)

    # Offset shared points by random scatter
    dist = offset# / np.sqrt(2)
    phi1s,theta1s = offset_points_on_sphere(phi_s, theta_s, dist, rng=rng)
    phi2s,theta2s = offset_points_on_sphere(phi_s, theta_s, dist, rng=rng)

    # Concatenate shared with unique points, putting shared points first
    phi1 = np.hstack([phi1s, phi1])
    theta1 = np.hstack([theta1s, theta1])
    phi2 = np.hstack([phi2s, phi2])
    theta2 = np.hstack([theta2s, theta2])

    phi1 = standardize_phi(phi1)
    phi2 = standardize_phi(phi2)

    return (phi1,theta1), (phi2,theta2)


def calc_distance(phi1, theta1, phi2, theta2):
    xyz1 = sph2cart(phi1, theta1)
    xyz2 = sph2cart(phi2, theta2)
    dist = np.linalg.norm(xyz2-xyz1, axis=1)
    return dist


def plot_data(phi1, theta1, phi2, theta2, n_shared, title=''):
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(6,7))
    fig.subplots_adjust(
        top=0.95,
        bottom=0.10,
        hspace=0.05
    )

    ax = fig.add_subplot(2,1,1, projection='mollweide')
    n = len(phi1) + len(phi2)
    kw = dict(s=9/np.sqrt(n/1024), alpha=0.5, edgecolors='none')
    ax.scatter(phi1, np.pi/2-theta1, c='r', **kw)
    ax.scatter(phi2, np.pi/2-theta2, c='b', **kw)
    ax.set_title(title)

    ax = fig.add_subplot(2,1,2)
    dists = calc_distance(
        phi1[:n_shared], theta1[:n_shared],
        phi2[:n_shared], theta2[:n_shared]
    )
    ax.hist(np.degrees(dists)*3600, bins=int(n_shared*0.1**2), density=True)
    ax.axvline(np.degrees(np.mean(dists))*3600, c='k', ls=':', alpha=0.5)
    ax.set_xlabel('distance (arcsec)')

    return fig

def plot_patch(lon1, lat1, lon2, lat2, projection='mollweide'):
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(6,3))
    fig.subplots_adjust(
        top=0.95,
        bottom=0.10,
        hspace=0.05
    )

    ax = fig.add_subplot(1,1,1, projection=projection)
    n = len(lon1) + len(lon1)
    kw = dict(s=4/np.sqrt(n/1024), alpha=0.5, edgecolors='none')
    ax.scatter(lon1.to('rad').value, lat1.to('rad').value, c='r', **kw)
    ax.scatter(lon2.to('rad').value, lat2.to('rad').value, c='b', **kw)

    return fig


class HEALPixCatalog(object):
    def __init__(self, lon, lat, nside, show_progress=False):
        self.nside = nside
        self.hpix = HEALPix(nside=nside, order='nested')

        pix_idx = self.hpix.lonlat_to_healpix(lon, lat)

        self.sort_idx = np.argsort(pix_idx)
        pix_idx = pix_idx[self.sort_idx]
        lon = lon[self.sort_idx]
        lat = lat[self.sort_idx]

        pix_idx_unique = np.unique(pix_idx)
        start_idx = np.searchsorted(pix_idx, pix_idx_unique)
        start_idx = np.hstack([start_idx, len(pix_idx)])

        if show_progress:
            from tqdm.auto import tqdm
            pix_idx_unique = tqdm(pix_idx_unique)

        self.hpix_dict = {}
        for pidx,i0,i1 in zip(pix_idx_unique,start_idx[:-1],start_idx[1:]):
            assert np.all(pix_idx[i0:i1] == pidx)
            self.hpix_dict[pidx] = (lon[i0:i1], lat[i0:i1], self.sort_idx[i0:i1])

    def fetch_pixel(self, pix_idx):
        if pix_idx in self.hpix_dict:
            return self.hpix_dict[pix_idx]
        return [], [], []

    def fetch_pixels(self, pix_idx):
        lon, lat, idx = [], [], []
        for pidx in pix_idx:
            if pidx in self.hpix_dict:
                lon_pix, lat_pix, idx_pix = self.hpix_dict[pidx]
                lon.append(lon_pix)
                lat.append(lat_pix)
                idx.append(idx_pix)
        if len(lon):
            lon = np.hstack(lon)
            lat = np.hstack(lat)
            idx = np.hstack(idx)
        return lon, lat, idx

    def fetch_patch(self, lon, lat):
        pix_idx = np.unique(self.hpix.lonlat_to_healpix(lon, lat))
        pix_idx_neighbors = np.unique(self.hpix.neighbours(pix_idx).flat)
        pix_idx = np.unique(np.hstack([pix_idx, pix_idx_neighbors]))
        return self.fetch_pixels(pix_idx)

    def get_pix_indices(self):
        return self.hpix_dict.keys()


def dist2_matrix(lon1, lat1, lon2, lat2):
    xyz1 = sph2cart(lon1.to('rad').value, 0.5*np.pi-lat1.to('rad').value)
    xyz2 = sph2cart(lon2.to('rad').value, 0.5*np.pi-lat2.to('rad').value)
    d2 = scipy.spatial.distance.cdist(xyz1, xyz2, 'sqeuclidean')
    return d2


def match_catalogs(base_cat, over_cat, dist_max, show_progress=False):
    idx1_match, idx2_match, dist_match = [], [], []

    dmax_rad = dist_max.to('rad').value

    iterator = over_cat.get_pix_indices()

    if show_progress:
        from tqdm.auto import tqdm
        iterator = tqdm(iterator)

    for pix_idx in iterator:
        lon2, lat2, idx2 = over_cat.fetch_pixel(pix_idx)
        lon1, lat1, idx1 = base_cat.fetch_patch(lon2, lat2)

        if (len(lon1) == 0) or (len(lon2) == 0):
            continue

        d2 = dist2_matrix(lon1, lat1, lon2, lat2) # shape = (patch1, patch2)
        # Index in patch1 of nearest point to each coordinate in patch2
        idx1_min = np.argmin(d2, axis=0) # shape = (patch2,)
        # Minimum distance^2 from each coordinate in patch2
        d2_min = d2[idx1_min, np.arange(d2.shape[1])] # shape = (patch2,)

        idx_sel = (d2_min < dmax_rad**2)
        idx1_match.append(idx1[idx1_min[idx_sel]])
        idx2_match.append(idx2[idx_sel])
        dist_match.append(np.sqrt(d2_min[idx_sel])*units.rad)

    if len(idx1_match):
        idx1_match = np.hstack(idx1_match)
        idx2_match = np.hstack(idx2_match)
        dist_match = np.hstack(dist_match)
    else:
        dist_match = np.array([]) * units.rad

    return idx1_match, idx2_match, dist_match


def main():
    import matplotlib.pyplot as plt
    from time import perf_counter

    n1, n2, n_shared = 1024*16, 1024*4, 1024*16
    offset = np.radians(1./3600.)
    match_radius = 5*units.arcsec
    nside_base = 16
    nside_over = 8
    rng = np.random.default_rng(seed=123)

    print('Generating mock data ...')
    (phi1,theta1),(phi2,theta2) = gen_data(n1, n2, n_shared, offset, rng=rng)

    print('Plotting mock data ...')
    fig = plot_data(phi1, theta1, phi2, theta2, n_shared, title='mock data')
    fig.savefig('plots/mock_data.png', dpi=200)
    plt.close(fig)

    print('Partitioning base catalog ...')
    t0 = perf_counter()
    base_cat = HEALPixCatalog(
        phi1*units.rad,
        (0.5*np.pi-theta1)*units.rad,
        nside_base,
        show_progress=True
    )
    t1 = perf_counter()
    print(f'  --> {1000*(t1-t0)} ms')

    print('Partitioning overlay catalog ...')
    t0 = perf_counter()
    over_cat = HEALPixCatalog(
        phi2*units.rad,
        (0.5*np.pi-theta2)*units.rad,
        nside_over
    )
    t1 = perf_counter()
    print(f'  --> {1000*(t1-t0)} ms')

    print('Fetching patch for crossmatch ...')
    lon2, lat2, idx2 = over_cat.fetch_pixels([0])
    lon1, lat1, idx1 = base_cat.fetch_patch(lon2, lat2)

    print('Plotting patch ...')
    fig = plot_patch(lon1, lat1, lon2, lat2)
    fig.savefig('plots/mock_data_patch.png', dpi=200)
    plt.close(fig)

    print('Matching catalogs ...')
    t0 = perf_counter()
    idx1_match,idx2_match,d = match_catalogs(base_cat, over_cat, match_radius)
    t1 = perf_counter()
    print(f'  --> {1000*(t1-t0)} ms')

    n_true_matches = np.count_nonzero(idx1_match == idx2_match)
    n_false_matches = len(idx1_match) - n_true_matches
    print(f'{n_true_matches} of {n_shared} true matches found. '
          f'{n_false_matches} spurious matches found.')
    fig = plot_data(
        phi1[idx1_match], theta1[idx1_match],
        phi2[idx2_match], theta2[idx2_match],
        len(idx1_match),
        title='matches'
    )
    fig.savefig('plots/mock_data_matches.png', dpi=200)
    plt.close(fig)

    return 0


if __name__ == '__main__':
    main()

