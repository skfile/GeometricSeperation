"""
dataset_funcs.py
----------------
This module contains functions to generate various datasets (manifolds and synthetic shapes),
as well as utility functions for grid creation and coordinate manipulation.
"""

import numpy as np
from sklearn.datasets import make_blobs

def add_zero_column(X, n=1):
    """
    Add n zero columns to X.

    Parameters
    ----------
    X : array-like, shape (n_samples, n_features)
        Input data.
    n : int, default=1
        Number of zero columns to add.

    Returns
    -------
    X_new : array of shape (n_samples, n_features + n)
        Data with added zero columns.
    """
    X = np.asarray(X)
    if X.ndim == 1:
        X = X[:, np.newaxis]
    zeros = np.zeros((X.shape[0], n))
    return np.hstack((X, zeros))


def nd_tri_grid(side, dim, strict_side=True, centering=True):
    """
    Generates an N-dimensional regular triangular grid.

    Parameters
    ----------
    side : int
        Grid size parameter.

    dim : int
        Number of dimensions.

    strict_side : bool, default=True
        If True, ensures the grid size matches 'side'.

    centering : bool, default=True
        If True, centers the grid at the origin.

    Returns
    -------
    M : ndarray
        Grid coordinates.

    X : ndarray of shape (n_samples, dim)
        Flattened grid points.
    """
    from numpy import sqrt, floor, zeros
    v = side
    ndim_vect = []
    stp_vect = []
    for p in range(1, dim + 1):
        q = sqrt((p + 1) / (2 * p))
        w = int(floor(v / q))
        ndim_vect.append(w)
        stp_vect.append(q)

    grid_shape = ndim_vect + [dim]
    M = zeros(grid_shape)

    def make_array_index(p, Ndim):
        subsfield = [slice(None)] * (Ndim + 1)
        subsfield[-1] = p
        return tuple(subsfield)

    for p in range(dim):
        q = np.sqrt((p + 2) / (2 * (p + 1)))
        w = ndim_vect[p]
        s = np.arange(w).reshape([w] + [1] * (dim - p - 1))
        sample_mat = s * q
        coord_index = make_array_index(p, dim)
        M[coord_index] += sample_mat

    for p in range(1, dim):
        for m in range(p):
            find_vect = (np.arange(M.shape[p]) % 2 == 1).reshape([1] * p + [-1] + [1] * (dim - p - 1))
            shift_vect = stp_vect[m] / (m + 2)
            shift_mat = find_vect * shift_vect
            coord_index = make_array_index(m, dim)
            M[coord_index] += shift_mat

    if centering:
        for p in range(dim):
            coord_index = make_array_index(p, dim)
            M[coord_index] -= M[coord_index].mean()

    if strict_side:
        slices = [slice(0, side)] * dim + [slice(None)]
        M = M[tuple(slices)]

    X = M.reshape(-1, dim)
    return M, X


def get_base_X(side, dim, sampling, strict_side=True, seed=0):
    """
    Generate base coordinates using various sampling methods.

    Parameters
    ----------
    side : int
        Size parameter for the grid.

    dim : int
        Number of dimensions.

    sampling : str
        Sampling method: 'trigrid', 'sqgrid', 'uniform', 'normal', 'sunflower'.

    strict_side : bool, default=True
        If True, ensures the grid size matches 'side' in 'trigrid'.

    seed : int, default=0
        Random seed for reproducibility.

    Returns
    -------
    base_X : array of shape (n_samples, dim)
        Generated coordinates.
    """
    np.random.seed(seed)
    if sampling == 'trigrid' and dim == 1:
        sampling = 'sqgrid'

    if sampling == 'trigrid':
        _, base_X = nd_tri_grid(side, dim, strict_side)
    elif sampling == 'sqgrid':
        rng = np.arange(side, dtype='float64') - side // 2
        coords = np.meshgrid(*([rng] * dim), indexing='ij')
        base_X = np.vstack([c.flatten() for c in coords]).T
    elif sampling == 'uniform':
        num_points = side ** dim
        base_X = np.random.uniform(-side // 2, side // 2, size=(num_points, dim))
    elif sampling == 'normal':
        num_points = side ** dim
        base_X = np.random.normal(scale=side // 2, size=(num_points, dim))
    elif sampling == 'sunflower':
        if dim != 2:
            raise ValueError("Sunflower sampling is only defined for 2D.")
        num_pts = side ** 2
        indices = np.arange(0, num_pts) + 0.5
        r = np.sqrt(indices / num_pts)
        theta = np.pi * (1 + 5 ** 0.5) * indices
        x = r * np.cos(theta)
        y = r * np.sin(theta)
        base_X = np.column_stack((x, y))
        base_X -= base_X.mean(axis=0)
        base_X *= side // 2
    else:
        raise ValueError(f"Sampling method '{sampling}' not recognized.")

    return base_X


def dset_grid_cat_plane(arc_len=1.5, z_len=2, a=0.2, n=13, noise_std=0, seed=0, plot=False):
    """
    Generates a grid of concatenated planes dataset.
    """
    np.random.seed(seed)
    s = np.tile(np.linspace(-arc_len, arc_len, int(2 * arc_len) * n), z_len * n)
    if noise_std > 0:
        s += np.random.randn(s.size) * noise_std
    sx = a * np.log((s + np.sqrt(s ** 2 + a ** 2)) / a)
    sy = a * np.cosh(np.log((s + np.sqrt(s ** 2 + a ** 2)) / a))

    Z = np.repeat(np.linspace(0, z_len, z_len * n), 2 * arc_len * n)
    if noise_std > 0:
        Z += np.random.randn(Z.size) * noise_std

    data = np.vstack((sx, sy, Z)).T
    data -= data.mean(axis=0)
    colors = s
    Y = np.column_stack((s, Z))

    return {'X': data, 'c': colors, '2d_coords': [0, 2], 'Y': Y}


def dset_crown(radius=1, scale=2., s=0.5, step=0.05, noise_std=0.025, seed=0, plot=False):
    """
    Generates a crown-shaped dataset.
    """
    np.random.seed(seed)
    ds = np.arange(-2, 2, step)
    max_zs = scale * np.exp(-ds ** 2 / (2 * s ** 2))
    z_step = np.sqrt((1 - radius * np.cos(step / 4 * 2 * np.pi)) ** 2 +
                     (radius * np.sin(step / 4 * 2 * np.pi)) ** 2)

    X_list, Y_list, Z_list = [], [], []
    all_rs = []
    for zi, max_z in enumerate(np.round(max_zs, 2)):
        r = zi * step
        if max_z <= z_step:
            zs = [0]
        else:
            zs = np.arange(0, max_z, z_step)
        rs = [r] * len(zs)
        all_rs += rs
        rs = np.array(rs)

        xs = radius * np.cos(rs / 4 * 2 * np.pi)
        ys = radius * np.sin(rs / 4 * 2 * np.pi)

        X_list.extend(xs)
        Y_list.extend(ys)
        Z_list.extend(zs)

    data = np.column_stack((X_list, Y_list, Z_list))
    if noise_std > 0:
        data += np.random.randn(*data.shape) * noise_std
    data -= data.mean(axis=0)
    colors = all_rs

    return {'X': data, 'c': colors, '2d_coords': [1, 2]}


def dset_spin_top(start=-1.5, stop=0.75, scale=0.25, s=0.5, step=0.075,
                  noise_std=0.025, seed=0, plot=False):
    """
    Generates a spin-top shaped dataset.
    """
    np.random.seed(seed)
    ds = np.arange(start, stop + step, step)
    max_rs = scale * np.exp(-ds ** 2 / (2 * s ** 2))

    X_list, Y_list, Z_list = [], [], []
    for zi, max_r in enumerate(np.round(max_rs, 2)):
        z = zi * step
        if max_r <= step:
            rs = [0]
        else:
            rs = np.arange(0, max_r, step)
        for r in rs:
            if r == 0:
                xs, ys = np.zeros(1), np.zeros(1)
            else:
                thetas = np.linspace(0, 2 * np.pi, int(2 * np.pi / (step / r)))[:-1]
                xs, ys = r * np.cos(thetas), r * np.sin(thetas)
            zs = np.full_like(xs, z)
            X_list.extend(xs)
            Y_list.extend(ys)
            Z_list.extend(zs)

    data = np.column_stack((X_list, Y_list, Z_list))
    if noise_std > 0:
        data += np.random.randn(*data.shape) * noise_std
    data -= data.mean(axis=0)
    colors = Z_list

    return {'X': data, 'c': colors, '2d_coords': [1, 2]}


def dset_grid_spiral(noise_std=0.01, global_noise_std=0, m=200, seed=10,
                     normal_noise=0, length_phi=15, plot=False):
    """
    Generates a 2D spiral dataset.
    """
    np.random.seed(seed)
    phi = length_phi * np.linspace(0, 1, m) + noise_std * np.random.randn(m)
    xi = np.random.rand(m)
    X = (1 / 6) * (phi + normal_noise * xi) * np.sin(phi)
    Y = (1 / 6) * (phi + normal_noise * xi) * np.cos(phi)

    data = np.column_stack((X, Y))
    data -= data.mean(axis=0)
    if global_noise_std > 0:
        data += global_noise_std * np.random.rand(*data.shape)
    colors = phi

    return {'X': data, 'c': colors, '2d_coords': [0, 1]}


def dset_2d_gaussian(cluster_ns=[100, 100], cluster_stds=[0.5, 0.75],
                     centers=None, seed=0, plot=False):
    """
    Generates a 2D Gaussian blobs dataset.
    """
    data, y = make_blobs(n_samples=cluster_ns, centers=centers,
                         cluster_std=cluster_stds, random_state=seed, shuffle=False)
    data -= data.mean(axis=0)
    colors = y

    return {'X': data, 'c': colors, '2d_coords': [0, 1]}


def dset_2d_plane(side_nx=1., n_points_x=15, side_ny=1., n_points_y=15,
                  param_noise=False, gauss_noise_std=0.05, seed=0, plot=False):
    """
    Generates a 2D plane dataset.
    """
    np.random.seed(seed)
    if param_noise:
        X = np.random.rand(n_points_x * n_points_y) - 0.5
        Y = np.random.rand(n_points_x * n_points_y) - 0.5
    else:
        X = np.linspace(-side_nx, side_nx, n_points_x)
        Y = np.linspace(-side_ny, side_ny, n_points_y)
        X, Y = np.meshgrid(X, Y)
        X = X.flatten()
        Y = Y.flatten()

    data = np.column_stack((X, Y)) + gauss_noise_std * np.random.randn(X.size, 2)
    data -= data.mean(axis=0)
    colors = X

    return {'X': data, 'c': colors, '2d_coords': [0, 1]}


def dset_stingray(plot=False, seed=0):
    """
    Generates a 'stingray' shaped dataset with a 2D body and a 1D tail.
    """
    from scipy.spatial.distance import squareform, pdist
    # Body
    gauss = lambda x, s: np.exp(-x**2/s**2)
    body_widths = np.r_[gauss(np.linspace(-.4,0,15),.25),gauss(np.linspace(0,2,30),1)]
    body_widths -= body_widths.min()
    body_y = np.linspace(-.4,2,len(body_widths))
    body_y -= body_y.min()
    xrng = np.ptp(body_y)
    body_y /= xrng
    body_widths /= xrng

    np.random.seed(seed)
    # Tail
    n = 40
    i = np.arange(n)
    th = np.linspace(1.05*np.pi/3,np.pi,n)
    r = np.cos(th) + 1
    scl = .3
    x, y = r*scl*np.cos(th), r*scl*np.sin(th)
    llim, rlim = 0,n
    tail_x = -x[llim:rlim]*2 + body_y.max() + (body_y[-1]-body_y[-2])
    tail_y = y[llim:rlim]

    tail_x = tail_x[::-1]
    tail_y = tail_y[::-1]

    tailX = np.hstack([tail_x[:,None],tail_y[:,None]])
    D1 = squareform(pdist(tailX))
    delta_tail = np.mean(np.diag(D1,k=1))*1.75

    to_del = []
    for ii in range(1,n-1):
        for prev in range(ii-1,-1,-1):
            if prev not in to_del:
                break
        if D1[ii,prev] < delta_tail and D1[ii,ii+1] < delta_tail:
            to_del.append(ii)

    tailX = np.delete(tailX,to_del,axis=0)
    tailX = tailX[:15]

    xsamples = len(body_y)
    pts = []
    for xi,xv in enumerate(body_y[:-1]):
        if xi % 2 == 1: continue
        h = body_widths[xi]
        ysamples = max(1,int(round(h*xsamples)))
        yvals = np.linspace(-h,h,ysamples)
        pts += list(zip([xv]*ysamples,yvals))

    # append tail
    pts += list(tailX)
    pts = np.array(pts)

    np.random.seed(1)
    pct = .2
    N = pts.shape[0]
    sub = np.random.choice(range(N),int(pct*N),False)
    data = np.delete(pts,sub,axis=0)
    data -= data.mean(axis=0)

    return {'X':data,'c':None,'2d_coords':[0,1]}