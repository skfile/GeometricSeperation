# sampling.py
# -----------
"""
Unifies manifold sampling (sphere, torus, etc.) with the new mesh-based approach.
We also handle "general" datasets (spiral, crown, etc.).
"""

import numpy as np
from typing import Tuple, Optional, List
import logging

from src.dataset_funcs import (
    dset_grid_spiral, dset_spin_top, dset_stingray, dset_crown, dset_grid_cat_plane
)
from src.mesh_sampling import (
    get_mesh_or_complex,
    uniform_or_biased_sampling,
    get_potential_func,
    SimplicialComplex,
    Mesh
)

###############################################################################
# Simple "uniform"/"biased" sampling for non-manifold data
###############################################################################
def sample_uniform(X: np.ndarray, fraction: float = 1.0, seed: int = 0) -> Tuple[np.ndarray, np.ndarray]:
    """
    Uniformly samples a fraction of points from X. Returns (X_sub, idx_sub).
    """
    rng = np.random.default_rng(seed)
    n_full = len(X)
    n_sub = int(np.floor(n_full * fraction))
    if not (0 < fraction <= 1.0):
        raise ValueError("fraction must be in the interval (0, 1].")
    idx = rng.choice(n_full, size=n_sub, replace=False)
    return X[idx], idx


def sample_biased(X: np.ndarray, colors: np.ndarray, fraction: float = 1.0, seed: int = 0) -> Tuple[np.ndarray, np.ndarray]:
    """
    Biasedly samples a fraction of points from X, weighting by 'colors'.
    Returns (X_sub, idx_sub).
    """
    rng = np.random.default_rng(seed)
    n_full = len(X)
    n_sub = int(np.floor(n_full * fraction))
    if not (0 < fraction <= 1.0):
        raise ValueError("fraction must be in the interval (0, 1].")

    raw = np.exp(-colors)
    ssum = raw.sum()
    if ssum < 1e-14:
        raise ValueError("All potential weights near zero. Adjust potential function.")
    probs = raw / ssum

    idx = rng.choice(n_full, size=n_sub, replace=False, p=probs)
    return X[idx], idx


###############################################################################
# Manifold-based shapes: sample_data_manifold
###############################################################################
def sample_data_manifold(
    name: str,
    total_points: int = 1000,
    fraction: float = 1.0,
    method: str = "uniform",
    dim: Optional[int] = None,
    seed: int = 42,
    potential_name: Optional[str] = None,
    potential_params: Optional[dict] = None,
    noise: bool = False,
    **kwargs
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    High-level function for manifold sampling (mesh-based or parametric).
    Returns (X_full, A_full, X_sub, A_sub, colors_full, colors_sub).

    No hidden logic tries to guess shape subdivisions. We rely on the user-defined
    shape generator defaults or explicit config. For certain shapes like 'k_sphere'
    or '4d_simplex', we pass total_points if no user-provided param is set.

    Parameters
    ----------
    name : str
        Name of the manifold shape ("2d_annulus", "3d_torus", "k_sphere", etc.).
    total_points : int, optional
        The total number of points to generate for the full shape. Some shapes
        interpret it as 'n_points' or 'n_vertices' if not otherwise set.
    fraction : float, optional
        The fraction of points to sub-sample from the full set. 0 < fraction <=1.
    method : str, optional
        "uniform" or "biased".
    dim : int, optional
        If needed by user (unused by default).
    seed : int, optional
        Random seed for reproducibility.
    potential_name : str, optional
        Name of potential function if method=="biased".
    potential_params : dict, optional
        Additional potential function params if method=="biased".
    kwargs : dict
        Additional shape parameters (e.g., n_ring, n_radial, n_theta, n_phi, etc.)
        that the shape generator can accept.

    Returns
    -------
    X_full, A_full, X_sub, A_sub, colors_full, colors_sub : tuple
        Where:
        - X_full : (N, d) full coordinates
        - A_full : optional (N, N) adjacency
        - X_sub : (M, d) sub-sampled coordinates
        - A_sub : optional (M, M) adjacency for sub-sample
        - colors_full : array of length N
        - colors_sub : array of length M
    """
    import logging
    logger = logging.getLogger("mesh_sampling")

    if potential_params is None:
        potential_params = {}

    if method == "biased" and not potential_name:
        raise ValueError("For 'biased' sampling, must provide 'potential_name'.")

    potential_func = None
    if method == "biased":
        potential_func = get_potential_func(potential_name, potential_params)

    from src.mesh_sampling import get_mesh_or_complex, Mesh, SimplicialComplex, uniform_or_biased_sampling

    shape_kwargs = dict(kwargs)
    name_lower = name.lower()

    if name_lower == "k_sphere" and 'n_points' not in shape_kwargs:
        shape_kwargs['n_points'] = total_points
    elif name_lower == "4d_simplex" and 'n_vertices' not in shape_kwargs:
        shape_kwargs['n_vertices'] = total_points

    try:
        shape_obj = get_mesh_or_complex(name_lower, **shape_kwargs)
        if shape_obj is None:
            raise ValueError(f"Shape '{name}' not created. Check parameters.")
    except Exception as e:
        raise ValueError(f"Error generating shape '{name}': {e}")

    # Count how many vertices
    if isinstance(shape_obj, Mesh):
        vcount = shape_obj.num_vertices()
    elif isinstance(shape_obj, SimplicialComplex):
        vcount = shape_obj.num_vertices()
    else:
        vcount = len(shape_obj.vertices)

    # Subsample size
    requested_samples = int(np.floor(total_points * fraction))
    n_samples = min(requested_samples, vcount)
    if requested_samples > vcount:
        logger.warning(f"Shape '{name}': requested {requested_samples} > {vcount}. Capping to {n_samples}.")

    # Actually sample
    try:
        X_full, A_full, X_sub, A_sub = uniform_or_biased_sampling(
            shape_obj,
            n_samples=n_samples,
            sampling_method=method,
            potential_func=potential_func,
            random_state=seed
        )
    except Exception as e:
        raise ValueError(f"Error sampling shape '{name}': {e}")

    # Create color arrays from coordinate 0
    if X_full.shape[1] > 0:
        colors_full = X_full[:, 0]
    else:
        colors_full = np.zeros(len(X_full))

    if X_sub.shape[1] > 0:
        colors_sub = X_sub[:, 0]
    else:
        colors_sub = np.zeros(len(X_sub))

    if noise:
        n_full, d_full = X_full.shape
        rng = np.random.default_rng(seed)
        noise_full = rng.normal(loc=0.0, scale=0.1, size=(n_full, d_full))
        X_full += noise_full

        n_sub, d_sub = X_sub.shape
        noise_sub = rng.normal(loc=0.0, scale=0.1, size=(n_sub, d_sub))
        X_sub += noise_sub

    return X_full, A_full, X_sub, A_sub, colors_full, colors_sub


def sample_data_general(
    name: str,
    total_points: int = 1000,
    fraction: float = 1.0,
    method: str = "uniform",
    seed: int = 42,
    noise: bool = False
) -> Tuple:
    """
    LEGACY IMPLEMENTATION: Samples data from general (non-manifold) datasets such as spirals, crowns, etc.
    Returns (X_full, X_sample, colors_full, colors_sample).
    """
    name_lower = name.lower()
    if name_lower == "spiral":
        data_dict = dset_grid_spiral(noise_std=0.01, global_noise_std=0, m=total_points, seed=seed)
    elif name_lower == "crown":
        data_dict = dset_crown(seed=seed)
    elif name_lower == "stingray":
        data_dict = dset_stingray(seed=seed)
    elif name_lower == "grid_cat_plane":
        data_dict = dset_grid_cat_plane(seed=seed)
    elif name_lower == "spin_top":
        data_dict = dset_spin_top(seed=seed)
    else:
        raise ValueError(f"Dataset '{name}' not supported in 'general' mode.")

    X_full = data_dict['X']
    colors_full = data_dict.get('c', np.zeros(len(X_full)))
    colors_full = np.asarray(colors_full)

    # Sub-sample
    if method == "uniform":
        X_sample, idx = sample_uniform(X_full, fraction=fraction, seed=seed)
        colors_sample = colors_full[idx]
    elif method == "biased":
        X_sample, idx = sample_biased(X_full, colors_full, fraction=fraction, seed=seed)
        colors_sample = colors_full[idx]
    else:
        raise ValueError(f"Sampling method '{method}' not supported for 'general'.")

    return X_full, X_sample, colors_full, colors_sample