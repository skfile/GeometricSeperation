import numpy as np
from typing import Optional, Tuple
from scipy.sparse.csgraph import connected_components
from scipy.spatial.distance import pdist, squareform
from scipy.spatial.distance import cdist

import logging

logger = logging.getLogger("utils")

DISCONNECTED_BRIDGING_FACTOR = 10.0

def connected_comp_helper(A: Optional[np.ndarray],
                          X: np.ndarray,
                          connect: bool = True) -> Optional[np.ndarray]:
    """
    Ensures graph connectivity by bridging disconnected components if connect=True.
    We replace 'inf' or 0 edges between components with bridging_val = largest_edge * DISCONNECTED_BRIDGING_FACTOR.
    """

    if A is None:
        logger.warning("Adjacency is None => skipping connectivity.")
        return A

    if A.shape[0] != A.shape[1]:
        raise ValueError(f"Adjacency must be square. shape={A.shape}")

    # Check if A is a sparse matrix
    from scipy import sparse
    is_sparse = sparse.issparse(A)
    orig_format = None
    
    if is_sparse:
        # Store original format
        orig_format = A.getformat()
        # Always convert to LIL for efficient modification of sparsity structure
        A = A.tolil()
    
    n_components, comp_labels = connected_components(A, directed=False, return_labels=True)
    if n_components > 1:
        if connect:
            logger.info(f"Graph has {n_components} disconnected components => bridging them.")
            
            # Handle sparse and dense matrices differently for finding the largest edge
            if is_sparse:
                data = A.data
                finite_data = [x for row in data for x in row if np.isfinite(x) and x > 0]
                if not finite_data:
                    bridging_val = 1e6
                else:
                    largest_edge = max(finite_data)
                    bridging_val = largest_edge * DISCONNECTED_BRIDGING_FACTOR
            else:
                finite_mask = np.isfinite(A) & (A > 0)
                if not np.any(finite_mask):
                    bridging_val = 1e6
                else:
                    largest_edge = np.max(A[finite_mask])
                    bridging_val = largest_edge * DISCONNECTED_BRIDGING_FACTOR

            # For each adjacent pair of components c, c+1, we connect them
            # in the minimal-dist pair.
            for c in range(n_components - 1):
                comp_i = np.where(comp_labels == c)[0]
                comp_j = np.where(comp_labels == c+1)[0]

                dist_ij = cdist(X[comp_i], X[comp_j], metric='euclidean')
                
                min_idx = np.unravel_index(np.argmin(dist_ij), dist_ij.shape)
                vi = comp_i[min_idx[0]]
                vj = comp_j[min_idx[1]]
                A[vi, vj] = bridging_val
                A[vj, vi] = bridging_val
        else:
            logger.info(f"Graph has {n_components} disconnected parts, not bridging.")

    # Convert back to original format if sparse
    if is_sparse:
        if orig_format == 'csr':
            A = A.tocsr()
        elif orig_format == 'csc':
            A = A.tocsc()
        elif orig_format == 'coo':
            A = A.tocoo()
    
    return A

def remove_duplicates(X: np.ndarray, tol: float = 1e-8) -> Tuple[np.ndarray, np.ndarray]:
    """
    Removes nearly-duplicate points based on a tolerance, returning the unique subset and the indices.
    """
    if X.ndim != 2:
        raise ValueError(f"X must be 2D, shape={X.shape}")
    # naive approach: sort, find diffs
    sorted_idx = np.lexsort(np.argsort(X, axis=1))
    sorted_X = X[sorted_idx]
    diffs = np.diff(sorted_X, axis=0)
    dist = np.linalg.norm(diffs, axis=1)
    keep_mask = np.insert(dist > tol, 0, True)
    X_unique = sorted_X[keep_mask]
    # Re-map to original indices
    unique_indices = np.where(keep_mask)[0]
    return X_unique, unique_indices

def preprocess_distance_matrix(D, large_value_multiplier=20.0):
    """
    Replaces inf with largest_finite*large_value_multiplier if any inf appear in D.
    """
    if not np.isfinite(D).all():
        finite_mask = np.isfinite(D)
        if not np.any(finite_mask):
            raise ValueError("All distances are infinite => cannot proceed.")
        max_finite = np.max(D[finite_mask])
        large_val = max_finite * large_value_multiplier
        n_infs = np.sum(~finite_mask)
        logging.getLogger("experiment_logger").info(f"Replaced {n_infs} inf distances with {large_val}.")
        D = np.where(np.isinf(D), large_val, D)
    return D

def normalize_distance_matrix(D):
    """
    Scale matrix so that max=1 by ignoring NaNs and replacing them with 0. If max=0 => return D unchanged.
    """
    D = np.nan_to_num(D, nan=0.0, posinf=1e10, neginf=-1e10)
    
    if np.all(np.isnan(D)):
        return np.zeros_like(D)
        
    dmax = np.nanmax(D)
    
    if dmax > 1e-10:
        normalized = D / dmax
        return np.nan_to_num(normalized, nan=0.0, posinf=0.0, neginf=0.0)
    
    return np.nan_to_num(D, nan=0.0)

def measure_from_potential(X, potential_name, potential_params, min_sum_threshold=1e-14):
    """
    Evaluate measure ~ exp(- potential(x)), then normalize.
    """
    from src.mesh_sampling import get_potential_func
    logger = logging.getLogger("experiment_logger")
    pot_func = get_potential_func(potential_name, potential_params)
    pot_vals = np.apply_along_axis(pot_func, 1, X)
    log_w = -pot_vals
    mx = np.max(log_w)
    log_w -= mx
    w = np.exp(log_w)
    s = w.sum()
    if s < min_sum_threshold:
        logger.warning(f"Potential measure sum < {min_sum_threshold} => fallback to uniform.")
        measure = np.ones(len(X)) / len(X)
    else:
        measure = w / s
    return measure