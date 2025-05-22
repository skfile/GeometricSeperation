#!/usr/bin/env python3
"""
metrics.py
-----------------------------
Distance and metric computation for manifold geometry comparison.

This module implements distance metrics for comparing manifold geometries, including:
- Gromov-Wasserstein distance for comparing metric measure spaces
- Gromov-Hausdorff approximation for measuring distance between metric spaces
- Single-linkage ultrametric construction for hierarchical comparison
"""

import numpy as np
import ot
from scipy.spatial.distance import pdist, squareform, cdist
from scipy.cluster.hierarchy import linkage
from src.utils import normalize_distance_matrix
from collections import defaultdict
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
import os

##########################################################
# 1) Gromov-Wasserstein
##########################################################

def compute_gromov_wasserstein(
    C1: np.ndarray, 
    C2: np.ndarray, 
    p: np.ndarray = None, 
    q: np.ndarray = None, 
    loss_fun: str = 'square_loss',
    max_iter: int = 10000, 
    tol: float = 1e-4
) -> float:
    """
    Compute Gromov-Wasserstein distance between two metric measure spaces.
    
    Parameters
    ----------
    C1 : np.ndarray
        First cost/distance matrix
    C2 : np.ndarray
        Second cost/distance matrix
    p : np.ndarray, optional
        Distribution over the first space (uniform if None)
    q : np.ndarray, optional
        Distribution over the second space (uniform if None)
    loss_fun : str, default='square_loss'
        Type of loss function ('square_loss', 'kl_loss')
    max_iter : int, default=10000
        Maximum number of iterations in the optimization
    tol : float, default=1e-4
        Convergence tolerance
        
    Returns
    -------
    float
        The Gromov-Wasserstein cost (not the square root)
    """
    # Handle default uniform distributions
    if p is None:
        p = np.ones(C1.shape[0]) / C1.shape[0]
    if q is None:
        q = np.ones(C2.shape[0]) / C2.shape[0]
        
    gw_cost = ot.gromov.gromov_wasserstein2(
        C1, C2, p, q,
        loss_fun=loss_fun,
        max_iter=max_iter,
        tol=tol
    )
    return gw_cost

def compute_gromov_hausdorff_approx(X: np.ndarray, Y: np.ndarray, metric: str = 'euclidean') -> float:
    """
    Approximate the Gromov-Hausdorff distance between two point clouds.
    
    Computes an approximation to the Gromov-Hausdorff distance using
    the Gromov-Wasserstein distance with uniform distributions.
    
    Parameters
    ----------
    X : np.ndarray
        First point cloud, shape (n_samples_X, n_features)
    Y : np.ndarray
        Second point cloud, shape (n_samples_Y, n_features)
    metric : str, default='euclidean'
        Distance metric to use
        
    Returns
    -------
    float
        Approximation to the GH distance (raw GW cost, not sqrt)
        To get the GH distance approximation, take the square root
    """
    distX = pdist(X, metric=metric)
    distY = pdist(Y, metric=metric)
    Cx = squareform(distX)
    Cy = squareform(distY)
    N, M = len(X), len(Y)
    p = np.ones(N)/N
    q = np.ones(M)/M
    gw_cost = ot.gromov.gromov_wasserstein2(
        Cx, Cy, p, q, loss_fun='square_loss',
        max_iter=10000, tol=1e-4
    )
    return gw_cost

def compute_gw_parallel(data_pairs, metric='euclidean', max_workers=None):
    """
    Compute Gromov-Wasserstein distances for multiple pairs of datasets in parallel.
    
    Parameters:
    -----------
    data_pairs : list of tuples
        Each tuple contains (X, Y) where X and Y are the datasets to compare
    metric : str
        Distance metric to use
    max_workers : int or None
        Maximum number of worker processes (None = use CPU count)
        
    Returns:
    --------
    list of costs, one for each pair
    """
    if max_workers is None:
        max_workers = os.cpu_count()
    
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        results = list(executor.map(
            lambda pair: compute_gromov_hausdorff_approx(pair[0], pair[1], metric), 
            data_pairs
        ))
    return results

##########################################################
# 2) Single-Linkage Ultrametric & GH on Ultrametrics
##########################################################

def compute_single_linkage_ultrametric(points, metric='euclidean'):
    """
    Optimized version of single-linkage ultrametric computation
    """
    import numpy as np
    from scipy.spatial import cKDTree
    from math import log, ceil
    import time

    n, d = points.shape
    if n < 2:
        return np.zeros((n, n), dtype=float)
    
    # For larger datasets, use scipy's faster implementation
    if n > 1000:
        from scipy.cluster.hierarchy import single, fcluster
        from scipy.spatial.distance import pdist, squareform
        
        # Compute condensed distance matrix and linkage
        start_time = time.time()
        condensed_dist = pdist(points, metric=metric)
        Z = single(condensed_dist)
        
        # Extract ultrametric matrix from linkage matrix
        U = np.zeros((n, n), dtype=float)
        for i in range(n-1):
            cluster1, cluster2, height, _ = Z[i]
            cluster1, cluster2 = int(cluster1), int(cluster2)
            
            # If these are original points
            if cluster1 < n:
                indices1 = [cluster1]
            else:
                # Get all points in this merged cluster
                cluster_id = cluster1 - n + 1
                indices1 = np.where(fcluster(Z, cluster_id, criterion='maxclust') == 1)[0]
                
            if cluster2 < n:
                indices2 = [cluster2]
            else:
                cluster_id = cluster2 - n + 1
                indices2 = np.where(fcluster(Z, cluster_id, criterion='maxclust') == 1)[0]
                
            # Update ultrametric distances
            for idx1 in indices1:
                for idx2 in indices2:
                    U[idx1, idx2] = height
                    U[idx2, idx1] = height
        
        return U

    # Original implementation for smaller datasets
    alpha = np.sqrt(2)
    k = max(2, ceil(d * log(n)))  # Ensure k is at least 2

    # Step 1: Compute rk(xi) for each point xi
    tree = cKDTree(points)
    distances, _ = tree.query(points, k=k+1, p=2)
    rk = distances[:, -1]  # distance to k-th nearest neighbor

    # Step 2: Prepare all possible edges with distance <= alpha * max(rk)
    max_rk = np.max(rk)
    cutoff_distance = alpha * max_rk

    # Compute all pairs within cutoff_distance using cKDTree
    pairs = tree.query_pairs(r=cutoff_distance, p=2)

    # Convert set of pairs to a sorted list based on distance using vectorized operations
    if pairs:
        pair_list = np.array(list(pairs))
        diffs = points[pair_list[:, 0]] - points[pair_list[:, 1]]
        pair_distances = np.linalg.norm(diffs, axis=1)
        sorted_indices = np.argsort(pair_distances)
        sorted_pairs = pair_list[sorted_indices]
        sorted_distances = pair_distances[sorted_indices]
    else:
        sorted_pairs = np.empty((0, 2), dtype=int)
        sorted_distances = np.array([])

    # Initialize Union-Find structure with cluster membership tracking
    parent = np.arange(n)
    rank_union = np.zeros(n, dtype=int)
    # Create a dictionary to efficiently track cluster members
    cluster_members = {i: {i} for i in range(n)}  # Using sets for fast membership operations

    def find(u):
        # Path compression only on the path we're exploring
        if parent[u] != u:
            parent[u] = find(parent[u])
        return parent[u]

    def union(u, v):
        pu, pv = find(u), find(v)
        if pu == pv:
            return False, None  # Already in the same set
        
        # Union by rank with cluster merging
        if rank_union[pu] < rank_union[pv]:
            parent[pu] = pv
            cluster_members[pv].update(cluster_members[pu])
            members = cluster_members[pv]
            del cluster_members[pu]
            root = pv
        else:
            parent[pv] = pu
            cluster_members[pu].update(cluster_members[pv])
            members = cluster_members[pu]
            del cluster_members[pv]
            root = pu
            if rank_union[pu] == rank_union[pv]:
                rank_union[pu] += 1
                
        return True, (root, members)

    # Initialize ultrametric matrix with zeros on the diagonal and infinities elsewhere
    U = np.full((n, n), np.inf)
    np.fill_diagonal(U, 0)

    # Process sorted pairs and merge clusters
    for (i, j), dist in zip(sorted_pairs, sorted_distances):
        # Determine the current r as the maximum of rk[i] and rk[j]
        current_r = max(rk[i], rk[j])
        # The condition to include the edge is dist <= alpha * r
        if dist > alpha * current_r:
            continue  # Do not include this edge

        # Attempt to union the clusters
        merged, result = union(i, j)
        if merged:
            root, members = result
            # Convert the set to a list for indexing - use numpy for speed
            member_array = np.array(list(members))
            # Use vectorized operations for faster updates
            for i in range(len(member_array)):
                m1 = member_array[i]
                m2s = member_array[i+1:]
                U[m1, m2s] = np.minimum(U[m1, m2s], current_r)
                U[m2s, m1] = U[m1, m2s]

    # After processing all pairs, some pairs might still be infinity if they were never connected
    # To handle this, we can set their ultrametric distance to the maximum rk
    U[U == np.inf] = max_rk

    return U

def compute_ultrametrics_parallel(datasets, metric='euclidean', max_workers=None):
    """
    Compute single-linkage ultrametrics for multiple datasets in parallel.
    
    Parameters:
    -----------
    datasets : list of arrays
        Each array is a dataset to process
    metric : str
        Distance metric to use
    max_workers : int or None
        Maximum number of worker processes (None = use CPU count)
        
    Returns:
    --------
    list of ultrametric matrices, one for each dataset
    """
    if max_workers is None:
        max_workers = os.cpu_count()
    
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        results = list(executor.map(
            lambda data: compute_single_linkage_ultrametric(data, metric), 
            datasets
        ))
    return results

def approximate_gh_on_ultrametrics(U1, U2, loss_fun='square_loss', max_iter=10000, tol=1e-4):
    """
    Approx GH distance = sqrt( Gromov-Wasserstein(U1, U2) ) with uniform weights.
    U1, U2: NxN and MxM ultrametric distance matrices (not necessarily same size).
    """
    import ot
    N = U1.shape[0]
    M = U2.shape[0]
    p = np.ones(N)/N
    q = np.ones(M)/M
    # Normalize them so max=1
    U1n = normalize_distance_matrix(U1)
    U2n = normalize_distance_matrix(U2)

    cost = ot.gromov.gromov_wasserstein2(
        U1n, U2n, p, q, loss_fun=loss_fun, max_iter=max_iter, tol=tol
    )
    return np.sqrt(abs(cost))

def approximate_gh_on_ultrametrics_parallel(ultrametric_pairs, loss_fun='square_loss', 
                                           max_iter=10000, tol=1e-4, max_workers=None):
    """
    Compute approximate Gromov-Hausdorff distances between multiple pairs of ultrametrics in parallel.
    
    Parameters:
    -----------
    ultrametric_pairs : list of tuples
        Each tuple contains (U1, U2) where U1 and U2 are the ultrametric matrices to compare
    loss_fun, max_iter, tol : parameters for GW computation
    max_workers : int or None
        Maximum number of worker processes (None = use CPU count)
        
    Returns:
    --------
    list of GH distances, one for each pair
    """
    if max_workers is None:
        max_workers = os.cpu_count()
        
    def _compute_single_gh(pair):
        U1, U2 = pair
        return approximate_gh_on_ultrametrics(U1, U2, loss_fun, max_iter, tol)
    
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        results = list(executor.map(_compute_single_gh, ultrametric_pairs))
    return results