"""
kernels.py
-----------------------------
Graph and kernel construction module for manifold separation analysis.

This module provides functions to construct various similarity kernels and
neighborhood graphs from point cloud data, including k-NN graphs, Gaussian
kernels, and adaptive neighborhood methods.
"""

import numpy as np
import pandas as pd
import logging

from src.utils import *
from src.graph_methods import *
from sklearn.neighbors import kneighbors_graph
from scipy.sparse.csgraph import shortest_path as sp_shortest_path
from scipy.spatial.distance import pdist, squareform

logger = logging.getLogger(__name__)

def ian_kernel(X: np.ndarray) -> np.ndarray:
    """
    Compute Iterated Adaptive Neighborhoods (IAN) kernel for the input data.
    
    Parameters
    ----------
    X : np.ndarray
        Input data matrix of shape (n_samples, n_features)
    
    Returns
    -------
    np.ndarray
        The IAN adjacency matrix as a dense array, or None if computation fails
    """
    try:
        from ian.ian import IAN
        if X.shape[0] < 10:
            return None
        G, wG, optScales, disc_pts = IAN('exact', X, verbose=0, obj='greedy')
        return wG.toarray()
    except Exception as e:
        print(f"IAN kernel error: {e}")
        return None

def compute_gaussian_kernel(X: np.ndarray, sigma: float) -> np.ndarray:
    """
    Compute the Gaussian (RBF) kernel matrix with bandwidth sigma.
    
    Parameters
    ----------
    X : np.ndarray
        Input data matrix of shape (n_samples, n_features)
    sigma : float
        Bandwidth parameter controlling the kernel width
        
    Returns
    -------
    np.ndarray
        Gaussian kernel matrix of shape (n_samples, n_samples)
    """
    D = squareform(pdist(X, 'sqeuclidean'))
    K = np.exp(-D / (2 * sigma**2))
    return K

def compute_knn_shortest_path_kernel(
    X: np.ndarray, 
    knn: int, 
    shortest_path: bool = False, 
    pruning_method: Optional[str] = None, 
    metric: str = 'euclidean'
) -> np.ndarray:
    """
    Build a k-NN graph and optionally compute shortest paths or perform pruning.
    
    Parameters
    ----------
    X : np.ndarray
        Input data matrix of shape (n_samples, n_features)
    knn : int
        Number of nearest neighbors to use
    shortest_path : bool, default=False
        If True, returns all-pairs shortest path distances instead of adjacency
    pruning_method : str, optional
        Method to prune the graph ('mst', 'density', 'bisection', etc.)
    metric : str, default='euclidean'
        Distance metric to use for nearest neighbor search
        
    Returns
    -------
    np.ndarray
        Graph matrix as adjacency or shortest-path distances
    """
    # Validate metric
    valid_metrics = {
        'euclidean', 'manhattan', 'minkowski', 'chebyshev', 'cosine'
    }
    if not (callable(metric) or metric in valid_metrics):
        logger.warning(f"Invalid metric '{metric}'. Falling back to 'euclidean'.")
        metric = 'euclidean'

    if X.shape[0] < knn + 1:
        logger.warning(f"Dataset has fewer points ({X.shape[0]}) than knn+1 ({knn+1}). Returning None.")
        return None

    # Build the sparse adjacency
    knn_graph_sparse = kneighbors_graph(
        X, n_neighbors=knn, mode="distance", include_self=False
    )

    # Convert to dense to ensure we can do connected_comp_helper or direct manipulations
    knn_graph_dense = knn_graph_sparse.toarray()
    knn_graph_dense = connected_comp_helper(knn_graph_dense, X, connect=True)

    # If user asked for a pruning method
    if pruning_method is not None:
        from src.graph_methods import (
            build_knn_graph,
            prune_random,
            prune_distance,
            prune_bisection,
            prune_mst,
            prune_density
        )
        G_original, A_sp = build_knn_graph(X, knn)  # returns G + sparse adjacency
        # Apply pruning
        if pruning_method == 'random':
            G_pruned, knn_graph, _ = prune_random(G_original, X, p=0.5)
        elif pruning_method == 'distance':
            G_pruned, knn_graph, _ = prune_distance(G_original, X, thresh=0.5)
        elif pruning_method == 'bisection':
            G_pruned, knn_graph, _ = prune_bisection(G_original, X, n=5)
        elif pruning_method == 'mst':
            G_pruned, knn_graph, _ = prune_mst(G_original, X, thresh=1.0)
        elif pruning_method == 'density':
            G_pruned, knn_graph, _ = prune_density(G_original, X, thresh=0.1)
        else:
            raise ValueError(f"Unknown pruning method: {pruning_method}")
        # Convert adjacency to dense format
        G = knn_graph.toarray()
        return G

    # If we want the all-pairs shortest path matrix
    if shortest_path:
        dist_mat = sp_shortest_path(knn_graph_sparse, directed=False)
        # Symmetrize
        dist_mat = 0.5 * (dist_mat + dist_mat.T)
        return dist_mat
    else:
        # Just return the adjacency
        return knn_graph_dense

def tsne_kernel(data, perplexity=30):
    """
    Manually build the t-SNE probability kernel from pairwise distances:
      P_{j|i} ~ exp(-||x_i - x_j||^2 / 2 sigma_i^2)
    Then symmetrize.
    """
    from src.embedding_algorithms import computeTSNEsigmas, computeTSNEkernel
    # The user’s code references these two functions, presumably in embedding_algorithms,
    # but not shown in your snippet. Assume they exist and work.
    from scipy.spatial.distance import pdist, squareform

    D2 = pdist(data, 'sqeuclidean')
    D2_square = squareform(D2)
    sigmas = computeTSNEsigmas(D2_square, perplexity)
    if sigmas is None or np.any(sigmas <= 0):
        return None
    myK = computeTSNEkernel(
        D2_square, sigmas,
        normalize=False, symmetrize=True, return_sparse=False
    )
    if myK is None or not np.allclose(myK, myK.T, atol=1e-8):
        return None
    return myK

def isomap_kernel(data, knn, metric='euclidean'):
    """
    Just the shortest-path distances from a knn graph, akin to isomap's main step.
    """
    if data.shape[0] < knn + 1:
        return None
    # Validate metric
    valid_metrics = {'euclidean','manhattan','minkowski','chebyshev','cosine'}
    if not (callable(metric) or metric in valid_metrics):
        logger.warning(f"[isomap_kernel] Invalid metric '{metric}'. Falling back to 'euclidean'.")
        metric = 'euclidean'

    from sklearn.neighbors import kneighbors_graph
    from scipy.sparse.csgraph import shortest_path
    sp_knn = kneighbors_graph(data, n_neighbors=knn, mode="distance",
                              include_self=False)
    dist_matrix = shortest_path(sp_knn, directed=False)
    dist_matrix = 0.5*(dist_matrix + dist_matrix.T)
    dist_matrix = connected_comp_helper(dist_matrix, data, connect=True)
    return dist_matrix

def kernel_dispatcher(name, **params):
    """
    Dispatches kernel functions based on the given name and parameters.
    e.g. name='gaussian', 'knn_shortest_path', 'isomap', 'tsne', 'IAN'
    """
    name = name.lower()
    if name.startswith('gaussian'):
        sigma = params.get('sigma')
        if sigma is None:
            raise ValueError("sigma parameter is required for Gaussian kernel.")
        return lambda X: compute_gaussian_kernel(X, sigma)

    elif name.startswith('knn_shortest_path'):
        knn = params.get('knn')
        if knn is None:
            raise ValueError("knn parameter is required for knn_shortest_path kernel.")
        shortest_path = params.get('shortest_path', False)
        pruning_method = params.get('pruning_method')
        metric = params.get('metric', 'euclidean')  # Default
        return lambda X: compute_knn_shortest_path_kernel(
            X, knn,
            shortest_path=shortest_path,
            pruning_method=pruning_method,
            metric=metric
        )

    elif name.startswith('isomap'):
        knn = params.get('knn')
        if knn is None:
            raise ValueError("knn parameter is required for isomap kernel.")
        metric = params.get('metric', 'euclidean')
        return lambda X: isomap_kernel(X, knn, metric=metric)

    elif name.startswith('tsne'):
        perplexity = params.get('perplexity', 30)
        return lambda X: tsne_kernel(X, perplexity=perplexity)

    elif name == 'ian':
        return ian_kernel

    else:
        raise ValueError(f"Unknown kernel name '{name}'. "
                         "Available: 'gaussian', 'knn_shortest_path', 'isomap', 'tsne', 'IAN'.")