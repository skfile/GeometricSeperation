"""
embedding_algorithms.py
-----------------------
Contains embedding methods such as Diffusion Maps, Isomap, t-SNE, Spectral Embeddings, LLE, and UMAP.
Also includes helper functions for locally linear embedding, etc.

TSNE Implementation adapted from https://github.com/dyballa/IAN.git.
"""

import numpy as np
import time       
import warnings
import scipy
import umap
import scipy as sp
from sklearn.manifold import (TSNE, SpectralEmbedding, Isomap as SkIsomap,
                              LocallyLinearEmbedding)
from sklearn.decomposition import KernelPCA
from sklearn.neighbors import NearestNeighbors, kneighbors_graph
from sklearn.manifold import smacof
from sklearn.utils.validation import check_non_negative, check_random_state

def ian_kernel(X):
    try:
        from ian.ian import IAN
        if X.shape[0] < 10:
            return None
        G, wG, optScales, disc_pts = IAN('exact', X, verbose=0, obj='greedy')
        return wG.toarray()
    except Exception as e:
        print(f"IAN kernel error: {e}")
        return None

############################################
# Fixing connectivity in distances or graphs
############################################

def _fix_connected_components_distance(dist_matrix, ambient_distances, component_labels, n_connected_components):
    """
    Fix disconnected components by adding minimal edges from ambient distances.
    dist_matrix: NxN adjacency or distance matrix
    ambient_distances: NxN full Euclidean distances
    component_labels: which component each node belongs to
    n_connected_components: number of components
    """
    dist_matrix = dist_matrix.copy()
    n = dist_matrix.shape[0]
    for c in range(n_connected_components):
        for d in range(c + 1, n_connected_components):
            c_nodes = np.where(component_labels == c)[0]
            d_nodes = np.where(component_labels == d)[0]
            submatrix = ambient_distances[np.ix_(c_nodes, d_nodes)]
            min_val = np.min(submatrix)
            min_pos = np.unravel_index(np.argmin(submatrix), submatrix.shape)
            c_node = c_nodes[min_pos[0]]
            d_node = d_nodes[min_pos[1]]
            dist_matrix[c_node, d_node] = min_val
            dist_matrix[d_node, c_node] = min_val
    return dist_matrix

##################################################
# Basic UMAP embedding from a shortest-path matrix
##################################################

def UMAP(A, n_neighbors, n_components, X=None):
    """
    UMAP embedding from a shortest-path distance matrix A.
    If A is not connected, tries to fix connectivity using X (the original data).
    """
    from scipy.sparse.csgraph import connected_components, shortest_path
    n_connected_components, component_labels = connected_components(A, directed=False)
    if n_connected_components > 1:
        if X is None:
            raise ValueError("The graph is not connected. Please provide the original data.")
        warnings.warn(f"The graph has {n_connected_components} components. Fixing connectivity.")
        ambient_distances = sp.spatial.distance.pdist(X, metric="euclidean")
        ambient_distances = sp.spatial.distance.squareform(ambient_distances)
        A = _fix_connected_components_distance(A, ambient_distances, component_labels, n_connected_components)

    distances = shortest_path(A, directed=False)
    assert np.allclose(distances, distances.T)
    umap_obj = umap.UMAP(n_neighbors=n_neighbors, n_components=n_components, metric='precomputed')
    Y = umap_obj.fit_transform(distances)
    return Y

##################################################
# Basic t-SNE from a shortest-path matrix
##################################################

def tsne(A, n_components, X=None):
    """
    t-SNE embedding from a shortest-path distance matrix A.
    If A is not connected, tries to fix connectivity using X (the original data).
    """
    from scipy.sparse.csgraph import connected_components, shortest_path
    n_connected_components, component_labels = connected_components(A, directed=False)
    if n_connected_components > 1:
        if X is None:
            raise ValueError("The graph is not connected. Please provide the original data.")
        warnings.warn(f"The graph has {n_connected_components} components. Fixing connectivity.")
        ambient_distances = sp.spatial.distance.pdist(X, metric="euclidean")
        ambient_distances = sp.spatial.distance.squareform(ambient_distances)
        A = _fix_connected_components_distance(A, ambient_distances, component_labels, n_connected_components)

    distances = shortest_path(A, directed=False)
    assert np.allclose(distances, distances.T)

    # Sklearn TSNE expects NxD data unless we specify metric='precomputed' differently.
    tsne_obj = TSNE(n_components=n_components, init='random', metric='precomputed')
    Y = tsne_obj.fit_transform(distances)
    return Y

############################################
# Basic Spectral Embedding
############################################

def spectral_embedding(A, n_components, affinity=True):
    """
    Spectral embedding from a given graph adjacency or distance matrix.
    If affinity=True, we interpret A as distances and build an RBF kernel.
    If affinity=False, we interpret A directly as adjacency.
    """
    if not affinity:
        W = A.copy()
    else:
        A_scaled = A / np.max(A)
        W = np.exp(-A_scaled**2)
        W[np.where(A == 0)] = 0
        np.fill_diagonal(W, 1)
    se = SpectralEmbedding(n_components=n_components, affinity='precomputed')
    Y = se.fit_transform(W)
    return Y

#################################################
# Locally Linear Embedding Helpers
#################################################

def locally_linear_embedding(distances, embedding, *, n_components, n_neighbors, reg=1e-3,
                             eigen_solver="auto", tol=1e-5, max_iter=200, random_state=None, n_jobs=None):
    """
    Locally Linear Embedding given a distance matrix and ambient embedding.
    ...
    (same code as before)
    """
    # your existing code here unchanged
    raise NotImplementedError("Same as your snippet, omitted for brevity")


############################################
# t-SNE Probability Helpers
############################################

from sklearn.manifold import TSNE
from ian.cutils import _binary_search_perplexity, get_tsne_sigmas
import warnings
from scipy.sparse import csr_matrix, issparse
from sklearn.utils.validation import check_non_negative, check_random_state
from src.graph_methods import build_knn_graph, prune_random, prune_distance, prune_mst, prune_bisection, prune_density

def my_joint_probabilities(sqdistances, desired_perplexity, precomputed_sigmas=None, verbose=False):
    """Compute joint probabilities p_ij from distances, and, 
     optionally, using precomputed sigmas (kernel scales).
     Code adapted from the scikit-learn package: https://scikit-learn.org/
     
     
    Parameters
    ----------
    sqdistances : square ndarray of shape (n_samples-by-n_samples) with squared distances
    desired_perplexity : float
        Desired perplexity of the joint probability distributions.
    verbose : int
        Verbosity level.
    Returns
    -------
    P : ndarray of shape (n_samples-by-n_samples)
        Condensed joint probability matrix.
    """
    # Compute conditional probabilities such that they approximately match
    # the desired perplexity
    sqdistances = sqdistances.astype(np.float32, copy=False)
    if precomputed_sigmas is None:
        conditional_P = _binary_search_perplexity(
            sqdistances, desired_perplexity, int(verbose)
        )
    else:
        EPSILON_DBL = 1e-8
        
        #instead of optimizing sigmas based on perplexity, use pre-computed ones
        conditional_P = np.zeros_like(sqdistances)
        for i in range(conditional_P.shape[0]):
            conditional_P[i] = np.exp(-sqdistances[i]/(2*precomputed_sigmas[i]**2))
            conditional_P[i,i] = 0
            sum_Pi = max(conditional_P[i].sum(),EPSILON_DBL)
            conditional_P[i] /= sum_Pi
            
    np.testing.assert_allclose(conditional_P.sum(1),1,1e-5,1e-5)
            
    assert np.all(np.isfinite(conditional_P)), "All probabilities should be finite"
    
    MACHINE_EPSILON = np.finfo(np.double).eps

    P = conditional_P + conditional_P.T
    sum_P = np.maximum(np.sum(P), MACHINE_EPSILON)
    P = np.maximum(squareform(P) / sum_P, MACHINE_EPSILON)
    return P

def my_joint_probabilities_nn(distances, desired_perplexity, precomputed_sigmas=None, verbose=False):
    """Compute joint probabilities p_ij from distances using just nearest
    neighbors.
    This method is approximately equal to _joint_probabilities. The latter
    is O(N), but limiting the joint probability to nearest neighbors improves
    this substantially to O(uN).
    Code adapted from the scikit-learn package: https://scikit-learn.org/
    Parameters
    ----------
    distances : sparse matrix of shape (n_samples, n_samples)
        Distances of samples to its n_neighbors nearest neighbors. All other
        distances are left to zero (and are not materialized in memory).
        Matrix should be of CSR format.
    desired_perplexity : float
        Desired perplexity of the joint probability distributions.
    verbose : int
        Verbosity level.
    Returns
    -------
    P : sparse matrix of shape (n_samples, n_samples)
        Condensed joint probability matrix with only nearest neighbors. Matrix
        will be of CSR format.
    """

    t0 = time.time()
    # Compute conditional probabilities such that they approximately match
    # the desired perplexity
    distances.sort_indices()
    n_samples = distances.shape[0]
    distances_data = distances.data.reshape(n_samples, -1)
    distances_data = distances_data.astype(np.float32, copy=False)

    if precomputed_sigmas is None:
        conditional_P = _binary_search_perplexity(
            distances_data, desired_perplexity, int(verbose)
        )
    else:
        EPSILON_DBL = 1e-8
        
        #instead of optimizing sigmas based on perplexity, use pre-computed ones
        conditional_P = np.zeros_like(distances_data)
        for i in range(conditional_P.shape[0]):
            conditional_P[i] = np.exp(-distances_data[i]/(2*precomputed_sigmas[i]**2))
            sum_Pi = max(conditional_P[i].sum(),EPSILON_DBL)
            conditional_P[i] /= sum_Pi      
            
    np.testing.assert_allclose(conditional_P.sum(1),1,1e-5,1e-5)
            
    assert np.all(np.isfinite(conditional_P)), "All probabilities should be finite"

    # Symmetrize the joint probability distribution using sparse operations
    P = csr_matrix(
        (conditional_P.ravel(), distances.indices, distances.indptr),
        shape=(n_samples, n_samples),
    )
    P = P + P.T
    
    MACHINE_EPSILON = np.finfo(np.double).eps

    # Normalize the joint probability distribution
    sum_P = np.maximum(P.sum(), MACHINE_EPSILON)
    P /= sum_P

    assert np.all(np.abs(P.data) <= 1.0)
    if verbose >= 2:
        duration = time.time() - t0
        print("[t-SNE] Computed conditional probabilities in {:.3f}s".format(duration))
    return P

def my_tsne_fit(tsne, distances, precomputed_sigmas=None, skip_num_points=0, verbose=False):
    
    """Code adapted from the scikit-learn package: https://scikit-learn.org/"""
    
    assert tsne.metric == "precomputed"
    #if tsne.metric == "precomputed":
    if isinstance(tsne.init, str) and tsne.init == "pca":
        raise ValueError(
            'The parameter init="pca" cannot be used with metric="precomputed".'
        )
        
    if tsne.method not in ["barnes_hut", "exact"]:
        raise ValueError("'method' must be 'barnes_hut' or 'exact'")
    if tsne.angle < 0.0 or tsne.angle > 1.0:
        raise ValueError("'angle' must be between 0.0 - 1.0")
    
    if tsne.learning_rate == "warn":
        # See issue #18018
        warnings.warn(
            "The default learning rate in TSNE will change "
            "from 200.0 to 'auto' in 1.2.",
            FutureWarning,
        )
        tsne._learning_rate = 200.0
    else:
        tsne._learning_rate = tsne.learning_rate
    if tsne._learning_rate == "auto":
        # See issue #18018
        tsne._learning_rate = distances.shape[0] / tsne.early_exaggeration / 4
        tsne._learning_rate = np.maximum(tsne._learning_rate, 50)
    else:
        if not (tsne._learning_rate > 0):
            raise ValueError("'learning_rate' must be a positive number or 'auto'.")
    tsne.learning_rate_ = tsne._learning_rate #for compatibility with different versions of scikit-learn

    if tsne.method == "barnes_hut":
        distances = tsne._validate_data(
            distances,
            accept_sparse=["csr"],
            ensure_min_samples=2,
            dtype=[np.float32, np.float64],
        )
    else:
        distances = tsne._validate_data(
            distances, accept_sparse=["csr", "csc", "coo"], dtype=[np.float32, np.float64]
        )

    check_non_negative(
        distances,
        "TSNE.fit(). With metric='precomputed', distances "
        "should contain positive distances.",
    )

    if tsne.method == "exact" and issparse(distances):
        raise TypeError(
            'TSNE with method="exact" does not accept sparse '
            'precomputed distance matrix. Use method="barnes_hut" '
            "or provide the dense distance matrix."
        )

    if tsne.method == "barnes_hut" and tsne.n_components > 3:
        raise ValueError(
            "'n_components' should be inferior to 4 for the "
            "barnes_hut algorithm as it relies on "
            "quad-tree or oct-tree."
        )
    random_state = check_random_state(tsne.random_state)

    if tsne.early_exaggeration < 1.0:
        raise ValueError(
            "early_exaggeration must be at least 1, but is {}".format(
                tsne.early_exaggeration
            )
        )

    if tsne.n_iter < 250:
        raise ValueError("n_iter should be at least 250")


    n_samples = distances.shape[0]
    
    if n_samples == distances.shape[1]: #sq distance matrix ("exact" method)
        P = my_joint_probabilities(distances, tsne.perplexity, precomputed_sigmas, verbose)
    else: #knn approximation
        P = my_joint_probabilities_nn(distances, tsne.perplexity, precomputed_sigmas, verbose)
    if tsne.method == "barnes_hut":
        P = sp.sparse.csr_matrix(squareform(P))
        val_P = P.data.astype(np.float32, copy=False)
        neighbors = P.indices.astype(np.int64, copy=False)
        indptr = P.indptr.astype(np.int64, copy=False)
      
    if isinstance(tsne.init, np.ndarray):
        distances_embedded = tsne.init
    elif tsne.init == "random":
        # The embedding is initialized with iid samples from Gaussians with
        # standard deviation 1e-4.
        distances_embedded = 1e-4 * random_state.randn(n_samples, tsne.n_components).astype(
            np.float32
        )
    else:
        raise ValueError("'init' must be 'random', or a numpy array")

    # Degrees of freedom of the Student's t-distribution. The suggestion
    # degrees_of_freedom = n_components - 1 comes from
    # "Learning a Parametric Embedding by Preserving Local Structure"
    # Laurens van der Maaten, 2009.
    degrees_of_freedom = max(tsne.n_components - 1, 1)

    # Set internal variables expected by _tsne()
    tsne._EXPLORATION_N_ITER = 250 if tsne.method == 'barnes_hut' else 100
    tsne._N_ITER_CHECK = 50
    tsne._max_iter = tsne.n_iter

    tsne.embedding_ = tsne._tsne(
        P,
        degrees_of_freedom,
        n_samples,
        X_embedded=distances_embedded,
        neighbors=None,
        skip_num_points=skip_num_points)
    
    return P

def my_tsne_fit_transform(tsne, sqdistances, precomputed_sigmas=None, recompute=True, skip_num_points=0, verbose=False):
    """ Allows one to fit a TSNE object (from sklearn.manifold) using custom sigmas (kernel scales)"""
    
    if recompute or not hasattr(tsne, 'embedding_'):
        my_tsne_fit(tsne, sqdistances, precomputed_sigmas, skip_num_points, verbose)
    return tsne.embedding_

def spectral_init_from_umap(graph, m, random_state=123456789):
    
    """ Computes a spectral embedding (akin to Laplacian eigenmaps) for initializing t-SNE 
    (analogously to what is done in UMAP, for a fair comparison b/w the two).
    The code below was adapted from the `spectral_layout` function in the umap-learn package. """

    n_components, labels = sp.sparse.csgraph.connected_components(graph)

    if n_components > 1:
        print(f'Found more than 1 connected component ({n_components}).')
        print('UMAP does additional pre-processing in this case.')
        print("Please run UMAP's spectral_layout for a fair comparison.")
        
    diag_data = np.asarray(graph.sum(axis=0))

    # Normalized Laplacian
    I = sp.sparse.identity(graph.shape[0], dtype=np.float64)
    diag_data[diag_data == 0] = 1e-10  # Avoid division by zero
    D = sp.sparse.spdiags(
        1.0 / np.sqrt(diag_data), 0, graph.shape[0], graph.shape[0]
    )
    L = I - D * graph * D

    k = m + 1
    eigenvectors, eigenvalues, _ = sp.sparse.linalg.svds(L, k=k, return_singular_vectors='u',random_state=random_state)

    order = np.argsort(eigenvalues)[1:k]
    return eigenvectors[:, order]

def computeTSNEsigmas(sqdistances, desired_perplexity, verbose=False):

    # Compute conditional probabilities such that they approximately match
    # the desired perplexity
    n_samples = sqdistances.shape[0]

    if sp.sparse.issparse(sqdistances):
        D2.sort_indices()
        n_samples = D2.shape[0]
        distances_data = D2.data.reshape(n_samples, -1)
        distances_data = distances_data
        distances_data = distances_data.astype(np.float32, copy=False)
    else:
        distances_data = sqdistances.astype(np.float32, copy=False)
    
    return get_tsne_sigmas(distances_data, desired_perplexity, verbose)

def computeTSNEkernel(D2, sigmas, normalize=True, symmetrize=True, return_sparse=True):
    
    """ Computes the t-SNE kernel from a dense square matrix of squared distances and 
    precomputed sigmas """

    n_samples = D2.shape[0]

    if sp.sparse.issparse(D2):
        D2.sort_indices()
        n_samples = D2.shape[0]
        distances_data = D2.data.reshape(n_samples, -1)
        distances_data = distances_data
        distances_data = distances_data.astype(np.float32, copy=False)
    else:
        distances_data = D2.astype(np.float32, copy=False)

    conditional_P = np.zeros_like(distances_data, dtype=np.float32)
    for i in range(conditional_P.shape[0]):
        conditional_P[i] = np.exp(-distances_data[i]/(2*sigmas[i]**2))
        if not sp.sparse.issparse(D2):
            conditional_P[i,i] = 0
        if normalize:
            sum_Pi = max(conditional_P[i].sum(),1e-8)
            conditional_P[i] /= sum_Pi

    if sp.sparse.issparse(D2):
        conditional_P = csr_matrix( (conditional_P.ravel(), D2.indices, D2.indptr),
                        shape=(n_samples, n_samples))
    if symmetrize:
        conditional_P = (conditional_P + conditional_P.T)
        if not normalize:
            conditional_P *= .5

    if normalize:
        # Normalize the joint probability distribution
        sum_P = np.maximum(conditional_P.sum(), np.finfo(D2.dtype).eps)
        conditional_P /= sum_P
        assert np.all(np.abs(conditional_P.data) <= 1)

    if return_sparse:
        conditional_P = sp.sparse.csr_matrix(conditional_P)

    return conditional_P

#################################################
# Diffusion Map from kernel K
#################################################

def diffusionMapFromK(K, n_components=2, alpha=0.5):
    """
    Compute diffusion map from a given NxN kernel matrix K using alpha-normalization.
    K must be symmetric, nonnegative.
    """
    if sp.sparse.issparse(K):
        K = K.toarray()

    N = K.shape[0]
    eps = np.finfo(K.dtype).eps

    # Symmetry check
    if not np.allclose(K, K.T):
        raise ValueError("Kernel matrix is not symmetric.")

    # Optional alpha-normalization
    if alpha > 0:
        D = K.sum(axis=1, keepdims=True)
        denominator = np.power(D @ D.T, alpha)
        denominator[denominator <= eps] = eps
        K = K / denominator

    # Threshold small entries
    K[K < 1e-5] = 0.0

    sqrtD = np.sqrt(K.sum(axis=1, keepdims=True)) + eps
    Ms = K / (sqrtD @ sqrtD.T)
    Ms = 0.5 * (Ms + Ms.T)  # enforce symmetry again

    k = n_components + 1
    if k >= N:
        k = N-1
    try:
        Ms_sparse = sp.sparse.csc_matrix(Ms)
        U, lambdas, _ = sp.sparse.linalg.svds(Ms_sparse, k=k, tol=1e-5, maxiter=5000)
        # The result from svds is not necessarily sorted in descending order of eigenvalues
        idx = np.argsort(-lambdas)
        lambdas, U = lambdas[idx], U[:, idx]
    except sp.sparse.linalg.ArpackNoConvergence:
        print("ARPACK failed for sparse eigendecomposition. Falling back to dense.")
        lambdas, U = np.linalg.eigh(Ms)
        idx = np.argsort(-lambdas)
        lambdas, U = lambdas[idx], U[:, idx]
        if k < N:
            lambdas = lambdas[:k]
            U = U[:, :k]

    # Avoid dividing by 0:
    for i in range(len(sqrtD)):
        if sqrtD[i, 0] < 1e-15:
            sqrtD[i, 0] = 1e-15
    Psi = U / sqrtD  # standard approach in diffusion maps
    diffmap = Psi * lambdas  # time t=1

    # Omit the trivial top eigenvector => keep columns [1,2,...,n_components]
    return diffmap[:, 1:(n_components+1)], lambdas[1:(n_components+1)]

def compute_embedding(X_or_graph,
                      method='diffusionmap',
                      n_components=2,
                      alpha=0.5,
                      **kwargs):
    """
    A unified embedding dispatcher.

    If method == 'diffusionmap':
      (A) if X_or_graph is NxD => ian_kernel => K => diffusionMapFromK
      (B) if X_or_graph is NxN => assume it's already a kernel => diffusionMapFromK

    For all other methods (umap, tsne, spectral, isomap, lle, pca):
       interpret X_or_graph as NxD raw coords,
       call the standard library approach.

    Returns
    -------
    emb : ndarray shape (N, n_components)
    extra : anything else needed (e.g. eigenvalues) or None
    """
    method = method.lower()

    # 1) Diffusion Map => custom kernel with IAN if NxD
    if method == 'diffusionmap':
        if X_or_graph.shape[0] == X_or_graph.shape[1]:
            # NxN kernel
            K = X_or_graph
        else:
            # NxD => use IAN kernel
            K = ian_kernel(X_or_graph)
            if K is None:
                warnings.warn("IAN kernel failed or returned None.")
                return None, None

        emb, eigvals = diffusionMapFromK(K, n_components=n_components, alpha=alpha)
        return emb, eigvals

    # 2) UMAP => scikit-learn's umap library, NxD raw coords
    elif method == 'umap':
        # interpret X_or_graph as NxD
        # if user wants n_neighbors in kwargs
        if 'n_neighbors' not in kwargs:
            kwargs['n_neighbors'] = 15
        reducer = umap.UMAP(n_neighbors=kwargs['n_neighbors'], n_components=n_components)
        emb = reducer.fit_transform(X_or_graph)  # NxD
        return emb, None

    # 3) t-SNE => scikit TSNE on NxD
    elif method == 'tsne':
        tsne_model = TSNE(n_components=n_components, init='random', **kwargs)
        emb = tsne_model.fit_transform(X_or_graph)  # NxD
        return emb, None

    # 4) spectral => scikit's SpectralEmbedding on NxD
    elif method == 'spectral':
        # Use standard 'nearest_neighbors' or 'rbf' affinity => NxD
        se = SpectralEmbedding(n_components=n_components,
                               affinity='nearest_neighbors', **kwargs)
        emb = se.fit_transform(X_or_graph)  # NxD
        return emb, None

    # 5) isomap => scikit's Isomap on NxD
    elif method == 'isomap':
        # Increase n_neighbors to help avoid disconnected components
        n_neighbors = kwargs.get('n_neighbors', 15)  # Increased from 5 to 15
        
        # For higher dimensional data, use even more neighbors
        if X_or_graph.shape[1] > 10:
            n_neighbors = max(n_neighbors, min(30, X_or_graph.shape[0] // 4))
        
        # Check for connectivity before running Isomap
        from scipy.sparse import csr_matrix
        from scipy.sparse.csgraph import connected_components
        knn_graph = kneighbors_graph(X_or_graph, n_neighbors=n_neighbors, 
                                     mode='distance', include_self=False)
        n_components, _ = connected_components(knn_graph, directed=False)
        
        if n_components > 1:
            warnings.warn(f"Fixing {n_components} disconnected components for Isomap by increasing neighbors")
            # Try with more neighbors to get a connected graph
            n_neighbors = min(n_neighbors * 2, X_or_graph.shape[0] // 2)
        
        iso = SkIsomap(n_neighbors=n_neighbors, n_components=n_components)
        emb = iso.fit_transform(X_or_graph)  # NxD
        return emb, None

    # 6) lle => scikit's LocallyLinearEmbedding on NxD
    elif method == 'lle':
        lle = LocallyLinearEmbedding(n_neighbors=kwargs.get('n_neighbors', 12),
                                     n_components=n_components, **kwargs)
        emb = lle.fit_transform(X_or_graph)
        return emb, None

    # 7) pca => scikit's PCA on NxD
    elif method == 'pca':
        pca = PCA(n_components=n_components)
        emb = pca.fit_transform(X_or_graph)
        return emb, None


    elif method == "knn_embedding":
        G, A_knn = build_knn_graph(X_or_graph, kwargs.get('n_neighbors', 32))
        K = A_knn.toarray()
        emb, eigvals = diffusionMapFromK(K, n_components=n_components, alpha=alpha)
        return emb, eigvals

    elif method == "knn_distance_embedding":
        G, A_knn = build_knn_graph(X_or_graph, kwargs.get('n_neighbors', 32))
        _, A_pruned, _ = prune_distance(G, X_or_graph, kwargs.get('distance_thresh', 1.0))
        K = A_pruned.toarray()
        emb, eigvals = diffusionMapFromK(K, n_components=n_components, alpha=alpha)
        return emb, eigvals

    elif method == "knn_mst_embedding":
        G, A_knn = build_knn_graph(X_or_graph, kwargs.get('n_neighbors', 32))
        _, A_pruned, _ = prune_mst(G, X_or_graph, kwargs.get('mst_thresh', 5.0))
        K = A_pruned.toarray()
        emb, eigvals = diffusionMapFromK(K, n_components=n_components, alpha=alpha)
        return emb, eigvals

    elif method == "knn_bisection_embedding":
        G, A_knn = build_knn_graph(X_or_graph, kwargs.get('n_neighbors', 32))
        _, A_pruned, _ = prune_bisection(G, X_or_graph, kwargs.get('bisection_n', 5))
        K = A_pruned.toarray()
        emb, eigvals = diffusionMapFromK(K, n_components=n_components, alpha=alpha)
        return emb, eigvals
    
    elif method == "knn_density_embedding":
        G, A_knn = build_knn_graph(X_or_graph, kwargs.get('n_neighbors', 32))
        _, A_pruned, _ = prune_density(G, X_or_graph, kwargs.get('kde_thresh', 0.1))
        K = A_pruned.toarray()
        emb, eigvals = diffusionMapFromK(K, n_components=n_components, alpha=alpha)
        return emb, eigvals

    else:
        raise ValueError(f"Unknown embedding method '{method}'.")
