#!/usr/bin/env python3

"""
mesh_sampling_revised.py
---------------------------
Generates data sets (union-of-shapes, simplex-based Gaussians, or MNIST) plus
subsamples (uniform or biased). Builds adjacency using multiple possible methods.

Key changes/fixes:
 - The code is largely the same, but we ensure offsets are used properly.
 - We unify adjacency construction and ensure it is block-diagonal for union-of-shapes.
"""

import numpy as np
import logging
from typing import List, Dict, Any, Optional, Callable, Tuple
from scipy.spatial import ConvexHull, KDTree
from scipy.spatial.distance import pdist, cdist
from scipy.sparse import block_diag
from sklearn.preprocessing import StandardScaler
from scipy.sparse.csgraph import shortest_path

from src.parallel_utils import parallel_context
from src.mnist_dataset import load_mnist_data, ensure_all_mnist_classes
from src.scRNA_dataset import load_scRNA_data
try:
    import pynndescent
    _HAS_PYNNDESCENT = True
except ImportError:
    _HAS_PYNNDESCENT = False

logger = logging.getLogger("mesh_opt_revised")
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
logger.addHandler(handler)

############################################################
# Potential Registry
############################################################

POTENTIAL_REGISTRY = {}

def potential_distance_origin(x: np.ndarray, scale: float = 1.0) -> float:
    return (np.linalg.norm(x) ** 2) / scale

def potential_offset(x: np.ndarray, offset=0.0):
    y = x.copy()
    y += offset
    return np.linalg.norm(y)

POTENTIAL_REGISTRY["distance_origin"] = potential_distance_origin
POTENTIAL_REGISTRY["offset"] = potential_offset

def get_potential_func(name: str, params: dict) -> Callable[[np.ndarray], float]:
    if name not in POTENTIAL_REGISTRY:
        raise ValueError(f"Unknown potential '{name}'. Known: {list(POTENTIAL_REGISTRY.keys())}")
    basef = POTENTIAL_REGISTRY[name]

    def wrapper(pt: np.ndarray) -> float:
        return basef(pt, **params)

    return wrapper

############################################################
# HD shapes
############################################################

def generate_hd_sphere(dim: int, radius: float, n_points: Optional[int], rng: np.random.Generator,
                       offset: Optional[np.ndarray] = None) -> np.ndarray:
    if n_points is None:
        n_points = 1000  # fallback or auto-sample heuristic
    X = rng.normal(size=(n_points, dim))
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms[norms < 1e-15] = 1e-15
    X /= norms
    X *= radius
    if offset is not None:
        offset = np.asarray(offset).reshape(1, -1)  # Ensure offset is a 2D array with shape (1, dim)
        X = X + offset  # Broadcasting will now work correctly
    return X

def generate_hd_ellipsoid(dim: int, axes: List[float], n_points: Optional[int], rng: np.random.Generator,
                          offset: Optional[np.ndarray] = None) -> np.ndarray:
    if len(axes) != dim:
        raise ValueError("Ellipsoid axes length does not match dimension.")
    if n_points is None:
        n_points = 1000
    # sample from uniform directions
    X = rng.normal(size=(n_points, dim))
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms[norms < 1e-15] = 1e-15
    X /= norms
    # radial scaling
    r = rng.uniform(0, 1, (n_points, 1)) ** (1.0 / dim)
    X *= (r * np.array(axes))
    if offset is not None:
        offset = np.asarray(offset).reshape(1, -1)  # Ensure offset is a 2D array with shape (1, dim)
        X = X + offset  # Broadcasting will now work correctly
    return X

def generate_hd_torus(dim: int, major_radius: float, minor_radius: float, n_points: Optional[int],
                      rng: np.random.Generator, offset: Optional[np.ndarray] = None) -> np.ndarray:
    if dim < 2:
        raise ValueError("Torus dimension must be at least 2.")
    if n_points is None:
        n_points = 1000
    # Basic param for 2D ring in the first 2 dims, plus random circle for higher dims
    angles = rng.uniform(0, 2*np.pi, size=n_points)
    circle2d = np.zeros((n_points, 2))
    circle2d[:, 0] = (major_radius + minor_radius * np.cos(angles)) * 1.0
    circle2d[:, 1] = (minor_radius * np.sin(angles)) * 1.0

    if dim > 2:
        remain = dim - 2
        Y = rng.normal(size=(n_points, remain))
        # scale each row to lie within radius=minor_radius
        norms = np.linalg.norm(Y, axis=1, keepdims=True)
        norms[norms < 1e-15] = 1e-15
        Y /= norms
        radius_vals = rng.uniform(0, 1, size=(n_points, 1)) ** (1.0 / remain)
        Y *= (radius_vals * minor_radius)
        X = np.hstack((circle2d, Y))
    else:
        X = circle2d

    if offset is not None:
        offset = np.asarray(offset).reshape(1, -1)  # Ensure offset is a 2D array with shape (1, dim)
        X = X + offset  # Broadcasting will now work correctly
    return X

def generate_hd_hyperbolic(dim: int, radius: float, curvature: float, n_points: Optional[int], 
                          rng: np.random.Generator, offset: Optional[np.ndarray] = None) -> np.ndarray:
    """
    Generate points in the Poincaré disk model of hyperbolic space.
    
    Args:
        dim: Dimension of the hyperbolic space
        radius: Maximum radius within the Poincaré disk (must be < 1.0)
        curvature: Curvature parameter (negative values for hyperbolic space)
        n_points: Number of points to generate
        rng: Random number generator
        offset: Optional offset vector
        
    Returns:
        Array of points in hyperbolic space
    """
    if n_points is None:
        n_points = 1000
    if radius >= 1.0:
        radius = 0.99  # Ensure points stay within the Poincaré disk
    
    # Generate random directions uniformly
    X = rng.normal(size=(n_points, dim))
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms[norms < 1e-15] = 1e-15
    X /= norms
    
    # Generate radii according to hyperbolic volume element
    # For the Poincaré disk model, points should be more densely packed near the boundary
    r = rng.uniform(0, 1, size=(n_points, 1)) ** (1.0 / dim)
    # Transform for hyperbolic distribution (more points near boundary)
    r = radius * r / (1 + (1 - radius) * r)
    
    # Scale directions by radii
    X *= r
    
    if offset is not None:
        # Ensure offset is a 2D array with shape (1, dim)
        offset = np.asarray(offset).reshape(1, -1)
        # In hyperbolic space, addition is more complex
        # For simplicity, we'll just add the offset in Euclidean space
        # Fix the broadcasting dimension
        if isinstance(offset, np.ndarray) and offset.ndim == 1:
            X_offset = X + offset[:, np.newaxis]  # Reshape offset to (N,1) for proper broadcasting
        else:
            X_offset = X + offset 
        # Ensure points remain within the Poincaré disk after offset
        norms = np.linalg.norm(X, axis=1, keepdims=True)
        mask = norms >= 1.0
        if np.any(mask):
            X[mask.flatten()] *= (0.99 / norms[mask])
    
    return X

############################################################
# Simplex Gaussians
############################################################

def generate_simplex_vertices(dim: int, scale: float = 1.0) -> np.ndarray:
    if dim < 1:
        raise ValueError("Dimension must be >= 1.")
    # Standard approach: start with the identity trick
    # e.g. in 2D => an equilateral triangle with side ~ scale, etc.
    vertices = np.zeros((dim + 1, dim))
    for i in range(1, dim + 1):
        vertices[i - 1, i - 1] = 1.0
    vertices[-1, :] = -1.0/dim
    # measure the edge length
    from scipy.spatial.distance import pdist
    edge_length = np.mean(pdist(vertices))
    # scale so that the average edge is ~ `scale`. You can do it differently if desired.
    if edge_length < 1e-15:
        edge_length = 1.0
    factor = scale / edge_length
    vertices *= factor
    return vertices

def build_adjacency_knn(X: np.ndarray, k: int, approximate: bool = False,
                        k_max: int = 1000) -> np.ndarray:
    """
    Create NxN adjacency by k-NN distances.
    """
    N = X.shape[0]
    adjacency = np.zeros((N, N), dtype=float)

    if k <= 0 or k > N - 1:
        k = min(max(1, k), N - 1)
        logger.info(f"build_adjacency_knn: adjusted k to {k} for dataset size {N}.")

    if approximate and _HAS_PYNNDESCENT:
        try:
            index = pynndescent.NNDescent(X, n_neighbors=k + 1, random_state=42, n_jobs=-1)
            neighbors, dists = index.neighbor_graph
            for i in range(N):
                for distv, idxv in zip(dists[i, 1:], neighbors[i, 1:]):
                    adjacency[i, idxv] = distv
                    adjacency[idxv, i] = distv
        except Exception as e:
            logger.warning(f"pynndescent failed: {e}. Reverting to exact.")
            approximate = False

    if not approximate:
        tree = KDTree(X)
        dists, idxs = tree.query(X, k=k+1)
        for i in range(N):
            for distv, idxv in zip(dists[i,1:], idxs[i,1:]):
                adjacency[i, idxv] = distv
                adjacency[idxv, i] = distv

    return adjacency

def generate_simplex_gaussian_mixture(cfg: dict, rng: np.random.Generator) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generates a mixture of Gaussians with means placed on a scaled simplex in `dim` dimensions.
    Each Gaussian's adjacency is built separately => block diagonal.
    """
    n_points_per_gauss = cfg["n_points_per_gauss"]
    dim = cfg["dims"]
    sigma_max_list = cfg["sigma_max_list"]
    offset = cfg.get("offset", 1.0)

    n_components = len(n_points_per_gauss)
    if len(sigma_max_list) != n_components:
        raise ValueError("sigma_max_list must match n_points_per_gauss in length.")

    # We create the set of dim+1 vertices
    full_simplex_vertices = generate_simplex_vertices(dim, scale=offset)
    num_vertices = dim + 1

    # If we have more Gaussians than vertices, we pick with replacement
    if n_components <= num_vertices:
        chosen_indices = rng.choice(num_vertices, size=n_components, replace=False)
    else:
        chosen_indices = rng.choice(num_vertices, size=n_components, replace=True)

    blocks = []
    adj_blocks = []
    labels = []
    for i in range(n_components):
        Ni = n_points_per_gauss[i]
        si = sigma_max_list[i]
        mu = full_simplex_vertices[chosen_indices[i]]
        cov = (si ** 2) * np.eye(dim)
        block = rng.multivariate_normal(mu, cov, size=Ni)
        blocks.append(block)

        adjacency = build_adjacency_knn(block, k=5)
        adj_blocks.append(adjacency)

        labels.extend([i]*Ni)

    X_full = np.vstack(blocks)
    from scipy.sparse import block_diag
    # keep adjacency sparse for efficient graph algorithms
    A_full = block_diag(adj_blocks)

    return X_full, A_full, np.array(labels, dtype=int)

############################################################
# Adjacency building for union-of-shapes
############################################################

def build_adjacency_threshold(X: np.ndarray, radius: float, parallelize: bool = False) -> np.ndarray:
    from joblib import Parallel, delayed
    N = X.shape[0]
    adjacency = np.zeros((N, N), dtype=float)
    tree = KDTree(X)

    def process_point(i: int):
        idxs = tree.query_ball_point(X[i], r=radius)
        conns = {}
        for j in idxs:
            if j == i:
                continue
            dist_ij = np.linalg.norm(X[i] - X[j])
            conns[j] = dist_ij
        return i, conns

    if parallelize:
        with parallel_context(n_jobs=None) as n_jobs:
            results = Parallel(n_jobs=n_jobs)(
                delayed(process_point)(i) for i in range(len(X))
            )
    else:
        results = [process_point(i) for i in range(N)]

    for i, conns in results:
        for j, distij in conns.items():
            adjacency[i, j] = distij
            adjacency[j, i] = distij

    return adjacency

def build_adjacency_convexhull(X: np.ndarray, iterative_passes: int = 1) -> np.ndarray:
    """
    Very rough approach: we take random subsets -> find hull edges -> place them in adjacency.
    """
    N = X.shape[0]
    adjacency = np.zeros((N, N), dtype=float)
    rng = np.random.default_rng(42)
    for pass_i in range(iterative_passes):
        sub_size = min(N, 1000)
        sub_idx = rng.choice(N, sub_size, replace=False)
        subX = X[sub_idx]
        try:
            hull = ConvexHull(subX, qhull_options="QJ i")
            for simplex in hull.simplices:
                vA = sub_idx[simplex[0]]
                vB = sub_idx[simplex[1]]
                d = np.linalg.norm(X[vA] - X[vB])
                adjacency[vA, vB] = d
                adjacency[vB, vA] = d
        except Exception as e:
            logger.warning(f"ConvexHull pass {pass_i+1} => {e}")
    return adjacency

def build_shape_adjacency(X: np.ndarray, adjacency_method: str, adjacency_param: dict) -> np.ndarray:
    if adjacency_method == "knn":
        k = adjacency_param.get("k", 5)
        approximate = adjacency_param.get("approximate", False)
        k_max = adjacency_param.get("k_max", 1000)
        return build_adjacency_knn(X, k, approximate, k_max)
    elif adjacency_method == "threshold":
        r = adjacency_param.get("radius", 1.0)
        do_parallel = adjacency_param.get("parallelize", False)
        return build_adjacency_threshold(X, r, do_parallel)
    elif adjacency_method == "convexhull":
        passes = adjacency_param.get("iterative_passes", 1)
        return build_adjacency_convexhull(X, passes)
    else:
        raise ValueError(f"Unknown adjacency_method={adjacency_method}")

def build_full_adjacency(adjacency_list: List[np.ndarray]) -> np.ndarray:
    if not adjacency_list:
        # empty adjacency
        return block_diag([])
    # return sparse block-diagonal adjacency
    return block_diag(adjacency_list)

def generate_union_data_revised(shapes: List[dict], adjacency_method: str, adjacency_param: dict,
                                offset: float, rng: np.random.Generator) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    For each shape, we generate points in the shape, build adjacency, then combine block-diagonal.
    If multiple shapes => place them at different offsets in a simplex arrangement if you want,
    or do something simpler. Below we do a simple approach: if k>1, we embed them on a (k-1)-D simplex.
    """
    k = len(shapes)
    if k == 0:
        return np.array([]), np.array([[]]), np.array([])
    # If only 1 shape, offset=0
    if k == 1:
        dim = shapes[0]["dim"]
        # single shape => no real shift
        simplex_vertices = np.zeros((1, dim))
    else:
        dim_for_offset = max([s["dim"] for s in shapes])
        # we embed them in a (k-1)-D simplex, then possibly pad to shape's dimension
        # For simplicity we'll just do a high-level approach:
        # generate (k-1)+1 = k vertices in dimension k-1
        # scale by 'offset'
        if (k-1) < 1:
            # fallback
            logger.warning("Trying to union multiple shapes but k-1 <1 => fallback no offset.")
            simplex_vertices = np.zeros((k, dim_for_offset))
        else:
            base_simplex = generate_simplex_vertices(k-1, scale=offset)  # shape: (k, k-1)
            # pad to dimension=dim_for_offset
            if (k-1) < dim_for_offset:
                # pad
                padded = np.zeros((k, dim_for_offset))
                padded[:, : (k-1)] = base_simplex
                simplex_vertices = padded
            else:
                # exactly matches or bigger
                simplex_vertices = base_simplex

    X_chunks = []
    adjacency_chunks = []
    labels = []
    for i, shape in enumerate(shapes):
        stype = shape["shape_type"]
        d = shape["dim"]
        n = shape.get("n_points", 1000)
        
        # Get center for this shape and ensure it's the right shape
        if i < len(simplex_vertices):
            # Extract only the dimensions we need for this shape
            center = np.zeros(d)
            center_slice = simplex_vertices[i][:d]
            center[:len(center_slice)] = center_slice
        else:
            center = np.zeros(d)
            
        if stype == "sphere_hd":
            rad = shape["radius"]
            block = generate_hd_sphere(d, rad, n, rng, offset=center)
        elif stype == "ellipsoid_hd":
            axes = shape["axes"]
            block = generate_hd_ellipsoid(d, axes, n, rng, offset=center)
        elif stype == "torus_hd":
            maj = shape["major_radius"]
            minr = shape["minor_radius"]
            block = generate_hd_torus(d, maj, minr, n, rng, offset=center)
        elif stype == "hyperbolic_hd":
            rad = shape.get("radius", 0.9)  # Default to 0.9 to stay in Poincaré disk
            curv = shape.get("curvature", -1.0)  # Default curvature for hyperbolic space
            block = generate_hd_hyperbolic(d, rad, curv, n, rng, offset=center)
        else:
            raise ValueError(f"Unknown shape_type {stype}")

        # If shape dim < max needed dimension, pad
        max_dim = max([sh["dim"] for sh in shapes])
        if block.shape[1] < max_dim:
            pad = max_dim - block.shape[1]
            block = np.hstack((block, np.zeros((block.shape[0], pad))))

        adjacency = build_shape_adjacency(block, adjacency_method, adjacency_param)
        X_chunks.append(block)
        adjacency_chunks.append(adjacency)
        labels.extend([i]*block.shape[0])

    X_full = np.vstack(X_chunks)
    A_full = build_full_adjacency(adjacency_chunks)
    labels = np.array(labels, dtype=int)
    return X_full, A_full, labels

############################################################
# Sampling
############################################################

def sample_indices_uniform(V: int, n: int, rng: np.random.Generator) -> np.ndarray:
    if n > V:
        n = V
    return rng.choice(V, size=n, replace=False)

def sample_indices_biased(X: np.ndarray, n: int, potfunc: Callable[[np.ndarray], float],
                          rng: np.random.Generator) -> np.ndarray:
    V = X.shape[0]
    if n > V:
        n = V
    potvals = np.apply_along_axis(potfunc, 1, X)
    log_w = -potvals
    max_log_w = np.max(log_w)
    log_w -= max_log_w
    w = np.exp(log_w)
    ssum = w.sum()
    if ssum < 1e-15:
        logger.warning("Sum of potential weights is too small. Using uniform instead.")
        return rng.choice(V, size=n, replace=False)
    probabilities = w / ssum
    return rng.choice(V, size=n, replace=False, p=probabilities)

############################################################
# Minimax Offset
############################################################

def compute_minimax_offset(X: np.ndarray, labels: np.ndarray, use_kmeans: bool = False, n_clusters: int = 10) -> float:
    """
    Compute the minimax offset between shapes:
    1. For each pair of shapes, find the minimum distance between any two points
    2. Return the maximum of these minimum distances
    
    Parameters:
    -----------
    X : numpy.ndarray
        Data points
    labels : numpy.ndarray
        Shape labels for each point
    use_kmeans : bool, default=False
        If True, re-cluster the data using k-means before computing offset
    n_clusters : int, default=10
        Number of clusters to use for k-means (only used if use_kmeans=True)
    
    Returns:
    --------
    float
        The minimax offset value
    """
    if X is None or len(X) == 0:
        logger.debug("compute_minimax_offset: No data points provided")
        return np.nan
    
    # If use_kmeans is True, use k-means clustering to determine labels
    if use_kmeans:
        try:
            from sklearn.cluster import KMeans
            kmeans = KMeans(n_clusters=n_clusters, random_state=42)
            cluster_labels = kmeans.fit_predict(X)
            logger.debug(f"compute_minimax_offset: Using k-means clustering with {n_clusters} clusters")
            labels_to_use = cluster_labels
        except Exception as e:
            logger.warning(f"compute_minimax_offset: Error in k-means clustering: {e}. Using provided labels.")
            labels_to_use = labels
    else:
        # Use the provided labels
        labels_to_use = labels
    
    if labels_to_use is None or len(labels_to_use) == 0:
        logger.debug("compute_minimax_offset: No labels provided")
        return np.nan
        
    unique_labels = np.unique(labels_to_use)
    if len(unique_labels) <= 1:
        logger.debug("compute_minimax_offset: Only one unique label found - no multiple shapes to compute offset between")
        return np.nan
        
    logger.debug(f"compute_minimax_offset: Computing minimax offset between {len(unique_labels)} shapes")
    min_dists = []
    
    # For each pair of shapes
    for i, label1 in enumerate(unique_labels):
        points1 = X[labels_to_use == label1]
        if len(points1) == 0:
            logger.debug(f"compute_minimax_offset: No points found for shape label {label1}")
            continue
            
        for label2 in unique_labels[i+1:]:
            points2 = X[labels_to_use == label2]
            if len(points2) == 0:
                logger.debug(f"compute_minimax_offset: No points found for shape label {label2}")
                continue
                
            # Compute all pairwise distances
            dists = cdist(points1, points2)
            
            # Find the minimum distance between any two points
            min_dist = np.min(dists)
            min_dists.append(min_dist)
            logger.debug(f"compute_minimax_offset: Minimum distance between shapes {label1} and {label2}: {min_dist:.4f}")
    
    if len(min_dists) == 0:
        logger.debug("compute_minimax_offset: No valid shape pairs found")
        return np.nan
        
    # Return the maximum of the minimum distances
    result = float(np.max(min_dists))
    logger.debug(f"compute_minimax_offset: Final minimax offset: {result:.4f}")
    return result

############################################################
# Main public function
############################################################

def generate_dataset_and_subsamples(config: dict) -> dict:
    """
    Master function: read config['type'] => build full dataset + adjacency => apply noise => build subsamples.
    """
    rng = np.random.default_rng(config.get("random_state", 42))

    data_type = config["type"]
    data_type_l = data_type.lower()
    adjacency_method = config.get("adjacency_method", "knn")
    adjacency_param = {}
    if adjacency_method == "knn":
        adjacency_param["k"] = config.get("knn_k", 5)
        adjacency_param["k_max"] = config.get("knn_k_max", 1000)
        adjacency_param["approximate"] = config.get("use_pynndescent", False)
    elif adjacency_method == "threshold":
        adjacency_param["radius"] = config.get("threshold_r", 1.0)
        adjacency_param["parallelize"] = config.get("threshold_parallel", False)
    elif adjacency_method == "convexhull":
        adjacency_param["iterative_passes"] = config.get("hull_passes", 1)
    else:
        pass  # or raise?

    if data_type_l == "unions_of_shapes":
        shapes = config["shapes"]
        offset = config.get("offset", 10.0)
        X_full, A_full, labels = generate_union_data_revised(shapes, adjacency_method,
                                                             adjacency_param, offset, rng)
        sep_distance = None
        logger.info(f"Generated union of {len(shapes)} shapes with {X_full.shape[0]} total points")
        
    elif data_type_l == "gaussian_tsep":
        gauss_cfg = config["gaussian_cfg"]
        X_full, A_full, labels = generate_simplex_gaussian_mixture(gauss_cfg, rng)
        sep_distance = gauss_cfg.get("offset", 1.0)
        logger.info(f"Generated Gaussian simplex mixture with {X_full.shape[0]} points, offset={sep_distance}")

    # support scRNA (case‐insensitive)
    elif data_type_l == "scrna":
        # load single‐cell RNA data
        matrix_file = config.get("matrix_file", "matrix.mtx")
        n_samp = config.get("n_samples", None)
        shuffle = config.get("shuffle", True)
        X_full = load_scRNA_data(
            matrix_file=matrix_file,
            n_samples=n_samp,
            shuffle=shuffle,
            random_state=config.get("random_state", 42)
        )
        # build adjacency using chosen method
        A_full = build_shape_adjacency(X_full, adjacency_method, adjacency_param)
        labels = None
        sep_distance = None
        logger.info(f"Loaded scRNA data with {X_full.shape[0]} cells and {X_full.shape[1]} genes")

    elif data_type_l == "mnist":
        n_samp = config.get("n_samples", 10000)
        X_mnist, y_mnist = load_mnist_data(n_samples=n_samp, shuffle=True)
            
        X_full = X_mnist
        # build adjacency using chosen method
        A_full = build_shape_adjacency(X_full, adjacency_method, adjacency_param)
        labels = y_mnist
        
        # For MNIST, we'll use the minimax offset between digit clusters as sep_distance
        # The actual computation will happen in the general minimax offset section
        sep_distance = None
        logger.info(f"Loaded MNIST data with {X_full.shape[0]} samples, {len(np.unique(labels))} classes")
    else:
        raise ValueError(f"Unknown data_type={data_type}")

    # Optionally add noise
    noise_opt = config.get("noise", False)
    noise_scale = config.get("noise_scale", 0.01)
    if (isinstance(noise_opt, bool) and noise_opt) or (isinstance(noise_opt, list) and any(noise_opt)):
        logger.info(f"Adding Gaussian noise scale={noise_scale}")
        X_full += rng.normal(loc=0.0, scale=noise_scale, size=X_full.shape)

    # Compute minimax offset if labels are available
    minimax_offset = np.nan
    minimax_offset_scaled = np.nan
    minimax_offset_kmeans = np.nan
    minimax_offset_kmeans_scaled = np.nan
    
    # Get the kmeans option from config, default to False
    use_kmeans = config.get("use_kmeans_offset", False)
    n_clusters = config.get("n_clusters", 10)
    
    if labels is not None:
        logger.info("Computing minimax offset metrics")
        
        # Original minimax offset using provided labels
        minimax_offset = compute_minimax_offset(X_full, labels)
        if not np.isnan(minimax_offset):
            logger.info(f"Minimax offset (using original labels): {minimax_offset:.4f}")
        
        # Compute k-means based minimax offset if requested
        if use_kmeans:
            minimax_offset_kmeans = compute_minimax_offset(X_full, labels, use_kmeans=True, n_clusters=n_clusters)
            if not np.isnan(minimax_offset_kmeans):
                logger.info(f"Minimax offset (using k-means clustering): {minimax_offset_kmeans:.4f}")
        
        # For MNIST datasets, decide which offset to use as sep_distance
        if data_type_l == "mnist":
            if use_kmeans and not np.isnan(minimax_offset_kmeans):
                sep_distance = minimax_offset_kmeans
                logger.info(f"Using k-means based minimax offset as separation distance for MNIST: {sep_distance:.4f}")
            else:
                sep_distance = minimax_offset
                logger.info(f"Using original labels based minimax offset as separation distance for MNIST: {sep_distance:.4f}")
        
        # Scaled minimax offset
        try:
            scaler = StandardScaler()
            X_full_scaled = scaler.fit_transform(X_full)
            
            # Scaled offset with original labels
            minimax_offset_scaled = compute_minimax_offset(X_full_scaled, labels)
            if not np.isnan(minimax_offset_scaled):
                logger.info(f"Minimax offset scaled (using original labels): {minimax_offset_scaled:.4f}")
            
            # Scaled offset with k-means if requested
            if use_kmeans:
                minimax_offset_kmeans_scaled = compute_minimax_offset(X_full_scaled, labels, use_kmeans=True, n_clusters=n_clusters)
                if not np.isnan(minimax_offset_kmeans_scaled):
                    logger.info(f"Minimax offset scaled (using k-means clustering): {minimax_offset_kmeans_scaled:.4f}")
                
        except Exception as e:
            logger.warning(f"Error computing scaled minimax offset: {e}")
    else:
        logger.info("No labels available for minimax offset computation")

    # Build sub-samples
    frac_list = config.get("fractions", [1.0])
    meth_list = config.get("methods", ["uniform"])

    pot_name = config.get("potential_name", None)
    pot_params = config.get("potential_params", {})
    potfunc = None
    if "biased" in meth_list:
        if not pot_name:
            raise ValueError("Need potential_name for 'biased' sampling.")
        potfunc = get_potential_func(pot_name, pot_params)

    N = X_full.shape[0]
    sub_samples = []
    for fval in frac_list:
        n_sub = int(np.floor(fval * N))
        if n_sub < 1:
            logger.warning(f"fraction={fval} => n_sub=0 => skip.")
            continue
        for mm in meth_list:
            if mm == "uniform":
                idx_sub = sample_indices_uniform(N, n_sub, rng)
            elif mm == "biased":
                idx_sub = sample_indices_biased(X_full, n_sub, potfunc, rng)
            else:
                raise ValueError(f"Unknown sampling method {mm}")
                
            # For MNIST datasets, ensure all digit classes are represented in the subsample
            if data_type_l == "mnist" and labels is not None:
                # Only apply if we have a reasonable number of samples
                # (at least 10 for one from each class)
                if n_sub >= 10:
                    logger.info(f"Ensuring all MNIST digit classes (0-9) are represented in the subsample")
                    idx_sub = ensure_all_mnist_classes(idx_sub, labels, num_classes=10, random_state=rng)
                else:
                    logger.warning(f"Sample size {n_sub} is too small to ensure all 10 MNIST classes are represented")

            X_sub = X_full[idx_sub]
            if labels is not None:
                y_sub = labels[idx_sub]
            else:
                y_sub = None
            sub_samples.append({
                "method": mm,
                "fraction": fval,
                "X_sub": X_sub,
                "indices_sub": idx_sub,
                "labels_sub": y_sub
            })

    return {
        "X_full": X_full,
        "A_full": A_full,
        "sub_samples": sub_samples,
        "sep_distance": sep_distance,
        "labels": labels if labels is not None else None,
        "minimax_offset": minimax_offset,
        "minimax_offset_scaled": minimax_offset_scaled,
        "minimax_offset_kmeans": minimax_offset_kmeans,
        "minimax_offset_kmeans_scaled": minimax_offset_kmeans_scaled
    }