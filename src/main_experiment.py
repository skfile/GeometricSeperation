#!/usr/bin/env python3
"""
main_experiment.py
-----------------------------
Core experiment pipeline for manifold separation threshold detection.

This module implements the complete pipeline for testing the geometric threshold hypothesis:
1. Dataset generation with controlled offsets between manifold components
2. Subsampling with uniform, biased, or noisy methods
3. Graph/kernel construction with various neighborhood methods
4. Embedding computation using dimensionality reduction techniques
5. Clustering of embeddings to detect component separation
6. Evaluation using Gromov-Wasserstein distance and clustering metrics
7. Result aggregation and storage

The pipeline supports parallel processing and is configured via JSON files.
"""

import os
import sys
import json
import argparse
import numpy as np
import pandas as pd
import traceback
import logging
import matplotlib
import time
from typing import Dict, List, Any, Optional, Tuple, Union, Callable

import multiprocessing.resource_tracker as _rt
_rt.ResourceTracker.__del__ = lambda self: None  # silence resource_tracker cleanup errors

logger = logging.getLogger("experiment_logger")
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter("[%(asctime)s] [%(levelname)s] %(message)s"))
logger.addHandler(handler)

sys.path.insert(0, os.path.abspath(os.path.dirname(os.path.dirname(__file__))))

from joblib import Parallel, delayed
from multiprocessing import cpu_count
from sklearn.metrics import adjusted_rand_score
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import cdist, pdist
from scipy.sparse.csgraph import shortest_path
from sklearn.preprocessing import StandardScaler  
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN, SpectralClustering
from scipy import sparse

from src.parallel_utils import parallel_context
from src.mesh_sampling import generate_dataset_and_subsamples
from src.kernels import kernel_dispatcher
from src.metrics import (
    compute_gromov_wasserstein,
    compute_single_linkage_ultrametric,
    approximate_gh_on_ultrametrics
)
from src.embedding_algorithms import compute_embedding
from src.utils import (
    connected_comp_helper,
    preprocess_distance_matrix,
    normalize_distance_matrix,
)

try:
    from src.gw_utils import gromov_wasserstein, parallel_gw_computation, batch_gw_computation
    HAS_GW_UTILS = True
    logger.info("Successfully imported optimized GW utilities.")
except ImportError as e:
    HAS_GW_UTILS = False
    logger.warning(f"Could not import gw_utils: {e}. Using default GW implementation.")

matplotlib.use('Agg')
import matplotlib.pyplot as plt

plt.ioff()
plt.close('all')

import warnings
warnings.filterwarnings("ignore", category=FutureWarning, module='sklearn.utils.deprecation')

def validate_configuration(config):
    needed = ["datasets", "kernel_methods", "embedding_methods", "GW_loss_fun", "results_dir"]
    for r in needed:
        if r not in config:
            raise ValueError(f"Missing '{r}' in top-level config.")
    if not isinstance(config["datasets"], list):
        raise ValueError("config['datasets'] must be a list.")
    
    # Validate clustering parameters if present
    if "clustering" in config:
        clustering = config["clustering"]
        if not isinstance(clustering, dict):
            raise ValueError("config['clustering'] must be a dictionary.")
        
        if "method" in clustering:
            valid_methods = ["kmeans", "agglomerative", "dbscan", "spectral", "hierarchical"]
            if clustering["method"] not in valid_methods:
                raise ValueError(f"Invalid clustering method. Must be one of {valid_methods}.")


def gather_shapes_info(dset_cfg):
    """
    Convert shape parameters to a JSON string for logging in the CSV.
    Example: store offset, shapes array with dimension, radius/axes, n_points, etc.
    """
    shapes_info_list = []
    offset = dset_cfg.get("offset", 0.0)
    shapes = dset_cfg.get("shapes", [])
    for sh in shapes:
        si = {
            "shape_type": sh.get("shape_type"),
            "dim": sh.get("dim"),
            "n_points": sh.get("n_points")
        }
        # Add radius or axes if present
        if sh.get("shape_type") == "sphere_hd":
            si["radius"] = sh.get("radius")
        elif sh.get("shape_type") == "ellipsoid_hd":
            si["axes"] = sh.get("axes")
        elif sh.get("shape_type") == "torus_hd":
            si["major_radius"] = sh.get("major_radius")
            si["minor_radius"] = sh.get("minor_radius")
        shapes_info_list.append(si)

    info_dict = {
        "offset": offset,
        "shapes": shapes_info_list
    }
    return json.dumps(info_dict)


def compute_clusters(X, method, n_clusters=None, **kwargs):
    """Compute cluster labels using the specified clustering method
    
    Parameters:
    -----------
    X : array-like, shape (n_samples, n_features)
        Input data
    method : str
        Clustering method: 'kmeans', 'agglomerative', 'dbscan', 'spectral', or 'hierarchical'
    n_clusters : int, optional
        Number of clusters (required for kmeans, agglomerative, spectral)
    **kwargs : additional parameters for specific clustering algorithms
    
    Returns:
    --------
    labels : array-like, shape (n_samples,)
        Cluster labels for each point
    """
    if X is None:
        return None
    
    method = method.lower()
    
    if method == 'kmeans':
        if n_clusters is None:
            raise ValueError("n_clusters is required for KMeans clustering")
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, **kwargs)
        return kmeans.fit_predict(X)
    
    elif method == 'agglomerative':
        if n_clusters is None:
            raise ValueError("n_clusters is required for AgglomerativeClustering")
        agg = AgglomerativeClustering(n_clusters=n_clusters, **kwargs)
        return agg.fit_predict(X)
    
    elif method == 'dbscan':
        eps = kwargs.get('eps', 0.5)
        min_samples = kwargs.get('min_samples', 5)
        dbscan = DBSCAN(eps=eps, min_samples=min_samples, **kwargs)
        return dbscan.fit_predict(X)
    
    elif method == 'spectral':
        if n_clusters is None:
            raise ValueError("n_clusters is required for SpectralClustering")
        spectral = SpectralClustering(n_clusters=n_clusters, random_state=42, **kwargs)
        return spectral.fit_predict(X)
    
    elif method == 'hierarchical':
        linkage_method = kwargs.get('linkage_method', 'ward')
        Z = linkage(X, method=linkage_method)
        if n_clusters is None:
            t = kwargs.get('t', 0.5)  # Distance threshold
            criterion = kwargs.get('criterion', 'distance')
            return fcluster(Z, t=t, criterion=criterion)
        else:
            return fcluster(Z, t=n_clusters, criterion='maxclust')
    
    else:
        raise ValueError(f"Unknown clustering method: {method}")


def compute_minimax_offset(X, labels):
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
    
    Returns:
    --------
    float
        The minimax offset value
    """
    if X is None or labels is None or len(X) == 0:
        return np.nan
        
    unique_labels = np.unique(labels)
    if len(unique_labels) <= 1:
        return np.nan
        
    min_dists = []
    
    # For each pair of shapes
    for i, label1 in enumerate(unique_labels):
        points1 = X[labels == label1]
        if len(points1) == 0:
            continue
            
        for label2 in unique_labels[i+1:]:
            points2 = X[labels == label2]
            if len(points2) == 0:
                continue
                
            # Compute all pairwise distances
            dists = cdist(points1, points2)
            
            # Find the minimum distance between any two points
            min_dist = np.min(dists)
            min_dists.append(min_dist)
    
    if len(min_dists) == 0:
        return np.nan
        
    # Return the maximum of the minimum distances
    return float(np.max(min_dists))


def run_experiment_for_dataset(
    dset_cfg: Dict[str, Any], 
    kernel_methods: List[Dict[str, Any]],
    embedding_methods: List[Dict[str, Any]], 
    GW_loss_fun: str,
    clustering_cfg: Optional[Dict[str, Any]] = None, 
    n_jobs: int = 1
) -> List[Dict[str, Any]]:
    """
    Run complete analysis pipeline for a single dataset configuration.
    
    Parameters
    ----------
    dset_cfg : Dict[str, Any]
        Dataset configuration containing name, method, parameters and offset
    kernel_methods : List[Dict[str, Any]]
        List of kernel/graph construction methods to evaluate
    embedding_methods : List[Dict[str, Any]]
        List of embedding methods to evaluate
    GW_loss_fun : str
        Loss function for Gromov-Wasserstein calculations
    clustering_cfg : Optional[Dict[str, Any]], default=None
        Configuration for clustering methods
    n_jobs : int, default=1
        Number of parallel jobs
        
    Returns
    -------
    List[Dict[str, Any]]
        Experiment results for all combinations of subsamples, kernels and embeddings
    """
    logger.info(f"Processing Dataset: {dset_cfg['name']} with Offset: {dset_cfg.get('offset', 0.0)}")
    results = []
    
    try:
        offset_value = dset_cfg.get('offset', 0.0)
        dset_cfg_copy = dset_cfg.copy()
        
        if isinstance(offset_value, (list, tuple, np.ndarray)):
            offset_value = np.asarray(offset_value)
            
        dset_cfg_copy['offset'] = offset_value
        dd = generate_dataset_and_subsamples(dset_cfg_copy)
    except Exception as e:
        logger.error(f"Error generating dataset '{dset_cfg['name']}' with Offset {dset_cfg.get('offset', 0.0)}: {e}")
        return results

    X_full = dd["X_full"]
    A_full = dd["A_full"]
    sub_samples = dd["sub_samples"]
    sep_dist = dd["sep_distance"]  # t_value for gaussian_tsep, or None
    n_total_points = X_full.shape[0]
    
    # Get the precomputed minimax offset values from mesh_sampling.py
    minimax_offset = dd.get("minimax_offset", np.nan)
    minimax_offset_scaled = dd.get("minimax_offset_scaled", np.nan)
    logger.info(f"Using precomputed minimax offset: {minimax_offset:.4f}, scaled: {minimax_offset_scaled:.4f}")

    # Precompute fill distances once per subsample (raw and scaled)
    scaler_fs = StandardScaler().fit(X_full)
    X_full_scaled = scaler_fs.transform(X_full)
    d = X_full_scaled.shape[1]
    n = X_full_scaled.shape[0]
    k_knn = int(np.round(d + np.log2(n / 0.05)))
    for ssub in sub_samples:
        X_sub = ssub["X_sub"]
        idx_sub = ssub["indices_sub"]
        
        ssub["fill_dist_orig"] = float(np.max(np.min(cdist(X_full, X_sub), axis=1)))
        X_sub_scaled = scaler_fs.transform(X_sub)
        ssub["fill_dist_scaled"] = float(np.max(np.min(cdist(X_full_scaled, X_sub_scaled), axis=1)))
        from sklearn.neighbors import NearestNeighbors
        nbrs_orig = NearestNeighbors(n_neighbors=min(k_knn+1, X_sub.shape[0]), algorithm='auto').fit(X_sub)
        distances_orig, _ = nbrs_orig.kneighbors(X_sub)
        if X_sub.shape[0] > k_knn:
            knn_dists_orig = distances_orig[:, k_knn]
        else:
            knn_dists_orig = distances_orig[:, -1]  # fallback: use farthest neighbor if not enough points
        ssub["fill_distance_knn_mean"] = float(np.mean(knn_dists_orig))
        ssub["fill_distance_knn_max"] = float(np.max(knn_dists_orig))
        
        # --- kNN fill distances (scaled) ---
        nbrs = NearestNeighbors(n_neighbors=min(k_knn+1, X_sub_scaled.shape[0]), algorithm='auto').fit(X_sub_scaled)
        distances, _ = nbrs.kneighbors(X_sub_scaled)
        # distances[:, 0] is always 0 (self), so k-th neighbor is at index k_knn
        if X_sub_scaled.shape[0] > k_knn:
            knn_dists = distances[:, k_knn]
        else:
            knn_dists = distances[:, -1]  # fallback: use farthest neighbor if not enough points
        ssub["fill_distance_knn_mean_scaled"] = float(np.mean(knn_dists))
        ssub["fill_distance_knn_max_scaled"] = float(np.max(knn_dists))

    # Build ground-truth cost
    try:
        # ensure adjacency is in CSR sparse format for shortest_path
        from scipy.sparse import issparse, csr_matrix
        if not issparse(A_full):
            A_sparse = csr_matrix(A_full)
        else:
            A_sparse = A_full.tocsr()
        GT_sp = shortest_path(A_sparse, directed=False)
        GT_sp = 0.5 * (GT_sp + GT_sp.T)
        GT_sp = preprocess_distance_matrix(GT_sp)
        GT_sp = normalize_distance_matrix(GT_sp)
    except Exception as e:
        logger.warning(f"[{dset_cfg['name']} Offset:{dset_cfg.get('offset',0.0)}] Error building GT cost => {e}")
        return results

    # Single-link
    try:
        U_full_HD = compute_single_linkage_ultrametric(X_full, metric='euclidean')
    except Exception as e:
        logger.warning(f"[{dset_cfg['name']} Offset:{dset_cfg.get('offset',0.0)}] single-link full => {e}")
        U_full_HD = None

    # Get ground truth labels from dataset
    labels_full = dd.get("labels", None)
    
    # If clustering config is provided and no labels exist, generate clusters
    if labels_full is None and clustering_cfg and "gt_method" in clustering_cfg:
        gt_method = clustering_cfg.get("gt_method")
        gt_n_clusters = dset_cfg.get("gt_n_clusters", clustering_cfg.get("gt_n_clusters") if clustering_cfg else None)
        
        try:
            logger.info(f"Generating ground truth clusters using {gt_method} with {gt_n_clusters} clusters")
            labels_full = compute_clusters(
                X_full, 
                method=gt_method, 
                n_clusters=gt_n_clusters,
                **clustering_cfg.get("gt_params", {})
            )
            logger.info(f"Generated {len(np.unique(labels_full))} ground truth clusters")
        except Exception as e:
            logger.warning(f"[{dset_cfg['name']} Offset:{dset_cfg.get('offset',0.0)}] Error computing ground truth clusters => {e}")
            labels_full = None

    # Potential-based measure
    from src.utils import measure_from_potential

    q_unif = np.ones(n_total_points) / n_total_points
    pot_name = dset_cfg.get("potential_name", None)
    pot_params = dset_cfg.get("potential_params", {})
    q_biased = None
    if "biased" in dset_cfg.get("methods", []):
        if not pot_name:
            raise ValueError("Need potential_name for 'biased' sampling.")
        try:
            q_biased = measure_from_potential(X_full, pot_name, pot_params)
        except Exception as e:
            logger.warning(f"[{dset_cfg['name']} Offset:{dset_cfg.get('offset',0.0)}] Error computing biased measure => {e}")

    # Embed full
    emb_full_map = {}
    U_full_Emb_map = {}
    
    # Convert full adjacency/kernel to dense if sparse
    if sparse.issparse(A_full):
        A_full_data = A_full.toarray()
    else:
        A_full_data = A_full

    # get dataset‐specific clustering params, fallback to global
    ds_n_clusters = dset_cfg.get("n_clusters", clustering_cfg.get("n_clusters") if clustering_cfg else None)
    ds_gt_clusters = dset_cfg.get("gt_n_clusters", clustering_cfg.get("gt_n_clusters") if clustering_cfg else None)

    if ds_gt_clusters is None:
        ds_gt_clusters = ds_n_clusters

    # Compute full embedding for each method
    for em in embedding_methods:
        low_em = em.lower()
        try:
            embf, _ = compute_embedding(A_full_data, method=low_em, n_components=2)
            if embf is not None:
                UfEmb = compute_single_linkage_ultrametric(embf, metric='euclidean')
            else:
                UfEmb = None
            emb_full_map[em] = embf
            U_full_Emb_map[em] = UfEmb
            
            # Compute clustering on full embedding if clustering config is provided
            if clustering_cfg and labels_full is not None and embf is not None:
                method = clustering_cfg.get("method")
                n_clusters = ds_n_clusters
                
                if method and n_clusters:
                    try:
                        logger.info(f"Computing clusters on full {em} embedding")
                        full_emb_labels = compute_clusters(
                            embf,
                            method=method,
                            n_clusters=n_clusters,
                            **clustering_cfg.get("params", {})
                        )
                        
                        # Compute ARI between ground truth and embedding clusters
                        if full_emb_labels is not None:
                            ari_full_emb = adjusted_rand_score(labels_full, full_emb_labels)
                            logger.info(f"Full {em} embedding ARI: {ari_full_emb:.4f}")
                            emb_full_map[f"{em}_ari"] = ari_full_emb
                    except Exception as e:
                        logger.warning(f"[{dset_cfg['name']} Offset:{dset_cfg.get('offset',0.0)}] Error computing clusters on full {em} embedding => {e}")
                        emb_full_map[f"{em}_ari"] = None
        
        except Exception as e:
            logger.warning(f"[{dset_cfg['name']} Offset:{dset_cfg.get('offset',0.0)}] embed full with {em} => {e}")
            emb_full_map[em] = None
            U_full_Emb_map[em] = None
            emb_full_map[f"{em}_ari"] = None
            

    # Prepare shapes info
    shapes_info_str = ""
    dataset_type = dset_cfg.get("type", "unions_of_shapes")
    has_gaussian_tsep = (dataset_type == "gaussian_tsep")
    has_union = (dataset_type == "unions_of_shapes" and dset_cfg.get("shapes", None) is not None)

    if has_union:
        shapes_info_str = gather_shapes_info(dset_cfg)
    elif has_gaussian_tsep:
        gcfg = dset_cfg.get("gaussian_cfg", {})
        shapes_info_str = json.dumps({
            "dims": gcfg.get("dims"),
            "n_points_per_gauss": gcfg.get("n_points_per_gauss"),
            "sigma_max_list": gcfg.get("sigma_max_list"),
            "offset": dset_cfg.get("offset", 0.0)
        })

    # Build cost with each kernel in parallel => measure GW
    def process_kernel(km, ssub):
        kname = km["name"]
        kparams = km.get("params", {})
        s_meth = ssub["method"]
        frac = ssub["fraction"]
        X_sub = ssub["X_sub"]
        idx_sub = ssub["indices_sub"]
        n_sub = X_sub.shape[0]
        
        # retrieve precomputed fill distances
        fill_dist_orig = ssub.get("fill_dist_orig")
        fill_dist_scaled = ssub.get("fill_dist_scaled")
        fill_distance_knn_mean_scaled = ssub.get("fill_distance_knn_mean_scaled", None)
        fill_distance_knn_max_scaled = ssub.get("fill_distance_knn_max_scaled", None)
        fill_distance_knn_mean = ssub.get("fill_distance_knn_mean", None)
        fill_distance_knn_max = ssub.get("fill_distance_knn_max", None)
        
        p_unif_sub = np.ones(n_sub) / n_sub
        
        kernel_results = []
        
        logger.info(f"Start building kernel='{kname}' for sub-sample with {X_sub.shape[0]} points.")
        t0 = time.time()

        # First compute the single-linkage ultrametric for the subsample
        U_sub_HD = None
        try:
            U_sub_HD = compute_single_linkage_ultrametric(X_sub, metric='euclidean')
            logger.info(f"Successfully computed single-linkage ultrametric for subsample with {X_sub.shape[0]} points")
        except Exception as e:
            logger.warning(f"[{dset_cfg['name']} Offset:{dset_cfg.get('offset',0.0)}] Error computing single-linkage for subsample => {e}")

        try:
            sub_kfunc = kernel_dispatcher(kname, **kparams)
            sub_adj = sub_kfunc(X_sub)
        except Exception as e:
            logger.warning(f"[{dset_cfg['name']} Offset:{dset_cfg.get('offset',0.0)}] Error initializing kernel '{kname}' => {e}")
            return kernel_results

        try:
            logger.info(f"Finished building kernel='{kname}' in {time.time() - t0:.2f}s. Now cleaning + shortest_path.")
            sub_adj = connected_comp_helper(sub_adj, X_sub, connect=True)
            sub_cost = shortest_path(sub_adj, directed=True)
            sub_cost = 0.5 * (sub_cost + sub_cost.T)
            sub_cost = preprocess_distance_matrix(sub_cost)
            sub_cost = normalize_distance_matrix(sub_cost)
        except Exception as costE:
            logger.warning(
                f"[{dset_cfg['name']} Offset:{dset_cfg.get('offset',0.0)}] building sub cost kernel={kname}, frac={frac} => {costE}"
            )
            return kernel_results

        logger.info(f"kernel='{kname}' => adjacency shape={sub_adj.shape}, next computing Gromov-Wasserstein ...")

        # Convert subsample adjacency to dense if sparse
        if sparse.issparse(sub_adj):
            sub_data = sub_adj.toarray()
        else:
            sub_data = sub_adj

        # Gromov-Wasserstein - Use optimized implementation if available
        gw_u_u, gw_nu_u = None, None
        try:
            if HAS_GW_UTILS:
                gw_u_u, _ = gromov_wasserstein(
                    GT_sp, sub_cost, q_unif, p_unif_sub, 
                    loss_fun=GW_loss_fun, use_gpu=True
                )
            else:
                gw_u_u = compute_gromov_wasserstein(
                    GT_sp, sub_cost, q_unif, p_unif_sub, loss_fun=GW_loss_fun
                )
        except Exception as e:
            logger.warning(f"[{dset_cfg['name']} Offset:{dset_cfg.get('offset',0.0)}] GW computation uniform-uniform => {e}")

        if q_biased is not None:
            try:
                if HAS_GW_UTILS:
                    gw_nu_u, _ = gromov_wasserstein(
                        GT_sp, sub_cost, q_biased, p_unif_sub, 
                        loss_fun=GW_loss_fun, use_gpu=True
                    )
                else:
                    gw_nu_u = compute_gromov_wasserstein(
                        GT_sp, sub_cost, q_biased, p_unif_sub, loss_fun=GW_loss_fun
                    )
            except Exception as e:
                logger.warning(f"[{dset_cfg['name']} Offset:{dset_cfg.get('offset',0.0)}] GW computation biased-uniform => {e}")

        # Compute ARI if ground truth labels are available
        ari_value = None
        if labels_full is not None and clustering_cfg and clustering_cfg.get("method"):
            try:
                # Extract just the labels for the subsample points
                subsample_gt_labels = labels_full[idx_sub] if idx_sub is not None else None
                
                if subsample_gt_labels is not None:
                    # Compute clusters on the subsample points
                    method = clustering_cfg.get("method")
                    n_clusters = ds_n_clusters
                    
                    sub_labels = compute_clusters(
                        X_sub, 
                        method=method, 
                        n_clusters=n_clusters, 
                        **clustering_cfg.get("params", {})
                    )
                    
                    if sub_labels is not None:
                        # Compute ARI between ground truth subsample and computed clusters
                        ari_value = adjusted_rand_score(subsample_gt_labels, sub_labels)
                        logger.info(f"Computed ARI={ari_value:.4f} for kernel={kname}, method={s_meth}, frac={frac}")
            except Exception as e:
                logger.warning(f"[{dset_cfg['name']} Offset:{dset_cfg.get('offset',0.0)}] Error computing ARI => {e}")
                ari_value = None

        # Process each embedding method for this kernel
        for em in embedding_methods:
            gh_hd, gh_emb = None, None
            ari_sub_emb = None
            sub_e = None
            
            try:
                low_em = em.lower()
                if low_em == "diffusionmap":
                    # Enforce symmetry to avoid 'Kernel matrix is not symmetric' error
                    if sparse.issparse(sub_data):
                        sub_data_sym = 0.5 * (sub_data + sub_data.T)
                    else:
                        sub_data_sym = 0.5 * (sub_data + sub_data.T)
                    sub_e, _ = compute_embedding(sub_data_sym, method=low_em, n_components=2)
                elif low_em == "tsne":
                    perp_s = min(50, n_sub - 1)
                    sub_e, _ = compute_embedding(X_sub, method='tsne', n_components=2, perplexity=perp_s)
                else:
                    sub_e, _ = compute_embedding(X_sub, method=low_em, n_components=2)
                
                if sub_e is not None:
                    U_subEmb = compute_single_linkage_ultrametric(sub_e, metric='euclidean')
                else:
                    U_subEmb = None
                    
                # Compute ARI on subsample embedding if possible
                if labels_full is not None and clustering_cfg and clustering_cfg.get("method") and sub_e is not None:
                    try:
                        # Extract just the labels for the subsample points
                        subsample_gt_labels = labels_full[idx_sub] if idx_sub is not None else None
                        
                        if subsample_gt_labels is not None:
                            # Compute clusters on the subsample embedding
                            method = clustering_cfg.get("method")
                            n_clusters = ds_n_clusters
                            
                            sub_emb_labels = compute_clusters(
                                sub_e, 
                                method=method, 
                                n_clusters=n_clusters, 
                                **clustering_cfg.get("params", {})
                            )
                            
                            if sub_emb_labels is not None:
                                # Compute ARI between ground truth subsample and embedding clusters
                                ari_sub_emb = adjusted_rand_score(subsample_gt_labels, sub_emb_labels)
                                logger.info(f"Computed Embedding ARI={ari_sub_emb:.4f} for kernel={kname}, method={s_meth}, frac={frac}, embedding={em}")
                    except Exception as e:
                        logger.warning(f"[{dset_cfg['name']} Offset:{dset_cfg.get('offset',0.0)}] Error computing ARI on subsample embedding => {e}")
                        ari_sub_emb = None
                
            except Exception as e:
                U_subEmb = None
                logger.warning(f"[{dset_cfg['name']} Offset:{dset_cfg.get('offset',0.0)}] embed sub with {em} => {e}")

            # GH(HD) - now U_sub_HD is defined
            if (U_full_HD is not None) and (U_sub_HD is not None):
                try:
                    gh_hd = approximate_gh_on_ultrametrics(U_sub_HD, U_full_HD)
                except Exception as e:
                    logger.warning(f"[{dset_cfg['name']} Offset:{dset_cfg.get('offset',0.0)}] GH(HD) => {e}")

            # GH(Emb)
            UfEmb = U_full_Emb_map.get(em, None)
            if (UfEmb is not None) and (U_subEmb is not None):
                try:
                    gh_emb = approximate_gh_on_ultrametrics(U_subEmb, UfEmb)
                except Exception as e:
                    logger.warning(f"[{dset_cfg['name']} Offset:{dset_cfg.get('offset',0.0)}] GH(Emb) => {e}")

            row = {
                "Dataset": dset_cfg["name"],
                "Offset": dset_cfg.get("offset", 0.0),
                "Noise": dset_cfg.get("noise", False),
                "Embedding_Method": em,
                "Sampling_Method": s_meth,
                "Sample_Percentage": frac,
                "Potential_Name": pot_name,
                "Potential_Params": json.dumps(pot_params),
                "Kernel_Method": kname,
                "Kernel_Params": json.dumps(kparams),
                "GroundTruth_Kernel": dset_cfg.get("ground_truth_kernel", "mesh"),
                "GroundTruth_Kernel_Params": json.dumps(dset_cfg.get("ground_truth_kernel_params", {})),
                "GW_Uniform_Uniform": gw_u_u,
                "GW_NonUniform_Uniform": gw_nu_u,
                "GH_Ultrametric_HD": gh_hd,
                "GH_Ultrametric_Emb": gh_emb,
                "Fill_Distance": fill_dist_orig,
                "Fill_Distance_Scaled": fill_dist_scaled,
                "Fill_Distance_KNN_Mean_Scaled": fill_distance_knn_mean_scaled,
                "Fill_Distance_KNN_Max_Scaled": fill_distance_knn_max_scaled,
                "Fill_Distance_KNN_Mean": fill_distance_knn_mean,
                "Fill_Distance_KNN_Max": fill_distance_knn_max,
                "ARI": ari_value,
                "SeparationDistance": sep_dist,
                "Shapes_Info": shapes_info_str,
                "NTotalPoints": n_total_points,
                "Minimax_Offset": minimax_offset,
                "Minimax_Offset_Scaled": minimax_offset_scaled,
                "ARI_Sub_Embedding": ari_sub_emb,
                "ARI_Full_Embedding": emb_full_map.get(f"{em}_ari", None)
            }
            kernel_results.append(row)
            
        return kernel_results

    # Process each subsample
    for ssub in sub_samples:
        # Process kernels in parallel
        kernel_jobs = []
        for km in kernel_methods:
            kernel_jobs.append((km, ssub))
        
        # Use parallel processing for kernel computation
        with parallel_context(n_jobs=n_jobs) as actual_n_jobs:
            logger.info(f"Using {actual_n_jobs} processes for parallel kernel processing")
            parallel_results = Parallel(n_jobs=actual_n_jobs)(
                delayed(process_kernel)(km, ssub) for km, ssub in kernel_jobs
            )
            
        # Flatten results
        for res_list in parallel_results:
            results.extend(res_list)

    return results


def main():
    parser = argparse.ArgumentParser(description="HPC-friendly main_experiment.")
    parser.add_argument("--config", type=str, required=True, help="Path to config JSON.")
    parser.add_argument("--n_jobs", type=int, default=-1, help="Number of parallel jobs.")
    args = parser.parse_args()

    # Logging
    fh = logging.FileHandler("experiment.log", mode='a')
    fh.setFormatter(logging.Formatter("[%(asctime)s] [%(levelname)s] %(message)s"))
    logger.addHandler(fh)

    if not os.path.exists(args.config):
        logger.error(f"Config file not found: {args.config}")
        return

    with open(args.config, 'r') as f:
        config = json.load(f)

    try:
        validate_configuration(config)
    except Exception as e:
        logger.error(f"Config validation error => {e}")
        return

    kernel_methods = config["kernel_methods"]
    embedding_methods = config["embedding_methods"]
    GW_loss_fun = config["GW_loss_fun"]
    results_dir = config["results_dir"]
    clustering_cfg = config.get("clustering", None)
    
    if clustering_cfg:
        logger.info(f"Using clustering configuration: {json.dumps(clustering_cfg)}")
    
    os.makedirs(results_dir, exist_ok=True)

    out_csv = os.path.join(results_dir, "all_experiment_results.csv")
    columns = [
        "Dataset", "Offset", "Noise", "Embedding_Method", "Sampling_Method", "Sample_Percentage",
        "Potential_Name", "Potential_Params",
        "Kernel_Method", "Kernel_Params",
        "GroundTruth_Kernel", "GroundTruth_Kernel_Params",
        "GW_Uniform_Uniform", "GW_NonUniform_Uniform",
        "GH_Ultrametric_HD", "GH_Ultrametric_Emb",
        "Fill_Distance", "Fill_Distance_Scaled", "Fill_Distance_KNN_Mean_Scaled", "Fill_Distance_KNN_Max_Scaled",
        "Fill_Distance_KNN_Mean", "Fill_Distance_KNN_Max", "ARI", "SeparationDistance",
        "Shapes_Info", "NTotalPoints", "Minimax_Offset", "Minimax_Offset_Scaled", "ARI_Sub_Embedding", "ARI_Full_Embedding"
    ]
    # Initialize CSV with headers
    if not os.path.exists(out_csv):
        df_init = pd.DataFrame(columns=columns)
        df_init.to_csv(out_csv, index=False)
        logger.info(f"Initialized results CSV at {out_csv}")

    # Stage 1: Generate Datasets in Parallel
    logger.info("Starting Stage 1: Generating and Processing Datasets")

    datasets = config["datasets"]
    expanded_datasets = []

    # Expand each dataset by its 'offset' array
    for dset in datasets:
        name = dset.get("name", "UnnamedDataset")
        if dset.get("type", "") == "gaussian_tsep":
            offsets = dset.get("offset", [0.0])
        else:
            offsets = dset.get("offset", [0.0])
        if not isinstance(offsets, list):
            offsets = [offsets]
        for offset in offsets:
            # Create a copy of the dataset configuration with the specific offset
            dset_copy = dset.copy()
            dset_copy["offset"] = offset
            # Remove 'offset' from 'gaussian_cfg' if present
            if dset_copy.get("type", "") == "gaussian_tsep":
                dset_copy["gaussian_cfg"] = dset_copy.get("gaussian_cfg", {}).copy()
                dset_copy["gaussian_cfg"].pop("offset", None)
            dset_copy["name"] = f"{name}_offset_{offset}"
            
            # Handle adjacency method based on dimension
            if dset_copy.get("type", "") == "unions_of_shapes" and "shapes" in dset_copy:
                max_dim = 0
                for shape in dset_copy["shapes"]:
                    dim = shape.get("dim", 0)
                    max_dim = max(max_dim, dim)
                
                # Use convexhull for low dimensions, pynndescent for high dimensions
                if max_dim < 10:
                    dset_copy["adjacency_method"] = "convexhull"
                else:
                    dset_copy["adjacency_method"] = "knn"
                    dset_copy["knn_k"] = min(2**max_dim, 1024)
                    dset_copy["use_pynndescent"] = True
                
                logger.info(f"Setting adjacency method for {dset_copy['name']} to {dset_copy['adjacency_method']} based on max dimension {max_dim}")
            
            # Propagate global potential settings
            dset_copy["potential_name"] = config.get("potential_name", None)
            dset_copy["potential_params"] = config.get("potential_params", {})

            expanded_datasets.append(dset_copy)

    num_datasets = len(expanded_datasets)
    n_jobs_stage1 = min(num_datasets, args.n_jobs if args.n_jobs > 0 else cpu_count())
    
    # If only one dataset but multiple processors requested, use processors inside dataset computation
    intra_dataset_jobs = 1
    if num_datasets == 1 and args.n_jobs > 1:
        n_jobs_stage1 = 1
        intra_dataset_jobs = args.n_jobs
        logger.info(f"Single dataset detected - using {intra_dataset_jobs} processors for internal parallelization")

    logger.info(f"Total Datasets to Process: {num_datasets}")
    logger.info(f"Using {n_jobs_stage1} parallel jobs for dataset processing.")

    # Define a helper function for Stage 1
    def generate_and_process(dset_cfg):
        try:
            results = run_experiment_for_dataset(
                dset_cfg, kernel_methods, embedding_methods, GW_loss_fun,
                clustering_cfg=clustering_cfg,
                n_jobs=intra_dataset_jobs
            )
            return results
        except Exception as e:
            logger.error(f"Error processing dataset '{dset_cfg['name']}': {e}")
            traceback.print_exc()
            return []

    # Parallel Dataset Processing
    with parallel_context(n_jobs=n_jobs_stage1, backend='loky') as n_jobs:
        stage1_results = Parallel(n_jobs=n_jobs, verbose=10)(
            delayed(generate_and_process)(dset_cfg) for dset_cfg in expanded_datasets
        )

    # Flatten and Append to CSV
    for idx, res in enumerate(stage1_results):
        dataset_name = expanded_datasets[idx]["name"]
        if res:
            df_stage = pd.DataFrame(res, columns=columns)
            df_stage.to_csv(out_csv, index=False, mode='a', header=False)
            logger.info(f"Appended {len(df_stage)} rows for Dataset: {dataset_name} to {out_csv}.")
        else:
            logger.warning(f"No results to append for Dataset: {dataset_name}.")

    logger.info(f"All experiments completed. Results are saved in {out_csv}.")

    # Run plotting scripts to organize outputs
    import subprocess
    # Run general plotter
    try:
        subprocess.run([
            sys.executable, os.path.join(os.path.dirname(__file__), 'general_plotter.py'),
            '--config', args.config,
            '--output_dir', results_dir
        ], check=True)
        logger.info("Generated comprehensive plots via general_plotter.py")
    except Exception as e:
        logger.warning(f"general_plotter.py failed: {e}")
    # Run final analysis plots regardless of general_plotter success
    try:
        subprocess.run([
            sys.executable, os.path.join(os.path.dirname(__file__), 'final_analysis_plots.py'),
            '--input_csv', out_csv,
            '--output_dir', results_dir
        ], check=True)
        logger.info("Generated analysis plots via final_analysis_plots.py")
    except Exception as e:
        logger.warning(f"final_analysis_plots.py failed: {e}")
    logger.info("All plotting scripts completed.")


if __name__ == "__main__":
    main()
