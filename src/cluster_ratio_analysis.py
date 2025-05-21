#!/usr/bin/env python3

"""
cluster_ratio_analysis.py
-------------------------
Analyzes how the ratio of minimax offsets to fill distances changes as we vary
the number of clusters in k-means. This helps identify the "natural" number of 
clusters in real-world datasets like MNIST and scRNA.

Key metrics:
- For each k in k-means: compute clusters, calculate fill distances and minimax offsets
- Calculate the percentage of cluster pairs with offset/fill_distance ratio > threshold
- Plot this percentage against k for different sampling resolutions

Usage:
  python cluster_ratio_analysis.py --dataset mnist|scRNA [--output_dir output_directory] [--sample_sizes 2000,1000,500,250]
"""

import os
import sys
import time
import argparse
import logging
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import ListedColormap
import seaborn as sns
from scipy.spatial.distance import cdist
from sklearn.cluster import KMeans, SpectralClustering
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors, kneighbors_graph
from scipy import sparse
from scipy.sparse.linalg import eigsh

# Ensure project root is on PYTHONPATH for local imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.mnist_dataset import load_mnist_data
from src.scRNA_dataset import load_scRNA_data
# Import necessary utilities but not compute_minimax_offset since we have our own implementation
from src.mesh_sampling import KDTree, cdist
from src.embedding_algorithms import compute_embedding

# Configure logging
logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')
logger = logging.getLogger("cluster_ratio_analysis")

# Set up high-quality figure parameters
SMALL_SIZE = 12
MEDIUM_SIZE = 14
BIGGER_SIZE = 16

plt.rc('font', size=SMALL_SIZE)           # Default text sizes
plt.rc('axes', titlesize=MEDIUM_SIZE)     # Axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)     # x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)     # x tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)     # y tick labels
plt.rc('legend', fontsize=SMALL_SIZE)     # Legend
plt.rc('figure', titlesize=BIGGER_SIZE)   # Figure title

sns.set_style("whitegrid")
colors = sns.color_palette("husl", 8)


def create_colormap(n_colors):
    """Create a colormap with n distinct colors for cluster visualization."""
    base_cmap = cm.get_cmap('tab10' if n_colors <= 10 else 'tab20')
    if n_colors <= 20:
        return base_cmap
    
    # For more than 20 colors, create a custom colormap
    return ListedColormap(sns.color_palette("husl", n_colors))


def sample_data(X, n_samples, y=None, random_state=42, stratify=True):
    """
    Sample n_samples points from X, optionally using stratified sampling based on y.
    
    Parameters:
    -----------
    X : numpy.ndarray
        Full dataset
    n_samples : int
        Number of samples to take
    y : numpy.ndarray, optional
        Labels for stratification
    random_state : int
        Random seed
    stratify : bool
        Whether to use stratified sampling based on y
    
    Returns:
    --------
    X_sampled : numpy.ndarray
        Sampled data
    indices : numpy.ndarray
        Indices of sampled points
    """
    if n_samples >= X.shape[0]:
        return X, np.arange(X.shape[0])
    
    rng = np.random.default_rng(random_state)
    
    # If no labels or stratification not requested, do simple random sampling
    if y is None or not stratify:
        indices = rng.choice(X.shape[0], size=n_samples, replace=False)
        return X[indices], indices
        
    # Stratified sampling based on labels
    unique_labels = np.unique(y)
    n_classes = len(unique_labels)
    
    # Calculate the proportion of each class in the full dataset
    class_counts = np.array([np.sum(y == label) for label in unique_labels])
    class_proportions = class_counts / np.sum(class_counts)
    
    # Calculate number of samples per class
    samples_per_class = np.floor(class_proportions * n_samples).astype(int)
    
    # Adjust for rounding errors
    remaining = n_samples - np.sum(samples_per_class)
    if remaining > 0:
        # Add remaining samples to classes with the largest proportions
        sorted_idx = np.argsort(-class_proportions)
        for i in range(min(remaining, n_classes)):
            samples_per_class[sorted_idx[i]] += 1
            
    # Sample from each class
    indices = []
    for i, label in enumerate(unique_labels):
        class_indices = np.where(y == label)[0]
        if len(class_indices) < samples_per_class[i]:
            logger.warning(f"Class {label} has fewer samples ({len(class_indices)}) than requested ({samples_per_class[i]})")
            # Take all available samples for this class
            indices.extend(class_indices)
            # Redistribute the deficit to other classes
            deficit = samples_per_class[i] - len(class_indices)
            if deficit > 0:
                other_classes = [j for j in range(n_classes) if j != i]
                if other_classes:
                    additions = np.zeros(n_classes, dtype=int)
                    for j in range(deficit):
                        additions[other_classes[j % len(other_classes)]] += 1
                    samples_per_class += additions
                    samples_per_class[i] = len(class_indices)
        else:
            # Randomly sample from this class
            sampled = rng.choice(class_indices, size=samples_per_class[i], replace=False)
            indices.extend(sampled)
    
    # Shuffle the combined indices
    indices = np.array(indices)
    if len(indices) > 0:
        indices = rng.permutation(indices)
        
    return X[indices], indices


def compute_fill_distance_point(X, point_idx, knn_k=None):
    """
    Compute fill distance for a specific point using kNN criterion.
    
    Parameters:
    -----------
    X : numpy.ndarray
        Data points containing the target point
    point_idx : int
        Index of the point for which to compute fill distance
    knn_k : int, optional
        Number of neighbors for kNN. If None, calculated based on data dimensionality.
    
    Returns:
    --------
    fill_distance : float
        kNN distance for the specified point
    """
    # If only one point exists, return 0
    if X.shape[0] <= 1:
        return 0.0
    
    # If k is not specified, calculate based on data dimensionality and size
    if knn_k is None:
        d = X.shape[1]
        n = X.shape[0]
        knn_k = int(np.round(d + np.log2(n / 0.05)))
    
    # Ensure k is not larger than number of points - 1
    knn_k = min(knn_k, X.shape[0] - 1)
    
    # Compute k nearest neighbors
    nbrs = NearestNeighbors(n_neighbors=knn_k+1, algorithm='auto').fit(X)
    distances, _ = nbrs.kneighbors(X[point_idx:point_idx+1])
    
    # Extract k-th neighbor distance (first element is the point itself with distance 0)
    knn_dist = distances[0, knn_k]
    
    return knn_dist


def compute_fill_distances(X, knn_k=None):
    """
    Compute fill distances for each point using kNN criterion.
    
    Parameters:
    -----------
    X : numpy.ndarray
        Data points
    knn_k : int, optional
        Number of neighbors for kNN. If None, calculated based on data dimensionality.
    
    Returns:
    --------
    fill_distance_max : float
        Maximum kNN distance
    """
    if X.shape[0] <= 1:
        return 0.0
    
    # If k is not specified, calculate based on data dimensionality and size
    if knn_k is None:
        d = X.shape[1]
        n = X.shape[0]
        knn_k = int(np.round(d + np.log2(n / 0.05)))
    
    # Ensure k is not larger than number of points - 1
    knn_k = min(knn_k, X.shape[0] - 1)
    
    # Compute k nearest neighbors
    nbrs = NearestNeighbors(n_neighbors=knn_k+1, algorithm='auto').fit(X)
    distances, _ = nbrs.kneighbors(X)
    
    # distances[:, 0] is distance to self (0), so k-th nearest neighbor is at index k
    knn_dists = distances[:, knn_k]
    
    return np.max(knn_dists)


def compute_cluster_metrics(X, k, random_state=42, clustering_method='kmeans', affinity='nearest_neighbors', 
                   n_neighbors=16, fill_distance_type='global'):
    """
    Perform clustering (k-means or spectral) and compute:
    1. Fill distances for each cluster
    2. Minimax offsets between each pair of clusters
    3. Ratios of offset to fill distance
    
    Parameters:
    -----------
    X : numpy.ndarray
        Data points
    k : int
        Number of clusters
    random_state : int
        Random seed for reproducibility
    clustering_method : str
        Clustering method to use ('kmeans' or 'spectral')
    affinity : str
        Affinity type for spectral clustering ('rbf', 'nearest_neighbors', etc.)
    n_neighbors : int
        Number of neighbors for affinity='nearest_neighbors'
    fill_distance_type : str
        Method for computing fill distances: 
        'global' - compute cluster-wide fill distance (traditional method)
        'local' - compute local fill distance at closest points between clusters
        
    Returns:
    --------
    dict
        Dictionary containing cluster assignments, fill distances, offsets, and ratios
    """
    # Perform clustering based on selected method
    if clustering_method.lower() == 'kmeans':
        # K-means clustering
        logger.info(f"Using K-means clustering with k={k}")
        clusterer = KMeans(n_clusters=k, random_state=random_state)
        labels = clusterer.fit_predict(X)
        centers = clusterer.cluster_centers_
    
    elif clustering_method.lower() == 'spectral':
        # Spectral clustering
        logger.info(f"Using spectral clustering with k={k}, affinity={affinity}, n_neighbors={n_neighbors}")
        labels = optimized_spectral_clustering(X, k, affinity=affinity, n_neighbors=n_neighbors, random_state=random_state)
        
        # Compute "virtual" centers for each cluster since SpectralClustering doesn't provide centers
        centers = np.zeros((k, X.shape[1]))
        for i in range(k):
            cluster_points = X[labels == i]
            if len(cluster_points) > 0:
                centers[i] = np.mean(cluster_points, axis=0)
    
    else:
        raise ValueError(f"Unsupported clustering method: {clustering_method}. Use 'kmeans' or 'spectral'.")
    
    # Compute global fill distance for each cluster (will be used for global method)
    fill_distances = []
    for i in range(k):
        cluster_points = X[labels == i]
        if len(cluster_points) > 1:  # Skip empty or singleton clusters
            fill_dist = compute_fill_distances(cluster_points)
            fill_distances.append(fill_dist)
        else:
            fill_distances.append(np.nan)
    
    # Storage for local fill distances when using local method
    local_fill_distances = np.zeros((k, k))
    
    # Compute minimax offset between each pair of clusters
    offsets = np.zeros((k, k))
    closest_points = {}  # Store closest point pairs for local fill distance calculation
    
    for i in range(k):
        for j in range(i+1, k):
            if fill_distance_type == 'local':
                # For local method, also get the closest points
                offset, (idx1, idx2) = compute_minimax_offset(X, labels, cluster_indices=[i, j], return_points=True)
                closest_points[(i, j)] = (idx1, idx2)
            else:
                # For global method, just get the offset
                offset = compute_minimax_offset(X, labels, cluster_indices=[i, j])
                
            offsets[i, j] = offset
            offsets[j, i] = offset
    
    # For local fill distance method, compute fill distances at closest points
    if fill_distance_type == 'local':
        for i in range(k):
            cluster_i_points = X[labels == i]
            for j in range(i+1, k):
                if (i, j) in closest_points:
                    idx1, idx2 = closest_points[(i, j)]
                    
                    # Skip if we don't have valid closest points
                    if idx1 is None or idx2 is None:
                        local_fill_distances[i, j] = np.nan
                        local_fill_distances[j, i] = np.nan
                        continue
                    
                    try:
                        # Get local indices for closest points
                        if idx1 is not None and idx2 is not None:
                            # Compute fill distance for the point from cluster i
                            cluster_i_points = X[labels == i]
                            if len(cluster_i_points) > 1:  # Need at least 2 points to compute kNN
                                # Find the position of idx1 within cluster i's points
                                cluster_i_indices = np.where(labels == i)[0]
                                local_idx1 = np.where(cluster_i_indices == idx1)[0][0]
                                fill_dist1 = compute_fill_distance_point(cluster_i_points, local_idx1)
                            else:
                                fill_dist1 = np.nan
                                
                            # Compute fill distance for the point from cluster j
                            cluster_j_points = X[labels == j]
                            if len(cluster_j_points) > 1:  # Need at least 2 points to compute kNN
                                # Find the position of idx2 within cluster j's points
                                cluster_j_indices = np.where(labels == j)[0]
                                local_idx2 = np.where(cluster_j_indices == idx2)[0][0]
                                fill_dist2 = compute_fill_distance_point(cluster_j_points, local_idx2)
                            else:
                                fill_dist2 = np.nan
                        else:
                            fill_dist1 = fill_dist2 = np.nan
                    except Exception as e:
                        logger.warning(f"Error computing local fill distances for clusters {i},{j}: {e}")
                        fill_dist1 = fill_dist2 = np.nan
                    
                    # Use max of the two local fill distances
                    local_max_fill = max(fill_dist1, fill_dist2)
                    local_fill_distances[i, j] = local_max_fill
                    local_fill_distances[j, i] = local_max_fill
    
    # Compute ratios of offset to fill distance for each pair of clusters
    ratios = np.zeros((k, k))
    for i in range(k):
        for j in range(i+1, k):
            if fill_distance_type == 'local':
                # Use local fill distance for the ratio
                max_fill = local_fill_distances[i, j]
            else:
                # Use global fill distance for the ratio
                max_fill = max(fill_distances[i], fill_distances[j])
                
            if max_fill > 0 and not np.isnan(max_fill):
                ratio = offsets[i, j] / max_fill
            else:
                ratio = np.nan
                
            ratios[i, j] = ratio
            ratios[j, i] = ratio
    
    result = {
        "labels": labels,
        "centers": centers,
        "fill_distances": np.array(fill_distances),
        "offsets": offsets,
        "ratios": ratios,
        "fill_distance_type": fill_distance_type
    }
    
    # Add local fill distances if computed
    if fill_distance_type == 'local':
        result["local_fill_distances"] = local_fill_distances
        
    return result


def compute_minimax_offset(X, labels, cluster_indices=None, return_points=False):
    """
    Compute the minimax offset between clusters:
    1. For each pair of specified clusters, find the minimum distance between any two points
    2. Return this minimum distance and optionally the points that achieve this minimum
    
    Parameters:
    -----------
    X : numpy.ndarray
        Data points
    labels : numpy.ndarray
        Cluster labels for each point
    cluster_indices : list, optional
        List of two cluster indices to compute offset between.
        If None, uses the first two unique labels found.
    return_points : bool, optional
        If True, also return the indices of the closest points in each cluster
    
    Returns:
    --------
    float or tuple
        If return_points=False: The minimax offset value between the specified clusters
        If return_points=True: A tuple (min_dist, (idx1, idx2)) with the minimum distance and 
                              the indices of the closest points in each cluster
    """
    if X is None or labels is None or len(X) == 0:
        if return_points:
            return np.nan, (None, None)
        return np.nan
    
    # If specific cluster indices are provided, use those
    if cluster_indices and len(cluster_indices) == 2:
        label1, label2 = cluster_indices
    else:
        # Otherwise use the first two unique labels
        unique_labels = np.unique(labels)
        if len(unique_labels) < 2:
            if return_points:
                return np.nan, (None, None)
            return np.nan
        label1, label2 = unique_labels[:2]
    
    # Get points for each cluster
    points1 = X[labels == label1]
    points2 = X[labels == label2]
    
    if len(points1) == 0 or len(points2) == 0:
        if return_points:
            return np.nan, (None, None)
        return np.nan
    
    # Get original indices for points in each cluster
    indices1 = np.where(labels == label1)[0]
    indices2 = np.where(labels == label2)[0]
    
    # Compute all pairwise distances
    dists = cdist(points1, points2)
    
    # Find the minimum distance and corresponding points
    min_idx = np.unravel_index(np.argmin(dists), dists.shape)
    min_dist = dists[min_idx]
    
    # If requested, return the indices of the closest points
    if return_points:
        closest_point1_idx = indices1[min_idx[0]]  # Original index in X for the first point
        closest_point2_idx = indices2[min_idx[1]]  # Original index in X for the second point
        return min_dist, (closest_point1_idx, closest_point2_idx)
    
    return min_dist


def calculate_ratio_percentages(ratios, threshold=0.5):
    """
    Calculate percentage of cluster pairs with ratio > threshold.
    
    Parameters:
    -----------
    ratios : numpy.ndarray
        Matrix of ratios
    threshold : float
        Threshold for counting ratios
        
    Returns:
    --------
    float
        Percentage of cluster pairs with ratio > threshold
    """
    # Extract upper triangle (excluding diagonal)
    upper_triangle = ratios[np.triu_indices_from(ratios, k=1)]
    
    # Count ratios above threshold (ignoring NaNs)
    valid_ratios = upper_triangle[~np.isnan(upper_triangle)]
    if len(valid_ratios) == 0:
        return 0.0
    
    count_above = np.sum(valid_ratios > threshold)
    percentage = (count_above / len(valid_ratios)) * 100.0
    return percentage


def visualize_clusters_tsne(X, labels, title, filename):
    """
    Create t-SNE visualization of clusters.
    
    Parameters:
    -----------
    X : numpy.ndarray
        Data points
    labels : numpy.ndarray
        Cluster labels
    title : str
        Plot title
    filename : str
        Output filename
    """
    # Apply t-SNE for visualization
    tsne = TSNE(n_components=2, random_state=42)
    X_tsne = tsne.fit_transform(X)
    
    # Create plot
    plt.figure(figsize=(10, 8))
    
    unique_labels = np.unique(labels)
    n_clusters = len(unique_labels)
    colormap = create_colormap(n_clusters)
    
    for i, label in enumerate(unique_labels):
        mask = labels == label
        plt.scatter(X_tsne[mask, 0], X_tsne[mask, 1], color=colormap(i % n_clusters), 
                   alpha=0.7, s=30, edgecolor='none', label=f'Cluster {label}')
    
    plt.title(title)
    plt.legend(loc='best', bbox_to_anchor=(1.05, 1), fontsize='small')
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved cluster visualization to {filename}")


def visualize_ratio_heatmap(ratios, title, filename):
    """
    Create heatmap visualization of ratio matrix.
    
    Parameters:
    -----------
    ratios : numpy.ndarray
        Matrix of ratios
    title : str
        Plot title
    filename : str
        Output filename
    """
    plt.figure(figsize=(10, 8))
    
    # Mask the diagonal (self-ratios)
    mask = np.eye(ratios.shape[0], dtype=bool)
    
    # Create heatmap
    ax = sns.heatmap(ratios, annot=True, fmt=".2f", cmap="YlOrRd", mask=mask, 
                   vmin=0, vmax=2, cbar_kws={'label': 'Offset/Fill Distance Ratio'})
    
    # Add lines between clusters
    for i in range(ratios.shape[0]):
        ax.axhline(i, color='white', lw=1)
        ax.axvline(i, color='white', lw=1)
    
    plt.title(title)
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()
    logger.info(f"Saved ratio heatmap to {filename}")


def plot_ratio_percentages(results_df, output_file, group_by_method=False):
    """
    Create plot showing percentage of ratios > threshold vs. k.
    
    Parameters:
    -----------
    results_df : pandas.DataFrame
        DataFrame containing results for different sample sizes and k values
    output_file : str
        Output filename
    group_by_method : bool
        If True, group results by clustering method instead of sample size
    """
    plt.figure(figsize=(10, 8))
    
    if group_by_method and 'clustering_method' in results_df.columns:
        # Group by clustering method
        methods = sorted(results_df['clustering_method'].unique())
        sample_sizes = sorted(results_df['sample_size'].unique())
        
        # Create a different color palette for each method
        method_colors = dict(zip(methods, sns.color_palette("tab10", len(methods))))
        
        # Create a color for each size-method combination
        for i, method in enumerate(methods):
            method_df = results_df[results_df['clustering_method'] == method]
            
            for j, size in enumerate(sample_sizes):
                size_df = method_df[method_df['sample_size'] == size]
                if len(size_df) > 0:
                    k_values = size_df['k']
                    percentages = size_df['percentage_above_threshold']
                    plt.plot(k_values, percentages, 'o-', 
                           color=method_colors[method], 
                           linestyle=['-', '--', '-.', ':'][j % 4],
                           label=f'{method}, {size} samples', 
                           linewidth=2, markersize=8)
    else:
        # Group by sample size (original behavior)
        sample_sizes = sorted(results_df['sample_size'].unique())
        
        # Plot line for each sample size
        for i, size in enumerate(sample_sizes):
            # If we have multiple clustering methods, further group by method
            if 'clustering_method' in results_df.columns and len(results_df['clustering_method'].unique()) > 1:
                methods = sorted(results_df['clustering_method'].unique())
                for j, method in enumerate(methods):
                    subset = results_df[(results_df['sample_size'] == size) & 
                                     (results_df['clustering_method'] == method)]
                    if len(subset) > 0:
                        k_values = subset['k']
                        percentages = subset['percentage_above_threshold']
                        plt.plot(k_values, percentages, 'o-', 
                               color=colors[i % len(colors)],
                               linestyle=['-', '--', '-.', ':'][j % 4],
                               label=f'{size} samples, {method}',
                               linewidth=2, markersize=8)
            else:
                # Single method case
                size_df = results_df[results_df['sample_size'] == size]
                k_values = size_df['k']
                percentages = size_df['percentage_above_threshold']
                plt.plot(k_values, percentages, 'o-', color=colors[i % len(colors)], 
                       label=f'{size} samples', linewidth=2, markersize=8)
    
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.xlabel('Number of Clusters (k)')
    plt.ylabel('% of Cluster Pairs with Ratio > 0.5')
    plt.title('Percentage of Well-Separated Cluster Pairs vs. Number of Clusters')
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_file, dpi=300)
    plt.close()
    logger.info(f"Saved ratio percentage plot to {output_file}")


def analyze_dataset(dataset_name, sample_sizes, output_dir, clustering_method='kmeans', 
                 affinity='rbf', n_neighbors=10, threshold=0.5, fill_distance_type='global'):
    """
    Perform complete analysis on a dataset:
    1. Load data
    2. For each sample size:
       a. Sample the data
       b. For each k value:
          i. Compute clusters, fill distances, offsets, and ratios using the specified clustering method
          ii. Calculate percentage of ratios > threshold
          iii. Create visualizations
    3. Create final plots and save results
    
    Parameters:
    -----------
    dataset_name : str
        Name of dataset ('mnist' or 'scRNA')
    sample_sizes : list
        List of sample sizes to analyze
    output_dir : str
        Output directory for results
    clustering_method : str
        Method used for clustering ('kmeans' or 'spectral')
    affinity : str
        Affinity to use with spectral clustering ('rbf', 'nearest_neighbors', 'precomputed')
    n_neighbors : int
        Number of neighbors for 'nearest_neighbors' affinity
    threshold : float
        Threshold value for calculating percentage of well-separated clusters
    """
    logger.info(f"Analyzing {dataset_name} dataset")
    
    # Load dataset
    if dataset_name.lower() == 'mnist':
        X, true_labels = load_mnist_data(n_samples=max(sample_sizes), shuffle=True, stratify=True)
        dataset_label = "MNIST"
        # Sensible k range for MNIST (digits 0-9)
        k_range = range(2, 21)
        logger.info(f"Loaded MNIST dataset with {X.shape[0]} samples and stratified by digit class")
    
    elif dataset_name.lower() == 'scrna':
        # Use the scRNA_data directory for the updated loader
        X, cell_types = load_scRNA_data(data_dir="scRNA_data", n_samples=max(sample_sizes), 
                                     shuffle=True, stratify=True)
        true_labels = cell_types  # Use cell types as the true labels
        dataset_label = "scRNA"
        # Sensible k range for scRNA (typically more clusters)
        k_range = range(2, 31)
        logger.info(f"Loaded scRNA dataset with {X.shape[0]} samples and stratified by cell type")
    
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")
    
    # Scale the data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Create results dataframe
    results = []
    
    # Analyze each sample size
    for sample_size in sample_sizes:
        logger.info(f"Processing sample size: {sample_size}")
        
        # Create sample size specific output directory
        sample_output_dir = os.path.join(output_dir, f"{dataset_label.lower()}_{sample_size}")
        os.makedirs(sample_output_dir, exist_ok=True)
        
        # Sample the data with stratification based on true labels
        X_sampled, indices = sample_data(X_scaled, sample_size, y=true_labels, stratify=True)
        
        # Get corresponding true labels for the sample
        if true_labels is not None:
            true_labels_sampled = true_labels[indices]
            
            # Log the distribution of true labels in the sample to verify stratification
            if len(np.unique(true_labels)) <= 20:  # Only log if there aren't too many classes
                unique_labels = np.unique(true_labels_sampled)
                label_counts = {str(label): np.sum(true_labels_sampled == label) for label in unique_labels}
                logger.info(f"Label distribution in {sample_size} sample: {label_counts}")
        else:
            true_labels_sampled = None
        
        # For each k value
        for k in k_range:
            logger.info(f"  Computing metrics for k={k}")
            
            # Compute cluster metrics with the specified clustering method
            metrics = compute_cluster_metrics(
                X_sampled, 
                k, 
                random_state=42,
                clustering_method=clustering_method,
                affinity=affinity,
                n_neighbors=n_neighbors,
                fill_distance_type=fill_distance_type
            )
            
            # Calculate percentage of ratios above threshold
            percentage = calculate_ratio_percentages(metrics["ratios"], threshold=threshold)
            
            # Add to results
            results.append({
                "dataset": dataset_label,
                "sample_size": sample_size,
                "k": k,
                "clustering_method": clustering_method,
                "percentage_above_threshold": percentage,
                "mean_ratio": np.nanmean(metrics["ratios"][np.triu_indices_from(metrics["ratios"], k=1)]),
                "min_ratio": np.nanmin(metrics["ratios"][np.triu_indices_from(metrics["ratios"], k=1)]),
                "max_ratio": np.nanmax(metrics["ratios"][np.triu_indices_from(metrics["ratios"], k=1)]),
                "mean_offset": np.nanmean(metrics["offsets"][np.triu_indices_from(metrics["offsets"], k=1)]),
                "mean_fill_distance": np.nanmean(metrics["fill_distances"]),
                "threshold": threshold
            })
            
            # Create visualizations for specific k values (to avoid too many plots)
            if k in [2, 3, 5, 6, 10, 15, 20]:  # Selected k values for visualization
                # Create a method suffix for filenames
                method_suffix = f"_{clustering_method}" if clustering_method != 'kmeans' else ""
                if fill_distance_type == 'local':
                    method_suffix += "_local"
                
                # Create k-specific output directory for embeddings
                k_output_dir = os.path.join(sample_output_dir, f"k{k}{method_suffix}")
                os.makedirs(k_output_dir, exist_ok=True)
                
                # Multi-embedding cluster visualization (UMAP, t-SNE, PCA)
                visualize_clusters_multi_embedding(
                    X_sampled,
                    metrics["labels"],
                    f"{dataset_label} - {sample_size} samples, k={k} ({clustering_method})",
                    k_output_dir,
                    embedding_methods=['tsne', 'umap', 'pca']
                )
                
                # Traditional t-SNE visualization (for backward compatibility)
                visualize_clusters_tsne(
                    X_sampled, 
                    metrics["labels"],
                    f"{dataset_label} - {sample_size} samples, k={k} ({clustering_method})",
                    os.path.join(k_output_dir, f"clusters_tsne_legacy.png")
                )
                
                # Ratio heatmap
                visualize_ratio_heatmap(
                    metrics["ratios"],
                    f"{dataset_label} - {sample_size} samples, k={k} - Offset/Fill Distance Ratios ({clustering_method})",
                    os.path.join(k_output_dir, f"ratio_heatmap.png")
                )
                
                # Multi-embedding visualization
                visualize_clusters_multi_embedding(
                    X_sampled,
                    metrics["labels"],
                    f"{dataset_label} - {sample_size} samples, k={k} ({clustering_method})",
                    sample_output_dir
                )
    
    # Convert results to dataframe
    results_df = pd.DataFrame(results)
    
    # Create method suffix for filenames
    method_suffix = f"_{clustering_method}" if clustering_method != 'kmeans' else ""
    
    # Save results to CSV
    csv_file = os.path.join(output_dir, f"{dataset_label.lower()}_cluster_ratio_results{method_suffix}.csv")
    results_df.to_csv(csv_file, index=False)
    logger.info(f"Saved results to {csv_file}")
    
    # Create summary plot
    plot_file = os.path.join(output_dir, f"{dataset_label.lower()}_ratio_percentage_plot{method_suffix}.png")
    plot_ratio_percentages(results_df, plot_file)
    
    # If we have multiple methods, create a comparison plot
    if 'clustering_method' in results_df.columns and len(results_df['clustering_method'].unique()) > 1:
        comparison_plot = os.path.join(output_dir, f"{dataset_label.lower()}_method_comparison_plot.png")
        plot_ratio_percentages(results_df, comparison_plot, group_by_method=True)
    
    # Create individual plots for each sample size
    for sample_size in sample_sizes:
        sample_df = results_df[results_df['sample_size'] == sample_size]
        
        # Check if we have multiple clustering methods
        if 'clustering_method' in sample_df.columns and len(sample_df['clustering_method'].unique()) > 1:
            # Create a plot that compares methods for this sample size
            sample_plot_file = os.path.join(output_dir, f"{dataset_label.lower()}_{sample_size}_method_comparison.png")
            
            plt.figure(figsize=(8, 6))
            
            for i, method in enumerate(sorted(sample_df['clustering_method'].unique())):
                method_df = sample_df[sample_df['clustering_method'] == method]
                plt.plot(method_df['k'], method_df['percentage_above_threshold'], 'o-', 
                       label=method, color=colors[i % len(colors)],
                       linewidth=2, markersize=8)
            
            plt.grid(True, linestyle='--', alpha=0.7)
            plt.xlabel('Number of Clusters (k)')
            plt.ylabel(f'% of Cluster Pairs with Ratio > {threshold}')
            plt.title(f'{dataset_label} - {sample_size} Samples\nComparison of Clustering Methods')
            plt.legend()
            plt.tight_layout()
            plt.savefig(sample_plot_file, dpi=300)
            plt.close()
        
        # Create individual plots for each method
        methods = ['kmeans'] if 'clustering_method' not in sample_df.columns else sorted(sample_df['clustering_method'].unique())
        for method in methods:
            # Filter by method if applicable
            if 'clustering_method' in sample_df.columns:
                method_df = sample_df[sample_df['clustering_method'] == method]
            else:
                method_df = sample_df
                
            method_suffix = f"_{method}" if method != 'kmeans' else ""
            sample_plot_file = os.path.join(output_dir, f"{dataset_label.lower()}_{sample_size}_ratio_plot{method_suffix}.png")
            
            plt.figure(figsize=(8, 6))
            plt.plot(method_df['k'], method_df['percentage_above_threshold'], 'o-', 
                   color='blue', linewidth=2, markersize=8)
            plt.grid(True, linestyle='--', alpha=0.7)
            plt.xlabel('Number of Clusters (k)')
            plt.ylabel(f'% of Cluster Pairs with Ratio > {threshold}')
            plt.title(f'{dataset_label} - {sample_size} Samples, {method.capitalize()}\nPercentage of Well-Separated Cluster Pairs')
            plt.tight_layout()
            plt.savefig(sample_plot_file, dpi=300)
            plt.close()
    
    return results_df


def visualize_clusters_multi_embedding(X, labels, title_prefix, output_dir, embedding_methods=None):
    """
    Create visualizations of clusters using multiple embedding techniques
    (t-SNE, UMAP, PCA) for better analysis.
    
    Parameters:
    -----------
    X : numpy.ndarray
        Data points
    labels : numpy.ndarray
        Cluster labels
    title_prefix : str
        Prefix for plot titles
    output_dir : str
        Directory to save visualizations
    embedding_methods : list, optional
        List of embedding methods to use. Defaults to ['tsne', 'umap', 'pca']
        
    Returns:
    --------
    dict
        Dictionary containing embedding results for each method
    """
    if embedding_methods is None:
        embedding_methods = ['tsne', 'umap', 'pca']
        
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize results dictionary
    embedding_results = {}
    
    # Get cluster information
    unique_labels = np.unique(labels)
    n_clusters = len(unique_labels)
    colormap = create_colormap(n_clusters)
    
    # Process each embedding method
    for method_name in embedding_methods:
        try:
            logger.info(f"Generating {method_name.upper()} embedding...")
            start_time = time.time()
            
            if method_name.lower() == 'tsne':
                # Use t-SNE with tuned parameters for better performance
                tsne = TSNE(
                    n_components=2,
                    perplexity=min(30, X.shape[0] // 5),  # Adaptive perplexity
                    n_iter=1000,
                    random_state=42,
                    init='pca'  # Use PCA initialization for better stability
                )
                embedded_data = tsne.fit_transform(X)
                
            elif method_name.lower() == 'umap':
                # Import UMAP here to avoid requiring it for other functions
                try:
                    import umap
                    # UMAP with tuned parameters
                    reducer = umap.UMAP(
                        n_components=2,
                        n_neighbors=min(15, X.shape[0] // 5),  # Adaptive neighbors
                        min_dist=0.1,
                        metric='euclidean',
                        random_state=42
                    )
                    embedded_data = reducer.fit_transform(X)
                except ImportError:
                    logger.warning("UMAP not installed. Install with 'pip install umap-learn'")
                    continue
                    
            elif method_name.lower() == 'pca':
                # PCA is straightforward and fast
                pca = PCA(n_components=2)
                embedded_data = pca.fit_transform(X)
                # Record variance explained
                embedding_results[f'{method_name}_variance_explained'] = pca.explained_variance_ratio_.sum()
                
            else:
                # Use the general compute_embedding function for other methods
                # (will use embedding_algorithms.py implementations)
                embedded_data, _ = compute_embedding(X, method=method_name.lower(), n_components=2)
            
            logger.info(f"{method_name.upper()} completed in {time.time() - start_time:.2f} seconds")
            
            # Save the embedding data for return
            embedding_results[method_name] = embedded_data
            
            # Create visualization figure
            plt.figure(figsize=(10, 8))
            
            # Plot each cluster with a different color
            for i, label in enumerate(unique_labels):
                mask = labels == label
                plt.scatter(embedded_data[mask, 0], embedded_data[mask, 1], 
                          color=colormap(i % n_clusters), 
                          alpha=0.7, s=30, edgecolor='none',
                          label=f'Cluster {label}')
                
            # Add title and legend
            plt.title(f"{title_prefix} - {method_name.upper()} Projection")
            plt.legend(loc='best', bbox_to_anchor=(1.05, 1), fontsize='small')
            plt.tight_layout()
            
            # Save figure
            filename = os.path.join(output_dir, f"clusters_{method_name.lower()}.png")
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            plt.close()
            logger.info(f"Saved {method_name.upper()} visualization to {filename}")
            
        except Exception as e:
            logger.error(f"Error generating {method_name} embedding: {e}")
            continue
            
    return embedding_results


def optimized_spectral_clustering(X, n_clusters, affinity='rbf', n_neighbors=10, random_state=42):
    """
    Optimized implementation of spectral clustering that uses sparse matrices and
    efficient eigensolvers for better performance.
    
    Parameters:
    -----------
    X : numpy.ndarray
        Data matrix of shape (n_samples, n_features)
    n_clusters : int
        Number of clusters
    affinity : str
        Affinity type: 'rbf', 'nearest_neighbors', 'precomputed'
    n_neighbors : int
        Number of neighbors for nearest_neighbors affinity
    random_state : int
        Random seed for reproducibility
    
    Returns:
    --------
    labels : numpy.ndarray
        Cluster labels for each point
    """
    n_samples = X.shape[0]
    
    # For small datasets, use sklearn's implementation
    if n_samples < 1000:
        clusterer = SpectralClustering(
            n_clusters=n_clusters,
            affinity=affinity,
            random_state=random_state,
            n_neighbors=n_neighbors
        )
        return clusterer.fit_predict(X)
    
    # For larger datasets, use optimized implementation
    logger.info("Using optimized spectral clustering implementation for large dataset")
    
    # Step 1: Create affinity matrix
    if affinity == 'precomputed':
        # Assume X is already a precomputed affinity matrix
        affinity_matrix = X
    elif affinity == 'nearest_neighbors':
        # Create sparse KNN graph
        connectivity = kneighbors_graph(
            X, n_neighbors=n_neighbors, 
            include_self=False,
            mode='connectivity'
        )
        # Make the graph symmetric
        affinity_matrix = 0.5 * (connectivity + connectivity.T)
    elif affinity == 'rbf':
        # Use KNN + RBF for sparse similarity matrix
        nn = NearestNeighbors(n_neighbors=min(n_neighbors, n_samples-1))
        nn.fit(X)
        distances, indices = nn.kneighbors(X)
        
        # Compute sigma as the median distance
        sigma = np.median(distances[:, -1]) if distances.size > 0 else 1.0
        
        # Create sparse affinity matrix using precomputed neighbors
        rows, cols, vals = [], [], []
        for i in range(n_samples):
            for j_idx, j in enumerate(indices[i]):
                if i != j:  # Skip self-connections
                    dist = distances[i, j_idx]
                    weight = np.exp(-dist**2 / (2 * sigma**2))
                    rows.append(i)
                    cols.append(j)
                    vals.append(weight)
        
        affinity_matrix = sparse.csr_matrix(
            (vals, (rows, cols)), 
            shape=(n_samples, n_samples)
        )
        # Make sure it's symmetric
        affinity_matrix = 0.5 * (affinity_matrix + affinity_matrix.T)
    else:
        raise ValueError(f"Unknown affinity type: {affinity}")
    
    # Step 2: Compute normalized graph Laplacian
    # Degree matrix
    degrees = np.array(affinity_matrix.sum(axis=1)).flatten()
    
    # Check if graph is connected
    if np.min(degrees) < 1e-10:
        logger.warning("Graph contains isolated nodes, adding small connections")
        # Add small connection to avoid disconnected graph
        n_connected = np.sum(degrees > 1e-10)
        if n_connected < n_samples:
            logger.warning(f"Only {n_connected} out of {n_samples} nodes are connected")
            affinity_matrix = affinity_matrix + sparse.eye(n_samples) * 1e-8
            degrees = np.array(affinity_matrix.sum(axis=1)).flatten()
    
    # Compute D^(-1/2)
    d_inv_sqrt = sparse.diags(1.0 / np.sqrt(degrees))
    
    # Normalized Laplacian: I - D^(-1/2) * A * D^(-1/2)
    laplacian = sparse.eye(n_samples) - d_inv_sqrt @ affinity_matrix @ d_inv_sqrt
    
    # Step 3: Find eigenvalues and eigenvectors
    logger.info("Computing eigendecomposition...")
    try:
        # Use eigsh which is faster for symmetric matrices
        eigenvalues, eigenvectors = eigsh(
            laplacian, k=n_clusters, which='SM', tol=1e-3
        )
    except Exception as e:
        logger.warning(f"Error in sparse eigendecomposition: {e}. Falling back to dense.")
        # Fall back to dense calculation for problematic cases
        eigenvalues, eigenvectors = np.linalg.eigh(laplacian.toarray())
        eigenvectors = eigenvectors[:, :n_clusters]
    
    # Step 4: Normalize rows of eigenvectors
    for i in range(eigenvectors.shape[0]):
        norm = np.sqrt(np.sum(eigenvectors[i, :]**2))
        if norm > 1e-10:
            eigenvectors[i, :] = eigenvectors[i, :] / norm
    
    # Step 5: Cluster using K-means
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    labels = kmeans.fit_predict(eigenvectors)
    
    return labels


def main():
    parser = argparse.ArgumentParser(description="Analyze cluster ratio metrics for real-world datasets")
    parser.add_argument("--dataset", choices=['mnist', 'scRNA'], required=True, 
                      help="Dataset to analyze (mnist or scRNA)")
    parser.add_argument("--output_dir", default="cluster_ratio_analysis", 
                      help="Output directory for results")
    parser.add_argument("--sample_sizes", default="2000,1000,500,250",
                      help="Comma-separated list of sample sizes")
    parser.add_argument("--random_seed", type=int, default=42,
                      help="Random seed for reproducibility")
    parser.add_argument("--clustering", choices=['kmeans', 'spectral'], default='kmeans',
                      help="Clustering method to use (kmeans or spectral)")
    parser.add_argument("--affinity", choices=['rbf', 'nearest_neighbors', 'precomputed'], default='rbf',
                      help="Affinity type for spectral clustering")
    parser.add_argument("--n_neighbors", type=int, default=10,
                      help="Number of neighbors for spectral clustering (required for all affinity types, but only used with nearest_neighbors)")
    parser.add_argument("--threshold", type=float, default=0.5,
                      help="Threshold value for ratio of offset to fill distance")
    parser.add_argument("--fill_distance_type", choices=['global', 'local'], default='global',
                      help="Method for computing fill distances: 'global' uses cluster-wide fill distance, 'local' uses fill distance at closest points")
    
    args = parser.parse_args()
    
    # Parse sample sizes
    sample_sizes = [int(size) for size in args.sample_sizes.split(',')]
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Set random seed
    np.random.seed(args.random_seed)
    
    # Analyze the dataset
    start_time = time.time()
    dataset_results = analyze_dataset(
        args.dataset, 
        sample_sizes, 
        args.output_dir,
        clustering_method=args.clustering,
        affinity=args.affinity,
        n_neighbors=args.n_neighbors,
        threshold=args.threshold,
        fill_distance_type=args.fill_distance_type
    )
    end_time = time.time()
    
    logger.info(f"Analysis completed in {end_time - start_time:.2f} seconds")
    
    # Create a simple report
    # Create file suffix based on clustering method and fill distance type
    method_suffix = f"_{args.clustering}" if args.clustering != 'kmeans' else ""
    if args.fill_distance_type == 'local':
        method_suffix += "_local"
    report_file = os.path.join(args.output_dir, f"{args.dataset}_analysis_report{method_suffix}.md")
    
    with open(report_file, 'w') as f:
        f.write(f"# Cluster Ratio Analysis for {args.dataset.upper()}\n\n")
        f.write(f"Analysis date: {time.strftime('%Y-%m-%d %H:%M')}\n\n")
        
        f.write("## Analysis Parameters\n\n")
        f.write(f"- Clustering method: {args.clustering}\n")
        if args.clustering == 'spectral':
            f.write(f"- Affinity: {args.affinity}\n")
            if args.affinity == 'nearest_neighbors':
                f.write(f"- Number of neighbors: {args.n_neighbors}\n")
        f.write(f"- Ratio threshold: {args.threshold}\n")
        f.write(f"- Fill distance type: {args.fill_distance_type}\n\n")
        
        f.write("## Summary\n\n")
        f.write("This analysis examines how the ratio of minimax offsets to fill distances changes\n")
        f.write(f"as we vary the number of clusters (k) using {args.clustering} clustering. ")
        f.write(f"A high ratio (>{args.threshold}) indicates well-separated clusters.\n\n")
        
        f.write("## Key Findings\n\n")
        
        # For each sample size, find the k with highest percentage
        f.write("Optimal number of clusters based on maximum percentage of well-separated pairs:\n\n")
        f.write("| Sample Size | Best k | % Well-Separated Pairs |\n")
        f.write("|------------|--------|------------------------|\n")
        
        for size in sample_sizes:
            size_df = dataset_results[dataset_results['sample_size'] == size]
            best_row = size_df.loc[size_df['percentage_above_threshold'].idxmax()]
            f.write(f"| {size} | {int(best_row['k'])} | {best_row['percentage_above_threshold']:.2f}% |\n")
        
        f.write("\n## Files Generated\n\n")
        f.write("- CSV results file with all metrics\n")
        f.write("- Overall plot showing percentage vs. k for all sample sizes\n")
        f.write("- Individual plots for each sample size\n")
        f.write("- Multi-embedding visualizations (t-SNE, UMAP, PCA) for selected k values\n")
        f.write("- Ratio heatmaps for selected k values\n")
        f.write("\n## Directory Structure\n\n")
        f.write("```\n")
        f.write(f"{args.output_dir}/\n")
        f.write(f"├── {args.dataset}_cluster_ratio_results{method_suffix}.csv  # Main results CSV\n")
        f.write(f"├── {args.dataset}_ratio_percentage_plot{method_suffix}.png  # Main plot\n")
        f.write(f"├── {args.dataset}_analysis_report{method_suffix}.md  # This report\n")
        f.write(f"└── {args.dataset}_[sample_size]/  # Sample-specific directories\n")
        f.write("    ├── k[value]_[method]/  # Cluster-specific directories\n")
        f.write("    │   ├── clusters_pca.png  # PCA visualization\n")
        f.write("    │   ├── clusters_tsne.png  # t-SNE visualization\n")
        f.write("    │   ├── clusters_umap.png  # UMAP visualization\n")
        f.write("    │   └── ratio_heatmap.png  # Cluster pair ratio heatmap\n")
        f.write("    └── [sample_size]_ratio_plot{method_suffix}.png  # Sample-specific plot\n")
        f.write("```\n")
        
        if args.clustering == 'spectral':
            f.write("\n## Spectral Clustering Notes\n\n")
            f.write("Spectral clustering may produce different results compared to k-means because it:\n")
            f.write("- Can detect non-convex clusters that k-means cannot find\n")
            f.write("- Uses pairwise similarities rather than Euclidean distances\n")
            f.write("- May be more sensitive to the local structure of the data\n")
            
            if args.affinity == 'rbf' and dataset_results.iloc[0]['sample_size'] > 5000:
                f.write("\n**Performance Note**: For large datasets (>5000 samples), consider using ")
                f.write("`--affinity nearest_neighbors` for better performance.\n")
    
    logger.info(f"Report generated at {report_file}")
    logger.info(f"All results saved to {args.output_dir}")


if __name__ == "__main__":
    main()
