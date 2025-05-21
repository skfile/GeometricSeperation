#!/usr/bin/env python3
"""
cluster_visualization.py
-------------------------
Specialized visualization tool for analyzing and visualizing:
1. Clusters in the dataset (original and embedded space)
2. Fill distances between different clusters
3. Visual representation of cluster separation

This script provides insights into how well the clusters are separated and
how sampling affects the cluster structure.

Usage:
  python cluster_visualization.py --config config_file.json [--output_dir output_directory/] [--dataset dataset_name]

Dependencies:
  - Uses code from mesh_sampling.py for data generation
  - Uses compute_embedding from embedding_algorithms.py for dimensionality reduction
  - matplotlib for visualization
"""

import os
import sys
import json
import argparse
import logging
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib import cm
import seaborn as sns
from scipy.spatial.distance import cdist
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from scipy.sparse import issparse

# Ensure project root is on PYTHONPATH for local src imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.mesh_sampling import generate_dataset_and_subsamples, compute_minimax_offset
from src.embedding_algorithms import compute_embedding

# Set up logging
logger = logging.getLogger("cluster_visualization")
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
logger.addHandler(handler)

def create_colormap(n_colors):
    """Create a colormap with n distinct colors for cluster visualization."""
    base_cmap = cm.get_cmap('tab10' if n_colors <= 10 else 'tab20')
    if n_colors <= 20:
        return base_cmap
    
    # For more than 20 colors, create a custom colormap
    return ListedColormap(sns.color_palette("husl", n_colors))

def visualize_original_data_with_clusters(X, labels, title, filename):
    """
    Visualize the raw data with cluster labels.
    
    Parameters:
    -----------
    X : numpy.ndarray
        The data points (n_samples, n_features)
    labels : numpy.ndarray
        Cluster labels for each point
    title : str
        Title for the plot
    filename : str
        Output filename for the plot
    """
    # Reduce dimensionality for visualization if needed
    if X.shape[1] > 2:
        pca = PCA(n_components=2)
        X_2d = pca.fit_transform(X)
        dim_reduction = "PCA"
    else:
        X_2d = X
        dim_reduction = "Original"
    
    unique_labels = np.unique(labels)
    n_clusters = len(unique_labels)
    colormap = create_colormap(n_clusters)
    
    plt.figure(figsize=(10, 8))
    
    for i, label in enumerate(unique_labels):
        mask = labels == label
        plt.scatter(X_2d[mask, 0], X_2d[mask, 1], c=[colormap(i)], label=f'Cluster {label}',
                   alpha=0.7, s=30, edgecolors='none')
    
    plt.title(f"{title}\n({dim_reduction} projection)")
    plt.legend(loc='best', fontsize='small')
    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.close()
    logger.info(f"Saved cluster visualization to {filename}")

def calculate_cluster_distances(X, labels):
    """
    Calculate minimum distances between clusters.
    
    Parameters:
    -----------
    X : numpy.ndarray
        The data points
    labels : numpy.ndarray
        Cluster labels for each point
        
    Returns:
    --------
    distance_matrix : numpy.ndarray
        Matrix of minimum distances between clusters
    """
    unique_labels = np.unique(labels)
    n_clusters = len(unique_labels)
    
    distance_matrix = np.zeros((n_clusters, n_clusters))
    
    for i, label1 in enumerate(unique_labels):
        points1 = X[labels == label1]
        
        for j, label2 in enumerate(unique_labels):
            if i == j:
                continue  # Skip same cluster comparisons
                
            points2 = X[labels == label2]
            
            # Calculate all pairwise distances between clusters
            dists = cdist(points1, points2)
            
            # Store the minimum distance
            min_dist = np.min(dists)
            distance_matrix[i, j] = min_dist
            distance_matrix[j, i] = min_dist  # Symmetric
    
    return distance_matrix, unique_labels

def visualize_cluster_distances(distance_matrix, labels, title, filename):
    """
    Create a heatmap visualization of distances between clusters.
    
    Parameters:
    -----------
    distance_matrix : numpy.ndarray
        Matrix of minimum distances between clusters
    labels : array-like
        Cluster labels (for axis labels)
    title : str
        Title for the plot
    filename : str
        Output filename for the plot
    """
    plt.figure(figsize=(10, 8))
    
    # Use seaborn for a nice heatmap
    ax = sns.heatmap(distance_matrix, annot=True, fmt=".2f", cmap="YlGnBu",
                    xticklabels=[f'C{l}' for l in labels],
                    yticklabels=[f'C{l}' for l in labels])
    
    plt.title(title)
    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.close()
    logger.info(f"Saved cluster distance heatmap to {filename}")

def visualize_fill_distances(X_full, X_sub, labels_full, title, filename):
    """
    Visualize the fill distances from full data to subsampled data.
    
    Parameters:
    -----------
    X_full : numpy.ndarray
        The full dataset
    X_sub : numpy.ndarray
        The subsampled dataset
    labels_full : numpy.ndarray
        Labels for the full dataset
    title : str
        Title for the plot
    filename : str
        Output filename for the plot
    """
    # Calculate distances from each full point to the nearest subsample point
    distances = np.min(cdist(X_full, X_sub), axis=1)
    
    # Reduce dimensionality for visualization if needed
    if X_full.shape[1] > 2:
        pca = PCA(n_components=2)
        X_full_2d = pca.fit_transform(X_full)
        X_sub_2d = pca.transform(X_sub)
        dim_reduction = "PCA"
    else:
        X_full_2d = X_full
        X_sub_2d = X_sub
        dim_reduction = "Original"
    
    plt.figure(figsize=(12, 10))
    
    # Create a scatter plot where color represents distance to nearest subsample
    scatter = plt.scatter(X_full_2d[:, 0], X_full_2d[:, 1], c=distances, cmap='viridis',
                         alpha=0.7, s=30, edgecolors='none')
    
    # Plot the subsample points
    plt.scatter(X_sub_2d[:, 0], X_sub_2d[:, 1], c='red', marker='x', s=50, label='Subsample')
    
    # Add a colorbar
    cbar = plt.colorbar(scatter)
    cbar.set_label('Distance to nearest subsample point')
    
    plt.title(f"{title}\nFill Distances ({dim_reduction} projection)")
    plt.legend(loc='best')
    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.close()
    logger.info(f"Saved fill distance visualization to {filename}")

def analyze_cluster_representation(full_labels, sub_indices):
    """
    Analyze how well each cluster is represented in the subsample.
    
    Parameters:
    -----------
    full_labels : numpy.ndarray
        Labels for the full dataset
    sub_indices : numpy.ndarray
        Indices of points in the subsample
        
    Returns:
    --------
    dict
        Statistics about cluster representation
    """
    unique_labels = np.unique(full_labels)
    n_clusters = len(unique_labels)
    
    # Count occurrences of each label in full dataset
    full_counts = {label: np.sum(full_labels == label) for label in unique_labels}
    
    # Count occurrences of each label in subsample
    sub_labels = full_labels[sub_indices]
    sub_counts = {label: np.sum(sub_labels == label) for label in unique_labels}
    
    # Calculate representation percentages
    representation = {label: (sub_counts[label] / full_counts[label] * 100) for label in unique_labels}
    
    return {
        'full_counts': full_counts,
        'sub_counts': sub_counts,
        'representation_pct': representation
    }

def visualize_cluster_representation(representation_stats, title, filename):
    """
    Create a bar chart showing how well each cluster is represented in the subsample.
    
    Parameters:
    -----------
    representation_stats : dict
        Statistics from analyze_cluster_representation
    title : str
        Title for the plot
    filename : str
        Output filename for the plot
    """
    labels = list(representation_stats['representation_pct'].keys())
    values = list(representation_stats['representation_pct'].values())
    full_counts = list(representation_stats['full_counts'].values())
    sub_counts = list(representation_stats['sub_counts'].values())
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # First subplot: representation percentages
    bars = ax1.bar(labels, values, color='skyblue')
    ax1.set_xlabel('Cluster')
    ax1.set_ylabel('Representation (%)')
    ax1.set_title('Cluster Representation in Subsample')
    ax1.set_ylim(0, 100)
    
    # Add percentage labels on top of bars
    for bar, percentage in zip(bars, values):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 2,
                f'{percentage:.1f}%',
                ha='center', va='bottom', rotation=0)
    
    # Second subplot: raw counts comparison
    x = np.arange(len(labels))
    width = 0.35
    
    ax2.bar(x - width/2, full_counts, width, label='Full Dataset')
    ax2.bar(x + width/2, sub_counts, width, label='Subsample')
    
    ax2.set_xlabel('Cluster')
    ax2.set_ylabel('Count')
    ax2.set_title('Cluster Size: Full vs Subsample')
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels)
    ax2.legend()
    
    # Add count labels on top of bars
    for i, (full, sub) in enumerate(zip(full_counts, sub_counts)):
        ax2.text(i - width/2, full + max(full_counts)*0.02, str(full), ha='center', va='bottom')
        ax2.text(i + width/2, sub + max(full_counts)*0.02, str(sub), ha='center', va='bottom')
    
    plt.suptitle(title)
    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.close()
    logger.info(f"Saved cluster representation analysis to {filename}")

def visualize_embeddings(X, labels, methods, title_prefix, filename_prefix):
    """
    Create visualizations of the data using multiple embedding methods.
    
    Parameters:
    -----------
    X : numpy.ndarray
        The data points
    labels : numpy.ndarray
        Cluster labels for each point
    methods : list
        List of embedding methods to use
    title_prefix : str
        Prefix for the plot titles
    filename_prefix : str
        Prefix for output filenames
    """
    unique_labels = np.unique(labels)
    n_clusters = len(unique_labels)
    colormap = create_colormap(n_clusters)
    
    for method in methods:
        logger.info(f"Computing {method} embedding...")
        
        try:
            if method.lower() == 'pca':
                # Use scikit-learn's PCA directly for simplicity
                embedding = PCA(n_components=2).fit_transform(X)
            elif method.lower() == 'tsne':
                # Use scikit-learn's t-SNE directly for simplicity
                embedding = TSNE(n_components=2, random_state=42).fit_transform(X)
            else:
                # For other methods like UMAP, diffusion maps, etc., use the compute_embedding function
                embedding, _ = compute_embedding(X, method=method.lower(), n_components=2)
            
            plt.figure(figsize=(10, 8))
            
            for i, label in enumerate(unique_labels):
                mask = labels == label
                plt.scatter(embedding[mask, 0], embedding[mask, 1], c=[colormap(i)], 
                           label=f'Cluster {label}', alpha=0.7, s=30, edgecolors='none')
            
            plt.title(f"{title_prefix} - {method} embedding")
            plt.legend(loc='best', fontsize='small')
            plt.tight_layout()
            
            filename = f"{filename_prefix}_{method.lower()}.png"
            plt.savefig(filename, dpi=150)
            plt.close()
            logger.info(f"Saved {method} embedding visualization to {filename}")
            
        except Exception as e:
            logger.error(f"Error computing {method} embedding: {e}")

def main():
    parser = argparse.ArgumentParser(description="Visualize clusters, fill distances, and separation in datasets")
    parser.add_argument("--config", required=True, help="Path to config JSON used for experiments")
    parser.add_argument("--output_dir", default="cluster_visualizations", help="Directory to store visualizations")
    parser.add_argument("--dataset", default=None, help="Specific dataset name to process (optional)")
    parser.add_argument("--embedding_methods", default="pca,tsne,umap,diffusionmap", 
                       help="Comma-separated list of embedding methods to use")
    args = parser.parse_args()
    
    # Load config
    with open(args.config, 'r') as f:
        config = json.load(f)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Parse embedding methods
    embedding_methods = args.embedding_methods.split(',')
    
    # Process datasets
    datasets = config["datasets"]
    
    for dset in datasets:
        dataset_name = dset.get("name", "UnnamedDataset")
        
        # Skip if not the specified dataset (if specified)
        if args.dataset and dataset_name != args.dataset:
            continue
            
        logger.info(f"Processing dataset: {dataset_name}")
        
        # Create dataset-specific output directory
        dataset_output_dir = os.path.join(args.output_dir, dataset_name)
        os.makedirs(dataset_output_dir, exist_ok=True)
        
        # Generate dataset with subsamples
        try:
            dataset = generate_dataset_and_subsamples(dset)
        except Exception as e:
            logger.error(f"Error generating dataset {dataset_name}: {e}")
            continue
            
        X_full = dataset.get("X_full")
        labels = dataset.get("labels")
        
        if X_full is None:
            logger.warning(f"No data points found for dataset {dataset_name}")
            continue
            
        if labels is None:
            logger.warning(f"No labels found for dataset {dataset_name}, skipping cluster visualizations")
            continue
        
        # Visualize original dataset with cluster labels
        visualize_original_data_with_clusters(
            X_full, 
            labels,
            f"Dataset: {dataset_name} - Original Data with Clusters", 
            os.path.join(dataset_output_dir, "original_clusters.png")
        )
        
        # Calculate and visualize cluster distances
        distance_matrix, unique_labels = calculate_cluster_distances(X_full, labels)
        visualize_cluster_distances(
            distance_matrix,
            unique_labels,
            f"Dataset: {dataset_name} - Minimum Distances Between Clusters",
            os.path.join(dataset_output_dir, "cluster_distances.png")
        )
        
        # Create embeddings of the full dataset with different methods
        visualize_embeddings(
            X_full,
            labels,
            embedding_methods,
            f"Dataset: {dataset_name}",
            os.path.join(dataset_output_dir, "embedding")
        )
        
        # Process each subsample
        for i, ssub in enumerate(dataset.get("sub_samples", [])):
            subsample_method = ssub.get("method", "unknown")
            subsample_fraction = ssub.get("fraction", 0.0)
            
            X_sub = ssub.get("X_sub")
            indices_sub = ssub.get("indices_sub")
            
            if X_sub is None or indices_sub is None:
                continue
                
            subsample_name = f"{subsample_method}_{subsample_fraction:.2f}"
            
            # Create subsample output directory
            subsample_output_dir = os.path.join(dataset_output_dir, subsample_name)
            os.makedirs(subsample_output_dir, exist_ok=True)
            
            # Visualize fill distances
            visualize_fill_distances(
                X_full,
                X_sub,
                labels,
                f"Dataset: {dataset_name}, Subsample: {subsample_name}",
                os.path.join(subsample_output_dir, "fill_distances.png")
            )
            
            # Analyze cluster representation in subsample
            representation_stats = analyze_cluster_representation(labels, indices_sub)
            visualize_cluster_representation(
                representation_stats,
                f"Dataset: {dataset_name}, Subsample: {subsample_name}",
                os.path.join(subsample_output_dir, "cluster_representation.png")
            )
            
            # Get fill distance values for reference
            fill_dist = ssub.get("fill_dist_orig", np.nan)
            fill_dist_scaled = ssub.get("fill_dist_scaled", np.nan)
            
            # Get minimax offset for reference
            minimax_offset = dataset.get("minimax_offset", np.nan)
            minimax_offset_scaled = dataset.get("minimax_offset_scaled", np.nan)
            
            # Save summary information
            with open(os.path.join(subsample_output_dir, "summary.txt"), 'w') as f:
                f.write(f"Dataset: {dataset_name}\n")
                f.write(f"Subsample method: {subsample_method}\n")
                f.write(f"Subsample fraction: {subsample_fraction:.4f}\n")
                f.write(f"Full dataset size: {X_full.shape[0]}\n")
                f.write(f"Subsample size: {X_sub.shape[0]}\n")
                f.write(f"Number of clusters: {len(np.unique(labels))}\n")
                f.write(f"Fill distance: {fill_dist:.4f}\n")
                f.write(f"Scaled fill distance: {fill_dist_scaled:.4f}\n")
                f.write(f"Minimax offset: {minimax_offset:.4f}\n")
                f.write(f"Scaled minimax offset: {minimax_offset_scaled:.4f}\n")
                if not np.isnan(minimax_offset) and not np.isnan(fill_dist) and fill_dist > 0:
                    f.write(f"Offset/Fill ratio: {minimax_offset/fill_dist:.4f}\n")
                
                # Add cluster representation statistics
                f.write("\nCluster representation statistics:\n")
                for label in np.unique(labels):
                    pct = representation_stats['representation_pct'][label]
                    full_count = representation_stats['full_counts'][label]
                    sub_count = representation_stats['sub_counts'][label]
                    f.write(f"  Cluster {label}: {sub_count}/{full_count} points ({pct:.2f}%)\n")
        
        logger.info(f"All visualizations for dataset {dataset_name} completed")
    
    logger.info("All datasets processed successfully")

if __name__ == "__main__":
    main()
