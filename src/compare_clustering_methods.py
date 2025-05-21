#!/usr/bin/env python3

"""
compare_clustering_methods.py
-----------------------------
Compares the cluster ratio analysis results from different clustering methods
(k-means, spectral with rbf kernel, spectral with nearest neighbors)
and creates visualization plots.

Usage:
  python compare_clustering_methods.py --dataset mnist|scRNA --output_dir output_directory
"""

import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

def load_results(base_dir, dataset):
    """
    Load results from different clustering methods for a given dataset
    
    Parameters:
    -----------
    base_dir : str
        Base directory containing results
    dataset : str
        Dataset name ('mnist' or 'scRNA')
        
    Returns:
    --------
    dict
        Dictionary of DataFrames with results for each clustering method
    """
    results = {}
    
    # Define the clustering methods and their directories
    methods = {
        'kmeans': 'kmeans',
        'spectral_rbf': 'spectral_rbf',
        'spectral_nn': 'spectral_nn'
    }
    
    for method_name, method_dir in methods.items():
        csv_path = os.path.join(base_dir, dataset, method_dir, 
                             f"{dataset}_cluster_ratio_results{'_spectral' if 'spectral' in method_name else ''}.csv")
        
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path)
            # Add a method column if it doesn't exist
            if 'clustering_method' not in df.columns:
                df['clustering_method'] = 'kmeans' if method_name == 'kmeans' else 'spectral'
            
            # Add a readable label for the method
            if method_name == 'kmeans':
                df['method_label'] = 'K-means'
            elif method_name == 'spectral_rbf':
                df['method_label'] = 'Spectral (RBF)'
            elif method_name == 'spectral_nn':
                df['method_label'] = 'Spectral (NN)'
                
            results[method_name] = df
    
    return results

def plot_method_comparison(results, dataset, output_dir):
    """
    Create comparison plots of the clustering methods
    
    Parameters:
    -----------
    results : dict
        Dictionary of DataFrames with results for each clustering method
    dataset : str
        Dataset name ('mnist' or 'scRNA')
    output_dir : str
        Output directory for saving plots
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Get all the sample sizes from the first method (assuming all have the same)
    if not results:
        print(f"No results found for {dataset}")
        return
    
    # Get the first available results DataFrame
    first_method = list(results.keys())[0]
    sample_sizes = sorted(results[first_method]['sample_size'].unique())
    
    # Method comparison plots for each sample size
    for size in sample_sizes:
        plt.figure(figsize=(10, 6))
        
        for method_name, df in results.items():
            size_df = df[df['sample_size'] == size]
            
            if not size_df.empty:
                plt.plot(size_df['k'], size_df['percentage_above_threshold'], 'o-', 
                       label=size_df['method_label'].iloc[0],
                       linewidth=2, markersize=8)
        
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.xlabel('Number of Clusters (k)')
        plt.ylabel('% of Cluster Pairs with Ratio > Threshold')
        plt.title(f'{dataset.upper()} - {size} Samples\nComparison of Clustering Methods')
        plt.legend()
        plt.tight_layout()
        
        output_file = os.path.join(output_dir, f"{dataset}_{size}_method_comparison.png")
        plt.savefig(output_file, dpi=300)
        plt.close()
        print(f"Created comparison plot for {size} samples: {output_file}")
    
    # Create a multi-panel figure with all sample sizes
    n_sizes = len(sample_sizes)
    fig_height = 4 * ((n_sizes + 1) // 2)  # Adjust height based on number of subplots
    fig, axes = plt.subplots(nrows=(n_sizes + 1) // 2, ncols=2, 
                           figsize=(14, fig_height), squeeze=False)
    
    # Method colors for consistency
    method_colors = {
        'kmeans': 'blue',
        'spectral_rbf': 'red',
        'spectral_nn': 'green'
    }
    
    for i, size in enumerate(sample_sizes):
        ax = axes[i // 2, i % 2]
        
        for method_name, df in results.items():
            size_df = df[df['sample_size'] == size]
            
            if not size_df.empty:
                ax.plot(size_df['k'], size_df['percentage_above_threshold'], 'o-', 
                      label=size_df['method_label'].iloc[0],
                      color=method_colors.get(method_name, 'gray'),
                      linewidth=2, markersize=6)
        
        ax.grid(True, linestyle='--', alpha=0.7)
        ax.set_xlabel('Number of Clusters (k)')
        ax.set_ylabel('% Well-Separated Pairs')
        ax.set_title(f'{size} Samples')
        
        # Only show legend in the first subplot
        if i == 0:
            ax.legend()
    
    # Hide any unused subplots
    for i in range(len(sample_sizes), (((n_sizes + 1) // 2) * 2)):
        axes[i // 2, i % 2].axis('off')
    
    plt.suptitle(f'Clustering Method Comparison for {dataset.upper()}', fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.97])  # Make room for the suptitle
    
    overview_file = os.path.join(output_dir, f"{dataset}_all_methods_comparison.png")
    plt.savefig(overview_file, dpi=300)
    plt.close()
    print(f"Created comprehensive comparison plot: {overview_file}")
    
    # Create a heatmap showing the difference between spectral and k-means
    if 'kmeans' in results and ('spectral_rbf' in results or 'spectral_nn' in results):
        plt.figure(figsize=(12, 8))
        
        # Choose which spectral method to compare against
        spectral_method = 'spectral_rbf' if 'spectral_rbf' in results else 'spectral_nn'
        
        # Create a matrix to hold the differences
        kmeans_df = results['kmeans']
        spectral_df = results[spectral_method]
        
        # Find common k values
        k_values = sorted(set(kmeans_df['k']).intersection(set(spectral_df['k'])))
        
        # Initialize the difference matrix
        diff_matrix = np.zeros((len(sample_sizes), len(k_values)))
        
        for i, size in enumerate(sample_sizes):
            for j, k in enumerate(k_values):
                kmeans_val = kmeans_df[(kmeans_df['sample_size'] == size) & (kmeans_df['k'] == k)]['percentage_above_threshold'].values
                spectral_val = spectral_df[(spectral_df['sample_size'] == size) & (spectral_df['k'] == k)]['percentage_above_threshold'].values
                
                if len(kmeans_val) > 0 and len(spectral_val) > 0:
                    # Calculate how much better spectral is than k-means (positive = spectral is better)
                    diff_matrix[i, j] = spectral_val[0] - kmeans_val[0]
        
        # Create heatmap
        sns.heatmap(diff_matrix, 
                  xticklabels=k_values, 
                  yticklabels=sample_sizes,
                  cmap='RdBu_r',  # Red-Blue diverging colormap
                  center=0,       # Center the colormap at 0
                  annot=True,     # Show values
                  fmt=".1f")      # Format as float with 1 decimal
        
        plt.title(f'{dataset.upper()}: {results[spectral_method]["method_label"].iloc[0]} vs K-means\n' +
                f'(Positive values = Spectral is better)')
        plt.xlabel('Number of Clusters (k)')
        plt.ylabel('Sample Size')
        plt.tight_layout()
        
        diff_file = os.path.join(output_dir, f"{dataset}_spectral_vs_kmeans_diff.png")
        plt.savefig(diff_file, dpi=300)
        plt.close()
        print(f"Created difference heatmap: {diff_file}")

def main():
    parser = argparse.ArgumentParser(description="Compare clustering methods for cluster ratio analysis")
    parser.add_argument("--dataset", choices=['mnist', 'scRNA'], required=True, 
                      help="Dataset to analyze (mnist or scRNA)")
    parser.add_argument("--output_dir", default="cluster_ratio_analysis/comparison", 
                      help="Output directory for comparison results")
    
    args = parser.parse_args()
    
    # Load results from different clustering methods
    base_dir = 'cluster_ratio_analysis'
    results = load_results(base_dir, args.dataset)
    
    if not results:
        print(f"No results found for {args.dataset}. Make sure to run the analysis first.")
        return
    
    # Create comparison plots
    plot_method_comparison(results, args.dataset, args.output_dir)
    
    print(f"Comparison complete. Results saved in {args.output_dir}")

if __name__ == "__main__":
    main()
