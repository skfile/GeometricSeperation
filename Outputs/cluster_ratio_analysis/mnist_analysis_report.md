# Cluster Ratio Analysis for MNIST

Analysis date: 2025-05-14 10:53

## Analysis Parameters

- Clustering method: kmeans
- Ratio threshold: 0.5
- Fill distance type: global

## Summary

This analysis examines how the ratio of minimax offsets to fill distances changes
as we vary the number of clusters (k) using kmeans clustering. A high ratio (>0.5) indicates well-separated clusters.

## Key Findings

Optimal number of clusters based on maximum percentage of well-separated pairs:

| Sample Size | Best k | % Well-Separated Pairs |
|------------|--------|------------------------|
| 2000 | 11 | 18.18% |
| 1000 | 20 | 31.03% |
| 500 | 7 | 47.62% |
| 250 | 20 | 68.31% |

## Files Generated

- CSV results file with all metrics
- Overall plot showing percentage vs. k for all sample sizes
- Individual plots for each sample size
- Multi-embedding visualizations (t-SNE, UMAP, PCA) for selected k values
- Ratio heatmaps for selected k values

## Directory Structure

```
cluster_ratio_analysis/
├── mnist_cluster_ratio_results.csv  # Main results CSV
├── mnist_ratio_percentage_plot.png  # Main plot
├── mnist_analysis_report.md  # This report
└── mnist_[sample_size]/  # Sample-specific directories
    ├── k[value]_[method]/  # Cluster-specific directories
    │   ├── clusters_pca.png  # PCA visualization
    │   ├── clusters_tsne.png  # t-SNE visualization
    │   ├── clusters_umap.png  # UMAP visualization
    │   └── ratio_heatmap.png  # Cluster pair ratio heatmap
    └── [sample_size]_ratio_plot{method_suffix}.png  # Sample-specific plot
```
