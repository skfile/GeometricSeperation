# GeometricSeperation: Codebase Analysis and Optimization Guide

## Overview
We provide a Python framework for kernel-based sampling experiments on real and synthetic datasets. It supports configurable pipelines for data generation, kernel construction, embedding, clustering, and comprehensive analysis/visualization. The codebase is designed for high-performance computing (HPC) environments, with support parallel processing and batch experiments.

---

## Codebase Structure and File Roles

### Top-Level Scripts
- **requirements.txt**: Python dependencies for the project.
- **experiment.log**: Log file for experiment runs.
- **config_*.json**: Experiment configuration files specifying datasets, kernels, embeddings, clustering, and other parameters.

### Main Experiment Pipeline
- **src/main_experiment.py**: The central driver for experiments. Handles config parsing, dataset expansion (by offset), parallelization, and result aggregation. It:
  - Validates config and expands datasets by offset.
  - Manages parallel execution at the dataset and kernel level using `joblib` and a custom `parallel_context`.
  - Calls `run_experiment_for_dataset` for each dataset, which:
    - Generates data and subsamples (via `mesh_sampling.py`).
    - Builds adjacency/cost matrices and computes ground-truth labels.
    - Runs kernel construction, Gromov-Wasserstein (GW), and embedding methods in parallel.
    - Computes clustering and ARI metrics.
    - Aggregates results into a CSV.
  - After all runs, triggers plotting/analysis scripts.
- **src/main_experiment_slurm.py**: SLURM array job driver. Each SLURM task runs a single (dataset, kernel) pair, writing a CSV shard. Used for large-scale distributed experiments.
- **src/test_experiment.py**: Dry-run pipeline for local validation. Skips heavy computations (GW, large embeddings), short-circuiting them for fast correctness checks and memory profiling.

### Data Generation and Sampling
- **src/mesh_sampling.py**: Generates synthetic datasets (unions of shapes, simplex-based Gaussians, MNIST, scRNA) and subsamples (uniform/biased). Handles adjacency construction, noise injection, and block-diagonal graph assembly.
- **src/dataset_funcs.py**: Utility functions for generating grids, synthetic manifolds, and custom datasets (e.g., grid, spiral, crown, stingray).
- **src/mnist_dataset.py, src/scRNA_dataset.py**: Loaders for MNIST and single-cell RNA data.

### Kernel, Graph, and Embedding Methods
- **src/kernels.py**: Dispatches kernel construction (Gaussian, kNN, Isomap, t-SNE, IAN, etc.), including pruning and shortest-path options. Integrates with `connected_comp_helper` for graph connectivity.
- **src/graph_methods.py**: Graph construction and pruning utilities (kNN, MST, density, bisection, random pruning).
- **src/embedding_algorithms.py**: Embedding methods (Diffusion Maps, UMAP, t-SNE, Spectral, LLE, PCA, etc.), with helpers for kernel/graph-based embeddings and connectivity fixes.
- **src/metrics.py, src/gw_utils.py**: Metric computation (Gromov-Wasserstein, single-linkage ultrametric, GH distance, etc.).

### Utilities and Parallelism
- **src/parallel_utils.py**: Context manager and helpers for parallel execution. Handles job count, backend selection, and temp folder cleanup. Used throughout for robust parallelization.
- **src/utils.py**: General utilities for graph connectivity, distance matrix normalization, duplicate removal, and potential-based measure computation.
- **src/sampling.py**: Additional sampling strategies (not always used directly).

### Visualization and Analysis
- **src/general_plotter.py**: Generates comprehensive visualizations for each dataset, including raw data, embeddings, kernel graphs, and adjacency matrices. Produces multi-matrix figures for all offsets/fractions.
- **src/final_analysis_plots.py**: Advanced statistical analysis and plotting. Computes ARI/GW/GH metrics vs. fill distance, offset, and other features. Produces summary plots, logistic regressions, and LaTeX tables for publication.
- **src/visualization.py**: Additional plotting utilities (used by other scripts).

_Last updated: May 5, 2025_