# A Geometric Threshold for Separation Detection in Manifold Inference

This repository contains the main synthetic and real world experiment code of the paper "A Geometric Threshold for Separation Detection in Manifold Inference".

## Abstract

To analyze manifold inference, theorists typically assume continuity, while users seek separation of manifold components (e.g., MNIST). We address this component versus continuity gap directly, by asking whether a manifold consists of distinct components under different samplings. We propose a single dimensionless quantity — the ratio between the ambient distance that separates the manifold components and the largest gap in the sample cover — that tests this for a broad family of neighborhood graphs and embeddings.  For a given sampling,
when the ratio exceeds a critical threshold, manifold components remain disconnected, clustering accuracy stays high, and global distortion (measured by Gromov–Wasserstein distance) remains low; if it falls below the threshold, bridging edges appear, clusters merge, and distortion rises.  We give upper and lower bounds for these thresholds as a function of the dimension and curvature of the data manifolds. This result bridges between classical (Gaussian) mixture separation principles and more topological manifold inference.

## Requirements
We advise using Python 3.12 for compatibility with the latest libraries. To install the neccessary dependencies, you can use the provided `requirements.txt` file. Run the following command in your terminal:

```bash
pip install -r requirements.txt
```

For a conda environment:

```bash
conda create -n geometric-separation python=3.12
conda activate geometric-separation
pip install -r requirements.txt
```

## Dataset Preparation

### Synthetic Data

The synthetic datasets are generated automatically by the code. They include:

- Various geometric shapes (spheres, ellipsoids, hyperboloids, tori) in different dimensions
- Uniform, biased, and noisy sampling options
- Configurable offsets between manifold components

### Real-world Data

For real-world experiments, we use:

1. **MNIST**: The dataset is automatically loaded when running experiments from `mnist.npz`.

2. **scRNA-seq data**: The 3k PBMCs dataset is included in the repository under `scRNA_data/`.

## Running Experiments

### Main Experiments

To run the main experiments that reproduce the results in the paper, use the following command for different configurations provided in the `configs/` directory. For example,

```bash
python src/main_experiment.py --config configs/config_main_1.json
```

And for biased and noisy sampling experiments, they are labeled accordingly:

```bash
python src/main_experiment.py --config configs/config_main_1_biased.json
python src/main_experiment.py --config configs/config_main_1_noisy.json
```

### MNIST Experiments

To run the MNIST experiments:

```bash
python src/main_experiment.py --config configs/config_mnist.json
```

### scRNA-seq Experiments

To run the scRNA-seq experiments:

```bash
python src/main_experiment.py --config configs/config_scRNA.json
```

## Cluster Distance Analysis on Real World

For the cluster ratio analysis on real-world datasets:

```bash
python src/cluster_ratio_analysis.py --dataset mnist --sample_size 1000,500
python src/cluster_ratio_analysis.py --dataset scRNA --sample_size 1000,500
```

Here, `--sample_size` specifies the number of samples to be used for the analysis and the `--dataset` argument can be either `mnist` or `scRNA`.

## Results

Our experiments validate the threshold theory across multiple datasets:

### Synthetic Data Results

We show that when the offset–fill-distance ratio exceeds a critical threshold (which depends on manifold dimension and curvature), the clusters remain distinct in the embedding space. This is measured by:

1. Gromov-Wasserstein distance between embeddings
2. Clustering accuracy (Adjusted Rand Index)

### Real-world Data Results

For MNIST and scRNA-seq data, we demonstrate that our threshold criterion correctly predicts when clusters will remain distinct despite subsampling.

## Code Structure

- `src/`: Source code
  - `main_experiment.py`: Main experiment pipeline
  - `mesh_sampling.py`: Dataset generation and subsampling
  - `kernels.py`: Kernel and graph construction methods
  - `embedding_algorithms.py`: Dimensionality reduction algorithms
  - `metrics.py`: Evaluation metrics including Gromov-Wasserstein
  - `utils.py`: Utility functions
  - `mnist_dataset.py`, `scRNA_dataset.py`: Real-world dataset handlers
  - `cluster_ratio_analysis.py`: Analysis of cluster separation ratios

- `configs/`: Configuration files
- `scRNA_data/`: scRNA-seq dataset


## License

This code is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
