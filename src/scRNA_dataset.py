#!/usr/bin/env python3
"""
scRNA_dataset.py
----------------
Loads single-cell RNA data from the preprocessed data files in scRNA_data directory.
Supports stratified sampling based on cell type to maintain proportional representation.
"""

import os
import numpy as np
import pandas as pd
import logging
from functools import lru_cache
import scipy.io
import scipy.sparse

logger = logging.getLogger("scRNA_loader")
handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
logger.addHandler(handler)
logger.setLevel(logging.INFO)

@lru_cache(maxsize=2)
def load_scRNA_data(data_dir="scRNA_data", n_samples=None, shuffle=True, random_state=42, stratify=True):
    """
    Loads scRNA data from the processed npz file with metadata for stratification.
    
    Parameters:
    -----------
    data_dir: directory containing the scRNA data files
    n_samples: number of cells to return; if None, return all
    shuffle: whether to shuffle cells
    random_state: seed for shuffling
    stratify: whether to perform stratified sampling by cell type
    
    Returns:
    --------
    X: array of shape (n_cells, n_genes) - expression data
    cell_types: array of cell type labels (if metadata available)
    """
    # Set up paths
    expression_path = os.path.join(data_dir, "expression_matrix.npz")
    metadata_path = os.path.join(data_dir, "cell_metadata.csv")
    
    # Load expression data
    logger.info(f"Loading scRNA expression data from {expression_path}")
    expr_data = np.load(expression_path, allow_pickle=True)
    
    # Check if the data is stored as a sparse matrix
    if all(key in expr_data for key in ['data', 'indices', 'indptr', 'shape', 'format']):
        logger.info("Detected sparse matrix format. Reconstructing sparse matrix...")
        import scipy.sparse as sp
        
        # Reconstruct the sparse matrix
        format_str = str(expr_data['format']) if isinstance(expr_data['format'], np.ndarray) else expr_data['format']
        if 'csr' in format_str.lower():
            X = sp.csr_matrix((expr_data['data'], expr_data['indices'], expr_data['indptr']), 
                              shape=tuple(expr_data['shape']))
        elif 'csc' in format_str.lower():
            X = sp.csc_matrix((expr_data['data'], expr_data['indices'], expr_data['indptr']), 
                              shape=tuple(expr_data['shape']))
        else:
            logger.warning(f"Unknown sparse matrix format: {format_str}. Attempting CSR reconstruction.")
            X = sp.csr_matrix((expr_data['data'], expr_data['indices'], expr_data['indptr']), 
                              shape=tuple(expr_data['shape']))
        
        # Convert to dense array for easier processing
        logger.info("Converting sparse matrix to dense array...")
        X = X.toarray()
    else:
        # Try direct loading as a dense array
        X = expr_data['data'] if 'data' in expr_data else expr_data['X']
    
    # Load metadata if available for stratification
    try:
        logger.info(f"Loading cell metadata from {metadata_path}")
        metadata = pd.read_csv(metadata_path)
        cell_types = metadata['cell_type'].values
        cell_type_ids = metadata['cell_type_id'].values if 'cell_type_id' in metadata.columns else None
    except (FileNotFoundError, KeyError) as e:
        logger.warning(f"Could not load cell metadata: {e}. Stratification will be disabled.")
        stratify = False
        cell_types = None
        
    rng = np.random.default_rng(random_state)
    
    # If n_samples is not specified or exceeds the dataset size, use all data
    if n_samples is None or n_samples >= X.shape[0]:
        n_samples = X.shape[0]
    
    if stratify and cell_types is not None:
        logger.info("Performing stratified sampling by cell type")
        
        # Get unique cell types and their counts
        unique_cell_types = np.unique(cell_types)
        n_cell_types = len(unique_cell_types)
        
        # Calculate how many samples to take from each cell type
        type_counts = np.array([np.sum(cell_types == ct) for ct in unique_cell_types])
        type_proportions = type_counts / np.sum(type_counts)
        samples_per_type = np.floor(type_proportions * n_samples).astype(int)
        
        # Adjust for rounding errors to ensure exactly n_samples
        remaining = n_samples - np.sum(samples_per_type)
        if remaining > 0:
            # Add remaining samples to the largest cell types
            sorted_types = np.argsort(-type_counts)[:remaining]
            for idx in sorted_types:
                samples_per_type[idx] += 1
        
        # Sample from each cell type
        selected_indices = []
        for i, ct in enumerate(unique_cell_types):
            ct_indices = np.where(cell_types == ct)[0]
            if shuffle:
                ct_indices = rng.permutation(ct_indices)
            selected_indices.extend(ct_indices[:samples_per_type[i]])
        
        # Shuffle the combined indices if needed
        if shuffle:
            selected_indices = rng.permutation(selected_indices)
            
        # Select the data
        X = X[selected_indices]
        if cell_types is not None:
            cell_types = cell_types[selected_indices]
    else:
        # Traditional random sampling
        if shuffle:
            indices = rng.permutation(X.shape[0])
            X = X[indices]
            if cell_types is not None:
                cell_types = cell_types[indices]
        
        # Subset to n_samples
        X = X[:n_samples]
        if cell_types is not None:
            cell_types = cell_types[:n_samples]
    
    # Verify data was loaded properly
    if isinstance(X, np.ndarray) and X.ndim == 2:
        logger.info(f"Loaded scRNA data: {X.shape[0]} cells, {X.shape[1]} features")
    else:
        # If X is not a 2D array, log error and try to fix
        logger.error(f"Unexpected data structure: {type(X)}, shape: {getattr(X, 'shape', None)}")
        if hasattr(X, 'toarray') and callable(getattr(X, 'toarray')):
            logger.info("Converting to dense array...")
            X = X.toarray()
            logger.info(f"After conversion: {X.shape[0]} cells, {X.shape[1]} features")
        else:
            raise ValueError(f"Failed to load expression data properly. Received type: {type(X)}")
    
    # Log cell type information if available
    if cell_types is not None:
        uniq_types = np.unique(cell_types)
        logger.info(f"Cell types: {len(uniq_types)} unique types among {len(cell_types)} labeled cells")
    else:
        logger.info("No cell type information available")
        
    return X, cell_types
