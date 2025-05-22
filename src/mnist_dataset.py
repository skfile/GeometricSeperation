"""
mnist_dataset.py
------------------------------
Module for loading and processing the MNIST dataset with optimizations.

"""

import numpy as np
import logging
import os
from functools import lru_cache

logger = logging.getLogger("mnist_loader")
handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
logger.addHandler(handler)

@lru_cache(maxsize=4)
def load_mnist_data(n_samples=10000, shuffle=True, random_state=42, use_memmap=True, stratify=True):
    """
    Loads MNIST from a local npz file with optimizations.
    Adds caching and memory mapping for faster access.
    
    Parameters:
    -----------
    n_samples: number of samples to load
    shuffle: whether to shuffle the data
    random_state: random seed for shuffling
    use_memmap: whether to use memory mapping for large datasets
    stratify: whether to perform stratified sampling by digit class
    
    Returns:
    --------
    X: data samples (n_samples x 784)
    y: labels
    """
    local_path = "mnist.npz"
    logger.info(f"Loading MNIST from local file: {local_path}")
    
    try:
        # Load all data first
        data_npz = np.load(local_path, mmap_mode='r' if use_memmap and n_samples > 10000 else None)
        data = data_npz['x_train'].reshape(60000, 784)
        labels = data_npz['y_train']
        
        rng = np.random.default_rng(random_state)
        
        if stratify:
            logger.info("Performing stratified sampling by digit class")
            unique_classes = np.unique(labels)
            n_classes = len(unique_classes)
            
            class_counts = np.array([np.sum(labels == c) for c in unique_classes])
            class_proportions = class_counts / np.sum(class_counts)
            samples_per_class = np.floor(class_proportions * n_samples).astype(int)
            
            while np.sum(samples_per_class) < n_samples:
                # Add the remaining samples to random classes
                remaining = n_samples - np.sum(samples_per_class)
                # Choose random classes without replacement
                add_to_classes = rng.choice(n_classes, size=min(remaining, n_classes), replace=False)
                for idx in add_to_classes:
                    samples_per_class[idx] += 1
                    if np.sum(samples_per_class) == n_samples:
                        break
            
            # Sample from each class
            selected_indices = []
            for i, c in enumerate(unique_classes):
                class_indices = np.where(labels == c)[0]
                if shuffle:
                    # Shuffle indices for this class
                    class_indices = rng.permutation(class_indices)
                # Take required number of samples
                selected_indices.extend(class_indices[:samples_per_class[i]])
            
            # Shuffle the combined indices if needed
            if shuffle:
                selected_indices = rng.permutation(selected_indices)
            
            # Select the data
            data = data[selected_indices]
            labels = labels[selected_indices]
        else:
            # Traditional random sampling
            if shuffle:
                idxs = rng.permutation(data.shape[0])
                data = data[idxs]
                labels = labels[idxs]
            
            # Limit to n_samples
            n_samples = min(n_samples, data.shape[0])
            data = data[:n_samples]
            labels = labels[:n_samples]
        
        X = data.astype(np.float32)
        y = labels
        
        logger.info(f"MNIST data loaded: {X.shape[0]} samples with {X.shape[1]} features")
        return X, y
    except Exception as e:
        logger.error(f"Error loading MNIST data: {e}")
        raise

def ensure_all_mnist_classes(indices, labels, num_classes=10, random_state=None):
    """
    Ensures that the selected indices have at least one sample from each MNIST digit class.
    If any classes are missing, it will replace some of the existing samples with
    samples from the missing classes.
    
    Parameters:
    -----------
    indices: np.ndarray
        The indices of the selected samples
    labels: np.ndarray
        The labels of all data points
    num_classes: int
        The number of classes to ensure representation for (default: 10 for MNIST digits 0-9)
    random_state: int or np.random.Generator
        Random seed or generator for reproducibility
        
    Returns:
    --------
    new_indices: np.ndarray
        Modified indices with at least one sample from each class
    """
    if random_state is None or isinstance(random_state, int):
        rng = np.random.default_rng(random_state)
    else:
        rng = random_state
        
    # Check which classes are present in the current selection
    selected_labels = labels[indices]
    unique_selected = np.unique(selected_labels)
    all_classes = np.arange(num_classes)
    
    # Find missing classes
    missing_classes = np.setdiff1d(all_classes, unique_selected)
    
    if len(missing_classes) == 0:
        logger.debug("All MNIST classes are already represented in the sample")
        return indices
    
    logger.info(f"Ensuring representation for {len(missing_classes)} missing MNIST classes: {missing_classes}")
    
    # Create a copy of indices that we'll modify
    new_indices = indices.copy()
    
    # For each missing class, find examples in the full dataset and add one
    for missing_class in missing_classes:
        # Get all indices where the label is the missing class
        candidates = np.where(labels == missing_class)[0]
        
        if len(candidates) == 0:
            logger.warning(f"No examples of class {missing_class} found in the full dataset")
            continue
            
        # Select a random sample from this class
        selected_idx = rng.choice(candidates)
        
        # Replace one of the existing indices with this sample
        replace_position = rng.integers(len(indices))
        new_indices[replace_position] = selected_idx
    
    # Verify that we now have all classes represented
    new_selected_labels = labels[new_indices]
    new_unique_selected = np.unique(new_selected_labels)
    
    if len(np.setdiff1d(all_classes, new_unique_selected)) > 0:
        logger.warning("Could not ensure all MNIST classes are represented")
    else:
        logger.info(f"Successfully ensured representation for all {num_classes} MNIST classes")
    
    return new_indices