"""
parallel_utils.py
-----------------
Utility functions to manage parallel computation for performance-intensive operations.
"""

import os
import numpy as np
import multiprocessing
from functools import partial
from contextlib import contextmanager

@contextmanager
def parallel_context(n_jobs=None, backend='loky', temp_folder=None):
    """
    Context manager for parallel execution with proper resource management.
    
    Parameters:
    -----------
    n_jobs : int or None
        Number of parallel jobs. If None, uses CPU count.
    backend : str
        Backend for joblib (loky, threading, multiprocessing).
    temp_folder : str or None
        Temporary folder for memmapped caching.
        
    Yields:
    -------
    n_jobs : int
        Number of jobs to use.
    """
    if n_jobs is None:
        n_jobs = max(1, os.cpu_count() - 1)  # Leave one CPU free
    try:
        yield n_jobs
    finally:
        if temp_folder and os.path.exists(temp_folder):
            import shutil
            try:
                shutil.rmtree(temp_folder)
            except:
                pass

def parallel_distance_matrix(X, Y=None, metric='euclidean', n_jobs=None, chunk_size=1000):
    """
    Parallel computation of distance matrix, optimized for large matrices.
    
    Parameters:
    -----------
    X : ndarray (n_samples_X, n_features)
        First set of points
    Y : ndarray (n_samples_Y, n_features) or None
        Second set of points, if None, use X
    metric : str
        Distance metric to use
    n_jobs : int or None
        Number of parallel jobs
    chunk_size : int
        Size of chunks for parallel processing
        
    Returns:
    --------
    D : ndarray
        Distance matrix
    """
    from scipy.spatial.distance import cdist
    from joblib import Parallel, delayed
    
    if Y is None:
        Y = X
        symmetric = True
    else:
        symmetric = False
    
    n_x = X.shape[0]
    n_y = Y.shape[0]
    
    # For small matrices, just use cdist directly
    if n_x * n_y < chunk_size * chunk_size:
        return cdist(X, Y, metric=metric)
    
    # Divide into chunks for parallel processing
    x_chunks = [X[i:min(i+chunk_size, n_x)] for i in range(0, n_x, chunk_size)]
    y_chunks = [Y[i:min(i+chunk_size, n_y)] for i in range(0, n_y, chunk_size)]
    
    # Set up parallel processing
    with parallel_context(n_jobs=n_jobs) as n_jobs:
        results = Parallel(n_jobs=n_jobs)(
            delayed(cdist)(x_chunk, y_chunk, metric=metric)
            for x_chunk in x_chunks
            for y_chunk in y_chunks
        )
    
    # Reassemble the distance matrix
    D = np.zeros((n_x, n_y))
    idx = 0
    for i, x_chunk in enumerate(x_chunks):
        x_size = x_chunk.shape[0]
        for j, y_chunk in enumerate(y_chunks):
            y_size = y_chunk.shape[0]
            D[i*chunk_size:i*chunk_size+x_size, j*chunk_size:j*chunk_size+y_size] = results[idx]
            idx += 1
            
    return D
