import numpy as np
import ot
import multiprocessing
from functools import partial
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import logging
import traceback

# Handle GPU availability gracefully
try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    print("Warning: PyTorch not available. GPU acceleration will be disabled.")

# Set up logging
logger = logging.getLogger("gw_utils")
handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
logger.addHandler(handler)
logger.setLevel(logging.INFO)

def square_loss_gpu(C1, C2, p, q, G):
    """GPU-accelerated square loss for GW"""
    if not HAS_TORCH:
        raise ImportError("PyTorch is required for GPU acceleration")
        
    if not torch.is_tensor(C1):
        C1 = torch.tensor(C1, device='cuda' if torch.cuda.is_available() else 'cpu')
        C2 = torch.tensor(C2, device='cuda' if torch.cuda.is_available() else 'cpu')
        p = torch.tensor(p, device=C1.device)
        q = torch.tensor(q, device=C1.device)
        G = torch.tensor(G, device=C1.device)
    
    C1_times_G = torch.matmul(C1, G)
    loss = torch.sum((C1_times_G.mm(C2.t()) - C1.mm(G).mm(C2.t())) ** 2)
    return loss.item() if torch.is_tensor(loss) else loss

def gromov_wasserstein(C1, C2, p=None, q=None, loss_fun='square_loss', 
                       epsilon=1e-2, max_iter=100, use_gpu=True):
    """
    Compute Gromov-Wasserstein distance between metric spaces.
    Optimized implementation with GPU support.
    
    Parameters:
    -----------
    C1, C2: distance/cost matrices
    p, q: distributions (uniform by default)
    loss_fun: loss function ('square_loss' or 'kl_loss')
    epsilon: regularization parameter
    max_iter: maximum number of iterations
    use_gpu: whether to use GPU acceleration
    
    Returns:
    --------
    gw_dist: Gromov-Wasserstein distance
    transport_plan: OT plan
    """
    if use_gpu and HAS_TORCH and torch.cuda.is_available():
        logger.info("Using GPU acceleration for GW computation")
        try:
            device = torch.device('cuda')
            C1_t = torch.tensor(C1, device=device)
            C2_t = torch.tensor(C2, device=device)
            
            if p is None:
                p = torch.ones(C1.shape[0], device=device) / C1.shape[0]
            else:
                p = torch.tensor(p, device=device)
                
            if q is None:
                q = torch.ones(C2.shape[0], device=device) / C2.shape[0]
            else:
                q = torch.tensor(q, device=device)
            
            # Use entropic GW which is faster
            gw_dist, log = ot.gromov.entropic_gromov_wasserstein2(
                C1_t.cpu().numpy(), C2_t.cpu().numpy(), 
                p.cpu().numpy(), q.cpu().numpy(),
                loss_fun=loss_fun, epsilon=epsilon, 
                max_iter=max_iter, log=True
            )
            transport_plan = log['T']
        except Exception as e:
            logger.warning(f"GPU computation failed: {e}, falling back to CPU")
            use_gpu = False
    
    if not use_gpu or not HAS_TORCH or not torch.cuda.is_available():
        if p is None:
            p = np.ones(C1.shape[0]) / C1.shape[0]
        if q is None:
            q = np.ones(C2.shape[0]) / C2.shape[0]
            
        gw_dist, log = ot.gromov.entropic_gromov_wasserstein2(
            C1, C2, p, q,
            loss_fun=loss_fun, epsilon=epsilon, 
            max_iter=max_iter, log=True
        )
        transport_plan = log['T']
    
    return gw_dist, transport_plan

def parallel_gw_computation(data_list, ref_data, n_jobs=-1, use_gpu=True):
    """
    Compute GW distances in parallel between reference data and a list of datasets
    
    Parameters:
    -----------
    data_list: list of datasets to compare with ref_data
    ref_data: reference dataset or list of reference datasets matching data_list
    n_jobs: number of processes (-1 for using all available cores)
    use_gpu: whether to use GPU acceleration
    
    Returns:
    --------
    gw_distances: list of GW distances
    """
    n_jobs = multiprocessing.cpu_count() if n_jobs == -1 else n_jobs
    logger.info(f"Running parallel GW computation with {n_jobs} processes")
    
    # Check if ref_data is a single dataset or a list matching data_list
    if isinstance(ref_data, list):
        if len(ref_data) != len(data_list):
            raise ValueError(f"Length mismatch: data_list ({len(data_list)}) vs ref_data ({len(ref_data)})")
        pairs = list(zip(data_list, ref_data))
        
        # Define the worker function for paired data
        def compute_gw_paired(pair):
            data, ref = pair
            try:
                return gromov_wasserstein(data, ref, use_gpu=use_gpu)[0]
            except Exception as e:
                logger.error(f"Error in GW computation: {e}")
                logger.error(traceback.format_exc())
                return float('nan')
        
        # Run in parallel
        with ProcessPoolExecutor(max_workers=n_jobs) as executor:
            gw_distances = list(executor.map(compute_gw_paired, pairs))
    else:
        # Define the worker function for fixed reference
        def compute_gw(data):
            try:
                return gromov_wasserstein(data, ref_data, use_gpu=use_gpu)[0]
            except Exception as e:
                logger.error(f"Error in GW computation: {e}")
                logger.error(traceback.format_exc())
                return float('nan')
        
        # Run in parallel
        with ProcessPoolExecutor(max_workers=n_jobs) as executor:
            gw_distances = list(executor.map(compute_gw, data_list))
    
    return gw_distances

def batch_gw_computation(C1_list, C2_list, batch_size=10, use_gpu=True, n_jobs=-1):
    """
    Compute GW distances in batches to optimize memory usage
    
    Parameters:
    -----------
    C1_list: list of first cost matrices
    C2_list: list of second cost matrices (can be a single matrix used for all comparisons)
    batch_size: size of batches for processing
    use_gpu: whether to use GPU acceleration
    n_jobs: number of parallel jobs
    
    Returns:
    --------
    results: list of GW distances
    """
    results = []
    total_comps = len(C1_list)
    
    # Check if C2_list is a single matrix or a list
    if not isinstance(C2_list, list):
        C2_list = [C2_list] * total_comps
    elif len(C2_list) == 1:
        C2_list = C2_list * total_comps
    elif len(C2_list) != total_comps:
        raise ValueError(f"Length mismatch: C1_list ({total_comps}) vs C2_list ({len(C2_list)})")
    
    try:
        for i in range(0, total_comps, batch_size):
            batch_C1 = C1_list[i:min(i+batch_size, total_comps)]
            batch_C2 = C2_list[i:min(i+batch_size, total_comps)]
            
            if use_gpu and HAS_TORCH and torch.cuda.is_available():
                # Process batch on GPU
                batch_results = []
                for C1, C2 in zip(batch_C1, batch_C2):
                    try:
                        gw_dist, _ = gromov_wasserstein(C1, C2, use_gpu=True)
                        batch_results.append(gw_dist)
                    except Exception as e:
                        logger.error(f"GPU computation failed: {e}, falling back to CPU")
                        try:
                            gw_dist, _ = gromov_wasserstein(C1, C2, use_gpu=False)
                            batch_results.append(gw_dist)
                        except Exception as e2:
                            logger.error(f"CPU computation also failed: {e2}")
                            batch_results.append(float('nan'))
            else:
                # Process batch in parallel on CPU
                batch_results = parallel_gw_computation(
                    batch_C1, 
                    batch_C2, 
                    n_jobs=n_jobs,
                    use_gpu=False
                )
            
            results.extend(batch_results)
            logger.info(f"Processed batch {i//batch_size + 1}/{(total_comps-1)//batch_size + 1}")
    except Exception as e:
        logger.error(f"Error in batch processing: {e}")
        logger.error(traceback.format_exc())
    
    return results
