#!/usr/bin/env python3
"""
final_analysis_plots.py
----------------------------------------
A comprehensive analysis script focusing on:

1) ARI vs. {FillDist, Offset, OffsetOverSigma, OffsetOverFillDistance, Ratio_OS_over_Fill}
2) Graph GW metric (GW_Uniform_Uniform)
3) GW Tree Embedding (GH_Ultrametric_Emb)
4) Logistic regressions (ARI>=0.8, GW<=0.3, GH<=0.3)
5) Aggregator analyses vs. the 4 main columns:
   - OffsetOverSigma (existing)
   - Ratio_OS_over_Fill (existing)
   - Offset (NEW)
   - OffsetOverFillDistance = Offset/Fill_Distance (NEW)
   - OffsetOverFillDistanceScaled = Offset/Fill_Distance_Scaled (NEW)
6) "sharp boundary" analysis for each x-col:
   - find ARI jump threshold
   - measure how strongly GW/GH separate at that threshold

Usage:
  python final_analysis_plots.py --input_csv your_data.csv --output_dir results/
"""

import os
import sys
import json
import argparse
import numpy as np
import pandas as pd

import seaborn as sns
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from joblib import Parallel, delayed
from scipy.stats import kruskal, spearmanr, rankdata, mannwhitneyu
from scipy.integrate import trapezoid
import statsmodels.api as sm
from sklearn.exceptions import ConvergenceWarning

import warnings
import logging

##############################################################################
# AGGREGATION CONFIGURATION
##############################################################################

# Dimension bins for aggregating data
DIMENSION_BINS = [1, 5, 10, 15, 20, 30, 50, 100]  # Creates bins: 1-5, 5-10, 10-15, 15-20, 20-30, 30-50, 50-100
DIMENSION_BIN_LABELS = [f"dim_{DIMENSION_BINS[i]}-{DIMENSION_BINS[i+1]}" for i in range(len(DIMENSION_BINS)-1)]

# Noise detection keywords - datasets with these terms are considered "noisy"
NOISE_KEYWORDS = ["noisy", "noise", "perturb"]

# Bias detection keywords - datasets with these terms are considered "biased"
BIAS_KEYWORDS = ["bias", "biased"]

##############################################################################
# PLOT CONFIGURATION
##############################################################################

# Standard figure configuration
PLOT_CONFIG = {
    # General settings
    "figure.dpi": 300,
    "font.size": 8,
    "font.family": "serif",
    
    # Axes and lines
    "axes.linewidth": 0.5,
    "axes.titlesize": 9,
    "axes.labelsize": 8,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "lines.linewidth": 1.0,
    "lines.markersize": 4,
    
    # Legend
    "legend.fontsize": 7,
    "legend.frameon": False,
    
    # Grid
    "grid.linewidth": 0.2,
    "grid.alpha": 0.4,
}

# Apply settings
plt.rcParams.update(PLOT_CONFIG)

# Color palettes
COLOR_PALETTES = {
    "default": sns.color_palette("colorblind", 8),
    "diverging": sns.color_palette("RdBu_r", 8),
    "sequential": sns.color_palette("viridis", 8),
    "paired": sns.color_palette("Paired", 8),
    "method_colors": {
        "Knn": "#1f77b4",
        "Knn Mst": "#ff7f0e",
        "Knn Perturb": "#2ca02c"
    }
}

# Standard labels and titles (organized by plot type)
LABELS = {
    "fill_distance": {
        "x": "Fill Distance",
        "y": "Metric Value",
        "title": "Metrics vs Fill Distance"
    },
    "fill_distance_scaled": {
        "x": "Scaled Fill Distance",
        "y": "Metric Value",
        "title": "Metrics vs Scaled Fill Distance"
    },
    "offset": {
        "x": "Offset",
        "y": "Metric Value",
        "title": "Metrics vs Offset"
    },
    "offset_over_sigma": {
        "x": "Offset/Sigma",
        "y": "Metric Value",
        "title": "Metrics vs Offset/Sigma"
    },
    "sample_percentage": {
        "x": "Sample Percentage",
        "y": "Metric Value",
        "title": "Metrics vs Sample Percentage"
    },
    "ratio_os_fill": {
        "x": "Ratio OS/Fill",
        "y": "Metric Value",
        "title": "Metrics vs Ratio OS/Fill"
    },
    "offset_over_fill": {
        "x": "Offset/Fill Distance",
        "y": "Metric Value",
        "title": "Metrics vs Offset/Fill Distance"
    }
}

# Metric value display names
# These labels control how metrics are displayed in plot titles and axes
METRIC_LABELS = {
    # Core metrics
    "ARI": "ARI",
    "GW_Uniform_Uniform": "Graph GW Metric",
    "GH_Ultrametric_Emb": "GW Tree Embedding",
    
    # Basic distance measurements
    "Fill_Distance": "Fill Distance",
    "Fill_Distance_Scaled": "Scaled Fill Distance",
    "Offset": "Offset",
    "Sample_Percentage": "Sample Percentage",
    
    # Ratio metrics
    "OffsetOverSigma": "Offset/Sigma",
    "Ratio_OS_over_Fill": "Ratio OS/Fill",
    "Ratio": "Ratio OS/Fill",
    "OffsetOverFillDistance": "Offset/Fill Distance",
    "OffsetOverFillDistanceScaled": "Offset/Fill Distance Scaled",
    
    # KNN variants
    "Fill_Distance_KNN_Mean": "Fill Distance KNN Mean",
    "Fill_Distance_KNN_Max": "Fill Distance KNN Max",
    "Fill_Distance_KNN_Mean_Scaled": "Scaled Fill Distance KNN Mean",
    "Fill_Distance_KNN_Max_Scaled": "Scaled Fill Distance KNN Max",
    "OffsetOverFillDistanceKNNMean": "Offset/Fill Distance KNN Mean",
    "OffsetOverFillDistanceKNNMax": "Offset/Fill Distance KNN Max",
    "OffsetOverFillDistanceKNNMeanScaled": "Offset/Fill Distance KNN Mean Scaled",
    "OffsetOverFillDistanceKNNMaxScaled": "Offset/Fill Distance KNN Max Scaled",
    
    # Minimax variants
    "Minimax_Offset": "Minimax Offset",
    "Minimax_Offset_Scaled": "Minimax Offset Scaled",
    "MinimaxOffsetOverSigma": "Minimax Offset/Sigma",
    "MinimaxOffsetOverFillDistance": "Minimax Offset/Fill Distance",
    "MinimaxOffsetOverFillDistanceScaled": "Minimax Offset/Fill Distance Scaled",
    "Ratio_MinimaxOS_over_Fill": "Ratio MinimaxOS/Fill",
    "MinimaxOffsetOverFillDistanceKNNMean": "Minimax Offset/Fill Distance KNN Mean",
    "MinimaxOffsetOverFillDistanceKNNMax": "Minimax Offset/Fill Distance KNN Max",
    "MinimaxOffsetOverFillDistanceKNNMeanScaled": "Minimax Offset/Fill Distance KNN Mean Scaled",
    "MinimaxOffsetOverFillDistanceKNNMaxScaled": "Minimax Offset/Fill Distance KNN Max Scaled",
    "MinimaxOffsetScaledOverFillDistance": "Minimax Offset Scaled/Fill Distance",
    "MinimaxOffsetScaledOverFillDistanceKNNMean": "Minimax Offset Scaled/Fill Distance KNN Mean",
    "MinimaxOffsetScaledOverFillDistanceKNNMax": "Minimax Offset Scaled/Fill Distance KNN Max"
}

##############################################################################
# ARGUMENTS
##############################################################################

def parse_args():
    p = argparse.ArgumentParser(description="ICML final analysis extended with offset/fill analysis.")
    p.add_argument("--input_csv", required=True, help="CSV with HPC experiment results.")
    p.add_argument("--output_dir", required=True, help="Directory for outputs.")
    p.add_argument("--n_jobs", type=int, default=-1, help="Parallel jobs for dataset-level analysis.")
    return p.parse_args()

##############################################################################
# LOGGING
##############################################################################

def setup_logging(output_dir):
    os.makedirs(output_dir, exist_ok=True)
    log_file = os.path.join(output_dir, "analysis.log")
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )
    logger = logging.getLogger()
    return logger

##############################################################################
# PREPROCESSING
##############################################################################

def beautify_string(s):
    if not isinstance(s, str):
        return s
    parts = s.replace("_"," ").split()
    parts = [p.capitalize() for p in parts]
    return " ".join(parts)

def rename_kernel_method(row):
    km = row.get("Kernel_Method","")
    kp = row.get("Kernel_Params","")
    try:
        pdict = json.loads(kp)
    except:
        pdict = {}
    if km == "knn_shortest_path":
        pr = pdict.get("pruning_method", None)
        if pr:
            row["Kernel_Method"] = f"knn_{pr}"
        else:
            row["Kernel_Method"] = "knn"
    row["Kernel_Method"] = beautify_string(row["Kernel_Method"])
    return row

def rename_embedding_method(row):
    em = row.get("Embedding_Method","")
    if isinstance(em,str):
        row["Embedding_Method"] = beautify_string(em)
    return row

def rename_potential_name(row):
    pn = row.get("Potential_Name","")
    if isinstance(pn,str):
        row["Potential_Name"] = beautify_string(pn)
    return row

def parse_shapes_for_sigma_and_dim(json_str):
    """
    Parse shapes => return (sigma_value, dimension, shapes_are_all_spheres).
    sigma_value = max radius or max(axes) among shapes
    dimension = from the first shape's 'dim'
    shapes_are_all_spheres = True if each shape_type == 'sphere_hd'
    """
    try:
        info = json.loads(json_str)
    except:
        return (np.nan, np.nan, False)
    if not isinstance(info, dict) or "shapes" not in info:
        return (np.nan, np.nan, False)

    shapes = info["shapes"]
    if len(shapes)<1:
        return (np.nan, np.nan, False)

    first_shape = shapes[0]
    dim = first_shape.get("dim", np.nan)
    max_val = 0.0
    all_spheres = True

    for shp in shapes:
        st = shp.get("shape_type","")
        if st != "sphere_hd":
            all_spheres = False
        if st=="sphere_hd":
            r=shp.get("radius",0.0)
            max_val=max(max_val, r)
        elif st=="ellipsoid_hd":
            axes=shp.get("axes",[])
            if axes:
                max_val=max(max_val, max(axes))
        elif st=="torus_hd":
            maj=shp.get("major_radius",0.0)
            mnr=shp.get("minor_radius",0.0)
            max_val=max(max_val, maj+mnr)

    if max_val<=0:
        max_val=np.nan
    return (max_val, dim, all_spheres)

def parse_shapes_info(row):
    si = row.get("Shapes_Info","")
    if not (isinstance(si,str) and si.startswith("{")):
        return row

    sigma_val, dim_val, all_spheres = parse_shapes_for_sigma_and_dim(si)
    row["SigmaValue"] = sigma_val

    curr_dim = row.get("dimension", np.nan)
    if pd.isna(curr_dim) and not pd.isna(dim_val):
        row["dimension"] = dim_val

    # Mark if all shapes are spheres
    row["Spheres_Only"] = bool(all_spheres)
    return row

def extract_base_dataset(df):
    def get_base(ds):
        if isinstance(ds, str) and "_offset_" in ds:
            return ds.split("_offset_")[0]
        return ds
    df["Base_Dataset"] = df["Dataset"].apply(get_base)
    return df

def replace_negative_metrics(df, col_list):
    for c in col_list:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce')
            df.loc[df[c]<0, c] = 0
    return df

def validate_input_data(df, logger):
    needed = [
        "Dataset","Kernel_Method","Kernel_Params","Embedding_Method","Potential_Name","Shapes_Info",
        "Sample_Percentage","Offset","Fill_Distance","Fill_Distance_Scaled","ARI",
        "GW_Uniform_Uniform","GH_Ultrametric_Emb"
    ]
    missing = [n for n in needed if n not in df.columns]
    if missing:
        logger.error(f"Missing columns => {missing}")
        sys.exit(1)
    logger.info("All required columns are present.")

def is_noisy_dataset(dataset_name):
    """
    Determine if a dataset is considered 'noisy' based on name patterns.
    
    Parameters
    ----------
    dataset_name : str
        Name of the dataset
        
    Returns
    -------
    bool
        True if the dataset appears to be a noisy variant
    """
    if not isinstance(dataset_name, str):
        return False
        
    return any(keyword in dataset_name.lower() for keyword in NOISE_KEYWORDS)

def is_biased_dataset(dataset_name):
    """
    Determine if a dataset is considered 'biased' based on name patterns.
    
    Parameters
    ----------
    dataset_name : str
        Name of the dataset
        
    Returns
    -------
    bool
        True if the dataset appears to be a biased variant
    """
    if not isinstance(dataset_name, str):
        return False
        
    return any(keyword in dataset_name.lower() for keyword in BIAS_KEYWORDS)
    
def categorize_by_dimension(dimension_value):
    """
    Assign a dimension category based on the dimension value.
    
    Parameters
    ----------
    dimension_value : float or int
        The dimension value to categorize
        
    Returns
    -------
    str
        The dimension category label
    """
    if pd.isna(dimension_value):
        return "unknown_dim"
        
    # Find the appropriate dimension bin
    for i in range(len(DIMENSION_BINS) - 1):
        if DIMENSION_BINS[i] <= dimension_value < DIMENSION_BINS[i+1]:
            return DIMENSION_BIN_LABELS[i]
            
    # If dimension is larger than the largest bin boundary
    if dimension_value >= DIMENSION_BINS[-1]:
        return f"dim_{DIMENSION_BINS[-1]}plus"
        
    # If dimension is smaller than the smallest bin boundary
    return f"dim_below_{DIMENSION_BINS[0]}"

def format_dimension_category(dim_cat):
    """
    Format dimension category for display in plot titles.
    
    Parameters
    ----------
    dim_cat : str
        The dimension category string (e.g. 'dim_10-15')
        
    Returns
    -------
    str
        Formatted dimension category (e.g. 'Dimension 10 to 15')
    """
    if not isinstance(dim_cat, str):
        return str(dim_cat)
        
    # Handle special cases
    if dim_cat == "unknown_dim":
        return "Unknown Dimension"
        
    # For standard dimension bins (e.g., 'dim_10-15')
    if dim_cat.startswith("dim_"):
        # Extract the numeric parts
        parts = dim_cat[4:].split('-')
        if len(parts) == 2:
            try:
                start = int(parts[0])
                end = int(parts[1])
                return f"Dimension {start} to {end}"
            except ValueError:
                pass
        
        # For cases like 'dim_50plus'
        if dim_cat.endswith("plus"):
            try:
                val = int(dim_cat[4:-4])  # Extract number from 'dim_50plus'
                return f"Dimension {val}+"
            except ValueError:
                pass
                
        # For cases like 'dim_below_1'
        if "below" in dim_cat:
            try:
                val = int(dim_cat.split("_")[-1])
                return f"Dimension Below {val}"
            except (ValueError, IndexError):
                pass
    
    # Default: just capitalize and replace underscores with spaces
    return dim_cat.replace("_", " ").capitalize()

def do_preprocessing(df, logger):
    for c in ["dimension","SigmaValue","Spheres_Only"]:
        if c not in df.columns:
            df[c] = np.nan

    df = extract_base_dataset(df)
    df = df.apply(rename_kernel_method, axis=1)
    df = df.apply(rename_embedding_method, axis=1)
    df = df.apply(rename_potential_name, axis=1)
    df = df.apply(parse_shapes_info, axis=1)
    
    # Add noise and bias flags
    df["IsNoisy"] = df["Dataset"].apply(is_noisy_dataset)
    df["IsBiased"] = df["Dataset"].apply(is_biased_dataset)

    # numeric
    df["Offset"] = pd.to_numeric(df["Offset"], errors='coerce')
    df["Fill_Distance"] = pd.to_numeric(df["Fill_Distance"], errors='coerce')
    df["Fill_Distance_Scaled"] = pd.to_numeric(df["Fill_Distance_Scaled"], errors='coerce')
    df["dimension"] = pd.to_numeric(df["dimension"], errors='coerce')
    df["SigmaValue"] = pd.to_numeric(df["SigmaValue"], errors='coerce')
    
    # Handle minimax offset values
    df["Minimax_Offset"] = pd.to_numeric(df["Minimax_Offset"], errors='coerce')
    df["Minimax_Offset_Scaled"] = pd.to_numeric(df["Minimax_Offset_Scaled"], errors='coerce')

    df = replace_negative_metrics(df, ["GW_Uniform_Uniform","GH_Ultrametric_Emb"])

    # offsetOverSigma
    df["OffsetOverSigma"] = np.nan
    mask_ = (~df["Offset"].isna()) & (df["SigmaValue"]>0)
    df.loc[mask_,"OffsetOverSigma"] = df.loc[mask_,"Offset"] / df.loc[mask_,"SigmaValue"]
    
    # MinimaxOffsetOverSigma
    df["MinimaxOffsetOverSigma"] = np.nan
    mask_minimax = (~df["Minimax_Offset"].isna()) & (df["SigmaValue"]>0)
    df.loc[mask_minimax,"MinimaxOffsetOverSigma"] = df.loc[mask_minimax,"Minimax_Offset"] / df.loc[mask_minimax,"SigmaValue"]

    # ratio
    df["Ratio_OS_over_Fill"] = np.nan
    mask_f = (df["Fill_Distance"]>0) & mask_
    df.loc[mask_f,"Ratio_OS_over_Fill"] = df.loc[mask_f,"OffsetOverSigma"] / df.loc[mask_f,"Fill_Distance"]
    
    # ratio with minimax
    df["Ratio_MinimaxOS_over_Fill"] = np.nan
    mask_minimax_f = (df["Fill_Distance"]>0) & mask_minimax
    df.loc[mask_minimax_f,"Ratio_MinimaxOS_over_Fill"] = df.loc[mask_minimax_f,"MinimaxOffsetOverSigma"] / df.loc[mask_minimax_f,"Fill_Distance"]

    # NEW => offsetOverFillDistance
    df["OffsetOverFillDistance"] = np.nan
    mask_off = (df["Fill_Distance"]>0) & (~df["Offset"].isna())
    df.loc[mask_off,"OffsetOverFillDistance"] = df.loc[mask_off,"Offset"] / df.loc[mask_off,"Fill_Distance"]
    
    # NEW => minimaxOffsetOverFillDistance
    df["MinimaxOffsetOverFillDistance"] = np.nan
    mask_minimax_off = (df["Fill_Distance"]>0) & (~df["Minimax_Offset"].isna())
    df.loc[mask_minimax_off,"MinimaxOffsetOverFillDistance"] = df.loc[mask_minimax_off,"Minimax_Offset"] / df.loc[mask_minimax_off,"Fill_Distance"]

    # NEW => offsetOverFillDistanceScaled
    df["OffsetOverFillDistanceScaled"] = np.nan
    mask_off_scaled = (df["Fill_Distance_Scaled"]>0) & (~df["Offset"].isna())
    df.loc[mask_off_scaled,"OffsetOverFillDistanceScaled"] = df.loc[mask_off_scaled,"Offset"] / df.loc[mask_off_scaled,"Fill_Distance_Scaled"]
    
    # NEW => minimaxOffsetOverFillDistanceScaled
    df["MinimaxOffsetOverFillDistanceScaled"] = np.nan
    mask_minimax_off_scaled = (df["Fill_Distance_Scaled"]>0) & (~df["Minimax_Offset_Scaled"].isna())
    df.loc[mask_minimax_off_scaled,"MinimaxOffsetOverFillDistanceScaled"] = df.loc[mask_minimax_off_scaled,"Minimax_Offset_Scaled"] / df.loc[mask_minimax_off_scaled,"Fill_Distance_Scaled"]

    # Add new fill distance columns (knn mean/max scaled)
    df["Fill_Distance_KNN_Mean_Scaled"] = pd.to_numeric(df["Fill_Distance_KNN_Mean_Scaled"], errors='coerce') if "Fill_Distance_KNN_Mean_Scaled" in df.columns else np.nan
    df["Fill_Distance_KNN_Max_Scaled"] = pd.to_numeric(df["Fill_Distance_KNN_Max_Scaled"], errors='coerce') if "Fill_Distance_KNN_Max_Scaled" in df.columns else np.nan
    
    # Add new fill distance columns (knn mean/max unscaled)
    df["Fill_Distance_KNN_Mean"] = pd.to_numeric(df["Fill_Distance_KNN_Mean"], errors='coerce') if "Fill_Distance_KNN_Mean" in df.columns else np.nan
    df["Fill_Distance_KNN_Max"] = pd.to_numeric(df["Fill_Distance_KNN_Max"], errors='coerce') if "Fill_Distance_KNN_Max" in df.columns else np.nan

    # Add new offset/fill ratios for knn mean/max scaled
    df["OffsetOverFillDistanceKNNMeanScaled"] = np.nan
    mask_knnmean = (df["Fill_Distance_KNN_Mean_Scaled"]>0) & (~df["Offset"].isna())
    df.loc[mask_knnmean,"OffsetOverFillDistanceKNNMeanScaled"] = df.loc[mask_knnmean,"Offset"] / df.loc[mask_knnmean,"Fill_Distance_KNN_Mean_Scaled"]

    df["OffsetOverFillDistanceKNNMaxScaled"] = np.nan
    mask_knnmax = (df["Fill_Distance_KNN_Max_Scaled"]>0) & (~df["Offset"].isna())
    df.loc[mask_knnmax,"OffsetOverFillDistanceKNNMaxScaled"] = df.loc[mask_knnmax,"Offset"] / df.loc[mask_knnmax,"Fill_Distance_KNN_Max_Scaled"]
    
    # Add new offset/fill ratios for knn mean/max unscaled
    df["OffsetOverFillDistanceKNNMean"] = np.nan
    mask_knnmean_unscaled = (df["Fill_Distance_KNN_Mean"]>0) & (~df["Offset"].isna())
    df.loc[mask_knnmean_unscaled,"OffsetOverFillDistanceKNNMean"] = df.loc[mask_knnmean_unscaled,"Offset"] / df.loc[mask_knnmean_unscaled,"Fill_Distance_KNN_Mean"]

    df["OffsetOverFillDistanceKNNMax"] = np.nan
    mask_knnmax_unscaled = (df["Fill_Distance_KNN_Max"]>0) & (~df["Offset"].isna())
    df.loc[mask_knnmax_unscaled,"OffsetOverFillDistanceKNNMax"] = df.loc[mask_knnmax_unscaled,"Offset"] / df.loc[mask_knnmax_unscaled,"Fill_Distance_KNN_Max"]
    
    # Add new minimax offset/fill ratios for knn mean/max scaled
    df["MinimaxOffsetOverFillDistanceKNNMeanScaled"] = np.nan
    mask_minimax_knnmean = (df["Fill_Distance_KNN_Mean_Scaled"]>0) & (~df["Minimax_Offset"].isna())
    df.loc[mask_minimax_knnmean,"MinimaxOffsetOverFillDistanceKNNMeanScaled"] = df.loc[mask_minimax_knnmean,"Minimax_Offset"] / df.loc[mask_minimax_knnmean,"Fill_Distance_KNN_Mean_Scaled"]

    df["MinimaxOffsetOverFillDistanceKNNMaxScaled"] = np.nan
    mask_minimax_knnmax = (df["Fill_Distance_KNN_Max_Scaled"]>0) & (~df["Minimax_Offset"].isna())
    df.loc[mask_minimax_knnmax,"MinimaxOffsetOverFillDistanceKNNMaxScaled"] = df.loc[mask_minimax_knnmax,"Minimax_Offset"] / df.loc[mask_minimax_knnmax,"Fill_Distance_KNN_Max_Scaled"]
    
    # Add new minimax offset/fill ratios for knn mean/max unscaled
    df["MinimaxOffsetOverFillDistanceKNNMean"] = np.nan
    mask_minimax_knnmean_unscaled = (df["Fill_Distance_KNN_Mean"]>0) & (~df["Minimax_Offset"].isna())
    df.loc[mask_minimax_knnmean_unscaled,"MinimaxOffsetOverFillDistanceKNNMean"] = df.loc[mask_minimax_knnmean_unscaled,"Minimax_Offset"] / df.loc[mask_minimax_knnmean_unscaled,"Fill_Distance_KNN_Mean"]

    df["MinimaxOffsetOverFillDistanceKNNMax"] = np.nan
    mask_minimax_knnmax_unscaled = (df["Fill_Distance_KNN_Max"]>0) & (~df["Minimax_Offset"].isna())
    df.loc[mask_minimax_knnmax_unscaled,"MinimaxOffsetOverFillDistanceKNNMax"] = df.loc[mask_minimax_knnmax_unscaled,"Minimax_Offset"] / df.loc[mask_minimax_knnmax_unscaled,"Fill_Distance_KNN_Max"]

    # NEW => minimaxOffsetScaledOverFillDistance
    df["MinimaxOffsetScaledOverFillDistance"] = np.nan
    mask_minimax_scaled_off = (df["Fill_Distance"]>0) & (~df["Minimax_Offset_Scaled"].isna())
    df.loc[mask_minimax_scaled_off,"MinimaxOffsetScaledOverFillDistance"] = df.loc[mask_minimax_scaled_off,"Minimax_Offset_Scaled"] / df.loc[mask_minimax_scaled_off,"Fill_Distance"]
    
    # NEW => minimaxOffsetScaledOverFillDistanceKNNMean
    df["MinimaxOffsetScaledOverFillDistanceKNNMean"] = np.nan
    mask_minimax_scaled_knnmean_unscaled = (df["Fill_Distance_KNN_Mean"]>0) & (~df["Minimax_Offset_Scaled"].isna())
    df.loc[mask_minimax_scaled_knnmean_unscaled,"MinimaxOffsetScaledOverFillDistanceKNNMean"] = df.loc[mask_minimax_scaled_knnmean_unscaled,"Minimax_Offset_Scaled"] / df.loc[mask_minimax_scaled_knnmean_unscaled,"Fill_Distance_KNN_Mean"]
    
    # NEW => minimaxOffsetScaledOverFillDistanceKNNMax
    df["MinimaxOffsetScaledOverFillDistanceKNNMax"] = np.nan
    mask_minimax_scaled_knnmax_unscaled = (df["Fill_Distance_KNN_Max"]>0) & (~df["Minimax_Offset_Scaled"].isna())
    df.loc[mask_minimax_scaled_knnmax_unscaled,"MinimaxOffsetScaledOverFillDistanceKNNMax"] = df.loc[mask_minimax_scaled_knnmax_unscaled,"Minimax_Offset_Scaled"] / df.loc[mask_minimax_scaled_knnmax_unscaled,"Fill_Distance_KNN_Max"]

    # dimension category using our custom function
    df["DimensionCategory"] = df["dimension"].apply(categorize_by_dimension)

    logger.info(f"Preprocessing complete. {len(df)} rows remain.")
    return df

##############################################################################
# UTILITY FUNCTIONS
##############################################################################

def save_plot(fig, output_path, dpi=300):
    """
    Save figure in both PDF and SVG formats.
    
    Parameters
    ----------
    fig : matplotlib.figure.Figure
        The figure to save
    output_path : str
        The output path without extension
    dpi : int, optional
        Resolution in dots per inch, by default 300
    """
    # Save PDF version
    pdf_path = f"{output_path}.pdf"
    fig.savefig(pdf_path, bbox_inches="tight", dpi=dpi)
    
    # Save SVG version
    svg_path = f"{output_path}.svg"
    fig.savefig(svg_path, bbox_inches="tight", dpi=dpi)
    
    return pdf_path, svg_path

##############################################################################
# Helpers for binning & line plotting (3 subplots)
##############################################################################

def bin_data_column(sub, xcol, ycol, groupcol, nbins=8):
    out={}
    sub = sub.dropna(subset=[xcol,ycol,groupcol])
    if sub.empty:
        return out
    mn,mx = sub[xcol].min(), sub[xcol].max()
    if mn==mx:
        # all x are the same => single bin
        bins=[mn, mn+1e-9]
        sub["binid"]=0
    else:
        bins=np.linspace(mn,mx,nbins+1)
        sub["binid"] = pd.cut(sub[xcol], bins=bins, labels=False, include_lowest=True)
    for gv in sorted(sub[groupcol].dropna().unique()):
        cch=sub[sub[groupcol]==gv]
        xv,yv,sv=[],[],[]
        for bbb in sorted(cch["binid"].unique()):
            chunk = cch[cch["binid"]==bbb]
            if chunk.empty:
                continue
            xv.append(chunk[xcol].mean())
            yv.append(chunk[ycol].mean())
            sv.append(chunk[ycol].std() if len(chunk)>1 else 0)
        if xv:
            ax_=np.array(xv)
            ay_=np.array(yv)
            as_=np.array(sv)
            sidx = np.argsort(ax_)
            out[gv] = (ax_[sidx], ay_[sidx], as_[sidx])
    return out

def multi_plot_3metrics_vs_x(axes, df_sub, xcol, x_label, add_metric_labels=False):
    """
    Create a 3-panel plot showing ARI, GW, and GH metrics vs. a specified x variable
    
    Parameters
    ----------
    axes : list
        List of 3 matplotlib axes: [axA, axB, axC]
          A => ARI (no grouping)
          B => GW_Uniform_Uniform => group=Kernel_Method
          C => GH_Ultrametric_Emb => group=Embedding_Method
    df_sub : pandas.DataFrame
        Data frame containing the data to plot
    xcol : str
        Name of the column to use for the x-axis
    x_label : str
        Label to use for the x-axis
    add_metric_labels : bool, optional
        Whether to add metric value labels to the plot, by default True
    """
    axA, axB, axC = axes
    
    # Get customizable labels from config
    ari_label = METRIC_LABELS.get("ARI", "ARI")
    gw_metric_label = METRIC_LABELS.get("GW_Uniform_Uniform", "Graph GW Metric")
    gh_metric_label = METRIC_LABELS.get("GH_Ultrametric_Emb", "GW Tree Embedding")
    
    # Get x-axis label (use provided or get from config)
    x_axis_label = METRIC_LABELS.get(xcol, x_label)

    # Function to add value labels
    def add_value_labels(ax, x, y, labels=None, offset=(0, 0.02), fontsize=6):
        if labels is None:
            labels = [f"{v:.2f}" for v in y]
        for i, (x_val, y_val, label) in enumerate(zip(x, y, labels)):
            ax.annotate(label,
                      (x_val, y_val),
                      textcoords="offset points",
                      xytext=offset,
                      ha='center',
                      fontsize=fontsize)

    #--- ARI
    subA = df_sub.dropna(subset=[xcol,"ARI"])
    if subA.empty:
        axA.text(0.5,0.5,"No ARI data",ha='center',va='center',transform=axA.transAxes,color='red')
        axA.set_title(ari_label)
    else:
        # single group "All"
        subA = subA.copy()
        subA["TmpGroup"]="All"
        ddA=bin_data_column(subA,xcol,"ARI","TmpGroup", nbins=8)
        if ddA:
            for gval,(xx,yy,ss) in ddA.items():
                line = axA.errorbar(xx,yy,yerr=ss,marker='o',linestyle='-',capsize=2)
                if add_metric_labels:
                    add_value_labels(axA, xx, yy)
        else:
            axA.text(0.5,0.5,"No ARI bin",ha='center',va='center',transform=axA.transAxes,color='red')
        axA.set_title(ari_label)
        axA.set_xlabel(x_axis_label)
        axA.set_ylabel(ari_label)
        axA.grid(True, alpha=0.3, linestyle='--')

    #--- Graph GW
    subB=df_sub.dropna(subset=[xcol,"GW_Uniform_Uniform","Kernel_Method"])
    if subB.empty:
        axB.text(0.5,0.5,"No GraphGW data",ha='center',va='center',transform=axB.transAxes,color='red')
        axB.set_title(gw_metric_label)
    else:
        ddB=bin_data_column(subB,xcol,"GW_Uniform_Uniform","Kernel_Method", nbins=8)
        if ddB:
            for kv,(xx,yy,ss) in ddB.items():
                line = axB.errorbar(xx,yy,yerr=ss,marker='o',linestyle='-',capsize=2,label=str(kv))
                if add_metric_labels:
                    add_value_labels(axB, xx, yy)
            axB.legend(fontsize=7)
        else:
            axB.text(0.5,0.5,"No GraphGW bin",ha='center',va='center',transform=axB.transAxes,color='red')
        axB.set_title(gw_metric_label)
        axB.set_xlabel(x_axis_label)
        axB.set_ylabel(gw_metric_label)
        axB.grid(True, alpha=0.3, linestyle='--')

    #--- Tree Embedding
    subC=df_sub.dropna(subset=[xcol,"GH_Ultrametric_Emb","Embedding_Method"])
    if subC.empty:
        axC.text(0.5,0.5,"No TreeEmb data",ha='center',va='center',transform=axC.transAxes,color='red')
        axC.set_title(gh_metric_label)
    else:
        ddC=bin_data_column(subC,xcol,"GH_Ultrametric_Emb","Embedding_Method", nbins=8)
        if ddC:
            for ev,(xx,yy,ss) in ddC.items():
                line = axC.errorbar(xx,yy,yerr=ss,marker='o',linestyle='-',capsize=2,label=str(ev))
                if add_metric_labels:
                    add_value_labels(axC, xx, yy)
            axC.legend(fontsize=7)
        else:
            axC.text(0.5,0.5,"No TreeEmb bin",ha='center',va='center',transform=axC.transAxes,color='red')
        axC.set_title(gh_metric_label)
        axC.set_xlabel(x_axis_label)
        axC.set_ylabel(gh_metric_label)
        axC.grid(True, alpha=0.3, linestyle='--')

##############################################################################
# SUMMARY TABLE GENERATION FUNCTIONS
##############################################################################

def generate_dim_summary_table(df, dim_cat, outdir, logger):
    """
    Generate a summary table for a specific dimension category.
    
    Parameters
    ----------
    df : pandas.DataFrame
        The dataframe containing the data for this dimension category
    dim_cat : str
        The dimension category name
    outdir : str
        Output directory for this dimension category
    logger : logging.Logger
        Logger instance
    """
    logger.info(f"Generating summary table for dimension category {dim_cat}")
    
    # Basic statistics across metrics
    metrics = ["ARI", "GW_Uniform_Uniform", "GH_Ultrametric_Emb"]
    stats = {}
    
    # Calculate statistics for each metric
    for metric in metrics:
        valid_data = df[metric].dropna()
        if not valid_data.empty:
            stats[metric] = {
                "count": len(valid_data),
                "mean": valid_data.mean(),
                "std": valid_data.std(),
                "min": valid_data.min(),
                "max": valid_data.max(),
                "median": valid_data.median(),
                "above_threshold": (valid_data > 0.8).sum() if metric == "ARI" else (valid_data < 0.3).sum(),
                "threshold_percent": (valid_data > 0.8).mean() * 100 if metric == "ARI" else (valid_data < 0.3).mean() * 100
            }
    
    # Calculate average metrics by key factors
    key_cols = ["Offset", "Fill_Distance", "OffsetOverSigma", "OffsetOverFillDistance"]
    averages_by_factor = {}
    
    for col in key_cols:
        df_valid = df.dropna(subset=[col])
        if df_valid.empty:
            continue
            
        # Create bins for the column values
        try:
            bins = pd.qcut(df_valid[col], 4)
            df_valid["bin"] = bins
        
            # Group by bin and calculate average metrics
            grouped = df_valid.groupby("bin")[metrics].mean().reset_index()
            averages_by_factor[col] = grouped
        except ValueError as e:
            logger.warning(f"Cannot create quartiles for {col} in {dim_cat}: {e}")
            continue
    
    # Create a markdown summary table
    md_table = f"# Summary for Dimension Category: {dim_cat}\n\n"
    md_table += f"Total samples: {len(df)}\n\n"
    
    # Add metrics statistics table
    md_table += "## Metrics Statistics\n\n"
    md_table += "| Metric | Count | Mean | Std | Min | Max | Median | Success Rate |\n"
    md_table += "| ------ | ----- | ---- | --- | --- | --- | ------ | ------------ |\n"
    
    for metric in metrics:
        if metric in stats:
            s = stats[metric]
            threshold_type = "> 0.8" if metric == "ARI" else "< 0.3"
            md_table += f"| {METRIC_LABELS.get(metric, metric)} | {s['count']} | {s['mean']:.4f} | {s['std']:.4f} | {s['min']:.4f} | {s['max']:.4f} | {s['median']:.4f} | {s['above_threshold']} ({s['threshold_percent']:.2f}%) {threshold_type} |\n"
    
    # Add tables for averages by factor
    for col, grouped in averages_by_factor.items():
        md_table += f"\n## Average Metrics by {METRIC_LABELS.get(col, col)}\n\n"
        md_table += "| Range | ARI | Graph GW Metric | GW Tree Embedding |\n"
        md_table += "| ----- | --- | --------------- | ----------------- |\n"
        
        for _, row in grouped.iterrows():
            bin_range = str(row["bin"])
            md_table += f"| {bin_range} | {row['ARI']:.4f} | {row['GW_Uniform_Uniform']:.4f} | {row['GH_Ultrametric_Emb']:.4f} |\n"
    
    # Save the markdown table
    md_path = os.path.join(outdir, f"{dim_cat}_summary.md")
    with open(md_path, "w") as f:
        f.write(md_table)
    
    # Generate a CSV with more detailed statistics
    csv_path = os.path.join(outdir, f"{dim_cat}_summary_data.csv")
    
    # Create a dataframe for the CSV
    summary_df = pd.DataFrame()
    for metric in metrics:
        if metric in stats:
            for stat_name, stat_value in stats[metric].items():
                summary_df.loc[metric, stat_name] = stat_value
    
    summary_df.to_csv(csv_path)
    logger.info(f"Summary table for {dim_cat} saved to {md_path} and {csv_path}")

def generate_noise_bias_summary_table(df, category, outdir, logger):
    """
    Generate a summary table for a noise or bias category.
    
    Parameters
    ----------
    df : pandas.DataFrame
        The dataframe containing the data for this category
    category : str
        The category name (e.g., "Noisy", "Clean", "Biased", "Unbiased")
    outdir : str
        Output directory for this category
    logger : logging.Logger
        Logger instance
    """
    logger.info(f"Generating summary table for {category} data")
    
    # Basic statistics across metrics
    metrics = ["ARI", "GW_Uniform_Uniform", "GH_Ultrametric_Emb"]
    stats = {}
    
    # Calculate statistics for each metric
    for metric in metrics:
        valid_data = df[metric].dropna()
        if not valid_data.empty:
            stats[metric] = {
                "count": len(valid_data),
                "mean": valid_data.mean(),
                "std": valid_data.std(),
                "min": valid_data.min(),
                "max": valid_data.max(),
                "median": valid_data.median(),
                "above_threshold": (valid_data > 0.8).sum() if metric == "ARI" else (valid_data < 0.3).sum(),
                "threshold_percent": (valid_data > 0.8).mean() * 100 if metric == "ARI" else (valid_data < 0.3).mean() * 100
            }
    
    # Group by dimension category
    dim_stats = {}
    for dim_cat in df["DimensionCategory"].dropna().unique():
        df_dim = df[df["DimensionCategory"] == dim_cat]
        dim_stats[dim_cat] = {}
        
        for metric in metrics:
            valid_data = df_dim[metric].dropna()
            if not valid_data.empty:
                dim_stats[dim_cat][metric] = {
                    "count": len(valid_data),
                    "mean": valid_data.mean(),
                    "std": valid_data.std()
                }
    
    # Create a markdown summary table
    md_table = f"# Summary for {category} Data\n\n"
    md_table += f"Total samples: {len(df)}\n\n"
    
    # Add overall metrics statistics table
    md_table += "## Overall Metrics Statistics\n\n"
    md_table += "| Metric | Count | Mean | Std | Min | Max | Median | Success Rate |\n"
    md_table += "| ------ | ----- | ---- | --- | --- | --- | ------ | ------------ |\n"
    
    for metric in metrics:
        if metric in stats:
            s = stats[metric]
            threshold_type = "> 0.8" if metric == "ARI" else "< 0.3"
            md_table += f"| {METRIC_LABELS.get(metric, metric)} | {s['count']} | {s['mean']:.4f} | {s['std']:.4f} | {s['min']:.4f} | {s['max']:.4f} | {s['median']:.4f} | {s['above_threshold']} ({s['threshold_percent']:.2f}%) {threshold_type} |\n"
    
    # Add metrics by dimension table
    md_table += f"\n## Metrics by Dimension Category\n\n"
    md_table += "| Dimension | Metric | Count | Mean | Std |\n"
    md_table += "| --------- | ------ | ----- | ---- | --- |\n"
    
    for dim_cat in sorted(dim_stats.keys()):
        for metric in metrics:
            if metric in dim_stats[dim_cat]:
                s = dim_stats[dim_cat][metric]
                md_table += f"| {dim_cat} | {METRIC_LABELS.get(metric, metric)} | {s['count']} | {s['mean']:.4f} | {s['std']:.4f} |\n"
    
    # Save the markdown table
    md_path = os.path.join(outdir, f"{category}_summary.md")
    with open(md_path, "w") as f:
        f.write(md_table)
    
    # Generate a CSV with more detailed statistics
    csv_path = os.path.join(outdir, f"{category}_summary_data.csv")
    
    # Create a dataframe for the CSV
    summary_df = pd.DataFrame()
    for metric in metrics:
        if metric in stats:
            for stat_name, stat_value in stats[metric].items():
                summary_df.loc[metric, stat_name] = stat_value
    
    summary_df.to_csv(csv_path)
    logger.info(f"Summary table for {category} saved to {md_path} and {csv_path}")

def generate_noise_comparison_table(df_noisy, df_clean, outdir, logger):
    """
    Generate a comparison table between noisy and clean data.
    
    Parameters
    ----------
    df_noisy : pandas.DataFrame
        The dataframe containing noisy data
    df_clean : pandas.DataFrame
        The dataframe containing clean data
    outdir : str
        Output directory for the comparison table
    logger : logging.Logger
        Logger instance
    """
    logger.info("Generating noise comparison table")
    
    metrics = ["ARI", "GW_Uniform_Uniform", "GH_Ultrametric_Emb"]
    comparison = {}
    
    # Calculate statistics for each metric
    for metric in metrics:
        noisy_data = df_noisy[metric].dropna()
        clean_data = df_clean[metric].dropna()
        
        if not noisy_data.empty and not clean_data.empty:
            comparison[metric] = {
                "noisy_count": len(noisy_data),
                "noisy_mean": noisy_data.mean(),
                "noisy_std": noisy_data.std(),
                "clean_count": len(clean_data),
                "clean_mean": clean_data.mean(),
                "clean_std": clean_data.std(),
                "mean_diff": clean_data.mean() - noisy_data.mean(),
                "mean_diff_percent": ((clean_data.mean() - noisy_data.mean()) / noisy_data.mean()) * 100 if noisy_data.mean() != 0 else 0
            }
            
            # Add statistical significance test if possible
            try:
                if len(noisy_data) >= 5 and len(clean_data) >= 5:
                    # Mann-Whitney U test (does not assume normality)
                    stat, p_value = mannwhitneyu(noisy_data, clean_data, alternative='two-sided')
                    comparison[metric]["p_value"] = p_value
                    comparison[metric]["significant"] = p_value < 0.05
                else:
                    comparison[metric]["p_value"] = None
                    comparison[metric]["significant"] = None
            except Exception as e:
                logger.warning(f"Statistical test failed for {metric}: {e}")
                comparison[metric]["p_value"] = None
                comparison[metric]["significant"] = None
    
    # Create a markdown comparison table
    md_table = "# Comparison: Noisy vs. Clean Data\n\n"
    md_table += f"Noisy samples: {len(df_noisy)}, Clean samples: {len(df_clean)}\n\n"
    
    md_table += "## Metrics Comparison\n\n"
    md_table += "| Metric | Noisy Mean (n) | Clean Mean (n) | Difference | % Change | p-value | Significant? |\n"
    md_table += "| ------ | -------------- | -------------- | ---------- | -------- | ------- | ------------ |\n"
    
    for metric in metrics:
        if metric in comparison:
            c = comparison[metric]
            sig_text = "Yes" if c["significant"] == True else "No" if c["significant"] == False else "N/A"
            p_val_text = f"{c['p_value']:.4f}" if c["p_value"] is not None else "N/A"
            
            md_table += f"| {METRIC_LABELS.get(metric, metric)} | {c['noisy_mean']:.4f} ({c['noisy_count']}) | {c['clean_mean']:.4f} ({c['clean_count']}) | {c['mean_diff']:.4f} | {c['mean_diff_percent']:.2f}% | {p_val_text} | {sig_text} |\n"
    
    # Save the markdown table
    md_path = os.path.join(outdir, "noisy_vs_clean_comparison.md")
    with open(md_path, "w") as f:
        f.write(md_table)
    
    # Generate a CSV with more detailed statistics
    csv_path = os.path.join(outdir, "noisy_vs_clean_comparison_data.csv")
    
    # Create a dataframe for the CSV
    comparison_df = pd.DataFrame()
    for metric in metrics:
        if metric in comparison:
            for stat_name, stat_value in comparison[metric].items():
                comparison_df.loc[metric, stat_name] = stat_value
    
    comparison_df.to_csv(csv_path)
    logger.info(f"Noise comparison table saved to {md_path} and {csv_path}")

def generate_bias_comparison_table(df_biased, df_unbiased, outdir, logger):
    """
    Generate a comparison table between biased and unbiased data.
    
    Parameters
    ----------
    df_biased : pandas.DataFrame
        The dataframe containing biased data
    df_unbiased : pandas.DataFrame
        The dataframe containing unbiased data
    outdir : str
        Output directory for the comparison table
    logger : logging.Logger
        Logger instance
    """
    logger.info("Generating bias comparison table")
    
    metrics = ["ARI", "GW_Uniform_Uniform", "GH_Ultrametric_Emb"]
    comparison = {}
    
    # Calculate statistics for each metric
    for metric in metrics:
        biased_data = df_biased[metric].dropna()
        unbiased_data = df_unbiased[metric].dropna()
        
        if not biased_data.empty and not unbiased_data.empty:
            comparison[metric] = {
                "biased_count": len(biased_data),
                "biased_mean": biased_data.mean(),
                "biased_std": biased_data.std(),
                "unbiased_count": len(unbiased_data),
                "unbiased_mean": unbiased_data.mean(),
                "unbiased_std": unbiased_data.std(),
                "mean_diff": unbiased_data.mean() - biased_data.mean(),
                "mean_diff_percent": ((unbiased_data.mean() - biased_data.mean()) / biased_data.mean()) * 100 if biased_data.mean() != 0 else 0
            }
            
            # Add statistical significance test if possible
            try:
                if len(biased_data) >= 5 and len(unbiased_data) >= 5:
                    # Mann-Whitney U test (does not assume normality)
                    stat, p_value = mannwhitneyu(biased_data, unbiased_data, alternative='two-sided')
                    comparison[metric]["p_value"] = p_value
                    comparison[metric]["significant"] = p_value < 0.05
                else:
                    comparison[metric]["p_value"] = None
                    comparison[metric]["significant"] = None
            except Exception as e:
                logger.warning(f"Statistical test failed for {metric}: {e}")
                comparison[metric]["p_value"] = None
                comparison[metric]["significant"] = None
    
    # Create a markdown comparison table
    md_table = "# Comparison: Biased vs. Unbiased Data\n\n"
    md_table += f"Biased samples: {len(df_biased)}, Unbiased samples: {len(df_unbiased)}\n\n"
    
    md_table += "## Metrics Comparison\n\n"
    md_table += "| Metric | Biased Mean (n) | Unbiased Mean (n) | Difference | % Change | p-value | Significant? |\n"
    md_table += "| ------ | --------------- | ----------------- | ---------- | -------- | ------- | ------------ |\n"
    
    for metric in metrics:
        if metric in comparison:
            c = comparison[metric]
            sig_text = "Yes" if c["significant"] == True else "No" if c["significant"] == False else "N/A"
            p_val_text = f"{c['p_value']:.4f}" if c["p_value"] is not None else "N/A"
            
            md_table += f"| {METRIC_LABELS.get(metric, metric)} | {c['biased_mean']:.4f} ({c['biased_count']}) | {c['unbiased_mean']:.4f} ({c['unbiased_count']}) | {c['mean_diff']:.4f} | {c['mean_diff_percent']:.2f}% | {p_val_text} | {sig_text} |\n"
    
    # Save the markdown table
    md_path = os.path.join(outdir, "biased_vs_unbiased_comparison.md")
    with open(md_path, "w") as f:
        f.write(md_table)
    
    # Generate a CSV with more detailed statistics
    csv_path = os.path.join(outdir, "biased_vs_unbiased_comparison_data.csv")
    
    # Create a dataframe for the CSV
    comparison_df = pd.DataFrame()
    for metric in metrics:
        if metric in comparison:
            for stat_name, stat_value in comparison[metric].items():
                comparison_df.loc[metric, stat_name] = stat_value
    
    comparison_df.to_csv(csv_path)
    logger.info(f"Bias comparison table saved to {md_path} and {csv_path}")

def generate_combined_summary_table(df, prefix, outdir, logger):
    """
    Generate a summary table for a combined dimension, noise, bias category.
    
    Parameters
    ----------
    df : pandas.DataFrame
        The dataframe containing the data for this combined category
    prefix : str
        Prefix for filenames
    outdir : str
        Output directory for this combined category
    logger : logging.Logger
        Logger instance
    """
    logger.info(f"Generating summary table for {prefix}")
    
    # Basic statistics across metrics
    metrics = ["ARI", "GW_Uniform_Uniform", "GH_Ultrametric_Emb"]
    stats = {}
    
    # Calculate statistics for each metric
    for metric in metrics:
        valid_data = df[metric].dropna()
        if not valid_data.empty:
            stats[metric] = {
                "count": len(valid_data),
                "mean": valid_data.mean(),
                "std": valid_data.std(),
                "min": valid_data.min(),
                "max": valid_data.max(),
                "median": valid_data.median(),
                "above_threshold": (valid_data > 0.8).sum() if metric == "ARI" else (valid_data < 0.3).sum(),
                "threshold_percent": (valid_data > 0.8).mean() * 100 if metric == "ARI" else (valid_data < 0.3).mean() * 100
            }
    
    # Calculate correlation between key factors and metrics
    key_cols = ["Offset", "Fill_Distance", "OffsetOverSigma", "OffsetOverFillDistance"]
    correlations = {}
    
    for col in key_cols:
        correlations[col] = {}
        for metric in metrics:
            # Calculate Spearman rank correlation
            df_valid = df.dropna(subset=[col, metric])
            if len(df_valid) >= 5:  # Need a minimum number of points
                try:
                    corr, p_val = spearmanr(df_valid[col], df_valid[metric])
                    correlations[col][metric] = {
                        "correlation": corr,
                        "p_value": p_val,
                        "significant": p_val < 0.05
                    }
                except Exception as e:
                    logger.warning(f"Correlation calculation failed for {col} vs {metric}: {e}")
                    correlations[col][metric] = {
                        "correlation": None,
                        "p_value": None,
                        "significant": None
                    }
            else:
                correlations[col][metric] = {
                    "correlation": None,
                    "p_value": None,
                    "significant": None
                }
    
    # Create a markdown summary table
    md_table = f"# Summary for {prefix}\n\n"
    md_table += f"Total samples: {len(df)}\n\n"
    
    # Add metrics statistics table
    md_table += "## Metrics Statistics\n\n"
    md_table += "| Metric | Count | Mean | Std | Min | Max | Median | Success Rate |\n"
    md_table += "| ------ | ----- | ---- | --- | --- | --- | ------ | ------------ |\n"
    
    for metric in metrics:
        if metric in stats:
            s = stats[metric]
            threshold_type = "> 0.8" if metric == "ARI" else "< 0.3"
            md_table += f"| {METRIC_LABELS.get(metric, metric)} | {s['count']} | {s['mean']:.4f} | {s['std']:.4f} | {s['min']:.4f} | {s['max']:.4f} | {s['median']:.4f} | {s['above_threshold']} ({s['threshold_percent']:.2f}%) {threshold_type} |\n"
    
    # Add correlation table
    md_table += f"\n## Correlations with Metrics\n\n"
    md_table += "| Factor | Metric | Correlation | p-value | Significant? |\n"
    md_table += "| ------ | ------ | ----------- | ------- | ------------ |\n"
    
    for col in key_cols:
        for metric in metrics:
            if col in correlations and metric in correlations[col]:
                c = correlations[col][metric]
                
                if c["correlation"] is not None:
                    sig_text = "Yes" if c["significant"] == True else "No" if c["significant"] == False else "N/A"
                    p_val_text = f"{c['p_value']:.4f}" if c["p_value"] is not None else "N/A"
                    
                    md_table += f"| {METRIC_LABELS.get(col, col)} | {METRIC_LABELS.get(metric, metric)} | {c['correlation']:.4f} | {p_val_text} | {sig_text} |\n"
    
    # Save the markdown table
    md_path = os.path.join(outdir, f"{prefix}_summary.md")
    with open(md_path, "w") as f:
        f.write(md_table)
    
    # Generate a CSV with more detailed statistics
    csv_path = os.path.join(outdir, f"{prefix}_summary_data.csv")
    
    # Create a dataframe for the CSV
    summary_df = pd.DataFrame()
    for metric in metrics:
        if metric in stats:
            for stat_name, stat_value in stats[metric].items():
                summary_df.loc[metric, stat_name] = stat_value
    
    summary_df.to_csv(csv_path)
    logger.info(f"Summary table for {prefix} saved to {md_path} and {csv_path}")

##############################################################################
# Single dataset analysis
##############################################################################

def single_base_dataset_analysis(df_sub, base_name, outdir, logger):
    """
    For each dataset, we create 2x additional plots for the new columns:
      - vs. 'Offset'
      - vs. 'OffsetOverFillDistance'
      - vs. 'OffsetOverFillDistanceScaled'
    and also the existing ones:
      - vs. 'Fill_Distance'
      - vs. 'Sample_Percentage'
      - vs. 'OffsetOverSigma' (by sample fraction)
      - vs. 'Ratio_OS_over_Fill'
    """
    if df_sub.empty:
        logger.warning(f"[{base_name}] No data => skip.")
        return
    logger.info(f"=== Single dataset => {base_name} ===")

    # (A) vs FillDistance
    fig,axes=plt.subplots(1,3,figsize=(14,4))
    fig.suptitle(f"{base_name}: vs FillDistance",fontsize=13)
    multi_plot_3metrics_vs_x(axes, df_sub, "Fill_Distance","FillDist")
    outA=os.path.join(outdir,f"{base_name}_3plot_vs_FillDistance")
    fig.tight_layout(rect=[0,0,1,0.92])
    save_plot(fig, outA, dpi=300)
    plt.close(fig)

    # (A2) vs Scaled FillDistance
    fig,axes=plt.subplots(1,3,figsize=(14,4))
    fig.suptitle(f"{base_name}: vs Scaled FillDistance",fontsize=13)
    multi_plot_3metrics_vs_x(axes, df_sub, "Fill_Distance_Scaled","FillDistScaled")
    outAscaled=os.path.join(outdir,f"{base_name}_3plot_vs_FillDistanceScaled")
    fig.tight_layout(rect=[0,0,1,0.92])
    save_plot(fig, outAscaled, dpi=300)
    plt.close(fig)

    # (B) vs Sample_Percentage
    fig,axes=plt.subplots(1,3,figsize=(14,4))
    fig.suptitle(f"{base_name}: vs SamplePct",fontsize=13)
    multi_plot_3metrics_vs_x(axes, df_sub, "Sample_Percentage","SamplePct")
    outB=os.path.join(outdir,f"{base_name}_3plot_vs_SamplePct")
    fig.tight_layout(rect=[0,0,1,0.92])
    save_plot(fig, outB, dpi=300)
    plt.close(fig)

    # (C) offsetOverSigma plots => by sample fraction
    spcts=sorted(df_sub["Sample_Percentage"].dropna().unique())
    for spc in spcts:
        subp=df_sub[(df_sub["Sample_Percentage"]==spc)&(~df_sub["OffsetOverSigma"].isna())]
        if subp.empty:
            continue
        fig,axes=plt.subplots(1,3,figsize=(14,4))
        fig.suptitle(f"{base_name}: sample={spc} => vs OffsetOverSigma",fontsize=13)
        multi_plot_3metrics_vs_x(axes, subp, "OffsetOverSigma","Offset/Sigma")
        outC=os.path.join(outdir,f"{base_name}_sample_{spc:.2f}_3plot_vs_OffsetOverSigma")
        fig.tight_layout(rect=[0,0,1,0.92])
        save_plot(fig, outC, dpi=300)
        plt.close(fig)

    # ratio => ratio_plots
    ratio_dir=os.path.join(outdir,"ratio_plots")
    os.makedirs(ratio_dir,exist_ok=True)
    df_tmp = df_sub.copy()
    df_tmp["Ratio"] = df_tmp["Ratio_OS_over_Fill"]
    fig,axes=plt.subplots(1,3,figsize=(14,4))
    fig.suptitle(f"{base_name}: vs RatioOverFill",fontsize=13)
    multi_plot_3metrics_vs_x(axes, df_tmp, "Ratio","Ratio OS/FILL")
    outR=os.path.join(ratio_dir,f"{base_name}_3plot_vs_RatioOverFill")
    fig.tight_layout(rect=[0,0,1,0.92])
    save_plot(fig, outR, dpi=300)
    plt.close(fig)

    # (NEW) vs Offset
    fig,axes=plt.subplots(1,3,figsize=(14,4))
    fig.suptitle(f"{base_name}: vs Offset",fontsize=13)
    multi_plot_3metrics_vs_x(axes, df_sub, "Offset","Offset")
    outOf = os.path.join(outdir,f"{base_name}_3plot_vs_Offset")
    fig.tight_layout(rect=[0,0,1,0.92])
    save_plot(fig, outOf, dpi=300)
    plt.close(fig)

    # (NEW) vs OffsetOverFillDistance
    fig,axes=plt.subplots(1,3,figsize=(14,4))
    fig.suptitle(f"{base_name}: vs Offset/FillDist",fontsize=13)
    multi_plot_3metrics_vs_x(axes, df_sub, "OffsetOverFillDistance","Offset/FillDist")
    outOf2 = os.path.join(outdir,f"{base_name}_3plot_vs_OffsetOverFillDistance")
    fig.tight_layout(rect=[0,0,1,0.92])
    save_plot(fig, outOf2, dpi=300)
    plt.close(fig)

    # (NEW) vs OffsetOverFillDistanceScaled
    fig,axes=plt.subplots(1,3,figsize=(14,4))
    fig.suptitle(f"{base_name}: vs Offset/FillDistScaled",fontsize=13)
    multi_plot_3metrics_vs_x(axes, df_sub, "OffsetOverFillDistanceScaled","Offset/FillDistScaled")
    outOf3 = os.path.join(outdir,f"{base_name}_3plot_vs_OffsetOverFillDistanceScaled")
    fig.tight_layout(rect=[0,0,1,0.92])
    save_plot(fig, outOf3, dpi=300)
    plt.close(fig)

    # (NEW) vs Fill_Distance_KNN_Mean_Scaled
    fig,axes=plt.subplots(1,3,figsize=(14,4))
    fig.suptitle(f"{base_name}: vs FillDistanceKNNMeanScaled",fontsize=13)
    multi_plot_3metrics_vs_x(axes, df_sub, "Fill_Distance_KNN_Mean_Scaled","FillDistKNNMeanScaled")
    outKNNmean=os.path.join(outdir,f"{base_name}_3plot_vs_FillDistanceKNNMeanScaled")
    fig.tight_layout(rect=[0,0,1,0.92])
    save_plot(fig, outKNNmean, dpi=300)
    plt.close(fig)

    # (NEW) vs Fill_Distance_KNN_Max_Scaled
    fig,axes=plt.subplots(1,3,figsize=(14,4))
    fig.suptitle(f"{base_name}: vs FillDistanceKNNMaxScaled",fontsize=13)
    multi_plot_3metrics_vs_x(axes, df_sub, "Fill_Distance_KNN_Max_Scaled","FillDistKNNMaxScaled")
    outKNNmax=os.path.join(outdir,f"{base_name}_3plot_vs_FillDistanceKNNMaxScaled")
    fig.tight_layout(rect=[0,0,1,0.92])
    save_plot(fig, outKNNmax, dpi=300)
    plt.close(fig)

    # (NEW) vs OffsetOverFillDistanceKNNMeanScaled
    fig,axes=plt.subplots(1,3,figsize=(14,4))
    fig.suptitle(f"{base_name}: vs Offset/FillDistKNNMeanScaled",fontsize=13)
    multi_plot_3metrics_vs_x(axes, df_sub, "OffsetOverFillDistanceKNNMeanScaled","Offset/FillDistKNNMeanScaled")
    outKNNmean2 = os.path.join(outdir,f"{base_name}_3plot_vs_OffsetOverFillDistanceKNNMeanScaled")
    fig.tight_layout(rect=[0,0,1,0.92])
    save_plot(fig, outKNNmean2, dpi=300)
    plt.close(fig)

    # (NEW) vs OffsetOverFillDistanceKNNMaxScaled
    fig,axes=plt.subplots(1,3,figsize=(14,4))
    fig.suptitle(f"{base_name}: vs Offset/FillDistKNNMaxScaled",fontsize=13)
    multi_plot_3metrics_vs_x(axes, df_sub, "OffsetOverFillDistanceKNNMaxScaled","Offset/FillDistKNNMaxScaled")
    outKNNmax2 = os.path.join(outdir,f"{base_name}_3plot_vs_OffsetOverFillDistanceKNNMaxScaled")
    fig.tight_layout(rect=[0,0,1,0.92])
    save_plot(fig, outKNNmax2, dpi=300)
    plt.close(fig)
    
    # (NEW) vs Fill_Distance_KNN_Mean (unscaled)
    fig,axes=plt.subplots(1,3,figsize=(14,4))
    fig.suptitle(f"{base_name}: vs FillDistanceKNNMean (Unscaled)",fontsize=13)
    multi_plot_3metrics_vs_x(axes, df_sub, "Fill_Distance_KNN_Mean","FillDistKNNMean")
    outKNNmean_unscaled=os.path.join(outdir,f"{base_name}_3plot_vs_FillDistanceKNNMean")
    fig.tight_layout(rect=[0,0,1,0.92])
    save_plot(fig, outKNNmean_unscaled, dpi=300)
    plt.close(fig)

    # (NEW) vs Fill_Distance_KNN_Max (unscaled)
    fig,axes=plt.subplots(1,3,figsize=(14,4))
    fig.suptitle(f"{base_name}: vs FillDistanceKNNMax (Unscaled)",fontsize=13)
    multi_plot_3metrics_vs_x(axes, df_sub, "Fill_Distance_KNN_Max","FillDistKNNMax")
    outKNNmax_unscaled=os.path.join(outdir,f"{base_name}_3plot_vs_FillDistanceKNNMax")
    fig.tight_layout(rect=[0,0,1,0.92])
    save_plot(fig, outKNNmax_unscaled, dpi=300)
    plt.close(fig)

    # (NEW) vs OffsetOverFillDistanceKNNMean (unscaled)
    fig,axes=plt.subplots(1,3,figsize=(14,4))
    fig.suptitle(f"{base_name}: vs Offset/FillDistKNNMean (Unscaled)",fontsize=13)
    multi_plot_3metrics_vs_x(axes, df_sub, "OffsetOverFillDistanceKNNMean","Offset/FillDistKNNMean")
    outKNNmean_unscaled2 = os.path.join(outdir,f"{base_name}_3plot_vs_OffsetOverFillDistanceKNNMean")
    fig.tight_layout(rect=[0,0,1,0.92])
    save_plot(fig, outKNNmean_unscaled2, dpi=300)
    plt.close(fig)

    # (NEW) vs OffsetOverFillDistanceKNNMax (unscaled)
    fig,axes=plt.subplots(1,3,figsize=(14,4))
    fig.suptitle(f"{base_name}: vs Offset/FillDistKNNMax (Unscaled)",fontsize=13)
    multi_plot_3metrics_vs_x(axes, df_sub, "OffsetOverFillDistanceKNNMax","Offset/FillDistKNNMax")
    outKNNmax_unscaled2 = os.path.join(outdir,f"{base_name}_3plot_vs_OffsetOverFillDistanceKNNMax")
    fig.tight_layout(rect=[0,0,1,0.92])
    save_plot(fig, outKNNmax_unscaled2, dpi=300)
    plt.close(fig)

    # (NEW) vs Minimax_Offset
    fig,axes=plt.subplots(1,3,figsize=(14,4))
    fig.suptitle(f"{base_name}: vs Minimax Offset",fontsize=13)
    multi_plot_3metrics_vs_x(axes, df_sub, "Minimax_Offset","Minimax Offset")
    outMinimax = os.path.join(outdir,f"{base_name}_3plot_vs_Minimax_Offset")
    fig.tight_layout(rect=[0,0,1,0.92])
    save_plot(fig, outMinimax, dpi=300)
    plt.close(fig)

    # (NEW) vs Minimax_Offset_Scaled
    fig,axes=plt.subplots(1,3,figsize=(14,4))
    fig.suptitle(f"{base_name}: vs Minimax Offset Scaled",fontsize=13)
    multi_plot_3metrics_vs_x(axes, df_sub, "Minimax_Offset_Scaled","Minimax Offset Scaled")
    outMinimaxScaled = os.path.join(outdir,f"{base_name}_3plot_vs_Minimax_Offset_Scaled")
    fig.tight_layout(rect=[0,0,1,0.92])
    save_plot(fig, outMinimaxScaled, dpi=300)
    plt.close(fig)

    # (NEW) vs MinimaxOffsetOverSigma
    fig,axes=plt.subplots(1,3,figsize=(14,4))
    fig.suptitle(f"{base_name}: vs Minimax Offset/Sigma",fontsize=13)
    multi_plot_3metrics_vs_x(axes, df_sub, "MinimaxOffsetOverSigma","Minimax Offset/Sigma")
    outMinimaxOS = os.path.join(outdir,f"{base_name}_3plot_vs_MinimaxOffsetOverSigma")
    fig.tight_layout(rect=[0,0,1,0.92])
    save_plot(fig, outMinimaxOS, dpi=300)
    plt.close(fig)

    # (NEW) vs MinimaxOffsetOverFillDistance
    fig,axes=plt.subplots(1,3,figsize=(14,4))
    fig.suptitle(f"{base_name}: vs Minimax Offset/FillDist",fontsize=13)
    multi_plot_3metrics_vs_x(axes, df_sub, "MinimaxOffsetOverFillDistance","Minimax Offset/FillDist")
    outMinimaxOFD = os.path.join(outdir,f"{base_name}_3plot_vs_MinimaxOffsetOverFillDistance")
    fig.tight_layout(rect=[0,0,1,0.92])
    save_plot(fig, outMinimaxOFD, dpi=300)
    plt.close(fig)

    # (NEW) vs MinimaxOffsetOverFillDistanceScaled
    fig,axes=plt.subplots(1,3,figsize=(14,4))
    fig.suptitle(f"{base_name}: vs Minimax Offset/FillDistScaled",fontsize=13)
    multi_plot_3metrics_vs_x(axes, df_sub, "MinimaxOffsetOverFillDistanceScaled","Minimax Offset/FillDistScaled")
    outMinimaxOFDS = os.path.join(outdir,f"{base_name}_3plot_vs_MinimaxOffsetOverFillDistanceScaled")
    fig.tight_layout(rect=[0,0,1,0.92])
    save_plot(fig, outMinimaxOFDS, dpi=300)
    plt.close(fig)
    
    # (NEW) vs Ratio_MinimaxOS_over_Fill
    minimax_ratio_dir=os.path.join(outdir,"minimax_ratio_plots")
    os.makedirs(minimax_ratio_dir,exist_ok=True)
    df_minimax = df_sub.copy()
    df_minimax["Ratio"] = df_minimax["Ratio_MinimaxOS_over_Fill"]
    fig,axes=plt.subplots(1,3,figsize=(14,4))
    fig.suptitle(f"{base_name}: vs Ratio MinimaxOS/Fill",fontsize=13)
    multi_plot_3metrics_vs_x(axes, df_minimax, "Ratio","Ratio MinimaxOS/FILL")
    outRminimax=os.path.join(minimax_ratio_dir,f"{base_name}_3plot_vs_Ratio_MinimaxOS_over_Fill")
    fig.tight_layout(rect=[0,0,1,0.92])
    save_plot(fig, outRminimax, dpi=300)
    plt.close(fig)

    # (NEW) vs MinimaxOffsetOverFillDistanceKNNMeanScaled
    fig,axes=plt.subplots(1,3,figsize=(14,4))
    fig.suptitle(f"{base_name}: vs Minimax Offset/FillDistKNNMeanScaled",fontsize=13)
    multi_plot_3metrics_vs_x(axes, df_sub, "MinimaxOffsetOverFillDistanceKNNMeanScaled","Minimax Offset/FillDistKNNMeanScaled")
    outMinimaxKNNMeanScaled = os.path.join(outdir,f"{base_name}_3plot_vs_MinimaxOffsetOverFillDistanceKNNMeanScaled")
    fig.tight_layout(rect=[0,0,1,0.92])
    save_plot(fig, outMinimaxKNNMeanScaled, dpi=300)
    plt.close(fig)

    # (NEW) vs MinimaxOffsetOverFillDistanceKNNMaxScaled
    fig,axes=plt.subplots(1,3,figsize=(14,4))
    fig.suptitle(f"{base_name}: vs Minimax Offset/FillDistKNNMaxScaled",fontsize=13)
    multi_plot_3metrics_vs_x(axes, df_sub, "MinimaxOffsetOverFillDistanceKNNMaxScaled","Minimax Offset/FillDistKNNMaxScaled")
    outMinimaxKNNMaxScaled = os.path.join(outdir,f"{base_name}_3plot_vs_MinimaxOffsetOverFillDistanceKNNMaxScaled")
    fig.tight_layout(rect=[0,0,1,0.92])
    save_plot(fig, outMinimaxKNNMaxScaled, dpi=300)
    plt.close(fig)
    
    # (NEW) vs MinimaxOffsetOverFillDistanceKNNMean
    fig,axes=plt.subplots(1,3,figsize=(14,4))
    fig.suptitle(f"{base_name}: vs Minimax Offset/FillDistKNNMean",fontsize=13)
    multi_plot_3metrics_vs_x(axes, df_sub, "MinimaxOffsetOverFillDistanceKNNMean","Minimax Offset/FillDistKNNMean")
    outMinimaxKNNMean = os.path.join(outdir,f"{base_name}_3plot_vs_MinimaxOffsetOverFillDistanceKNNMean")
    fig.tight_layout(rect=[0,0,1,0.92])
    save_plot(fig, outMinimaxKNNMean, dpi=300)
    plt.close(fig)
    
    # (NEW) vs MinimaxOffsetOverFillDistanceKNNMax
    fig,axes=plt.subplots(1,3,figsize=(14,4))
    fig.suptitle(f"{base_name}: vs Minimax Offset/FillDistKNNMax",fontsize=13)
    multi_plot_3metrics_vs_x(axes, df_sub, "MinimaxOffsetOverFillDistanceKNNMax","Minimax Offset/FillDistKNNMax")
    outMinimaxKNNMax = os.path.join(outdir,f"{base_name}_3plot_vs_MinimaxOffsetOverFillDistanceKNNMax")
    fig.tight_layout(rect=[0,0,1,0.92])
    save_plot(fig, outMinimaxKNNMax, dpi=300)
    plt.close(fig)

    # (NEW) vs MinimaxOffsetScaledOverFillDistance
    fig,axes=plt.subplots(1,3,figsize=(14,4))
    fig.suptitle(f"{base_name}: vs Minimax Offset Scaled/FillDist",fontsize=13)
    multi_plot_3metrics_vs_x(axes, df_sub, "MinimaxOffsetScaledOverFillDistance","Minimax Offset Scaled/FillDist")
    outMinimaxSFD = os.path.join(outdir,f"{base_name}_3plot_vs_MinimaxOffsetScaledOverFillDistance")
    fig.tight_layout(rect=[0,0,1,0.92])
    save_plot(fig, outMinimaxSFD, dpi=300)
    plt.close(fig)
    
    # (NEW) vs MinimaxOffsetScaledOverFillDistanceKNNMean
    fig,axes=plt.subplots(1,3,figsize=(14,4))
    fig.suptitle(f"{base_name}: vs Minimax Offset Scaled/FillDistKNNMean",fontsize=13)
    multi_plot_3metrics_vs_x(axes, df_sub, "MinimaxOffsetScaledOverFillDistanceKNNMean","Minimax Offset Scaled/FillDistKNNMean")
    outMinimaxSKNNMean = os.path.join(outdir,f"{base_name}_3plot_vs_MinimaxOffsetScaledOverFillDistanceKNNMean")
    fig.tight_layout(rect=[0,0,1,0.92])
    save_plot(fig, outMinimaxSKNNMean, dpi=300)
    plt.close(fig)
    
    # (NEW) vs MinimaxOffsetScaledOverFillDistanceKNNMax
    fig,axes=plt.subplots(1,3,figsize=(14,4))
    fig.suptitle(f"{base_name}: vs Minimax Offset Scaled/FillDistKNNMax",fontsize=13)
    multi_plot_3metrics_vs_x(axes, df_sub, "MinimaxOffsetScaledOverFillDistanceKNNMax","Minimax Offset Scaled/FillDistKNNMax")
    outMinimaxSKNNMax = os.path.join(outdir,f"{base_name}_3plot_vs_MinimaxOffsetScaledOverFillDistanceKNNMax")
    fig.tight_layout(rect=[0,0,1,0.92])
    save_plot(fig, outMinimaxSKNNMax, dpi=300)
    plt.close(fig)

    # aggregator tables
    aggregator_latex_tables(df_sub, base_name, outdir, logger, xcol="OffsetOverSigma")  
    # correlation + logistic
    cor_and_logistic_all_metrics(df_sub, base_name, outdir, logger)

    # sharp boundary analysis => do for all 4 xcols:
    do_sharp_boundary_analysis(df_sub, base_name, outdir, logger)

##############################################################################
# Missing Functions (Placeholder implementations)
##############################################################################

def aggregator_latex_tables(df_sub, base_name, outdir, logger, xcol="OffsetOverSigma"):
    """
    Placeholder for the aggregator_latex_tables function.
    """
    pass

def cor_and_logistic_all_metrics(df_sub, base_name, outdir, logger):
    """
    Placeholder for the cor_and_logistic_all_metrics function.
    """
    pass

def do_sharp_boundary_analysis(df_sub, base_name, outdir, logger):
    """
    Placeholder for the do_sharp_boundary_analysis function.
    """
    pass

def aggregate_analysis_vs_x(df_, prefix, outdir, logger, xcol="OffsetOverSigma", xcol_label="Offset/Sigma", clamp_val=None):
    """
    Placeholder for the aggregate_analysis_vs_x function.
    """
    pass

##############################################################################
# AGGREGATION FUNCTIONS
##############################################################################

def aggregate_by_dimension(df, outdir, logger):
    """
    Aggregate data by dimension categories and create plots and tables.
    
    Parameters
    ----------
    df : pandas.DataFrame
        The dataframe containing all the data
    outdir : str
        Base output directory
    logger : logging.Logger
        Logger instance
    """
    logger.info("=== Aggregating data by dimension categories ===")
    
    # Create dimension aggregation directory
    dim_outdir = os.path.join(outdir, "dimension_aggregated")
    os.makedirs(dim_outdir, exist_ok=True)
    
    # Group data by dimension category
    dimension_categories = df["DimensionCategory"].dropna().unique()
    
    for dim_cat in sorted(dimension_categories):
        df_dim = df[df["DimensionCategory"] == dim_cat].copy()
        
        if df_dim.empty:
            logger.warning(f"No data for dimension category {dim_cat}")
            continue
        
        # Create subdirectory for this dimension category
        dim_cat_dir = os.path.join(dim_outdir, dim_cat)
        os.makedirs(dim_cat_dir, exist_ok=True)
        
        logger.info(f"Processing dimension category {dim_cat} with {len(df_dim)} data points")
        
        # Run aggregated analysis on this dimension slice
        dim_prefix = f"Dim_{dim_cat}"
        
        # Generate standard plots
        process_dimensional_data(df_dim, dim_cat, dim_cat_dir, logger)
        
        # Generate aggregation tables
        generate_dim_summary_table(df_dim, dim_cat, dim_cat_dir, logger)

def aggregate_by_noise(df, outdir, logger):
    """
    Aggregate data by noise status (noisy vs. not noisy).
    
    Parameters
    ----------
    df : pandas.DataFrame
        The dataframe containing all the data
    outdir : str
        Base output directory
    logger : logging.Logger
        Logger instance
    """
    logger.info("=== Aggregating data by noise status ===")
    
    # Create noise aggregation directory
    noise_outdir = os.path.join(outdir, "noise_aggregated")
    os.makedirs(noise_outdir, exist_ok=True)
    
    # Process noisy data
    df_noisy = df[df["IsNoisy"] == True].copy()
    if not df_noisy.empty:
        noisy_dir = os.path.join(noise_outdir, "noisy")
        os.makedirs(noisy_dir, exist_ok=True)
        logger.info(f"Processing noisy data with {len(df_noisy)} data points")
        process_noise_bias_data(df_noisy, "Noisy", noisy_dir, logger)
        generate_noise_bias_summary_table(df_noisy, "Noisy", noisy_dir, logger)
    
    # Process clean data
    df_clean = df[df["IsNoisy"] == False].copy()
    if not df_clean.empty:
        clean_dir = os.path.join(noise_outdir, "clean")
        os.makedirs(clean_dir, exist_ok=True)
        logger.info(f"Processing clean data with {len(df_clean)} data points")
        process_noise_bias_data(df_clean, "Clean", clean_dir, logger)
        generate_noise_bias_summary_table(df_clean, "Clean", clean_dir, logger)
    
    # Generate comparison tables between noisy and clean
    if not df_noisy.empty and not df_clean.empty:
        generate_noise_comparison_table(df_noisy, df_clean, noise_outdir, logger)

def aggregate_by_bias(df, outdir, logger):
    """
    Aggregate data by bias status (biased vs. unbiased).
    
    Parameters
    ----------
    df : pandas.DataFrame
        The dataframe containing all the data
    outdir : str
        Base output directory
    logger : logging.Logger
        Logger instance
    """
    logger.info("=== Aggregating data by bias status ===")
    
    # Create bias aggregation directory
    bias_outdir = os.path.join(outdir, "bias_aggregated")
    os.makedirs(bias_outdir, exist_ok=True)
    
    # Process biased data
    df_biased = df[df["IsBiased"] == True].copy()
    if not df_biased.empty:
        biased_dir = os.path.join(bias_outdir, "biased")
        os.makedirs(biased_dir, exist_ok=True)
        logger.info(f"Processing biased data with {len(df_biased)} data points")
        process_noise_bias_data(df_biased, "Biased", biased_dir, logger)
        generate_noise_bias_summary_table(df_biased, "Biased", biased_dir, logger)
    
    # Process unbiased data
    df_unbiased = df[df["IsBiased"] == False].copy()
    if not df_unbiased.empty:
        unbiased_dir = os.path.join(bias_outdir, "unbiased")
        os.makedirs(unbiased_dir, exist_ok=True)
        logger.info(f"Processing unbiased data with {len(df_unbiased)} data points")
        process_noise_bias_data(df_unbiased, "Unbiased", unbiased_dir, logger)
        generate_noise_bias_summary_table(df_unbiased, "Unbiased", unbiased_dir, logger)
    
    # Generate comparison tables between biased and unbiased
    if not df_biased.empty and not df_unbiased.empty:
        generate_bias_comparison_table(df_biased, df_unbiased, bias_outdir, logger)

def aggregate_by_combined_factors(df, outdir, logger):
    """
    Aggregate data by combined dimension, noise, and bias factors.
    
    Parameters
    ----------
    df : pandas.DataFrame
        The dataframe containing all the data
    outdir : str
        Base output directory
    logger : logging.Logger
        Logger instance
    """
    logger.info("=== Aggregating data by combined factors ===")
    
    # Create combined aggregation directory
    combined_outdir = os.path.join(outdir, "combined_aggregated")
    os.makedirs(combined_outdir, exist_ok=True)
    
    # Group by dimension category
    dimension_categories = df["DimensionCategory"].dropna().unique()
    
    for dim_cat in sorted(dimension_categories):
        df_dim = df[df["DimensionCategory"] == dim_cat].copy()
        
        if df_dim.empty:
            continue
            
        # Now split by noise and bias
        for is_noisy in [True, False]:
            for is_biased in [True, False]:
                df_subset = df_dim[(df_dim["IsNoisy"] == is_noisy) & (df_dim["IsBiased"] == is_biased)].copy()
                
                if df_subset.empty:
                    continue
                
                # Create descriptive directory name
                noise_str = "noisy" if is_noisy else "clean"
                bias_str = "biased" if is_biased else "unbiased"
                subset_dir = os.path.join(combined_outdir, f"{dim_cat}_{noise_str}_{bias_str}")
                os.makedirs(subset_dir, exist_ok=True)
                
                subset_prefix = f"{dim_cat}_{noise_str}_{bias_str}"
                logger.info(f"Processing {subset_prefix} with {len(df_subset)} data points")
                
                # Generate plots for this combination
                process_combined_data(df_subset, subset_prefix, subset_dir, logger)
                
                # Generate summary table
                generate_combined_summary_table(df_subset, subset_prefix, subset_dir, logger)

def process_dimensional_data(df, dim_cat, outdir, logger):
    """
    Process data for a specific dimension category, generating plots.
    
    Parameters
    ----------
    df : pandas.DataFrame
        The dataframe containing the data for this dimension category
    dim_cat : str
        The dimension category name
    outdir : str
        Output directory for this dimension category
    logger : logging.Logger
        Logger instance
    """
    # Format the dimension category for display
    formatted_dim_cat = format_dimension_category(dim_cat)
    
    # Define all the metrics we want to plot against
    key_metrics = [
        # Core metrics
        ("OffsetOverSigma", "Offset/Sigma"),
        ("Offset", "Offset"),
        ("Fill_Distance", "Fill Distance"),
        ("Fill_Distance_Scaled", "Scaled Fill Distance"),
        ("OffsetOverFillDistance", "Offset/Fill Distance"),
        ("OffsetOverFillDistanceScaled", "Offset/Fill Distance Scaled"),
        ("Sample_Percentage", "Sample Percentage"),
        ("Ratio_OS_over_Fill", "Ratio OS/Fill"),
        
        # KNN variants
        ("Fill_Distance_KNN_Mean", "Fill Distance KNN Mean"),
        ("Fill_Distance_KNN_Max", "Fill Distance KNN Max"),
        ("Fill_Distance_KNN_Mean_Scaled", "Fill Distance KNN Mean Scaled"),
        ("Fill_Distance_KNN_Max_Scaled", "Fill Distance KNN Max Scaled"),
        ("OffsetOverFillDistanceKNNMean", "Offset/Fill Distance KNN Mean"),
        ("OffsetOverFillDistanceKNNMax", "Offset/Fill Distance KNN Max"),
        ("OffsetOverFillDistanceKNNMeanScaled", "Offset/Fill Distance KNN Mean Scaled"),
        ("OffsetOverFillDistanceKNNMaxScaled", "Offset/Fill Distance KNN Max Scaled"),
        
        # Minimax variants
        ("Minimax_Offset", "Minimax Offset"),
        ("Minimax_Offset_Scaled", "Minimax Offset Scaled"),
        ("MinimaxOffsetOverSigma", "Minimax Offset/Sigma"),
        ("MinimaxOffsetOverFillDistance", "Minimax Offset/Fill Distance"),
        ("MinimaxOffsetOverFillDistanceScaled", "Minimax Offset/Fill Distance Scaled"),
        ("Ratio_MinimaxOS_over_Fill", "Ratio MinimaxOS/Fill"),
        ("MinimaxOffsetOverFillDistanceKNNMean", "Minimax Offset/Fill Distance KNN Mean"),
        ("MinimaxOffsetOverFillDistanceKNNMax", "Minimax Offset/Fill Distance KNN Max"),
        ("MinimaxOffsetOverFillDistanceKNNMeanScaled", "Minimax Offset/Fill Distance KNN Mean Scaled"),
        ("MinimaxOffsetOverFillDistanceKNNMaxScaled", "Minimax Offset/Fill Distance KNN Max Scaled"),
        ("MinimaxOffsetScaledOverFillDistance", "Minimax Offset Scaled/Fill Distance"),
        ("MinimaxOffsetScaledOverFillDistanceKNNMean", "Minimax Offset Scaled/Fill Distance KNN Mean"),
        ("MinimaxOffsetScaledOverFillDistanceKNNMax", "Minimax Offset Scaled/Fill Distance KNN Max")
    ]
    
    # Process each metric
    for col_name, col_label in key_metrics:
        # Skip if column doesn't exist or has no valid data
        if col_name not in df.columns or df[col_name].dropna().empty:
            logger.info(f"Skipping {col_name} for {dim_cat} - no data available")
            continue
            
        # Get display label for the metric from our dictionary
        display_label = METRIC_LABELS.get(col_name, col_label)
        
        # Create the plot
        fig, axes = plt.subplots(1, 3, figsize=(14, 4))
        fig.suptitle(f"{formatted_dim_cat} - vs. {display_label}", fontsize=13)
        multi_plot_3metrics_vs_x(axes, df, col_name, display_label)
        out_path = os.path.join(outdir, f"{dim_cat}_3plot_vs_{col_name}")
        fig.tight_layout(rect=[0, 0, 1, 0.92])
        save_plot(fig, out_path, dpi=300)
        plt.close(fig)

def process_noise_bias_data(df, category, outdir, logger):
    """
    Process data for noise or bias categories, generating plots.
    
    Parameters
    ----------
    df : pandas.DataFrame
        The dataframe containing the data for this category
    category : str
        The category name (e.g., "Noisy", "Clean", "Biased", "Unbiased")
    outdir : str
        Output directory for this category
    logger : logging.Logger
        Logger instance
    """
    # Define all the metrics we want to plot against
    key_metrics = [
        # Core metrics
        ("OffsetOverSigma", "Offset/Sigma"),
        ("Offset", "Offset"),
        ("Fill_Distance", "Fill Distance"),
        ("Fill_Distance_Scaled", "Scaled Fill Distance"),
        ("OffsetOverFillDistance", "Offset/Fill Distance"),
        ("OffsetOverFillDistanceScaled", "Offset/Fill Distance Scaled"),
        ("Sample_Percentage", "Sample Percentage"),
        ("Ratio_OS_over_Fill", "Ratio OS/Fill"),
        
        # KNN variants
        ("Fill_Distance_KNN_Mean", "Fill Distance KNN Mean"),
        ("Fill_Distance_KNN_Max", "Fill Distance KNN Max"),
        ("Fill_Distance_KNN_Mean_Scaled", "Fill Distance KNN Mean Scaled"),
        ("Fill_Distance_KNN_Max_Scaled", "Fill Distance KNN Max Scaled"),
        ("OffsetOverFillDistanceKNNMean", "Offset/Fill Distance KNN Mean"),
        ("OffsetOverFillDistanceKNNMax", "Offset/Fill Distance KNN Max"),
        ("OffsetOverFillDistanceKNNMeanScaled", "Offset/Fill Distance KNN Mean Scaled"),
        ("OffsetOverFillDistanceKNNMaxScaled", "Offset/Fill Distance KNN Max Scaled"),
        
        # Minimax variants
        ("Minimax_Offset", "Minimax Offset"),
        ("Minimax_Offset_Scaled", "Minimax Offset Scaled"),
        ("MinimaxOffsetOverSigma", "Minimax Offset/Sigma"),
        ("MinimaxOffsetOverFillDistance", "Minimax Offset/Fill Distance"),
        ("MinimaxOffsetOverFillDistanceScaled", "Minimax Offset/Fill Distance Scaled"),
        ("Ratio_MinimaxOS_over_Fill", "Ratio MinimaxOS/Fill"),
        ("MinimaxOffsetOverFillDistanceKNNMean", "Minimax Offset/Fill Distance KNN Mean"),
        ("MinimaxOffsetOverFillDistanceKNNMax", "Minimax Offset/Fill Distance KNN Max"),
        ("MinimaxOffsetOverFillDistanceKNNMeanScaled", "Minimax Offset/Fill Distance KNN Mean Scaled"),
        ("MinimaxOffsetOverFillDistanceKNNMaxScaled", "Minimax Offset/Fill Distance KNN Max Scaled"),
        ("MinimaxOffsetScaledOverFillDistance", "Minimax Offset Scaled/Fill Distance"),
        ("MinimaxOffsetScaledOverFillDistanceKNNMean", "Minimax Offset Scaled/Fill Distance KNN Mean"),
        ("MinimaxOffsetScaledOverFillDistanceKNNMax", "Minimax Offset Scaled/Fill Distance KNN Max")
    ]
    
    # Process each metric
    for col_name, col_label in key_metrics:
        # Skip if column doesn't exist or has no valid data
        if col_name not in df.columns or df[col_name].dropna().empty:
            logger.info(f"Skipping {col_name} for {category} - no data available")
            continue
            
        # Get display label for the metric from our dictionary
        display_label = METRIC_LABELS.get(col_name, col_label)
        
        # Create the plot
        fig, axes = plt.subplots(1, 3, figsize=(14, 4))
        fig.suptitle(f"{category} Data - vs. {display_label}", fontsize=13)
        multi_plot_3metrics_vs_x(axes, df, col_name, display_label)
        out_path = os.path.join(outdir, f"{category}_3plot_vs_{col_name}")
        fig.tight_layout(rect=[0, 0, 1, 0.92])
        save_plot(fig, out_path, dpi=300)
        plt.close(fig)

def process_combined_data(df, prefix, outdir, logger):
    """
    Process data for combined dimension, noise, and bias categories.
    
    Parameters
    ----------
    df : pandas.DataFrame
        The dataframe containing the data for this combined category
    prefix : str
        Prefix for filenames
    outdir : str
        Output directory for this combined category
    logger : logging.Logger
        Logger instance
    """
    # Format the prefix for display (extract the dimension category part if present)
    formatted_prefix = prefix
    if "_dim_" in prefix.lower():
        parts = prefix.split("_", 1)
        dim_part = parts[0]
        remaining = parts[1] if len(parts) > 1 else ""
        formatted_dim_cat = format_dimension_category(dim_part)
        formatted_prefix = f"{formatted_dim_cat} {remaining}"
    
    # Define all the metrics we want to plot against
    key_metrics = [
        # Core metrics
        ("OffsetOverSigma", "Offset/Sigma"),
        ("Offset", "Offset"),
        ("Fill_Distance", "Fill Distance"),
        ("Fill_Distance_Scaled", "Scaled Fill Distance"),
        ("OffsetOverFillDistance", "Offset/Fill Distance"),
        ("OffsetOverFillDistanceScaled", "Offset/Fill Distance Scaled"),
        ("Sample_Percentage", "Sample Percentage"),
        ("Ratio_OS_over_Fill", "Ratio OS/Fill"),
        
        # KNN variants
        ("Fill_Distance_KNN_Mean", "Fill Distance KNN Mean"),
        ("Fill_Distance_KNN_Max", "Fill Distance KNN Max"),
        ("Fill_Distance_KNN_Mean_Scaled", "Fill Distance KNN Mean Scaled"),
        ("Fill_Distance_KNN_Max_Scaled", "Fill Distance KNN Max Scaled"),
        ("OffsetOverFillDistanceKNNMean", "Offset/Fill Distance KNN Mean"),
        ("OffsetOverFillDistanceKNNMax", "Offset/Fill Distance KNN Max"),
        ("OffsetOverFillDistanceKNNMeanScaled", "Offset/Fill Distance KNN Mean Scaled"),
        ("OffsetOverFillDistanceKNNMaxScaled", "Offset/Fill Distance KNN Max Scaled"),
        
        # Minimax variants
        ("Minimax_Offset", "Minimax Offset"),
        ("Minimax_Offset_Scaled", "Minimax Offset Scaled"),
        ("MinimaxOffsetOverSigma", "Minimax Offset/Sigma"),
        ("MinimaxOffsetOverFillDistance", "Minimax Offset/Fill Distance"),
        ("MinimaxOffsetOverFillDistanceScaled", "Minimax Offset/Fill Distance Scaled"),
        ("Ratio_MinimaxOS_over_Fill", "Ratio MinimaxOS/Fill"),
        ("MinimaxOffsetOverFillDistanceKNNMean", "Minimax Offset/Fill Distance KNN Mean"),
        ("MinimaxOffsetOverFillDistanceKNNMax", "Minimax Offset/Fill Distance KNN Max"),
        ("MinimaxOffsetOverFillDistanceKNNMeanScaled", "Minimax Offset/Fill Distance KNN Mean Scaled"),
        ("MinimaxOffsetOverFillDistanceKNNMaxScaled", "Minimax Offset/Fill Distance KNN Max Scaled"),
        ("MinimaxOffsetScaledOverFillDistance", "Minimax Offset Scaled/Fill Distance"),
        ("MinimaxOffsetScaledOverFillDistanceKNNMean", "Minimax Offset Scaled/Fill Distance KNN Mean"),
        ("MinimaxOffsetScaledOverFillDistanceKNNMax", "Minimax Offset Scaled/Fill Distance KNN Max")
    ]
    
    # Process each metric
    for col_name, col_label in key_metrics:
        # Skip if column doesn't exist or has no valid data
        if col_name not in df.columns or df[col_name].dropna().empty:
            logger.info(f"Skipping {col_name} for {prefix} - no data available")
            continue
            
        # Get display label for the metric from our dictionary
        display_label = METRIC_LABELS.get(col_name, col_label)
        
        # Create the plot
        fig, axes = plt.subplots(1, 3, figsize=(14, 4))
        fig.suptitle(f"{formatted_prefix} - vs. {display_label}", fontsize=13)
        multi_plot_3metrics_vs_x(axes, df, col_name, display_label)
        out_path = os.path.join(outdir, f"{prefix}_3plot_vs_{col_name}")
        fig.tight_layout(rect=[0, 0, 1, 0.92])
        save_plot(fig, out_path, dpi=300)
        plt.close(fig)

##############################################################################
# MAIN
##############################################################################

def main():
    args = parse_args()
    logger = setup_logging(args.output_dir)
    logger.info("=== Starting plotting script ===")

    try:
        df=pd.read_csv(args.input_csv)
        logger.info(f"Read {len(df)} rows from {args.input_csv}")
    except Exception as e:
        logger.error(f"CSV read fail => {e}")
        sys.exit(1)

    validate_input_data(df, logger)
    df=do_preprocessing(df, logger)
    if df.empty:
        logger.warning("No data => done.")
        return

    # single dataset analysis
    base_list=sorted(df["Base_Dataset"].dropna().unique())
    logger.info(f"Found {len(base_list)} base datasets: {base_list}")

    def process_dataset(base_ds):
        sub_=df[df["Base_Dataset"]==base_ds]
        if sub_.empty:
            return
        outd=os.path.join(args.output_dir, base_ds)
        os.makedirs(outd,exist_ok=True)
        single_base_dataset_analysis(sub_, base_ds, outd, logger)

    Parallel(n_jobs=args.n_jobs)(
        delayed(process_dataset)(bd) for bd in base_list
    )

    # aggregator for each xcol. We'll replicate the approach
    # (OffsetOverSigma, Ratio_OS_over_Fill) => existing
    # (Offset, OffsetOverFillDistance, OffsetOverFillDistanceScaled) => new
    def do_agg_for_xcol(df_, prefix, xcol, xlbl):
        aggregate_analysis_vs_x(df_, prefix, args.output_dir, logger,
                                xcol=xcol, xcol_label=xlbl, clamp_val=None)
        mx_ = df_[xcol].dropna().max()
        if mx_>10:
            aggregate_analysis_vs_x(df_, prefix, args.output_dir, logger,
                                    xcol=xcol, xcol_label=xlbl, clamp_val=50.0)

    do_agg_for_xcol(df, "AllAggregated_offsetSigma","OffsetOverSigma","Offset/Sigma")
    do_agg_for_xcol(df, "AllAggregated_ratio","Ratio_OS_over_Fill","Ratio OS/FILL")
    do_agg_for_xcol(df, "AllAggregated_offset","Offset","Offset")
    do_agg_for_xcol(df, "AllAggregated_offsetOverFill","OffsetOverFillDistance","Offset/FillDist")
    do_agg_for_xcol(df, "AllAggregated_offsetOverFillScaled","OffsetOverFillDistanceScaled","Offset/FillDistScaled")
    do_agg_for_xcol(df, "AllAggregated_fillKNNMeanScaled","Fill_Distance_KNN_Mean_Scaled","FillDistKNNMeanScaled")
    do_agg_for_xcol(df, "AllAggregated_fillKNNMaxScaled","Fill_Distance_KNN_Max_Scaled","FillDistKNNMaxScaled")
    do_agg_for_xcol(df, "AllAggregated_offsetOverFillKNNMeanScaled","OffsetOverFillDistanceKNNMeanScaled","Offset/FillDistKNNMeanScaled")
    do_agg_for_xcol(df, "AllAggregated_offsetOverFillKNNMaxScaled","OffsetOverFillDistanceKNNMaxScaled","Offset/FillDistKNNMaxScaled")
    
    # New minimax offset aggregations
    do_agg_for_xcol(df, "AllAggregated_minimaxOffset","Minimax_Offset","Minimax Offset")
    do_agg_for_xcol(df, "AllAggregated_minimaxOffsetScaled","Minimax_Offset_Scaled","Minimax Offset Scaled")
    do_agg_for_xcol(df, "AllAggregated_minimaxOffsetSigma","MinimaxOffsetOverSigma","Minimax Offset/Sigma")
    do_agg_for_xcol(df, "AllAggregated_minimaxRatio","Ratio_MinimaxOS_over_Fill","Ratio MinimaxOS/FILL") 
    do_agg_for_xcol(df, "AllAggregated_minimaxOffsetOverFill","MinimaxOffsetOverFillDistance","Minimax Offset/FillDist")
    do_agg_for_xcol(df, "AllAggregated_minimaxOffsetOverFillScaled","MinimaxOffsetOverFillDistanceScaled","Minimax Offset/FillDistScaled")
    do_agg_for_xcol(df, "AllAggregated_minimaxOffsetOverFillKNNMean","MinimaxOffsetOverFillDistanceKNNMean","Minimax Offset/FillDistKNNMean")
    do_agg_for_xcol(df, "AllAggregated_minimaxOffsetOverFillKNNMax","MinimaxOffsetOverFillDistanceKNNMax","Minimax Offset/FillDistKNNMax")
    do_agg_for_xcol(df, "AllAggregated_minimaxOffsetOverFillKNNMeanScaled","MinimaxOffsetOverFillDistanceKNNMeanScaled","Minimax Offset/FillDistKNNMeanScaled")
    do_agg_for_xcol(df, "AllAggregated_minimaxOffsetOverFillKNNMaxScaled","MinimaxOffsetOverFillDistanceKNNMaxScaled","Minimax Offset/FillDistKNNMaxScaled")

    # Aggregation by dimension, noise, and bias
    logger.info("=== Starting new aggregation analyses ===")
    
    # Aggregate by dimension
    aggregate_by_dimension(df, args.output_dir, logger)
    
    # Aggregate by noise status
    aggregate_by_noise(df, args.output_dir, logger)
    
    # Aggregate by bias status
    aggregate_by_bias(df, args.output_dir, logger)
    
    # Aggregate by combined factors (dimension, noise, bias)
    aggregate_by_combined_factors(df, args.output_dir, logger)

    logger.info("=== Done. ===")

if __name__=="__main__":
    main()
