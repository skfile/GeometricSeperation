#!/usr/bin/env python3
"""
general_plotter.py
-----------------------------
Generates a comprehensive set of visualizations for each dataset in your config, covering:
  1) Raw high-dimensional data (projected to 2D via PCA) as a matrix of subplots over (offset, fraction).
  2) For each embedding method in config, a matrix of subplots (offset x fraction) of the embedded data in 2D.
  3) A 3D plot of the ground-truth adjacency A_full (either in the original dimension if dim<=3, 
     or using PCA(3) otherwise) with edges rendered.
  4) A matrix of subplots showing each kernel adjacency, for each (offset, fraction).

We place the offset on the rows, fraction on the columns.

Features:
  - Publication-quality plots suitable for NeurIPS/ICML submission
  - SVG output of individual subplots for fine-grained control
  - Support for MNIST and scRNA data
  - Customizable color palettes and plot parameters
  - Enhanced handling of noisy/biased data

Usage:
  python general_plotter.py --config my_config.json [--output_dir my_plots/] [--plot_style neurips]
                           [--color_palette colorblind] [--svg_subplots]

Dependencies:
  - Uses your existing code from `mesh_sampling.py` for data generation.
  - Also uses `kernel_dispatcher` from `src.kernels`.
  - Built with HPC non-interactive plotting in mind (matplotlib 'Agg' backend).
"""

import os
import sys
import time
import json
import argparse
import logging
from pathlib import Path

# Ensure project root is on PYTHONPATH for local src imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from mpl_toolkits.mplot3d import Axes3D  # needed for 3D plots
from scipy.sparse.csgraph import shortest_path
from scipy.spatial.distance import pdist
import warnings

from src.mesh_sampling import generate_dataset_and_subsamples
from src.kernels import kernel_dispatcher
from src.utils import connected_comp_helper
from src.embedding_algorithms import compute_embedding

##############################################################################
# CONFIGURATION CONSTANTS
##############################################################################

# Plot styles available
PLOT_STYLES = {
    "neurips": {
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
    },
    "standard": {
        # A more colorful and bolder style
        "figure.dpi": 300,
        "font.size": 10,
        "font.family": "sans-serif",
        
        "axes.linewidth": 0.8,
        "axes.titlesize": 12,
        "axes.labelsize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "lines.linewidth": 1.5,
        "lines.markersize": 6,
        
        "legend.fontsize": 9,
        "legend.frameon": True,
        
        "grid.linewidth": 0.5,
        "grid.alpha": 0.3,
    },
    "minimal": {
        # Very clean, minimalist style
        "figure.dpi": 300,
        "font.size": 8,
        "font.family": "sans-serif",
        
        "axes.linewidth": 0.5,
        "axes.titlesize": 9,
        "axes.labelsize": 8,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "lines.linewidth": 1.0,
        "lines.markersize": 4,
        
        "legend.fontsize": 7,
        "legend.frameon": False,
        
        "grid.linewidth": 0.0,  # No grid
        "grid.alpha": 0.0,
        
        "axes.grid": False,
        "axes.spines.top": False,
        "axes.spines.right": False,
    },
}

# Color palette options
COLOR_PALETTES = {
    "colorblind": sns.color_palette("colorblind", 8),
    "neurips": [
        "#377eb8",  # Blue
        "#ff7f00",  # Orange
        "#4daf4a",  # Green
        "#f781bf",  # Pink
        "#a65628",  # Brown
        "#984ea3",  # Purple
        "#999999",  # Gray
        "#e41a1c",  # Red
    ],
    "bright": sns.color_palette("bright", 8),
    "pastel": sns.color_palette("pastel", 8),
    "dark": sns.color_palette("dark", 8),
    "deep": sns.color_palette("deep", 8),
    "muted": sns.color_palette("muted", 8),
    "paired": sns.color_palette("Paired", 8),
    "set1": sns.color_palette("Set1", 8),
    "tab10": sns.color_palette("tab10", 8),
    "viridis": sns.color_palette("viridis", 8),
    "plasma": sns.color_palette("plasma", 8),
}

# Noise detection keywords - datasets with these terms are considered "noisy"
NOISE_KEYWORDS = ["noisy", "noise", "perturb"]

# Bias detection keywords - datasets with these terms are considered "biased"
BIAS_KEYWORDS = ["bias", "biased"]

##############################################################################
# LOGGING SETUP
##############################################################################

def setup_logging(log_dir=None, level=logging.INFO):
    """
    Configure logging to both console and file (if log_dir is provided)
    """
    logger = logging.getLogger("general_plotter")
    logger.setLevel(level)
    
    # Clear any existing handlers
    if logger.handlers:
        logger.handlers.clear()
        
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
    logger.addHandler(console_handler)
    
    # File handler (if log_dir is specified)
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        log_file = os.path.join(log_dir, f"general_plotter_{timestamp}.log")
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
        logger.addHandler(file_handler)
        
    return logger


##############################################################################
# DATA UTILITY FUNCTIONS
##############################################################################

def is_noisy_dataset(dataset_name):
    """Check if dataset is noisy based on keywords in name."""
    if not dataset_name:
        return False
    name_lower = dataset_name.lower()
    return any(keyword in name_lower for keyword in NOISE_KEYWORDS)

def is_biased_dataset(dataset_name):
    """Check if dataset is biased based on keywords in name."""
    if not dataset_name:
        return False
    name_lower = dataset_name.lower()
    return any(keyword in name_lower for keyword in BIAS_KEYWORDS)

def is_mnist_dataset(dataset_config):
    """Check if dataset is MNIST based on config type."""
    return dataset_config.get("type", "").lower() == "mnist"

def is_scrna_dataset(dataset_config):
    """Check if dataset is scRNA based on config type."""
    return dataset_config.get("type", "").lower() in ["scrna", "singlecell", "scrnaseq"]

def pca_2d(X):
    """Project X (NxD) to 2D via PCA."""
    if X.shape[1] <= 2:
        return X
    pca = PCA(n_components=2)
    return pca.fit_transform(X)

def pca_3d(X):
    """
    Project NxD data X to Nx3 using PCA if dimension>3,
    or just pad zeros if dimension<3.
    """
    d = X.shape[1]
    if d == 3:
        return X
    elif d < 3:
        # pad zeros
        pad_cols = 3 - d
        padding = np.zeros((X.shape[0], pad_cols), dtype=X.dtype)
        return np.hstack((X, padding))
    else:
        pca = PCA(n_components=3)
        return pca.fit_transform(X)

def build_kernel_adjacency(X, kname, kparams):
    """
    Build adjacency from X using kernel_dispatcher.
    Then interpret any positive entry as an edge.
    """
    kfunc = kernel_dispatcher(kname, **kparams)
    mat = kfunc(X)   # NxN
    if mat is None:
        return None
    mat = connected_comp_helper(mat, X, connect=True)
    # We'll only interpret mat>0.0 as an edge
    return mat

def embed_for_method(X, A, method):
    """
    Return Nx2 embedding for the given method.
    If method in ['diffusionmap','umap'], interpret A as adjacency.
    Else interpret X as NxD.
    """
    m = method.lower()
    if m in ['diffusionmap','umap']:
        if A is None:
            return None
        emb, _ = compute_embedding(A, method=m, n_components=2)
    else:
        emb, _ = compute_embedding(X, method=m, n_components=2)
    return emb

##############################################################################
# PLOTTING UTILITY FUNCTIONS
##############################################################################

def plot_matrix_rows_cols(row_values, col_values, figsize=(5,4), sharex=True, sharey=True):
    """
    Helper to create a (len(row_values) x len(col_values)) grid of subplots.
    Returns: fig, axes
    """
    nrows, ncols = len(row_values), len(col_values)
    fig, axes = plt.subplots(
        nrows=nrows, ncols=ncols,
        figsize=(figsize[0]*ncols, figsize[1]*nrows),
        sharex=sharex, sharey=sharey,
        squeeze=False
    )
    return fig, axes

def scatter_2d(ax, X2d, color='gray', label=None, alpha=0.7, size=8, edgecolor=None):
    """Create a scatter plot with the given parameters."""
    if edgecolor is None:
        ax.scatter(X2d[:,0], X2d[:,1], s=size, color=color, alpha=alpha, label=label)
    else:
        ax.scatter(X2d[:,0], X2d[:,1], s=size, color=color, alpha=alpha, label=label, edgecolor=edgecolor)

def annotate_top_left(ax, text_str, fontsize=8):
    """
    Place a small annotation "N=xxx" or similar in the top-left corner.
    If `ax` is a 3D Axes, use `text2D`; if 2D, use `text`.
    """
    if isinstance(ax, Axes3D):
        # For 3D axes, we have to place text in 2D coordinates with ax.text2D
        ax.text2D(
            0.02, 0.98,
            s=text_str,
            transform=ax.transAxes,
            ha='left', va='top',
            fontsize=fontsize,
            bbox=dict(boxstyle='round,pad=0.3', fc='white', ec='gray', alpha=0.7)
        )
    else:
        ax.text(
            0.02, 0.98,
            text_str,
            transform=ax.transAxes,
            ha='left', va='top',
            fontsize=fontsize,
            bbox=dict(boxstyle='round,pad=0.3', fc='white', ec='gray', alpha=0.7)
        )

def save_figure(fig, output_path, dpi=300, formats=None):
    """
    Save figure in multiple formats
    
    Parameters:
    - fig: matplotlib figure
    - output_path: path without extension
    - formats: list of formats to save in, e.g. ['png', 'svg', 'pdf']
    """
    if formats is None:
        formats = ['png']
    
    # Make sure the directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Save in each requested format
    for fmt in formats:
        full_path = f"{output_path}.{fmt}"
        fig.savefig(full_path, dpi=dpi, bbox_inches='tight')
        
def save_subplot_as_svg(fig, ax, i, j, output_dir, base_name, suffix="", dpi=300):
    """
    Extract a single subplot and save it as SVG
    
    Parameters:
    - fig: original figure containing subplots
    - ax: the specific subplot axes
    - i, j: subplot coordinates in the grid
    - output_dir: directory to save SVGs
    - base_name: base filename
    - suffix: additional identifier for filename
    """
    import logging
    logger = logging.getLogger("general_plotter")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    filename = f"{base_name}_subplot_{i}_{j}{suffix}.svg"
    filepath = os.path.join(output_dir, filename)
    
    try:
        # Create a new figure with the same size as the subplot
        bbox = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
        width, height = bbox.width, bbox.height
        fig_single = plt.figure(figsize=(width, height))
    
    # Copy content from the subplot to new figure
    ax_single = fig_single.add_subplot(111)
    
    # Copy main content by regenerating the plot
    for line in ax.get_lines():
        ax_single.plot(line.get_xdata(), line.get_ydata(), 
                      color=line.get_color(), linestyle=line.get_linestyle(),
                      linewidth=line.get_linewidth(), marker=line.get_marker(),
                      markersize=line.get_markersize(), alpha=line.get_alpha(),
                      label=line.get_label())
    
    # Copy scatter plots
    for collection in ax.collections:
        if hasattr(collection, 'get_offsets'):  # For scatter plots
            try:
                offsets = collection.get_offsets()
                if len(offsets) > 0:
                    # Handle facecolor properly
                    try:
                        c = collection.get_facecolor()
                        # Check if it's empty or has the right format
                        if not hasattr(c, 'size') or c.size == 0:
                            c = 'b'
                        # If it's a 2D array with multiple colors (one per point)
                        elif len(c.shape) > 1 and c.shape[0] > 1:
                            # Keep the colors as they are
                            pass
                        # If it's a single RGBA color
                        else:
                            # Use the first color (for consistency)
                            c = c[0] if len(c) > 0 else 'b'
                    except:
                        c = 'b'
                    
                    # Handle sizes properly
                    try:
                        s = collection.get_sizes()
                        if not hasattr(s, 'size') or s.size == 0:
                            s = 20
                        # If multiple sizes, keep them as is
                        elif s.size > 1 and s.size == offsets.shape[0]:
                            pass
                        # If single size
                        else:
                            s = s[0] if len(s) > 0 else 20
                    except:
                        s = 20
                    
                    # Handle alpha properly
                    alpha = collection.get_alpha() if collection.get_alpha() is not None else 1.0
                    
                    # Handle labels for legend
                    label = collection.get_label() if hasattr(collection, 'get_label') else None
                    if label == '_nolegend_':
                        label = None
                    
                    # Handle edgecolors if present
                    try:
                        ec = collection.get_edgecolors()
                        edgecolor = 'none' if ec.size == 0 else ec
                    except:
                        edgecolor = 'none'
                        
                    # Use 'color' parameter instead of 'c' to avoid deprecation warnings
                    ax_single.scatter(offsets[:, 0], offsets[:, 1], 
                                     color=c, s=s, alpha=alpha, label=label, 
                                     edgecolor=edgecolor)
            except Exception as e:
                # Log the error but continue with other elements
                import logging
                logging.warning(f"Failed to copy scatter plot: {str(e)}")
    
    # Copy properties
    ax_single.set_title(ax.get_title())
    ax_single.set_xlabel(ax.get_xlabel())
    ax_single.set_ylabel(ax.get_ylabel())
    ax_single.set_xlim(ax.get_xlim())
    ax_single.set_ylim(ax.get_ylim())

    # Copy legend if present and if there are labeled artists
    if ax.get_legend():
        handles, labels = ax.get_legend_handles_labels()
        if handles and labels:  # Only create legend if we have labeled items
            ax_single.legend(handles, labels)
    
    # Copy text annotations
    for text in ax.texts:
        bbox_props = None
        if hasattr(text, '_bbox_patch') and text._bbox_patch is not None:
            # Create a new bbox properties dictionary instead of copying the boxstyle object directly
            try:
                # Extract style properties safely
                boxstyle = text._bbox_patch.get_boxstyle().name if hasattr(text._bbox_patch.get_boxstyle(), 'name') else 'round'
                
                bbox_props = dict(
                    boxstyle=boxstyle,
                    fc=text._bbox_patch.get_facecolor(),
                    ec=text._bbox_patch.get_edgecolor(),
                    alpha=text._bbox_patch.get_alpha() if hasattr(text._bbox_patch, 'get_alpha') else 1.0,
                    pad=0.3  # Default padding if not specified
                )
            except Exception as e:
                # Fallback to a simple bounding box if extraction fails
                import logging
                logging.warning(f"Failed to copy text bbox properties: {str(e)}")
                bbox_props = dict(boxstyle='round', fc='white', ec='gray', alpha=0.7)
        
        ax_single.text(text.get_position()[0], text.get_position()[1], text.get_text(),
                      transform=ax_single.transAxes if text.get_transform() == ax.transAxes else None,
                      ha=text.get_ha(), va=text.get_va(), fontsize=text.get_fontsize(),
                      bbox=bbox_props)
    
        # Save the figure
        fig_single.savefig(filepath, dpi=dpi, bbox_inches='tight', format='svg')
        plt.close(fig_single)
        logger.debug(f"Saved subplot SVG: {filepath}")
    except Exception as e:
        logger.error(f"Error saving subplot to SVG for {base_name} subplot ({i},{j}): {str(e)}")
        # Clean up in case of failure
        try:
            plt.close(fig_single)
        except:
            pass

def plot_3d_graph(X3d, A, title, outpng, max_edges=2000, save_svg=False, output_dir=None, base_name=None):
    """
    3D plot of X3d (Nx3) + edges from adjacency A. 
    """
    fig = plt.figure(figsize=(7, 6))
    ax = fig.add_subplot(111, projection='3d')

    ax.scatter(X3d[:, 0], X3d[:, 1], X3d[:, 2], s=12, c='blue', alpha=0.8)
    edges = np.argwhere(A > 0)
    edges = edges[edges[:, 0] < edges[:, 1]]

    if len(edges) > max_edges:
        logger.info(f"Skipping edges since len(edges)={len(edges)} > max_edges={max_edges}")
    else:
        for (i, j) in edges:
            xs = [X3d[i, 0], X3d[j, 0]]
            ys = [X3d[i, 1], X3d[j, 1]]
            zs = [X3d[i, 2], X3d[j, 2]]
            ax.plot(xs, ys, zs, 'k-', alpha=0.2, linewidth=0.5)

    ax.set_title(title, fontsize=10)
    annotate_top_left(ax, f"N={X3d.shape[0]}")

    plt.tight_layout()
    
    # Save file
    plt.savefig(outpng, dpi=300, bbox_inches='tight')
    
    # Optionally save SVG version
    if save_svg and output_dir and base_name:
        svg_filename = os.path.join(output_dir, f"{base_name}_3D_groundtruth.svg")
        plt.savefig(svg_filename, dpi=300, bbox_inches='tight', format='svg')
    
    plt.close(fig)
    
def plot_adjacency_on_embedding(ax, X2d, A, max_edges=1000, line_color='k', line_alpha=0.15, line_width=0.5):
    """
    Overlays adjacency edges from A on the 2D embedding X2d.
    """
    edges = np.argwhere(A>0)
    edges = edges[edges[:,0]<edges[:,1]]
    if len(edges) <= max_edges:
        for (i, j) in edges:
            xvals = [X2d[i,0], X2d[j,0]]
            yvals = [X2d[i,1], X2d[j,1]]
            ax.plot(xvals, yvals, '-', color=line_color, alpha=line_alpha, linewidth=line_width)


##############################################################################
# MAIN PLOTTING FUNCTIONS
##############################################################################

def plot_dataset_matrix(data_map, d_name, offsets, fractions, results_dir, color_palette, save_svg=False):
    """
    Create a matrix of subplots showing the raw high-dimensional data (2D via PCA)
    for each combination of offset and fraction.
    """
    logger.info(f"Creating raw high-dimensional data matrix for {d_name}...")
    
    # Create figure
    fig_hd, axes_hd = plot_matrix_rows_cols(offsets, fractions, figsize=(4,3), sharex=False, sharey=False)
    fig_hd.suptitle(f"{d_name} - Raw HD data (PCA(2)) matrix", fontsize=14)
    
    # SVG output directory
    if save_svg:
        svg_dir = os.path.join(results_dir, f"{d_name}_subplots_hddata")
        os.makedirs(svg_dir, exist_ok=True)
    
    for i, off in enumerate(offsets):
        dd_off = data_map.get((off), None)
        if dd_off is None:
            for j, frac in enumerate(fractions):
                axes_hd[i][j].text(0.5, 0.5, "(no data)", ha='center', va='center', transform=axes_hd[i][j].transAxes, color='red')
            continue
        
        Xf_off = dd_off["X_full"]
        subs_off = dd_off["sub_samples"]

        # Skip if data is empty
        if Xf_off is None or not hasattr(Xf_off, 'shape') or Xf_off.shape[0] < 2:
            for j, frac in enumerate(fractions):
                axes_hd[i][j].text(0.5, 0.5, "(no data)", ha='center', va='center', transform=axes_hd[i][j].transAxes, color='red')
            continue

        # Project to 2D
        try:
            Xf_2d = pca_2d(Xf_off)
        except Exception as e:
            logger.warning(f"PCA failed for offset={off}: {e}")
            for j, frac in enumerate(fractions):
                axes_hd[i][j].text(0.5, 0.5, "(PCA failed)", ha='center', va='center', transform=axes_hd[i][j].transAxes, color='red')
            continue

        for j, frac in enumerate(fractions):
            ax = axes_hd[i][j]
            ax.set_title(f"offset={off}, frac={frac}", fontsize=10)

            # Not enough points
            if Xf_2d.shape[0] < 2:
                ax.text(0.5, 0.5, "(not enough points)", ha='center', va='center', transform=ax.transAxes, color='red')
                continue
                
            # Scatter full dataset
            scatter_2d(ax, Xf_2d, color=color_palette[0], label='Full', alpha=0.5, size=6)

            # Find sub-sample for this fraction
            sub_idx = None
            sub_method = None
            for ssub in subs_off:
                if abs(ssub["fraction"] - frac) < 1e-12:
                    sub_idx = ssub["indices_sub"]
                    sub_method = ssub["method"]
                    break

            if sub_idx is not None and len(sub_idx) > 0:
                X_sub_2d = Xf_2d[sub_idx]
                if X_sub_2d.shape[0] >= 1:
                    scatter_2d(ax, X_sub_2d, color=color_palette[1], label=f"{sub_method}", alpha=0.8, size=12, edgecolor='black')
                else:
                    ax.text(0.5, 0.5, "(empty sub-sample)", ha='center', va='center', transform=ax.transAxes, color='red')
            else:
                ax.text(0.5, 0.5, "(no sub-sample)", ha='center', va='center', transform=ax.transAxes, color='red')
                
            annotate_top_left(ax, f"N={Xf_off.shape[0]}")

            if i==0 and j==0:
                ax.legend(fontsize=7, loc='best')
                
            # Save individual subplot as SVG
            if save_svg:
                save_subplot_as_svg(fig_hd, ax, i, j, svg_dir, d_name, suffix="_hddata")

    out_hd_png = os.path.join(results_dir, f"{d_name}_matrix_hddata.png")
    fig_hd.tight_layout(rect=[0,0,1,0.95])
    fig_hd.savefig(out_hd_png, dpi=300, bbox_inches='tight')
    
    # Optionally save as SVG as well
    if save_svg:
        out_hd_svg = os.path.join(results_dir, f"{d_name}_matrix_hddata.svg")
        fig_hd.savefig(out_hd_svg, format='svg', bbox_inches='tight')
    
    plt.close(fig_hd)
    logger.info(f"Saved raw HD data matrix => {out_hd_png}")

def plot_embedding_matrix(data_map, d_name, offsets, fractions, results_dir, emb_method, color_palette, save_svg=False):
    """
    Create a matrix of subplots for a specific embedding method
    """
    logger.info(f"Building matrix of subplots for embedding method='{emb_method}' ...")
    
    # Create figure
    fig_emb, axes_emb = plot_matrix_rows_cols(offsets, fractions, figsize=(4,3), sharex=False, sharey=False)
    fig_emb.suptitle(f"{d_name} - Embedding '{emb_method}' matrix", fontsize=14)
    
    # SVG output directory
    if save_svg:
        svg_dir = os.path.join(results_dir, f"{d_name}_subplots_{emb_method}")
        os.makedirs(svg_dir, exist_ok=True)

    for i, off in enumerate(offsets):
        dd_off = data_map.get(off)
        if dd_off is None:
            continue
        
        Xf_off = dd_off["X_full"]
        Af_off = dd_off["A_full"]
        subs_off = dd_off["sub_samples"]

        # Embed in 2D
        emb_2d = embed_for_method(Xf_off, Af_off, emb_method)
        if emb_2d is None:
            # Skip entire row
            for j, frac in enumerate(fractions):
                axes_emb[i][j].text(0.5, 0.5, "(embedding failed)", ha='center', va='center', transform=axes_emb[i][j].transAxes, color='red')
            continue

        for j, frac in enumerate(fractions):
            ax = axes_emb[i][j]
            ax.set_title(f"offset={off}, frac={frac}", fontsize=10)
            scatter_2d(ax, emb_2d, color=color_palette[0], label='Full', alpha=0.5, size=6)
            
            # Sub-sample highlight
            sub_idx = None
            sub_method = None
            for ssub in subs_off:
                if abs(ssub["fraction"] - frac)<1e-12:
                    sub_idx = ssub["indices_sub"]
                    sub_method = ssub["method"]
                    break
                    
            if sub_idx is not None:
                sub_2d = emb_2d[sub_idx]
                scatter_2d(ax, sub_2d, color=color_palette[1], label=sub_method, alpha=0.8, size=12, edgecolor='black')

            annotate_top_left(ax, f"N={Xf_off.shape[0]}")

            if i==0 and j==0:
                ax.legend(fontsize=7, loc='best')
                
            # Save individual subplot as SVG
            if save_svg:
                save_subplot_as_svg(fig_emb, ax, i, j, svg_dir, d_name, suffix=f"_{emb_method}")

    out_emb_png = os.path.join(results_dir, f"{d_name}_matrix_{emb_method}.png")
    fig_emb.tight_layout(rect=[0,0,1,0.95])
    fig_emb.savefig(out_emb_png, dpi=300, bbox_inches='tight')
    
    # Optionally save as SVG as well
    if save_svg:
        out_emb_svg = os.path.join(results_dir, f"{d_name}_matrix_{emb_method}.svg")
        fig_emb.savefig(out_emb_svg, format='svg', bbox_inches='tight')
        
    plt.close(fig_emb)
    logger.info(f"Saved embedding '{emb_method}' matrix => {out_emb_png}")

def plot_kernel_matrix(data_map, d_name, offsets, fractions, results_dir, kname, kparams, color_palette, save_svg=False):
    """
    Create a matrix of subplots for a specific kernel adjacency
    """
    logger.info(f"Building matrix of kernel='{kname}' adjacency for each (offset, fraction) ...")
    
    # Create figure
    fig_kern, axes_kern = plot_matrix_rows_cols(offsets, fractions, figsize=(4,3), sharex=False, sharey=False)
    fig_kern.suptitle(f"{d_name} - Kernel='{kname}' adjacency (sub-samples)", fontsize=14)
    
    # SVG output directory
    if save_svg:
        svg_dir = os.path.join(results_dir, f"{d_name}_subplots_kernel_{kname}")
        os.makedirs(svg_dir, exist_ok=True)

    for i, off in enumerate(offsets):
        dd_off = data_map.get(off)
        if dd_off is None:
            continue
            
        subs_off = dd_off["sub_samples"]
        Xf_off = dd_off["X_full"]

        # For each fraction, find sub-sample, build adjacency, and plot
        for j, frac in enumerate(fractions):
            ax = axes_kern[i][j]
            ax.set_title(f"offset={off}, frac={frac}", fontsize=9)

            sub_idx = None
            sub_method = None
            X_sub = None
            for ssub in subs_off:
                if abs(ssub["fraction"]-frac)<1e-12:
                    X_sub = ssub["X_sub"]
                    sub_idx = ssub["indices_sub"]
                    sub_method = ssub["method"]
                    break
                    
            if X_sub is None:
                ax.text(0.5,0.5,"(missing sub-sample)",ha='center',va='center',transform=ax.transAxes)
                continue

            # Build adjacency
            try:
                K_mat = build_kernel_adjacency(X_sub, kname, kparams)
                if K_mat is None:
                    ax.text(0.5,0.5,"(kernel returned None)",ha='center',va='center',transform=ax.transAxes)
                    continue

                # PCA(2) on X_sub
                X_sub_2d = pca_2d(X_sub)
                scatter_2d(ax, X_sub_2d, color=color_palette[2], alpha=0.9, size=10)

                # Overlay edges
                plot_adjacency_on_embedding(ax, X_sub_2d, K_mat, max_edges=2000)
                annotate_top_left(ax, f"N={len(X_sub)}")
                
                # Save individual subplot as SVG
                if save_svg:
                    save_subplot_as_svg(fig_kern, ax, i, j, svg_dir, d_name, suffix=f"_kernel_{kname}")

            except Exception as e:
                logger.warning(f"Error building kernel '{kname}' adjacency for offset={off}, fraction={frac}: {e}")
                ax.text(0.5,0.5,"(kernel error)",ha='center',va='center',transform=ax.transAxes)

    out_kern_png = os.path.join(results_dir, f"{d_name}_matrix_kernel_{kname}.png")
    fig_kern.tight_layout(rect=[0,0,1,0.95])
    fig_kern.savefig(out_kern_png, dpi=300, bbox_inches='tight')
    
    # Optionally save as SVG as well
    if save_svg:
        out_kern_svg = os.path.join(results_dir, f"{d_name}_matrix_kernel_{kname}.svg")
        fig_kern.savefig(out_kern_svg, format='svg', bbox_inches='tight')
        
    plt.close(fig_kern)
    logger.info(f"Saved kernel adjacency matrix for kernel='{kname}' => {out_kern_png}")

def process_dataset(dset, config, results_dir, color_palette, save_svg=False):
    """
    Process a single dataset configuration from the config file
    """
    d_name = dset.get("name", "UnnamedDataset")
    logger.info(f"\n=== PROCESSING DATASET: {d_name} ===")
    
    # Special handling for different dataset types
    dataset_note = ""
    if is_noisy_dataset(d_name):
        dataset_note += " [NOISY]"
    if is_biased_dataset(d_name):
        dataset_note += " [BIASED]"
    if is_mnist_dataset(dset):
        dataset_note += " [MNIST]"
    if is_scrna_dataset(dset):
        dataset_note += " [scRNA]"
    if dataset_note:
        logger.info(f"Dataset characteristics:{dataset_note}")
    
    # Expand offsets from the dataset
    offsets = dset.get("offset", [0.0])
    if not isinstance(offsets, list):
        offsets = [offsets]

    fractions = dset.get("fractions", [1.0])
    if not isinstance(fractions, list):
        fractions = [fractions]
    
    # Create data container
    data_map = {}

    for off in offsets:
        # Create a copy of the dataset config, set offset
        dcopy = dict(dset)
        dcopy["offset"] = off
        
        # Possibly remove offset from gaussian_cfg if needed
        if dcopy.get("type", "") == "gaussian_tsep" and "gaussian_cfg" in dcopy:
            dcopy["gaussian_cfg"] = dict(dcopy["gaussian_cfg"])
            dcopy["gaussian_cfg"].pop("offset", None)

        logger.info(f"Generating dataset offset={off} for '{d_name}' ...")
        try:
            dd = generate_dataset_and_subsamples(dcopy)
        except Exception as e:
            logger.error(f"Error generating dataset offset={off}: {e}")
            continue

        # Extract data from results
        Xf = dd.get("X_full", None)
        Af = dd.get("A_full", None)
        subs = dd.get("sub_samples", [])

        # Skip if data is invalid
        if Xf is None or not hasattr(Xf, 'shape') or Xf.shape[0] < 2:
            logger.warning(f"X_full is empty or invalid for offset={off}, skipping plots for this offset.")
            data_map[(off)] = None
            continue
            
        data_map[(off)] = dd

    # 1. Plot raw high-dimensional data matrix
    plot_dataset_matrix(data_map, d_name, offsets, fractions, results_dir, color_palette, save_svg)
    
    # 2. Plot embedding matrices for each embedding method
    embedding_methods = config.get("embedding_methods", [])
    for emb_method in embedding_methods:
        plot_embedding_matrix(data_map, d_name, offsets, fractions, results_dir, emb_method, color_palette, save_svg)
    
    # 3. Plot 3D ground-truth adjacency for each offset
    for off in offsets:
        dd_off = data_map.get(off)
        if dd_off is None:
            continue
        Xf_off = dd_off["X_full"]
        Af_off = dd_off["A_full"]
        if Af_off is None:
            logger.warning(f"No adjacency for offset={off}. skipping 3D plot.")
            continue
        Xf_3d = pca_3d(Xf_off)
        out_3d_png = os.path.join(results_dir, f"{d_name}_offset_{off}_3D_groundtruth.png")
        
        # Create SVG subdirectory for 3D plots
        svg_dir = None
        if save_svg:
            svg_dir = os.path.join(results_dir, f"{d_name}_3D_plots")
            os.makedirs(svg_dir, exist_ok=True)
            
        plot_3d_graph(Xf_3d, Af_off, title=f"{d_name} offset={off} (3D groundtruth)", 
                     outpng=out_3d_png, save_svg=save_svg,
                     output_dir=svg_dir, base_name=f"{d_name}_offset_{off}")
        
        logger.info(f"Saved 3D ground-truth adjacency => {out_3d_png}")
    
    # 4. Plot kernel adjacency matrices for each kernel method
    kernel_methods = config.get("kernel_methods", [])
    for km in kernel_methods:
        kname = km["name"]
        kparams = km.get("params", {})
        plot_kernel_matrix(data_map, d_name, offsets, fractions, results_dir, kname, kparams, color_palette, save_svg)

    logger.info(f"Completed processing dataset: {d_name}")

##############################################################################
# MAIN FUNCTION
##############################################################################

def main():
    parser = argparse.ArgumentParser(
        description="Generates publication-quality visualizations from dataset configurations.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--config", required=True, help="Path to config JSON used for data generation.")
    parser.add_argument("--output_dir", default=None, help="Directory to store resulting plots.")
    parser.add_argument("--plot_style", default="neurips", choices=list(PLOT_STYLES.keys()),
                       help="Plot style to use (neurips, standard, minimal).")
    parser.add_argument("--color_palette", default="viridis", choices=list(COLOR_PALETTES.keys()),
                       help="Color palette to use for plots.")
    parser.add_argument("--svg_subplots", action="store_true", 
                       help="Also save individual subplots as SVG files.")
    parser.add_argument("--log_level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                       help="Set logging level.")
    args = parser.parse_args()

    # Validate and load config file
    if not os.path.exists(args.config):
        print(f"Error: Config file not found: {args.config}")
        sys.exit(1)

    with open(args.config, 'r') as f:
        try:
            config = json.load(f)
        except json.JSONDecodeError:
            print(f"Error: Invalid JSON in config file: {args.config}")
            sys.exit(1)

    # Set up results directory
    results_dir = config.get("results_dir", "results_plots")
    if args.output_dir:
        results_dir = args.output_dir
    os.makedirs(results_dir, exist_ok=True)
    
    # Set up logging
    log_dir = os.path.join(results_dir, "logs")
    log_level = getattr(logging, args.log_level.upper())
    global logger
    logger = setup_logging(log_dir, log_level)
    
    logger.info(f"Starting general_plotter with config: {args.config}")
    logger.info(f"Results will be saved to: {results_dir}")
    logger.info(f"Using plot style: {args.plot_style}")
    logger.info(f"Using color palette: {args.color_palette}")
    
    # Apply plotting style
    plt.style.use('default')  # Reset to default first
    plt.rcParams.update(PLOT_STYLES[args.plot_style])
    
    # Get color palette
    color_palette = COLOR_PALETTES[args.color_palette]
    
    # Process each dataset
    datasets = config.get("datasets", [])
    if not datasets:
        logger.warning("No datasets found in config file!")
        
    for dset in datasets:
        process_dataset(dset, config, results_dir, color_palette, args.svg_subplots)

    logger.info("\nAll datasets processed. Comprehensive plotting complete!\n")


if __name__ == "__main__":
    main()