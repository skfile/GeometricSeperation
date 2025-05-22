#!/usr/bin/env python3
"""
final_analysis_plots.py
----------------------------------------
Creates plots showing geometric separation metrics against Minimax variant metrics:

1) ARI (Adjusted Rand Index)
2) Graph GW metric (GW_Uniform_Uniform)
3) GW Tree Embedding (GH_Ultrametric_Emb)

All plots are generated against Minimax variant metrics only.

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

DIMENSION_BINS = [1, 5, 10, 15, 20, 30, 50, 100]
DIMENSION_BIN_LABELS = [f"dim_{DIMENSION_BINS[i]}-{DIMENSION_BINS[i+1]}" for i in range(len(DIMENSION_BINS)-1)]
NOISE_KEYWORDS = ["noisy", "noise", "perturb"]
BIAS_KEYWORDS = ["bias", "biased"]

PLOT_CONFIG = {
    "figure.dpi": 300,
    "font.size": 8,
    "font.family": "serif",
    "axes.linewidth": 0.5,
    "axes.titlesize": 9,
    "axes.labelsize": 8,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "lines.linewidth": 1.0,
    "lines.markersize": 4,
    "legend.fontsize": 7,
    "legend.frameon": False,
    "grid.linewidth": 0.2,
    "grid.alpha": 0.4,
}

plt.rcParams.update(PLOT_CONFIG)

COLOR_PALETTES = {
    "default": sns.color_palette("colorblind", 8),
    "method_colors": {
        "Knn": "#1f77b4",
        "Knn Mst": "#ff7f0e",
        "Knn Perturb": "#2ca02c"
    }
}

METRIC_LABELS = {
    "ARI": "ARI",
    "GW_Uniform_Uniform": "Graph GW Metric",
    "GH_Ultrametric_Emb": "GW Tree Embedding",
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

def parse_args():
    p = argparse.ArgumentParser(description="Geometric separation analysis plot generator.")
    p.add_argument("--input_csv", required=True, help="CSV with HPC experiment results.")
    p.add_argument("--output_dir", required=True, help="Directory for outputs.")
    p.add_argument("--n_jobs", type=int, default=-1, help="Parallel jobs for dataset-level analysis.")
    return p.parse_args()

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
    return logging.getLogger()

def beautify_string(s):
    if not isinstance(s, str): return s
    parts = s.replace("_"," ").split()
    return " ".join([p.capitalize() for p in parts])

def rename_kernel_method(row):
    km = row.get("Kernel_Method","")
    kp = row.get("Kernel_Params","")
    try:
        pdict = json.loads(kp)
    except:
        pdict = {}
    if km == "knn_shortest_path":
        pr = pdict.get("pruning_method", None)
        row["Kernel_Method"] = f"knn_{pr}" if pr else "knn"
    row["Kernel_Method"] = beautify_string(row["Kernel_Method"])
    return row

def rename_embedding_method(row):
    em = row.get("Embedding_Method","")
    if isinstance(em,str):
        row["Embedding_Method"] = beautify_string(em)
    return row

def parse_shapes_for_sigma_and_dim(json_str):
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
        "Dataset", "Kernel_Method", "Kernel_Params", "Embedding_Method", "Shapes_Info",
        "Offset", "Fill_Distance", "Fill_Distance_Scaled", "ARI",
        "GW_Uniform_Uniform", "GH_Ultrametric_Emb"
    ]
    missing = [n for n in needed if n not in df.columns]
    if missing:
        logger.error(f"Missing columns => {missing}")
        sys.exit(1)

def is_noisy_dataset(dataset_name):
    if not isinstance(dataset_name, str):
        return False
    return any(keyword in dataset_name.lower() for keyword in NOISE_KEYWORDS)

def is_biased_dataset(dataset_name):
    if not isinstance(dataset_name, str):
        return False
    return any(keyword in dataset_name.lower() for keyword in BIAS_KEYWORDS)
    
def categorize_by_dimension(dimension_value):
    if pd.isna(dimension_value):
        return "unknown_dim"
    for i in range(len(DIMENSION_BINS) - 1):
        if DIMENSION_BINS[i] <= dimension_value < DIMENSION_BINS[i+1]:
            return DIMENSION_BIN_LABELS[i]
    if dimension_value >= DIMENSION_BINS[-1]:
        return f"dim_{DIMENSION_BINS[-1]}plus"
    return f"dim_below_{DIMENSION_BINS[0]}"

def format_dimension_category(dim_cat):
    if not isinstance(dim_cat, str):
        return str(dim_cat)
    if dim_cat == "unknown_dim":
        return "Unknown Dimension"
    if dim_cat.startswith("dim_"):
        parts = dim_cat[4:].split('-')
        if len(parts) == 2:
            try:
                start = int(parts[0])
                end = int(parts[1])
                return f"Dimension {start} to {end}"
            except ValueError:
                pass
        if dim_cat.endswith("plus"):
            try:
                val = int(dim_cat[4:-4])
                return f"Dimension {val}+"
            except ValueError:
                pass
        if "below" in dim_cat:
            try:
                val = int(dim_cat.split("_")[-1])
                return f"Dimension Below {val}"
            except (ValueError, IndexError):
                pass
    return dim_cat.replace("_", " ").capitalize()

def do_preprocessing(df, logger):
    for c in ["dimension","SigmaValue","Spheres_Only"]:
        if c not in df.columns:
            df[c] = np.nan

    df = extract_base_dataset(df)
    df = df.apply(rename_kernel_method, axis=1)
    df = df.apply(rename_embedding_method, axis=1)
    df = df.apply(parse_shapes_info, axis=1)
    
    df["IsNoisy"] = df["Dataset"].apply(is_noisy_dataset)
    df["IsBiased"] = df["Dataset"].apply(is_biased_dataset)

    df["Offset"] = pd.to_numeric(df["Offset"], errors='coerce')
    df["Fill_Distance"] = pd.to_numeric(df["Fill_Distance"], errors='coerce')
    df["Fill_Distance_Scaled"] = pd.to_numeric(df["Fill_Distance_Scaled"], errors='coerce')
    df["dimension"] = pd.to_numeric(df["dimension"], errors='coerce')
    df["SigmaValue"] = pd.to_numeric(df["SigmaValue"], errors='coerce')
    
    df["Minimax_Offset"] = pd.to_numeric(df["Minimax_Offset"], errors='coerce')
    df["Minimax_Offset_Scaled"] = pd.to_numeric(df["Minimax_Offset_Scaled"], errors='coerce')

    df = replace_negative_metrics(df, ["GW_Uniform_Uniform","GH_Ultrametric_Emb"])

    # Calculate derived metrics
    mask_minimax = (~df["Minimax_Offset"].isna()) & (df["SigmaValue"]>0)
    df.loc[mask_minimax,"MinimaxOffsetOverSigma"] = df.loc[mask_minimax,"Minimax_Offset"] / df.loc[mask_minimax,"SigmaValue"]
    
    mask_minimax_f = (df["Fill_Distance"]>0) & mask_minimax
    df.loc[mask_minimax_f,"Ratio_MinimaxOS_over_Fill"] = df.loc[mask_minimax_f,"MinimaxOffsetOverSigma"] / df.loc[mask_minimax_f,"Fill_Distance"]
    
    mask_minimax_off = (df["Fill_Distance"]>0) & (~df["Minimax_Offset"].isna())
    df.loc[mask_minimax_off,"MinimaxOffsetOverFillDistance"] = df.loc[mask_minimax_off,"Minimax_Offset"] / df.loc[mask_minimax_off,"Fill_Distance"]
    
    mask_minimax_off_scaled = (df["Fill_Distance_Scaled"]>0) & (~df["Minimax_Offset_Scaled"].isna())
    df.loc[mask_minimax_off_scaled,"MinimaxOffsetOverFillDistanceScaled"] = df.loc[mask_minimax_off_scaled,"Minimax_Offset_Scaled"] / df.loc[mask_minimax_off_scaled,"Fill_Distance_Scaled"]

    # KNN metrics
    df["Fill_Distance_KNN_Mean_Scaled"] = pd.to_numeric(df["Fill_Distance_KNN_Mean_Scaled"], errors='coerce') if "Fill_Distance_KNN_Mean_Scaled" in df.columns else np.nan
    df["Fill_Distance_KNN_Max_Scaled"] = pd.to_numeric(df["Fill_Distance_KNN_Max_Scaled"], errors='coerce') if "Fill_Distance_KNN_Max_Scaled" in df.columns else np.nan
    df["Fill_Distance_KNN_Mean"] = pd.to_numeric(df["Fill_Distance_KNN_Mean"], errors='coerce') if "Fill_Distance_KNN_Mean" in df.columns else np.nan
    df["Fill_Distance_KNN_Max"] = pd.to_numeric(df["Fill_Distance_KNN_Max"], errors='coerce') if "Fill_Distance_KNN_Max" in df.columns else np.nan
    
    mask_minimax_knnmean = (df["Fill_Distance_KNN_Mean_Scaled"]>0) & (~df["Minimax_Offset"].isna())
    df.loc[mask_minimax_knnmean,"MinimaxOffsetOverFillDistanceKNNMeanScaled"] = df.loc[mask_minimax_knnmean,"Minimax_Offset"] / df.loc[mask_minimax_knnmean,"Fill_Distance_KNN_Mean_Scaled"]

    mask_minimax_knnmax = (df["Fill_Distance_KNN_Max_Scaled"]>0) & (~df["Minimax_Offset"].isna())
    df.loc[mask_minimax_knnmax,"MinimaxOffsetOverFillDistanceKNNMaxScaled"] = df.loc[mask_minimax_knnmax,"Minimax_Offset"] / df.loc[mask_minimax_knnmax,"Fill_Distance_KNN_Max_Scaled"]
    
    mask_minimax_knnmean_unscaled = (df["Fill_Distance_KNN_Mean"]>0) & (~df["Minimax_Offset"].isna())
    df.loc[mask_minimax_knnmean_unscaled,"MinimaxOffsetOverFillDistanceKNNMean"] = df.loc[mask_minimax_knnmean_unscaled,"Minimax_Offset"] / df.loc[mask_minimax_knnmean_unscaled,"Fill_Distance_KNN_Mean"]

    mask_minimax_knnmax_unscaled = (df["Fill_Distance_KNN_Max"]>0) & (~df["Minimax_Offset"].isna())
    df.loc[mask_minimax_knnmax_unscaled,"MinimaxOffsetOverFillDistanceKNNMax"] = df.loc[mask_minimax_knnmax_unscaled,"Minimax_Offset"] / df.loc[mask_minimax_knnmax_unscaled,"Fill_Distance_KNN_Max"]

    mask_minimax_scaled_off = (df["Fill_Distance"]>0) & (~df["Minimax_Offset_Scaled"].isna())
    df.loc[mask_minimax_scaled_off,"MinimaxOffsetScaledOverFillDistance"] = df.loc[mask_minimax_scaled_off,"Minimax_Offset_Scaled"] / df.loc[mask_minimax_scaled_off,"Fill_Distance"]
    
    mask_minimax_scaled_knnmean_unscaled = (df["Fill_Distance_KNN_Mean"]>0) & (~df["Minimax_Offset_Scaled"].isna())
    df.loc[mask_minimax_scaled_knnmean_unscaled,"MinimaxOffsetScaledOverFillDistanceKNNMean"] = df.loc[mask_minimax_scaled_knnmean_unscaled,"Minimax_Offset_Scaled"] / df.loc[mask_minimax_scaled_knnmean_unscaled,"Fill_Distance_KNN_Mean"]
    
    mask_minimax_scaled_knnmax_unscaled = (df["Fill_Distance_KNN_Max"]>0) & (~df["Minimax_Offset_Scaled"].isna())
    df.loc[mask_minimax_scaled_knnmax_unscaled,"MinimaxOffsetScaledOverFillDistanceKNNMax"] = df.loc[mask_minimax_scaled_knnmax_unscaled,"Minimax_Offset_Scaled"] / df.loc[mask_minimax_scaled_knnmax_unscaled,"Fill_Distance_KNN_Max"]

    df["DimensionCategory"] = df["dimension"].apply(categorize_by_dimension)
    
    logger.info(f"Preprocessing complete. {len(df)} rows remain.")
    return df

def save_plot(fig, output_path, dpi=300):
    pdf_path = f"{output_path}.pdf"
    fig.savefig(pdf_path, bbox_inches="tight", dpi=dpi)
    svg_path = f"{output_path}.svg"
    fig.savefig(svg_path, bbox_inches="tight", dpi=dpi)
    return pdf_path, svg_path

def bin_data_column(sub, xcol, ycol, groupcol, nbins=8):
    out={}
    sub = sub.dropna(subset=[xcol,ycol,groupcol])
    if sub.empty:
        return out
    mn,mx = sub[xcol].min(), sub[xcol].max()
    if mn==mx:
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
    axA, axB, axC = axes
    
    ari_label = METRIC_LABELS.get("ARI", "ARI")
    gw_metric_label = METRIC_LABELS.get("GW_Uniform_Uniform", "Graph GW Metric")
    gh_metric_label = METRIC_LABELS.get("GH_Ultrametric_Emb", "GW Tree Embedding")
    x_axis_label = METRIC_LABELS.get(xcol, x_label)

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

    # ARI
    subA = df_sub.dropna(subset=[xcol,"ARI"])
    if subA.empty:
        axA.text(0.5,0.5,"No ARI data",ha='center',va='center',transform=axA.transAxes,color='red')
        axA.set_title(ari_label)
    else:
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

    # Graph GW
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

    # Tree Embedding
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

def single_base_dataset_analysis(df_sub, base_name, outdir, logger):
    if df_sub.empty:
        logger.warning(f"[{base_name}] No data => skip.")
        return
    logger.info(f"=== Single dataset => {base_name} ===")
    
    minimax_metrics = [
        ("Minimax_Offset", "Minimax Offset"),
        ("Minimax_Offset_Scaled", "Minimax Offset Scaled"),
        ("MinimaxOffsetOverSigma", "Minimax Offset/Sigma"),
        ("MinimaxOffsetOverFillDistance", "Minimax Offset/FillDist"),
        ("MinimaxOffsetOverFillDistanceScaled", "Minimax Offset/FillDistScaled"),
        ("Ratio_MinimaxOS_over_Fill", "Ratio MinimaxOS/FILL"),
        ("MinimaxOffsetOverFillDistanceKNNMeanScaled", "Minimax Offset/FillDistKNNMeanScaled"),
        ("MinimaxOffsetOverFillDistanceKNNMaxScaled", "Minimax Offset/FillDistKNNMaxScaled"),
        ("MinimaxOffsetOverFillDistanceKNNMean", "Minimax Offset/FillDistKNNMean"),
        ("MinimaxOffsetOverFillDistanceKNNMax", "Minimax Offset/FillDistKNNMax"),
        ("MinimaxOffsetScaledOverFillDistance", "Minimax Offset Scaled/FillDist"),
        ("MinimaxOffsetScaledOverFillDistanceKNNMean", "Minimax Offset Scaled/FillDistKNNMean"),
        ("MinimaxOffsetScaledOverFillDistanceKNNMax", "Minimax Offset Scaled/FillDistKNNMax")
    ]

    for col_name, col_label in minimax_metrics:
        if col_name not in df_sub.columns or df_sub[col_name].dropna().empty:
            continue
            
        fig, axes = plt.subplots(1, 3, figsize=(14, 4))
        fig.suptitle(f"{base_name}: vs {col_label}", fontsize=13)
        multi_plot_3metrics_vs_x(axes, df_sub, col_name, col_label)
        out_path = os.path.join(outdir, f"{base_name}_3plot_vs_{col_name}")
        fig.tight_layout(rect=[0, 0, 1, 0.92])
        save_plot(fig, out_path, dpi=300)
        plt.close(fig)

def process_dimensional_data(df, dim_cat, outdir, logger):
    formatted_dim_cat = format_dimension_category(dim_cat)
    
    minimax_metrics = [
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
    
    for col_name, col_label in minimax_metrics:
        if col_name not in df.columns or df[col_name].dropna().empty:
            logger.info(f"Skipping {col_name} for {dim_cat} - no data available")
            continue
            
        display_label = METRIC_LABELS.get(col_name, col_label)
        
        fig, axes = plt.subplots(1, 3, figsize=(14, 4))
        fig.suptitle(f"{formatted_dim_cat} - vs. {display_label}", fontsize=13)
        multi_plot_3metrics_vs_x(axes, df, col_name, display_label)
        out_path = os.path.join(outdir, f"{dim_cat}_3plot_vs_{col_name}")
        fig.tight_layout(rect=[0, 0, 1, 0.92])
        save_plot(fig, out_path, dpi=300)
        plt.close(fig)

def process_noise_bias_data(df, category, outdir, logger):
    minimax_metrics = [
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
    
    for col_name, col_label in minimax_metrics:
        if col_name not in df.columns or df[col_name].dropna().empty:
            logger.info(f"Skipping {col_name} for {category} - no data available")
            continue
            
        display_label = METRIC_LABELS.get(col_name, col_label)
        
        fig, axes = plt.subplots(1, 3, figsize=(14, 4))
        fig.suptitle(f"{category} Data - vs. {display_label}", fontsize=13)
        multi_plot_3metrics_vs_x(axes, df, col_name, display_label)
        out_path = os.path.join(outdir, f"{category}_3plot_vs_{col_name}")
        fig.tight_layout(rect=[0, 0, 1, 0.92])
        save_plot(fig, out_path, dpi=300)
        plt.close(fig)

def process_combined_data(df, prefix, outdir, logger):
    formatted_prefix = prefix
    if "_dim_" in prefix.lower():
        parts = prefix.split("_", 1)
        dim_part = parts[0]
        remaining = parts[1] if len(parts) > 1 else ""
        formatted_dim_cat = format_dimension_category(dim_part)
        formatted_prefix = f"{formatted_dim_cat} {remaining}"
    
    minimax_metrics = [
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
    
    for col_name, col_label in minimax_metrics:
        if col_name not in df.columns or df[col_name].dropna().empty:
            logger.info(f"Skipping {col_name} for {prefix} - no data available")
            continue
            
        display_label = METRIC_LABELS.get(col_name, col_label)
        
        fig, axes = plt.subplots(1, 3, figsize=(14, 4))
        fig.suptitle(f"{formatted_prefix} - vs. {display_label}", fontsize=13)
        multi_plot_3metrics_vs_x(axes, df, col_name, display_label)
        out_path = os.path.join(outdir, f"{prefix}_3plot_vs_{col_name}")
        fig.tight_layout(rect=[0, 0, 1, 0.92])
        save_plot(fig, out_path, dpi=300)
        plt.close(fig)

def aggregate_by_dimension(df, outdir, logger):
    logger.info("=== Aggregating data by dimension categories ===")
    
    dim_outdir = os.path.join(outdir, "dimension_aggregated")
    os.makedirs(dim_outdir, exist_ok=True)
    
    dimension_categories = df["DimensionCategory"].dropna().unique()
    
    for dim_cat in sorted(dimension_categories):
        df_dim = df[df["DimensionCategory"] == dim_cat].copy()
        
        if df_dim.empty:
            logger.warning(f"No data for dimension category {dim_cat}")
            continue
        
        dim_cat_dir = os.path.join(dim_outdir, dim_cat)
        os.makedirs(dim_cat_dir, exist_ok=True)
        
        logger.info(f"Processing dimension category {dim_cat} with {len(df_dim)} data points")
        
        process_dimensional_data(df_dim, dim_cat, dim_cat_dir, logger)

def aggregate_by_noise(df, outdir, logger):
    logger.info("=== Aggregating data by noise status ===")
    
    noise_outdir = os.path.join(outdir, "noise_aggregated")
    os.makedirs(noise_outdir, exist_ok=True)
    
    df_noisy = df[df["IsNoisy"] == True].copy()
    if not df_noisy.empty:
        noisy_dir = os.path.join(noise_outdir, "noisy")
        os.makedirs(noisy_dir, exist_ok=True)
        logger.info(f"Processing noisy data with {len(df_noisy)} data points")
        process_noise_bias_data(df_noisy, "Noisy", noisy_dir, logger)
    
    df_clean = df[df["IsNoisy"] == False].copy()
    if not df_clean.empty:
        clean_dir = os.path.join(noise_outdir, "clean")
        os.makedirs(clean_dir, exist_ok=True)
        logger.info(f"Processing clean data with {len(df_clean)} data points")
        process_noise_bias_data(df_clean, "Clean", clean_dir, logger)

def aggregate_by_bias(df, outdir, logger):
    logger.info("=== Aggregating data by bias status ===")
    
    bias_outdir = os.path.join(outdir, "bias_aggregated")
    os.makedirs(bias_outdir, exist_ok=True)
    
    df_biased = df[df["IsBiased"] == True].copy()
    if not df_biased.empty:
        biased_dir = os.path.join(bias_outdir, "biased")
        os.makedirs(biased_dir, exist_ok=True)
        logger.info(f"Processing biased data with {len(df_biased)} data points")
        process_noise_bias_data(df_biased, "Biased", biased_dir, logger)
    
    df_unbiased = df[df["IsBiased"] == False].copy()
    if not df_unbiased.empty:
        unbiased_dir = os.path.join(bias_outdir, "unbiased")
        os.makedirs(unbiased_dir, exist_ok=True)
        logger.info(f"Processing unbiased data with {len(df_unbiased)} data points")
        process_noise_bias_data(df_unbiased, "Unbiased", unbiased_dir, logger)

def aggregate_by_combined_factors(df, outdir, logger):
    logger.info("=== Aggregating data by combined factors ===")
    
    combined_outdir = os.path.join(outdir, "combined_aggregated")
    os.makedirs(combined_outdir, exist_ok=True)
    
    dimension_categories = df["DimensionCategory"].dropna().unique()
    
    for dim_cat in sorted(dimension_categories):
        df_dim = df[df["DimensionCategory"] == dim_cat].copy()
        
        if df_dim.empty:
            continue
            
        for is_noisy in [True, False]:
            for is_biased in [True, False]:
                df_subset = df_dim[(df_dim["IsNoisy"] == is_noisy) & (df_dim["IsBiased"] == is_biased)].copy()
                
                if df_subset.empty:
                    continue
                
                noise_str = "noisy" if is_noisy else "clean"
                bias_str = "biased" if is_biased else "unbiased"
                subset_dir = os.path.join(combined_outdir, f"{dim_cat}_{noise_str}_{bias_str}")
                os.makedirs(subset_dir, exist_ok=True)
                
                subset_prefix = f"{dim_cat}_{noise_str}_{bias_str}"
                logger.info(f"Processing {subset_prefix} with {len(df_subset)} data points")
                
                process_combined_data(df_subset, subset_prefix, subset_dir, logger)

def main():
    args = parse_args()
    logger = setup_logging(args.output_dir)
    logger.info("=== Starting plotting script ===")

    try:
        df = pd.read_csv(args.input_csv)
        logger.info(f"Read {len(df)} rows from {args.input_csv}")
    except Exception as e:
        logger.error(f"CSV read fail => {e}")
        sys.exit(1)

    validate_input_data(df, logger)
    df = do_preprocessing(df, logger)
    if df.empty:
        logger.warning("No data => done.")
        return

    base_list = sorted(df["Base_Dataset"].dropna().unique())
    logger.info(f"Found {len(base_list)} base datasets: {base_list}")

    def process_dataset(base_ds):
        sub_ = df[df["Base_Dataset"] == base_ds]
        if sub_.empty:
            return
        outd = os.path.join(args.output_dir, base_ds)
        os.makedirs(outd, exist_ok=True)
        single_base_dataset_analysis(sub_, base_ds, outd, logger)

    Parallel(n_jobs=args.n_jobs)(
        delayed(process_dataset)(bd) for bd in base_list
    )

    aggregate_by_dimension(df, args.output_dir, logger)
    aggregate_by_noise(df, args.output_dir, logger)
    aggregate_by_bias(df, args.output_dir, logger)
    aggregate_by_combined_factors(df, args.output_dir, logger)

    logger.info("=== Done. ===")

if __name__ == "__main__":
    main()
