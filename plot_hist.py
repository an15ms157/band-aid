#!/usr/bin/env python3
"""Quick inspection plots for Photon ROOT outputs.

What it does (processes all subfolders in a parent directory):
  - Lists all subfolders under the loc directory
  - For each subfolder:
    * Finds up to 5 ROOT files matching *.root
    * Searches for three histograms by name (traverses nested TLists):
      - SignalPt_BDT_MC (TH2F) - signal scatter plot
      - BackgroundPt_BDT_MC (TH2F) - background scatter plot  
      - XGB_ConvGamma_BDT_MC (TH1F) - BDT score overlay
    * Generates and saves three figures to plots/subfoldername/:
      - signal_bdt.png        : 3x2 grid of scatter plots (one per file)
      - background_bdt.png    : 3x2 grid of scatter plots (one per file)
      - xgb_score_overlay.png : overlay of 5 lines (one per file, log-y scale)
    * Each plot includes entry counts (N=total_entries) in labels

Usage:
    python plot_hist.py [--loc PARENT_DIR] [--max-files 5]

Defaults:
    loc: /home/abhishek/PhD/Work/work_A/photons/ML_analysis/training_output/XGB_on_Grid/models/MB_bkg/V4_16Jan2026/test
    max_files: 5 per subfolder

Example output structure:
    test/
    ├── 20a/
    ├── 20b/
    ├── ...
    └── plots/
        ├── 20a/
        │   ├── xgb_score_overlay.png
        │   ├── signal_bdt.png
        │   └── background_bdt.png
        ├── 20b/
        │   ├── xgb_score_overlay.png
        │   ├── signal_bdt.png
        │   └── background_bdt.png
        └── ...

Requirements:
    pip install uproot matplotlib numpy
"""

import argparse
import glob
from collections import defaultdict
from math import ceil
from pathlib import Path
from typing import Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import uproot

# Pattern: parent test directory containing subfolders
loc = "/home/abhishek/PhD/Work/work_A/photons/ML_analysis/training_output/XGB_on_Grid/models/MB_bkg/V4_16Jan2026/test"


def _get_name(obj) -> str:
    """Return a readable name for uproot objects (TList members, TH1/2/etc)."""
    for candidate in ("name", "_fName"):
        if hasattr(obj, candidate):
            val = getattr(obj, candidate)
            if isinstance(val, bytes):
                return val.decode("utf-8", errors="ignore")
            return str(val)
    try:
        return str(obj.member("fName"))
    except Exception:
        return "unknown"


def _is_tlist(obj) -> bool:
    try:
        if getattr(obj, "classname", "") == "TList":
            return True
    except Exception:
        pass
    try:
        from uproot.models.TList import Model_TList  # type: ignore
        return isinstance(obj, Model_TList)
    except Exception:
        return False


def _iter_objects(obj, path_prefix: Tuple[str, ...] = ()):
    """Yield (path, obj) for all objects reachable by iterating TLists or TDirectories."""
    yield path_prefix, obj
    # TList: iterable of children
    if _is_tlist(obj):
        for item in obj:
            name = _get_name(item)
            yield from _iter_objects(item, path_prefix + (name,))
    # TDirectory: use keys
    if hasattr(obj, "keys") and not _is_tlist(obj):
        try:
            for key in obj.keys():
                child = obj[key]
                yield from _iter_objects(child, path_prefix + (key,))
        except Exception:
            pass


def find_by_name(obj, target: str) -> Optional[Tuple[Tuple[str, ...], object]]:
    """Depth-first search for the first object whose name matches target."""
    for path, candidate in _iter_objects(obj):
        if _get_name(candidate) == target:
            return path, candidate
    return None


def load_th1(file_paths: Sequence[str], name: str):
    """Load TH1-like objects by name from each file, ensuring binning consistency."""
    edges = None
    counts_list = []
    labels = []
    for path in file_paths:
        with uproot.open(path) as f:
            root_obj = f[f.keys()[0]]  # GammaConvV1_xxxx
            found = find_by_name(root_obj, name)
            if not found:
                raise KeyError(f"Histogram '{name}' not found in {path}")
            _, h = found
            counts, bin_edges = h.to_numpy(flow=False)
            if edges is None:
                edges = bin_edges
            else:
                if len(edges) != len(bin_edges) or (edges != bin_edges).any():
                    raise ValueError(f"Bin edges differ in {path}; cannot overlay")
            counts_list.append(counts)
            labels.append(Path(path).name)  # Use ROOT filename
    return edges, counts_list, labels


def load_th2(file_paths: Sequence[str], name: str):
    """Load TH2-like objects by name from each file and return numpy content."""
    out = []
    labels = []
    for path in file_paths:
        with uproot.open(path) as f:
            root_obj = f[f.keys()[0]]
            found = find_by_name(root_obj, name)
            if not found:
                raise KeyError(f"Histogram '{name}' not found in {path}")
            _, h = found
            result = h.to_numpy(flow=False, dd=True)
            if len(result) == 2:
                counts, edges_tuple = result
                # edges_tuple is (x_edges, y_edges)
                if isinstance(edges_tuple[0], (list, tuple, np.ndarray)):
                    xedges, yedges = edges_tuple[0], edges_tuple[1]
                else:
                    # Fallback: if edges is 1D, use it for both
                    xedges, yedges = edges_tuple, edges_tuple
            else:
                counts, xedges, yedges = result
            out.append((counts, xedges, yedges))
            labels.append(Path(path).name)  # Use ROOT filename
    return out, labels


def plot_th1_overlay(bin_edges, counts_list, labels, title: str, output: str):
    bin_centers = 0.5 * (bin_edges[1:] + bin_edges[:-1])
    plt.figure(figsize=(10, 6))
    for counts, label in zip(counts_list, labels):
        n_entries = int(np.sum(counts))
        plt.step(bin_centers, counts, where="mid", label=f"{label} (N={n_entries})", linewidth=2)
    plt.yscale("log")
    plt.xlabel("Bin center")
    plt.ylabel("Counts")
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150)
    print(f"Saved {output_path}")


def plot_th2_grid(th2_data, labels, title: str, output: str):
    n = len(th2_data)
    rows, cols = 3, 2
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 3.5 * rows), squeeze=False)
    for idx, (ax, data, label) in enumerate(zip(axes.flat, th2_data, labels)):
        if idx >= n:
            break
        counts, xedges, yedges = data
        # Convert to numpy arrays if needed
        xedges = np.asarray(xedges)
        yedges = np.asarray(yedges)
        # Create scatter plot from 2D histogram
        x_bin_centers = 0.5 * (xedges[:-1] + xedges[1:])
        y_bin_centers = 0.5 * (yedges[:-1] + yedges[1:])
        X, Y = np.meshgrid(x_bin_centers, y_bin_centers)
        # Flatten arrays for scatter plot
        x_flat = X.flatten()
        y_flat = Y.flatten()
        counts_flat = counts.T.flatten()
        # Scatter plot with counts as color and size
        scatter = ax.scatter(x_flat, y_flat, c=counts_flat, s=np.sqrt(counts_flat) * 5, 
                            cmap="viridis", alpha=0.6, edgecolors="none")
        n_entries = int(np.sum(counts))
        ax.set_title(f"{label}\n(N={n_entries})")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        plt.colorbar(scatter, ax=ax, label="Counts")
    # remove unused axes
    for ax in axes.flat[n:]:
        ax.axis("off")
    fig.suptitle(title)
    fig.tight_layout(rect=(0, 0, 0.9, 0.95))
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    print(f"Saved {output_path}")

def main():
    parser = argparse.ArgumentParser(description="Plot key histograms from ROOT files (processes all subfolders)")
    parser.add_argument(
        "--loc",
        default=loc,
        help="Parent directory containing subfolders with ROOT files",
    )
    parser.add_argument("--max-files", type=int, default=5, help="Max number of files to plot per folder")
    args = parser.parse_args()

    loc_path = Path(args.loc)
    if not loc_path.exists():
        raise FileNotFoundError(f"Location not found: {args.loc}")

    # Get all subdirectories
    subfolders = sorted([d for d in loc_path.iterdir() if d.is_dir()])
    if not subfolders:
        raise FileNotFoundError(f"No subfolders found in: {args.loc}")

    print(f"Processing {len(subfolders)} subfolders...")
    
    # Process each subfolder
    for subfolder in subfolders:
        print(f"\nProcessing: {subfolder.name}")
        
        # Find ROOT files in this subfolder
        root_files = sorted(glob.glob(str(subfolder / "*.root")))[: args.max_files]
        if not root_files:
            print(f"  No ROOT files found, skipping...")
            continue
        
        print(f"  Found {len(root_files)} ROOT files")
        
        # Output directory
        output_dir = loc_path / "plots" / subfolder.name
        
        try:
            # TH1 overlay: 5 lines (one per file) on a single plot
            edges, counts_list, labels = load_th1(root_files, "XGB_ConvGamma_BDT_MC")
            plot_th1_overlay(edges, counts_list, labels, "XGB_ConvGamma_BDT_MC", 
                             str(output_dir / "xgb_score_overlay.png"))

            # TH2 grids: 3x2 subplot grid (one panel per file)
            sig_data, sig_labels = load_th2(root_files, "SignalPt_BDT_MC")
            plot_th2_grid(sig_data, sig_labels, "SignalPt_BDT_MC", 
                          str(output_dir / "signal_bdt.png"))

            bkg_data, bkg_labels = load_th2(root_files, "BackgroundPt_BDT_MC")
            plot_th2_grid(bkg_data, bkg_labels, "BackgroundPt_BDT_MC", 
                          str(output_dir / "background_bdt.png"))
            
            print(f"  ✓ Plots saved to {output_dir}")
        except Exception as e:
            print(f"  ✗ Error processing {subfolder.name}: {e}")
            continue


if __name__ == "__main__":
    main()
