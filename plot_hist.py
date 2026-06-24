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
  - Also generates summary plots in plots/:
    * merged_efficiency.png : merged efficiency from all MC folders
    * mb_purity.png         : MB purity from the detected MB folder pair

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
import re
from collections import defaultdict
from math import ceil
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import uproot

# Pattern: parent test directory containing subfolders
loc = "/home/abhishek/PhD/Work/work_A/photons/ML_analysis/training_output/XGB_on_Grid/models/2026/June/June19/test"

TH2_SIGNAL = "SignalPt_BDT_MC"
TH2_BACKGROUND = "BackgroundPt_BDT_MC"
TH1_XGB = "XGB_ConvGamma_BDT_MC"

RAW_GAMMA_HIST = "ESD_ConvGamma_Pt"
TRUE_PRIMARY_HIST = "ESD_TruePrimaryConvGamma_Pt"
TRUE_SECONDARY_HIST = "ESD_TrueSecondaryConvGamma_Pt"
MC_CONV_HIST = "MC_ConvGamma_Pt"

MERGED_EFFICIENCY_PLOT = "merged_efficiency.png"
MB_PURITY_PLOT = "mb_purity.png"

ALLOWED_MC_FOLDERS = ("20a", "20b", "20c", "20gpi0", "24pi0", "20geta", "24eta")
MB_FOLDER_PAIRS = (("20a", "20c"), ("20a", "20b"))


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


def _extract_suffix(path: Path) -> Optional[str]:
    """Extract the trailing BDT index from a ROOT file name."""
    match = re.search(r"(\d{2})$", path.stem)
    if match:
        return match.group(1)

    digits = re.findall(r"(\d+)", path.stem)
    if digits:
        return digits[-1][-2:]
    return None


def _subfolders_in(loc_path: Path) -> List[Path]:
    """List channel folders while skipping the output directory."""
    return sorted(
        folder for folder in loc_path.iterdir()
        if folder.is_dir() and folder.name != "plots"
    )


def _group_files_by_suffix(root_files: Sequence[Path]) -> Dict[str, Path]:
    """Map the trailing BDT index to its ROOT file."""
    grouped: Dict[str, Path] = {}
    for root_file in sorted(root_files):
        suffix = _extract_suffix(root_file)
        if suffix is not None:
            grouped[suffix] = root_file
    return grouped


def _infer_mb_folders(available_folders: Iterable[str]) -> List[str]:
    available = set(available_folders)
    for pair in MB_FOLDER_PAIRS:
        if all(name in available for name in pair):
            return list(pair)
    if "MB" in available:
        return ["MB"]
    raise ValueError(f"Could not infer MB folders from: {sorted(available)}")


def _infer_all_mc_folders(available_folders: Iterable[str]) -> List[str]:
    available = set(available_folders)
    folders = [name for name in ALLOWED_MC_FOLDERS if name in available]
    if not folders:
        raise ValueError(f"No known MC folders found in: {sorted(available)}")
    return folders


def _build_summary_groups(loc_path: Path):
    folder_map = {folder.name: folder for folder in _subfolders_in(loc_path)}
    mb_folders = _infer_mb_folders(folder_map.keys())
    all_mc_folders = _infer_all_mc_folders(folder_map.keys())

    per_folder_groups = {
        name: _group_files_by_suffix(sorted(folder_map[name].glob("*.root")))
        for name in set(mb_folders + all_mc_folders)
    }

    common_suffixes = sorted(set.intersection(*(set(group.keys()) for group in per_folder_groups.values())))
    if not common_suffixes:
        raise ValueError("No common BDT suffixes found across required folders")

    return folder_map, mb_folders, all_mc_folders, per_folder_groups, common_suffixes


def load_th1(file_paths: Sequence[str], name: str):
    """Load TH1-like objects by name from each file, ensuring binning consistency."""
    if len(file_paths) < 5:
        print(f"  ⚠ Warning: Only {len(file_paths)} file(s) found (expected up to 5)")
    
    edges = None
    counts_list = []
    labels = []
    for path in file_paths:
        try:
            with uproot.open(path) as f:
                if not f.keys():
                    print(f"  ⚠ Skipping {Path(path).name}: ROOT file is empty")
                    continue
                root_obj = f[f.keys()[0]]  # GammaConvV1_xxxx
                found = find_by_name(root_obj, name)
                if not found:
                    print(f"  ⚠ Skipping {Path(path).name}: Histogram '{name}' not found")
                    continue
                _, h = found
                counts, bin_edges = h.to_numpy(flow=False)
                if edges is None:
                    edges = bin_edges
                else:
                    if len(edges) != len(bin_edges) or (edges != bin_edges).any():
                        raise ValueError(f"Bin edges differ in {path}; cannot overlay")
                counts_list.append(counts)
                labels.append(Path(path).name)  # Use ROOT filename
        except Exception as e:
            print(f"  ⚠ Skipping {Path(path).name}: {e}")
            continue
    
    if not counts_list:
        raise ValueError(f"No valid histograms found for '{name}' in any file")
    return edges, counts_list, labels


def load_th2(file_paths: Sequence[str], name: str):
    """Load TH2-like objects by name from each file and return numpy content."""
    out = []
    labels = []
    for path in file_paths:
        try:
            with uproot.open(path) as f:
                if not f.keys():
                    print(f"  ⚠ Skipping {Path(path).name}: ROOT file is empty")
                    continue
                root_obj = f[f.keys()[0]]
                found = find_by_name(root_obj, name)
                if not found:
                    print(f"  ⚠ Skipping {Path(path).name}: Histogram '{name}' not found")
                    continue
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
        except Exception as e:
            print(f"  ⚠ Skipping {Path(path).name}: {e}")
            continue
    
    if not out:
        raise ValueError(f"No valid histograms found for '{name}' in any file")
    return out, labels


def load_hist_projected_to_pt(root_file: Path, hist_name: str):
    """Load a histogram and project it to the pT axis if needed."""
    with uproot.open(root_file) as f:
        if not f.keys():
            raise ValueError(f"ROOT file is empty: {root_file}")

        root_obj = f[f.keys()[0]]
        found = find_by_name(root_obj, hist_name)
        if not found:
            raise KeyError(f"Histogram '{hist_name}' not found in {root_file}")

        _, hist = found
        hist_tuple = hist.to_numpy(flow=False)
        values = np.asarray(hist_tuple[0], dtype=float)
        axes = [np.asarray(axis, dtype=float) for axis in hist_tuple[1:]]

        if values.ndim == 1:
            return values, axes[0]
        if values.ndim == 2:
            if len(axes[0]) >= len(axes[1]):
                return values.sum(axis=1), axes[0]
            return values.sum(axis=0), axes[1]

        raise ValueError(
            f"Unsupported histogram dimension {values.ndim} for {hist_name} in {root_file}"
        )


def sum_histograms(files: Sequence[Path], hist_name: str):
    """Sum the requested histogram across multiple ROOT files."""
    total = None
    ref_edges = None

    for root_file in files:
        values, edges = load_hist_projected_to_pt(root_file, hist_name)
        if total is None:
            total = np.zeros_like(values, dtype=float)
            ref_edges = edges
        else:
            if len(values) != len(total) or not np.allclose(edges, ref_edges):
                raise ValueError(f"Binning mismatch for {hist_name} in {root_file}")
        total += values

    if total is None or ref_edges is None:
        raise ValueError(f"No files provided for {hist_name}")

    return total, ref_edges


def safe_ratio(numerator: np.ndarray, denominator: np.ndarray):
    """Compute a ratio while masking bins with non-positive denominator."""
    ratio = np.full_like(numerator, np.nan, dtype=float)
    valid = denominator > 0
    ratio[valid] = numerator[valid] / denominator[valid]
    return ratio, valid


def ratio_error(denominator: np.ndarray, ratio: np.ndarray, valid: np.ndarray):
    """Binomial uncertainty for efficiency/purity-style ratios."""
    error = np.full_like(ratio, np.nan, dtype=float)
    clipped = np.clip(ratio[valid], 0.0, 1.0)
    error[valid] = np.sqrt(clipped * (1.0 - clipped) / denominator[valid])
    return error


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
    plt.close()
    print(f"Saved {output_path}")


def plot_th2_grid(th2_data, labels, title: str, output: str):
    n = len(th2_data)
    if n == 0:
        print(f"No data to plot for {title}")
        return
    
    # Warn if fewer than 5 files
    if n < 5:
        print(f"  ⚠ Warning: Only {n} file(s) found (expected up to 5)")
    
    # Landscape mode with GridSpec: 50% left (1 plot), 50% right (max 4 plots)
    fig = plt.figure(figsize=(16, 8))
    gs = fig.add_gridspec(2, 4, hspace=0.3, wspace=0.3)
    
    # Large plot on left (spans full height)
    ax_large = fig.add_subplot(gs[:, 0:2])
    
    # Small plots on right (up to 4)
    ax_small = []
    ax_small.append(fig.add_subplot(gs[0, 2]))
    ax_small.append(fig.add_subplot(gs[0, 3]))
    ax_small.append(fig.add_subplot(gs[1, 2]))
    ax_small.append(fig.add_subplot(gs[1, 3]))
    
    # Plot largest (first) on left
    if n > 0:
        counts, xedges, yedges = th2_data[0]
        xedges = np.asarray(xedges)
        yedges = np.asarray(yedges)
        x_bin_centers = 0.5 * (xedges[:-1] + xedges[1:])
        y_bin_centers = 0.5 * (yedges[:-1] + yedges[1:])
        X, Y = np.meshgrid(x_bin_centers, y_bin_centers)
        x_flat = X.flatten()
        y_flat = Y.flatten()
        counts_flat = counts.T.flatten()
        scatter = ax_large.scatter(x_flat, y_flat, c=counts_flat, s=np.sqrt(counts_flat) * 5, 
                                   cmap="viridis", alpha=0.6, edgecolors="none")
        n_entries = int(np.sum(counts))
        ax_large.set_title(f"{labels[0]}\n(N={n_entries})", fontsize=12, fontweight="bold")
        ax_large.set_xlabel("x")
        ax_large.set_ylabel("y")
        plt.colorbar(scatter, ax=ax_large, label="Counts")
    
    # Plot remaining (up to 4) on right
    for idx, ax_small_plot in enumerate(ax_small):
        data_idx = idx + 1
        if data_idx < n:
            counts, xedges, yedges = th2_data[data_idx]
            xedges = np.asarray(xedges)
            yedges = np.asarray(yedges)
            x_bin_centers = 0.5 * (xedges[:-1] + xedges[1:])
            y_bin_centers = 0.5 * (yedges[:-1] + yedges[1:])
            X, Y = np.meshgrid(x_bin_centers, y_bin_centers)
            x_flat = X.flatten()
            y_flat = Y.flatten()
            counts_flat = counts.T.flatten()
            scatter = ax_small_plot.scatter(x_flat, y_flat, c=counts_flat, s=np.sqrt(counts_flat) * 3, 
                                           cmap="viridis", alpha=0.6, edgecolors="none")
            n_entries = int(np.sum(counts))
            ax_small_plot.set_title(f"{labels[data_idx]}\n(N={n_entries})", fontsize=10)
            ax_small_plot.set_xlabel("x", fontsize=9)
            ax_small_plot.set_ylabel("y", fontsize=9)
            plt.colorbar(scatter, ax=ax_small_plot, label="Counts")
        else:
            ax_small_plot.axis("off")
    
    fig.suptitle(title, fontsize=14, fontweight="bold")
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {output_path}")


def plot_summary_curves(
    curves: Sequence[Tuple[str, np.ndarray, np.ndarray, np.ndarray]],
    title: str,
    ylabel: str,
    output: str,
    ylim: Optional[Tuple[float, float]] = None,
    yticks: Optional[Sequence[float]] = None,
):
    """Plot summary purity/efficiency curves for all detected BDT points."""
    if not curves:
        raise ValueError("No summary curves available to plot")

    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8.5, 5.4))
    positive_centers = []

    for label, edges, values, errors in curves:
        centers = 0.5 * (edges[1:] + edges[:-1])
        widths = edges[1:] - edges[:-1]
        finite = np.isfinite(values) & (centers > 0)
        positive_centers.extend(centers[finite].tolist())
        ax.errorbar(
            centers[finite],
            values[finite],
            xerr=0.5 * widths[finite],
            yerr=errors[finite],
            marker="o",
            markersize=3.5,
            linewidth=1.4,
            capsize=2,
            label=label,
        )

    ax.set_xscale("log")
    if positive_centers:
        ax.set_xlim(left=min(positive_centers) * 0.8)
    if ylim is not None:
        ax.set_ylim(*ylim)
    if yticks is not None:
        ax.set_yticks(yticks)
    ax.set_xlabel(r"$p_T$ (GeV/$c$)")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(frameon=False, title="BDT")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    print(f"Saved {output_path}")


def merged_efficiency_curves(loc_path: Path):
    """Build merged MC efficiency curves using the folder convention in the reference code."""
    folder_map, _, all_mc_folders, per_folder_groups, suffixes = _build_summary_groups(loc_path)
    curves = []

    for suffix in suffixes:
        files = [folder_map[name] / per_folder_groups[name][suffix].name for name in all_mc_folders]
        true_primary, edges = sum_histograms(files, TRUE_PRIMARY_HIST)
        generated, _ = sum_histograms(files, MC_CONV_HIST)
        efficiency, valid = safe_ratio(true_primary, generated)
        error = ratio_error(generated, efficiency, valid)
        curves.append((suffix, edges, efficiency, error))

    return curves


def mb_purity_curves(loc_path: Path):
    """Build MB purity curves using the automatically detected MB folder pair."""
    folder_map, mb_folders, _, per_folder_groups, suffixes = _build_summary_groups(loc_path)
    curves = []

    for suffix in suffixes:
        files = [folder_map[name] / per_folder_groups[name][suffix].name for name in mb_folders]
        true_primary, edges = sum_histograms(files, TRUE_PRIMARY_HIST)
        reco_all, _ = sum_histograms(files, RAW_GAMMA_HIST)
        true_secondary, _ = sum_histograms(files, TRUE_SECONDARY_HIST)
        denominator = reco_all - true_secondary
        purity, valid = safe_ratio(true_primary, denominator)
        error = ratio_error(denominator, purity, valid)
        curves.append((suffix, edges, purity, error))

    return curves, " + ".join(mb_folders)

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

    # Get all subdirectories except the output folder
    subfolders = _subfolders_in(loc_path)
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
            edges, counts_list, labels = load_th1(root_files, TH1_XGB)
            plot_th1_overlay(edges, counts_list, labels, TH1_XGB,
                             str(output_dir / "xgb_score_overlay.png"))

            # TH2 grids: 3x2 subplot grid (one panel per file)
            sig_data, sig_labels = load_th2(root_files, TH2_SIGNAL)
            plot_th2_grid(sig_data, sig_labels, TH2_SIGNAL,
                          str(output_dir / "signal_bdt.png"))

            bkg_data, bkg_labels = load_th2(root_files, TH2_BACKGROUND)
            plot_th2_grid(bkg_data, bkg_labels, TH2_BACKGROUND,
                          str(output_dir / "background_bdt.png"))
            
            print(f"  ✓ Plots saved to {output_dir}")
        except Exception as e:
            print(f"  ✗ Error processing {subfolder.name}: {e}")
            continue

    try:
        efficiency_curves = merged_efficiency_curves(loc_path)
        plot_summary_curves(
            efficiency_curves,
            "Merged efficiency vs pT",
            "Efficiency = true primary conv / generated conv gamma",
            str(loc_path / "plots" / MERGED_EFFICIENCY_PLOT),
        )
    except Exception as e:
        print(f"Skipping merged efficiency summary: {e}")

    try:
        purity_curves, mb_label = mb_purity_curves(loc_path)
        plot_summary_curves(
            purity_curves,
            f"MB purity vs pT ({mb_label})",
            "Purity = true primary conv / (reco conv - true secondary conv)",
            str(loc_path / "plots" / MB_PURITY_PLOT),
            ylim=(0.8, 1.1),
            yticks=np.arange(0.8, 1.11, 0.1),
        )
    except Exception as e:
        print(f"Skipping MB purity summary: {e}")


if __name__ == "__main__":
    main()
