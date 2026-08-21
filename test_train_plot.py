#!/usr/bin/env python3
"""Make the Photon QA summary plots for a GA train download."""

import argparse
import re
from pathlib import Path
from typing import Iterable, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import uproot


MC_FOLDERS = ("20a", "20b", "24api0", "20gpi0", "24aeta", "20geta")
MB_FOLDERS = ("20a", "20b")

RAW_GAMMA = "ESD_ConvGamma_Pt"
TRUE_PRIMARY_GAMMA = "ESD_TruePrimaryConvGamma_Pt"
TRUE_SECONDARY_GAMMA = "ESD_TrueSecondaryConvGamma_Pt"
MC_CONV_GAMMA = "MC_ConvGamma_Pt"
TRUE_PRIMARY_PI0 = "ESD_TruePrimaryPi0_MCPt_Pt_Fine"
MC_PI0_IN_ACC = "MC_Pi0WOWeightInAcc_Pt"
SIGNAL_BDT = "SignalPt_BDT_MC"
BACKGROUND_BDT = "BackgroundPt_BDT_MC"
XGB_SCORE = "XGB_ConvGamma_BDT_MC"


def object_name(obj) -> str:
    for attribute in ("name", "_fName"):
        if hasattr(obj, attribute):
            value = getattr(obj, attribute)
            return value.decode("utf-8", errors="ignore") if isinstance(value, bytes) else str(value)
    try:
        return str(obj.member("fName"))
    except Exception:
        return "unknown"


def is_tlist(obj) -> bool:
    if getattr(obj, "classname", "") == "TList":
        return True
    try:
        from uproot.models.TList import Model_TList
        return isinstance(obj, Model_TList)
    except Exception:
        return False


def walk_objects(obj):
    yield obj
    if is_tlist(obj):
        for item in obj:
            yield from walk_objects(item)
    elif hasattr(obj, "keys"):
        try:
            for key in obj.keys():
                yield from walk_objects(obj[key])
        except Exception:
            return


def find_histogram(root_file: Path, histogram_name: str):
    with uproot.open(root_file) as root:
        if not root.keys():
            raise ValueError(f"ROOT file is empty: {root_file}")
        for obj in walk_objects(root[root.keys()[0]]):
            if object_name(obj) == histogram_name:
                return obj.to_numpy(flow=False)
    raise KeyError(f"{histogram_name} not found in {root_file}")


def load_1d(root_file: Path, histogram_name: str) -> Tuple[np.ndarray, np.ndarray]:
    histogram = find_histogram(root_file, histogram_name)
    values = np.asarray(histogram[0], dtype=float)
    axes = [np.asarray(axis, dtype=float) for axis in histogram[1:]]
    if values.ndim == 1:
        return values, axes[0]
    if values.ndim == 2:
        # Secondary-conversion truth is binned in pT and source.  The pT
        # axis has the finer binning, so project over the other axis.
        if len(axes[0]) >= len(axes[1]):
            return values.sum(axis=1), axes[0]
        return values.sum(axis=0), axes[1]
    raise ValueError(f"Unsupported dimension for {histogram_name} in {root_file}")


def load_true_pi0(root_file: Path) -> Tuple[np.ndarray, np.ndarray]:
    histogram = find_histogram(root_file, TRUE_PRIMARY_PI0)
    values = np.asarray(histogram[0], dtype=float)
    if values.ndim != 2:
        raise ValueError(f"{TRUE_PRIMARY_PI0} is not two-dimensional in {root_file}")
    # The first axis is MC pT; summing the reconstructed-pT axis gives the
    # TaskV1 numerator before the efficiency correction.
    return values.sum(axis=1), np.asarray(histogram[1], dtype=float)


def load_2d(root_file: Path, histogram_name: str):
    histogram = find_histogram(root_file, histogram_name)
    values = np.asarray(histogram[0], dtype=float)
    if values.ndim != 2:
        raise ValueError(f"{histogram_name} is not two-dimensional in {root_file}")
    return values, np.asarray(histogram[1], dtype=float), np.asarray(histogram[2], dtype=float)


def rebin_to(values: np.ndarray, source_edges: np.ndarray, target_edges: np.ndarray) -> np.ndarray:
    """Rebin a fine-binned histogram, sharing partially overlapping bins."""
    if np.array_equal(source_edges, target_edges):
        return values
    if target_edges[0] < source_edges[0] or target_edges[-1] > source_edges[-1]:
        raise ValueError("Histogram binnings are not compatible")
    rebinned = np.zeros(len(target_edges) - 1, dtype=float)
    for source_index, count in enumerate(values):
        source_low = source_edges[source_index]
        source_high = source_edges[source_index + 1]
        for target_index in range(len(rebinned)):
            overlap = min(source_high, target_edges[target_index + 1]) - max(source_low, target_edges[target_index])
            if overlap > 0:
                rebinned[target_index] += count * overlap / (source_high - source_low)
    return rebinned


def sum_histograms(files: Iterable[Path], histogram_name: str) -> Tuple[np.ndarray, np.ndarray]:
    total = None
    edges = None
    for root_file in files:
        values, file_edges = load_1d(root_file, histogram_name)
        if total is None:
            total, edges = np.zeros_like(values), file_edges
        elif not np.array_equal(edges, file_edges):
            raise ValueError(f"Binning mismatch for {histogram_name} in {root_file}")
        total += values
    if total is None or edges is None:
        raise ValueError(f"No files found for {histogram_name}")
    return total, edges


def sum_pion_efficiency(files: Iterable[Path]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    numerator = None
    denominator = None
    edges = None
    for root_file in files:
        true_pi0, true_edges = load_true_pi0(root_file)
        generated_pi0, generated_edges = load_1d(root_file, MC_PI0_IN_ACC)
        true_pi0 = rebin_to(true_pi0, true_edges, generated_edges)
        if numerator is None:
            numerator = np.zeros_like(true_pi0)
            denominator = np.zeros_like(generated_pi0)
            edges = generated_edges
        elif not np.array_equal(edges, generated_edges):
            raise ValueError(f"Pion binning mismatch in {root_file}")
        numerator += true_pi0
        denominator += generated_pi0
    if numerator is None or denominator is None or edges is None:
        raise ValueError("No pion histograms found")
    return numerator, denominator, edges


def ratio_and_error(numerator: np.ndarray, denominator: np.ndarray):
    ratio = np.full_like(numerator, np.nan, dtype=float)
    error = np.full_like(numerator, np.nan, dtype=float)
    valid = denominator > 0
    ratio[valid] = numerator[valid] / denominator[valid]
    clipped = np.clip(ratio[valid], 0.0, 1.0)
    error[valid] = np.sqrt(clipped * (1.0 - clipped) / denominator[valid])
    return ratio, error


def gco_files(folder: Path) -> dict[str, Path]:
    files = {}
    for root_file in sorted(folder.glob("GCo_*.root")):
        match = re.search(r"(\d{2})$", root_file.stem)
        if match:
            files[match.group(1)] = root_file
    return files


def suffixes(loc: Path, folders: Sequence[str]) -> list[str]:
    missing = [name for name in folders if not (loc / name).is_dir()]
    if missing:
        raise FileNotFoundError(f"Missing folders: {', '.join(missing)}")
    available = [set(gco_files(loc / name)) for name in folders]
    common = sorted(set.intersection(*available))
    if not common:
        raise FileNotFoundError("No common GCo suffixes found")
    return common


def summary_curves(loc: Path, folders: Sequence[str], numerator_name: str, denominator_name: str):
    curves = []
    for suffix in suffixes(loc, folders):
        files = [gco_files(loc / name)[suffix] for name in folders]
        numerator, edges = sum_histograms(files, numerator_name)
        denominator, _ = sum_histograms(files, denominator_name)
        ratio, error = ratio_and_error(numerator, denominator)
        curves.append((f"GCo *{suffix}", edges, ratio, error))
    return curves


def purity_curves(loc: Path):
    curves = []
    for suffix in suffixes(loc, MB_FOLDERS):
        files = [gco_files(loc / name)[suffix] for name in MB_FOLDERS]
        true_primary, edges = sum_histograms(files, TRUE_PRIMARY_GAMMA)
        reco_all, _ = sum_histograms(files, RAW_GAMMA)
        true_secondary, _ = sum_histograms(files, TRUE_SECONDARY_GAMMA)
        ratio, error = ratio_and_error(true_primary, reco_all - true_secondary)
        curves.append((f"GCo *{suffix}", edges, ratio, error))
    return curves


def pion_curves(loc: Path):
    curves = []
    for suffix in suffixes(loc, MC_FOLDERS):
        files = [gco_files(loc / name)[suffix] for name in MC_FOLDERS]
        numerator, denominator, edges = sum_pion_efficiency(files)
        ratio, error = ratio_and_error(numerator, denominator)
        curves.append((f"GCo *{suffix}", edges, ratio, error))
    return curves


def plot_curves(curves, title: str, ylabel: str, output: Path, ylim=None):
    figure, axis = plt.subplots(figsize=(8.5, 5.4))
    all_centers = []
    for label, edges, values, errors in curves:
        centers = 0.5 * (edges[1:] + edges[:-1])
        widths = edges[1:] - edges[:-1]
        keep = np.isfinite(values) & (centers > 0)
        all_centers.extend(centers[keep])
        axis.errorbar(
            centers[keep], values[keep], xerr=widths[keep] / 2, yerr=errors[keep],
            marker="o", markersize=3.5, linewidth=1.4, capsize=2, label=label,
        )
    axis.set_xscale("log")
    if all_centers:
        axis.set_xlim(left=min(all_centers) * 0.8)
    if ylim:
        axis.set_ylim(*ylim)
    axis.set_xlabel(r"$p_T$ (GeV/$c$)")
    axis.set_ylabel(ylabel)
    axis.set_title(title)
    axis.grid(True, which="both", alpha=0.25)
    axis.legend(frameon=False, title="GCo")
    figure.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=180)
    plt.close(figure)
    print(f"Saved {output}")


def plot_xgb_score(files: Sequence[Path], output: Path):
    figure, axis = plt.subplots(figsize=(8.5, 5.4))
    for root_file in files:
        values, edges = load_1d(root_file, XGB_SCORE)
        centers = (edges[1:] + edges[:-1]) / 2
        axis.step(centers, values, where="mid", linewidth=1.5,
                  label=f"{root_file.name} (N={int(values.sum())})")
    axis.set_yscale("log")
    axis.set_xlabel("XGB score")
    axis.set_ylabel("Counts")
    axis.set_title(XGB_SCORE)
    axis.grid(True, which="both", alpha=0.25)
    axis.legend(fontsize=8)
    figure.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=180)
    plt.close(figure)
    print(f"Saved {output}")


def plot_bdt_grid(files: Sequence[Path], histogram_name: str, output: Path):
    figure = plt.figure(figsize=(16, 8))
    grid = figure.add_gridspec(2, 4, hspace=0.35, wspace=0.3)
    axes = [
        figure.add_subplot(grid[:, 0:2]),
        figure.add_subplot(grid[0, 2]),
        figure.add_subplot(grid[0, 3]),
        figure.add_subplot(grid[1, 2]),
        figure.add_subplot(grid[1, 3]),
    ]
    for index, (axis, root_file) in enumerate(zip(axes, files)):
        values, xedges, yedges = load_2d(root_file, histogram_name)
        xcenters = (xedges[1:] + xedges[:-1]) / 2
        ycenters = (yedges[1:] + yedges[:-1]) / 2
        xmesh, ymesh = np.meshgrid(xcenters, ycenters)
        counts = values.T
        selected = counts > 0
        image = axis.scatter(
            xmesh[selected], ymesh[selected], c=counts[selected],
            s=np.sqrt(counts[selected]) * (5 if index == 0 else 3),
            cmap="viridis", alpha=0.7, edgecolors="none",
        )
        axis.set_title(f"{root_file.name}\nN={int(values.sum())}", fontsize=12 if index == 0 else 9)
        axis.set_xlabel("pT")
        axis.set_ylabel("BDT score")
        figure.colorbar(image, ax=axis, label="Counts")
    for axis in axes[len(files):]:
        axis.axis("off")
    figure.suptitle(histogram_name)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(figure)
    print(f"Saved {output}")


def plot_input_histograms(loc: Path, max_files: int):
    for folder in sorted(path for path in loc.iterdir() if path.is_dir() and path.name != "plots"):
        files = sorted(folder.glob("GCo_*.root"))[:max_files]
        if not files:
            continue
        output = loc / "plots" / folder.name
        try:
            plot_xgb_score(files, output / "xgb_score_overlay.png")
            plot_bdt_grid(files, SIGNAL_BDT, output / "signal_bdt.png")
            plot_bdt_grid(files, BACKGROUND_BDT, output / "background_bdt.png")
        except Exception as error:
            print(f"Skipping input plots for {folder.name}: {error}")


def main():
    parser = argparse.ArgumentParser(description="Plot GA Photon train summaries")
    parser.add_argument("--loc", required=True, help="GA train download directory")
    parser.add_argument("--max-files", type=int, default=5, help="ROOT files per folder for BDT plots")
    args = parser.parse_args()
    loc = Path(args.loc)
    plots = loc / "plots"

    plot_input_histograms(loc, args.max_files)
    plot_curves(
        summary_curves(loc, MC_FOLDERS, TRUE_PRIMARY_GAMMA, MC_CONV_GAMMA),
        "Merged conversion-photon efficiency", "Efficiency", plots / "merged_efficiency.png",
    )
    plot_curves(
        purity_curves(loc),
        "MB conversion-photon purity", "Purity", plots / "mb_purity.png", (0.8, 1.1),
    )
    plot_curves(
        pion_curves(loc), "Combined pion efficiency (TaskV1)", "Efficiency",
        plots / "combined_pion_efficiency_taskv1.png",
    )


if __name__ == "__main__":
    main()
