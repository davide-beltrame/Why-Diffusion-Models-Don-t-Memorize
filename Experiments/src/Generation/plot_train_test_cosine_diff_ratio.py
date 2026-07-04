#!/usr/bin/env python3
"""Plot test-vs-train cosine difference and ratio for PCA and Wavelet levels."""

from __future__ import annotations

import argparse
import glob
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
import numpy as np


LEVELS = [
    "PCA_L0",
    "PCA_L1",
    "PCA_L2",
    "PCA_L3",
    "Wavelet_L0",
    "Wavelet_L1",
    "Wavelet_L2",
    "Wavelet_L3",
]

LEVEL_COLORS = {
    "L0": "#1f77b4",
    "L1": "#ff7f0e",
    "L2": "#2ca02c",
    "L3": "#d62728",
}

FAMILY_STYLE = {
    "PCA": {"linestyle": "-", "marker": "o"},
    "Wavelet": {"linestyle": "--", "marker": "s"},
}


def _find_input_npz(level_dir: Path, time_step: int, n_train: int) -> Path:
    pattern = (
        "backward_cosine_similarity_metric_indexA_vs_B_"
        f"Npairs_*_time_{time_step}_2y_issameFalse_N_train_{n_train}.npz"
    )
    matches = sorted(glob.glob(str(level_dir / "Comparisons" / pattern)))
    if not matches:
        raise FileNotFoundError(f"No comparison NPZ found for {level_dir} with pattern: {pattern}")
    return Path(matches[-1])


def _plot_level(npz_path: Path, out_png: Path, label: str) -> None:
    data = np.load(npz_path)

    required = ["training_times", "all_means_train", "all_means_test"]
    missing = [k for k in required if k not in data]
    if missing:
        raise KeyError(f"Missing keys in {npz_path}: {missing}")

    t = np.asarray(data["training_times"]).astype(float)
    train = np.asarray(data["all_means_train"]).astype(float)
    test = np.asarray(data["all_means_test"]).astype(float)

    if t.shape[0] != train.shape[0] or t.shape[0] != test.shape[0]:
        raise ValueError(
            f"Shape mismatch in {npz_path}: "
            f"training_times={t.shape}, train={train.shape}, test={test.shape}"
        )

    # Raw value difference and raw ratio.
    diff_test_minus_train = test - train
    ratio_test_over_train = np.divide(
        test,
        train,
        out=np.full_like(test, np.nan),
        where=np.abs(train) > 1e-12,
    )

    fig, (ax0, ax1) = plt.subplots(2, 1, figsize=(8.4, 6.3), sharex=True)

    ax0.plot(t, diff_test_minus_train, color="#1f77b4", linewidth=1.8)
    ax0.axhline(0.0, color="black", linestyle="--", linewidth=1.0, alpha=0.7)
    ax0.set_ylabel("test - train")
    ax0.set_title(f"{label}: diff = test - train, ratio = test / train")
    ax0.grid(True, alpha=0.3)

    ax1.plot(t, ratio_test_over_train, color="#d62728", linewidth=1.8)
    ax1.axhline(1.0, color="black", linestyle="--", linewidth=1.0, alpha=0.7)
    ax1.set_ylabel("test / train (raw)")
    ax1.set_xlabel("checkpoint")
    ax1.grid(True, alpha=0.3)

    # Disable Matplotlib y-offset notation (e.g., "+1") to show raw ratios.
    ax1.ticklabel_format(axis="y", style="plain", useOffset=False)
    ax1.yaxis.set_major_formatter(FormatStrFormatter("%.6f"))

    ax0.set_xscale("log")
    ax1.set_xscale("log")

    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _split_family_level(level_label: str) -> tuple[str, str]:
    family, level = level_label.split("_")
    return family, level


def _plot_aggregate(level_data: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]], out_png: Path) -> None:
    fig, (ax0, ax1) = plt.subplots(2, 1, figsize=(11.0, 8.0), sharex=True)

    for level_label in LEVELS:
        t, train, test = level_data[level_label]
        diff = test - train
        ratio = np.divide(test, train, out=np.full_like(test, np.nan), where=np.abs(train) > 1e-12)

        family, level = _split_family_level(level_label)
        color = LEVEL_COLORS[level]
        style = FAMILY_STYLE[family]
        legend_name = f"{family} {level}"

        ax0.plot(
            t,
            diff,
            color=color,
            linewidth=1.8,
            linestyle=style["linestyle"],
            marker=style["marker"],
            markersize=3.0,
            markevery=6,
            label=legend_name,
        )
        ax1.plot(
            t,
            ratio,
            color=color,
            linewidth=1.8,
            linestyle=style["linestyle"],
            marker=style["marker"],
            markersize=3.0,
            markevery=6,
            label=legend_name,
        )

    ax0.axhline(0.0, color="black", linestyle=":", linewidth=1.0, alpha=0.8)
    ax1.axhline(1.0, color="black", linestyle=":", linewidth=1.0, alpha=0.8)

    ax0.set_title("Aggregate trajectories: color = level, line style = family")
    ax0.set_ylabel("test - train")
    ax1.set_ylabel("test / train (raw)")
    ax1.set_xlabel("checkpoint")

    ax0.set_xscale("log")
    ax1.set_xscale("log")

    ax0.grid(True, alpha=0.3)
    ax1.grid(True, alpha=0.3)

    ax1.ticklabel_format(axis="y", style="plain", useOffset=False)
    ax1.yaxis.set_major_formatter(FormatStrFormatter("%.6f"))

    handles, labels = ax0.get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4, frameon=True, bbox_to_anchor=(0.5, 1.0))

    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.94))
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--saves-root",
        type=Path,
        default=Path(__file__).resolve().parents[2] / "Saves_new",
        help="Root directory containing PCA_L*/Wavelet_L* folders (default: wdmdm/Experiments/Saves_new)",
    )
    parser.add_argument("--time", type=int, default=100, help="Diffusion time used in the filename")
    parser.add_argument("--n-train", type=int, default=1024, help="N_train used in the filename")
    parser.add_argument(
        "--aggregate-name",
        type=str,
        default="aggregate_test_train_cosine_diff_ratio_time_{time}_N_train_{n_train}.png",
        help="Filename for the aggregate output under Saves_new/Comparisons/",
    )
    args = parser.parse_args()

    outputs = []
    level_data: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    for level in LEVELS:
        level_dir = args.saves_root / level
        npz_path = _find_input_npz(level_dir, args.time, args.n_train)

        data = np.load(npz_path)
        t = np.asarray(data["training_times"]).astype(float)
        train = np.asarray(data["all_means_train"]).astype(float)
        test = np.asarray(data["all_means_test"]).astype(float)
        level_data[level] = (t, train, test)

        stem = npz_path.stem.replace("backward_cosine_similarity_metric", "test_train_cosine_diff_ratio")
        out_png = npz_path.with_name(f"{stem}.png")
        _plot_level(npz_path=npz_path, out_png=out_png, label=level)
        outputs.append(out_png)

    aggregate_name = args.aggregate_name.format(time=args.time, n_train=args.n_train)
    aggregate_out = args.saves_root / "Comparisons" / aggregate_name
    _plot_aggregate(level_data=level_data, out_png=aggregate_out)
    outputs.append(aggregate_out)

    print("Generated plots:")
    for out in outputs:
        print(out)


if __name__ == "__main__":
    main()
