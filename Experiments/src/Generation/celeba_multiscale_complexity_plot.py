#!/usr/bin/env python3
"""Build the CelebA multiscale complexity figure from one evaluation root."""

import argparse
import csv
import json
import os
import shutil
from pathlib import Path

_cache_root = os.environ.get("TMPDIR", "/tmp")
os.environ.setdefault("MPLCONFIGDIR", os.path.join(_cache_root, "matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", os.path.join(_cache_root, "xdg-cache"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np


COLORS = {0: "#CC79A7", 1: "#0072B2", 2: "#009E73"}
LOSS_COLOR = "red"
LABELS = {0: "L3 vs L0", 1: "L3 vs L1", 2: "L3 vs L2"}


def parse_csv_ints(raw: str):
    vals = [s.strip() for s in raw.split(",") if s.strip()]
    if not vals:
        raise ValueError("Expected at least one comma-separated integer")
    return [int(v) for v in vals]


def load_cross_curve(result_root: Path, method: str, level: int, time: int):
    files = sorted(
        (result_root / "CrossLevel").glob(
            f"{method}_index*/cross_scores_L3_vs_L{level}_t{time}.npz"
        )
    )
    if not files:
        fallback = result_root / "CrossLevel" / method / f"cross_scores_L3_vs_L{level}_t{time}.npz"
        if fallback.exists():
            files = [fallback]
    if not files:
        raise FileNotFoundError(f"No cross-level files found for {method} L{level} t={time} in {result_root}")

    curves = []
    ckpt_ref = None
    ckpt_b = []
    sources = []
    source_args = []
    for fp in files:
        d = np.load(fp, allow_pickle=True)
        ckpt = np.asarray(d["ckpt_ids_a"], dtype=np.int64)
        means = np.asarray(d["means_cos"] if "means_cos" in d else d["means"], dtype=np.float64)
        if ckpt_ref is None:
            ckpt_ref = ckpt
        elif not np.array_equal(ckpt_ref, ckpt):
            raise ValueError(f"Checkpoint mismatch in {fp}")
        curves.append(means)
        ckpt_b.append(int(np.asarray(d["ckpt_b"]).reshape(-1)[0]))
        sources.append(str(fp))
        if "args_json" in d:
            source_args.append(json.loads(str(np.asarray(d["args_json"]).item())))

    curves = np.stack(curves, axis=0)
    if curves.shape[0] == 1:
        mean = curves[0]
        sem = np.zeros_like(mean)
    else:
        mean = np.nanmean(curves, axis=0)
        sem = np.nanstd(curves, axis=0, ddof=1) / np.sqrt(curves.shape[0])

    return {
        "checkpoints": ckpt_ref,
        "mean": mean,
        "sem": sem,
        "ckpt_b": int(np.median(ckpt_b)),
        "ckpt_b_all": ckpt_b,
        "n_indices": int(curves.shape[0]),
        "sources": sources,
        "source_args": source_args,
    }


def find_loss_file(loss_dir: Path, method: str, index: int):
    pat = f"timing_loss_avg_over_timesteps_vs_checkpoint_CelebA32_1024_32_Adam_512_0.0001_index{index}_{method}_L3*.npz"
    matches = sorted(loss_dir.glob(pat))
    if not matches:
        return None
    non_quick = [m for m in matches if "_quick" not in m.name]
    return non_quick[0] if non_quick else matches[0]


def load_loss_curve(result_root: Path, method: str):
    loss_dir = result_root / "Losses_over_checkpoints_timing"
    curves = []
    ckpt_ref = None
    sources = []
    min_ckpts = []
    for index in (0, 1):
        fp = find_loss_file(loss_dir, method, index)
        if fp is None:
            continue
        d = np.load(fp, allow_pickle=True)
        if "test_loss_means" not in d or np.asarray(d["test_loss_means"]).size == 0:
            continue
        ckpt = np.asarray(d["training_times"], dtype=np.int64)
        vals = np.asarray(d["test_loss_means"], dtype=np.float64)
        if ckpt_ref is None:
            ckpt_ref = ckpt
        elif not np.array_equal(ckpt_ref, ckpt):
            raise ValueError(f"Loss checkpoint mismatch in {fp}")
        curves.append(vals)
        sources.append(str(fp))
        min_ckpts.append(int(np.asarray(d["min_checkpoint"]).reshape(-1)[0]) if "min_checkpoint" in d else int(ckpt[np.argmin(vals)]))

    if not curves:
        return None
    curves = np.stack(curves, axis=0)
    return {
        "checkpoints": ckpt_ref,
        "mean": np.nanmean(curves, axis=0),
        "min_checkpoint": int(np.median(min_ckpts)),
        "sources": sources,
        "n_indices": int(curves.shape[0]),
    }


def collect_data(result_root: Path, methods, levels, time):
    data = {}
    for method in methods:
        data[method] = {
            "levels": {level: load_cross_curve(result_root, method, level, time) for level in levels},
            "loss": load_loss_curve(result_root, method),
        }
    return data


def inspect_data(data, levels):
    report = {}
    for method, method_data in data.items():
        ckpt = method_data["levels"][levels[0]]["checkpoints"]
        level_means = {level: method_data["levels"][level]["mean"] for level in levels}
        if all(level in level_means for level in (0, 1, 2)):
            ordered = (level_means[2] > level_means[1]) & (level_means[1] > level_means[0])
            ordered_fraction = float(np.mean(ordered))
            spread = level_means[2] - level_means[0]
            median_spread = float(np.median(spread))
        else:
            ordered_fraction = np.nan
            median_spread = np.nan

        all_vals = np.concatenate([level_means[level] for level in levels])
        report[method] = {
            "ordered_fraction": ordered_fraction,
            "median_L2_minus_L0": median_spread,
            "min_similarity": float(np.min(all_vals)),
            "max_similarity": float(np.max(all_vals)),
            "range_similarity": float(np.max(all_vals) - np.min(all_vals)),
            "first_checkpoint": int(ckpt[0]),
            "last_checkpoint": int(ckpt[-1]),
        }
    return report


def plot(data, methods, levels, out_dir: Path, out_stem: str, title: str | None, use_tex: bool = False):
    rc_params = {
        "text.usetex": use_tex,
        "figure.dpi": 140,
        "savefig.dpi": 300,
        "font.size": 12,
        "axes.labelsize": 12,
        "axes.titlesize": 12,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "legend.fontsize": 10.5,
        "legend.frameon": False,
    }
    if use_tex:
        rc_params["text.latex.preamble"] = r"\usepackage{amsfonts}"
    plt.rcParams.update(rc_params)

    fig, axes = plt.subplots(1, len(methods), figsize=(9.6, 4.2), sharey=True)
    if len(methods) == 1:
        axes = [axes]

    all_vals = []
    all_loss_vals = []
    x_min, x_max = np.inf, -np.inf
    for method in methods:
        for level in levels:
            curve = data[method]["levels"][level]
            ckpt = curve["checkpoints"]
            mean = curve["mean"]
            sem = curve["sem"]
            all_vals.append(mean - sem)
            all_vals.append(mean + sem)
            x_min = min(x_min, float(np.min(ckpt)))
            x_max = max(x_max, float(np.max(ckpt)))
        loss = data[method]["loss"]
        if loss is not None:
            all_loss_vals.append(loss["mean"])

    y_all = np.concatenate(all_vals)
    y_pad = 0.04 * max(float(np.max(y_all) - np.min(y_all)), 1e-6)
    y_lim = (max(0.0, float(np.min(y_all) - y_pad)), min(1.005, float(np.max(y_all) + y_pad)))
    if all_loss_vals:
        loss_all = np.concatenate(all_loss_vals)
        loss_pad = 0.06 * max(float(np.max(loss_all) - np.min(loss_all)), 1e-6)
        loss_lim = (float(np.min(loss_all) - loss_pad), float(np.max(loss_all) + loss_pad))
    else:
        loss_lim = None

    for ax_i, (ax, method) in enumerate(zip(axes, methods)):
        for level in levels:
            curve = data[method]["levels"][level]
            ckpt = curve["checkpoints"]
            mean = curve["mean"]
            sem = curve["sem"]
            color = COLORS[level]
            ax.plot(
                ckpt,
                mean,
                marker="o",
                markersize=3,
                markerfacecolor=color,
                color=color,
                linewidth=2,
                alpha=0.92,
                label=LABELS[level],
                zorder=3,
            )
            if np.any(sem > 0):
                ax.fill_between(ckpt, mean - sem, mean + sem, alpha=0.14, color=color, linewidth=0, zorder=1)

            nearest = int(np.argmin(np.abs(ckpt - curve["ckpt_b"])))
            ax.scatter(
                [ckpt[nearest]],
                [mean[nearest]],
                marker="D",
                s=24,
                color=color,
                edgecolor="black",
                linewidth=0.45,
                zorder=8,
            )

        loss = data[method]["loss"]
        if loss is not None:
            ax2 = ax.twinx()
            ax2.plot(
                loss["checkpoints"],
                loss["mean"],
                marker="d",
                markersize=3,
                markerfacecolor=LOSS_COLOR,
                color=LOSS_COLOR,
                linestyle="--",
                linewidth=2,
                alpha=0.9,
                zorder=2,
            )
            if loss_lim is not None:
                ax2.set_ylim(loss_lim)
            if ax_i == len(methods) - 1:
                ax2.set_ylabel("DSM test loss", color=LOSS_COLOR)
                ax2.tick_params(axis="y", labelcolor=LOSS_COLOR, which="both")
            else:
                ax2.tick_params(axis="y", right=False, labelright=False)
                ax2.spines["right"].set_visible(False)

        ax.set_xscale("log")
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_lim)
        ax.set_xlabel("L3 checkpoint")
        if ax_i == 0:
            ax.set_ylabel("Cosine similarity (noise predictions)")
        ax.set_title("Haar wavelet" if method == "Wavelet" else "PCA")

    handles = [
        Line2D([0], [0], color=COLORS[level], marker="o", lw=2, markersize=3, label=LABELS[level])
        for level in levels
    ]
    handles.append(Line2D([0], [0], color=LOSS_COLOR, marker="d", ls="--", lw=2, markersize=3, label="L3 test loss"))
    handles.append(Line2D([0], [0], color="black", marker="D", lw=0, markersize=5, label="selected Lk checkpoint"))
    fig.legend(
        handles=handles,
        loc="lower center",
        ncol=min(5, len(handles)),
        bbox_to_anchor=(0.5, 0.025),
        handletextpad=0.5,
        borderaxespad=0.0,
        handlelength=1.4,
        columnspacing=0.9,
        frameon=False,
    )
    if title:
        fig.suptitle(title, y=0.98)
    fig.tight_layout(rect=(0.0, 0.095, 1.0, 0.94 if title else 1.0))

    out_dir.mkdir(parents=True, exist_ok=True)
    paths = {}
    for ext in ("png", "pdf"):
        fp = out_dir / f"{out_stem}.{ext}"
        fig.savefig(fp, bbox_inches="tight")
        paths[ext] = str(fp)
        print(f"Saved: {fp}")
    plt.close(fig)
    return paths


def save_tables(data, report, methods, levels, out_dir: Path, out_stem: str, args):
    csv_path = out_dir / f"{out_stem}.csv"
    npz_path = out_dir / f"{out_stem}.npz"
    meta = {
        "args": vars(args),
        "independent_evaluation_root": True,
        "uncertainty": "SEM across available split indices; no shaded band is shown when only one index is available.",
        "inspection": report,
        "sources": {
            method: {
                "levels": {str(level): data[method]["levels"][level]["sources"] for level in levels},
                "loss": [] if data[method]["loss"] is None else data[method]["loss"]["sources"],
            }
            for method in methods
        },
        "cross_level_args": {
            method: {
                str(level): data[method]["levels"][level]["source_args"]
                for level in levels
            }
            for method in methods
        },
        "frozen_checkpoints": {
            method: {
                str(level): data[method]["levels"][level]["ckpt_b_all"]
                for level in levels
            }
            for method in methods
        },
    }

    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["method", "level", "checkpoint", "mean", "sem", "n_indices", "frozen_checkpoint"])
        for method in methods:
            for level in levels:
                curve = data[method]["levels"][level]
                for ck, mean, sem in zip(curve["checkpoints"], curve["mean"], curve["sem"]):
                    writer.writerow([method, level, int(ck), float(mean), float(sem), curve["n_indices"], curve["ckpt_b"]])

    save = {
        "methods": np.array(methods, dtype=object),
        "levels": np.array(levels, dtype=np.int64),
        "meta_json": np.array(json.dumps(meta, sort_keys=True)),
    }
    for method in methods:
        for level in levels:
            curve = data[method]["levels"][level]
            pfx = f"{method}_L{level}"
            save[f"{pfx}_checkpoints"] = curve["checkpoints"]
            save[f"{pfx}_mean"] = curve["mean"]
            save[f"{pfx}_sem"] = curve["sem"]
            save[f"{pfx}_frozen_checkpoint"] = np.array(curve["ckpt_b"], dtype=np.int64)
    np.savez(npz_path, **save)
    print(f"Saved: {csv_path}")
    print(f"Saved: {npz_path}")


def main():
    parser = argparse.ArgumentParser("Create the CelebA multiscale complexity figure.")
    parser.add_argument("--result-root", type=str, required=True)
    parser.add_argument("--methods", type=str, default="PCA,Wavelet")
    parser.add_argument("--levels", type=str, default="0,1,2")
    parser.add_argument("--time", type=int, default=100)
    parser.add_argument("--out-dir", type=str, default=None)
    parser.add_argument("--out-stem", type=str, default="celeba_wavelet_pca_complexity")
    parser.add_argument("--title", type=str, default=None)
    parser.add_argument("--use-tex", action="store_true", help="Render labels with an external LaTeX installation.")
    parser.add_argument("--copy-to-figs", type=str, default=None)
    args = parser.parse_args()

    result_root = Path(os.path.expanduser(args.result_root))
    methods = [x.strip() for x in args.methods.split(",") if x.strip()]
    levels = parse_csv_ints(args.levels)
    out_dir = Path(os.path.expanduser(args.out_dir)) if args.out_dir else result_root / "MultiscaleComplexity"

    data = collect_data(result_root, methods, levels, int(args.time))
    report = inspect_data(data, levels)
    paths = plot(data, methods, levels, out_dir, args.out_stem, args.title, use_tex=args.use_tex)
    save_tables(data, report, methods, levels, out_dir, args.out_stem, args)

    print("\nInspection summary:")
    print(json.dumps(report, indent=2, sort_keys=True))

    if args.copy_to_figs is not None:
        figs_dir = Path(os.path.expanduser(args.copy_to_figs))
        figs_dir.mkdir(parents=True, exist_ok=True)
        for ext, src in paths.items():
            dst = figs_dir / f"{args.out_stem}.{ext}"
            shutil.copy2(src, dst)
            print(f"Copied: {dst}")


if __name__ == "__main__":
    main()
