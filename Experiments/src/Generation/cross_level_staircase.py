#!/usr/bin/env python3
"""Aggregate cross-level sweeps into staircase plots.

Supports:
- multiple metrics saved by compare_scores_cross.py (cosine + MSE),
- multi-time overlays (e.g. t=100,200,400),
- optional loss overlays from Losses_over_checkpoints_timing,
- combined 2-panel figure and per-method single-panel figures.
"""

import argparse
import json
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np


plt.rcParams.update(
    {
        "figure.dpi": 140,
        "savefig.dpi": 200,
        "font.size": 11,
        "axes.labelsize": 12,
        "axes.titlesize": 12,
        "legend.frameon": False,
        "grid.alpha": 0.3,
    }
)


def parse_csv_ints(raw: str):
    vals = [s.strip() for s in raw.split(",") if s.strip()]
    if not vals:
        raise ValueError("Expected at least one value in a comma-separated list.")
    return [int(v) for v in vals]


def pick_metric_arrays(d: np.lib.npyio.NpzFile, metric: str):
    if metric == "cosine":
        if "means_cos" in d and "sems_cos" in d:
            return np.asarray(d["means_cos"]), np.asarray(d["sems_cos"]), "cosine"
        if "means" in d and "sems" in d:
            return np.asarray(d["means"]), np.asarray(d["sems"]), "cosine"
        raise KeyError("cosine metric requested but means_cos/sems_cos (or means/sems) missing")

    if metric == "mse":
        if "means_mse" in d and "sems_mse" in d:
            return np.asarray(d["means_mse"]), np.asarray(d["sems_mse"]), "mse"
        raise KeyError("mse metric requested but means_mse/sems_mse missing")

    # auto: prefer mse if present, else cosine.
    if "means_mse" in d and "sems_mse" in d:
        return np.asarray(d["means_mse"]), np.asarray(d["sems_mse"]), "mse"
    if "means_cos" in d and "sems_cos" in d:
        return np.asarray(d["means_cos"]), np.asarray(d["sems_cos"]), "cosine"
    if "means" in d and "sems" in d:
        return np.asarray(d["means"]), np.asarray(d["sems"]), "cosine"
    raise KeyError("No compatible metric arrays found in NPZ")


def load_curve(path: Path, metric: str):
    if not path.exists():
        raise FileNotFoundError(f"Missing cross-level file: {path}")

    d = np.load(path, allow_pickle=True)
    needed = {"ckpt_ids_a", "ckpt_b"}
    if not needed.issubset(set(d.files)):
        missing = sorted(list(needed.difference(set(d.files))))
        raise KeyError(f"{path} missing required keys: {missing}")

    ckpt_a = np.asarray(d["ckpt_ids_a"]).reshape(-1).astype(np.int64)
    ckpt_b = int(np.asarray(d["ckpt_b"]).reshape(-1)[0])
    means, sems, metric_name = pick_metric_arrays(d, metric)
    means = np.asarray(means).reshape(-1).astype(np.float64)
    sems = np.asarray(sems).reshape(-1).astype(np.float64)

    if not (ckpt_a.shape == means.shape == sems.shape):
        raise ValueError(
            f"Inconsistent shapes in {path}: ckpt={ckpt_a.shape}, means={means.shape}, sems={sems.shape}"
        )

    order = np.argsort(ckpt_a)
    ckpt_a = ckpt_a[order]
    means = means[order]
    sems = sems[order]

    if not (np.isfinite(means).all() and np.isfinite(sems).all()):
        raise ValueError(f"Non-finite values found in {path}")

    time_val = None
    if "time" in d:
        time_val = int(np.asarray(d["time"]).reshape(-1)[0])

    return {
        "ckpt_ids_a": ckpt_a,
        "ckpt_b": ckpt_b,
        "means": means,
        "sems": sems,
        "metric": metric_name,
        "time": time_val,
        "source": str(path),
    }


def find_loss_npz(loss_dir: Path, method: str, index: int):
    pat = f"timing_loss_avg_over_timesteps_vs_checkpoint_CelebA32_1024_32_Adam_512_0.0001_index{index}_{method}_L3*.npz"
    matches = sorted(loss_dir.glob(pat))
    if not matches:
        return None
    non_quick = [m for m in matches if "_quick" not in m.name]
    return non_quick[0] if non_quick else matches[0]


def load_loss_curve(loss_dir: Path, method: str, index: int):
    fp = find_loss_npz(loss_dir, method, index)
    if fp is None:
        return None

    d = np.load(fp, allow_pickle=True)
    if "training_times" not in d or "test_loss_means" not in d:
        return None

    out = {
        "training_times": np.asarray(d["training_times"]).reshape(-1).astype(np.int64),
        "test_loss_means": np.asarray(d["test_loss_means"]).reshape(-1).astype(np.float64),
        "source": str(fp),
    }
    if "min_checkpoint" in d:
        out["min_checkpoint"] = int(np.asarray(d["min_checkpoint"]).reshape(-1)[0])
    else:
        i = int(np.argmin(out["test_loss_means"]))
        out["min_checkpoint"] = int(out["training_times"][i])
    return out


def method_limits(level_curves: dict):
    x_min = None
    x_max = None
    y_min = None
    y_max = None

    for t_data in level_curves.values():
        for data in t_data.values():
            ckpt = np.asarray(data["ckpt_ids_a"]).reshape(-1)
            means = np.asarray(data["means"]).reshape(-1)

            cur_x_min = float(np.min(ckpt))
            cur_x_max = float(np.max(ckpt))
            cur_y_min = float(np.min(means))
            cur_y_max = float(np.max(means))

            x_min = cur_x_min if x_min is None else min(x_min, cur_x_min)
            x_max = cur_x_max if x_max is None else max(x_max, cur_x_max)
            y_min = cur_y_min if y_min is None else min(y_min, cur_y_min)
            y_max = cur_y_max if y_max is None else max(y_max, cur_y_max)

    if x_min is None or x_max is None or y_min is None or y_max is None:
        raise ValueError("Cannot compute axis limits from empty curves")

    if np.isclose(x_min, x_max):
        x_max = x_min + 1.0
    if np.isclose(y_min, y_max):
        y_max = y_min + 1e-6

    return x_min, x_max, y_min, y_max


def build_curve_map(cross_dir: Path, methods, levels, times):
    all_data = {}
    for method in methods:
        method_map = {}
        for level in levels:
            time_map = {}
            if times is None:
                fp = cross_dir / method / f"cross_scores_L3_vs_L{level}.npz"
                time_map[None] = fp
            else:
                for t in times:
                    fp = cross_dir / method / f"cross_scores_L3_vs_L{level}_t{t}.npz"
                    time_map[int(t)] = fp
            method_map[level] = time_map
        all_data[method] = method_map
    return all_data


def plot_method(
    ax,
    method: str,
    method_curves: dict,
    xscale: str,
    metric_label: str,
    overlay_loss: dict | None,
    style_by_time: dict,
):
    level_colors = {0: "#d62728", 1: "#1f77b4", 2: "#9467bd"}
    plotted_markers = set()

    ax_loss = None
    if overlay_loss is not None:
        ax_loss = ax.twinx()
        ax_loss.plot(
            overlay_loss["training_times"],
            overlay_loss["test_loss_means"],
            color="#222222",
            linestyle="--",
            linewidth=1.5,
            alpha=0.80,
            label="L3 test loss",
        )
        ax_loss.axvline(
            overlay_loss["min_checkpoint"],
            color="#222222",
            linestyle=":",
            linewidth=1.0,
            alpha=0.75,
        )
        ax_loss.set_ylabel("Test loss (MSE)", color="#222222")
        ax_loss.tick_params(axis="y", labelcolor="#222222")

    for level in sorted(method_curves):
        for time_key in sorted(method_curves[level], key=lambda x: -1 if x is None else x):
            data = method_curves[level][time_key]
            ckpt = data["ckpt_ids_a"]
            means = data["means"]
            sems = data["sems"]
            ckpt_b = int(data["ckpt_b"])

            color = level_colors.get(level, None)
            style = style_by_time.get(time_key, "-")
            ax.plot(
                ckpt,
                means,
                color=color,
                linewidth=2.0,
                linestyle=style,
                alpha=0.95,
            )
            ax.fill_between(ckpt, means - sems, means + sems, color=color, alpha=0.08)

            marker_key = (level, ckpt_b)
            if marker_key not in plotted_markers:
                idx = np.where(ckpt == ckpt_b)[0]
                if idx.size == 0:
                    nearest = int(np.argmin(np.abs(ckpt - ckpt_b)))
                    x_m = int(ckpt[nearest])
                    y_m = float(means[nearest])
                else:
                    i = int(idx[0])
                    x_m = int(ckpt[i])
                    y_m = float(means[i])
                ax.scatter([x_m], [y_m], marker="D", s=34, color=color, zorder=4)
                ax.axvline(ckpt_b, linestyle=":", linewidth=0.9, color=color, alpha=0.45)
                plotted_markers.add(marker_key)

    ax.set_title(method)
    ax.set_xlabel("L3 checkpoint (swept model)")
    ax.set_ylabel(metric_label)
    if xscale == "log":
        ax.set_xscale("log")
    ax.grid(True, linestyle=":", alpha=0.4)
    return ax_loss


def legend_handles(levels, times, overlay_losses):
    level_colors = {0: "#d62728", 1: "#1f77b4", 2: "#9467bd"}
    out = []

    for level in levels:
        out.append(Line2D([0], [0], color=level_colors.get(level, "#444444"), lw=2.0, label=f"L3 vs L{level}"))

    if times is not None and len(times) > 1:
        style_cycle = ["-", "--", ":", "-."]
        for i, t in enumerate(times):
            out.append(Line2D([0], [0], color="#666666", lw=2.0, linestyle=style_cycle[i % len(style_cycle)], label=f"t={t}"))

    out.append(Line2D([0], [0], color="#333333", marker="D", lw=0, markersize=6, label="frozen checkpoint"))

    if overlay_losses:
        out.append(Line2D([0], [0], color="#222222", lw=1.5, linestyle="--", label="L3 test loss"))

    return out


def save_single_method_figure(
    out_dir: Path,
    out_stem: str,
    method: str,
    method_curves: dict,
    xscale: str,
    metric_label: str,
    overlay_loss: dict | None,
    style_by_time: dict,
    levels: list[int],
    times: list[int] | None,
    overlay_losses: bool,
):
    fig, ax = plt.subplots(figsize=(8.0, 4.8))
    plot_method(
        ax=ax,
        method=method,
        method_curves=method_curves,
        xscale=xscale,
        metric_label=metric_label,
        overlay_loss=overlay_loss,
        style_by_time=style_by_time,
    )
    x_min, x_max, y_min, y_max = method_limits(method_curves)
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)

    h = legend_handles(levels=levels, times=times, overlay_losses=overlay_losses)
    fig.legend(handles=h, loc="lower center", ncol=min(6, max(2, len(h))))
    fig.tight_layout(rect=(0.0, 0.12, 1.0, 1.0))

    for ext in ("png", "pdf"):
        out_plot = out_dir / f"{out_stem}_{method}.{ext}"
        fig.savefig(out_plot, bbox_inches="tight")
        print(f"Saved plot: {out_plot}")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser("Create two-panel cross-level staircase plot.")
    parser.add_argument("--saves_dir", type=str, required=True, help="Path to Saves_new.")
    parser.add_argument("--methods", type=str, default="PCA,Wavelet")
    parser.add_argument("--levels", type=str, default="0,1,2")
    parser.add_argument("--times", type=str, default=None,
                        help="Optional comma-separated t values, e.g. '100,200,400'.")
    parser.add_argument("--metric", type=str, choices=["auto", "cosine", "mse"], default="auto")
    parser.add_argument("--xscale", type=str, choices=["linear", "log"], default="log")
    parser.add_argument("--overlay_losses", action="store_true")
    parser.add_argument("--loss_dir", type=str, default=None,
                        help="Defaults to <saves_dir>/Losses_over_checkpoints_timing")
    parser.add_argument("--loss_index", type=int, default=0)
    parser.add_argument("--out_dir", type=str, default=None,
                        help="Output directory for aggregate figure. Defaults to <saves_dir>/CrossLevel")
    parser.add_argument("--out_stem", type=str, default="cross_level_staircase")
    parser.add_argument("--title", type=str,
                        default="Cross-level staircase: L3 sweep against fixed Lk")
    parser.add_argument("--save_per_method", action="store_true",
                        help="Also save one figure per method.")
    args = parser.parse_args()

    saves_dir = Path(os.path.expanduser(args.saves_dir))
    cross_dir = saves_dir / "CrossLevel"
    out_dir = Path(os.path.expanduser(args.out_dir)) if args.out_dir else cross_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    if not methods:
        raise ValueError("--methods produced an empty list")

    levels = parse_csv_ints(args.levels)
    times = parse_csv_ints(args.times) if args.times is not None else None

    style_cycle = ["-", "--", ":", "-."]
    style_by_time = {None: "-"}
    if times is not None:
        style_by_time = {int(t): style_cycle[i % len(style_cycle)] for i, t in enumerate(times)}

    raw_map = build_curve_map(cross_dir, methods, levels, times)

    all_data = {}
    metric_used = None
    for method in methods:
        method_map = {}
        for level in levels:
            level_map = {}
            for time_key, fp in raw_map[method][level].items():
                curve = load_curve(fp, metric=args.metric)
                level_map[time_key] = curve
                if metric_used is None:
                    metric_used = curve["metric"]
            method_map[level] = level_map
        all_data[method] = method_map

    if metric_used == "mse":
        y_label = "MSE of score predictions"
    else:
        y_label = "Cosine similarity (score predictions)"

    loss_dir = Path(os.path.expanduser(args.loss_dir)) if args.loss_dir else saves_dir / "Losses_over_checkpoints_timing"
    loss_by_method = {}
    for method in methods:
        if args.overlay_losses:
            loss_by_method[method] = load_loss_curve(loss_dir, method=method, index=int(args.loss_index))
        else:
            loss_by_method[method] = None

    fig, axes = plt.subplots(1, len(methods), figsize=(7.6 * len(methods), 5.0), sharey=False)
    if len(methods) == 1:
        axes = [axes]

    for i, (ax, method) in enumerate(zip(axes, methods)):
        plot_method(
            ax=ax,
            method=method,
            method_curves=all_data[method],
            xscale=args.xscale,
            metric_label=y_label,
            overlay_loss=loss_by_method[method],
            style_by_time=style_by_time,
        )

        if i == 0:
            x_min, x_max, y_min, y_max = method_limits(all_data[method])
            ax.set_xlim(x_min, x_max)
            ax.set_ylim(y_min, y_max)

    h = legend_handles(levels=levels, times=times, overlay_losses=args.overlay_losses)
    fig.legend(handles=h, loc="lower center", ncol=min(6, max(2, len(h))))
    fig.suptitle(args.title, y=0.98)
    fig.tight_layout(rect=(0.0, 0.10, 1.0, 0.95))

    for ext in ("png", "pdf"):
        out_plot = out_dir / f"{args.out_stem}.{ext}"
        fig.savefig(out_plot, bbox_inches="tight")
        print(f"Saved plot: {out_plot}")
    plt.close(fig)

    if args.save_per_method:
        for method in methods:
            save_single_method_figure(
                out_dir=out_dir,
                out_stem=args.out_stem,
                method=method,
                method_curves=all_data[method],
                xscale=args.xscale,
                metric_label=y_label,
                overlay_loss=loss_by_method[method],
                style_by_time=style_by_time,
                levels=levels,
                times=times,
                overlay_losses=args.overlay_losses,
            )

    save_dict = {
        "methods": np.array(methods, dtype=object),
        "levels": np.array(levels, dtype=np.int64),
        "times": np.array([-1] if times is None else times, dtype=np.int64),
        "metric": np.array(metric_used if metric_used is not None else args.metric),
        "xscale": np.array(args.xscale),
        "title": np.array(args.title),
        "meta_json": np.array(json.dumps({"out_stem": args.out_stem, "overlay_losses": bool(args.overlay_losses)}, sort_keys=True)),
    }

    for method in methods:
        for level in levels:
            for time_key, d in all_data[method][level].items():
                t_tag = "tbase" if time_key is None else f"t{int(time_key)}"
                pfx = f"{method}_L{level}_{t_tag}"
                save_dict[f"{pfx}_ckpt_ids_a"] = d["ckpt_ids_a"]
                save_dict[f"{pfx}_ckpt_b"] = np.array(d["ckpt_b"], dtype=np.int64)
                save_dict[f"{pfx}_means"] = d["means"]
                save_dict[f"{pfx}_sems"] = d["sems"]

    out_npz = out_dir / f"{args.out_stem}_data.npz"
    np.savez(out_npz, **save_dict)
    print(f"Saved data: {out_npz}")


if __name__ == "__main__":
    main()
