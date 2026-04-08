#!/usr/bin/env python3
"""Compare PCA staircase tracks with separated time panels and loss overlays.

Track A example: 16-64-128 (new)
Track B example: 64-128-512 (reference)

Produces a two-panel figure:
- left: t=100 only
- right: t=200 and t=400
Both panels overlay L3 test loss on a secondary y-axis.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np


def parse_csv_ints(raw: str) -> list[int]:
    vals = [s.strip() for s in raw.split(",") if s.strip()]
    if not vals:
        raise ValueError("Empty csv integer list")
    return [int(v) for v in vals]


def load_curve(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")
    d = np.load(path, allow_pickle=True)
    ckpt = np.asarray(d["ckpt_ids_a"]).reshape(-1).astype(np.int64)
    means = np.asarray(d["means"]).reshape(-1).astype(np.float64)
    sems = np.asarray(d["sems"]).reshape(-1).astype(np.float64)
    ckpt_b = int(np.asarray(d["ckpt_b"]).reshape(-1)[0])

    order = np.argsort(ckpt)
    ckpt = ckpt[order]
    means = means[order]
    sems = sems[order]

    return {"ckpt": ckpt, "means": means, "sems": sems, "ckpt_b": ckpt_b}


def find_loss_curve(saves_dir: Path, method_l3: str, loss_index: int):
    loss_dir = saves_dir / "Losses_over_checkpoints_timing"
    pat = f"timing_loss_avg_over_timesteps_vs_checkpoint_CelebA32_1024_32_Adam_512_0.0001_index{loss_index}_{method_l3}_L3*.npz"
    files = sorted(loss_dir.glob(pat))
    if not files:
        return None
    non_quick = [f for f in files if "_quick" not in f.name]
    fp = non_quick[0] if non_quick else files[0]
    d = np.load(fp, allow_pickle=True)
    return {
        "ckpt": np.asarray(d["training_times"]).reshape(-1).astype(np.int64),
        "loss": np.asarray(d["test_loss_means"]).reshape(-1).astype(np.float64),
        "min_ckpt": int(np.asarray(d["min_checkpoint"]).reshape(-1)[0]) if "min_checkpoint" in d else None,
    }


def resolve_file(dir_path: Path, level: int, t: int, strict_timed: bool):
    timed = dir_path / f"cross_scores_L3_vs_L{level}_t{t}.npz"
    if timed.exists():
        return timed

    base = dir_path / f"cross_scores_L3_vs_L{level}.npz"
    if t == 100 and base.exists() and not strict_timed:
        return base

    raise FileNotFoundError(f"Missing timed curve for level L{level} at t={t} in {dir_path}")


def main():
    parser = argparse.ArgumentParser("Compare two PCA staircase tracks")
    parser.add_argument("--saves_dir", type=str, required=True)
    parser.add_argument("--track_a", type=str, default="PCA_16_64_128_1024")
    parser.add_argument("--track_b", type=str, default="PCA_64_128_512")
    parser.add_argument("--track_a_label", type=str, default="16-64-128")
    parser.add_argument("--track_b_label", type=str, default="64-128-512")
    parser.add_argument("--times", type=str, default="100,200,400")
    parser.add_argument("--levels", type=str, default="0,1,2")
    parser.add_argument("--method_l3", type=str, default="PCA")
    parser.add_argument("--loss_index", type=int, default=0)
    parser.add_argument("--out_dir", type=str, default=None)
    parser.add_argument("--out_stem", type=str, default="pca_staircase_16_64_128_vs_64_128_512")
    parser.add_argument("--strict_timed", action="store_true")
    args = parser.parse_args()

    saves_dir = Path(args.saves_dir).expanduser()
    out_dir = Path(args.out_dir).expanduser() if args.out_dir else saves_dir / "CrossLevel"
    out_dir.mkdir(parents=True, exist_ok=True)

    times = parse_csv_ints(args.times)
    levels = parse_csv_ints(args.levels)

    dir_a = saves_dir / "CrossLevel" / args.track_a
    dir_b = saves_dir / "CrossLevel" / args.track_b

    data = {"A": {}, "B": {}}
    for t in times:
        data["A"][t] = {}
        data["B"][t] = {}
        for lvl in levels:
            data["A"][t][lvl] = load_curve(resolve_file(dir_a, lvl, t, args.strict_timed))
            data["B"][t][lvl] = load_curve(resolve_file(dir_b, lvl, t, args.strict_timed))

    loss_curve = find_loss_curve(saves_dir, method_l3=args.method_l3, loss_index=int(args.loss_index))

    level_colors = {0: "#d62728", 1: "#1f77b4", 2: "#9467bd"}
    track_styles = {"A": "-", "B": "--"}
    time_alpha = {100: 0.95, 200: 0.85, 400: 0.75}

    fig, axes = plt.subplots(1, 2, figsize=(16, 6), sharex=True)
    panel_specs = [([100], "t=100"), ([200, 400], "t=200 and t=400")]

    for ax, (panel_times, panel_title) in zip(axes, panel_specs):
        ax_loss = ax.twinx() if loss_curve is not None else None

        if ax_loss is not None:
            ax_loss.plot(
                loss_curve["ckpt"],
                loss_curve["loss"],
                color="#222222",
                linestyle=":",
                linewidth=1.3,
                alpha=0.8,
                label="L3 test loss",
            )
            ax_loss.set_ylabel("L3 test loss (MSE)", color="#222222")
            ax_loss.tick_params(axis="y", labelcolor="#222222")

        for t in panel_times:
            for lvl in levels:
                for track, track_label in (("A", args.track_a_label), ("B", args.track_b_label)):
                    d = data[track][t][lvl]
                    ax.plot(
                        d["ckpt"],
                        d["means"],
                        linestyle=track_styles[track],
                        color=level_colors[lvl],
                        linewidth=2.0,
                        alpha=time_alpha.get(t, 0.9),
                    )
                    ax.fill_between(
                        d["ckpt"],
                        d["means"] - d["sems"],
                        d["means"] + d["sems"],
                        color=level_colors[lvl],
                        alpha=0.06,
                    )
                    idx = int(np.argmin(np.abs(d["ckpt"] - d["ckpt_b"])))
                    ax.scatter(
                        [d["ckpt"][idx]],
                        [d["means"][idx]],
                        marker="D",
                        s=28,
                        color=level_colors[lvl],
                        alpha=time_alpha.get(t, 0.9),
                    )

        ax.set_title(panel_title)
        ax.set_xlabel("L3 checkpoint (swept model)")
        ax.set_ylabel("Cosine similarity (score predictions)")
        ax.set_xscale("log")
        ax.grid(True, linestyle=":", alpha=0.35)

    legend_items = []
    for lvl in levels:
        legend_items.append(Line2D([0], [0], color=level_colors[lvl], lw=2.0, label=f"L3 vs L{lvl}"))
    legend_items.append(Line2D([0], [0], color="#333333", lw=2.0, linestyle="-", label=args.track_a_label))
    legend_items.append(Line2D([0], [0], color="#333333", lw=2.0, linestyle="--", label=args.track_b_label))
    legend_items.append(Line2D([0], [0], color="#333333", marker="D", lw=0, markersize=6, label="frozen checkpoint"))
    if loss_curve is not None:
        legend_items.append(Line2D([0], [0], color="#222222", lw=1.3, linestyle=":", label="L3 test loss"))

    fig.legend(handles=legend_items, loc="lower center", ncol=6)
    fig.suptitle(
        f"PCA staircase comparison: {args.track_a_label} vs {args.track_b_label}",
        y=0.98,
        fontsize=16,
    )
    fig.tight_layout(rect=(0.0, 0.10, 1.0, 0.93))

    for ext in ("png", "pdf"):
        out_fp = out_dir / f"{args.out_stem}.{ext}"
        fig.savefig(out_fp, bbox_inches="tight")
        print(f"Saved plot: {out_fp}")
    plt.close(fig)


if __name__ == "__main__":
    main()
