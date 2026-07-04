#!/usr/bin/env python3
"""Compose summary figures combining sample grids and staircase plots."""

import argparse
import json
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def parse_csv_ints(raw: str):
    vals = [s.strip() for s in raw.split(",") if s.strip()]
    if not vals:
        raise ValueError("Expected at least one value in a comma-separated list")
    return [int(v) for v in vals]


def load_image(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"Missing image: {path}")
    return plt.imread(path)


def compose_one_method(method: str, stair_img, sample_imgs, sample_ids, out_dir: Path):
    n_samples = len(sample_imgs)
    fig = plt.figure(figsize=(4.5 * max(2, n_samples), 8.2))
    gs = fig.add_gridspec(nrows=2, ncols=max(1, n_samples), height_ratios=[1.05, 1.4], hspace=0.16)

    for i, (sid, img) in enumerate(zip(sample_ids, sample_imgs)):
        ax = fig.add_subplot(gs[0, i])
        ax.imshow(img)
        ax.set_title(f"Sample {sid}")
        ax.axis("off")

    ax2 = fig.add_subplot(gs[1, :])
    ax2.imshow(stair_img)
    ax2.axis("off")
    ax2.set_title("Cross-level staircase with loss overlay")

    fig.suptitle(f"{method}: coarse-to-fine samples + staircase", y=0.99, fontsize=14)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.97))

    for ext in ("png", "pdf"):
        out_path = out_dir / f"summary_{method}.{ext}"
        fig.savefig(out_path, bbox_inches="tight")
        print(f"Saved: {out_path}")
    plt.close(fig)


def compose_combined(method_to_stair, method_to_samples, method_to_sample_ids, out_dir: Path):
    methods = list(method_to_stair.keys())
    n_methods = len(methods)
    fig = plt.figure(figsize=(7.2 * n_methods, 10.2))
    outer = fig.add_gridspec(nrows=1, ncols=n_methods, wspace=0.06)

    for j, method in enumerate(methods):
        samples = method_to_samples[method]
        sample_ids = method_to_sample_ids[method]
        n_samples = len(samples)
        gs = outer[j].subgridspec(nrows=2, ncols=max(1, n_samples), height_ratios=[1.0, 1.45], hspace=0.12)

        for i, (sid, img) in enumerate(zip(sample_ids, samples)):
            ax = fig.add_subplot(gs[0, i])
            ax.imshow(img)
            ax.set_title(f"{method} sample {sid}", fontsize=10)
            ax.axis("off")

        ax2 = fig.add_subplot(gs[1, :])
        ax2.imshow(method_to_stair[method])
        ax2.axis("off")
        ax2.set_title(f"{method} staircase+loss", fontsize=11)

    fig.suptitle("Coarse-to-fine summary: samples and cross-level staircases", y=0.995, fontsize=15)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.985))

    for ext in ("png", "pdf"):
        out_path = out_dir / f"summary_combined.{ext}"
        fig.savefig(out_path, bbox_inches="tight")
        print(f"Saved: {out_path}")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser("Compose summary figures from coarse-to-fine samples and staircases.")
    parser.add_argument("--saves_dir", type=str, required=True, help="Path to Saves_new")
    parser.add_argument("--methods", type=str, default="PCA,Wavelet")
    parser.add_argument("--sample_ids", type=str, default="0,1")
    parser.add_argument("--samples_dirname", type=str, default="CoarseFineInference")
    parser.add_argument("--cross_dirname", type=str, default="CrossLevel")
    parser.add_argument("--stair_stem", type=str, default="cross_level_staircase")
    parser.add_argument("--out_dir", type=str, default=None)
    args = parser.parse_args()

    saves_dir = Path(os.path.expanduser(args.saves_dir))
    out_dir = Path(os.path.expanduser(args.out_dir)) if args.out_dir else saves_dir / "SummaryFigures"
    out_dir.mkdir(parents=True, exist_ok=True)

    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    sample_ids = parse_csv_ints(args.sample_ids)

    method_to_stair = {}
    method_to_samples = {}
    method_to_sample_ids = {}

    for method in methods:
        stair_path = saves_dir / args.cross_dirname / f"{args.stair_stem}_{method}.png"
        method_to_stair[method] = load_image(stair_path)

        samples = []
        for sid in sample_ids:
            s_path = saves_dir / args.samples_dirname / method / f"showcase_{method}_sample{sid}.png"
            samples.append(load_image(s_path))
        method_to_samples[method] = samples
        method_to_sample_ids[method] = sample_ids

        compose_one_method(
            method=method,
            stair_img=method_to_stair[method],
            sample_imgs=samples,
            sample_ids=sample_ids,
            out_dir=out_dir,
        )

    compose_combined(method_to_stair, method_to_samples, method_to_sample_ids, out_dir)

    meta = {
        "methods": methods,
        "sample_ids": sample_ids,
        "samples_dirname": args.samples_dirname,
        "cross_dirname": args.cross_dirname,
        "stair_stem": args.stair_stem,
        "args": vars(args),
    }
    with open(out_dir / "summary_meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    print(f"Saved: {out_dir / 'summary_meta.json'}")


if __name__ == "__main__":
    main()
