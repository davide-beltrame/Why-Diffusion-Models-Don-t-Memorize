#!/usr/bin/env python3
"""Select frozen checkpoint IDs for cross-level evaluations.

Policies:
- max_similarity: checkpoint where within-level all_means_test is maximal.
- min_test_loss: checkpoint from timing-loss NPZ min_checkpoint (or argmin fallback).
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Select a checkpoint id from evaluation NPZ files.")
    parser.add_argument(
        "--policy",
        type=str,
        required=True,
        choices=["max_similarity", "min_test_loss"],
    )
    parser.add_argument(
        "--comparisons-dir",
        type=str,
        default=None,
        help="Directory containing backward cosine NPZ files (required for max_similarity).",
    )
    parser.add_argument(
        "--comparison-glob",
        type=str,
        default="backward_cosine_similarity_metric_indexA_vs_B_Npairs_*_time_100_2y_issameFalse_N_train_1024.npz",
        help="Glob pattern used inside --comparisons-dir.",
    )
    parser.add_argument(
        "--loss-file",
        type=str,
        default=None,
        help="Direct path to timing-loss NPZ (optional for min_test_loss).",
    )
    parser.add_argument(
        "--loss-dir",
        type=str,
        default=None,
        help="Directory used with --loss-glob to find timing-loss NPZ (optional for min_test_loss).",
    )
    parser.add_argument(
        "--loss-glob",
        type=str,
        default="timing_loss_avg_over_timesteps_vs_checkpoint_*.npz",
        help="Glob pattern used inside --loss-dir.",
    )
    parser.add_argument(
        "--prefer-non-quick",
        action="store_true",
        help="Prefer candidates without '_quick' in filename when selecting among matches.",
    )
    return parser.parse_args()


def choose_latest(candidates: list[Path], prefer_non_quick: bool) -> Path:
    if not candidates:
        raise ValueError("No candidate files found.")

    candidates = sorted(candidates, key=lambda p: p.stat().st_mtime, reverse=True)
    if prefer_non_quick:
        non_quick = [p for p in candidates if "_quick" not in p.name]
        if non_quick:
            return non_quick[0]
    return candidates[0]


def parse_max_similarity(npz_path: Path) -> int:
    data = np.load(npz_path, allow_pickle=True)
    if "training_times" not in data or "all_means_test" not in data:
        raise KeyError(f"Missing required keys in {npz_path}")

    times = np.asarray(data["training_times"]).reshape(-1).astype(np.int64)
    vals = np.asarray(data["all_means_test"]).reshape(-1).astype(np.float64)
    if times.size == 0 or vals.size == 0 or times.size != vals.size:
        raise ValueError(f"Invalid arrays in {npz_path}")

    max_val = float(np.max(vals))
    tie_idx = np.where(np.isclose(vals, max_val, rtol=0.0, atol=1e-12))[0]
    pick = int(np.min(times[tie_idx]))
    return pick


def resolve_max_similarity(args: argparse.Namespace) -> int:
    if not args.comparisons_dir:
        raise ValueError("--comparisons-dir is required for policy=max_similarity")

    comp_dir = Path(args.comparisons_dir)
    if not comp_dir.exists():
        raise FileNotFoundError(f"Comparisons directory not found: {comp_dir}")

    candidates = list(comp_dir.glob(args.comparison_glob))
    if not candidates:
        raise FileNotFoundError(
            f"No comparison NPZ found in {comp_dir} with pattern {args.comparison_glob}"
        )

    last_error: Exception | None = None
    for path in sorted(candidates, key=lambda p: p.stat().st_mtime, reverse=True):
        try:
            return parse_max_similarity(path)
        except Exception as exc:
            last_error = exc
            continue

    raise RuntimeError(f"Could not parse any comparison NPZ in {comp_dir}. Last error: {last_error}")


def parse_min_test_loss(npz_path: Path) -> int:
    data = np.load(npz_path, allow_pickle=True)

    if "min_checkpoint" in data:
        ckpt = int(np.asarray(data["min_checkpoint"]).reshape(-1)[0])
        if ckpt >= 0:
            return ckpt

    if "training_times" not in data or "test_loss_means" not in data:
        raise KeyError(f"Missing required keys in {npz_path}")

    times = np.asarray(data["training_times"]).reshape(-1).astype(np.int64)
    vals = np.asarray(data["test_loss_means"]).reshape(-1).astype(np.float64)
    if times.size == 0 or vals.size == 0 or times.size != vals.size:
        raise ValueError(f"Invalid arrays in {npz_path}")

    idx = int(np.argmin(vals))
    return int(times[idx])


def resolve_min_test_loss(args: argparse.Namespace) -> int:
    candidates: list[Path] = []

    if args.loss_file:
        candidates = [Path(args.loss_file)]
    elif args.loss_dir:
        loss_dir = Path(args.loss_dir)
        if not loss_dir.exists():
            raise FileNotFoundError(f"Loss directory not found: {loss_dir}")
        candidates = list(loss_dir.glob(args.loss_glob))
    else:
        raise ValueError("Provide --loss-file or --loss-dir for policy=min_test_loss")

    path = choose_latest(candidates, prefer_non_quick=args.prefer_non_quick)
    return parse_min_test_loss(path)


def main() -> int:
    args = parse_args()

    if args.policy == "max_similarity":
        ckpt = resolve_max_similarity(args)
    else:
        ckpt = resolve_min_test_loss(args)

    print(int(ckpt))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        raise SystemExit(1)
