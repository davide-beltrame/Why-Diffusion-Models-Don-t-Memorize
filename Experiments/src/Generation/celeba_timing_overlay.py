#!/usr/bin/env python3
"""
Build a checkpoint-aligned CelebA timing overlay from existing outputs:
- sample-split cosine similarity curves (per pair NPZs)
- deterministic train/test loss curves (per index NPZs)
- memorization f_mem curves (per index NPZs)

Output:
- merged CSV + NPZ table aligned by checkpoint
- timing overlay plot (.png/.pdf)
"""

import argparse
import csv
import glob
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import re


def sem_over_first_axis(x: np.ndarray) -> np.ndarray:
    n_eff = np.sum(np.isfinite(x), axis=0)
    out = np.full(x.shape[1], np.nan, dtype=np.float64)
    good = n_eff >= 2
    out[good] = np.nanstd(x[:, good], axis=0, ddof=1) / np.sqrt(n_eff[good])
    return out


def relative_scale(y: np.ndarray, sem: np.ndarray):
    y_min = float(np.nanmin(y))
    y_max = float(np.nanmax(y))
    span = y_max - y_min
    if span <= 1e-12:
        return np.zeros_like(y, dtype=np.float64), np.zeros_like(sem, dtype=np.float64), y_min, y_max
    return (y - y_min) / span, sem / span, y_min, y_max


def load_cosine_curves_for_seed(comp_dir: str, metric: str, indices):
    all_files = sorted(glob.glob(os.path.join(comp_dir, f"cosine_similarity_metric_{metric}_index*_vs_*.npz")))
    if len(all_files) == 0:
        raise FileNotFoundError(f"No cosine NPZ found in {comp_dir}")

    idx_set = set(indices)
    pair_pat = re.compile(r".*_index(\d+)_vs_(\d+)\.npz$")
    files = []
    for fp in all_files:
        m = pair_pat.match(os.path.basename(fp))
        if m is None:
            continue
        ia = int(m.group(1))
        ib = int(m.group(2))
        if ia in idx_set and ib in idx_set:
            files.append(fp)

    if len(files) == 0:
        raise FileNotFoundError(
            f"No cosine NPZ matched requested indices={sorted(indices)} in {comp_dir}"
        )

    raw_epochs = []
    raw_curves = []
    for fp in files:
        d = np.load(fp)
        raw_epochs.append(d["training_times"].astype(np.int64))
        raw_curves.append(d["means"].astype(np.float64))

    common_epochs = raw_epochs[0]
    for ep in raw_epochs[1:]:
        common_epochs = np.intersect1d(common_epochs, ep)

    if common_epochs.size == 0:
        raise ValueError("No common cosine checkpoints across selected pair files.")

    curves = []
    for ep, cv, fp in zip(raw_epochs, raw_curves, files):
        mask = np.isin(ep, common_epochs)
        if int(mask.sum()) != int(common_epochs.size):
            raise ValueError(f"Cosine file missing common checkpoints: {fp}")
        curves.append(cv[mask])

    curves = np.stack(curves, axis=0)
    return common_epochs.astype(np.int64), curves


def parse_seed_spec(saves_dir: str, n_train: int, seed_spec: str, fallback_seed: int):
    if seed_spec.strip().lower() == "auto":
        pat = os.path.join(saves_dir, f"Comparisons_Mallat_final_{n_train}_seed_*")
        dirs = sorted(glob.glob(pat))
        seeds = []
        for d in dirs:
            m = re.match(r".*_seed_(\d+)$", d)
            if m is not None:
                seeds.append(int(m.group(1)))
        if len(seeds) == 0:
            return [int(fallback_seed)]
        return sorted(set(seeds))
    return [int(x) for x in seed_spec.split(",") if x.strip() != ""]


def load_cosine_curves_multi_seed(saves_dir: str, n_train: int, metric: str, indices, seed_spec: str, fallback_seed: int):
    seeds = parse_seed_spec(saves_dir, n_train, seed_spec, fallback_seed)
    epochs_list = []
    curves_list = []
    used = []

    for sd in seeds:
        comp_dir = os.path.join(saves_dir, f"Comparisons_Mallat_final_{n_train}_seed_{sd}")
        if not os.path.isdir(comp_dir):
            continue
        try:
            ep, cv = load_cosine_curves_for_seed(comp_dir, metric, indices)
        except FileNotFoundError:
            continue
        epochs_list.append(ep)
        curves_list.append(cv)
        used.append(sd)

    if len(curves_list) == 0:
        raise FileNotFoundError("No valid cosine curves found across requested seed runs.")

    common = epochs_list[0]
    for ep in epochs_list[1:]:
        common = np.intersect1d(common, ep)
    if common.size == 0:
        raise ValueError("No common cosine checkpoints across seed runs.")

    stacked = []
    for ep, cv in zip(epochs_list, curves_list):
        mask = np.isin(ep, common)
        stacked.append(cv[:, mask])

    return common.astype(np.int64), np.concatenate(stacked, axis=0), used


def load_loss_curves(saves_dir: str, indices):
    loss_dir = os.path.join(saves_dir, "Losses_over_checkpoints_timing")
    test_curves = []
    train_curves = []
    epochs_ref = None

    for idx in indices:
        pattern = os.path.join(loss_dir, f"timing_loss_avg_over_timesteps_vs_checkpoint_*_index{idx}.npz")
        matches = sorted(glob.glob(pattern))
        if len(matches) == 0:
            raise FileNotFoundError(f"No timing loss NPZ for index {idx} in {loss_dir}")
        fp = matches[-1]
        d = np.load(fp, allow_pickle=True)

        epochs = d["training_times"].astype(np.int64)
        if epochs_ref is None:
            epochs_ref = epochs
        elif not np.array_equal(epochs_ref, epochs):
            raise ValueError(f"Loss checkpoint mismatch for index {idx}: {fp}")

        test_means = d["test_loss_means"]
        train_means = d["train_loss_means"]
        if test_means.size == 0 or train_means.size == 0:
            raise ValueError(f"Expected both train/test losses in {fp}. Re-run with --split both.")

        test_curves.append(test_means.astype(np.float64))
        train_curves.append(train_means.astype(np.float64))

    return epochs_ref, np.stack(train_curves, axis=0), np.stack(test_curves, axis=0)


def load_fmem_curves(saves_dir: str, indices, n_train: int, img_size: int, n_base: int, optim: str, batch_size: int, lr: float):
    fmem_curves = []
    epochs_ref = None

    for idx in indices:
        run_name = f"CelebA{img_size}_{n_train}_{n_base}_{optim}_{batch_size}_{lr:.4f}_index{idx}"
        fp = os.path.join(saves_dir, run_name, "Memorization", "fraction_memorized.npz")
        if not os.path.exists(fp):
            raise FileNotFoundError(f"Missing f_mem NPZ for index {idx}: {fp}")

        d = np.load(fp)
        epochs = d["training_times"].astype(np.int64)
        vals = d["fmem_percent"].astype(np.float64)

        if epochs_ref is None:
            epochs_ref = epochs
        elif not np.array_equal(epochs_ref, epochs):
            raise ValueError(f"f_mem checkpoint mismatch for index {idx}: {fp}")

        fmem_curves.append(vals)

    return epochs_ref, np.stack(fmem_curves, axis=0)


def main():
    parser = argparse.ArgumentParser("Aggregate CelebA timing metrics into one aligned overlay.")
    parser.add_argument("--saves_dir", type=str, required=True)
    parser.add_argument("--indices", type=str, required=True, help="Comma-separated model indices, e.g. 0,1,2,3")
    parser.add_argument("-n", "--num", type=int, default=1024)
    parser.add_argument("-s", "--img_size", type=int, default=32)
    parser.add_argument("-W", "--nbase", type=int, default=32)
    parser.add_argument("-O", "--optim", type=str, default="Adam")
    parser.add_argument("-B", "--batch_size", type=int, default=512)
    parser.add_argument("-LR", "--learning_rate", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--cosine_seeds", type=str, default="auto",
                        help="Comma-separated list of seed runs for cosine aggregation, or 'auto'.")
    parser.add_argument("--metric", type=str, default="pixel")
    parser.add_argument("--fmem_onset_threshold", type=float, default=0.1,
                        help="f_mem threshold in percent to mark onset.")
    parser.add_argument("--relative_plot", action="store_true",
                        help="Plot cosine/train/test on [0,1] relative scales to reduce magnitude mismatch.")
    parser.add_argument("--out_dir", type=str, default=None)
    args = parser.parse_args()

    indices = [int(x) for x in args.indices.split(",") if x.strip() != ""]
    if len(indices) == 0:
        raise ValueError("--indices produced an empty list")

    out_dir = args.out_dir if args.out_dir is not None else args.saves_dir
    os.makedirs(out_dir, exist_ok=True)

    epochs_cos, cosine_sims, cosine_seeds_used = load_cosine_curves_multi_seed(
        args.saves_dir, args.num, args.metric, indices, args.cosine_seeds, args.seed
    )
    epochs_loss, train_losses, test_losses = load_loss_curves(args.saves_dir, indices)
    epochs_fmem, fmem_vals = load_fmem_curves(
        args.saves_dir, indices, args.num, args.img_size, args.nbase, args.optim, args.batch_size, args.learning_rate
    )

    common = np.intersect1d(np.intersect1d(epochs_cos, epochs_loss), epochs_fmem)
    if common.size == 0:
        raise ValueError("No common checkpoints across cosine/loss/f_mem.")

    def align(epochs, arr2d):
        order = np.searchsorted(epochs, common)
        return arr2d[:, order]

    cosine_aligned = align(epochs_cos, cosine_sims)
    train_aligned = align(epochs_loss, train_losses)
    test_aligned = align(epochs_loss, test_losses)
    fmem_aligned = align(epochs_fmem, fmem_vals)

    cosine_dist = 1.0 - cosine_aligned
    cosine_dist_mean = np.nanmedian(cosine_dist, axis=0)
    cosine_dist_sem = sem_over_first_axis(cosine_dist)

    train_mean = np.nanmedian(train_aligned, axis=0)
    train_sem = sem_over_first_axis(train_aligned)
    test_mean = np.nanmedian(test_aligned, axis=0)
    test_sem = sem_over_first_axis(test_aligned)

    fmem_mean = np.nanmedian(fmem_aligned, axis=0)
    fmem_sem = sem_over_first_axis(fmem_aligned)

    idx_cos_min = int(np.nanargmin(cosine_dist_mean))
    idx_test_min = int(np.nanargmin(test_mean))
    idx_fmem_onset = np.where(fmem_mean >= args.fmem_onset_threshold)[0]
    if idx_fmem_onset.size > 0:
        idx_fmem_onset = int(idx_fmem_onset[0])
    else:
        idx_fmem_onset = -1

    table_npz = os.path.join(out_dir, "celeba_timing_overlay.npz")
    table_csv = os.path.join(out_dir, "celeba_timing_overlay.csv")

    np.savez(
        table_npz,
        checkpoints=common,
        cosine_dist_mean=cosine_dist_mean,
        cosine_dist_sem=cosine_dist_sem,
        train_loss_mean=train_mean,
        train_loss_sem=train_sem,
        test_loss_mean=test_mean,
        test_loss_sem=test_sem,
        fmem_percent_mean=fmem_mean,
        fmem_percent_sem=fmem_sem,
        idx_cos_min=np.array(idx_cos_min, dtype=np.int64),
        idx_test_min=np.array(idx_test_min, dtype=np.int64),
        idx_fmem_onset=np.array(idx_fmem_onset, dtype=np.int64),
        fmem_onset_threshold=np.array(args.fmem_onset_threshold, dtype=np.float64),
        indices=np.array(indices, dtype=np.int64),
        cosine_seeds_used=np.array(cosine_seeds_used, dtype=np.int64),
        n_cosine_curves=np.array(int(cosine_sims.shape[0]), dtype=np.int64),
    )

    with open(table_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "checkpoint",
            "cosine_dist_mean", "cosine_dist_sem",
            "train_loss_mean", "train_loss_sem",
            "test_loss_mean", "test_loss_sem",
            "fmem_percent_mean", "fmem_percent_sem",
        ])
        for i, ck in enumerate(common.tolist()):
            w.writerow([
                ck,
                cosine_dist_mean[i], cosine_dist_sem[i],
                train_mean[i], train_sem[i],
                test_mean[i], test_sem[i],
                fmem_mean[i], fmem_sem[i],
            ])

    fig, ax1 = plt.subplots(figsize=(7.0, 4.8))
    ax2 = ax1.twinx()

    c_cos = "#1f77b4"
    c_test = "#d62728"
    c_train = "#2ca02c"
    c_fmem = "#111111"

    use_relative = bool(args.relative_plot)
    if use_relative:
        cos_plot, cos_sem_plot, cos_min, cos_max = relative_scale(cosine_dist_mean, cosine_dist_sem)
        test_plot, test_sem_plot, test_min, test_max = relative_scale(test_mean, test_sem)
        train_plot, train_sem_plot, train_min, train_max = relative_scale(train_mean, train_sem)
    else:
        cos_plot, cos_sem_plot = cosine_dist_mean, cosine_dist_sem
        test_plot, test_sem_plot = test_mean, test_sem
        train_plot, train_sem_plot = train_mean, train_sem

    ax1.errorbar(common, cos_plot, yerr=cos_sem_plot, color=c_cos, marker="o", ms=3,
                 linewidth=1.3, capsize=2, label="sample-split cosine distance")
    ax1.errorbar(common, test_plot, yerr=test_sem_plot, color=c_test, marker="d", ms=3,
                 linewidth=1.3, capsize=2, linestyle="--", label="test DSM loss")
    ax1.errorbar(common, train_plot, yerr=train_sem_plot, color=c_train, marker="s", ms=3,
                 linewidth=1.1, capsize=2, linestyle="-.", label="train DSM loss")

    fmem_is_flat = np.nanmax(fmem_mean) - np.nanmin(fmem_mean) <= 1e-12
    if not fmem_is_flat:
        ax2.errorbar(common, fmem_mean, yerr=fmem_sem, color=c_fmem, marker="^", ms=3,
                     linewidth=1.2, capsize=2, linestyle=":", label="f_mem (%)")

    ax1.axvline(common[idx_cos_min], color=c_cos, linestyle=":", linewidth=1.0)
    ax1.axvline(common[idx_test_min], color=c_test, linestyle=":", linewidth=1.0)
    if idx_fmem_onset >= 0:
        ax1.axvline(common[idx_fmem_onset], color=c_fmem, linestyle=":", linewidth=1.0)

    ax1.set_xlabel("Checkpoint")
    if use_relative:
        ax1.set_ylabel("Relative scale (0-1, per curve)")
    else:
        ax1.set_ylabel("Cosine distance / DSM loss")
    if not fmem_is_flat:
        ax2.set_ylabel("f_mem (%)")
    else:
        ax2.set_yticks([])
    ax1.grid(True, linestyle=":", linewidth=0.8, alpha=0.35)

    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, loc="best", frameon=False)

    title = (
        f"CelebA timing overlay | cos-min@{common[idx_cos_min]} "
        f"test-min@{common[idx_test_min]} | seeds={len(cosine_seeds_used)}"
    )
    if idx_fmem_onset >= 0:
        title += f" fmem-onset@{common[idx_fmem_onset]}"
    if fmem_is_flat:
        title += " | f_mem flat"
    if use_relative:
        title += (
            f" | rel: cos[{cos_min:.3g},{cos_max:.3g}]"
            f" test[{test_min:.3g},{test_max:.3g}]"
        )
    ax1.set_title(title)

    fig.tight_layout()
    png_path = os.path.join(out_dir, "celeba_timing_overlay.png")
    pdf_path = os.path.join(out_dir, "celeba_timing_overlay.pdf")
    fig.savefig(png_path, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved table NPZ: {table_npz}")
    print(f"Saved table CSV: {table_csv}")
    print(f"Saved plot PNG: {png_path}")
    print(f"Saved plot PDF: {pdf_path}")


if __name__ == "__main__":
    main()
