#!/usr/bin/env python3
"""Aggregate per-pair cosine similarity and per-model test loss NPZ files
into a single NPZ and produce the dual-axis Fig 1(a) left panel plot.

Inputs (produced by sample_split_inference.py and loss_compute.py):
  - Comparisons_Mallat_final_{n}_seed_{seed}/cosine_similarity_metric_{metric}_index*_vs_*.npz
  - Losses_over_checkpoints_timing/timing_loss_avg_over_timesteps_vs_checkpoint_*_index*.npz

Outputs:
  - {out_dir}/cosine_and_loss_data.npz
  - {out_dir}/cos_dis_vs_test_loss.pdf/.png
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import os
import glob
import argparse
import numpy as np

# ---------------------------------------------------------------------------
plt.rcParams.update({
    "figure.dpi": 140,
    "savefig.dpi": 200,
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 12,
    "legend.frameon": False,
    "grid.alpha": 0.3,
})
# ---------------------------------------------------------------------------

parser = argparse.ArgumentParser(
    "Aggregate CelebA sample-split results into a dual-axis cosine-distance / test-loss plot."
)
parser.add_argument("--saves_dir", type=str, required=True,
                    help="Path to Saves_new/ (contains Comparisons_Mallat_final_* and Losses_over_checkpoints).")
parser.add_argument("-n", "--num", type=int, default=1024,
                    help="N_train used in folder names.")
parser.add_argument("--seed", type=int, default=0,
                    help="Seed used in Comparisons folder name.")
parser.add_argument("--metric", type=str, default="pixel",
                    help="Similarity metric tag in NPZ filenames.")
parser.add_argument("--out_dir", type=str, default=None,
                    help="Output directory (defaults to saves_dir).")
args = parser.parse_args()

saves_dir = args.saves_dir
out_dir = args.out_dir if args.out_dir is not None else saves_dir
os.makedirs(out_dir, exist_ok=True)

# ── 1. load cosine similarity curves (one NPZ per pair) ──────────────────────
comp_dir = os.path.join(
    saves_dir, f"Comparisons_Mallat_final_{args.num}_seed_{args.seed}"
)
cos_files = sorted(glob.glob(
    os.path.join(comp_dir, f"cosine_similarity_metric_{args.metric}_index*_vs_*.npz")
))
assert len(cos_files) > 0, f"No cosine similarity NPZ files in {comp_dir}"

cos_means_list = []
cos_epochs = None
for f in cos_files:
    d = np.load(f)
    if cos_epochs is None:
        cos_epochs = d["training_times"]
    else:
        assert np.array_equal(cos_epochs, d["training_times"]), \
            f"Epochs mismatch in {f}"
    cos_means_list.append(d["means"])

cosine_sims = np.stack(cos_means_list, axis=0)  # (n_pairs, n_cos_epochs)
n_pairs = cosine_sims.shape[0]
print(f"Loaded {n_pairs} cosine similarity curves, {len(cos_epochs)} checkpoints.")

# ── 2. load test loss curves (one NPZ per model) ─────────────────────────────
loss_dir = os.path.join(saves_dir, "Losses_over_checkpoints_timing")
loss_files = sorted(glob.glob(
    os.path.join(loss_dir, "timing_loss_avg_over_timesteps_vs_checkpoint_*_index*.npz")
))
if not loss_files:
    # Backward compatibility with pre-rebuttal result bundles.
    loss_dir = os.path.join(saves_dir, "Losses_over_checkpoints")
    loss_files = sorted(glob.glob(
        os.path.join(loss_dir, "test_mse_avg_over_timesteps_vs_checkpoint_*_index*.npz")
    ))
assert len(loss_files) > 0, f"No loss NPZ files in {loss_dir}"

loss_means_list = []
loss_epochs = None
for f in loss_files:
    d = np.load(f, allow_pickle=True)
    if loss_epochs is None:
        loss_epochs = d["training_times"]
    else:
        assert np.array_equal(loss_epochs, d["training_times"]), \
            f"Epochs mismatch in {f}"
    if "test_loss_means" in d:
        loss_means_list.append(d["test_loss_means"])
    else:
        loss_means_list.append(d["loss_means"])

test_losses_full = np.stack(loss_means_list, axis=0)  # (n_models, n_loss_epochs)
n_models = test_losses_full.shape[0]
print(f"Loaded {n_models} test loss curves, {len(loss_epochs)} checkpoints.")

# ── 3. align to common epoch grid ────────────────────────────────────────────
common_epochs = cos_epochs
loss_mask = np.isin(loss_epochs, common_epochs)
if loss_mask.sum() == len(common_epochs):
    test_losses = test_losses_full[:, loss_mask]
    print("Loss epochs are a superset of cosine epochs; selecting matching subset.")
else:
    test_losses = np.column_stack([
        np.interp(common_epochs, loss_epochs, test_losses_full[i])
        for i in range(n_models)
    ]).T
    print("Interpolated loss data onto cosine epoch grid.")

epochs = common_epochs

# ── 4. save aggregated NPZ ───────────────────────────────────────────────────
agg_path = os.path.join(out_dir, "cosine_and_loss_data.npz")
np.savez(
    agg_path,
    epochs=epochs,
    cosine_similarities=cosine_sims,
    test_losses=test_losses,
    loss_epochs_full=loss_epochs,
    test_losses_full=test_losses_full,
)
print(f"Saved aggregated data → {agg_path}")

# ── 5. plot ───────────────────────────────────────────────────────────────────
cosine_median = np.nanmedian(cosine_sims, axis=0)
loss_median = np.nanmedian(test_losses, axis=0)

n_eff_cos = np.sum(np.isfinite(cosine_sims), axis=0)
n_eff_loss = np.sum(np.isfinite(test_losses), axis=0)

cosine_sem = np.where(
    n_eff_cos >= 2,
    np.nanstd(cosine_sims, axis=0, ddof=1) / np.sqrt(n_eff_cos),
    np.nan,
)
loss_sem = np.where(
    n_eff_loss >= 2,
    np.nanstd(test_losses, axis=0, ddof=1) / np.sqrt(n_eff_loss),
    np.nan,
)

fig, ax1 = plt.subplots(figsize=(6, 6))
ax2 = ax1.twinx()

color_cos = "#1f77b4"
color_loss = "#d62728"

# faint individual pair traces
for i in range(n_pairs):
    ax1.plot(epochs, 1 - cosine_sims[i], alpha=min(1.0, 2 / n_pairs),
             linewidth=0.8, color=color_cos, zorder=1)

# median cosine distance ± SEM
ax1.errorbar(epochs, 1 - cosine_median, yerr=cosine_sem,
             marker="o", ls="-", color=color_cos, capsize=3,
             linewidth=1.5, markersize=4, zorder=3,
             label="Cosine distance (median)")
ax1.set_xlabel("Epochs")
ax1.set_ylabel("Cosine distance", color=color_cos)
ax1.tick_params(axis="y", labelcolor=color_cos)

tau_cos = epochs[np.argmin(1 - cosine_median)]
ax1.set_ylim(0.06, 0.26)
ax1.axvline(tau_cos, ls=":", color=color_cos, linewidth=0.8)

# median test loss ± SEM
ax2.errorbar(epochs, loss_median, yerr=loss_sem,
             marker="d", ls="--", color=color_loss, capsize=3,
             linewidth=1.5, markersize=4, zorder=3,
             label="Test loss (median)")
ax2.set_ylabel("Test loss (MSE)", color=color_loss)
ax2.tick_params(axis="y", labelcolor=color_loss)

tau_loss = epochs[np.argmin(loss_median)]
ax2.set_ylim(0.0247, 0.035)
ax2.axvline(tau_loss, ls=":", color=color_loss, linewidth=0.8)

ax1.set_xlim(epochs.min(), epochs.max())
ax1.grid(True, linestyle=":", linewidth=0.8)

fig.tight_layout()
for ext in ("pdf", "png"):
    fpath = os.path.join(out_dir, f"cos_dis_vs_test_loss.{ext}")
    fig.savefig(fpath, bbox_inches="tight")
    print(f"Saved plot → {fpath}")
plt.close(fig)
