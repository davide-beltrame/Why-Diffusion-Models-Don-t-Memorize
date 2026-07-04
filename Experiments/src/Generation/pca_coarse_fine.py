#!/usr/bin/env python3
"""
pca_coarse_fine.py — PCA-based coarse/fine decomposition of CelebA denoising predictions.

Tests whether coarse (leading PCA) directions are learned
early and remain split-independent, while fine (trailing PCA) directions become
split-dependent later — mirroring the mechanistic story from the hierarchical
data model (Sec. 4.1–4.2 of the paper).

Usage (HPC):
    cd $WORK/wdmdm/Experiments/src/Generation
    python pca_coarse_fine.py \
        -n 1024 -is 0 -ie 14 -s 32 -LR 0.0001 -O Adam -W 32 \
        -t 50 -Ns 512 -B 512 --pca_k 64

Reuses: cfg.py, loader.py, Unet.py, Diffusion.py (same imports as compare_scores.py).
Outputs saved under $SAVES/PCA_CoarseFine/.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams.update({
    "figure.dpi": 140,
    "savefig.dpi": 200,
    "font.size": 10,
    "axes.labelsize": 11,
    "axes.titlesize": 11,
    "legend.fontsize": 8,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
})

import numpy as np
import torch
import torch.nn.functional as F
import os
import sys
import json
import argparse
from tqdm import tqdm

sys.path.insert(1, "../Utils/")
import Diffusion
import Unet
import cfg
import loader

# ─── CLI ───────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser("PCA coarse/fine decomposition of CelebA denoiser predictions.")

parser.add_argument("-n", "--num", type=int, required=True, help="Training set size per index.")
parser.add_argument("-is", "--index_start", type=int, required=True, help="First model index.")
parser.add_argument("-ie", "--index_end", type=int, required=True, help="Last model index.")
parser.add_argument("-s", "--img_size", type=int, required=True, help="Image size (e.g. 32).")
parser.add_argument("-LR", "--learning_rate", type=float, required=True)
parser.add_argument("-O", "--optim", type=str, required=True)
parser.add_argument("-W", "--nbase", type=str, required=True)
parser.add_argument("-B", "--batch_size", type=int, required=True)
parser.add_argument("-t", "--time", type=int, default=50, help="Diffusion timestep for evaluation.")
parser.add_argument("-Ns", "--Nsamples", type=int, default=512, help="Number of eval samples (multiple of batch_gen).")
parser.add_argument("--pca_k", type=int, default=64, help="Number of leading PCA components = coarse subspace dimension.")
parser.add_argument("--device", type=str, default="cuda:0")
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--out_dire", type=str, default=None)

args = parser.parse_args()
print(args)

torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True


# ─── Config & data ────────────────────────────────────────────────────────────
DATASET = "CelebA"
config = cfg.load_config(DATASET)
n_base = int(args.nbase)
config.DEVICE = args.device
Nsamples = int(args.Nsamples)
size = int(args.img_size)
config.OPTIM = args.optim
config.BATCH_SIZE = int(args.batch_size)
config.LR = float(args.learning_rate)

# Load full image pool (40k train + 2k test)
config.n_images = 40000
big_train, big_test = cfg.load_training_data(config, 0, loadtest=True)

# ─── PCA basis ─────────────────────────────────────────────────────────────────
if args.out_dire is not None:
    config.path_save = args.out_dire

out_dir = os.path.join(config.path_save, "PCA_CoarseFine")
os.makedirs(out_dir, exist_ok=True)

pca_path = os.path.join(out_dir, f"pca_basis_K{args.pca_k}.npz")

# Build PCA on the full training pool (split-independent reference).
# Images are 1×32×32 = 1024-dim after centering with per-index-0 mean.
# big_train is a TransformedDataset; materialize it.
print("Computing PCA basis …")
pca_ref = torch.stack([big_train[i] for i in range(len(big_train))]).cpu()  # [N, 1, 32, 32]
pca_ref_flat = pca_ref.reshape(pca_ref.shape[0], -1).numpy()  # [N, D]
D = pca_ref_flat.shape[1]
K = min(args.pca_k, D)

pca_mean = pca_ref_flat.mean(axis=0)  # [D]
X_centered = pca_ref_flat - pca_mean  # [N, D]

# Economy SVD (only need top-K)
# For D=1024 this is fast even on CPU.
U, S, Vt = np.linalg.svd(X_centered, full_matrices=False)
components = Vt[:K]  # [K, D]  — rows are the K leading eigenvectors
explained_var = (S ** 2) / (X_centered.shape[0] - 1)
total_var = explained_var.sum()
cumulative_var_ratio = np.cumsum(explained_var) / total_var

print(f"PCA: D={D}, K={K}")
print(f"  Top-{K} components explain {cumulative_var_ratio[K-1]*100:.1f}% of variance.")

np.savez(
    pca_path,
    components=components,       # [K, D]
    pca_mean=pca_mean,           # [D]
    explained_var=explained_var,  # [D]
    cumulative_var_ratio=cumulative_var_ratio,  # [D]
    K=np.array(K),
)
print(f"Saved PCA basis to: {pca_path}")

# Convert to torch for projections
pca_components = torch.from_numpy(components).float().to(config.DEVICE)  # [K, D]
pca_mean_t = torch.from_numpy(pca_mean).float().to(config.DEVICE)        # [D]


# ─── Projection helpers ───────────────────────────────────────────────────────
def project_coarse(x_flat):
    """Project [B, D] tensor into the K-dim coarse subspace. Returns [B, D] in pixel space."""
    x_c = x_flat - pca_mean_t
    coeffs = x_c @ pca_components.T  # [B, K]
    return coeffs @ pca_components + pca_mean_t  # [B, D]

def project_fine(x_flat):
    """Residual = original minus coarse projection. Returns [B, D]."""
    return x_flat - project_coarse(x_flat)

def mse_in_subspace(a, b, proj_fn):
    """MSE between a and b after projecting both through proj_fn. a, b: [B, D]."""
    pa = proj_fn(a)
    pb = proj_fn(b)
    return ((pa - pb) ** 2).mean(dim=1)  # [B]


# ─── Eval images per index ─────────────────────────────────────────────────────
config.n_images = int(args.num)

# Per-index training images (for "train" condition) & shared test images
train_images = {}
for idx in range(args.index_start, args.index_end + 1):
    train_images[idx] = big_train[idx * args.num : (idx + 1) * args.num][:Nsamples]

test_images = big_test[:Nsamples]

# ─── Forward diffusion (fixed noise per index) ────────────────────────────────
df = Diffusion.DiffusionConfig(
    n_steps=config.TIMESTEPS,
    img_shape=config.IMG_SHAPE,
    device=config.DEVICE,
)
selected_time = torch.tensor([args.time], device=config.DEVICE, dtype=torch.long)

# Also precompute the scaling factors for x0 recovery:
# x0_pred = (x_t - sqrt(1-alpha_bar_t) * eps_pred) / sqrt(alpha_bar_t)
sqrt_alpha_bar_t = df.sqrt_alpha_cumulative[args.time].item()
sqrt_one_minus_alpha_bar_t = df.sqrt_one_minus_alpha_cumulative[args.time].item()

noisy_train = {}          # index -> (x_t, noise, x0_clean)
noisy_test = None         # (x_t, noise, x0_clean)

for idx in range(args.index_start, args.index_end + 1):
    torch.manual_seed(args.seed + idx)
    torch.cuda.manual_seed_all(args.seed + idx)

    X = torch.stack([train_images[idx][i] for i in range(len(train_images[idx]))]).to(config.DEVICE)
    ts = selected_time.repeat(X.shape[0])
    X_t, noise_t, _ = Diffusion.forward_diffusion(df, X, ts, config, return_std=True)
    noisy_train[idx] = (X_t, noise_t, X.clone())

# Test: use seed from first index for consistency
torch.manual_seed(args.seed + args.index_start)
torch.cuda.manual_seed_all(args.seed + args.index_start)

X_test = torch.stack([test_images[i] for i in range(len(test_images))]).to(config.DEVICE)
ts_test = selected_time.repeat(X_test.shape[0])
X_t_test, noise_t_test, _ = Diffusion.forward_diffusion(df, X_test, ts_test, config, return_std=True)
noisy_test = (X_t_test, noise_t_test, X_test.clone())


# ─── Model building ───────────────────────────────────────────────────────────
def build_type_model(index):
    return "{:s}{:d}_{:d}_{:d}_{:s}_{:d}_{:.4f}_index{:d}/".format(
        config.DATASET, size, config.n_images, n_base, config.OPTIM, config.BATCH_SIZE, config.LR, index
    )

def build_model():
    m = Unet.UNet(
        input_channels=config.IMG_SHAPE[0],
        output_channels=config.IMG_SHAPE[0],
        base_channels=n_base,
        base_channels_multiples=(1, 2, 4),
        apply_attention=(False, True, True),
        dropout_rate=0.1,
    )
    return m.to(config.DEVICE)


# ─── All unordered index pairs ────────────────────────────────────────────────
index_pairs = [
    (i, j)
    for i in range(args.index_start, args.index_end + 1)
    for j in range(i + 1, args.index_end + 1)
]
print(f"Evaluating {len(index_pairs)} unordered model pairs across {args.index_end - args.index_start + 1} indices.")


# ─── Checkpoint loop ──────────────────────────────────────────────────────────
batch_gen = 512
Ns = max(1, Nsamples // batch_gen)
training_times = np.linspace(10, 5000 - 1, 50, dtype=int)[1:]  # same grid as compare_scores.py

# Accumulators: shape will be [n_checkpoints]
results = {
    "err_coarse_train": [], "err_fine_train": [],
    "err_coarse_test":  [], "err_fine_test":  [],
    "dis_coarse_train": [], "dis_fine_train": [],
    "dis_coarse_test":  [], "dis_fine_test":  [],
    "test_loss": [],
}

loss_fn = torch.nn.MSELoss(reduction="none")
model_a = build_model()
model_b = build_model()

for j, ckpt in enumerate(training_times):
    print(f"\nCheckpoint epoch {ckpt}  ({j+1}/{len(training_times)})")
    ckpt_suffix = f"/Models/Model_epoch_{int(ckpt)}"

    # Per-pair accumulators for this checkpoint
    pair_err_coarse_train = []
    pair_err_fine_train   = []
    pair_err_coarse_test  = []
    pair_err_fine_test    = []
    pair_dis_coarse_train = []
    pair_dis_fine_train   = []
    pair_dis_coarse_test  = []
    pair_dis_fine_test    = []
    pair_test_loss        = []

    for k, (ia, ib) in enumerate(index_pairs):
        type_a = build_type_model(ia)
        type_b = build_type_model(ib)
        path_a = config.path_save + type_a + "Models" + f"/Model_epoch_{int(ckpt)}"
        path_b = config.path_save + type_b + "Models" + f"/Model_epoch_{int(ckpt)}"

        if not os.path.exists(path_a):
            raise FileNotFoundError(f"Missing checkpoint: {path_a}")
        if not os.path.exists(path_b):
            raise FileNotFoundError(f"Missing checkpoint: {path_b}")

        model_a = loader.load_model(model_a, path_a, verbose=False)
        model_b = loader.load_model(model_b, path_b, verbose=False)
        model_a.eval()
        model_b.eval()

        # --- TRAIN condition: use noisy images from split A ---
        xt_a, noise_a, x0_clean_a = noisy_train[ia]
        batch_results_train = {key: [] for key in ["ec", "ef", "dc", "df"]}

        for bi in range(Ns):
            sl = slice(bi * batch_gen, (bi + 1) * batch_gen)
            x_t = xt_a[sl]
            x0_gt = x0_clean_a[sl].reshape(x_t.shape[0], -1)  # [B, D]

            with torch.no_grad():
                t_in = selected_time.repeat(x_t.shape[0])
                eps_a = model_a(x_t, t_in)
                eps_b = model_b(x_t, t_in)

            # Convert eps -> x0 predictions
            x0_a = (x_t - sqrt_one_minus_alpha_bar_t * eps_a) / sqrt_alpha_bar_t
            x0_b = (x_t - sqrt_one_minus_alpha_bar_t * eps_b) / sqrt_alpha_bar_t
            x0_a_flat = x0_a.reshape(x_t.shape[0], -1)
            x0_b_flat = x0_b.reshape(x_t.shape[0], -1)

            # Reconstruction errors (model A vs clean target)
            batch_results_train["ec"].append(mse_in_subspace(x0_a_flat, x0_gt, project_coarse).cpu().numpy())
            batch_results_train["ef"].append(mse_in_subspace(x0_a_flat, x0_gt, project_fine).cpu().numpy())

            # Disagreement (model A vs model B)
            batch_results_train["dc"].append(mse_in_subspace(x0_a_flat, x0_b_flat, project_coarse).cpu().numpy())
            batch_results_train["df"].append(mse_in_subspace(x0_a_flat, x0_b_flat, project_fine).cpu().numpy())

        pair_err_coarse_train.append(np.concatenate(batch_results_train["ec"]).mean())
        pair_err_fine_train.append(np.concatenate(batch_results_train["ef"]).mean())
        pair_dis_coarse_train.append(np.concatenate(batch_results_train["dc"]).mean())
        pair_dis_fine_train.append(np.concatenate(batch_results_train["df"]).mean())

        # --- TEST condition: use shared test images ---
        xt_test, noise_test, x0_clean_test = noisy_test
        batch_results_test = {key: [] for key in ["ec", "ef", "dc", "df", "loss"]}

        for bi in range(Ns):
            sl = slice(bi * batch_gen, (bi + 1) * batch_gen)
            x_t = xt_test[sl]
            x0_gt = x0_clean_test[sl].reshape(x_t.shape[0], -1)
            noise_gt = noise_test[sl]

            with torch.no_grad():
                t_in = selected_time.repeat(x_t.shape[0])
                eps_a = model_a(x_t, t_in)
                eps_b = model_b(x_t, t_in)

            x0_a = (x_t - sqrt_one_minus_alpha_bar_t * eps_a) / sqrt_alpha_bar_t
            x0_b = (x_t - sqrt_one_minus_alpha_bar_t * eps_b) / sqrt_alpha_bar_t
            x0_a_flat = x0_a.reshape(x_t.shape[0], -1)
            x0_b_flat = x0_b.reshape(x_t.shape[0], -1)

            batch_results_test["ec"].append(mse_in_subspace(x0_a_flat, x0_gt, project_coarse).cpu().numpy())
            batch_results_test["ef"].append(mse_in_subspace(x0_a_flat, x0_gt, project_fine).cpu().numpy())
            batch_results_test["dc"].append(mse_in_subspace(x0_a_flat, x0_b_flat, project_coarse).cpu().numpy())
            batch_results_test["df"].append(mse_in_subspace(x0_a_flat, x0_b_flat, project_fine).cpu().numpy())

            # Also compute test loss (MSE in noise space, averaged over both models)
            loss_val = 0.5 * (
                loss_fn(eps_a, noise_gt).mean(dim=(1, 2, 3)) +
                loss_fn(eps_b, noise_gt).mean(dim=(1, 2, 3))
            )
            batch_results_test["loss"].append(loss_val.cpu().numpy())

        pair_err_coarse_test.append(np.concatenate(batch_results_test["ec"]).mean())
        pair_err_fine_test.append(np.concatenate(batch_results_test["ef"]).mean())
        pair_dis_coarse_test.append(np.concatenate(batch_results_test["dc"]).mean())
        pair_dis_fine_test.append(np.concatenate(batch_results_test["df"]).mean())
        pair_test_loss.append(np.concatenate(batch_results_test["loss"]).mean())

    # Average over all pairs for this checkpoint
    results["err_coarse_train"].append(np.mean(pair_err_coarse_train))
    results["err_fine_train"].append(np.mean(pair_err_fine_train))
    results["err_coarse_test"].append(np.mean(pair_err_coarse_test))
    results["err_fine_test"].append(np.mean(pair_err_fine_test))
    results["dis_coarse_train"].append(np.mean(pair_dis_coarse_train))
    results["dis_fine_train"].append(np.mean(pair_dis_fine_train))
    results["dis_coarse_test"].append(np.mean(pair_dis_coarse_test))
    results["dis_fine_test"].append(np.mean(pair_dis_fine_test))
    results["test_loss"].append(np.mean(pair_test_loss))

    # ── incremental save ──
    for key in results:
        results[key] = list(results[key])  # ensure list

    npz_save = os.path.join(out_dir, f"pca_coarse_fine_t{args.time}_K{K}.npz")
    np.savez(
        npz_save,
        training_times=training_times,
        **{k: np.array(v) for k, v in results.items()},
        pca_k=np.array(K),
        time=np.array(args.time),
        cumulative_var_ratio_K=np.array(cumulative_var_ratio[K - 1]),
        args_json=np.array(json.dumps(vars(args), sort_keys=True)),
    )

    # ── incremental plot ──
    xs = training_times[: j + 1]

    fig, axes = plt.subplots(2, 2, figsize=(11.5, 7.4), sharex=True)

    # (0,0): Reconstruction error — coarse vs fine, test images
    ax = axes[0, 0]
    ax.plot(xs, results["err_coarse_test"], "o-", label="Coarse", color="C0", markersize=3)
    ax.plot(xs, results["err_fine_test"], "s-", label="Fine", color="C1", markersize=3)
    ax.set_ylabel("Reconstruction MSE")
    ax.set_title("Reconstruction error (test images)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # (0,1): Reconstruction error — coarse vs fine, train images
    ax = axes[0, 1]
    ax.plot(xs, results["err_coarse_train"], "o-", label="Coarse", color="C0", markersize=3)
    ax.plot(xs, results["err_fine_train"], "s-", label="Fine", color="C1", markersize=3)
    ax.set_ylabel("Reconstruction MSE")
    ax.set_title("Reconstruction error (train images)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # (1,0): Model disagreement — coarse vs fine, test images + test loss
    ax = axes[1, 0]
    ax.plot(xs, results["dis_coarse_test"], "o-", label="Coarse disagr.", color="C2", markersize=3)
    ax.plot(xs, results["dis_fine_test"], "s-", label="Fine disagr.", color="C3", markersize=3)
    ax.set_ylabel("Disagreement MSE")
    ax.set_xlabel("Epoch")
    ax.set_title("Model disagreement (test images)")
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(True, alpha=0.3)
    # Test loss on right axis
    ax2 = ax.twinx()
    ax2.plot(xs, results["test_loss"], "d-", label="Test loss", color="C4", markersize=3, alpha=0.6)
    ax2.set_ylabel("Test loss MSE")
    ax2.legend(loc="upper right", fontsize=8)

    # (1,1): Model disagreement — coarse vs fine, train images
    ax = axes[1, 1]
    ax.plot(xs, results["dis_coarse_train"], "o-", label="Coarse disagr.", color="C2", markersize=3)
    ax.plot(xs, results["dis_fine_train"], "s-", label="Fine disagr.", color="C3", markersize=3)
    ax.set_ylabel("Disagreement MSE")
    ax.set_xlabel("Epoch")
    ax.set_title("Model disagreement (train images)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    fig.suptitle(
        f"PCA coarse/fine decomposition  (t={args.time}, K={K}, "
        f"top-K var={cumulative_var_ratio[K-1]*100:.1f}%)",
        fontsize=11,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    plot_path = os.path.join(out_dir, f"pca_coarse_fine_t{args.time}_K{K}.png")
    fig.savefig(plot_path, dpi=150, bbox_inches="tight")
    fig.savefig(os.path.join(out_dir, f"pca_coarse_fine_t{args.time}_K{K}.pdf"), bbox_inches="tight")
    plt.close("all")

    print(f"  Saved checkpoint {ckpt}: npz={npz_save}, plot={plot_path}")

print("\nDone.")
print(f"  NPZ: {npz_save}")
print(f"  Plot: {plot_path}")
print(f"  PCA basis: {pca_path}")
