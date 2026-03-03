#!/usr/bin/env python3
import matplotlib
matplotlib.use("Agg")  # non-interactive backend (safe for cluster runs)
import matplotlib.pyplot as plt

import os
import sys
import argparse
import json
import numpy as np
import torch
from tqdm import tqdm

sys.path.insert(1, "../Utils/")  # In case we run from Experiments/Generation
import Diffusion
import Unet
import cfg
import loader


# =============================================================================
# Args
# =============================================================================
parser = argparse.ArgumentParser(
    "Single-model test MSE averaged over ALL diffusion timesteps, as a function of training checkpoint."
)

# run identity (path)
parser.add_argument("-n", "--num", type=int, required=True, help="N_train used for the run folder name.")
parser.add_argument("-i", "--index", type=int, required=True, help="Index used for the run folder name.")
parser.add_argument("-s", "--img_size", type=int, required=True, help="Image size used to train.")
parser.add_argument("-LR", "--learning_rate", type=float, required=True, help="Learning rate.")
parser.add_argument("-O", "--optim", type=str, required=True, help="Optimizer name (SGD_Momentum or Adam).")
parser.add_argument("-W", "--nbase", type=str, required=True, help="Base channels (string in your runs).")
parser.add_argument("-B", "--batch_size", type=int, required=True, help="Batch size used to train.")
parser.add_argument("-D", "--dataset", type=str, default="CelebA", help="Dataset name.")
parser.add_argument("--device", type=str, default="cuda:0", help="Device for eval.")
parser.add_argument("--seed", type=int, default=0, help="Seed controlling eval subset + forward noise reproducibility.")

parser.add_argument(
    "--seed_run",
    type=int,
    default=None,
    help="If your saved run uses the '_seed{...}' suffix, set this. Else leave None.",
)


# eval controls
parser.add_argument("--eval_N", type=int, default=10, help="Test images PER timestep")
parser.add_argument("--eval_batch", type=int, default=100, help="Batch size for model forward (<= eval_N).")
parser.add_argument(
    "--t_base",
    type=int,
    choices=[0, 1],
    default=1,
    help="If timesteps are [1..T], use 1 (default). If [0..T-1], use 0.",
)

# outputs
parser.add_argument("--out_dire", type=str, default=None, help="Override config.path_save (optional).")
parser.add_argument("--xlog", action="store_true", help="Use log scale on x (checkpoints).")
parser.add_argument("--ylog", action="store_true", help="Use log scale on y (loss).")

args = parser.parse_args()
print(args)

torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True


# =============================================================================
# Helpers
# =============================================================================

def build_type_model(config, size, n_base, index: int, seed_run: int | None) -> str:
    if seed_run is None:
        return "{:s}{:d}_{:d}_{:d}_{:s}_{:d}_{:.4f}_index{:d}/".format(
            config.DATASET, size, config.n_images, n_base, config.OPTIM, config.BATCH_SIZE, config.LR, index
        )
    else:
        return "{:s}{:d}_{:d}_{:d}_{:s}_{:d}_{:.4f}_index{:d}_seed{:d}/".format(
            config.DATASET, size, config.n_images, n_base, config.OPTIM, config.BATCH_SIZE, config.LR, index, seed_run
        )


def build_model(config, n_base: int, device: torch.device):
    m = Unet.UNet(
        input_channels=config.IMG_SHAPE[0],
        output_channels=config.IMG_SHAPE[0],
        base_channels=n_base,
        base_channels_multiples=(1, 2, 4),
        apply_attention=(False, True, True),
        dropout_rate=0.1,
    )
    return m.to(device)


@torch.no_grad()
def eval_checkpoint_loss_avg_over_timesteps(
    model,
    df,
    config,
    X0: torch.Tensor,
    t_values: torch.Tensor,
    eval_batch: int,
    seed_base: int,
) -> tuple[float, float, int]:
    """
    Computes test MSE of noise prediction averaged over:
      - all timesteps in t_values
      - all samples in X0 (same subset for all checkpoints)

    Returns: (mean, sem, count)
      where 'count' is number of scalar loss samples aggregated = len(X0)*len(t_values)
    """
    loss_fn = torch.nn.MSELoss(reduction="none")

    global_sum = 0.0
    global_sumsq = 0.0
    global_count = 0

    # progress bar over timesteps
    for t in tqdm(t_values.tolist(), desc="  Timesteps", leave=False, dynamic_ncols=True):
        # deterministic per-t noise for comparability across checkpoints
        torch.manual_seed(seed_base + int(t))
        torch.cuda.manual_seed_all(seed_base + int(t))

        ts = torch.full((X0.shape[0],), int(t), device=X0.device, dtype=torch.long)
        X_t, noise_t, _ = Diffusion.forward_diffusion(df, X0, ts, config, return_std=True)

        # batch over the N_eval samples
        for start in range(0, X0.shape[0], eval_batch):
            end = min(start + eval_batch, X0.shape[0])
            x_batch = X_t[start:end]
            noise_batch = noise_t[start:end]
            t_batch = ts[start:end]

            noise_pred = model(x_batch, t_batch)
            l = loss_fn(noise_pred, noise_batch).mean(dim=(1, 2, 3))  # [B]

            l_np = l.float().cpu().numpy()
            global_sum += float(l_np.sum())
            global_sumsq += float((l_np ** 2).sum())
            global_count += int(l_np.size)

    mean = global_sum / global_count
    var = max(global_sumsq / global_count - mean**2, 0.0)
    std = np.sqrt(var)
    sem = std / np.sqrt(global_count)
    return float(mean), float(sem), int(global_count)


# =============================================================================
# Config + data
# =============================================================================
config = cfg.load_config(args.dataset)
config.DEVICE = args.device
config.OPTIM = args.optim
config.BATCH_SIZE = int(args.batch_size)
config.LR = float(args.learning_rate)

n_base = int(args.nbase)
size = int(args.img_size)
device = torch.device(config.DEVICE)

if args.out_dire is not None:
    config.path_save = args.out_dire

# Load test set once (large pool), then pick a fixed subset
config.n_images = 40000
_, bigb = cfg.load_training_data(config, 0, loadtest=True)
X_test_all = bigb

N_total = len(X_test_all)
N_eval = min(int(args.eval_N), N_total)
if N_eval < int(args.eval_N):
    print(f"[WARN] Requested eval_N={args.eval_N}, but testset has only {N_total}. Using N_eval={N_eval}.")

g = torch.Generator(device="cpu").manual_seed(args.seed)
idx = torch.randperm(N_total, generator=g)[:N_eval]
X0 = X_test_all[idx].to(device)

# diffusion config + timesteps
df = Diffusion.DiffusionConfig(
    n_steps=config.TIMESTEPS,
    img_shape=config.IMG_SHAPE,
    device=config.DEVICE,
)

t_values = torch.arange(0, df.n_steps, device=device, dtype=torch.long)
# sanity check (will catch the off-by-one immediately)
assert int(t_values.min()) == 0
assert int(t_values.max()) == df.n_steps - 1

# training checkpoints
training_times = np.linspace(10, 5000 - 1, 50, dtype=int)[1:]
print(f"Evaluating {len(training_times)} checkpoints.")

# IMPORTANT: set n_images for the run folder name
config.n_images = int(args.num)

type_model = build_type_model(config, size, n_base, args.index, args.seed_run)


# =============================================================================
# Evaluate all checkpoints
# =============================================================================
model = build_model(config, n_base, device)
model.eval()

loss_means = []
loss_sems = []
counts = []

for ckpt in tqdm(training_times.tolist(), desc="Checkpoints", dynamic_ncols=True):
    ckpt_path = os.path.join(config.path_save, type_model, "Models", f"Model_epoch_{int(ckpt)}")
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    model = loader.load_model(model, ckpt_path)
    model.eval()

    mean_loss, sem_loss, count = eval_checkpoint_loss_avg_over_timesteps(
        model=model,
        df=df,
        config=config,
        X0=X0,
        t_values=t_values,
        eval_batch=int(args.eval_batch),
        seed_base=int(args.seed),
    )

    loss_means.append(mean_loss)
    loss_sems.append(sem_loss)
    counts.append(count)

loss_means = np.array(loss_means, dtype=np.float64)
loss_sems = np.array(loss_sems, dtype=np.float64)
counts = np.array(counts, dtype=np.int64)

# minimum
min_idx = int(np.argmin(loss_means))
min_checkpoint = int(training_times[min_idx])
min_loss = float(loss_means[min_idx])
min_sem = float(loss_sems[min_idx])

print("\n=== Minimum test loss across checkpoints ===")
print(f"min_loss = {min_loss:.6g} ± {min_sem:.6g} at checkpoint {min_checkpoint} (idx {min_idx})")


# =============================================================================
# Save + plot
# =============================================================================
out_dir = os.path.join(config.path_save, "Losses_over_checkpoints")
os.makedirs(out_dir, exist_ok=True)

tag = f"{config.DATASET}{size}_{config.n_images}_{n_base}_{config.OPTIM}_{config.BATCH_SIZE}_{config.LR:.4f}_index{args.index}"
if args.seed_run is not None:
    tag += f"_seed{args.seed_run}"

npz_path = os.path.join(out_dir, f"test_mse_avg_over_timesteps_vs_checkpoint_{tag}.npz")
png_path = os.path.join(out_dir, f"test_mse_avg_over_timesteps_vs_checkpoint_{tag}.png")

np.savez(
    npz_path,
    training_times=np.array(training_times, dtype=np.int64),
    loss_means=loss_means,
    loss_sems=loss_sems,
    counts=counts,
    # minimum info
    min_idx=np.array(min_idx, dtype=np.int64),
    min_checkpoint=np.array(min_checkpoint, dtype=np.int64),
    min_loss=np.array(min_loss, dtype=np.float64),
    min_sem=np.array(min_sem, dtype=np.float64),
    # eval metadata
    eval_N=np.array(N_eval, dtype=np.int64),
    eval_batch=np.array(int(args.eval_batch), dtype=np.int64),
    t_base=np.array(int(args.t_base), dtype=np.int64),
    n_timesteps=np.array(int(t_values.numel()), dtype=np.int64),
    # full args for reproducibility
    args_json=np.array(json.dumps(vars(args), sort_keys=True)),
)

plt.figure(figsize=(8.5, 4.8))
plt.errorbar(training_times, loss_means, yerr=loss_sems, fmt="-o", capsize=2,
             linewidth=1.4, markersize=3.5, markerfacecolor='white', markeredgewidth=0.9)

# mark minimum
plt.axvline(min_checkpoint, linestyle="--", linewidth=1.0)
plt.scatter([min_checkpoint], [min_loss], zorder=5)

plt.grid(True, alpha=0.3)
plt.xlabel("Training checkpoint")
plt.ylabel("Test MSE (avg over all timesteps + samples)")
plt.title(f"Test loss vs checkpoint\nmin={min_loss:.4g} @ {min_checkpoint}")

# simple legend that explicitly reports the min checkpoint
plt.legend([f"loss curve (min @ {min_checkpoint})"], frameon=False, loc="best")

if args.xlog:
    plt.xscale("log")
if args.ylog:
    plt.yscale("log")

plt.tight_layout()
plt.savefig(png_path, bbox_inches="tight")
plt.close()

print("\nSaved:")
print("  NPZ:", npz_path)
print("  PNG:", png_path)
