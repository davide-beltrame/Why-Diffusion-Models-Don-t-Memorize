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
parser.add_argument("--split", type=str, choices=["train", "test", "both"], default="test",
                    help="Which split to evaluate. Default keeps legacy behavior (test).")
parser.add_argument("--subset_N", type=int, default=None,
                    help="If set, evaluate exactly subset_N images from selected split(s).")
parser.add_argument("--subset_seed", type=int, default=0,
                    help="Seed for deterministic subset selection in each split.")
parser.add_argument("--noise_seed_base", type=int, default=None,
                    help="Base seed for forward-noise reproducibility. Defaults to --seed if omitted.")
parser.add_argument("--checkpoint_grid", type=str, choices=["paper49", "cfg"], default="paper49",
                    help="Checkpoint grid: paper49 matches sample-split scripts; cfg uses cfg.get_training_times().")
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
parser.add_argument("--data-file", type=str, default=None,
                    help="Override: path to a pre-split train .pt tensor.")
parser.add_argument("--data-test-file", type=str, default=None,
                    help="Override: path to a pre-split test .pt tensor.")
parser.add_argument("--suffix", type=str, default=None,
                    help="Custom suffix appended to model folder name (e.g., _PCA_L0).")

args = parser.parse_args()
print(args)

torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True


# =============================================================================
# Helpers
# =============================================================================

def build_type_model(config, size, n_base, index: int, seed_run: int | None, suffix: str | None = None) -> str:
    if seed_run is None:
        base = "{:s}{:d}_{:d}_{:d}_{:s}_{:d}_{:.4f}_index{:d}".format(
            config.DATASET, size, config.n_images, n_base, config.OPTIM, config.BATCH_SIZE, config.LR, index
        )
    else:
        base = "{:s}{:d}_{:d}_{:d}_{:s}_{:d}_{:.4f}_index{:d}_seed{:d}".format(
            config.DATASET, size, config.n_images, n_base, config.OPTIM, config.BATCH_SIZE, config.LR, index, seed_run
        )
    if suffix:
        base += "_" + suffix.lstrip("_")
    return base + "/"


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

# Load train/test pools once, then pick fixed subsets
if args.data_file is not None:
    import torchvision.transforms as _tfms
    _train_raw = torch.load(os.path.expanduser(args.data_file), weights_only=True)
    _test_raw = torch.load(os.path.expanduser(args.data_test_file), weights_only=True) if args.data_test_file else None
    _mean = torch.mean(_train_raw, axis=[0, 2, 3])
    _std = torch.ones(config.IMG_SHAPE[0])
    _tfm = _tfms.Compose([_tfms.Normalize(_mean, _std)])
    big_train = torch.stack([_tfm(x) for x in _train_raw])
    big_test = torch.stack([_tfm(x) for x in _test_raw]) if _test_raw is not None else torch.empty(0)
    config.mean = _mean
    config.std = _std
    print(f'Loaded custom data: train={big_train.shape}, test={big_test.shape}')
else:
    config.n_images = 40000
    big_train, big_test = cfg.load_training_data(config, 0, loadtest=True)

def select_subset(X_all: torch.Tensor, want: int, seed: int) -> torch.Tensor:
    n_total = len(X_all)
    n_eval = min(int(want), n_total)
    if n_eval < int(want):
        print(f"[WARN] Requested {want}, but split has only {n_total}. Using {n_eval}.")
    g_local = torch.Generator(device="cpu").manual_seed(seed)
    idx_local = torch.randperm(n_total, generator=g_local)[:n_eval]
    return X_all[idx_local].to(device)


if args.subset_N is not None:
    n_req = int(args.subset_N)
else:
    n_req = int(args.eval_N)

X_train_eval = None
X_test_eval = None
if args.split in ("train", "both"):
    X_train_eval = select_subset(big_train, n_req, int(args.subset_seed))
if args.split in ("test", "both"):
    X_test_eval = select_subset(big_test, n_req, int(args.subset_seed) + 1)

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
if args.checkpoint_grid == "paper49":
    training_times = np.linspace(10, 5000 - 1, 50, dtype=int)[1:]
else:
    training_times = np.array(cfg.get_training_times(), dtype=int)
print(f"Evaluating {len(training_times)} checkpoints.")

# IMPORTANT: set n_images for the run folder name
config.n_images = int(args.num)

type_model = build_type_model(config, size, n_base, args.index, args.seed_run, suffix=args.suffix)


# =============================================================================
# Evaluate all checkpoints
# =============================================================================
model = build_model(config, n_base, device)
model.eval()

if args.noise_seed_base is None:
    noise_seed_base = int(args.seed)
else:
    noise_seed_base = int(args.noise_seed_base)

split_to_means = {"train": [], "test": []}
split_to_sems = {"train": [], "test": []}
split_to_counts = {"train": [], "test": []}

for ckpt in tqdm(training_times.tolist(), desc="Checkpoints", dynamic_ncols=True):
    ckpt_path = os.path.join(config.path_save, type_model, "Models", f"Model_epoch_{int(ckpt)}")
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    model = loader.load_model(model, ckpt_path)
    model.eval()

    if args.split in ("train", "both"):
        mean_loss, sem_loss, count = eval_checkpoint_loss_avg_over_timesteps(
            model=model,
            df=df,
            config=config,
            X0=X_train_eval,
            t_values=t_values,
            eval_batch=int(args.eval_batch),
            seed_base=noise_seed_base,
        )
        split_to_means["train"].append(mean_loss)
        split_to_sems["train"].append(sem_loss)
        split_to_counts["train"].append(count)

    if args.split in ("test", "both"):
        mean_loss, sem_loss, count = eval_checkpoint_loss_avg_over_timesteps(
            model=model,
            df=df,
            config=config,
            X0=X_test_eval,
            t_values=t_values,
            eval_batch=int(args.eval_batch),
            seed_base=noise_seed_base,
        )
        split_to_means["test"].append(mean_loss)
        split_to_sems["test"].append(sem_loss)
        split_to_counts["test"].append(count)

for k in ("train", "test"):
    split_to_means[k] = np.array(split_to_means[k], dtype=np.float64)
    split_to_sems[k] = np.array(split_to_sems[k], dtype=np.float64)
    split_to_counts[k] = np.array(split_to_counts[k], dtype=np.int64)

if args.split in ("test", "both") and split_to_means["test"].size > 0:
    min_idx = int(np.argmin(split_to_means["test"]))
    min_checkpoint = int(training_times[min_idx])
    min_loss = float(split_to_means["test"][min_idx])
    min_sem = float(split_to_sems["test"][min_idx])
    print("\n=== Minimum test loss across checkpoints ===")
    print(f"min_loss = {min_loss:.6g} ± {min_sem:.6g} at checkpoint {min_checkpoint} (idx {min_idx})")
else:
    min_idx = -1
    min_checkpoint = -1
    min_loss = np.nan
    min_sem = np.nan


# =============================================================================
# Save + plot
# =============================================================================
out_dir = os.path.join(config.path_save, "Losses_over_checkpoints_timing")
os.makedirs(out_dir, exist_ok=True)

tag = f"{config.DATASET}{size}_{config.n_images}_{n_base}_{config.OPTIM}_{config.BATCH_SIZE}_{config.LR:.4f}_index{args.index}"
if args.seed_run is not None:
    tag += f"_seed{args.seed_run}"
if args.suffix is not None:
    tag += f"_{args.suffix.lstrip('_')}"

npz_path = os.path.join(out_dir, f"timing_loss_avg_over_timesteps_vs_checkpoint_{tag}.npz")
png_path = os.path.join(out_dir, f"timing_loss_avg_over_timesteps_vs_checkpoint_{tag}.png")

np.savez(
    npz_path,
    training_times=np.array(training_times, dtype=np.int64),
    split=np.array(args.split),
    train_loss_means=split_to_means["train"],
    train_loss_sems=split_to_sems["train"],
    train_counts=split_to_counts["train"],
    test_loss_means=split_to_means["test"],
    test_loss_sems=split_to_sems["test"],
    test_counts=split_to_counts["test"],
    # minimum info
    min_idx=np.array(min_idx, dtype=np.int64),
    min_checkpoint=np.array(min_checkpoint, dtype=np.int64),
    min_loss=np.array(min_loss, dtype=np.float64),
    min_sem=np.array(min_sem, dtype=np.float64),
    # eval metadata
    subset_N=np.array(n_req, dtype=np.int64),
    eval_batch=np.array(int(args.eval_batch), dtype=np.int64),
    t_base=np.array(int(args.t_base), dtype=np.int64),
    checkpoint_grid=np.array(args.checkpoint_grid),
    subset_seed=np.array(int(args.subset_seed), dtype=np.int64),
    noise_seed_base=np.array(noise_seed_base, dtype=np.int64),
    n_timesteps=np.array(int(t_values.numel()), dtype=np.int64),
    # full args for reproducibility
    args_json=np.array(json.dumps(vars(args), sort_keys=True)),
)

plt.figure(figsize=(8.5, 4.8))
if split_to_means["train"].size > 0:
    plt.errorbar(training_times, split_to_means["train"], yerr=split_to_sems["train"], fmt="-o", capsize=2,
                 linewidth=1.2, markersize=3.2, markerfacecolor='white', markeredgewidth=0.9, label="train")
if split_to_means["test"].size > 0:
    plt.errorbar(training_times, split_to_means["test"], yerr=split_to_sems["test"], fmt="-s", capsize=2,
                 linewidth=1.2, markersize=3.2, markerfacecolor='white', markeredgewidth=0.9, label="test")

if min_checkpoint >= 0:
    plt.axvline(min_checkpoint, linestyle="--", linewidth=1.0)
    plt.scatter([min_checkpoint], [min_loss], zorder=5)

plt.grid(True, alpha=0.3)
plt.xlabel("Training checkpoint")
plt.ylabel("MSE (avg over all timesteps + samples)")
if min_checkpoint >= 0:
    plt.title(f"Loss vs checkpoint\nmin test={min_loss:.4g} @ {min_checkpoint}")
else:
    plt.title("Loss vs checkpoint")
if split_to_means["train"].size > 0 or split_to_means["test"].size > 0:
    plt.legend(frameon=False, loc="best")

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
