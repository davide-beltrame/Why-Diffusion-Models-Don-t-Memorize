#!/usr/bin/env python3
"""Cross-level score comparison for the CelebA complexity experiment.

Fixes model B at a single checkpoint and sweeps model A over all its
checkpoints, computing cosine similarity between their score predictions
on a shared test set. Used to build the "staircase" plot.

Usage:
    python compare_scores_cross.py \
        --model-a-dir ../../Saves_new/CelebA32_1024_32_Adam_512_0.0001_index0_PCA_L3/ \
        --model-b-dir ../../Saves_new/CelebA32_1024_32_Adam_512_0.0001_index0_PCA_L0/ \
        --ckpt-b 450 \
        --data-test-file ../../Data/CelebA_filtered/CelebA32_PCA_L3_test.pt \
        --out-dir ../../Saves_new/CrossLevel/ \
        -s 32 -W 32 -t 100 -Ns 512
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import os
import sys
import json
import argparse

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

sys.path.insert(1, "../Utils/")
import Diffusion
import Unet
import cfg
import loader

parser = argparse.ArgumentParser("Cross-level score cosine similarity.")
parser.add_argument("--model-a-dir", type=str, required=True,
                    help="Saves_new subfolder for model A (the one swept over checkpoints).")
parser.add_argument("--model-b-dir", type=str, required=True,
                    help="Saves_new subfolder for model B (frozen at --ckpt-b).")
parser.add_argument("--ckpt-b", type=int, required=True,
                    help="Checkpoint id at which model B is frozen.")
parser.add_argument("-s", "--img_size", type=int, default=32)
parser.add_argument("-W", "--nbase", type=int, default=32)
parser.add_argument("-t", "--time", type=int, default=100,
                    help="Diffusion timestep for score evaluation.")
parser.add_argument("--times", type=str, default=None,
                    help="Optional comma-separated timesteps (e.g. '100,200,400'). Overrides --time.")
parser.add_argument("-Ns", "--Nsamples", type=int, default=512)
parser.add_argument("--out-dir", type=str, required=True)
parser.add_argument("--device", type=str, default="cuda:0")
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--data-test-file", type=str, default=None,
                    help="Path to pre-split test .pt tensor (unfiltered).")
parser.add_argument("--label", type=str, default="cross",
                    help="Label for output filenames.")
args = parser.parse_args()
print(args)


def parse_times(args_obj, t_max):
    if args_obj.times is None:
        out = [int(args_obj.time)]
    else:
        parts = [p.strip() for p in args_obj.times.split(",") if p.strip()]
        if not parts:
            raise ValueError("--times was provided but no valid values were parsed")
        out = sorted(set(int(p) for p in parts))

    for t in out:
        if t < 0 or t >= t_max:
            raise ValueError(f"Invalid timestep {t}; expected in [0, {t_max - 1}]")
    return out

torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

# Config
config = cfg.load_config("CelebA")
config.DEVICE = args.device
n_base = args.nbase
size = args.img_size

# Load test data
if args.data_test_file is not None:
    import torchvision.transforms as _tfms
    _test_raw = torch.load(os.path.expanduser(args.data_test_file), weights_only=True)
    _mean = torch.mean(_test_raw, axis=[0, 2, 3])
    _std = torch.ones(config.IMG_SHAPE[0])
    _tfm = _tfms.Compose([_tfms.Normalize(_mean, _std)])
    test_data = torch.stack([_tfm(x) for x in _test_raw])
    config.mean = _mean
    config.std = _std
else:
    config.n_images = 40000
    _, test_data = cfg.load_training_data(config, 0, loadtest=True)
    # test_data is a TransformedDataset; materialise it
    test_data = torch.stack([test_data[i] for i in range(len(test_data))])

Nsamples = min(args.Nsamples, len(test_data))
test_data = test_data[:Nsamples].to(config.DEVICE)

# Diffusion config
df = Diffusion.DiffusionConfig(
    n_steps=config.TIMESTEPS,
    img_shape=config.IMG_SHAPE,
    device=config.DEVICE,
)
selected_times = parse_times(args, config.TIMESTEPS)
is_multi_time = len(selected_times) > 1

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


# Discover checkpoints of model A
models_dir_a = os.path.join(args.model_a_dir, "Models")
ckpt_files = [f for f in os.listdir(models_dir_a) if f.startswith("Model_epoch_")]
ckpt_ids = sorted([int(f.split("_")[-1]) for f in ckpt_files])
print(f"Found {len(ckpt_ids)} checkpoints for model A")

# Load model B (frozen checkpoint)
model_b = build_model()
path_b = os.path.join(args.model_b_dir, "Models", f"Model_epoch_{args.ckpt_b}")
if not os.path.exists(path_b):
    raise FileNotFoundError(f"Model B checkpoint not found: {path_b}")
model_b = loader.load_model(model_b, path_b)
model_b.eval()


def run_one_time(t_eval: int):
    selected_time = torch.tensor([t_eval], device=config.DEVICE, dtype=torch.long)

    # Noisy test images (deterministic noise)
    torch.manual_seed(args.seed + int(t_eval))
    torch.cuda.manual_seed_all(args.seed + int(t_eval))
    ts = selected_time.repeat(test_data.shape[0])
    X_t, _, _ = Diffusion.forward_diffusion(df, test_data, ts, config, return_std=True)

    # Precompute model B scores
    batch_gen = 512
    Ns = max(1, Nsamples // batch_gen)
    with torch.no_grad():
        scores_b_list = []
        for bi in range(Ns):
            sl = slice(bi * batch_gen, min((bi + 1) * batch_gen, Nsamples))
            t_in = selected_time.repeat(X_t[sl].shape[0])
            scores_b_list.append(model_b(X_t[sl], t_in))
        scores_b = torch.cat(scores_b_list, dim=0)

    # Sweep over model A checkpoints
    model_a = build_model()
    means_cos = []
    sems_cos = []
    means_mse = []
    sems_mse = []

    for ckpt_a in tqdm(ckpt_ids, desc=f"Sweep A checkpoints @ t={t_eval}"):
        path_a = os.path.join(models_dir_a, f"Model_epoch_{ckpt_a}")
        model_a = loader.load_model(model_a, path_a, verbose=False)
        model_a.eval()

        cos_list = []
        mse_list = []
        with torch.no_grad():
            for bi in range(Ns):
                sl = slice(bi * batch_gen, min((bi + 1) * batch_gen, Nsamples))
                t_in = selected_time.repeat(X_t[sl].shape[0])
                scores_a = model_a(X_t[sl], t_in)
                scores_b_sl = scores_b[sl]
                cos = F.cosine_similarity(scores_a.flatten(1), scores_b_sl.flatten(1), dim=1)
                mse = F.mse_loss(scores_a, scores_b_sl, reduction='none').mean(dim=(1, 2, 3))
                cos_list.append(cos.cpu().numpy())
                mse_list.append(mse.cpu().numpy())

        vals_cos = np.concatenate(cos_list)
        vals_mse = np.concatenate(mse_list)
        means_cos.append(float(np.mean(vals_cos)))
        sems_cos.append(float(np.std(vals_cos) / np.sqrt(len(vals_cos))))
        means_mse.append(float(np.mean(vals_mse)))
        sems_mse.append(float(np.std(vals_mse) / np.sqrt(len(vals_mse))))

    means_cos = np.array(means_cos)
    sems_cos = np.array(sems_cos)
    means_mse = np.array(means_mse)
    sems_mse = np.array(sems_mse)
    ckpt_ids_np = np.array(ckpt_ids)

    # Save
    os.makedirs(args.out_dir, exist_ok=True)
    tag = f"{args.label}_t{t_eval}" if is_multi_time else args.label
    npz_path = os.path.join(args.out_dir, f"cross_scores_{tag}.npz")
    np.savez(
        npz_path,
        ckpt_ids_a=ckpt_ids_np,
        ckpt_b=np.array(args.ckpt_b),
        time=np.array(int(t_eval), dtype=np.int64),
        means=means_cos,      # backward-compatible alias for cosine
        sems=sems_cos,        # backward-compatible alias for cosine
        means_cos=means_cos,
        sems_cos=sems_cos,
        means_mse=means_mse,
        sems_mse=sems_mse,
        args_json=np.array(json.dumps(vars(args), sort_keys=True)),
    )

    # Quick plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5), sharex=True)

    ax1.errorbar(
        ckpt_ids_np,
        means_cos,
        yerr=sems_cos,
        fmt="o-",
        capsize=2,
        linewidth=1.4,
        markersize=3,
        markerfacecolor="white",
        markeredgewidth=0.8,
        color="#1f77b4",
    )
    ax1.set_xlabel("Model A checkpoint")
    ax1.set_ylabel("Cosine similarity (A vs B)")
    ax1.grid(True, alpha=0.3)

    ax2.errorbar(
        ckpt_ids_np,
        means_mse,
        yerr=sems_mse,
        fmt="o-",
        capsize=2,
        linewidth=1.4,
        markersize=3,
        markerfacecolor="white",
        markeredgewidth=0.8,
        color="#d62728",
    )
    ax2.set_xlabel("Model A checkpoint")
    ax2.set_ylabel("MSE of score predictions")
    ax2.grid(True, alpha=0.3)

    fig.suptitle(f"Cross-level @ t={t_eval}: A sweep vs B@{args.ckpt_b}")
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.94))
    png_path = os.path.join(args.out_dir, f"cross_scores_{tag}.png")
    fig.savefig(png_path, bbox_inches="tight")
    plt.close(fig)

    print(f"\nSaved: {npz_path}")
    print(f"       {png_path}")


for t_eval in selected_times:
    run_one_time(int(t_eval))
