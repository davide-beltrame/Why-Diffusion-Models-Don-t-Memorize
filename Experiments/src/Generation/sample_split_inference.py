#!/usr/bin/env python3
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import os
import sys
import re
import math
import argparse
import itertools
from pathlib import Path
from typing import Optional, Dict, Tuple, List

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(1, "../Utils/")
import Diffusion
import Unet
import cfg
import loader

# Dataset helpers

def _is_dataset(obj) -> bool:
    return (hasattr(obj, "__len__") and hasattr(obj, "__getitem__") and not torch.is_tensor(obj))


def _extract_image_from_sample(sample):
    if torch.is_tensor(sample):
        return sample
    if isinstance(sample, (tuple, list)):
        return sample[0]
    if isinstance(sample, dict):
        for k in ("image", "img", "x", "data"):
            if k in sample:
                return sample[k]
        raise KeyError(f"Dict sample keys {list(sample.keys())} don't include a known image key.")
    raise TypeError(f"Unsupported sample type: {type(sample)}")


@torch.no_grad()
def _fetch_images_by_indices(dataset, indices_cpu: torch.Tensor) -> torch.Tensor:
    imgs = []
    for idx in indices_cpu.tolist():
        img = _extract_image_from_sample(dataset[idx])
        if torch.is_tensor(img) and img.dim() == 2:
            img = img.unsqueeze(0)
        if not torch.is_tensor(img):
            raise TypeError(f"Dataset returned non-tensor image at idx={idx}: {type(img)}")
        imgs.append(img.detach().cpu())
    return torch.stack(imgs, dim=0)


# Plotting helpers

def _tensor_to_imshow(img_chw: torch.Tensor):
    x = img_chw.detach().cpu().float()
    if x.numel() == 0:
        return np.zeros((1, 1)), "gray"
    if x.min().item() < 0.0:
        x = (x + 1.0) / 2.0
    x = x.clamp(0.0, 1.0)
    if x.shape[0] == 1:
        return x[0].numpy(), "gray"
    return x.permute(1, 2, 0).numpy(), None


def save_pairs_with_nn_png(
        nn_a: torch.Tensor, xa: torch.Tensor, xb: torch.Tensor, nn_b: torch.Tensor,
        pair_sims: torch.Tensor,
        out_path: str,
        title_nn_a="NN train A", title_a="Model A", title_b="Model B", title_nn_b="NN train B",
        nn_sims_a: Optional[torch.Tensor] = None, nn_sims_b: Optional[torch.Tensor] = None,
    ):
    K = xa.shape[0]
    out_path = str(out_path)

    if K == 0:
        fig, ax = plt.subplots(figsize=(7, 2))
        ax.text(0.5, 0.5, "No samples to display", ha="center", va="center")
        ax.axis("off")
        fig.tight_layout()
        fig.savefig(out_path, dpi=200)
        plt.close(fig)
        return
    fig, axes = plt.subplots(K, 4, figsize=(12, 2.6 * K))

    if K == 1:
        axes = np.array([axes])

    axes[0, 0].set_title(title_nn_a, fontsize=9)
    axes[0, 1].set_title(title_a, fontsize=9)
    axes[0, 2].set_title(title_b, fontsize=9)
    axes[0, 3].set_title(title_nn_b, fontsize=9)

    for i in range(K):
        nn_a_np, nn_a_cmap = _tensor_to_imshow(nn_a[i])
        a_np, a_cmap       = _tensor_to_imshow(xa[i])
        b_np, b_cmap       = _tensor_to_imshow(xb[i])
        nn_b_np, nn_b_cmap = _tensor_to_imshow(nn_b[i])

        axes[i, 0].imshow(nn_a_np, cmap=nn_a_cmap)
        axes[i, 1].imshow(a_np, cmap=a_cmap)
        axes[i, 2].imshow(b_np, cmap=b_cmap)
        axes[i, 3].imshow(nn_b_np, cmap=nn_b_cmap)

        axes[i, 0].set_ylabel(f"{pair_sims[i].item():.3f}", rotation=0, labelpad=30, va="center")

        if nn_sims_a is not None and nn_sims_a.numel() == K:
            axes[i, 0].set_xlabel(f"cos={nn_sims_a[i].item():.3f}")
        if nn_sims_b is not None and nn_sims_b.numel() == K:
            axes[i, 3].set_xlabel(f"cos={nn_sims_b[i].item():.3f}")

        for c in range(4):
            axes[i, c].set_xticks([])
            axes[i, c].set_yticks([])

    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


# Save arrays for figures (.npz)

def save_topk_pairs_npz(train_a, train_b, ia_cpu, ib_cpu, sims_cpu, out_path_npz):
    ia_cpu = ia_cpu.detach().cpu().to(torch.long)
    ib_cpu = ib_cpu.detach().cpu().to(torch.long)
    sims_cpu = sims_cpu.detach().cpu().to(torch.float32)

    A = fetch_train_by_idx(train_a, ia_cpu).detach().cpu().float()
    B = fetch_train_by_idx(train_b, ib_cpu).detach().cpu().float()

    np.savez_compressed(
        str(out_path_npz),
        ia=ia_cpu.numpy(),
        ib=ib_cpu.numpy(),
        sims=sims_cpu.numpy(),
        A_imgs=A.numpy(),  # [K,C,H,W]
        B_imgs=B.numpy(),  # [K,C,H,W]
    )


def save_rep_pairs_with_nn_npz(
    rep_idx_cpu: torch.Tensor,
    rep_sims_cpu: torch.Tensor,
    rep_xa_cpu: torch.Tensor,
    rep_xb_cpu: torch.Tensor,
    nn_idx_a_cpu: torch.Tensor,
    nn_sims_a_cpu: torch.Tensor,
    nn_imgs_a_cpu: torch.Tensor,
    nn_idx_b_cpu: torch.Tensor,
    nn_sims_b_cpu: torch.Tensor,
    nn_imgs_b_cpu: torch.Tensor,
    out_path_npz: str,
):
    np.savez_compressed(
        str(out_path_npz),
        rep_idx=rep_idx_cpu.detach().cpu().to(torch.long).numpy(),
        pair_sims=rep_sims_cpu.detach().cpu().to(torch.float32).numpy(),
        xa=rep_xa_cpu.detach().cpu().numpy(),      # [K,C,H,W]
        xb=rep_xb_cpu.detach().cpu().numpy(),      # [K,C,H,W]
        nn_idx_a=nn_idx_a_cpu.detach().cpu().to(torch.long).numpy(),
        nn_sims_a=nn_sims_a_cpu.detach().cpu().to(torch.float32).numpy(),
        nn_a=nn_imgs_a_cpu.detach().cpu().numpy(), # [K,C,H,W]
        nn_idx_b=nn_idx_b_cpu.detach().cpu().to(torch.long).numpy(),
        nn_sims_b=nn_sims_b_cpu.detach().cpu().to(torch.float32).numpy(),
        nn_b=nn_imgs_b_cpu.detach().cpu().numpy(), # [K,C,H,W]
    )


# Features + similarity

@torch.no_grad()
def _pixel_features(x: torch.Tensor) -> torch.Tensor:
    # per-image per-channel standardization, then cosine on flattened pixels
    x = x.float()
    x = (x - x.mean(dim=(2, 3), keepdim=True)) / (x.std(dim=(2, 3), keepdim=True) + 1e-12)
    f = x.flatten(1)
    return torch.nn.functional.normalize(f, dim=1)


@torch.no_grad()
def _to_0_1(x: torch.Tensor) -> torch.Tensor:
    x = x.float()
    if x.min().item() < 0.0:
        x = (x + 1.0) / 2.0
    return x.clamp(0.0, 1.0)


def make_resnet50_feature_extractor(device: str):
    try:
        import torchvision
        backbone = torchvision.models.resnet50(weights=torchvision.models.ResNet50_Weights.DEFAULT)
        feat = torch.nn.Sequential(*list(backbone.children())[:-1]).to(device).eval()
    except Exception as e:
        raise RuntimeError(
            "Failed to init ResNet-50. If weights aren't cached, use --metric pixel or pre-cache them."
        ) from e

    @torch.no_grad()
    def _resnet_features(x: torch.Tensor) -> torch.Tensor:
        x = _to_0_1(x)
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)
        elif x.shape[1] != 3:
            x = x[:, :3]

        x = torch.nn.functional.interpolate(x, size=(224, 224), mode="bilinear", align_corners=False)
        mean = torch.tensor([0.485, 0.456, 0.406], device=x.device).view(1, 3, 1, 1)
        std  = torch.tensor([0.229, 0.224, 0.225], device=x.device).view(1, 3, 1, 1)
        x = (x - mean) / std

        f = feat(x).flatten(1)
        return torch.nn.functional.normalize(f, dim=1)

    return _resnet_features


@torch.no_grad()
def compute_features(x: torch.Tensor, metric: str, resnet_fn=None) -> torch.Tensor:
    if metric == "pixel":
        return _pixel_features(x)
    if metric == "resnet":
        if resnet_fn is None:
            raise RuntimeError("resnet_fn is None but metric='resnet'")
        return resnet_fn(x)
    raise ValueError(metric)


@torch.no_grad()
def cosine_sim(xa: torch.Tensor, xb: torch.Tensor, metric: str, resnet_fn=None) -> torch.Tensor:
    fa = compute_features(xa, metric, resnet_fn)
    fb = compute_features(xb, metric, resnet_fn)
    return (fa * fb).sum(dim=1)  # [B]


# Training NN indexing

@torch.no_grad()
def build_train_index(
    train_images,
    metric: str,
    device: str,
    resnet_fn,
    batch_size: int,
    store_dtype: torch.dtype,
    num_workers: int = 0,
):
    """
    Returns:
      train_ref: CPU Tensor [N,C,H,W] OR Dataset
      feats_cpu: CPU [N,D] (store_dtype), normalized
    """
    if torch.is_tensor(train_images) or isinstance(train_images, np.ndarray):
        if not torch.is_tensor(train_images):
            train_images = torch.as_tensor(train_images)
        train_cpu = train_images.detach().cpu()
        feats = []
        for s in tqdm(range(0, train_cpu.shape[0], batch_size), desc="Index train feats", leave=False):
            x = train_cpu[s:s + batch_size].to(device, non_blocking=True)
            f = compute_features(x, metric, resnet_fn).detach().cpu().to(store_dtype)
            feats.append(f)
        return train_cpu, torch.cat(feats, dim=0)

    if _is_dataset(train_images):
        ds = train_images
        pin = ("cuda" in str(device))
        dl = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=num_workers,
                        pin_memory=pin, drop_last=False)
        feats = []
        for batch in tqdm(dl, desc="Index train feats", leave=False):
            imgs = _extract_image_from_sample(batch)
            if imgs.dim() == 3:
                imgs = imgs.unsqueeze(0)
            imgs = imgs.to(device, non_blocking=True)
            f = compute_features(imgs, metric, resnet_fn).detach().cpu().to(store_dtype)
            feats.append(f)
        feats_cpu = torch.cat(feats, dim=0)
        if len(ds) != feats_cpu.shape[0]:
            raise RuntimeError(f"Feature count mismatch: len(ds)={len(ds)} vs feats={feats_cpu.shape[0]}")
        return ds, feats_cpu

    raise TypeError(f"train_images must be Tensor/ndarray or Dataset, got {type(train_images)}")


@torch.no_grad()
def nearest_train_scores_and_idx(
        gen_imgs: torch.Tensor,               # [B,C,H,W] on DEVICE
        train_feats_cpu: torch.Tensor,        # [N,D] CPU, normalized
        metric: str,
        device: str,
        resnet_fn,
        chunk_size: int = 8192,
    ):
    if gen_imgs.numel() == 0:
        return torch.empty((0,), dtype=torch.float32), torch.empty((0,), dtype=torch.long)

    f_gen = compute_features(gen_imgs.to(device, non_blocking=True), metric, resnet_fn).float()  # [B,D]
    B = f_gen.shape[0]

    best_scores = torch.full((B,), -1e9, device=device, dtype=torch.float32)
    best_idx = torch.zeros((B,), device=device, dtype=torch.long)

    N = train_feats_cpu.shape[0]
    for s in range(0, N, chunk_size):
        feats = train_feats_cpu[s:s + chunk_size].to(device, non_blocking=True).float()  # [M,D]
        scores = f_gen @ feats.T  # cosine (already normalized; extra renorm is harmless)
        scores = scores / (f_gen.norm(dim=1, keepdim=True) * feats.norm(dim=1).unsqueeze(0) + 1e-12)
        chunk_best, chunk_idx = scores.max(dim=1)
        better = chunk_best > best_scores
        best_scores[better] = chunk_best[better]
        best_idx[better] = chunk_idx[better] + s

    return best_scores.detach().cpu(), best_idx.detach().cpu()


@torch.no_grad()
def fetch_train_by_idx(train_ref, idx_cpu: torch.Tensor) -> torch.Tensor:
    if torch.is_tensor(train_ref):
        return train_ref[idx_cpu]
    return _fetch_images_by_indices(train_ref, idx_cpu)


# Train to train debug (top-k correlated + dup-check)
@torch.no_grad()
def topk_corr_pairs(train_a, train_b, device="cuda:0", k=10):
    def _len(ds):
        if torch.is_tensor(ds) or isinstance(ds, np.ndarray): return int(ds.shape[0])
        if _is_dataset(ds): return len(ds)
        raise TypeError(type(ds))

    def _get(ds, idx_cpu: torch.Tensor) -> torch.Tensor:
        if torch.is_tensor(ds):
            x = ds[idx_cpu]
        elif isinstance(ds, np.ndarray):
            x = torch.as_tensor(ds[idx_cpu.numpy()])
        else:
            x = _fetch_images_by_indices(ds, idx_cpu)
        return x.unsqueeze(0) if x.dim() == 3 else x  # [B,C,H,W]

    def _feat(x):
        x = x.float()
        mu  = x.mean(dim=(1,2,3), keepdim=True)
        sig = x.std(dim=(1,2,3), keepdim=True, unbiased=False)
        x = (x - mu) / (sig + 1e-12)
        return torch.nn.functional.normalize(x.flatten(1), dim=1)

    nA, nB = _len(train_a), _len(train_b)
    ia = torch.arange(nA, dtype=torch.long)
    ib = torch.arange(nB, dtype=torch.long)

    Fa = _feat(_get(train_a, ia).to(device)).cpu()  # [nA,D]
    Fb = _feat(_get(train_b, ib).to(device)).cpu()  # [nB,D]

    S = Fa @ Fb.T  # [nA,nB]
    kk = min(int(k), S.numel())
    vals, flat = torch.topk(S.flatten(), k=kk, largest=True)
    ia_top = (flat // nB).to(torch.long)
    ib_top = (flat %  nB).to(torch.long)
    return ia_top.cpu(), ib_top.cpu(), vals.cpu()


def save_topk_pairs_png(train_a, train_b, ia_cpu, ib_cpu, sims_cpu, out_path,
                        title_a="Train A", title_b="Train B"):
    K = int(len(sims_cpu))
    fig, axes = plt.subplots(K, 2, figsize=(6.5, 2.6 * K))
    if K == 1:
        axes = np.array([axes])

    axes[0,0].set_title(title_a)
    axes[0,1].set_title(title_b)

    A = fetch_train_by_idx(train_a, ia_cpu)
    B = fetch_train_by_idx(train_b, ib_cpu)

    for i in range(K):
        a_np, a_cmap = _tensor_to_imshow(A[i])
        b_np, b_cmap = _tensor_to_imshow(B[i])

        axes[i,0].imshow(a_np, cmap=a_cmap)
        axes[i,1].imshow(b_np, cmap=b_cmap)

        axes[i,0].set_ylabel(f"{float(sims_cpu[i]):.3f}", rotation=0, labelpad=30, va="center")

        for c in range(2):
            axes[i,c].set_xticks([])
            axes[i,c].set_yticks([])

    fig.tight_layout()
    fig.savefig(str(out_path), dpi=200)
    plt.close(fig)


@torch.no_grad()
def dup_check_trainsets(train_a, train_b, device="cuda:0", na=None, nb=None, thresh=0.995):
    def _len(ds):
        if torch.is_tensor(ds) or isinstance(ds, np.ndarray): return int(ds.shape[0])
        if _is_dataset(ds): return len(ds)
        raise TypeError(type(ds))

    def _get(ds, idx):
        if torch.is_tensor(ds): x = ds[idx]
        elif isinstance(ds, np.ndarray): x = torch.as_tensor(ds[idx.numpy()])
        else: x = _fetch_images_by_indices(ds, idx)
        return x.unsqueeze(0) if x.dim() == 3 else x

    def _feat(x):
        x = x.float()
        mu  = x.mean(dim=(1,2,3), keepdim=True)
        sig = x.std(dim=(1,2,3), keepdim=True, unbiased=False)
        x = (x - mu) / (sig + 1e-12)
        return torch.nn.functional.normalize(x.flatten(1), dim=1)

    nA, nB = _len(train_a), _len(train_b)
    na = nA if na is None else min(int(na), nA)
    nb = nB if nb is None else min(int(nb), nB)

    ia = torch.arange(na, dtype=torch.long)
    ib = torch.arange(nb, dtype=torch.long)

    Fa = _feat(_get(train_a, ia).to(device)).cpu()  # [na,D]
    Fb = _feat(_get(train_b, ib).to(device)).cpu()  # [nb,D]

    S = Fa @ Fb.T
    best, j = S.max(dim=1)
    hits = (best >= float(thresh)).nonzero().flatten()

    matches = [(int(ia[i]), int(ib[int(j[i])]), float(best[i])) for i in hits.tolist()]
    matches.sort(key=lambda t: t[2], reverse=True)
    return {"max_sim": float(best.max()), "matches": matches[:20]}


# Utility: batch plan

def make_batch_plan(Nsamples: int, batch_gen: int):
    batch_sizes = [batch_gen] * (Nsamples // batch_gen)
    rem = Nsamples % batch_gen
    if rem:
        batch_sizes.append(rem)
    starts = [0]
    for bs in batch_sizes:
        starts.append(starts[-1] + bs)
    return batch_sizes, starts


# Index discovery / parsing

def _parse_indices_spec(spec: str) -> List[int]:
    """
    Supports:
      "0,1,2"
      "0-5" (inclusive)
      "0:5" (interpreted as inclusive)
      mix: "0-3,7,10:12"
    """
    out = []
    for part in spec.split(","):
        p = part.strip()
        if not p:
            continue
        if "-" in p:
            a, b = p.split("-", 1)
            a, b = int(a), int(b)
            if b < a:
                a, b = b, a
            out.extend(range(a, b + 1))
        elif ":" in p:
            a, b = p.split(":", 1)
            a, b = int(a), int(b)
            if b < a:
                a, b = b, a
            out.extend(range(a, b + 1))
        else:
            out.append(int(p))
    return sorted(set(out))


def discover_available_indices(path_save: Path, prefix: str) -> List[int]:
    """
    Discover indices by scanning directories under path_save matching:
      f"{prefix}{idx}" or f"{prefix}{idx}_seed{seed}"
    """
    if not path_save.exists():
        return []
    pat = re.compile(r"^" + re.escape(prefix) + r"(?P<idx>\d+)(?:_seed\d+)?$")
    idxs = set()
    for p in path_save.iterdir():
        if not p.is_dir():
            continue
        m = pat.match(p.name)
        if m:
            idxs.add(int(m.group("idx")))
    return sorted(idxs)


# Generation cache (per model-folder, per checkpoint)

class GenerationCache:
    """
    Cache generated samples on disk (and small in-memory map) keyed by (folder_name, checkpoint_id).
    Stores CPU tensors [Nsamples,C,H,W] in store_img_dtype.
    """
    def __init__(
        self,
        cache_dir: Path,
        device: str,
        df,
        config,
        build_model_fn,
        x_init_batches: List[torch.Tensor],
        batch_sizes: List[int],
        starts: List[int],
        store_img_dtype: torch.dtype,
        seed_base: int,
    ):
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.device = device
        self.df = df
        self.config = config
        self.build_model_fn = build_model_fn
        self.x_init_batches = x_init_batches
        self.batch_sizes = batch_sizes
        self.starts = starts
        self.store_img_dtype = store_img_dtype
        self.seed_base = int(seed_base)

        self._models: Dict[str, torch.nn.Module] = {}
        self._mem: Dict[Tuple[str, int], torch.Tensor] = {}

    def _get_model(self, folder: str) -> torch.nn.Module:
        if folder not in self._models:
            self._models[folder] = self.build_model_fn()
        return self._models[folder]

    def _cache_path(self, folder: str, checkpoint_id: int) -> Path:
        d = self.cache_dir / folder
        d.mkdir(parents=True, exist_ok=True)
        return d / f"gen_epoch_{int(checkpoint_id)}.pt"

    @torch.no_grad()
    def get(self, folder: str, checkpoint_id: int, ckpt_path: Path) -> torch.Tensor:
        key = (folder, int(checkpoint_id))
        if key in self._mem:
            return self._mem[key]

        fpath = self._cache_path(folder, checkpoint_id)
        if fpath.exists():
            x = torch.load(str(fpath), map_location="cpu")
            if not torch.is_tensor(x):
                raise RuntimeError(f"Cache file did not contain a tensor: {fpath}")
            self._mem[key] = x
            return x

        # Generate and save
        model = self._get_model(folder)
        model = loader.load_model(model, str(ckpt_path)).eval()

        Nsamples = self.starts[-1]
        C, H, W = self.config.IMG_SHAPE
        out = torch.empty((Nsamples, C, H, W), dtype=self.store_img_dtype)  # CPU

        nbatches = len(self.batch_sizes)
        for bi in tqdm(range(nbatches), desc=f"Gen-cache {folder} @ {checkpoint_id}", leave=False):
            x0 = self.x_init_batches[bi]
            seed_noise = self.seed_base + 10_000 * bi

            xa, _ = Diffusion.sample_diffusion_from_noise_det(
                model, config=self.config, df=self.df, dim=4,
                x_init=x0, seed=seed_noise
            )
            s0, s1 = self.starts[bi], self.starts[bi + 1]
            out[s0:s1] = xa.detach().cpu().to(self.store_img_dtype)

        tmp = str(fpath) + ".tmp"
        torch.save(out, tmp)
        os.replace(tmp, str(fpath))

        self._mem[key] = out
        return out


def main():
    parser = argparse.ArgumentParser(
        "Scan unordered model-index pairs; for each pair save similarity curve + histograms + rep pairs (+npz) "
        "and cache generated sequences per model/checkpoint."
    )
    # Original args (kept)
    parser.add_argument("-n", "--num", type=int, required=True)
    parser.add_argument("-s", "--img_size", type=int, required=True)
    parser.add_argument("-LR", "--learning_rate", type=float, required=True)
    parser.add_argument("-O", "--optim", type=str, required=True)
    parser.add_argument("-W", "--nbase", type=str, required=True)
    parser.add_argument("-B", "--batch_size", type=int, required=True)
    parser.add_argument("-Ns", "--Nsamples", type=int, required=True)

    # Pair scanning controls
    parser.add_argument("--scan_all_pairs", action="store_true",
                        help="If set, scan all unordered pairs among discovered/selected indices.")
    parser.add_argument("--indices", type=str, default=None,
                        help="Optional subset of indices to scan (e.g. '0,1,2' or '0-7').")
    parser.add_argument("--max_pairs", type=int, default=None,
                        help="Optional cap on number of pairs (useful for debugging).")

    # Compatibility single-pair mode (optional)
    parser.add_argument("-ia", "--index_a", type=int, default=None)
    parser.add_argument("-ib", "--index_b", type=int, default=None)

    # Seeds for generation / (only relevant for determinism; kept)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--seed", type=int, default=0)

    # Metric / plotting choices
    parser.add_argument("--metric", type=str, default="pixel", choices=["pixel", "resnet"])
    parser.add_argument("--rep_k", type=int, default=20, help="How many pairs closest to mean to plot.")
    parser.add_argument("--hist_bins", type=int, default=50)
    parser.add_argument("--batch_gen", type=int, default=100, help="Generation batch size (cap).")
    parser.add_argument("--store_img_dtype", type=str, default="float16", choices=["float16", "float32"],
                        help="CPU dtype to store generated images (and cached sequences).")
    parser.add_argument("--train_index_bs", type=int, default=256)
    parser.add_argument("--train_num_workers", type=int, default=0)
    parser.add_argument("--train_chunk_size", type=int, default=8192)
    parser.add_argument("--train_feat_dtype", type=str, default="float16", choices=["float16", "float32"])

    # For legacy "same index, different seeds" use-cases (not used in unordered i<j scanning by default)
    parser.add_argument("-sr", "--seeds_run", type=str, default="0,1")

    args = parser.parse_args()

    print(args)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    # Config
    DATASET = "CelebA"
    config = cfg.load_config(DATASET)

    n_base = int(args.nbase)
    config.DEVICE = args.device
    config.n_images = int(args.num)
    config.OPTIM = args.optim
    config.BATCH_SIZE = int(args.batch_size)
    config.LR = float(args.learning_rate)
    size = int(args.img_size)

    seeds_run = [int(s) for s in args.seeds_run.split(",") if s.strip() != ""]

    # Model folder naming
    def build_folder_name(index: int, seed_run_i: Optional[int] = None) -> str:
        base = f"{config.DATASET}{size}_{config.n_images}_{n_base}_{config.OPTIM}_{config.BATCH_SIZE}_{config.LR:.4f}_index{int(index)}"
        if seed_run_i is None:
            return base
        return base + f"_seed{int(seed_run_i)}"

    def build_model():
        m = Unet.UNet(
            input_channels=config.IMG_SHAPE[0],
            output_channels=config.IMG_SHAPE[0],
            base_channels=n_base,
            base_channels_multiples=(1, 2, 4),
            apply_attention=(False, True, True),
            dropout_rate=0.1,
        )
        return m.to(config.DEVICE).eval()

    # Metric backbone
    resnet_fn = None
    if args.metric == "resnet":
        resnet_fn = make_resnet50_feature_extractor(config.DEVICE)

    # Output dirs
    out_dir = Path(config.path_save) / f"Comparisons_Mallat_final_{int(args.num)}_seed_{int(args.seed)}"
    out_dir.mkdir(parents=True, exist_ok=True)

    tag = f"metric_{args.metric}"

    # Cache dirs
    cache_dir = out_dir / f"GenCache_seed{int(args.seed)}_Ns{int(args.Nsamples)}_dtype{args.store_img_dtype}"

    # Determine indices + pairs
    # If user passes --indices, use that. Otherwise, auto-discover from filesystem.
    if args.indices is not None:
        indices = _parse_indices_spec(args.indices)
    else:
        prefix = f"{config.DATASET}{size}_{config.n_images}_{n_base}_{config.OPTIM}_{config.BATCH_SIZE}_{config.LR:.4f}_index"
        indices = discover_available_indices(Path(config.path_save), prefix)

    indices = sorted(indices)

    # If not scanning all pairs, fall back to single-pair mode (must have ia/ib).
    if not args.scan_all_pairs:
        if args.index_a is None or args.index_b is None:
            raise ValueError("Provide -ia and -ib, or use --scan_all_pairs (optionally with --indices).")
        pairs = [(int(args.index_a), int(args.index_b))]
    else:
        if len(indices) < 2:
            raise RuntimeError(
                f"Not enough indices to form pairs. Discovered indices={indices}. "
                f"Try providing --indices explicitly."
            )
        pairs = list(itertools.combinations(indices, 2))  # unordered i<j
        if args.max_pairs is not None:
            pairs = pairs[: int(args.max_pairs)]

    print(f"Indices: {indices}")
    print(f"Num pairs: {len(pairs)}")
    if len(pairs) <= 20:
        print("Pairs:", pairs)

    # Diffusion config + generation plan (shared across all pairs)
    df = Diffusion.DiffusionConfig(
        n_steps=config.TIMESTEPS,
        img_shape=config.IMG_SHAPE,
        device=config.DEVICE,
    )

    Nsamples = int(args.Nsamples)
    batch_gen = min(int(args.batch_gen), Nsamples)
    batch_sizes, starts = make_batch_plan(Nsamples, batch_gen)
    nbatches = len(batch_sizes)

    # fixed x_init batches (shared across all models/pairs)
    g_init = torch.Generator(device=config.DEVICE)
    g_init.manual_seed(int(args.seed))
    x_init_batches = []
    for bs in batch_sizes:
        x0 = torch.randn(bs, *config.IMG_SHAPE, device=config.DEVICE, generator=g_init)
        x_init_batches.append(x0)

    # Checkpoints
    training_times = np.linspace(10, 5000 - 1, 50, dtype=int)[1:21]

    # dtype for stored/generated images (also used for cache)
    store_img_dtype = torch.float16 if args.store_img_dtype == "float16" else torch.float32
    store_feat_dtype = torch.float16 if args.train_feat_dtype == "float16" else torch.float32

    # Caches: training datasets + training feature indices
    train_ds_cache: Dict[int, object] = {}
    train_index_cache: Dict[int, Tuple[object, torch.Tensor]] = {}

    def get_train_ds(index: int):
        if index in train_ds_cache:
            return train_ds_cache[index]
        train_images, _ = cfg.load_training_data(config, int(index), loadtest=True)
        train_ds_cache[index] = train_images
        return train_images

    def get_train_index(index: int):
        if index in train_index_cache:
            return train_index_cache[index]
        train_images = get_train_ds(index)
        print(f"Indexing training features for index={index} (metric={args.metric}) ...")
        train_ref, train_feats_cpu = build_train_index(
            train_images, args.metric, config.DEVICE, resnet_fn,
            batch_size=int(args.train_index_bs),
            store_dtype=store_feat_dtype,
            num_workers=int(args.train_num_workers),
        )
        train_index_cache[index] = (train_ref, train_feats_cpu)
        print(f"Done indexing index={index}.")
        return train_ref, train_feats_cpu

    # Generation cache (shared across pairs)
    gen_cache = GenerationCache(
        cache_dir=cache_dir,
        device=config.DEVICE,
        df=df,
        config=config,
        build_model_fn=build_model,
        x_init_batches=x_init_batches,
        batch_sizes=batch_sizes,
        starts=starts,
        store_img_dtype=store_img_dtype,
        seed_base=int(args.seed),
    )

    # Pair loop
    for (index_a, index_b) in pairs:
        index_a = int(index_a)
        index_b = int(index_b)
        print("\n" + "=" * 80)
        print(f"PAIR: index {index_a} vs {index_b}")

        # Load datasets (cached)
        train_images_a = get_train_ds(index_a)
        train_images_b = get_train_ds(index_b)

        # Optional dup check + topk correlated train to train
        rep = dup_check_trainsets(train_images_a, train_images_b, device=config.DEVICE, thresh=0.995)
        print("[dup-check] max_sim:", rep["max_sim"])
        if rep["matches"]:
            print("[dup-check] possible duplicates (top 10):", rep["matches"][:10])

        ia10, ib10, sim10 = topk_corr_pairs(train_images_a, train_images_b, device=config.DEVICE, k=10)

        topk_png = Path(config.path_save) / f"top10_train{index_a}_vs_train{index_b}_corr.png"
        topk_npz = Path(config.path_save) / f"top10_train{index_a}_vs_train{index_b}_corr.npz"
        save_topk_pairs_png(train_images_a, train_images_b, ia10, ib10, sim10, topk_png)
        save_topk_pairs_npz(train_images_a, train_images_b, ia10, ib10, sim10, topk_npz)

        print("Saved:", topk_png)
        print("Saved:", topk_npz)
        print("Top pairs:", list(zip(ia10.tolist(), ib10.tolist(), sim10.tolist())))

        # Output dirs (pair-specific)
        curve_png = out_dir / f"cosine_similarity_{tag}_index{index_a}_vs_{index_b}.png"
        curve_npz = out_dir / f"cosine_similarity_{tag}_index{index_a}_vs_{index_b}.npz"
        hist_dir  = out_dir / f"SimilarityHist_{tag}_index{index_a}_vs_{index_b}"
        pairs_dir = out_dir / f"RepPairs_{tag}_index{index_a}_vs_{index_b}"
        hist_dir.mkdir(parents=True, exist_ok=True)
        pairs_dir.mkdir(parents=True, exist_ok=True)

        # Training NN indices (cached per index)
        train_ref_a, train_feats_a_cpu = get_train_index(index_a)
        train_ref_b, train_feats_b_cpu = get_train_index(index_b)

        # Curve state
        fig_curve, ax_curve = plt.subplots(figsize=(8, 4.5))
        fig_curve.tight_layout()
        means, sems = [], []

        for j, checkpoint_id in enumerate(training_times):
            print(f"Training time = {checkpoint_id} ({j + 1}/{len(training_times)})")

            # Checkpoint paths for each model
            folder_a = build_folder_name(index_a, None)
            folder_b = build_folder_name(index_b, None)

            path_a = Path(config.path_save) / folder_a / "Models" / f"Model_epoch_{checkpoint_id}"
            path_b = Path(config.path_save) / folder_b / "Models" / f"Model_epoch_{checkpoint_id}"
            if not path_a.exists():
                raise FileNotFoundError(f"Missing checkpoint: {path_a}")
            if not path_b.exists():
                raise FileNotFoundError(f"Missing checkpoint: {path_b}")

            # Get cached generations (CPU tensors [Nsamples,C,H,W])
            xa_all = gen_cache.get(folder_a, int(checkpoint_id), path_a)
            xb_all = gen_cache.get(folder_b, int(checkpoint_id), path_b)

            # Compute sims in batches (GPU) to keep memory sane
            sims_all = torch.empty((Nsamples,), dtype=torch.float32)  # CPU
            for bi in range(nbatches):
                s0, s1 = starts[bi], starts[bi + 1]
                xa = xa_all[s0:s1].to(config.DEVICE, non_blocking=True).float()
                xb = xb_all[s0:s1].to(config.DEVICE, non_blocking=True).float()
                sim = cosine_sim(xa, xb, metric=args.metric, resnet_fn=resnet_fn).detach().cpu()
                sims_all[s0:s1] = sim

            # Distribution stats + curve
            mean_sim = float(sims_all.mean().item())
            std_sim = float(sims_all.std(unbiased=False).item())
            sem_sim = std_sim / math.sqrt(max(Nsamples, 1))

            means.append(mean_sim)
            sems.append(sem_sim)

            x = np.array(training_times[:len(means)])
            y = np.array(means, dtype=float)
            yerr = np.array(sems, dtype=float)

            ax_curve.cla()
            ax_curve.grid(True, alpha=0.3)
            ax_curve.set_xlabel("Checkpoint (epoch)", fontsize=9)
            ax_curve.set_ylabel(f"Similarity ({args.metric})", fontsize=9)
            ax_curve.set_title(f"Mean similarity vs checkpoint (index {index_a} vs {index_b})", fontsize=9)
            ax_curve.errorbar(x, y, yerr=yerr, fmt="o-", capsize=2, markersize=3.5, linewidth=1.4)
            ax_curve.tick_params(labelsize=8)
            fig_curve.tight_layout()
            fig_curve.savefig(curve_png, dpi=200)

            np.savez(
                curve_npz,
                training_times=x,
                means=y,
                sems=yerr,
                Nsamples=Nsamples,
                seed=int(args.seed),
                index_a=index_a,
                index_b=index_b,
                metric=args.metric,
            )

            # Histogram
            fig_h, ax_h = plt.subplots(figsize=(7.5, 4.5))
            ax_h.hist(sims_all.numpy(), bins=int(args.hist_bins))
            ax_h.axvline(mean_sim, linestyle="--")
            ax_h.set_title(f"Similarity distribution @ epoch {checkpoint_id}\nmean={mean_sim:.4f}, std={std_sim:.4f}", fontsize=9)
            ax_h.set_xlabel("Similarity", fontsize=9)
            ax_h.set_ylabel("Count", fontsize=9)
            ax_h.tick_params(labelsize=8)
            fig_h.tight_layout()
            fig_h.savefig(hist_dir / f"hist_epoch_{checkpoint_id}.png", dpi=200)
            plt.close(fig_h)

            # Representative pairs closest to mean
            rep_k = min(int(args.rep_k), Nsamples)
            dist = (sims_all - mean_sim).abs()
            vals, rep_idx = torch.topk(dist, k=rep_k, largest=False)
            rep_idx = rep_idx[torch.argsort(vals)]  # sorted by closeness

            rep_xa = xa_all[rep_idx].float()
            rep_xb = xb_all[rep_idx].float()
            rep_sims = sims_all[rep_idx]

            # NN (only for rep_k)
            nn_sims_a, nn_idx_a = nearest_train_scores_and_idx(
                rep_xa.to(config.DEVICE), train_feats_a_cpu,
                metric=args.metric, device=config.DEVICE, resnet_fn=resnet_fn,
                chunk_size=int(args.train_chunk_size)
            )
            nn_sims_b, nn_idx_b = nearest_train_scores_and_idx(
                rep_xb.to(config.DEVICE), train_feats_b_cpu,
                metric=args.metric, device=config.DEVICE, resnet_fn=resnet_fn,
                chunk_size=int(args.train_chunk_size)
            )
            nn_imgs_a = fetch_train_by_idx(train_ref_a, nn_idx_a)
            nn_imgs_b = fetch_train_by_idx(train_ref_b, nn_idx_b)

            out_pairs_png = pairs_dir / f"rep{rep_k}_near_mean_epoch_{checkpoint_id}.png"
            out_pairs_npz = pairs_dir / f"rep{rep_k}_near_mean_epoch_{checkpoint_id}.npz"

            save_pairs_with_nn_png(
                nn_imgs_a, rep_xa.cpu(), rep_xb.cpu(), nn_imgs_b,
                rep_sims, out_pairs_png,
                title_nn_a="NN train A", title_a="Model A", title_b="Model B", title_nn_b="NN train B",
                nn_sims_a=nn_sims_a, nn_sims_b=nn_sims_b
            )

            save_rep_pairs_with_nn_npz(
                rep_idx_cpu=rep_idx,
                rep_sims_cpu=rep_sims,
                rep_xa_cpu=rep_xa.cpu(),
                rep_xb_cpu=rep_xb.cpu(),
                nn_idx_a_cpu=nn_idx_a,
                nn_sims_a_cpu=nn_sims_a,
                nn_imgs_a_cpu=nn_imgs_a,
                nn_idx_b_cpu=nn_idx_b,
                nn_sims_b_cpu=nn_sims_b,
                nn_imgs_b_cpu=nn_imgs_b,
                out_path_npz=out_pairs_npz,
            )

            # small cleanup (keep cache tensors)
            del sims_all, rep_xa, rep_xb, rep_sims
            torch.cuda.empty_cache()

        plt.close(fig_curve)
        print("Done pair!")
        print("Saved curve:", curve_png)
        print("Saved curve data:", curve_npz)
        print("Saved hists in:", hist_dir)
        print("Saved representative pairs in:", pairs_dir)

    print("\nALL DONE!")


if __name__ == "__main__":
    main()
