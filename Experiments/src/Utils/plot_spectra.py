#!/usr/bin/env python3
"""Plot PCA and wavelet spectra on CelebA tensors."""

import argparse
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch


def parse_args():
    parser = argparse.ArgumentParser("Plot PCA eigenvalue and wavelet energy spectra.")
    parser.add_argument("--data_path", type=str, default="../../Data/CelebA/CelebA32.pt")
    parser.add_argument("--n_samples", type=int, default=1024)
    parser.add_argument("--out_file", type=str, default="../../Saves_new/celeba_spectra.png")
    parser.add_argument(
        "--pca_levels",
        type=str,
        default="16,64,128,1024",
        help="Comma-separated PCA cut positions to annotate.",
    )
    parser.add_argument(
        "--wavelet-balanced-gains",
        type=str,
        default="0,0,0;0.45,0.15,0.05;0.8,0.45,0.2;1,1,1",
        help="Semicolon-separated gains for wavelet detail bands per level: g3,g2,g1;...",
    )
    return parser.parse_args()


def parse_pca_levels(raw: str) -> list[int]:
    vals = [s.strip() for s in raw.split(",") if s.strip()]
    if not vals:
        raise ValueError("--pca_levels must contain at least one integer")
    out = sorted({int(v) for v in vals})
    if any(v <= 0 for v in out):
        raise ValueError("All PCA levels must be positive")
    return out


def parse_balanced_gains(raw: str) -> list[tuple[float, float, float]]:
    blocks = [b.strip() for b in raw.split(";") if b.strip()]
    if len(blocks) != 4:
        raise ValueError("--wavelet-balanced-gains must define exactly 4 levels")

    out = []
    for b in blocks:
        parts = [p.strip() for p in b.split(",") if p.strip()]
        if len(parts) != 3:
            raise ValueError("Each gain block must contain 3 values: g3,g2,g1")
        g = tuple(float(x) for x in parts)
        if any(x < 0 for x in g):
            raise ValueError("Wavelet gains must be non-negative")
        out.append(g)
    return out


def main():
    args = parse_args()
    pca_levels = parse_pca_levels(args.pca_levels)
    balanced_gains = parse_balanced_gains(args.wavelet_balanced_gains)

    try:
        import pywt
    except ImportError as exc:
        raise SystemExit("pywt is required. Install with: pip install PyWavelets") from exc

    data_path = os.path.expanduser(args.data_path)
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Cannot find data at {data_path}")

    data = torch.load(data_path, weights_only=True)[: args.n_samples].numpy()
    n, c, h, w = data.shape
    data_flat = data.reshape(n, -1)

    print("Computing PCA via SVD...")
    x = data_flat - data_flat.mean(axis=0, keepdims=True)
    singular_values = np.linalg.svd(x, full_matrices=False, compute_uv=False)
    explained_variance = (singular_values ** 2) / max(1, (n - 1))
    explained_variance = explained_variance / explained_variance.sum()

    print("Computing wavelet energies...")
    wavelet = "haar"
    energies = {
        "Base (cA3)\n(L0 keeps this)": 0.0,
        "Level 3 (Coarse)\n(L1 adds this)": 0.0,
        "Level 2 (Medium)\n(L2 adds this)": 0.0,
        "Level 1 (Fine)\n(L3 adds this)": 0.0,
    }

    for img in data:
        coeffs = pywt.wavedec2(img[0], wavelet, level=3)
        energies["Base (cA3)\n(L0 keeps this)"] += np.sum(coeffs[0] ** 2)
        energies["Level 3 (Coarse)\n(L1 adds this)"] += sum(np.sum(c_ij ** 2) for c_ij in coeffs[1])
        energies["Level 2 (Medium)\n(L2 adds this)"] += sum(np.sum(c_ij ** 2) for c_ij in coeffs[2])
        energies["Level 1 (Fine)\n(L3 adds this)"] += sum(np.sum(c_ij ** 2) for c_ij in coeffs[3])

    for key in energies:
        energies[key] /= float(n)

    e_base = float(energies["Base (cA3)\n(L0 keeps this)"])
    e_coarse = float(energies["Level 3 (Coarse)\n(L1 adds this)"])
    e_medium = float(energies["Level 2 (Medium)\n(L2 adds this)"])
    e_fine = float(energies["Level 1 (Fine)\n(L3 adds this)"])

    # Legacy incremental additions (strict scale unfreezing).
    legacy_inc = [0.0, e_coarse, e_medium, e_fine]

    # Balanced incremental additions from gain schedule.
    bal_levels_energy = []
    for (g3, g2, g1) in balanced_gains:
        bal_levels_energy.append((g3 ** 2) * e_coarse + (g2 ** 2) * e_medium + (g1 ** 2) * e_fine)

    balanced_inc = [0.0]
    for i in range(1, len(bal_levels_energy)):
        balanced_inc.append(max(0.0, bal_levels_energy[i] - bal_levels_energy[i - 1]))

    print("Generating plots...")
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 5))

    ax1.plot(np.cumsum(explained_variance), lw=2, color="black")
    color_cycle = ["#2ca02c", "#1f77b4", "#ff7f0e", "#d62728", "#9467bd", "#8c564b"]
    for i, level in enumerate(pca_levels):
        if level <= explained_variance.size:
            ax1.axvline(level, color=color_cycle[i % len(color_cycle)], linestyle="--", label=f"L{i} ({level})")
    ax1.set_title("PCA Cumulative Explained Variance")
    ax1.set_xlabel("Number of Principal Components")
    ax1.set_ylabel("Cumulative Variance")
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    labels = list(energies.keys())
    vals = list(energies.values())
    ax2.bar(labels, vals, color=["#2ca02c", "#1f77b4", "#ff7f0e", "#d62728"])
    ax2.set_title("Wavelet Subband Average Energy (Haar, 3-level)")
    ax2.set_ylabel("Average L2 Energy")
    ax2.grid(True, axis="y", alpha=0.3)

    lv = np.arange(4)
    width = 0.38
    ax3.bar(lv - width / 2, legacy_inc, width=width, label="Legacy adds", color="#7f7f7f")
    ax3.bar(lv + width / 2, balanced_inc, width=width, label="Balanced adds", color="#17becf")
    ax3.set_xticks(lv)
    ax3.set_xticklabels(["L0", "L1", "L2", "L3"])
    ax3.set_title("Detail Energy Added Per Level")
    ax3.set_ylabel("Added detail energy")
    ax3.grid(True, axis="y", alpha=0.3)
    ax3.legend()

    text = (
        f"Base energy={e_base:.2f}\\n"
        f"Detail total={e_coarse + e_medium + e_fine:.2f}"
    )
    ax3.text(0.98, 0.98, text, transform=ax3.transAxes, ha="right", va="top", fontsize=9)

    plt.tight_layout()
    out_file = os.path.expanduser(args.out_file)
    os.makedirs(os.path.dirname(out_file), exist_ok=True)
    plt.savefig(out_file, dpi=200)
    plt.close(fig)
    print(f"Done! Plot saved to {out_file}")


if __name__ == "__main__":
    main()
