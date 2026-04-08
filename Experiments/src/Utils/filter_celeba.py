#!/usr/bin/env python3
"""Filter CelebA32 images via PCA or wavelet coarsening at multiple levels.

For each method × level, saves pre-split train (index 0, 1) and test tensors
ready for direct consumption by run_Unet.py --data-file.

Usage:
    python filter_celeba.py \
        --input-path ../../Data/CelebA/CelebA32.pt \
        --output-dir ../../Data/CelebA_filtered/ \
        --n-train 1024
"""

import argparse
import os
from typing import Optional

import numpy as np
import torch

# Default PCA components per level: number of principal components retained.
DEFAULT_PCA_COMPONENTS = [32, 128, 512, 1024]

# Wavelet levels: number of finest DWT scales whose details are zeroed out
# 0 → zero all 3 detail scales, 3 → zero none (original)
WAVELET_ZERO_SCALES = {0: 3, 1: 2, 2: 1, 3: 0}

# Balanced wavelet levels: multiplicative gains for detail bands
# (coarse, medium, fine) at each level.
# This yields a smoother detail injection than strict on/off scale gating.
WAVELET_BALANCED_GAINS = {
    0: (0.00, 0.00, 0.00),
    1: (0.45, 0.15, 0.05),
    2: (0.80, 0.45, 0.20),
    3: (1.00, 1.00, 1.00),
}


def parse_pca_components(raw: str) -> list[int]:
    vals = [s.strip() for s in raw.split(",") if s.strip()]
    if not vals:
        raise ValueError("--pca-components must contain at least one integer")
    out = [int(v) for v in vals]
    if any(v <= 0 for v in out):
        raise ValueError("All PCA component counts must be positive")
    return out


def pca_filter(images_flat: np.ndarray, n_components: int, basis_flat: Optional[np.ndarray] = None) -> np.ndarray:
    """Project images onto top-k PCA components and reconstruct.

    Args:
        images_flat: [N, D] array of flattened images to reconstruct.
        n_components: number of PCA components to retain.
        basis_flat: optional [M, D] array used to estimate PCA basis.
            If None, `images_flat` itself is used.

    Returns:
        Reconstructed [N, D] array.
    """
    if basis_flat is None:
        basis_flat = images_flat

    mean = basis_flat.mean(axis=0)
    centered_basis = basis_flat - mean
    # economy SVD
    _, _, Vt = np.linalg.svd(centered_basis, full_matrices=False)
    # keep top-k
    k = min(n_components, Vt.shape[0])
    centered = images_flat - mean
    proj = centered @ Vt[:k].T  # [N, k]
    recon = proj @ Vt[:k] + mean
    return recon


def wavelet_filter(
    images: np.ndarray,
    zero_scales: Optional[int] = None,
    detail_gains: Optional[tuple[float, float, float]] = None,
) -> np.ndarray:
    """Zero out high-frequency DWT subbands and reconstruct.

    Uses 3-level Haar DWT on each 32x32 image.

    Two mutually exclusive modes:
    - zero_scales: how many finest scales to zero (legacy behavior).
    - detail_gains: multiplicative gains for (coarse, medium, fine) detail bands.

    Args:
        images: [N, H, W] array.
        zero_scales: number of finest detail scales to discard.
        detail_gains: tuple of gains (g3, g2, g1) applied to detail scales.

    Returns:
        Reconstructed [N, H, W] array.
    """
    import pywt

    if (zero_scales is None) == (detail_gains is None):
        raise ValueError("Provide exactly one of zero_scales or detail_gains")

    if zero_scales is not None and zero_scales == 0:
        return images.copy()

    if detail_gains is not None:
        if len(detail_gains) != 3:
            raise ValueError("detail_gains must have length 3: (coarse, medium, fine)")
        if any(g < 0.0 for g in detail_gains):
            raise ValueError("detail_gains values must be non-negative")

    wavelet = "haar"
    max_level = 3
    result = np.empty_like(images)

    for i in range(images.shape[0]):
        coeffs = pywt.wavedec2(images[i], wavelet, level=max_level)
        # coeffs: [cA3, (cH3,cV3,cD3), (cH2,cV2,cD2), (cH1,cV1,cD1)]
        # index 1 = coarsest detail, index 3 = finest detail
        if zero_scales is not None:
            for s in range(1, zero_scales + 1):
                idx = max_level - s + 1
                coeffs[idx] = tuple(np.zeros_like(c) for c in coeffs[idx])
        else:
            g3, g2, g1 = detail_gains
            gains = {1: g3, 2: g2, 3: g1}
            for idx in (1, 2, 3):
                g = gains[idx]
                coeffs[idx] = tuple(g * c for c in coeffs[idx])

        result[i] = pywt.waverec2(coeffs, wavelet)

    return result


def parse_args():
    parser = argparse.ArgumentParser(
        description="Filter CelebA32 images via PCA or wavelet coarsening."
    )
    parser.add_argument(
        "--input-path",
        type=str,
        default="../../Data/CelebA/CelebA32.pt",
        help="Path to the full CelebA32.pt tensor.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="../../Data/CelebA_filtered/",
        help="Directory to save filtered tensors.",
    )
    parser.add_argument(
        "--n-train",
        type=int,
        default=1024,
        help="Number of training images per split.",
    )
    parser.add_argument(
        "--n-test",
        type=int,
        default=2048,
        help="Number of test images (taken from the end).",
    )
    parser.add_argument(
        "--pca-components",
        type=str,
        default=",".join(str(x) for x in DEFAULT_PCA_COMPONENTS),
        help="Comma-separated PCA components per level, e.g. '16,64,128,1024'.",
    )
    parser.add_argument(
        "--pca-tag",
        type=str,
        default="PCA",
        help="Tag used in PCA output filenames: CelebA32_<tag>_L<level>_*.pt",
    )
    parser.add_argument(
        "--skip-wavelet",
        action="store_true",
        help="If set, do not generate wavelet-filtered tensors.",
    )
    parser.add_argument(
        "--skip-pca",
        action="store_true",
        help="If set, do not generate PCA-filtered tensors.",
    )
    parser.add_argument(
        "--wavelet-mode",
        type=str,
        default="legacy",
        choices=["legacy", "balanced"],
        help="Wavelet level construction: legacy on/off scales or balanced detail gains.",
    )
    parser.add_argument(
        "--wavelet-tag",
        type=str,
        default=None,
        help="Tag used in wavelet output filenames: CelebA32_<tag>_L<level>_*.pt",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    input_path = os.path.expanduser(args.input_path)
    output_dir = os.path.expanduser(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)

    print(f"Loading {input_path} ...")
    full_tensor = torch.load(input_path, weights_only=True)  # [N, 1, 32, 32]
    print(f"  shape: {full_tensor.shape}, dtype: {full_tensor.dtype}")
    N = full_tensor.shape[0]

    n_train = args.n_train
    n_test = args.n_test
    pca_components = parse_pca_components(args.pca_components)
    pca_levels = {level: comp for level, comp in enumerate(pca_components)}

    if 2 * n_train + n_test > N:
        raise ValueError(
            f"Invalid split sizes for N={N}: 2*n_train + n_test = {2 * n_train + n_test} exceeds dataset size."
        )

    if args.skip_pca:
        print("PCA generation: skipped")
    else:
        print(f"PCA levels ({args.pca_tag}): {pca_levels}")
    if args.skip_wavelet:
        print("Wavelet generation: skipped")
    else:
        print(f"Wavelet mode: {args.wavelet_mode}")

    wavelet_tag = args.wavelet_tag
    if wavelet_tag is None:
        wavelet_tag = "Wavelet" if args.wavelet_mode == "legacy" else "WaveletBalanced"

    # Extract the two training splits and the test split (raw indices)
    idx0_slice = slice(0, n_train)
    idx1_slice = slice(n_train, 2 * n_train)
    test_slice = slice(N - n_test, N)

    # Flatten images once; PCA basis is fitted strictly on train splits only.
    full_np = full_tensor.squeeze(1).numpy()  # [N, 32, 32]
    full_flat = full_np.reshape(N, -1)  # [N, 1024]
    train_basis_flat = full_flat[: 2 * n_train]

    # ── PCA filtering ────────────────────────────────────────────────────────
    if not args.skip_pca:
        for level, n_comp in pca_levels.items():
            print(f"\n=== PCA Level {level} (k={n_comp}) ===")
            recon_flat = pca_filter(full_flat, n_comp, basis_flat=train_basis_flat)
            recon = torch.from_numpy(
                recon_flat.reshape(N, 1, 32, 32).astype(np.float32)
            )

            for tag, sl in [("index0", idx0_slice), ("index1", idx1_slice), ("test", test_slice)]:
                out = recon[sl]
                fname = f"CelebA32_{args.pca_tag}_L{level}_{tag}.pt"
                fpath = os.path.join(output_dir, fname)
                torch.save(out, fpath)
                print(f"  saved {fname}  shape={out.shape}")

    if not args.skip_wavelet:
        # ── Wavelet filtering ────────────────────────────────────────────────
        if args.wavelet_mode == "legacy":
            wavelet_levels = sorted(WAVELET_ZERO_SCALES.keys())
        else:
            wavelet_levels = sorted(WAVELET_BALANCED_GAINS.keys())

        for level in wavelet_levels:
            if args.wavelet_mode == "legacy":
                zero_sc = WAVELET_ZERO_SCALES[level]
                print(f"\n=== Wavelet Level {level} (zero_scales={zero_sc}) ===")
                recon_np = wavelet_filter(full_np, zero_scales=zero_sc)
            else:
                gains = WAVELET_BALANCED_GAINS[level]
                print(f"\n=== WaveletBalanced Level {level} (gains={gains}) ===")
                recon_np = wavelet_filter(full_np, detail_gains=gains)

            recon = torch.from_numpy(
                recon_np.reshape(N, 1, 32, 32).astype(np.float32)
            )

            for tag, sl in [("index0", idx0_slice), ("index1", idx1_slice), ("test", test_slice)]:
                out = recon[sl]
                fname = f"CelebA32_{wavelet_tag}_L{level}_{tag}.pt"
                fpath = os.path.join(output_dir, fname)
                torch.save(out, fpath)
                print(f"  saved {fname}  shape={out.shape}")

    print("\nDone. All filtered datasets saved to:", output_dir)


if __name__ == "__main__":
    main()
