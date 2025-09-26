#!/usr/bin/env python3
"""
07_threshold_sweep.py

Purpose
- Visually compare how the pre-CLAHE clipping step (sitk.Threshold lower) impacts final binarization.
- Sweep lower thresholds in [25, 40] with step 3 (25, 28, 31, 34, 37, 40), keeping all other params fixed, on a single slice.
- Produce mosaics: (a) binary images per threshold, (b) overlays of contours on grayscale per threshold.
- Export simple metrics per threshold (binary porosity and count of connected components) to a CSV/JSON.

How to use
1) Edit the GLOBALS below (IMAGES_DIR, SLICE_INDEX, PATTERN, OUT_DIR, etc.).
2) Run the script; no CLI flags.

Outputs (in OUT_DIR)
- mosaic_binary.png    : grid of binary images by threshold
- mosaic_overlay.png   : grid of red contours over grayscale by threshold
- sweep_metrics.json   : metrics per threshold
- sweep_metrics.csv    : same, in CSV format

Notes
- Intensity normalization follows 06: map to float [0,1] without per-slice min/max stretching (uint8→/255, uint16→/65535, float→clip to [0,1], /255 if values>1.5).
- The binarization threshold after CLAHE remains the same strategy as in 05 (replace with your fixed value if desired).
"""

from __future__ import annotations
import os
import re
import json
from typing import List, Tuple, Dict

import numpy as np
import SimpleITK as sitk
import matplotlib.pyplot as plt
from skimage import io as skio
from skimage.measure import find_contours, label

# ======================
# GLOBAL CONFIGURATIONS
# ======================

# Input images directory
IMAGES_DIR: str = "/mnt/c/Users/bherr/Downloads/CT-1um/images_reconstructed_1um"
# Optional filename substring filter
PATTERN: str | None = None
# Optional explicit slice index; default mid-slice if None
SLICE_INDEX: int | None = None
# Output directory
OUT_DIR: str = "results/CT/threshold_sweep"

# Sweep of sitk.Threshold lower (pre-CLAHE clip) in 0..255 domain
LOWER_THRESHOLDS_255 = list(range(25, 41, 3))  # 25,28,31,34,37,40

# Preprocessing params (fixed)
DENOISE_TIME_STEP: float = 0.0625
DENOISE_CONDUCTANCE: float = 10.0
DENOISE_ITERATIONS: int = 30
CLAHE_ALPHA: float = 0.5
CLAHE_BETA: float = 0.7

# Binarization threshold on CLAHE (0..1). If None, use mean(clahe) like original 05.
FIXED_BIN_THRESH_01: float | None = None

# Overlay options
CONTOUR_LINEWIDTH: float = 0.6


# ======================
# UTILITIES
# ======================

def natural_key(s: str) -> List[object]:
    return [int(text) if text.isdigit() else text.lower() for text in re.split(r"(\d+)", s)]


def collect_images(images_dir: str, pattern: str | None) -> List[str]:
    if not os.path.isdir(images_dir):
        raise FileNotFoundError(f"images-dir not found: {images_dir}")
    exts = {".bmp", ".png", ".jpg", ".jpeg", ".tif", ".tiff"}
    files = [os.path.join(images_dir, f) for f in os.listdir(images_dir) if os.path.splitext(f)[1].lower() in exts]
    if pattern:
        files = [f for f in files if pattern in os.path.basename(f)]
    files.sort(key=natural_key)
    if not files:
        raise RuntimeError(f"No images found in {images_dir} with pattern={pattern!r}")
    return files


def to_float01(img: np.ndarray) -> np.ndarray:
    if img.dtype == np.uint8:
        return img.astype(np.float32) / 255.0
    if img.dtype == np.uint16:
        return img.astype(np.float32) / 65535.0
    if np.issubdtype(img.dtype, np.floating):
        imgf = img.astype(np.float32)
        mx = float(np.nanmax(imgf))
        if mx > 1.5:
            imgf = imgf / 255.0
        return np.clip(imgf, 0.0, 1.0)
    imgf = img.astype(np.float32)
    imin, imax = float(np.min(imgf)), float(np.max(imgf))
    if imax <= imin:
        return np.zeros_like(imgf, dtype=np.float32)
    return np.clip((imgf - imin) / (imax - imin), 0.0, 1.0)


def run_pipeline_with_clip(gray01: np.ndarray, clip_lower_01: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    # Wrap as SITK image
    img_sitk = sitk.GetImageFromArray(gray01)

    # Denoise
    denoised = sitk.CurvatureAnisotropicDiffusion(
        img_sitk,
        timeStep=DENOISE_TIME_STEP,
        conductanceParameter=DENOISE_CONDUCTANCE,
        numberOfIterations=DENOISE_ITERATIONS,
    )

    # Clip low intensities (varying parameter)
    clipped = sitk.Threshold(denoised, lower=clip_lower_01, upper=1.0, outsideValue=0.0)

    # CLAHE
    clahe = sitk.AdaptiveHistogramEqualization(clipped, alpha=CLAHE_ALPHA, beta=CLAHE_BETA)
    clahe_np = sitk.GetArrayFromImage(clahe).astype(np.float32, copy=False)

    # Threshold on CLAHE
    if FIXED_BIN_THRESH_01 is None:
        thr = float(np.mean(clahe_np))
    else:
        thr = float(FIXED_BIN_THRESH_01)

    binary = sitk.BinaryThreshold(clahe, lowerThreshold=thr, upperThreshold=1.0, outsideValue=0, insideValue=1)
    binary_np = sitk.GetArrayFromImage(binary).astype(bool, copy=False)
    binary_np = np.squeeze(binary_np)

    return clahe_np, binary_np, thr, float(np.mean(binary_np))  # also return porosity for metrics


def plot_mosaic_binary(thresholds_255: List[int], binaries: List[np.ndarray], out_path: str) -> None:
    n = len(thresholds_255)
    cols = 3
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 4))
    axes = np.atleast_2d(axes)

    for i, (t, b) in enumerate(zip(thresholds_255, binaries)):
        r, c = divmod(i, cols)
        ax = axes[r, c]
        ax.imshow(np.squeeze(b)[1300:1700,1300:1700], cmap="gray")
        ax.set_title(f"lower={t}/255")
        ax.axis("off")
    # hide any empty cells
    for j in range(n, rows * cols):
        r, c = divmod(j, cols)
        axes[r, c].axis("off")

    plt.tight_layout()
    fig.savefig(out_path, dpi=250)
    plt.close(fig)


def plot_mosaic_overlay(thresholds_255: List[int], gray01: np.ndarray, binaries: List[np.ndarray], out_path: str) -> None:
    n = len(thresholds_255)
    cols = 3
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 4))
    axes = np.atleast_2d(axes)
    gray = np.squeeze(gray01)

    for i, (t, b) in enumerate(zip(thresholds_255, binaries)):
        r, c = divmod(i, cols)
        ax = axes[r, c]
        ax.imshow(gray[1300:1700,1300:1700], cmap="gray")
        contours = find_contours(np.squeeze(b)[1300:1700,1300:1700].astype(float), level=0.5)
        for cnt in contours:
            ax.plot(cnt[:, 1], cnt[:, 0], color="red", linewidth=CONTOUR_LINEWIDTH)
        ax.set_title(f"lower={t}/255")
        ax.axis("off")
    for j in range(n, rows * cols):
        r, c = divmod(j, cols)
        axes[r, c].axis("off")

    plt.tight_layout()
    fig.savefig(out_path, dpi=250)
    plt.close(fig)


def main() -> None:
    files = collect_images(IMAGES_DIR, PATTERN)
    idx = SLICE_INDEX if SLICE_INDEX is not None else len(files) // 2
    if idx < 0 or idx >= len(files):
        raise IndexError(f"slice-index {idx} out of range for {len(files)} files")

    path = files[idx]
    print(f"[sweep] Using slice index {idx} -> {os.path.basename(path)}")

    img = skio.imread(path)
    if img.ndim == 3:
        img = (0.2126 * img[..., 0] + 0.7152 * img[..., 1] + 0.0722 * img[..., 2]).astype(img.dtype)
    gray01 = to_float01(img)

    os.makedirs(OUT_DIR, exist_ok=True)

    clahe_list: List[np.ndarray] = []
    binaries: List[np.ndarray] = []
    metrics: List[Dict[str, float]] = []

    for t255 in LOWER_THRESHOLDS_255:
        t01 = float(t255) / 255.0
        clahe_np, binary_np, thr_used, porosity = run_pipeline_with_clip(gray01, t01)

        # simple components metric (2D)
        lbl = label(binary_np)
        n_comp = int(lbl.max())

        clahe_list.append(clahe_np)
        binaries.append(binary_np)
        metrics.append({
            "clip_lower_255": t255,
            "clip_lower_01": t01,
            "bin_thresh_used_01": thr_used,
            "binary_porosity": porosity,
            "connected_components": n_comp
        })

    plot_mosaic_binary(LOWER_THRESHOLDS_255, binaries, os.path.join(OUT_DIR, "mosaic_binary.png"))
    plot_mosaic_overlay(LOWER_THRESHOLDS_255, gray01, binaries, os.path.join(OUT_DIR, "mosaic_overlay.png"))

    with open(os.path.join(OUT_DIR, "sweep_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    # also CSV
    import csv
    with open(os.path.join(OUT_DIR, "sweep_metrics.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(metrics[0].keys()))
        w.writeheader()
        w.writerows(metrics)

    print(f"[sweep] Saved mosaics and metrics to: {os.path.abspath(OUT_DIR)}")


if __name__ == "__main__":
    main()
