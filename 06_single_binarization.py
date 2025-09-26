#!/usr/bin/env python3
"""
06_single_binarization.py

Purpose
- Run the exact grayscale→binary image-processing pipeline used for RVEs, but on a full mid-CT slice once per stack.
- Produce a stable, dataset-wide global threshold and parameters to avoid RVE-dependent bias.
- Save comparison plots (grayscale, CLAHE, binary, and overlay), and emit a JSON with the exact values to paste into 05_full_pipeline.py.

Notes
- This script targets CT stacks of 8-bit images (BMP/PNG/TIF). Intensities are normalized to [0,1] like in the main pipeline.
- Steps mirror 05_full_pipeline.py: SimpleITK denoise → clip low values → CLAHE → global threshold (mean by default) → binary → thin/remove small objects/holes (in 2D).
- Only parameters are reported; it does not edit 05_full_pipeline.py automatically.

How to use
1) Edit the GLOBALS below (paths and parameters).
2) Run this file directly with Python. No command-line flags are used.

Outputs
- single_binarization_params.json (recommended constants to set in 05_full_pipeline.py)
- mid_slice_grayscale.png, mid_slice_clahe.png, mid_slice_binary.png, mid_slice_overlay.png
"""

from __future__ import annotations
import json
import os
import re
import cv2
from typing import List, Tuple

import numpy as np
import SimpleITK as sitk
import matplotlib.pyplot as plt
from skimage import io as skio
from skimage.filters import threshold_otsu
import skimage.morphology as skm
from skimage.measure import find_contours


# ======================
# GLOBAL CONFIGURATIONS
# ======================

# Input CT/E8 images directory (edit this)
IMAGES_DIR: str = "/mnt/c/Users/bherr/Downloads/CT-1um/images_reconstructed_1um"

# Optional filename substring filter (e.g., 'binary_v3.bmp')
PATTERN: str | None = None

# Optional explicit slice index; default is the mid-slice if None
SLICE_INDEX: int | None = None

# Output directory for plots and JSON
OUT_DIR: str = "results/CT/single_binarization"

# Preprocessing params (match 05 pipeline defaults)
DENOISE_TIME_STEP: float = 0.0625
DENOISE_CONDUCTANCE: float = 10.0
DENOISE_ITERATIONS: int = 30
CLIP_LOWER: float = 30 / 255.0
CLAHE_ALPHA: float = 0.5
CLAHE_BETA: float = 0.7

# Threshold method selection: "mean" | "otsu" | "fixed"
THRESHOLD_METHOD: str = "mean"
THRESHOLD_FIXED: float | None = None  # used only when THRESHOLD_METHOD == "fixed"

# 2D Postprocessing (for visualization parity)
POST_REMOVE_SMALL_OBJECTS: int = 250
POST_THIN_MAX_ITER: int = 2
POST_REMOVE_SMALL_HOLES: int = 2000


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
    """Convert an image to float32 in [0,1] without per-slice min-max scaling.

    This mirrors 05_full_pipeline's assumption that intensities are in [0,1]
    (e.g., 8-bit images divided by 255). If the input is uint8/uint16, map by
    known max. If it's already float and appears to be in 0..255, scale down.
    As a last resort, clip to [0,1].
    """
    if img.dtype == np.uint8:
        return img.astype(np.float32) / 255.0
    if img.dtype == np.uint16:
        return img.astype(np.float32) / 65535.0
    if np.issubdtype(img.dtype, np.floating):
        imgf = img.astype(np.float32)
        mx = float(np.nanmax(imgf))
        if mx > 1.5:  # likely 0..255
            imgf = imgf / 255.0
        return np.clip(imgf, 0.0, 1.0)
    # Fallback: map by known min/max but avoid stretching too much
    imgf = img.astype(np.float32)
    imin, imax = float(np.min(imgf)), float(np.max(imgf))
    if imax <= imin:
        return np.zeros_like(imgf, dtype=np.float32)
    return np.clip((imgf - imin) / (imax - imin), 0.0, 1.0)


def pipeline_on_slice(img_norm: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    # img_norm is float32 in [0,1]; wrap as SimpleITK image (2D)
    img_sitk = sitk.GetImageFromArray(img_norm)

    # 1) Denoise (Curvature Anisotropic Diffusion)
    denoised = sitk.CurvatureAnisotropicDiffusion(
        img_sitk,
        timeStep=DENOISE_TIME_STEP,
        conductanceParameter=DENOISE_CONDUCTANCE,
        numberOfIterations=DENOISE_ITERATIONS,
    )

    # 2) Clip low intensities
    clipped = sitk.Threshold(denoised, lower=CLIP_LOWER, upper=1.0, outsideValue=0.0)

    # 3) CLAHE (Adaptive Histogram Equalization)
    clahe = sitk.AdaptiveHistogramEqualization(clipped, alpha=CLAHE_ALPHA, beta=CLAHE_BETA)

    # Convert CLAHE result to NumPy for threshold determination
    clahe_np = sitk.GetArrayFromImage(clahe).astype(np.float32, copy=False)

    # 4) Global threshold method selection
    if THRESHOLD_METHOD == "mean":
        thr = float(np.mean(clahe_np))
    elif THRESHOLD_METHOD == "otsu":
        thr = float(threshold_otsu(clahe_np))
    else:
        if THRESHOLD_FIXED is None:
            raise ValueError("THRESHOLD_FIXED must be set when THRESHOLD_METHOD='fixed'")
        thr = float(THRESHOLD_FIXED)

    # 5) Binary threshold using selected thr
    binary = sitk.BinaryThreshold(clahe, lowerThreshold=thr, upperThreshold=1.0, outsideValue=0, insideValue=1)
    binary_np = sitk.GetArrayFromImage(binary).astype(bool, copy=False)
    binary_np = np.squeeze(binary_np)

    # 6) 2D postprocessing analogous to 3D pipeline
    if binary_np.ndim == 2:
        if POST_REMOVE_SMALL_OBJECTS and POST_REMOVE_SMALL_OBJECTS > 0:
            binary_np = skm.remove_small_objects(binary_np, min_size=int(POST_REMOVE_SMALL_OBJECTS))
        if POST_THIN_MAX_ITER and POST_THIN_MAX_ITER > 0:
            binary_np = skm.thin(binary_np, max_num_iter=int(POST_THIN_MAX_ITER))
        if POST_REMOVE_SMALL_HOLES and POST_REMOVE_SMALL_HOLES > 0:
            binary_np = skm.remove_small_holes(binary_np, area_threshold=int(POST_REMOVE_SMALL_HOLES))
    else:
        # Fallback: keep binary as-is; downstream plots/metrics will squeeze too
        print(f"[single-binarization] Warning: binary array has ndim={binary_np.ndim}, skipping 2D post-processing")

    return img_norm, clahe_np, binary_np, thr


def save_plots(out_dir: str, gray: np.ndarray, clahe: np.ndarray, binary: np.ndarray) -> None:
    os.makedirs(out_dir, exist_ok=True)

    # Grayscale
    gray = np.squeeze(gray)
    plt.figure(figsize=(6, 6))
    plt.imshow(gray, cmap="gray")
    plt.axis("off")
    plt.title("Mid-slice grayscale")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "mid_slice_grayscale.png"), dpi=300)
    plt.close()

    # CLAHE
    clahe = np.squeeze(clahe)
    plt.figure(figsize=(6, 6))
    plt.imshow(clahe, cmap="gray", vmin=0, vmax=1)
    plt.axis("off")
    plt.title("CLAHE result")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "mid_slice_clahe.png"), dpi=300)
    plt.close()

    # Binary
    binary = np.squeeze(binary)
    plt.figure(figsize=(6, 6))
    plt.imshow(binary, cmap="gray")
    plt.axis("off")
    plt.title("Binary (global threshold)")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "mid_slice_binary.png"), dpi=300)
    plt.close()

    # Overlay: red outline of binary on grayscale
    contours = find_contours(binary.astype(float), level=0.5)
    plt.figure(figsize=(6, 6))
    plt.imshow(gray, cmap="gray")
    for c in contours:
        plt.plot(c[:, 1], c[:, 0], color="red", linewidth=1.0)
    plt.axis("off")
    plt.title("Overlay: binary contour on grayscale")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "mid_slice_overlay.png"), dpi=300)
    plt.close()


def main() -> None:
    files = collect_images(IMAGES_DIR, PATTERN)
    idx = SLICE_INDEX if SLICE_INDEX is not None else len(files) // 2
    if idx < 0 or idx >= len(files):
        raise IndexError(f"slice-index {idx} out of range for {len(files)} files")

    path = files[idx]
    print(f"[single-binarization] Using mid-slice index {idx} -> {os.path.basename(path)}")

    img = cv2.imread(path, cv2.IMREAD_UNCHANGED).astype(np.float32)
    print(f"img shape = {img.shape}, dtype={img.dtype}, min={np.min(img)}, max={np.max(img)}")
    gray = to_float01(img)
    print(f"Converted to float32 in [0,1]: dtype={gray.dtype}, min={np.min(gray):.3f}, max={np.max(gray):.3f}")
    # plot quick histogram of values
    plt.figure(figsize=(9, 4))
    plt.subplot(1,2,1)
    plt.hist(gray.ravel(), bins=256, range=(0.0, 1.0), color="gray")
    plt.title("Histogram of grayscale intensities")
    plt.xlabel("Intensity (0..1)")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.subplot(1,2,2)
    plt.imshow(gray, cmap="gray")
    plt.axis("off")
    plt.title("Mid-slice grayscale")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "mid_slice_histogram.png"), dpi=300)
    plt.close()

    gray, clahe_np, binary_np, thr = pipeline_on_slice(gray)
    print(f"[single-binarization] Shapes gray={np.squeeze(gray).shape}, clahe={np.squeeze(clahe_np).shape}, binary={np.squeeze(binary_np).shape}")

    # Save plots
    os.makedirs(OUT_DIR, exist_ok=True)
    save_plots(OUT_DIR, gray, clahe_np, binary_np)

    # Prepare recommendations for 05_full_pipeline.py
    thr_01 = float(np.clip(thr, 0.0, 1.0))
    thr_255 = float(thr_01 * 255.0)

    porosity = float(np.mean(binary_np))

    recommendations = {
        "images_dir": os.path.abspath(IMAGES_DIR),
        "slice_index": int(idx),
        "mid_slice_file": os.path.basename(path),
        "image_shape": list(gray.shape),
        "binary_porosity_mid_slice": porosity,
        "preprocess": {
            "timeStep": float(DENOISE_TIME_STEP),
            "conductanceParameter": float(DENOISE_CONDUCTANCE),
            "numberOfIterations": int(DENOISE_ITERATIONS),
            "clip_lower": float(CLIP_LOWER)
        },
        "clahe": {"alpha": float(CLAHE_ALPHA), "beta": float(CLAHE_BETA)},
        "threshold": {
            "method": THRESHOLD_METHOD,
            "value_0_1": thr_01,
            "value_0_255": thr_255
        },
        "post_2d": {
            "remove_small_objects_min_size": int(POST_REMOVE_SMALL_OBJECTS),
            "thin_max_iter": int(POST_THIN_MAX_ITER),
            "remove_small_holes_area": int(POST_REMOVE_SMALL_HOLES)
        },
        "paste_into_05": {
            "replace_line_comment": "Replace BinaryThreshold lowerThreshold=np.mean(clahe) with the constant value below",
            "code_snippet": [
                "# In 05_full_pipeline.py (LOOP1 binarization phase):",
                "# Before:",
                "# binary = sitk.BinaryThreshold(clahe, lowerThreshold=np.mean(clahe), upperThreshold=255/255, outsideValue=0, insideValue=1)",
                "# After (use dataset-wide fixed threshold computed here):",
                f"binary = sitk.BinaryThreshold(clahe, lowerThreshold={thr_01:.6f}, upperThreshold=255/255, outsideValue=0, insideValue=1)",
            ]
        }
    }

    with open(os.path.join(OUT_DIR, "single_binarization_params.json"), "w") as f:
        json.dump(recommendations, f, indent=2)

    print("\n[single-binarization] Suggested 05_full_pipeline.py replacement:")
    for line in recommendations["paste_into_05"]["code_snippet"]:
        print(line)
    print("\n[single-binarization] Threshold (0..1):", f"{thr_01:.6f}")
    print("[single-binarization] Threshold (0..255):", f"{thr_255:.2f}")
    print(f"[single-binarization] Saved outputs to: {os.path.abspath(OUT_DIR)}")


if __name__ == "__main__":
    main()
