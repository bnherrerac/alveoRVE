import os
import re
import json
import time
from typing import List

import numpy as np
import cv2
from numpy.lib.format import open_memmap


# Module-level configuration (no CLI by design)
#
# INPUT_DIR: directory with BMP slices named in increasing order (sorted by last integer in filename)
# OUTPUT_DIR: where to write <MODE>.npy and <MODE>.json (contiguous uint8 array and metadata)
# MODE: a short name matching the pipeline modes (e.g., "CT" or "E8"). 05_full_pipeline.py will look for MODE.npy / MODE.json here.
INPUT_DIR = "/mnt/c/Users/bherr/Downloads/E8-1um/images_reconstructed_1um"
OUTPUT_DIR = "/home/bnherrerac/CHASKi/alveoRVE/data/memmap_stacks"
MODE = "E8"
VERIFY_RANDOM_SLICES = 3  # set 0 to skip verification


def _numeric_key(name: str) -> int:
    m = re.findall(r"\d+", os.path.basename(name))
    return int(m[-1]) if m else -1


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out_npy = os.path.join(OUTPUT_DIR, f"{MODE}.npy")
    out_json = os.path.join(OUTPUT_DIR, f"{MODE}.json")

    files = [f for f in os.listdir(INPUT_DIR) if f.lower().endswith(".bmp")]
    if not files:
        raise RuntimeError(f"No .bmp files found in {INPUT_DIR}")
    paths = [os.path.join(INPUT_DIR, f) for f in sorted(files, key=_numeric_key)]

    # Read first image to get dimensions
    first = cv2.imread(paths[0], cv2.IMREAD_UNCHANGED)
    if first is None:
        raise RuntimeError(f"Failed to read first image: {paths[0]}")
    if first.ndim != 2:
        raise RuntimeError(f"Expected grayscale BMPs; got shape {first.shape}")
    if first.dtype != np.uint8:
        # We will store as uint8; ensure exact match to pipeline expectations
        first = first.astype(np.uint8)
    H, W = first.shape
    Z = len(paths)
    shape = (H, W, Z)
    dtype = np.uint8
    print(f"Packing stack: shape={shape} dtype={dtype}" )

    t0 = time.time()
    # Create a .npy file opened as a writable memmap and stream slices into it
    vol_mm = open_memmap(out_npy, mode='w+', dtype=dtype, shape=shape)
    vol_mm[:, :, 0] = first
    for zi, p in enumerate(paths[1:], start=1):
        img = cv2.imread(p, cv2.IMREAD_UNCHANGED)
        if img is None:
            raise RuntimeError(f"Failed to read image: {p}")
        if img.ndim != 2:
            raise RuntimeError(f"Non-grayscale image at {p} with shape {img.shape}")
        if img.dtype != np.uint8:
            img = img.astype(np.uint8)
        if img.shape != (H, W):
            raise RuntimeError(f"Inconsistent image shape at {p}: {img.shape} != {(H,W)}")
        vol_mm[:, :, zi] = img
        if zi % 200 == 0:
            print(f"  loaded slice {zi}/{Z}")
    # Flush changes
    del vol_mm

    # Save metadata
    meta = {
        "shape": [int(H), int(W), int(Z)],
        "dtype": str(dtype),
        "source": os.path.abspath(INPUT_DIR),
        "saved_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    with open(out_json, 'w') as f:
        json.dump(meta, f, indent=2, sort_keys=True)
    print(f"Wrote {out_npy} and {out_json} in {time.time()-t0:.1f}s")

    # Optional verification against original BMPs for a few random slices
    if VERIFY_RANDOM_SLICES > 0:
        import random
        idxs = random.sample(range(Z), k=min(VERIFY_RANDOM_SLICES, Z))
        arr = np.load(out_npy, mmap_mode='r')
        errs: List[str] = []
        for zi in idxs:
            bmp = cv2.imread(paths[zi], cv2.IMREAD_UNCHANGED)
            if bmp.dtype != np.uint8:
                bmp = bmp.astype(np.uint8)
            if not np.array_equal(arr[:, :, zi], bmp):
                errs.append(str(zi))
        if errs:
            print(f"[VERIFY] Mismatch at slices: {', '.join(errs)}")
        else:
            print("[VERIFY] Random slice checks passed (byte-identical)")


if __name__ == "__main__":
    main()
