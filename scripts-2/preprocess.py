import os
from pathlib import Path
import pandas as pd
import numpy as np
from PIL import Image
from tqdm.auto import tqdm

# Configuration constants
L_MAX = 100  # max number of fixations per sequence
CACHE_STATS = Path("fixstats.npz")
EPS = 1e-6   # to avoid division by zero

def compute_fixation_stats(fix_paths, cols):
    """
    Compute global mean and std for fixation features across all studies.
    Returns mu (4,), sigma (4,)
    """
    sum_ = np.zeros(len(cols), dtype=np.float64)
    sumsq_ = np.zeros(len(cols), dtype=np.float64)
    count = 0
    for p in tqdm(fix_paths, desc="Scanning fixations for stats"):
        df = pd.read_csv(p, usecols=cols)
        arr = df.values.astype(np.float64)
        sum_ += arr.sum(axis=0)
        sumsq_ += (arr**2).sum(axis=0)
        count += arr.shape[0]
    mu = sum_ / count
    var = sumsq_ / count - mu**2
    sigma = np.sqrt(np.maximum(var, EPS))
    # save for later
    np.savez(CACHE_STATS, mu=mu.astype(np.float32), sigma=sigma.astype(np.float32))
    return mu.astype(np.float32), sigma.astype(np.float32)

def process_image(image_path, out_dir):
    img = Image.open(image_path).convert("RGB")
    orig_w, orig_h = img.size
    img_resized = img.resize((512, 512), Image.BILINEAR)
    out_file = out_dir / f"{Path(image_path).stem}.png"
    img_resized.save(out_file)
    return orig_w, orig_h

def process_bbox(bbox_path, orig_size, out_dir):
    orig_w, orig_h = orig_size
    df = pd.read_csv(bbox_path)
    mask = np.zeros((512, 512), dtype=np.uint8)
    scale_x = 512.0 / orig_w
    scale_y = 512.0 / orig_h
    for _, r in df.iterrows():
        x1 = int(r['x1'] * scale_x)
        x2 = int(r['x2'] * scale_x)
        y1 = int(r['y1'] * scale_y)
        y2 = int(r['y2'] * scale_y)
        x1, x2 = np.clip([x1, x2], 0, 511)
        y1, y2 = np.clip([y1, y2], 0, 511)
        mask[y1:y2, x1:x2] = 1
    out_file = out_dir / f"{Path(bbox_path).stem}.png"
    Image.fromarray(mask * 255).save(out_file)

def process_fixations(fix_path, mu, sigma, out_dir):
    df = pd.read_csv(fix_path, usecols=["x_norm", "y_norm", "gaze_duration", "pupil_area_normalized"])
    arr = df.values.astype(np.float32)
    # Check for empty or all-NaN input
    if arr.size == 0 or np.isnan(arr).all():
        print(f"[WARN] Empty or all-NaN fixation file: {fix_path}. Output will be all zeros.")
        bad = True
        arr = np.zeros((L_MAX, 4), dtype=np.float32)
    else:
        bad = False
        # Clip raw
        arr[:, 0] = np.clip(arr[:, 0], 0.0, 1.0)
        arr[:, 1] = np.clip(arr[:, 1], 0.0, 1.0)
        arr[:, 2] = np.clip(arr[:, 2], 0.0, None)
        arr[:, 3] = np.clip(arr[:, 3], 0.0, 1.0)
        # Standardize with ε-guarded sigma
        arr = (arr - mu) / (sigma + EPS)
        # Clean any NaN/∞
        arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    # Truncate / pad
    length = min(arr.shape[0], L_MAX)
    seq = np.zeros((L_MAX, 4), dtype=np.float32)
    mask = np.zeros((L_MAX,), dtype=np.bool_)
    if not bad:
        seq[:length] = arr[:length]
        mask[:length] = True
    # Additional check: warn if output seq is all zeros
    if np.all(seq == 0):
        print(f"[WARN] Output fixation sequence is all zeros for file: {fix_path}")
        bad = True
    out_file = out_dir / f"{Path(fix_path).stem}.npz"
    np.savez_compressed(out_file, seq=seq, mask=mask, bad=bad)

def main():
    # Hard-coded paths (no argparse)
    CSV_PATH = Path(r'E:\MIMIC Research\Code\Report-Generation-and-Disease-Classification\final_dataset_fixed.csv')
    OUT_DIR  = Path(r'E:\MIMIC Research\Code\Report-Generation-and-Disease-Classification\output')

    df = pd.read_csv(CSV_PATH)
    img_out  = OUT_DIR / "img_png"
    bbox_out = OUT_DIR / "bbox_mask"
    fix_out  = OUT_DIR / "fix_seq"
    img_out .mkdir(parents=True, exist_ok=True)
    bbox_out.mkdir(parents=True, exist_ok=True)
    fix_out .mkdir(parents=True, exist_ok=True)

    # Compute or load fixation stats
    fix_paths = df['fixations_path'].dropna().unique()
    if CACHE_STATS.exists():
        data = np.load(CACHE_STATS)
        mu, sigma = data['mu'], data['sigma']
    else:
        mu, sigma = compute_fixation_stats(
            fix_paths,
            ["x_norm", "y_norm", "gaze_duration", "pupil_area_normalized"]
        )

    # Run
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Preprocessing dataset"):
        orig_size = process_image(row['image_path'], img_out)
        if pd.notna(row['bbox_path']):
            process_bbox(row['bbox_path'], orig_size, bbox_out)
        if pd.notna(row['fixations_path']):
            process_fixations(row['fixations_path'], mu, sigma, fix_out)

    print("Preprocessing complete.")

if __name__ == '__main__':
    main()
