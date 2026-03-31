"""
Cat color threshold tuner.

Detects the green bounding box already drawn on the images, crops the ROI
inside it, and computes HSV stats. No YOLO re-detection needed.

Usage:
    python tune_colors.py
"""

import os
import cv2
import numpy as np

BASE_DIR = "cat images"
CATS = ["bonnie", "jinny", "louise"]

# The bounding box color drawn by app.py: BGR (0, 230, 60)
# In HSV (OpenCV 0-180 scale) this is roughly hue=52, S=255, V=230
BOX_HSV_LOWER = np.array([42, 150, 150])
BOX_HSV_UPPER = np.array([62, 255, 255])
BOX_THICKNESS = 2   # pixels to inset so we don't sample the border itself


def find_box_roi(img: np.ndarray) -> np.ndarray | None:
    """Return the ROI inside the green bounding box, or None if not found."""
    hsv  = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, BOX_HSV_LOWER, BOX_HSV_UPPER)

    # Connect nearby green pixels so thin lines form solid contours
    kernel  = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    mask    = cv2.dilate(mask, kernel, iterations=1)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return None

    # Pick the largest contour (the rectangle outline)
    c  = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(c)

    if w < 20 or h < 20:
        return None

    # Inset by box thickness so we sample only the cat, not the border
    pad = BOX_THICKNESS + 2
    roi = img[y + pad: y + h - pad, x + pad: x + w - pad]
    return roi if roi.size > 0 else None


def hsv_stats(roi: np.ndarray) -> dict:
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    v = hsv[:, :, 2].astype(float)
    s = hsv[:, :, 1].astype(float)
    h = hsv[:, :, 0].astype(float)
    warm_frac = float(np.mean((h >= 8) & (h <= 25) & (s > 60)))
    return {
        "mean_v":    np.mean(v),
        "std_v":     np.std(v),
        "warm_frac": warm_frac,
    }


def process_folder(cat: str) -> list[dict]:
    folder = os.path.join(BASE_DIR, cat)
    images = [
        os.path.join(folder, f) for f in sorted(os.listdir(folder))
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ]
    if not images:
        print(f"  No images found in {folder}")
        return []

    samples = []
    for img_path in images:
        img = cv2.imread(img_path)
        if img is None:
            continue
        roi = find_box_roi(img)
        if roi is None:
            print(f"  {os.path.basename(img_path)}: no bounding box found, skipping")
            continue
        stats = hsv_stats(roi)
        samples.append(stats)
        print(f"  {os.path.basename(img_path):40s}  mean_v={stats['mean_v']:5.1f}  std_v={stats['std_v']:5.1f}  warm_frac={stats['warm_frac']:.3f}")

    return samples


def print_summary(results: dict[str, list[dict]]):
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    for cat, samples in results.items():
        if not samples:
            print(f"\n{cat}: no samples")
            continue
        mean_vs    = [s["mean_v"]    for s in samples]
        std_vs     = [s["std_v"]     for s in samples]
        warm_fracs = [s["warm_frac"] for s in samples]
        print(f"\n{cat} ({len(samples)} samples):")
        print(f"  mean_v:    {min(mean_vs):5.1f} – {max(mean_vs):5.1f}  (avg {sum(mean_vs)/len(mean_vs):.1f})")
        print(f"  std_v:     {min(std_vs):5.1f} – {max(std_vs):5.1f}  (avg {sum(std_vs)/len(std_vs):.1f})")
        print(f"  warm_frac: {min(warm_fracs):.3f} – {max(warm_fracs):.3f}  (avg {sum(warm_fracs)/len(warm_fracs):.3f})")

    print("\n" + "=" * 60)
    print("RAW DATA")
    print("=" * 60)
    print(f"{'cat':<12} {'mean_v':>8} {'std_v':>8} {'warm_frac':>10}")
    for cat, samples in results.items():
        for s in samples:
            print(f"{cat:<12} {s['mean_v']:8.1f} {s['std_v']:8.1f} {s['warm_frac']:10.3f}")


def main():
    results = {}
    for cat in CATS:
        print(f"\n--- {cat} ---")
        results[cat] = process_folder(cat)

    print_summary(results)


if __name__ == "__main__":
    main()
