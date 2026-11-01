"""
Data Acquisition Module for Google Open Images
Downloads and organizes images across eight categories with strict label separation.
Uses official Open Images S3 mirrors with parallel thread pool downloads.
"""

import os
import sys
import csv
import io
import time
import urllib.request
import urllib.error
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import defaultdict
import pandas as pd
from PIL import Image

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import (
    RAW_IMG_DIR,
    METADATA_DIR,
    TARGET_CLASSES,
    CATEGORY_NAMES,
    TARGET_PER_CLASS,
    RANDOM_SEED,
    MANIFEST_FILE,
    GROUND_TRUTH_FILE
)

ANNOTATION_URLS = {
    "validation": "https://storage.googleapis.com/openimages/v5/validation-annotations-human-imagelabels.csv",
    "test": "https://storage.googleapis.com/openimages/v5/test-annotations-human-imagelabels.csv",
}

S3_BASE_URLS = {
    "validation": "https://s3.amazonaws.com/open-images-dataset/validation/{}.jpg",
    "test": "https://s3.amazonaws.com/open-images-dataset/test/{}.jpg",
}


def fetch_annotations_index(splits=("validation", "test")):
    """
    Downloads and parses image-level annotations from Open Images.
    Returns:
        dict: category -> list of (image_id, split)
    """
    print("=" * 65)
    print("Fetching Open Images annotations index...")
    category_candidates = defaultdict(list)
    image_categories = defaultdict(set)

    for split in splits:
        url = ANNOTATION_URLS[split]
        print(f"Downloading annotation manifest for split: {split}...")
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        try:
            with urllib.request.urlopen(req, timeout=45) as resp:
                content = resp.read().decode("utf-8")
                reader = csv.reader(io.StringIO(content))
                header = next(reader)
                for row in reader:
                    # Row: ImageID, Source, LabelName, Confidence
                    if len(row) >= 4 and row[3] == "1":
                        mid = row[2]
                        if mid in TARGET_CLASSES:
                            cat = TARGET_CLASSES[mid]
                            img_id = row[0]
                            image_categories[img_id].add(cat)
                            category_candidates[cat].append((img_id, split))
        except Exception as e:
            print(f"Error fetching annotations for {split}: {e}")

    # Prioritize single-label images to reduce ambiguous multi-category visual clutter
    pure_candidates = defaultdict(list)
    multi_candidates = defaultdict(list)

    for cat, items in category_candidates.items():
        seen = set()
        for img_id, split in items:
            if img_id in seen:
                continue
            seen.add(img_id)
            if len(image_categories[img_id]) == 1:
                pure_candidates[cat].append((img_id, split))
            else:
                multi_candidates[cat].append((img_id, split))

    final_candidates = {}
    import random
    rng = random.Random(RANDOM_SEED)

    for cat in CATEGORY_NAMES:
        pures = pure_candidates[cat]
        multis = multi_candidates[cat]
        rng.shuffle(pures)
        rng.shuffle(multis)
        combined = pures + multis
        final_candidates[cat] = combined
        print(f"  Category '{cat}': {len(pures)} pure + {len(multis)} multi candidate images available.")

    return final_candidates


def download_single_image(args):
    """Worker function to download an individual image and verify validity."""
    img_id, split, category, dest_dir = args
    target_path = dest_dir / f"{img_id}.jpg"

    if target_path.exists() and target_path.stat().st_size > 1000:
        try:
            with Image.open(target_path) as img:
                img.verify()
            return True, img_id, split, category, str(target_path), target_path.stat().st_size, "Already Cached"
        except Exception:
            try:
                target_path.unlink()
            except Exception:
                pass

    url = S3_BASE_URLS[split].format(img_id)
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})

    for attempt in range(3):
        try:
            with urllib.request.urlopen(req, timeout=12) as response:
                data = response.read()
                if len(data) < 1000:
                    return False, img_id, split, category, None, 0, "File too small"

                # Verify PIL readable
                img = Image.open(io.BytesIO(data))
                img.verify()

                with open(target_path, "wb") as f:
                    f.write(data)
                return True, img_id, split, category, str(target_path), len(data), "OK"
        except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError) as e:
            time.sleep(0.5 * (attempt + 1))
        except Exception as e:
            return False, img_id, split, category, None, 0, str(e)

    return False, img_id, split, category, None, 0, "Max Retries Exceeded"


def download_dataset(target_per_class=TARGET_PER_CLASS, max_workers=16):
    """
    Downloads balanced target images per category into data/raw/{category}/.
    Generates strict separate metadata:
      - data/metadata/image_manifest.csv (NO LABELS)
      - data/metadata/ground_truth.csv (Ground truth strictly isolated)
    """
    print("=" * 65)
    print(f"Starting Open Images download: Target {target_per_class} images per category")
    print(f"Target Categories ({len(CATEGORY_NAMES)}): {CATEGORY_NAMES}")
    print(f"Total Target Dataset Size: {target_per_class * len(CATEGORY_NAMES)} images")
    print("=" * 65)

    candidates = fetch_annotations_index()
    download_tasks = []

    for cat in CATEGORY_NAMES:
        cat_dir = RAW_IMG_DIR / cat
        cat_dir.mkdir(parents=True, exist_ok=True)
        # Select buffer candidates in case of download or corrupt errors
        cat_cands = candidates[cat][: int(target_per_class * 1.5)]
        for img_id, split in cat_cands:
            download_tasks.append((img_id, split, cat, cat_dir))

    successful_by_cat = defaultdict(list)
    manifest_records = []
    ground_truth_records = []

    print(f"\nDispatching {len(download_tasks)} tasks across {max_workers} worker threads...")
    t0 = time.time()
    completed_count = 0

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_map = {executor.submit(download_single_image, task): task for task in download_tasks}
        for future in as_completed(future_map):
            completed_count += 1
            success, img_id, split, cat, path, size, status = future.result()

            if success and len(successful_by_cat[cat]) < target_per_class:
                successful_by_cat[cat].append((img_id, path, size))
                manifest_records.append({
                    "image_id": img_id,
                    "file_path": str(Path(path).relative_to(PROJECT_ROOT)),
                    "file_size_bytes": size
                })
                ground_truth_records.append({
                    "image_id": img_id,
                    "category": cat,
                    "split": split,
                    "confidence": 1.0
                })

            if completed_count % 200 == 0 or completed_count == len(download_tasks):
                progress = (completed_count / len(download_tasks)) * 100
                total_collected = sum(len(v) for v in successful_by_cat.values())
                print(f"Progress: {progress:5.1f}% | Completed tasks: {completed_count}/{len(download_tasks)} | Images verified: {total_collected}")

            # Check if all categories reached quota
            if all(len(successful_by_cat[c]) >= target_per_class for c in CATEGORY_NAMES):
                print("\nAll categories reached target quota! Finalizing active downloads...")
                break

    elapsed = time.time() - t0
    print(f"\nDownload phase completed in {elapsed:.1f}s.")

    # Save manifest (UNSUPERVISED - NO LABELS)
    df_manifest = pd.DataFrame(manifest_records).sort_values("image_id").reset_index(drop=True)
    df_manifest.to_csv(MANIFEST_FILE, index=False)
    print(f"Saved Image Manifest (NO LABELS) -> {MANIFEST_FILE} ({len(df_manifest)} entries)")

    # Save ground truth (ISOLATED - FOR POST-HOC EVALUATION ONLY)
    df_gt = pd.DataFrame(ground_truth_records).sort_values("image_id").reset_index(drop=True)
    df_gt.to_csv(GROUND_TRUTH_FILE, index=False)
    print(f"Saved Ground Truth (ISOLATED)    -> {GROUND_TRUTH_FILE} ({len(df_gt)} entries)")

    # Summary table
    print("\nFinal Image Quota Summary:")
    print("-" * 40)
    for cat in CATEGORY_NAMES:
        print(f"  {cat:<12}: {len(successful_by_cat[cat])} images")
    print("-" * 40)
    print(f"  Total Valid Images: {len(df_manifest)}")

    return df_manifest, df_gt


if __name__ == "__main__":
    download_dataset(target_per_class=TARGET_PER_CLASS, max_workers=16)
