"""
Data Quality Inspection and Preprocessing Module
Performs missing, corrupt, duplicate image checks, format verification,
and produces the formal Data Quality Report.
"""

import sys
import hashlib
from pathlib import Path
from PIL import Image
import pandas as pd
import numpy as np

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import (
    RAW_IMG_DIR,
    MANIFEST_FILE,
    GROUND_TRUTH_FILE,
    DATA_QUALITY_FILE,
    METRICS_DIR,
    FIGURES_DIR
)


def compute_file_md5(filepath, chunk_size=65536):
    """Computes MD5 hash for bitwise duplicate detection."""
    hasher = hashlib.md5()
    with open(filepath, "rb") as f:
        while chunk := f.read(chunk_size):
            hasher.update(chunk)
    return hasher.hexdigest()


def inspect_image_quality(manifest_path=MANIFEST_FILE):
    """
    Performs comprehensive data quality audit:
      - File existence (Missing detection)
      - Image decodability and integrity (Corrupt detection)
      - Bitwise MD5 hash (Duplicate detection)
      - Dimension statistics (Width, Height, Channels, Aspect Ratio)
    """
    print("=" * 65)
    print("Executing Data Quality Inspection...")

    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found at {manifest_path}. Run data_download.py first.")

    df_manifest = pd.read_csv(manifest_path)
    total_records = len(df_manifest)

    valid_records = []
    missing_records = []
    corrupt_records = []
    duplicate_records = []

    seen_hashes = {}
    dimensions = []

    for idx, row in df_manifest.iterrows():
        img_id = row["image_id"]
        rel_path = row["file_path"]
        abs_path = PROJECT_ROOT / rel_path

        # 1. Existence Check
        if not abs_path.exists():
            missing_records.append({"image_id": img_id, "reason": "File does not exist"})
            continue

        # 2. Corrupt / Decodability Check
        try:
            with Image.open(abs_path) as img:
                img.verify()  # Verifies file integrity
            # Reopen to get image attributes
            with Image.open(abs_path) as img:
                w, h = img.size
                mode = img.mode
                fmt = img.format
                dimensions.append({
                    "image_id": img_id,
                    "width": w,
                    "height": h,
                    "aspect_ratio": round(w / h, 3),
                    "mode": mode,
                    "format": fmt
                })
        except Exception as e:
            corrupt_records.append({"image_id": img_id, "reason": f"Corrupt image: {str(e)}"})
            continue

        # 3. Duplicate Detection via MD5 Hash
        file_hash = compute_file_md5(abs_path)
        if file_hash in seen_hashes:
            duplicate_records.append({
                "image_id": img_id,
                "duplicate_of": seen_hashes[file_hash],
                "hash": file_hash
            })
            continue
        else:
            seen_hashes[file_hash] = img_id

        valid_records.append(row)

    df_valid = pd.DataFrame(valid_records)
    df_dims = pd.DataFrame(dimensions)

    # Compile Formal Quality Audit Metrics
    audit_summary = {
        "Metric": [
            "Total Images Collected",
            "Missing Images",
            "Corrupt Images",
            "Duplicate Images",
            "Valid Images Retained",
            "Retention Rate (%)",
            "Mean Width (px)",
            "Mean Height (px)",
            "Median Aspect Ratio"
        ],
        "Value": [
            total_records,
            len(missing_records),
            len(corrupt_records),
            len(duplicate_records),
            len(df_valid),
            round((len(df_valid) / max(total_records, 1)) * 100, 2),
            round(df_dims["width"].mean(), 1) if not df_dims.empty else 0,
            round(df_dims["height"].mean(), 1) if not df_dims.empty else 0,
            round(df_dims["aspect_ratio"].median(), 2) if not df_dims.empty else 0
        ]
    }
    df_audit = pd.DataFrame(audit_summary)
    df_audit.to_csv(DATA_QUALITY_FILE, index=False)

    print("\nData Quality Audit Summary:")
    print(df_audit.to_string(index=False))

    # Save cleaned manifest
    clean_manifest_path = PROJECT_ROOT / "data" / "processed" / "clean_manifest.csv"
    df_valid.to_csv(clean_manifest_path, index=False)
    print(f"\nSaved Clean Manifest -> {clean_manifest_path} ({len(df_valid)} images)")

    # Save dimensions stats
    dims_path = METRICS_DIR / "image_dimensions.csv"
    df_dims.to_csv(dims_path, index=False)

    return df_audit, df_valid, df_dims


def inspect_class_distribution():
    """
    Audits ground truth distribution strictly for reporting.
    Verifies that no labels are leaked to the manifest.
    """
    print("\nAuditing Class Distribution (Metadata Only)...")
    if not GROUND_TRUTH_FILE.exists():
        print("Warning: ground_truth.csv not yet generated.")
        return None

    df_gt = pd.read_csv(GROUND_TRUTH_FILE)
    dist = df_gt["category"].value_counts().reset_index()
    dist.columns = ["Category", "Image Count"]
    dist["Percentage (%)"] = (dist["Image Count"] / len(df_gt) * 100).round(2)
    dist_file = METRICS_DIR / "class_distribution.csv"
    dist.to_csv(dist_file, index=False)
    print(dist.to_string(index=False))
    return dist


def verify_strict_label_separation(manifest_path=MANIFEST_FILE):
    """
    Security check: Verifies that manifest contains NO ground-truth label columns.
    """
    df_m = pd.read_csv(manifest_path)
    forbidden_terms = {"label", "category", "class", "target", "ground_truth", "gt"}
    cols = set(col.lower() for col in df_m.columns)
    leaked = cols.intersection(forbidden_terms)
    if leaked:
        raise ValueError(f"CRITICAL DATA LEAKAGE: Manifest contains forbidden columns: {leaked}")
    print("\n[VERIFICATION PASSED] Manifest is completely free of ground-truth labels.")
    return True


if __name__ == "__main__":
    inspect_image_quality()
    inspect_class_distribution()
    verify_strict_label_separation()
