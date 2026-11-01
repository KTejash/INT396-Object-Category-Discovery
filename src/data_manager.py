"""
Data Manager Module for Object Discovery Lab
Handles dataset loading, multi-image upload, ZIP extraction, validation,
MD5 bitwise duplicate detection, data quality auditing, and strict label isolation.
"""

import os
import sys
import io
import shutil
import zipfile
import hashlib
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Any
import numpy as np
import pandas as pd
from PIL import Image

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import (
    BASE_DIR,
    DATA_DIR,
    RAW_IMG_DIR,
    METADATA_DIR,
    MANIFEST_FILE,
    GROUND_TRUTH_FILE,
    RANDOM_SEED
)

CUSTOM_DATA_DIR = DATA_DIR / "custom"
CUSTOM_DATA_DIR.mkdir(parents=True, exist_ok=True)

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}


def compute_md5(data: bytes) -> str:
    """Computes MD5 hash for bitwise duplicate detection."""
    return hashlib.md5(data).hexdigest()


def compute_file_md5(filepath: Path, chunk_size: int = 65536) -> str:
    """Computes MD5 hash of a file on disk."""
    hasher = hashlib.md5()
    with open(filepath, "rb") as f:
        while chunk := f.read(chunk_size):
            hasher.update(chunk)
    return hasher.hexdigest()


class DatasetManager:
    """
    Unified manager for built-in benchmark datasets and custom user-uploaded datasets.
    """

    def __init__(self, dataset_name: str = "open_images_benchmark"):
        self.dataset_name = dataset_name
        self.dataset_dir = CUSTOM_DATA_DIR / dataset_name if dataset_name != "open_images_benchmark" else RAW_IMG_DIR
        self.metadata_dir = (CUSTOM_DATA_DIR / dataset_name / "metadata") if dataset_name != "open_images_benchmark" else METADATA_DIR
        self.metadata_dir.mkdir(parents=True, exist_ok=True)
        
        self.manifest_path = self.metadata_dir / "image_manifest.csv"
        self.ground_truth_path = self.metadata_dir / "ground_truth.csv"

    def list_available_datasets(self) -> List[str]:
        """Lists available datasets (built-in + any custom datasets)."""
        datasets = ["open_images_benchmark"]
        if CUSTOM_DATA_DIR.exists():
            for item in CUSTOM_DATA_DIR.iterdir():
                if item.is_dir() and item.name != "open_images_benchmark":
                    datasets.append(item.name)
        return datasets

    def load_manifest(self) -> pd.DataFrame:
        """Loads image manifest (guaranteed free of ground-truth labels)."""
        if self.manifest_path.exists():
            df = pd.read_csv(self.manifest_path)
            # Verify no leaked labels
            forbidden = {"label", "category", "class", "target", "ground_truth"}
            leaked = forbidden.intersection([c.lower() for c in df.columns])
            if leaked:
                raise ValueError(f"Security Alert: Manifest contains forbidden columns: {leaked}")
            return df
        return pd.DataFrame(columns=["image_id", "file_path", "width", "height", "aspect_ratio", "format", "file_size_bytes"])

    def load_ground_truth(self) -> Optional[pd.DataFrame]:
        """Loads ground-truth labels if available (for post-hoc evaluation only)."""
        if self.ground_truth_path.exists():
            return pd.read_csv(self.ground_truth_path)
        return None

    def audit_dataset(self) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
        """
        Performs full data quality inspection:
        - Decodability / corrupt image check
        - Bitwise duplicate detection
        - Dimensional analysis
        Returns:
            (audit_summary_df, valid_images_df, detailed_stats_dict)
        """
        df_manifest = self.load_manifest()
        if df_manifest.empty:
            summary_df = pd.DataFrame({
                "Metric": ["Total Records", "Valid Images", "Corrupt Images", "Duplicates", "Format Supported"],
                "Value": [0, 0, 0, 0, "N/A"]
            })
            return summary_df, pd.DataFrame(), {}

        valid_records = []
        corrupt_records = []
        duplicate_records = []
        seen_hashes = {}

        for _, row in df_manifest.iterrows():
            img_id = str(row["image_id"])
            rel_path = str(row["file_path"])
            abs_path = PROJECT_ROOT / rel_path

            if not abs_path.exists():
                corrupt_records.append({"image_id": img_id, "reason": "Missing from disk"})
                continue

            try:
                with open(abs_path, "rb") as f:
                    file_bytes = f.read()

                # Verify with PIL
                img = Image.open(io.BytesIO(file_bytes))
                img.verify()

                # Check bitwise duplicate
                h = compute_md5(file_bytes)
                if h in seen_hashes:
                    duplicate_records.append({
                        "image_id": img_id,
                        "duplicate_of": seen_hashes[h],
                        "hash": h
                    })
                    continue
                seen_hashes[h] = img_id

                # Reopen to read dimensions safely
                with Image.open(io.BytesIO(file_bytes)) as img_opened:
                    w, h_dim = img_opened.size
                    fmt = img_opened.format or Path(rel_path).suffix.replace(".", "").upper()

                valid_records.append({
                    "image_id": img_id,
                    "file_path": rel_path,
                    "width": w,
                    "height": h_dim,
                    "aspect_ratio": round(w / max(h_dim, 1), 3),
                    "format": fmt,
                    "file_size_bytes": len(file_bytes)
                })
            except Exception as e:
                corrupt_records.append({"image_id": img_id, "reason": str(e)})

        df_valid = pd.DataFrame(valid_records)
        total_collected = len(df_manifest)

        summary_metrics = {
            "Metric": [
                "Total Images Analyzed",
                "Valid Images Retained",
                "Corrupt / Unreadable Images",
                "Bitwise Duplicates Removed",
                "Retention Rate (%)",
                "Mean Width (px)",
                "Mean Height (px)",
                "Median Aspect Ratio"
            ],
            "Value": [
                total_collected,
                len(df_valid),
                len(corrupt_records),
                len(duplicate_records),
                round((len(df_valid) / max(total_collected, 1)) * 100, 2),
                round(df_valid["width"].mean(), 1) if not df_valid.empty else 0,
                round(df_valid["height"].mean(), 1) if not df_valid.empty else 0,
                round(df_valid["aspect_ratio"].median(), 2) if not df_valid.empty else 0
            ]
        }
        df_summary = pd.DataFrame(summary_metrics)

        stats = {
            "corrupt_count": len(corrupt_records),
            "duplicate_count": len(duplicate_records),
            "valid_count": len(df_valid),
            "formats": df_valid["format"].value_counts().to_dict() if not df_valid.empty else {}
        }

        return df_summary, df_valid, stats

    def create_custom_dataset_from_files(
        self,
        dataset_name: str,
        uploaded_files: List[Any],
        extract_labels_from_structure: bool = True,
        max_images: Optional[int] = None,
        random_seed: int = RANDOM_SEED
    ) -> Tuple[int, int, str]:
        """
        Creates a custom dataset from multiple uploaded files or a single/multiple ZIP archives.
        Supports JPG, JPEG, PNG, WEBP.
        Returns:
            (valid_count, corrupt_count, message)
        """
        safe_name = "".join(c if c.isalnum() or c in ("_", "-") else "_" for c in dataset_name).strip()
        if not safe_name:
            safe_name = "custom_dataset"

        target_dir = CUSTOM_DATA_DIR / safe_name
        target_images_dir = target_dir / "images"
        target_meta_dir = target_dir / "metadata"
        
        target_images_dir.mkdir(parents=True, exist_ok=True)
        target_meta_dir.mkdir(parents=True, exist_ok=True)

        extracted_raw_files: List[Tuple[str, bytes, Optional[str]]] = []  # (filename, bytes, category)

        for uploaded in uploaded_files:
            fname = uploaded.name
            ext = Path(fname).suffix.lower()

            # Handle ZIP archive
            if ext == ".zip":
                try:
                    with zipfile.ZipFile(uploaded) as z:
                        for member in z.namelist():
                            if member.startswith("__MACOSX/") or member.endswith("/"):
                                continue
                            m_ext = Path(member).suffix.lower()
                            if m_ext in SUPPORTED_EXTENSIONS:
                                data = z.read(member)
                                parts = Path(member).parts
                                category = parts[0] if len(parts) > 1 and extract_labels_from_structure else None
                                extracted_raw_files.append((Path(member).name, data, category))
                except Exception as e:
                    print(f"Error unzipping {fname}: {e}")
            elif ext in SUPPORTED_EXTENSIONS:
                data = uploaded.read()
                extracted_raw_files.append((fname, data, None))

        if not extracted_raw_files:
            return 0, 0, "No supported image files found (supported: JPG, JPEG, PNG, WEBP, ZIP)."

        # Optional subsampling / cap
        if max_images is not None and len(extracted_raw_files) > max_images:
            import random
            rng = random.Random(random_seed)
            rng.shuffle(extracted_raw_files)
            extracted_raw_files = extracted_raw_files[:max_images]

        manifest_records = []
        ground_truth_records = []
        seen_hashes = set()
        corrupt_count = 0

        for i, (fname, file_bytes, category) in enumerate(extracted_raw_files):
            try:
                # Validate image integrity with PIL
                img = Image.open(io.BytesIO(file_bytes))
                img.verify()
                
                # Reopen to read dimensions
                with Image.open(io.BytesIO(file_bytes)) as img_opened:
                    w, h = img_opened.size
                    fmt = img_opened.format or "JPEG"

                h_val = compute_md5(file_bytes)
                if h_val in seen_hashes:
                    continue  # skip duplicate
                seen_hashes.add(h_val)

                img_id = f"custom_{i:05d}_{h_val[:8]}"
                save_filename = f"{img_id}.{fmt.lower()}"
                save_path = target_images_dir / save_filename
                with open(save_path, "wb") as f:
                    f.write(file_bytes)

                rel_path = str(save_path.relative_to(PROJECT_ROOT))
                manifest_records.append({
                    "image_id": img_id,
                    "file_path": rel_path,
                    "width": w,
                    "height": h,
                    "aspect_ratio": round(w / max(h, 1), 3),
                    "format": fmt,
                    "file_size_bytes": len(file_bytes)
                })

                if category:
                    ground_truth_records.append({
                        "image_id": img_id,
                        "category": category,
                        "confidence": 1.0
                    })
            except Exception:
                corrupt_count += 1

        df_m = pd.DataFrame(manifest_records)
        df_m.to_csv(target_meta_dir / "image_manifest.csv", index=False)

        if ground_truth_records:
            df_gt = pd.DataFrame(ground_truth_records)
            df_gt.to_csv(target_meta_dir / "ground_truth.csv", index=False)

        return len(df_m), corrupt_count, f"Dataset '{safe_name}' created with {len(df_m)} valid images ({corrupt_count} skipped/corrupt)."

    def delete_images(self, image_ids_to_remove: List[str]) -> int:
        """Removes specified image IDs from the dataset manifest and disk."""
        df_m = self.load_manifest()
        if df_m.empty:
            return 0

        to_remove = set(image_ids_to_remove)
        keep_mask = ~df_m["image_id"].isin(to_remove)
        df_kept = df_m[keep_mask].reset_index(drop=True)

        removed_count = 0
        for _, row in df_m[~keep_mask].iterrows():
            abs_p = PROJECT_ROOT / str(row["file_path"])
            if abs_p.exists():
                try:
                    abs_p.unlink()
                    removed_count += 1
                except Exception:
                    pass

        df_kept.to_csv(self.manifest_path, index=False)

        # Also remove from ground truth if present
        if self.ground_truth_path.exists():
            df_gt = pd.read_csv(self.ground_truth_path)
            df_gt = df_gt[~df_gt["image_id"].isin(to_remove)].reset_index(drop=True)
            df_gt.to_csv(self.ground_truth_path, index=False)

        return removed_count
