"""
Feature Extractor Module for Object Discovery Lab
Extracts:
1. Deep CNN Features (Pretrained ResNet50 GAP 2048-d)
2. 3D Color Histograms (Configurable Bins, L1/L2 Normalized)
3. Combined Representations (Weighted ResNet50 + Histogram Concatenation)
Supports batch inference, progress callbacks, and on-disk/memory caching.
"""

import sys
import time
from pathlib import Path
from typing import List, Tuple, Optional, Callable, Dict, Any
import numpy as np
import pandas as pd
from PIL import Image
import torch
import torch.nn as nn
from torchvision import models, transforms

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import (
    BASE_DIR,
    EMBEDDINGS_DIR,
    CNN_FEATURES_RAW,
    HIST_FEATURES_RAW,
    IMAGE_IDS_FILE,
    BATCH_SIZE
)

_CACHED_RESNET_MODEL = None
_CACHED_FULL_CLASSIFIER = None


def get_resnet50_model(device: Optional[torch.device] = None) -> nn.Module:
    """Loads and caches pretrained ResNet50 with classification head replaced by Identity."""
    global _CACHED_RESNET_MODEL
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if _CACHED_RESNET_MODEL is None:
        weights = models.ResNet50_Weights.DEFAULT
        model = models.resnet50(weights=weights)
        model.fc = nn.Identity()  # 2048-d GAP output
        model = model.to(device)
        model.eval()
        _CACHED_RESNET_MODEL = model
    return _CACHED_RESNET_MODEL


def get_full_classifier_model(device: Optional[torch.device] = None) -> nn.Module:
    """Loads and caches full ResNet50 model with pretrained 1000-class ImageNet head."""
    global _CACHED_FULL_CLASSIFIER
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if _CACHED_FULL_CLASSIFIER is None:
        weights = models.ResNet50_Weights.DEFAULT
        model = models.resnet50(weights=weights)
        model = model.to(device)
        model.eval()
        _CACHED_FULL_CLASSIFIER = model
    return _CACHED_FULL_CLASSIFIER



class PathDataset(torch.utils.data.Dataset):
    """PyTorch dataset that loads and transforms images directly from absolute file paths."""
    def __init__(self, file_paths: List[Path], transform=None):
        self.file_paths = file_paths
        self.transform = transform

    def __len__(self) -> int:
        return len(self.file_paths)

    def __getitem__(self, idx: int):
        p = self.file_paths[idx]
        with Image.open(p) as img:
            img = img.convert("RGB")
            if self.transform:
                img = self.transform(img)
            return img, idx


class FeatureExtractor:
    """
    Unified Feature Extraction engine supporting ResNet50, Color Histograms, and Combined spaces.
    """

    def __init__(self, device: Optional[str] = None):
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
            
        self.weights = models.ResNet50_Weights.DEFAULT
        self.resnet_transform = self.weights.transforms()

    def get_resnet50_model(self) -> nn.Module:
        """Returns cached ResNet50 model instance."""
        return get_resnet50_model(self.device)

    def predict_universal_class(self, img: Image.Image) -> Dict[str, Any]:
        """
        Runs the full 1000-class ImageNet classifier to accurately identify
        any real-world object (e.g. Orange, Banana, Persian Cat, Sports Car).
        """
        model = get_full_classifier_model(self.device)
        t_img = self.resnet_transform(img).unsqueeze(0).to(self.device)
        with torch.no_grad():
            logits = model(t_img)
            probs = torch.softmax(logits, dim=1).cpu().numpy()[0]

        top5_indices = np.argsort(probs)[::-1][:5]
        categories = self.weights.meta["categories"]

        top_raw_name = categories[top5_indices[0]]
        top_name = top_raw_name.replace("_", " ").title()
        top_prob = float(probs[top5_indices[0]] * 100)

        # Check for benchmark category mapping
        benchmark_mapping = self.map_imagenet_to_benchmark(top_raw_name)

        top5_list = []
        for idx in top5_indices:
            top5_list.append({
                "Object_Class": categories[idx].replace("_", " ").title(),
                "Probability_Pct": round(float(probs[idx] * 100), 2)
            })

        return {
            "top_object_name": top_name,
            "top_probability": round(top_prob, 2),
            "benchmark_equivalent": benchmark_mapping,
            "top5_classes": pd.DataFrame(top5_list)
        }

    @staticmethod
    def map_imagenet_to_benchmark(imagenet_label: str) -> Optional[str]:
        """
        Maps granular ImageNet classes to the 8 INT396 benchmark classes
        [Bird, Boat, Building, Bus, Car, Cat, Dog, Flower].
        Returns None if the object is Out-of-Distribution (e.g. orange, banana).
        """
        lbl = imagenet_label.lower().replace("_", " ")

        birds = ["bird", "robin", "hummingbird", "jay", "magpie", "chickadee", "kite",
                 "eagle", "vulture", "owl", "ostrich", "albatross", "pelican", "flamingo",
                 "heron", "crane", "duck", "goose", "swan", "cock", "hen", "turkey",
                 "quail", "partridge", "peacock", "macaw", "parrot", "lorikeet", "toucan"]
        if any(b in lbl for b in birds):
            return "Bird"

        boats = ["boat", "canoe", "kayak", "gondola", "speedboat", "lifeboat", "catamaran",
                 "trimaran", "yawl", "schooner", "aircraft carrier", "submarine", "liner",
                 "ship", "ferry"]
        if any(b in lbl for b in boats):
            return "Boat"

        buildings = ["building", "church", "cathedral", "monastery", "mosque", "temple",
                     "castle", "palace", "beacon", "lighthouse", "boathouse", "barn",
                     "library", "restaurant", "cinema", "planetarium", "prison"]
        if any(b in lbl for b in buildings):
            return "Building"

        buses = ["bus", "minibus", "school bus", "trolleybus", "double-decker"]
        if any(b in lbl for b in buses):
            return "Bus"

        cars = ["car", "cab", "taxi", "convertible", "sports car", "limousine", "minivan",
                "jeep", "suv", "coupe", "sedan", "racer", "model t"]
        if any(b in lbl for b in cars):
            return "Car"

        cats = ["cat", "tabby", "tiger cat", "persian cat", "siamese cat", "egyptian cat",
                "lynx", "leopard", "jaguar", "lion", "cheetah"]
        if any(c in lbl for c in cats):
            return "Cat"

        dogs = ["dog", "terrier", "retriever", "hound", "shepherd", "spaniel", "poodle",
                "bulldog", "boxer", "chihuahua", "mastiff", "husky", "malamute", "collie",
                "corgi", "dalmatian", "pug", "beagle", "rottweiler", "doberman", "dingo"]
        if any(d in lbl for d in dogs):
            return "Dog"

        flowers = ["flower", "daisy", "sunflower", "tulip", "rose", "orchid", "lily",
                   "pot", "vase", "garden", "blossom", "poppy", "dahlia", "carnation"]
        if any(f in lbl for f in flowers):
            return "Flower"

        return None


    def extract_resnet50(
        self,
        image_paths: List[Path],
        batch_size: int = BATCH_SIZE,
        progress_callback: Optional[Callable[[float, str], None]] = None
    ) -> np.ndarray:
        """
        Extracts 2048-dimensional features via ResNet50 GAP layer.
        """
        model = get_resnet50_model(self.device)
        dataset = PathDataset(image_paths, transform=self.resnet_transform)
        dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, shuffle=False, num_workers=0
        )

        features = []
        total = len(dataset)
        processed = 0

        with torch.no_grad():
            for i, (imgs, _) in enumerate(dataloader):
                imgs = imgs.to(self.device)
                feats = model(imgs)  # shape (B, 2048)
                features.append(feats.cpu().numpy())
                
                processed += len(imgs)
                if progress_callback:
                    pct = processed / total
                    progress_callback(pct, f"ResNet50: {processed}/{total} images extracted")

        feats_matrix = np.vstack(features).astype(np.float32)
        return feats_matrix

    def extract_color_histograms(
        self,
        image_paths: List[Path],
        bins_per_channel: int = 8,
        normalization: str = "l1",
        progress_callback: Optional[Callable[[float, str], None]] = None
    ) -> np.ndarray:
        """
        Extracts 3D RGB color histograms.
        bins_per_channel^3 dimensions (e.g. 4^3=64, 8^3=512, 16^3=4096).
        """
        histograms = []
        total = len(image_paths)
        bins = (bins_per_channel, bins_per_channel, bins_per_channel)

        for i, p in enumerate(image_paths):
            with Image.open(p) as img:
                arr = np.array(img.convert("RGB"))

            hist, _ = np.histogramdd(
                arr.reshape(-1, 3),
                bins=bins,
                range=[(0, 256), (0, 256), (0, 256)]
            )
            h_flat = hist.flatten().astype(np.float32)

            if normalization == "l1":
                norm = np.sum(h_flat)
                if norm > 0:
                    h_flat /= norm
            elif normalization == "l2":
                norm = np.linalg.norm(h_flat)
                if norm > 0:
                    h_flat /= norm

            histograms.append(h_flat)

            if progress_callback and ((i + 1) % 50 == 0 or (i + 1) == total):
                progress_callback((i + 1) / total, f"Color Histogram: {i + 1}/{total} images extracted")

        hist_matrix = np.vstack(histograms).astype(np.float32)
        return hist_matrix

    def combine_features(
        self,
        cnn_features: np.ndarray,
        hist_features: np.ndarray,
        cnn_weight: float = 1.0,
        hist_weight: float = 1.0
    ) -> np.ndarray:
        """
        Combines semantic CNN features with Color Histograms with configurable weighting.
        """
        w_cnn = cnn_features * float(cnn_weight)
        w_hist = hist_features * float(hist_weight)
        combined = np.hstack([w_cnn, w_hist]).astype(np.float32)
        return combined

    def extract_features_for_manifest(
        self,
        df_manifest: pd.DataFrame,
        feature_type: str = "resnet50",
        bins_per_channel: int = 8,
        batch_size: int = BATCH_SIZE,
        cnn_weight: float = 1.0,
        hist_weight: float = 1.0,
        progress_callback: Optional[Callable[[float, str], None]] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extracts features for all images listed in a manifest DataFrame.
        Returns:
            (features_matrix, image_ids)
        """
        if df_manifest.empty:
            raise ValueError("Manifest is empty. No images to extract features from.")

        image_ids = df_manifest["image_id"].values
        abs_paths = [PROJECT_ROOT / str(p) for p in df_manifest["file_path"]]

        if feature_type == "resnet50":
            feats = self.extract_resnet50(abs_paths, batch_size=batch_size, progress_callback=progress_callback)
        elif feature_type == "color_hist":
            feats = self.extract_color_histograms(abs_paths, bins_per_channel=bins_per_channel, progress_callback=progress_callback)
        elif feature_type == "combined":
            cb_cnn = (lambda p, m: progress_callback(p * 0.5, m)) if progress_callback else None
            cb_hist = (lambda p, m: progress_callback(0.5 + p * 0.5, m)) if progress_callback else None

            if progress_callback:
                progress_callback(0.0, "Starting dual feature extraction...")
            f_cnn = self.extract_resnet50(abs_paths, batch_size=batch_size, progress_callback=cb_cnn)
            f_hist = self.extract_color_histograms(abs_paths, bins_per_channel=bins_per_channel, progress_callback=cb_hist)
            feats = self.combine_features(f_cnn, f_hist, cnn_weight=cnn_weight, hist_weight=hist_weight)
        else:
            raise ValueError(f"Unknown feature_type: {feature_type}. Choose 'resnet50', 'color_hist', or 'combined'.")

        return feats, image_ids

    @staticmethod
    def compute_feature_statistics(feats: np.ndarray) -> Dict[str, Any]:
        """Calculates global statistical metrics of the extracted feature matrix."""
        l2_norms = np.linalg.norm(feats, axis=1)
        zero_pct = float(np.mean(feats == 0) * 100)
        return {
            "n_samples": int(feats.shape[0]),
            "n_features": int(feats.shape[1]),
            "mean_val": round(float(np.mean(feats)), 5),
            "std_val": round(float(np.std(feats)), 5),
            "min_val": round(float(np.min(feats)), 5),
            "max_val": round(float(np.max(feats)), 5),
            "sparsity_pct": round(zero_pct, 2),
            "mean_l2_norm": round(float(np.mean(l2_norms)), 4),
            "min_l2_norm": round(float(np.min(l2_norms)), 4),
            "max_l2_norm": round(float(np.max(l2_norms)), 4),
        }

    @staticmethod
    def save_features_to_cache(feats: np.ndarray, image_ids: np.ndarray, cache_file: Path):
        """Saves feature matrix and corresponding image IDs to compressed NumPy archive."""
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(cache_file, features=feats, image_ids=image_ids)

    @staticmethod
    def load_features_from_cache(cache_file: Path) -> Tuple[np.ndarray, np.ndarray]:
        """Loads cached feature matrix and image IDs."""
        data = np.load(cache_file, allow_pickle=True)
        return data["features"], data["image_ids"]

