"""
VisionCluster AI - Phase 1 Environment & Directory Setup Inspection
Checks dependencies, hardware acceleration (PyTorch / CUDA / CPU),
and initializes the project directory structure.
"""

import sys
import importlib.util
from pathlib import Path

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.visioncluster_config import (
    PROJECT_ROOT,
    REQUIRED_DIRECTORIES,
    create_project_directories,
)


def check_dependencies():
    """Verifies that all required packages are present in the environment."""
    packages = {
        "numpy": "NumPy",
        "pandas": "Pandas",
        "PIL": "Pillow",
        "sklearn": "Scikit-learn",
        "torch": "PyTorch",
        "torchvision": "Torchvision",
        "streamlit": "Streamlit",
        "plotly": "Plotly",
        "cv2": "OpenCV",
    }

    print("\nDEPENDENCY CHECK")
    print("-" * 45)

    missing = []

    for module, name in packages.items():
        installed = importlib.util.find_spec(module) is not None
        status = "INSTALLED" if installed else "MISSING"
        print(f"{name:20} : {status}")

        if not installed:
            missing.append(name)

    return missing


def check_directories():
    """Checks whether the required project directories exist."""
    print("\nPROJECT DIRECTORY CHECK")
    print("-" * 45)

    for directory in REQUIRED_DIRECTORIES:
        exists = directory.exists()
        rel_path = directory.relative_to(PROJECT_ROOT) if directory.is_relative_to(PROJECT_ROOT) else directory
        print(f"{str(rel_path):35} : {'READY' if exists else 'MISSING'}")


def main():
    print("=" * 60)
    print("       VISIONCLUSTER AI - PHASE 1 ENVIRONMENT CHECK")
    print("=" * 60)

    print(f"\nPython version: {sys.version.split()[0]}")
    print(f"Python executable: {sys.executable}")
    print(f"Project root: {PROJECT_ROOT}")

    create_project_directories()
    check_directories()

    missing = check_dependencies()

    print("\nDEVICE CHECK")
    print("-" * 45)

    if importlib.util.find_spec("torch"):
        import torch

        print(f"PyTorch version: {torch.__version__}")
        print(f"CUDA available: {torch.cuda.is_available()}")

        if torch.cuda.is_available():
            print(f"GPU: {torch.cuda.get_device_name(0)}")
        else:
            print("Using CPU for machine learning.")
    else:
        print("PyTorch not installed yet.")

    print("\n" + "=" * 60)

    if missing:
        print("Setup incomplete. Missing packages:", ", ".join(missing))
    else:
        print("All required packages are available and verified.")

    print("Phase 1 environment inspection completed successfully.")
    print("=" * 60)


if __name__ == "__main__":
    main()
