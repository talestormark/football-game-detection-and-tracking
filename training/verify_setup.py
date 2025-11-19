#!/usr/bin/env python3
"""
Verify training setup before launching full training job.
"""

import sys
from pathlib import Path

def check_dataset():
    """Verify dataset structure and files."""
    print("Checking dataset...")
    dataset_dir = Path('/cluster/work/tmstorma/Football2025/dataset')

    required_files = [
        'data.yaml',
        'train.txt',
        'val.txt',
        'gt_tracking.json'
    ]

    for file in required_files:
        filepath = dataset_dir / file
        if filepath.exists():
            print(f"  ✓ {file}")
        else:
            print(f"  ✗ {file} - MISSING!")
            return False

    # Check image and label directories
    train_images = dataset_dir / 'images' / 'train'
    val_images = dataset_dir / 'images' / 'val'
    train_labels = dataset_dir / 'labels' / 'train'
    val_labels = dataset_dir / 'labels' / 'val'

    if train_images.exists():
        n_train_imgs = len(list(train_images.glob('*.png')))
        print(f"  ✓ Train images: {n_train_imgs}")
    else:
        print(f"  ✗ Train images directory missing!")
        return False

    if val_images.exists():
        n_val_imgs = len(list(val_images.glob('*.png')))
        print(f"  ✓ Val images: {n_val_imgs}")
    else:
        print(f"  ✗ Val images directory missing!")
        return False

    if train_labels.exists():
        n_train_labels = len(list(train_labels.glob('*.txt')))
        print(f"  ✓ Train labels: {n_train_labels}")
    else:
        print(f"  ✗ Train labels directory missing!")
        return False

    if val_labels.exists():
        n_val_labels = len(list(val_labels.glob('*.txt')))
        print(f"  ✓ Val labels: {n_val_labels}")
    else:
        print(f"  ✗ Val labels directory missing!")
        return False

    print(f"  Dataset OK: {n_train_imgs} train, {n_val_imgs} val images\n")
    return True


def check_environment():
    """Check Python environment and packages."""
    print("Checking Python environment...")
    print(f"  Python: {sys.version.split()[0]}")

    try:
        import torch
        print(f"  ✓ PyTorch: {torch.__version__}")
        print(f"  ✓ CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"    CUDA version: {torch.version.cuda}")
    except ImportError:
        print(f"  ✗ PyTorch not installed!")
        return False

    try:
        import ultralytics
        print(f"  ✓ Ultralytics: {ultralytics.__version__}")
    except ImportError:
        print(f"  ✗ Ultralytics not installed!")
        return False

    try:
        import cv2
        print(f"  ✓ OpenCV: {cv2.__version__}")
    except ImportError:
        print(f"  ✗ OpenCV not installed!")
        return False

    print("  Environment OK\n")
    return True


def main():
    print("="*60)
    print("YOLOv8 Training Setup Verification")
    print("="*60)
    print()

    dataset_ok = check_dataset()
    env_ok = check_environment()

    print("="*60)
    if dataset_ok and env_ok:
        print("✓ All checks passed! Ready to start training.")
        print("\nTo submit training job:")
        print("  cd /cluster/work/tmstorma/Football2025/training")
        print("  sbatch train_yolov8_gpu.sh")
    else:
        print("✗ Some checks failed. Please fix the issues above.")
        sys.exit(1)
    print("="*60)


if __name__ == '__main__':
    main()
