#!/usr/bin/env python3
"""
YOLOv8 Training Script for Football Object Detection
4 classes: home, away, referee, ball
"""

import os
from pathlib import Path
from ultralytics import YOLO
import torch

def main():
    print("="*60)
    print("YOLOv8 Football Object Detection Training")
    print("="*60)

    # Check GPU availability
    print(f"\nPyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        print(f"GPU count: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
            print(f"    Memory: {torch.cuda.get_device_properties(i).total_memory / 1024**3:.1f} GB")
    print()

    # Paths
    dataset_yaml = Path('/cluster/work/tmstorma/Football2025/dataset/data.yaml')
    project_dir = Path('/cluster/work/tmstorma/Football2025/training/runs')
    resume_checkpoint = Path('/cluster/work/tmstorma/Football2025/training/runs/yolov8s_4class2/weights/last.pt')

    print(f"Dataset config: {dataset_yaml}")
    print(f"Output directory: {project_dir}")
    print(f"Resume checkpoint: {resume_checkpoint if resume_checkpoint.exists() else 'None (training from scratch)'}")
    print()

    # Verify dataset exists
    if not dataset_yaml.exists():
        raise FileNotFoundError(f"Dataset config not found: {dataset_yaml}")

    # Training configuration
    config = {
        'data': str(dataset_yaml),
        'epochs': 100,
        'imgsz': 1280,  # Critical for ball detection
        'batch': 16,    # Adjust based on GPU memory
        'lr0': 0.01,
        'optimizer': 'AdamW',
        'patience': 15,
        'save': True,
        'save_period': 5,  # Save checkpoint every 5 epochs
        'workers': 8,
        'device': 0,  # GPU 0
        'project': str(project_dir),
        'name': 'yolov8s_4class2',
        'exist_ok': True,  # Allow resuming from existing run
        'pretrained': True,
        'verbose': True,
        'val': True,
        'plots': True,
        'resume': resume_checkpoint.exists(),  # Resume if checkpoint exists
    }

    print("Training Configuration:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    print()

    # Load YOLOv8-s model (resume from checkpoint or start fresh)
    if resume_checkpoint.exists():
        print(f"Resuming from checkpoint: {resume_checkpoint}")
        model = YOLO(str(resume_checkpoint))
        print("Checkpoint loaded successfully")
    else:
        print("Loading YOLOv8-s model with COCO pretrained weights...")
        model = YOLO('yolov8s.pt')
        print(f"Model loaded: {model.model}")
    print()

    # Start training
    print("Starting training...")
    print("="*60)
    results = model.train(**config)

    print()
    print("="*60)
    print("Training completed!")
    print("="*60)

    # Print best metrics
    print("\nBest Metrics:")
    print(f"  mAP@0.5: {results.results_dict.get('metrics/mAP50(B)', 'N/A')}")
    print(f"  mAP@0.5:0.95: {results.results_dict.get('metrics/mAP50-95(B)', 'N/A')}")
    print(f"  Precision: {results.results_dict.get('metrics/precision(B)', 'N/A')}")
    print(f"  Recall: {results.results_dict.get('metrics/recall(B)', 'N/A')}")

    # Save final model location
    best_model_path = project_dir / 'yolov8s_4class2' / 'weights' / 'best.pt'
    print(f"\nBest model saved to: {best_model_path}")

    return results


if __name__ == '__main__':
    main()
