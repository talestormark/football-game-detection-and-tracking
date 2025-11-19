#!/usr/bin/env python3
"""
Object Tracking on Generalization Datasets
Step 3.4: Test tracking on RBK-VIKING and RBK-BODO-part3
"""

from ultralytics import YOLO
from pathlib import Path
import json

def track_dataset(model, dataset_name, images_dir, tracker_config, output_base):
    """Track a single dataset"""
    print("\n" + "="*60)
    print(f"Tracking {dataset_name}")
    print("="*60)

    # Count images
    images = sorted(images_dir.glob('*.png'))
    print(f"Found {len(images)} images")

    # Run tracking
    output_dir = output_base / dataset_name
    results = model.track(
        source=str(images_dir),
        tracker=str(tracker_config),
        save=True,
        conf=0.3,
        iou=0.7,
        show_labels=True,
        show_conf=True,
        save_txt=True,
        project=str(output_dir.parent),
        name=output_dir.name,
        exist_ok=True,
        stream=True,
        verbose=False  # Less verbose for long sequences
    )

    # Collect statistics
    track_ids_per_class = {0: set(), 1: set(), 2: set(), 3: set()}
    class_names = {0: 'home', 1: 'away', 2: 'referee', 3: 'ball'}
    total_detections = 0
    frames_processed = 0

    print("Processing frames...", end='', flush=True)
    for i, result in enumerate(results):
        if i % 100 == 0:
            print(f"\rProcessing frames... {i}/{len(images)}", end='', flush=True)

        frames_processed += 1
        if result.boxes is not None and len(result.boxes) > 0:
            boxes = result.boxes
            total_detections += len(boxes)

            if boxes.id is not None:
                for cls_id, track_id in zip(boxes.cls.cpu().numpy(), boxes.id.cpu().numpy()):
                    track_ids_per_class[int(cls_id)].add(int(track_id))

    print(f"\rProcessing frames... {frames_processed}/{len(images)} - Done!")

    # Summary
    summary = {
        'dataset': dataset_name,
        'frames_processed': frames_processed,
        'total_detections': total_detections,
        'avg_detections_per_frame': total_detections / frames_processed if frames_processed > 0 else 0,
        'unique_tracks_per_class': {
            class_names[cls_id]: len(track_ids)
            for cls_id, track_ids in track_ids_per_class.items()
        }
    }

    # Save summary
    summary_path = output_dir / 'tracking_summary.json'
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\nSummary:")
    print(f"  Frames: {frames_processed}")
    print(f"  Detections: {total_detections} ({total_detections/frames_processed:.1f}/frame)")
    print(f"  Unique tracks: {sum(len(ids) for ids in track_ids_per_class.values())}")
    for cls_id, track_ids in track_ids_per_class.items():
        if len(track_ids) > 0:
            print(f"    - {class_names[cls_id]}: {len(track_ids)}")

    return summary

def main():
    print("="*60)
    print("Generalization Testing - Object Tracking")
    print("="*60)

    # Paths
    model_path = Path('/cluster/work/tmstorma/Football2025/training/runs/yolov8s_4class2/weights/best.pt')
    tracker_config = Path('/cluster/work/tmstorma/Football2025/tracking/bytetrack_custom.yaml')
    base_images_dir = Path('/cluster/projects/vc/courses/TDT17/other/Football2025')
    output_base = Path('/cluster/work/tmstorma/Football2025/tracking/runs/generalization')

    print(f"\nConfiguration:")
    print(f"  Model: {model_path}")
    print(f"  Tracker: {tracker_config}")
    print(f"  Output: {output_base}")

    # Load model
    print("\nLoading model...")
    model = YOLO(str(model_path))
    print("Model loaded successfully")

    # Datasets to test
    datasets = [
        ('RBK-VIKING', base_images_dir / 'RBK-VIKING' / 'img1'),
        ('RBK-BODO-part3', base_images_dir / 'RBK-BODO-part3' / 'img1')
    ]

    results_all = []

    for dataset_name, images_dir in datasets:
        if not images_dir.exists():
            print(f"\nWARNING: {dataset_name} not found at {images_dir}")
            continue

        summary = track_dataset(model, dataset_name, images_dir, tracker_config, output_base)
        results_all.append(summary)

    # Overall summary
    print("\n" + "="*60)
    print("Generalization Testing Complete!")
    print("="*60)

    print("\nOverall Results:")
    for summary in results_all:
        print(f"\n{summary['dataset']}:")
        print(f"  Frames: {summary['frames_processed']}")
        print(f"  Avg detections/frame: {summary['avg_detections_per_frame']:.1f}")
        print(f"  Unique tracks: {sum(summary['unique_tracks_per_class'].values())}")

    print(f"\nOutputs saved to: {output_base}")
    print("\nNote: These datasets lack proper team labels (Step 1b finding)")
    print("Evaluation is qualitative only - inspect tracked videos manually")

if __name__ == '__main__':
    main()
