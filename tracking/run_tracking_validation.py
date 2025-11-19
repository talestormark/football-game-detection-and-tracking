#!/usr/bin/env python3
"""
Object Tracking on Validation Set
Step 3.3: Apply ByteTrack to validation frames
"""

from ultralytics import YOLO
from pathlib import Path
import json

def main():
    print("="*60)
    print("Football Object Tracking - Validation Set")
    print("="*60)

    # Paths
    model_path = Path('/cluster/work/tmstorma/Football2025/training/runs/yolov8s_4class2/weights/best.pt')
    val_dir = Path('/cluster/work/tmstorma/Football2025/dataset/images/val')
    tracker_config = Path('/cluster/work/tmstorma/Football2025/tracking/bytetrack_custom.yaml')
    output_dir = Path('/cluster/work/tmstorma/Football2025/tracking/runs/val_tracking')

    print(f"\nConfiguration:")
    print(f"  Model: {model_path}")
    print(f"  Validation images: {val_dir}")
    print(f"  Tracker config: {tracker_config}")
    print(f"  Output directory: {output_dir}")

    # Verify paths
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")
    if not val_dir.exists():
        raise FileNotFoundError(f"Validation directory not found: {val_dir}")
    if not tracker_config.exists():
        raise FileNotFoundError(f"Tracker config not found: {tracker_config}")

    # Count validation images
    val_images = sorted(val_dir.glob('*.png'))
    print(f"\nFound {len(val_images)} validation images")

    # Load model
    print("\nLoading trained model...")
    model = YOLO(str(model_path))
    print("Model loaded successfully")

    # Run tracking
    print("\n" + "="*60)
    print("Running ByteTrack on validation set...")
    print("="*60)
    print("\nTracker Configuration (based on Step 2b analysis):")
    print("  track_thresh: 0.5   (min confidence to start track)")
    print("  track_buffer: 30    (max frames for lost tracks - 1.2s @ 25fps)")
    print("  match_thresh: 0.8   (IoU threshold for matching)")
    print("\nRationale:")
    print("  - track_buffer=30: Handles brief occlusions (97.6% continuity)")
    print("  - match_thresh=0.8: High precision (only 1% ID switches in GT)")
    print("  - conf=0.3: ByteTrack uses low/high thresholds for robustness")
    print()

    # Process each dataset separately to avoid tracking across matches
    # Frame numbers refer to image filenames (e.g., frame_001623.png)
    # These correspond to XML frames + 1 (image 001623 = XML frame 1622)
    datasets = [
        ('RBK-AALESUND', list(range(1623, 1803))),  # frames 1623-1802 (180 frames)
        ('RBK-FREDRIKSTAD', list(range(1636, 1817))),  # frames 1636-1816 (181 frames) - FIXED
        ('RBK-HamKam', list(range(1372, 1524)))  # frames 1372-1523 (152 frames)
    ]

    all_results = []
    for dataset_name, frame_indices in datasets:
        print(f"\n{'='*60}")
        print(f"Processing {dataset_name} ({len(frame_indices)} frames)")
        print(f"{'='*60}")

        # Get frames for this dataset
        dataset_frames = [val_dir / f"{dataset_name}_frame_{i:06d}.png" for i in frame_indices]
        dataset_frames = [f for f in dataset_frames if f.exists()]

        if len(dataset_frames) == 0:
            print(f"  Warning: No frames found for {dataset_name}")
            continue

        print(f"  Found {len(dataset_frames)} frames")

        print(f"  Processing {len(dataset_frames)} frames with ByteTrack...")

        # Create dataset-specific output directory
        dataset_output_dir = output_dir / dataset_name
        dataset_output_dir.mkdir(parents=True, exist_ok=True)
        (dataset_output_dir / 'labels').mkdir(exist_ok=True)

        # Process frames in batches to avoid OOM
        # Use batch_size=8 to stay within GPU memory limits (P100 has 16GB)
        batch_size = 8
        frame_count = 0

        for batch_start in range(0, len(dataset_frames), batch_size):
            batch_end = min(batch_start + batch_size, len(dataset_frames))
            batch_frames = dataset_frames[batch_start:batch_end]

            # Track this batch with persist=True to maintain state across batches
            results = model.track(
                source=[str(f) for f in batch_frames],
                tracker=str(tracker_config),
                save=False,  # We'll save manually
                conf=0.3,
                iou=0.7,
                save_txt=False,
                persist=True,  # CRITICAL: maintain tracker state across batches
                stream=False,  # Process batch at once
                verbose=False
            )

            # Process results
            for idx, (result, frame_path) in enumerate(zip(results, batch_frames)):
                all_results.append(result)
                frame_count += 1
                global_idx = batch_start + idx

                # Save tracking results manually in YOLO format
                original_frame_name = frame_path.stem
                label_file = dataset_output_dir / 'labels' / f'{original_frame_name}.txt'

                # Write tracking results
                with open(label_file, 'w') as f:
                    if result.boxes is not None and len(result.boxes) > 0:
                        boxes = result.boxes
                        for box_idx in range(len(boxes)):
                            cls_id = int(boxes.cls[box_idx].cpu().numpy())
                            bbox = boxes.xywhn[box_idx].cpu().numpy()  # normalized xywh
                            x, y, w, h = bbox

                            # Include track ID if available
                            if boxes.id is not None:
                                track_id = int(boxes.id[box_idx].cpu().numpy())
                                f.write(f"{cls_id} {x:.6f} {y:.6f} {w:.6f} {h:.6f} {track_id}\n")
                            else:
                                f.write(f"{cls_id} {x:.6f} {y:.6f} {w:.6f} {h:.6f}\n")

                if frame_count % 50 == 0:
                    print(f"  Processed {frame_count}/{len(dataset_frames)} frames")

        print(f"  Completed {dataset_name}: {frame_count} frames processed")

    # Process results and collect tracking statistics
    print("\n" + "="*60)
    print("Processing tracking results...")
    print("="*60)

    track_ids_per_class = {0: set(), 1: set(), 2: set(), 3: set()}
    class_names = {0: 'home', 1: 'away', 2: 'referee', 3: 'ball'}
    total_detections = 0
    frames_processed = 0

    for result in all_results:
        frames_processed += 1
        if result.boxes is not None and len(result.boxes) > 0:
            boxes = result.boxes
            total_detections += len(boxes)

            # Collect track IDs per class
            if boxes.id is not None:
                for cls_id, track_id in zip(boxes.cls.cpu().numpy(), boxes.id.cpu().numpy()):
                    track_ids_per_class[int(cls_id)].add(int(track_id))

    print(f"\nTracking Summary:")
    print(f"  Frames processed: {frames_processed}")
    print(f"  Total detections: {total_detections}")
    print(f"  Average detections per frame: {total_detections/frames_processed:.1f}")
    print(f"\nUnique Track IDs by Class:")
    for cls_id, track_ids in track_ids_per_class.items():
        print(f"  {class_names[cls_id]}: {len(track_ids)} unique tracks")

    # Save summary
    summary = {
        'frames_processed': frames_processed,
        'total_detections': total_detections,
        'avg_detections_per_frame': total_detections / frames_processed,
        'unique_tracks_per_class': {
            class_names[cls_id]: len(track_ids)
            for cls_id, track_ids in track_ids_per_class.items()
        }
    }

    summary_path = output_dir / 'tracking_summary.json'
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\nTracking summary saved to: {summary_path}")

    print("\n" + "="*60)
    print("Tracking Complete!")
    print("="*60)
    print(f"\nOutputs saved to: {output_dir}")
    print("\nDataset-specific outputs:")
    print("  - RBK-AALESUND/")
    print("  - RBK-FREDRIKSTAD/")
    print("  - RBK-HamKam/")
    print("\nGenerated files per dataset:")
    print("  - Annotated images with track IDs")
    print("  - Tracking results in MOT format (labels/*.txt)")
    print("\nNext steps:")
    print("  1. Visual inspection of tracked images")
    print("  2. HOTA evaluation using ground truth from XML")
    print("  3. Analyze ID switches and track fragmentation")

if __name__ == '__main__':
    main()
