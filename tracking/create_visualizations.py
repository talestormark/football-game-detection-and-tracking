#!/usr/bin/env python3
"""
Create tracking visualizations and highlight videos
"""

import cv2
import numpy as np
import xml.etree.ElementTree as ET
from pathlib import Path
from collections import defaultdict
import json

# Class configuration
CLASS_NAMES = {0: 'home', 1: 'away', 2: 'referee', 3: 'ball'}
CLASS_COLORS = {
    0: (0, 255, 0),      # home - green
    1: (255, 0, 0),      # away - blue
    2: (0, 255, 255),    # referee - yellow
    3: (255, 255, 255)   # ball - white
}

def parse_xml_annotations(xml_path, start_frame, end_frame):
    """Parse ground truth annotations from XML"""
    tree = ET.parse(xml_path)
    root = tree.getroot()

    annotations = defaultdict(list)

    for track in root.findall('.//track'):
        label = track.get('label')
        track_id = int(track.get('id'))

        # Map label to class_id
        if label == 'ball':
            class_id = 3
        else:
            team = next((attr.get('value') for attr in track.findall('attribute')
                        if attr.get('name') == 'team'), None)
            if team == 'home':
                class_id = 0
            elif team == 'away':
                class_id = 1
            elif team == 'referee':
                class_id = 2
            else:
                continue

        for box in track.findall('box'):
            frame_num = int(box.get('frame'))
            if start_frame <= frame_num <= end_frame:
                xtl = float(box.get('xtl'))
                ytl = float(box.get('ytl'))
                xbr = float(box.get('xbr'))
                ybr = float(box.get('ybr'))

                annotations[frame_num].append({
                    'track_id': track_id,
                    'class_id': class_id,
                    'bbox': [xtl, ytl, xbr, ybr]
                })

    return annotations

def parse_tracking_predictions(label_dir, dataset_name, start_frame, img_width, img_height):
    """Parse tracking predictions from YOLO format"""
    predictions = defaultdict(list)

    for label_file in sorted(Path(label_dir).glob(f'{dataset_name}_frame_*.txt')):
        frame_str = label_file.stem.replace(f'{dataset_name}_frame_', '')
        frame_num = int(frame_str)

        with open(label_file) as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 6:
                    class_id = int(parts[0])
                    x_center = float(parts[1])
                    y_center = float(parts[2])
                    width = float(parts[3])
                    height = float(parts[4])
                    track_id = int(parts[5])

                    # Convert to pixel coordinates
                    xtl = (x_center - width/2) * img_width
                    ytl = (y_center - height/2) * img_height
                    xbr = (x_center + width/2) * img_width
                    ybr = (y_center + height/2) * img_height

                    predictions[frame_num].append({
                        'track_id': track_id,
                        'class_id': class_id,
                        'bbox': [xtl, ytl, xbr, ybr]
                    })

    return predictions

def draw_boxes(img, annotations, mode='gt'):
    """Draw bounding boxes on image"""
    img_vis = img.copy()

    for obj in annotations:
        class_id = obj['class_id']
        track_id = obj['track_id']
        bbox = obj['bbox']

        color = CLASS_COLORS[class_id]
        xtl, ytl, xbr, ybr = [int(x) for x in bbox]

        # Draw bounding box
        cv2.rectangle(img_vis, (xtl, ytl), (xbr, ybr), color, 2)

        # Draw label
        label = f"{CLASS_NAMES[class_id]} ID:{track_id}"
        font_scale = 0.5
        thickness = 1

        # Background for text
        (text_width, text_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX,
                                                        font_scale, thickness)
        cv2.rectangle(img_vis, (xtl, ytl - text_height - 5),
                     (xtl + text_width, ytl), color, -1)

        # Text
        cv2.putText(img_vis, label, (xtl, ytl - 5),
                   cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), thickness)

    # Add mode label
    mode_label = "Ground Truth" if mode == 'gt' else "Prediction"
    cv2.putText(img_vis, mode_label, (10, 30),
               cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)

    return img_vis

def create_side_by_side_comparison(dataset_config):
    """Create side-by-side GT vs Prediction comparison video"""
    dataset_name, start_frame, end_frame, xml_path, img_dir, label_dir, img_width, img_height = dataset_config

    print(f"\n{'='*80}")
    print(f"Creating comparison video for {dataset_name}")
    print(f"{'='*80}")

    # Parse annotations
    gt_annotations = parse_xml_annotations(xml_path, start_frame, end_frame)
    pred_annotations = parse_tracking_predictions(label_dir, dataset_name, start_frame, img_width, img_height)

    # Setup video writer
    output_path = f'/cluster/work/tmstorma/Football2025/tracking/visualizations/{dataset_name}_comparison.mp4'
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, 25.0, (img_width * 2, img_height))

    frames_written = 0
    for frame_num in range(start_frame, end_frame + 1):
        img_path = Path(img_dir) / f'frame_{frame_num:06d}.png'

        if not img_path.exists():
            continue

        img = cv2.imread(str(img_path))
        if img is None:
            continue

        # Draw GT and predictions
        gt_img = draw_boxes(img, gt_annotations.get(frame_num, []), mode='gt')
        pred_img = draw_boxes(img, pred_annotations.get(frame_num, []), mode='pred')

        # Combine side by side
        combined = np.hstack([gt_img, pred_img])

        out.write(combined)
        frames_written += 1

        if frames_written % 50 == 0:
            print(f"  Processed {frames_written} frames...")

    out.release()
    print(f"  Saved comparison video: {output_path}")
    print(f"  Total frames: {frames_written}")

    return output_path

def create_trajectory_visualization(dataset_config, max_frames=150):
    """Create trajectory visualization showing movement paths"""
    dataset_name, start_frame, end_frame, xml_path, img_dir, label_dir, img_width, img_height = dataset_config

    print(f"\n{'='*80}")
    print(f"Creating trajectory visualization for {dataset_name}")
    print(f"{'='*80}")

    # Parse predictions
    pred_annotations = parse_tracking_predictions(label_dir, dataset_name, start_frame, img_width, img_height)

    # Collect trajectories
    trajectories = defaultdict(list)
    for frame_num in sorted(pred_annotations.keys())[:max_frames]:
        for obj in pred_annotations[frame_num]:
            track_id = obj['track_id']
            class_id = obj['class_id']
            bbox = obj['bbox']
            center_x = (bbox[0] + bbox[2]) / 2
            center_y = (bbox[1] + bbox[3]) / 2
            trajectories[track_id].append({
                'frame': frame_num,
                'class_id': class_id,
                'center': (int(center_x), int(center_y))
            })

    # Create base image
    first_frame = start_frame
    img_path = Path(img_dir) / f'frame_{first_frame:06d}.png'
    base_img = cv2.imread(str(img_path))

    # Draw trajectories
    overlay = base_img.copy()

    for track_id, points in trajectories.items():
        if len(points) < 5:  # Skip short tracks
            continue

        class_id = points[0]['class_id']
        color = CLASS_COLORS[class_id]

        # Draw trajectory line
        for i in range(len(points) - 1):
            pt1 = points[i]['center']
            pt2 = points[i + 1]['center']
            cv2.line(overlay, pt1, pt2, color, 2)

        # Draw current position
        current_pos = points[-1]['center']
        cv2.circle(overlay, current_pos, 5, color, -1)

        # Draw ID label
        cv2.putText(overlay, f"ID:{track_id}",
                   (current_pos[0] + 10, current_pos[1]),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

    # Add title
    cv2.putText(overlay, f"Trajectories - {dataset_name} (First {max_frames} frames)",
               (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)

    # Save
    output_path = f'/cluster/work/tmstorma/Football2025/tracking/visualizations/{dataset_name}_trajectories.png'
    cv2.imwrite(output_path, overlay)
    print(f"  Saved trajectory visualization: {output_path}")

    return output_path

def create_highlights_video(dataset_configs, output_path, frames_per_dataset=50):
    """Create highlights video from all datasets"""
    print(f"\n{'='*80}")
    print(f"Creating highlights video")
    print(f"{'='*80}")

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = None
    total_frames = 0

    for dataset_config in dataset_configs:
        dataset_name, start_frame, end_frame, xml_path, img_dir, label_dir, img_width, img_height = dataset_config

        print(f"\n  Processing {dataset_name}...")

        # Parse predictions
        pred_annotations = parse_tracking_predictions(label_dir, dataset_name, start_frame, img_width, img_height)

        # Sample frames evenly
        all_frames = sorted(pred_annotations.keys())
        step = max(1, len(all_frames) // frames_per_dataset)
        sampled_frames = all_frames[::step][:frames_per_dataset]

        for frame_num in sampled_frames:
            img_path = Path(img_dir) / f'frame_{frame_num:06d}.png'

            if not img_path.exists():
                continue

            img = cv2.imread(str(img_path))
            if img is None:
                continue

            # Initialize writer with first frame dimensions
            if out is None:
                out = cv2.VideoWriter(output_path, fourcc, 25.0, (img_width, img_height))

            # Draw predictions
            vis_img = draw_boxes(img, pred_annotations.get(frame_num, []), mode='pred')

            # Add dataset label
            cv2.putText(vis_img, dataset_name, (10, img_height - 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

            out.write(vis_img)
            total_frames += 1

    if out is not None:
        out.release()

    print(f"\n  Saved highlights video: {output_path}")
    print(f"  Total frames: {total_frames}")

    return output_path

def main():
    # Dataset configurations
    datasets = [
        ('RBK-AALESUND', 1622, 1801,
         '/cluster/projects/vc/courses/TDT17/other/Football2025/RBK-AALESUND/annotations.xml',
         '/cluster/projects/vc/courses/TDT17/other/Football2025/RBK-AALESUND/data/images/train',
         '/cluster/work/tmstorma/Football2025/tracking/runs/val_tracking/RBK-AALESUND/labels',
         1920, 1080),
        ('RBK-FREDRIKSTAD', 1635, 1815,
         '/cluster/projects/vc/courses/TDT17/other/Football2025/RBK-FREDRIKSTAD/annotations.xml',
         '/cluster/projects/vc/courses/TDT17/other/Football2025/RBK-FREDRIKSTAD/data/images/train',
         '/cluster/work/tmstorma/Football2025/tracking/runs/val_tracking/RBK-FREDRIKSTAD/labels',
         1280, 720),
        ('RBK-HamKam', 1371, 1522,
         '/cluster/projects/vc/courses/TDT17/other/Football2025/RBK-HamKam/annotations.xml',
         '/cluster/projects/vc/courses/TDT17/other/Football2025/RBK-HamKam/data/images/train',
         '/cluster/work/tmstorma/Football2025/tracking/runs/val_tracking/RBK-HamKam/labels',
         1920, 1080)
    ]

    print("="*80)
    print("Tracking Visualization Generator")
    print("="*80)

    # Create output directory
    Path('/cluster/work/tmstorma/Football2025/tracking/visualizations').mkdir(parents=True, exist_ok=True)

    # 1. Create side-by-side comparison videos
    print("\n[1/3] Creating side-by-side comparison videos...")
    for dataset_config in datasets:
        create_side_by_side_comparison(dataset_config)

    # 2. Create trajectory visualizations
    print("\n[2/3] Creating trajectory visualizations...")
    for dataset_config in datasets:
        create_trajectory_visualization(dataset_config, max_frames=150)

    # 3. Create highlights video
    print("\n[3/3] Creating highlights video...")
    highlights_path = '/cluster/work/tmstorma/Football2025/tracking/visualizations/validation_highlights.mp4'
    create_highlights_video(datasets, highlights_path, frames_per_dataset=50)

    print("\n" + "="*80)
    print("Visualization generation complete!")
    print("="*80)
    print("\nOutputs:")
    print("  - visualizations/RBK-AALESUND_comparison.mp4")
    print("  - visualizations/RBK-FREDRIKSTAD_comparison.mp4")
    print("  - visualizations/RBK-HamKam_comparison.mp4")
    print("  - visualizations/RBK-AALESUND_trajectories.png")
    print("  - visualizations/RBK-FREDRIKSTAD_trajectories.png")
    print("  - visualizations/RBK-HamKam_trajectories.png")
    print("  - visualizations/validation_highlights.mp4")

if __name__ == '__main__':
    main()
