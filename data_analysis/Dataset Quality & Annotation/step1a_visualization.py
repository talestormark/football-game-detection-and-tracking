#!/usr/bin/env python3
"""
Step 1a - Visualize Random Annotated Frames
Draws bounding boxes with team colors and track IDs
"""

import xml.etree.ElementTree as ET
from pathlib import Path
import cv2
import random
from collections import defaultdict

# Dataset paths
DATASETS = {
    'RBK-VIKING': {
        'xml': '/cluster/projects/vc/courses/TDT17/other/Football2025/RBK-VIKING/annotations.xml',
        'images': '/cluster/projects/vc/courses/TDT17/other/Football2025/RBK-VIKING/data/images/train'
    },
    'RBK-AALESUND': {
        'xml': '/cluster/projects/vc/courses/TDT17/other/Football2025/RBK-AALESUND/annotations.xml',
        'images': '/cluster/projects/vc/courses/TDT17/other/Football2025/RBK-AALESUND/data/images/train'
    },
    'RBK-FREDRIKSTAD': {
        'xml': '/cluster/projects/vc/courses/TDT17/other/Football2025/RBK-FREDRIKSTAD/annotations.xml',
        'images': '/cluster/projects/vc/courses/TDT17/other/Football2025/RBK-FREDRIKSTAD/data/images/train'
    },
    'RBK-HamKam': {
        'xml': '/cluster/projects/vc/courses/TDT17/other/Football2025/RBK-HamKam/annotations.xml',
        'images': '/cluster/projects/vc/courses/TDT17/other/Football2025/RBK-HamKam/data/images/train'
    },
    'RBK-BODO-part3': {
        'xml': '/cluster/projects/vc/courses/TDT17/other/Football2025/RBK-BODO/part3/RBK_BODO_PART3/annotations.xml',
        'images': '/cluster/projects/vc/courses/TDT17/other/Football2025/RBK-BODO/part3/RBK_BODO_PART3/data/images/train'
    },
}

# Color coding (BGR format for OpenCV)
COLORS = {
    'home': (255, 0, 0),      # Blue
    'away': (0, 0, 255),      # Red
    'referee': (0, 255, 255), # Yellow
    'ball': (0, 255, 0),      # Green
}

OUTPUT_DIR = Path('/cluster/work/tmstorma/Football2025/visualizations')

def get_frame_annotations(xml_path, frame_num):
    """Extract all annotations for a specific frame"""
    tree = ET.parse(xml_path)
    root = tree.getroot()

    annotations = []
    for track in root.findall('track'):
        track_id = track.get('id')
        label = track.get('label')

        for box in track.findall('box'):
            if int(box.get('frame')) == frame_num:
                xtl = int(float(box.get('xtl')))
                ytl = int(float(box.get('ytl')))
                xbr = int(float(box.get('xbr')))
                ybr = int(float(box.get('ybr')))

                # Get team attribute for players
                team = 'ball'
                if label == 'player':
                    team_attr = box.find("attribute[@name='team']")
                    if team_attr is not None:
                        team = team_attr.text

                annotations.append({
                    'track_id': track_id,
                    'label': label,
                    'team': team,
                    'bbox': (xtl, ytl, xbr, ybr)
                })

    return annotations

def visualize_frame(dataset_name, xml_path, images_dir, frame_num, output_path):
    """Visualize annotations for one frame"""
    # Load image
    img_path = Path(images_dir) / f'frame_{frame_num:06d}.png'
    if not img_path.exists():
        print(f"  Image not found: {img_path}")
        return False

    img = cv2.imread(str(img_path))
    if img is None:
        print(f"  Failed to load: {img_path}")
        return False

    # Get annotations
    annotations = get_frame_annotations(xml_path, frame_num)

    # Draw each bounding box
    for ann in annotations:
        xtl, ytl, xbr, ybr = ann['bbox']
        color = COLORS.get(ann['team'], (255, 255, 255))

        # Draw rectangle
        cv2.rectangle(img, (xtl, ytl), (xbr, ybr), color, 2)

        # Draw label with track ID
        label_text = f"ID:{ann['track_id']} {ann['team']}"
        cv2.putText(img, label_text, (xtl, ytl - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # Add title
    title = f"{dataset_name} - Frame {frame_num}"
    cv2.putText(img, title, (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    # Save
    cv2.imwrite(str(output_path), img)
    return True

def analyze_dataset(dataset_name, dataset_info):
    """Visualize random frames from one dataset"""
    print(f"\n{'='*80}")
    print(f"Dataset: {dataset_name}")
    print('='*80)

    xml_path = dataset_info['xml']
    images_dir = dataset_info['images']

    xml_file = Path(xml_path)
    if not xml_file.exists():
        print(f"XML not found: {xml_path}")
        return

    # Get total frames
    tree = ET.parse(xml_file)
    root = tree.getroot()
    meta = root.find('meta')
    task = meta.find('task')
    total_frames = int(task.find('size').text)

    # Pick 3 random frames
    num_samples = min(3, total_frames)
    random_frames = sorted(random.sample(range(total_frames), num_samples))

    print(f"Total frames: {total_frames}")
    print(f"Visualizing frames: {random_frames}")

    # Create output directory
    dataset_output = OUTPUT_DIR / dataset_name
    dataset_output.mkdir(parents=True, exist_ok=True)

    # Visualize each frame
    for frame_num in random_frames:
        output_path = dataset_output / f'frame_{frame_num:06d}.png'
        success = visualize_frame(dataset_name, xml_path, images_dir,
                                  frame_num, output_path)
        if success:
            print(f"  Saved: {output_path}")

def main():
    print("="*80)
    print("STEP 1a: VISUALIZATION OF RANDOM FRAMES")
    print("="*80)

    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Set random seed for reproducibility
    random.seed(42)

    for dataset_name, dataset_info in DATASETS.items():
        analyze_dataset(dataset_name, dataset_info)

    print("\n" + "="*80)
    print("VISUALIZATION COMPLETE")
    print(f"Output directory: {OUTPUT_DIR}")
    print("="*80)

if __name__ == '__main__':
    main()
