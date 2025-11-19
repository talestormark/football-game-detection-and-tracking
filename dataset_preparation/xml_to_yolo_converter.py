import xml.etree.ElementTree as ET
import json
import os
from pathlib import Path
from collections import defaultdict

class XMLToYOLOConverter:
    """
    Convert CVAT XML annotations to YOLO format with 4 classes:
    0: home, 1: away, 2: referee, 3: ball
    """

    def __init__(self):
        self.class_map = {
            ('player', 'home'): 0,
            ('player', 'away'): 1,
            ('player', 'referee'): 2,
            ('ball', None): 3
        }

    def parse_xml(self, xml_path):
        """
        Parse XML annotation file.
        Returns: dict mapping frame_id to list of boxes, and image dimensions
        """
        tree = ET.parse(xml_path)
        root = tree.getroot()

        # Get image dimensions from XML
        size_elem = root.find('.//original_size')
        if size_elem is not None:
            xml_width = int(size_elem.find('width').text)
            xml_height = int(size_elem.find('height').text)
        else:
            # Fallback to default
            xml_width = 1920
            xml_height = 1080

        # Frame to boxes mapping
        frame_annotations = defaultdict(list)

        # Ground truth tracking for HOTA evaluation
        gt_tracking = defaultdict(list)

        # Parse all tracks
        for track in root.findall('track'):
            track_id = int(track.get('id'))
            label = track.get('label')

            # Parse all boxes in this track
            for box in track.findall('box'):
                frame_id = int(box.get('frame'))
                outside = int(box.get('outside', '0'))

                # Skip boxes marked as outside
                if outside == 1:
                    continue

                # Get bounding box coordinates
                xtl = float(box.get('xtl'))
                ytl = float(box.get('ytl'))
                xbr = float(box.get('xbr'))
                ybr = float(box.get('ybr'))

                # Get team attribute for players
                team = None
                if label == 'player':
                    team_attr = box.find("attribute[@name='team']")
                    if team_attr is not None:
                        team = team_attr.text

                # Skip event_labels
                if label == 'event_labels':
                    continue

                # Determine class
                class_key = (label, team)
                if class_key not in self.class_map:
                    continue

                class_id = self.class_map[class_key]

                # Store annotation
                frame_annotations[frame_id].append({
                    'track_id': track_id,
                    'class_id': class_id,
                    'xtl': xtl,
                    'ytl': ytl,
                    'xbr': xbr,
                    'ybr': ybr
                })

                # Store ground truth tracking
                gt_tracking[frame_id].append({
                    'track_id': track_id,
                    'class_id': class_id,
                    'bbox': [xtl, ytl, xbr, ybr]
                })

        return frame_annotations, gt_tracking, xml_width, xml_height

    def bbox_to_yolo(self, xtl, ytl, xbr, ybr, img_width, img_height):
        """
        Convert bbox from (xtl, ytl, xbr, ybr) to YOLO format:
        (x_center, y_center, width, height) normalized to [0, 1]

        Args:
            xtl, ytl, xbr, ybr: Bounding box coordinates in pixels
            img_width, img_height: Image dimensions for normalization
        """
        x_center = (xtl + xbr) / (2 * img_width)
        y_center = (ytl + ybr) / (2 * img_height)
        width = (xbr - xtl) / img_width
        height = (ybr - ytl) / img_height

        # Clip to [0, 1] range
        x_center = max(0.0, min(1.0, x_center))
        y_center = max(0.0, min(1.0, y_center))
        width = max(0.0, min(1.0, width))
        height = max(0.0, min(1.0, height))

        return x_center, y_center, width, height

    def convert_dataset(self, dataset_name, xml_path, output_dir,
                       train_frames, val_frames):
        """
        Convert a dataset's XML annotations to YOLO format.

        Args:
            dataset_name: Name of dataset (e.g., 'RBK-AALESUND')
            xml_path: Path to annotations.xml
            output_dir: Base output directory
            train_frames: List of frame indices for training
            val_frames: List of frame indices for validation
        """
        print(f"\nProcessing {dataset_name}...")

        # Parse XML and get image dimensions
        frame_annotations, gt_tracking, img_width, img_height = self.parse_xml(xml_path)
        print(f"  Image dimensions from XML: {img_width}x{img_height}")

        # Create output directories
        train_labels_dir = Path(output_dir) / 'labels' / 'train'
        val_labels_dir = Path(output_dir) / 'labels' / 'val'
        train_labels_dir.mkdir(parents=True, exist_ok=True)
        val_labels_dir.mkdir(parents=True, exist_ok=True)

        # Statistics
        stats = {
            'train': {'total': 0, 'classes': defaultdict(int)},
            'val': {'total': 0, 'classes': defaultdict(int)}
        }

        # Convert training frames
        for frame_id in train_frames:
            if frame_id not in frame_annotations:
                # Empty frame (no annotations)
                label_file = train_labels_dir / f'{dataset_name}_frame_{frame_id+1:06d}.txt'
                label_file.write_text('')
                continue

            boxes = frame_annotations[frame_id]
            label_file = train_labels_dir / f'{dataset_name}_frame_{frame_id+1:06d}.txt'

            with open(label_file, 'w') as f:
                for box in boxes:
                    class_id = box['class_id']
                    x_c, y_c, w, h = self.bbox_to_yolo(
                        box['xtl'], box['ytl'], box['xbr'], box['ybr'],
                        img_width, img_height
                    )
                    f.write(f"{class_id} {x_c:.6f} {y_c:.6f} {w:.6f} {h:.6f}\n")

                    stats['train']['total'] += 1
                    stats['train']['classes'][class_id] += 1

        # Convert validation frames
        for frame_id in val_frames:
            if frame_id not in frame_annotations:
                # Empty frame
                label_file = val_labels_dir / f'{dataset_name}_frame_{frame_id+1:06d}.txt'
                label_file.write_text('')
                continue

            boxes = frame_annotations[frame_id]
            label_file = val_labels_dir / f'{dataset_name}_frame_{frame_id+1:06d}.txt'

            with open(label_file, 'w') as f:
                for box in boxes:
                    class_id = box['class_id']
                    x_c, y_c, w, h = self.bbox_to_yolo(
                        box['xtl'], box['ytl'], box['xbr'], box['ybr'],
                        img_width, img_height
                    )
                    f.write(f"{class_id} {x_c:.6f} {y_c:.6f} {w:.6f} {h:.6f}\n")

                    stats['val']['total'] += 1
                    stats['val']['classes'][class_id] += 1

        # Print statistics
        print(f"  Train: {len(train_frames)} frames, {stats['train']['total']} boxes")
        print(f"    home={stats['train']['classes'][0]}, away={stats['train']['classes'][1]}, "
              f"referee={stats['train']['classes'][2]}, ball={stats['train']['classes'][3]}")
        print(f"  Val: {len(val_frames)} frames, {stats['val']['total']} boxes")
        print(f"    home={stats['val']['classes'][0]}, away={stats['val']['classes'][1]}, "
              f"referee={stats['val']['classes'][2]}, ball={stats['val']['classes'][3]}")

        # Extract ground truth tracking for validation set only
        val_gt_tracking = {
            str(frame_id): gt_tracking[frame_id]
            for frame_id in val_frames
            if frame_id in gt_tracking
        }

        return stats, val_gt_tracking


def main():
    """
    Main conversion pipeline.
    """
    # Base directories
    source_dir = Path('/cluster/projects/vc/courses/TDT17/other/Football2025')
    output_dir = Path('/cluster/work/tmstorma/Football2025/dataset')

    # Dataset configurations
    datasets = {
        'RBK-AALESUND': {
            'xml_path': source_dir / 'RBK-AALESUND' / 'annotations.xml',
            'total_frames': 1802,
            'train_end': 1621,  # frames 0-1621 (1622 frames)
            'val_start': 1622,  # frames 1622-1801 (180 frames)
        },
        'RBK-FREDRIKSTAD': {
            'xml_path': source_dir / 'RBK-FREDRIKSTAD' / 'annotations.xml',
            'total_frames': 1816,
            'train_end': 1634,  # frames 0-1634 (1635 frames)
            'val_start': 1635,  # frames 1635-1815 (181 frames)
        },
        'RBK-HamKam': {
            'xml_path': source_dir / 'RBK-HamKam' / 'annotations.xml',
            'total_frames': 1523,
            'train_end': 1370,  # frames 0-1370 (1371 frames)
            'val_start': 1371,  # frames 1371-1522 (152 frames)
        }
    }

    # Initialize converter
    converter = XMLToYOLOConverter()

    # Global statistics
    all_stats = {'train': defaultdict(int), 'val': defaultdict(int)}
    all_gt_tracking = {}

    # Process each dataset
    for dataset_name, config in datasets.items():
        train_frames = list(range(0, config['train_end'] + 1))
        val_frames = list(range(config['val_start'], config['total_frames']))

        stats, gt_tracking = converter.convert_dataset(
            dataset_name=dataset_name,
            xml_path=config['xml_path'],
            output_dir=output_dir,
            train_frames=train_frames,
            val_frames=val_frames
        )

        # Accumulate statistics
        all_stats['train']['frames'] += len(train_frames)
        all_stats['train']['boxes'] += stats['train']['total']
        for class_id, count in stats['train']['classes'].items():
            all_stats['train'][f'class_{class_id}'] += count

        all_stats['val']['frames'] += len(val_frames)
        all_stats['val']['boxes'] += stats['val']['total']
        for class_id, count in stats['val']['classes'].items():
            all_stats['val'][f'class_{class_id}'] += count

        # Store ground truth tracking
        all_gt_tracking[dataset_name] = gt_tracking

    # Save ground truth tracking
    gt_tracking_path = output_dir / 'gt_tracking.json'
    with open(gt_tracking_path, 'w') as f:
        json.dump(all_gt_tracking, f, indent=2)
    print(f"\nSaved ground truth tracking to {gt_tracking_path}")

    # Print overall statistics
    print("\n" + "="*60)
    print("OVERALL STATISTICS")
    print("="*60)
    print(f"Training: {all_stats['train']['frames']} frames, {all_stats['train']['boxes']} boxes")
    print(f"  home={all_stats['train']['class_0']}, away={all_stats['train']['class_1']}, "
          f"referee={all_stats['train']['class_2']}, ball={all_stats['train']['class_3']}")
    print(f"Validation: {all_stats['val']['frames']} frames, {all_stats['val']['boxes']} boxes")
    print(f"  home={all_stats['val']['class_0']}, away={all_stats['val']['class_1']}, "
          f"referee={all_stats['val']['class_2']}, ball={all_stats['val']['class_3']}")

    print("\nConversion complete!")


if __name__ == '__main__':
    main()
