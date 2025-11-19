#!/usr/bin/env python3
"""
Prepare data for HOTA evaluation by converting:
1. Ground truth XML annotations -> MOT format
2. Tracking predictions -> MOT format

MOT format: <frame>, <id>, <bb_left>, <bb_top>, <bb_width>, <bb_height>, <conf>, <class>, <visibility>
"""

import xml.etree.ElementTree as ET
from pathlib import Path
import json
from collections import defaultdict

def parse_xml_ground_truth(xml_path, dataset_name):
    """
    Parse XML annotations and extract ground truth tracks
    Returns: dict of frame_idx -> list of tracks
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()

    # Ground truth tracks per frame
    gt_tracks = defaultdict(list)

    # Class mapping
    class_map = {
        'home': 0,
        'away': 1,
        'referee': 2,
        'ball': 3
    }

    for track in root.findall('.//track'):
        label = track.get('label')

        # Determine class
        if label == 'ball':
            class_id = 3
        else:
            # Player or referee - check team attribute
            team = None
            for attr in track.findall('.//attribute'):
                if attr.get('name') == 'team':
                    team = attr.text.strip().lower()
                    break

            if team in class_map:
                class_id = class_map[team]
            else:
                # Skip tracks without valid team
                continue

        track_id = int(track.get('id'))

        # Process all boxes in this track
        for box in track.findall('.//box'):
            frame_xml = int(box.get('frame'))  # XML frame (0-indexed)
            frame_mot = frame_xml + 1  # Convert to MOT format (1-indexed)

            xtl = float(box.get('xtl'))
            ytl = float(box.get('ytl'))
            xbr = float(box.get('xbr'))
            ybr = float(box.get('ybr'))

            # Convert to MOT format (top-left width height)
            bb_left = xtl
            bb_top = ytl
            bb_width = xbr - xtl
            bb_height = ybr - ytl

            gt_tracks[frame_mot].append({
                'frame': frame_mot,
                'track_id': track_id,
                'bb_left': bb_left,
                'bb_top': bb_top,
                'bb_width': bb_width,
                'bb_height': bb_height,
                'class_id': class_id,
                'conf': 1.0,  # Ground truth has confidence 1.0
                'visibility': 1.0
            })

    return gt_tracks

def parse_tracking_predictions(label_dir, dataset_name, frame_offset, img_width=1920, img_height=1080):
    """
    Parse tracking predictions from YOLO format
    Returns: dict of frame_idx -> list of tracks
    """
    pred_tracks = defaultdict(list)

    # Image dimensions for denormalization (passed as parameters)

    label_files = sorted(label_dir.glob(f"{dataset_name}_frame_*.txt"))

    for label_file in label_files:
        # Extract frame number from filename (e.g., RBK-AALESUND_frame_001623.txt)
        frame_str = label_file.stem.split('_')[-1]
        frame_idx = int(frame_str)

        # Frame number in filename already accounts for off-by-one with XML
        # (image frame_N.png corresponds to XML frame N-1)
        # For MOT format, we use XML frame + 1
        # So: filename_frame = XML_frame + 1, MOT_frame = XML_frame + 1
        # Therefore: MOT_frame = filename_frame
        frame_mot = frame_idx

        # Read tracking predictions
        with open(label_file, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 6:
                    # Skip detections without track IDs (first frame issue)
                    continue

                class_id = int(parts[0])
                x_center_norm = float(parts[1])
                y_center_norm = float(parts[2])
                width_norm = float(parts[3])
                height_norm = float(parts[4])
                track_id = int(parts[5])

                # Denormalize to pixel coordinates
                x_center = x_center_norm * img_width
                y_center = y_center_norm * img_height
                width = width_norm * img_width
                height = height_norm * img_height

                # Convert to MOT format (top-left)
                bb_left = x_center - width / 2
                bb_top = y_center - height / 2

                pred_tracks[frame_mot].append({
                    'frame': frame_mot,
                    'track_id': track_id,
                    'bb_left': bb_left,
                    'bb_top': bb_top,
                    'bb_width': width,
                    'bb_height': height,
                    'class_id': class_id,
                    'conf': 1.0,  # Predictions already filtered by confidence
                    'visibility': 1.0
                })

    return pred_tracks

def write_mot_format(tracks_dict, output_path):
    """
    Write tracks to MOT format file
    Format: <frame>, <id>, <bb_left>, <bb_top>, <bb_width>, <bb_height>, <conf>, <class>, <visibility>
    """
    with open(output_path, 'w') as f:
        for frame_idx in sorted(tracks_dict.keys()):
            for track in tracks_dict[frame_idx]:
                # frame_idx is already in MOT format (1-indexed)
                frame_mot = frame_idx
                f.write(f"{frame_mot},{track['track_id']},"
                       f"{track['bb_left']:.2f},{track['bb_top']:.2f},"
                       f"{track['bb_width']:.2f},{track['bb_height']:.2f},"
                       f"{track['conf']:.2f},{track['class_id']},{track['visibility']:.2f}\n")

def main():
    print("="*80)
    print("Preparing Data for HOTA Evaluation")
    print("="*80)

    # Paths
    xml_base = Path('/cluster/projects/vc/courses/TDT17/other/Football2025')
    tracking_base = Path('/cluster/work/tmstorma/Football2025/tracking/runs/val_tracking')
    output_base = Path('/cluster/work/tmstorma/Football2025/tracking/hota_data')

    # Datasets with their validation frame ranges and resolutions
    datasets = [
        ('RBK-AALESUND', 1622, 1802, 1920, 1080),     # frames 1622-1801 (180 frames, 0-indexed in XML)
        ('RBK-FREDRIKSTAD', 1635, 1816, 1280, 720),   # frames 1635-1815 (181 frames, 1280x720 resolution)
        ('RBK-HamKam', 1371, 1523, 1920, 1080)        # frames 1371-1522 (152 frames)
    ]

    for dataset_name, start_frame, end_frame, img_width, img_height in datasets:
        print(f"\n{'='*80}")
        print(f"Processing {dataset_name}")
        print(f"{'='*80}")
        print(f"  Resolution: {img_width}x{img_height}")

        # Paths for this dataset
        xml_path = xml_base / dataset_name / 'annotations.xml'
        label_dir = tracking_base / dataset_name / 'labels'

        # Output directories
        gt_dir = output_base / 'gt' / dataset_name
        pred_dir = output_base / 'trackers' / 'ByteTrack' / dataset_name

        gt_dir.mkdir(parents=True, exist_ok=True)
        pred_dir.mkdir(parents=True, exist_ok=True)

        print(f"  XML: {xml_path}")
        print(f"  Predictions: {label_dir}")

        # Parse ground truth
        print(f"  Parsing ground truth...")
        gt_tracks = parse_xml_ground_truth(xml_path, dataset_name)

        # Filter to validation frames only
        # start_frame and end_frame are XML frames (0-indexed)
        # gt_tracks uses MOT frames (1-indexed), so we need to adjust the range
        start_frame_mot = start_frame + 1
        end_frame_mot = end_frame + 1
        gt_tracks_filtered = {
            frame: tracks for frame, tracks in gt_tracks.items()
            if start_frame_mot <= frame < end_frame_mot
        }

        print(f"    Found {len(gt_tracks_filtered)} frames with annotations")

        # Parse predictions
        print(f"  Parsing predictions...")
        pred_tracks = parse_tracking_predictions(label_dir, dataset_name, start_frame, img_width, img_height)

        print(f"    Found {len(pred_tracks)} frames with predictions")

        # Write MOT format files
        gt_file = gt_dir / 'gt.txt'
        pred_file = pred_dir / 'data.txt'

        print(f"  Writing ground truth to: {gt_file}")
        write_mot_format(gt_tracks_filtered, gt_file)

        print(f"  Writing predictions to: {pred_file}")
        write_mot_format(pred_tracks, pred_file)

        # Write seqinfo
        seqinfo_path = gt_dir.parent / 'seqinfo.ini'
        with open(seqinfo_path, 'w') as f:
            f.write(f"[Sequence]\n")
            f.write(f"name={dataset_name}\n")
            f.write(f"imDir=images\n")
            f.write(f"frameRate=25\n")
            f.write(f"seqLength={end_frame - start_frame}\n")
            f.write(f"imWidth={img_width}\n")
            f.write(f"imHeight={img_height}\n")
            f.write(f"imExt=.png\n")

        print(f"  Completed {dataset_name}")

    print("\n" + "="*80)
    print("Data Preparation Complete!")
    print("="*80)
    print(f"\nOutput structure:")
    print(f"  {output_base}/")
    print(f"    gt/")
    print(f"      RBK-AALESUND/gt.txt")
    print(f"      RBK-FREDRIKSTAD/gt.txt")
    print(f"      RBK-HamKam/gt.txt")
    print(f"    trackers/")
    print(f"      ByteTrack/")
    print(f"        RBK-AALESUND/data.txt")
    print(f"        RBK-FREDRIKSTAD/data.txt")
    print(f"        RBK-HamKam/data.txt")
    print(f"\nNext: Install TrackEval and run HOTA evaluation")

if __name__ == '__main__':
    main()
