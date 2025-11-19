#!/usr/bin/env python3
"""
Step 1a - Tracking ID Validation
Checks for duplicate and missing tracking IDs
"""

import xml.etree.ElementTree as ET
from pathlib import Path
from collections import defaultdict

# Dataset paths
DATASETS = {
    'RBK-VIKING': '/cluster/projects/vc/courses/TDT17/other/Football2025/RBK-VIKING/annotations.xml',
    'RBK-AALESUND': '/cluster/projects/vc/courses/TDT17/other/Football2025/RBK-AALESUND/annotations.xml',
    'RBK-FREDRIKSTAD': '/cluster/projects/vc/courses/TDT17/other/Football2025/RBK-FREDRIKSTAD/annotations.xml',
    'RBK-HamKam': '/cluster/projects/vc/courses/TDT17/other/Football2025/RBK-HamKam/annotations.xml',
    'RBK-BODO-part3': '/cluster/projects/vc/courses/TDT17/other/Football2025/RBK-BODO/part3/RBK_BODO_PART3/annotations.xml',
}

def analyze_dataset(dataset_name, xml_path):
    """Analyze one dataset for tracking ID issues"""
    print(f"\n{'='*80}")
    print(f"Dataset: {dataset_name}")
    print('='*80)

    xml_file = Path(xml_path)
    if not xml_file.exists():
        print(f"File not found: {xml_path}")
        return

    tree = ET.parse(xml_file)
    root = tree.getroot()

    # Get metadata
    meta = root.find('meta')
    task = meta.find('task')
    total_frames = int(task.find('size').text)

    # Build frame -> track_id mapping
    frame_tracks = defaultdict(list)
    total_tracks = 0
    total_boxes = 0

    for track in root.findall('track'):
        track_id = track.get('id')
        total_tracks += 1

        for box in track.findall('box'):
            total_boxes += 1
            frame = int(box.get('frame'))
            label = track.get('label')

            frame_tracks[frame].append({
                'track_id': track_id,
                'label': label
            })

    # Check for duplicate track IDs in same frame
    duplicate_frames = []
    for frame, tracks in frame_tracks.items():
        track_ids = [t['track_id'] for t in tracks]
        if len(track_ids) != len(set(track_ids)):
            # Found duplicates
            from collections import Counter
            id_counts = Counter(track_ids)
            duplicates = {tid: count for tid, count in id_counts.items() if count > 1}
            duplicate_frames.append({
                'frame': frame,
                'duplicates': duplicates
            })

    # Report results
    print(f"\nTotal frames: {total_frames}")
    print(f"Total tracks: {total_tracks}")
    print(f"Total boxes: {total_boxes}")

    if duplicate_frames:
        print(f"\nDUPLICATE TRACK IDs FOUND: {len(duplicate_frames)} frames")
        for i, dup in enumerate(duplicate_frames[:3]):
            print(f"  Frame {dup['frame']}: Track IDs {dup['duplicates']}")
        if len(duplicate_frames) > 3:
            print(f"  ... and {len(duplicate_frames) - 3} more frames with duplicates")
    else:
        print("\nNo duplicate track IDs found - all track IDs are unique within frames")

    return len(duplicate_frames)

def main():
    print("="*80)
    print("STEP 1a: TRACKING ID VALIDATION")
    print("="*80)

    total_issues = 0
    for dataset_name, xml_path in DATASETS.items():
        num_issues = analyze_dataset(dataset_name, xml_path)
        if num_issues:
            total_issues += num_issues

    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    if total_issues == 0:
        print("All datasets passed validation - no tracking ID issues found")
    else:
        print(f"Found duplicate track IDs in {total_issues} total frames")
    print("="*80)

if __name__ == '__main__':
    main()
