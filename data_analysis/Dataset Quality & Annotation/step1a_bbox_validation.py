#!/usr/bin/env python3
"""
Step 1a - Bounding Box Boundary Validation
Checks all bounding boxes for coordinate and boundary issues
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
    'RBK-BODO-part1': '/cluster/projects/vc/courses/TDT17/other/Football2025/RBK-BODO/part1/RBK_BODO_PART1/annotations.xml',
    'RBK-BODO-part2': '/cluster/projects/vc/courses/TDT17/other/Football2025/RBK-BODO/part2/RBK_BODO_PART2/annotations.xml',
    'RBK-BODO-part3': '/cluster/projects/vc/courses/TDT17/other/Football2025/RBK-BODO/part3/RBK_BODO_PART3/annotations.xml',
}


# Image dimensions
IMG_WIDTH = 1920
IMG_HEIGHT = 1080

def validate_bbox(xtl, ytl, xbr, ybr, frame, track_id, label):
    """Check if bounding box has any issues"""
    issues = []

    # Check for negative coordinates
    if xtl < 0 or ytl < 0 or xbr < 0 or ybr < 0:
        issues.append(f"Negative coordinates: ({xtl}, {ytl}, {xbr}, {ybr})")

    # Check for zero-area boxes
    if xtl >= xbr or ytl >= ybr:
        issues.append(f"Zero-area box: ({xtl}, {ytl}, {xbr}, {ybr})")

    # Check for out-of-bounds coordinates
    if xbr > IMG_WIDTH or ybr > IMG_HEIGHT:
        issues.append(f"Out of bounds: ({xtl}, {ytl}, {xbr}, {ybr}) > ({IMG_WIDTH}, {IMG_HEIGHT})")

    return issues

def analyze_dataset(dataset_name, xml_path):
    """Analyze one dataset for bounding box issues"""
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

    # Track issues
    anomalies = defaultdict(list)
    total_boxes = 0

    # Check all tracks and boxes
    for track in root.findall('track'):
        track_id = track.get('id')
        label = track.get('label')

        for box in track.findall('box'):
            total_boxes += 1
            frame = box.get('frame')

            # Extract coordinates
            xtl = float(box.get('xtl'))
            ytl = float(box.get('ytl'))
            xbr = float(box.get('xbr'))
            ybr = float(box.get('ybr'))

            # Validate
            issues = validate_bbox(xtl, ytl, xbr, ybr, frame, track_id, label)
            if issues:
                for issue in issues:
                    anomalies[issue].append({
                        'frame': frame,
                        'track_id': track_id,
                        'label': label
                    })

    # Report results
    print(f"\nTotal frames: {total_frames}")
    print(f"Total boxes checked: {total_boxes}")

    if anomalies:
        print(f"\nANOMALIES FOUND: {len(anomalies)} types")
        for issue_type, occurrences in anomalies.items():
            print(f"\n  {issue_type}")
            print(f"  Count: {len(occurrences)}")
            # Show first 3 examples
            for i, ex in enumerate(occurrences[:3]):
                print(f"    - Frame {ex['frame']}, Track {ex['track_id']}, Label: {ex['label']}")
            if len(occurrences) > 3:
                print(f"    ... and {len(occurrences) - 3} more")
    else:
        print("\nNo anomalies found - all bounding boxes are valid")

    return len(anomalies)

def main():
    print("="*80)
    print("STEP 1a: BOUNDING BOX VALIDATION")
    print("="*80)

    total_anomalies = 0
    for dataset_name, xml_path in DATASETS.items():
        num_issues = analyze_dataset(dataset_name, xml_path)
        if num_issues:
            total_anomalies += num_issues

    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    if total_anomalies == 0:
        print("All datasets passed validation - no bounding box issues found")
    else:
        print(f"Found issues in {total_anomalies} datasets")
    print("="*80)

if __name__ == '__main__':
    main()
