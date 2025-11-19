#!/usr/bin/env python3
"""
Step 1b - Count Class Instances
Counts bounding boxes by class: home_player, away_player, referee, ball
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

def count_classes(xml_path):
    """Count bounding boxes by class"""
    tree = ET.parse(xml_path)
    root = tree.getroot()

    class_counts = {
        'home_player': 0,
        'away_player': 0,
        'referee': 0,
        'ball': 0
    }

    for track in root.findall('track'):
        label = track.get('label')

        for box in track.findall('box'):
            if label == 'ball':
                class_counts['ball'] += 1
            elif label == 'player':
                # Get team attribute
                team_attr = box.find("attribute[@name='team']")
                if team_attr is not None:
                    team = team_attr.text
                    if team == 'home':
                        class_counts['home_player'] += 1
                    elif team == 'away':
                        class_counts['away_player'] += 1
                    elif team == 'referee':
                        class_counts['referee'] += 1

    return class_counts

def main():
    print("="*80)
    print("STEP 1b: CLASS INSTANCE COUNTING")
    print("="*80)

    all_results = {}
    total_counts = {
        'home_player': 0,
        'away_player': 0,
        'referee': 0,
        'ball': 0
    }

    # Count classes for each dataset
    for dataset_name, xml_path in DATASETS.items():
        print(f"\n{'='*80}")
        print(f"Dataset: {dataset_name}")
        print('='*80)

        xml_file = Path(xml_path)
        if not xml_file.exists():
            print(f"File not found: {xml_path}")
            continue

        counts = count_classes(xml_path)
        all_results[dataset_name] = counts

        # Calculate total for this dataset
        dataset_total = sum(counts.values())

        print(f"\nClass counts:")
        for class_name, count in counts.items():
            pct = (count / dataset_total * 100) if dataset_total > 0 else 0
            print(f"  {class_name:15s}: {count:6d} ({pct:5.2f}%)")
            total_counts[class_name] += count

        print(f"  {'TOTAL':15s}: {dataset_total:6d}")

    # Overall summary
    print("\n" + "="*80)
    print("OVERALL SUMMARY")
    print("="*80)

    grand_total = sum(total_counts.values())
    print(f"\nTotal instances across all datasets:")
    for class_name, count in total_counts.items():
        pct = (count / grand_total * 100) if grand_total > 0 else 0
        print(f"  {class_name:15s}: {count:6d} ({pct:5.2f}%)")
    print(f"  {'GRAND TOTAL':15s}: {grand_total:6d}")

    # Save results for report generation
    import json
    output_file = Path('/cluster/work/tmstorma/Football2025/data_analysis/Class Distribution/class_counts.json')
    with open(output_file, 'w') as f:
        json.dump({
            'datasets': all_results,
            'totals': total_counts
        }, f, indent=2)
    print(f"\nResults saved to: {output_file}")
    print("="*80)

if __name__ == '__main__':
    main()
