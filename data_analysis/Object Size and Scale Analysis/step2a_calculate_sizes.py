#!/usr/bin/env python3
"""
Step 2a - Calculate Bounding Box Areas and Aspect Ratios
Analyzes object sizes across all datasets
"""

import xml.etree.ElementTree as ET
from pathlib import Path
import json
import numpy as np
from collections import defaultdict

# Dataset paths
DATASETS = {
    'RBK-VIKING': '/cluster/projects/vc/courses/TDT17/other/Football2025/RBK-VIKING/annotations.xml',
    'RBK-AALESUND': '/cluster/projects/vc/courses/TDT17/other/Football2025/RBK-AALESUND/annotations.xml',
    'RBK-FREDRIKSTAD': '/cluster/projects/vc/courses/TDT17/other/Football2025/RBK-FREDRIKSTAD/annotations.xml',
    'RBK-HamKam': '/cluster/projects/vc/courses/TDT17/other/Football2025/RBK-HamKam/annotations.xml',
    'RBK-BODO-part3': '/cluster/projects/vc/courses/TDT17/other/Football2025/RBK-BODO/part3/RBK_BODO_PART3/annotations.xml',
}

# Image dimensions
IMG_WIDTH = 1920
IMG_HEIGHT = 1080
IMG_AREA = IMG_WIDTH * IMG_HEIGHT

def analyze_boxes(xml_path):
    """Extract size and aspect ratio information from all boxes"""
    tree = ET.parse(xml_path)
    root = tree.getroot()

    # Store measurements by class
    measurements = defaultdict(lambda: {
        'areas': [],
        'widths': [],
        'heights': [],
        'aspect_ratios': [],
        'relative_areas': []
    })

    for track in root.findall('track'):
        label = track.get('label')

        for box in track.findall('box'):
            # Extract coordinates
            xtl = float(box.get('xtl'))
            ytl = float(box.get('ytl'))
            xbr = float(box.get('xbr'))
            ybr = float(box.get('ybr'))

            # Calculate dimensions
            width = xbr - xtl
            height = ybr - ytl
            area = width * height
            aspect_ratio = width / height if height > 0 else 0
            relative_area = (area / IMG_AREA) * 100  # Percentage

            # Determine class
            class_name = 'ball' if label == 'ball' else None
            if label == 'player':
                team_attr = box.find("attribute[@name='team']")
                if team_attr is not None:
                    team = team_attr.text
                    if team == 'referee':
                        class_name = 'referee'
                    else:
                        class_name = f"{team}_player"

            if class_name:
                measurements[class_name]['areas'].append(area)
                measurements[class_name]['widths'].append(width)
                measurements[class_name]['heights'].append(height)
                measurements[class_name]['aspect_ratios'].append(aspect_ratio)
                measurements[class_name]['relative_areas'].append(relative_area)

    return measurements

def compute_statistics(values):
    """Compute summary statistics"""
    if not values:
        return {}

    arr = np.array(values)
    return {
        'min': float(np.min(arr)),
        'max': float(np.max(arr)),
        'mean': float(np.mean(arr)),
        'median': float(np.median(arr)),
        'std': float(np.std(arr)),
        'q25': float(np.percentile(arr, 25)),
        'q75': float(np.percentile(arr, 75)),
        'count': len(values)
    }

def main():
    print("="*80)
    print("STEP 2a: CALCULATING BOUNDING BOX SIZES AND ASPECT RATIOS")
    print("="*80)

    # Aggregate measurements across all datasets
    all_measurements = defaultdict(lambda: {
        'areas': [],
        'widths': [],
        'heights': [],
        'aspect_ratios': [],
        'relative_areas': []
    })

    # Process each dataset
    for dataset_name, xml_path in DATASETS.items():
        print(f"\nProcessing: {dataset_name}...")

        xml_file = Path(xml_path)
        if not xml_file.exists():
            print(f"  File not found: {xml_path}")
            continue

        measurements = analyze_boxes(xml_path)

        # Aggregate into overall measurements
        for class_name, data in measurements.items():
            for metric in ['areas', 'widths', 'heights', 'aspect_ratios', 'relative_areas']:
                all_measurements[class_name][metric].extend(data[metric])

    # Compute statistics for each class
    print("\n" + "="*80)
    print("STATISTICS BY CLASS")
    print("="*80)

    statistics = {}

    for class_name in ['home_player', 'away_player', 'referee', 'ball']:
        if class_name not in all_measurements:
            print(f"\n{class_name}: No data")
            continue

        print(f"\n{class_name.upper().replace('_', ' ')}:")
        print("-" * 40)

        data = all_measurements[class_name]

        # Area statistics
        area_stats = compute_statistics(data['areas'])
        print(f"  Area (pixels²):")
        print(f"    Mean: {area_stats['mean']:.0f}, Median: {area_stats['median']:.0f}")
        print(f"    Range: [{area_stats['min']:.0f}, {area_stats['max']:.0f}]")

        # Aspect ratio statistics
        ar_stats = compute_statistics(data['aspect_ratios'])
        print(f"  Aspect Ratio (W/H):")
        print(f"    Mean: {ar_stats['mean']:.2f}, Median: {ar_stats['median']:.2f}")
        print(f"    Range: [{ar_stats['min']:.2f}, {ar_stats['max']:.2f}]")

        # Relative size
        rel_stats = compute_statistics(data['relative_areas'])
        print(f"  Relative Size (% of image):")
        print(f"    Mean: {rel_stats['mean']:.3f}%, Median: {rel_stats['median']:.3f}%")

        print(f"  Sample count: {area_stats['count']:,}")

        # Store for later use
        statistics[class_name] = {
            'area': area_stats,
            'width': compute_statistics(data['widths']),
            'height': compute_statistics(data['heights']),
            'aspect_ratio': ar_stats,
            'relative_area': rel_stats
        }

    # Save results
    output_file = Path('/cluster/work/tmstorma/Football2025/data_analysis/Object Size and Scale Analysis/size_statistics.json')

    output_data = {
        'statistics': statistics,
        'raw_measurements': {
            class_name: {
                metric: values for metric, values in data.items()
            }
            for class_name, data in all_measurements.items()
        }
    }

    with open(output_file, 'w') as f:
        json.dump(output_data, f, indent=2)

    print("\n" + "="*80)
    print(f"Results saved to: {output_file}")
    print("="*80)

if __name__ == '__main__':
    main()
