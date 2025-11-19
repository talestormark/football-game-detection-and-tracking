#!/usr/bin/env python3
"""
Quick check of team attribute distribution across all Football2025 datasets
"""

import xml.etree.ElementTree as ET
from pathlib import Path
from collections import Counter

# All dataset paths
datasets = {
    'RBK-VIKING': '/cluster/projects/vc/courses/TDT17/other/Football2025/RBK-VIKING/annotations.xml',
    'RBK-AALESUND': '/cluster/projects/vc/courses/TDT17/other/Football2025/RBK-AALESUND/annotations.xml',
    'RBK-FREDRIKSTAD': '/cluster/projects/vc/courses/TDT17/other/Football2025/RBK-FREDRIKSTAD/annotations.xml',
    'RBK-HamKam': '/cluster/projects/vc/courses/TDT17/other/Football2025/RBK-HamKam/annotations.xml',
    'RBK-BODO-part1': '/cluster/projects/vc/courses/TDT17/other/Football2025/RBK-BODO/part1/RBK_BODO_PART1/annotations.xml',
    'RBK-BODO-part2': '/cluster/projects/vc/courses/TDT17/other/Football2025/RBK-BODO/part2/RBK_BODO_PART2/annotations.xml',
    'RBK-BODO-part3': '/cluster/projects/vc/courses/TDT17/other/Football2025/RBK-BODO/part3/RBK_BODO_PART3/annotations.xml',
}

print("="*80)
print("TEAM ATTRIBUTE ANALYSIS - ALL FOOTBALL2025 DATASETS")
print("="*80)

for dataset_name, xml_path in datasets.items():
    print(f"\n{'='*80}")
    print(f"Dataset: {dataset_name}")
    print("="*80)

    xml_file = Path(xml_path)

    if not xml_file.exists():
        print(f"  ❌ File not found: {xml_path}")
        continue

    try:
        tree = ET.parse(xml_file)
        root = tree.getroot()

        # Get metadata
        meta = root.find('meta')
        task = meta.find('task')
        total_frames = int(task.find('size').text)

        # Count tracks and boxes by class
        track_counts = Counter()
        box_counts = Counter()
        team_counts = Counter()

        for track in root.findall('track'):
            label = track.get('label')
            track_counts[label] += 1

            for box in track.findall('box'):
                box_counts[label] += 1

                # Check for team attribute
                if label == 'player':
                    team_attr = box.find("attribute[@name='team']")
                    if team_attr is not None:
                        team_counts[team_attr.text] += 1

        print(f"\n📊 Basic Stats:")
        print(f"  Total frames: {total_frames}")
        print(f"  Total tracks: {sum(track_counts.values())}")
        print(f"  Total boxes: {sum(box_counts.values())}")

        print(f"\n📦 Boxes by Class:")
        for label, count in sorted(box_counts.items()):
            pct = 100 * count / sum(box_counts.values())
            print(f"  {label:20s}: {count:6d} ({pct:5.2f}%)")

        print(f"\n👥 Team Distribution (player boxes only):")
        if team_counts:
            for team, count in sorted(team_counts.items()):
                pct = 100 * count / sum(team_counts.values())
                print(f"  {team:20s}: {count:6d} ({pct:5.2f}%)")

            # Check if teams are distinguished
            unique_teams = len(team_counts)
            if unique_teams == 1:
                print(f"\n  ⚠️  WARNING: Only 1 team type annotated!")
            elif unique_teams == 2:
                print(f"\n  ⚠️  PARTIAL: 2 team types (missing referee or one team)")
            elif unique_teams == 3:
                print(f"\n  ✅ GOOD: All 3 team types annotated (home/away/referee)")
        else:
            print("  ❌ No team attributes found!")

    except Exception as e:
        print(f"  ❌ Error parsing file: {e}")

print("\n" + "="*80)
print("SUMMARY")
print("="*80)
print("\nDatasets with proper team distinction will show all 3 types:")
print("  - home")
print("  - away")
print("  - referee")
print("\nDatasets with only 'home' need re-annotation or team classification post-processing.")
print("="*80)
