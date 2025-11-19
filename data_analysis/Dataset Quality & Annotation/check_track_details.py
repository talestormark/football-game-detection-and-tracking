#!/usr/bin/env python3
"""
Analyze track details to understand the high track count
"""

import xml.etree.ElementTree as ET
from pathlib import Path
from collections import Counter, defaultdict

xml_path = '/cluster/projects/vc/courses/TDT17/other/Football2025/RBK-AALESUND/annotations.xml'

tree = ET.parse(xml_path)
root = tree.getroot()

# Analyze tracks
track_info = []

for track in root.findall('track'):
    track_id = track.get('id')
    label = track.get('label')

    # Count boxes in this track
    boxes = track.findall('box')
    num_boxes = len(boxes)

    # Get frame range
    frames = [int(box.get('frame')) for box in boxes]
    frame_start = min(frames)
    frame_end = max(frames)

    # Get team for players
    team = None
    if label == 'player':
        first_box = boxes[0]
        team_attr = first_box.find("attribute[@name='team']")
        if team_attr is not None:
            team = team_attr.text

    track_info.append({
        'id': track_id,
        'label': label,
        'team': team,
        'boxes': num_boxes,
        'frame_start': frame_start,
        'frame_end': frame_end,
        'duration': frame_end - frame_start + 1
    })

# Sort by label and team
track_info.sort(key=lambda x: (x['label'], x['team'] or '', int(x['id'])))

# Print summary
print("="*80)
print("RBK-AALESUND TRACK ANALYSIS")
print("="*80)

label_counts = Counter([t['label'] for t in track_info])
print(f"\nTracks by label:")
for label, count in label_counts.items():
    print(f"  {label}: {count}")

team_counts = Counter([t['team'] for t in track_info if t['team']])
print(f"\nPlayer tracks by team:")
for team, count in sorted(team_counts.items()):
    print(f"  {team}: {count}")

print(f"\n{'='*80}")
print("DETAILED TRACK LIST")
print("="*80)
print(f"{'ID':<5} {'Label':<12} {'Team':<10} {'Boxes':<7} {'Start':<7} {'End':<7} {'Duration':<8}")
print("-"*80)

for t in track_info:
    team_str = t['team'] or '-'
    print(f"{t['id']:<5} {t['label']:<12} {team_str:<10} {t['boxes']:<7} {t['frame_start']:<7} {t['frame_end']:<7} {t['duration']:<8}")
