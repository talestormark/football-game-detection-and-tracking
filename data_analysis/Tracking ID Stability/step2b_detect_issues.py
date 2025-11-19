#!/usr/bin/env python3
"""
Step 2b - Detect ID Switches and Track Issues
Identifies problematic tracks and potential ID switches
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

def get_track_data(xml_path):
    """Extract detailed track information"""
    tree = ET.parse(xml_path)
    root = tree.getroot()

    tracks = []

    for track in root.findall('track'):
        track_id = track.get('id')
        label = track.get('label')

        frames = []
        positions = []

        for box in track.findall('box'):
            frame_num = int(box.get('frame'))
            xtl = float(box.get('xtl'))
            ytl = float(box.get('ytl'))
            xbr = float(box.get('xbr'))
            ybr = float(box.get('ybr'))

            center_x = (xtl + xbr) / 2
            center_y = (ytl + ybr) / 2

            frames.append(frame_num)
            positions.append((center_x, center_y))

        if not frames:
            continue

        frames = sorted(frames)

        # Get team for players
        team = None
        if label == 'player':
            first_box = track.findall('box')[0]
            team_attr = first_box.find("attribute[@name='team']")
            if team_attr is not None:
                team = team_attr.text

        tracks.append({
            'track_id': track_id,
            'label': label,
            'team': team,
            'frames': frames,
            'positions': positions,
            'frame_start': frames[0],
            'frame_end': frames[-1]
        })

    return tracks

def detect_position_jumps(track):
    """Detect sudden large position jumps within a track"""
    jumps = []

    positions = track['positions']
    frames = track['frames']

    for i in range(len(positions) - 1):
        # Calculate distance between consecutive positions
        dx = positions[i+1][0] - positions[i][0]
        dy = positions[i+1][1] - positions[i][1]
        distance = np.sqrt(dx**2 + dy**2)

        # Frame gap
        frame_gap = frames[i+1] - frames[i]

        # Speed in pixels per frame
        speed = distance / frame_gap if frame_gap > 0 else 0

        # Flag unusually fast movement (potential ID switch)
        # Threshold: 300 pixels per frame (very fast movement)
        if speed > 300:
            jumps.append({
                'from_frame': frames[i],
                'to_frame': frames[i+1],
                'distance': distance,
                'speed': speed,
                'frame_gap': frame_gap
            })

    return jumps

def find_potential_id_switches(tracks):
    """Find pairs of tracks that might be the same object with different IDs"""
    switches = []

    # Only check same class tracks
    tracks_by_class = defaultdict(list)
    for track in tracks:
        class_name = track['team'] if track['team'] else track['label']
        if class_name != 'event_labels':  # Skip event labels
            tracks_by_class[class_name].append(track)

    for class_name, class_tracks in tracks_by_class.items():
        # Compare pairs of tracks
        for i, track1 in enumerate(class_tracks):
            for track2 in class_tracks[i+1:]:
                # Check if track2 starts shortly after track1 ends
                time_gap = track2['frame_start'] - track1['frame_end']

                if 1 <= time_gap <= 10:  # Small gap between tracks
                    # Check if end position of track1 is close to start position of track2
                    end_pos1 = track1['positions'][-1]
                    start_pos2 = track2['positions'][0]

                    dx = start_pos2[0] - end_pos1[0]
                    dy = start_pos2[1] - end_pos1[1]
                    distance = np.sqrt(dx**2 + dy**2)

                    # If distance is small, might be same object
                    if distance < 200:  # pixels
                        switches.append({
                            'track1_id': track1['track_id'],
                            'track2_id': track2['track_id'],
                            'class': class_name,
                            'time_gap': time_gap,
                            'position_distance': distance,
                            'track1_end': track1['frame_end'],
                            'track2_start': track2['frame_start']
                        })

    return switches

def analyze_dataset(dataset_name, xml_path):
    """Analyze tracking issues for one dataset"""
    print(f"\n{'='*80}")
    print(f"Dataset: {dataset_name}")
    print('='*80)

    xml_file = Path(xml_path)
    if not xml_file.exists():
        print(f"File not found: {xml_path}")
        return None

    tracks = get_track_data(xml_path)

    print(f"\nTotal tracks: {len(tracks)}")

    # Detect position jumps
    tracks_with_jumps = []
    total_jumps = 0

    for track in tracks:
        jumps = detect_position_jumps(track)
        if jumps:
            tracks_with_jumps.append({
                'track_id': track['track_id'],
                'label': track['label'],
                'team': track['team'],
                'jumps': jumps
            })
            total_jumps += len(jumps)

    print(f"\nPosition Jump Analysis:")
    print(f"  Tracks with sudden jumps: {len(tracks_with_jumps)}")
    print(f"  Total jumps detected: {total_jumps}")

    if tracks_with_jumps:
        print(f"\n  Examples of large jumps:")
        for i, track_info in enumerate(tracks_with_jumps[:3]):
            jump = track_info['jumps'][0]
            class_name = track_info['team'] if track_info['team'] else track_info['label']
            print(f"    Track {track_info['track_id']} ({class_name}): {jump['distance']:.0f} pixels in {jump['frame_gap']} frames (speed: {jump['speed']:.0f} px/frame)")

    # Detect potential ID switches
    switches = find_potential_id_switches(tracks)

    print(f"\nPotential ID Switch Analysis:")
    print(f"  Suspicious track pairs: {len(switches)}")

    if switches:
        print(f"\n  Examples:")
        for i, switch in enumerate(switches[:5]):
            print(f"    {switch['class']}: Track {switch['track1_id']} → Track {switch['track2_id']}")
            print(f"      Gap: {switch['time_gap']} frames, Distance: {switch['position_distance']:.0f} pixels")

    return {
        'dataset': dataset_name,
        'total_tracks': len(tracks),
        'tracks_with_jumps': len(tracks_with_jumps),
        'total_jumps': total_jumps,
        'potential_switches': len(switches),
        'jump_details': tracks_with_jumps,
        'switch_details': switches
    }

def main():
    print("="*80)
    print("STEP 2b: DETECTING ID SWITCHES AND TRACK ISSUES")
    print("="*80)

    all_results = {}

    for dataset_name, xml_path in DATASETS.items():
        result = analyze_dataset(dataset_name, xml_path)
        if result:
            all_results[dataset_name] = result

    # Overall summary
    print("\n" + "="*80)
    print("OVERALL SUMMARY")
    print("="*80)

    total_jumps = sum(r['total_jumps'] for r in all_results.values())
    total_switches = sum(r['potential_switches'] for r in all_results.values())

    print(f"\nAcross all datasets:")
    print(f"  Tracks with position jumps: {sum(r['tracks_with_jumps'] for r in all_results.values())}")
    print(f"  Total position jumps: {total_jumps}")
    print(f"  Potential ID switches: {total_switches}")

    if total_jumps == 0 and total_switches == 0:
        print(f"\n✓ No significant tracking issues detected")
    else:
        print(f"\n⚠️  Some tracking discontinuities detected")
        print(f"     These may be legitimate (occlusions, objects leaving view)")
        print(f"     Or indicate annotation quality issues")

    # Save results
    output_file = Path('/cluster/work/tmstorma/Football2025/data_analysis/Tracking ID Stability/tracking_issues.json')

    with open(output_file, 'w') as f:
        json.dump(all_results, f, indent=2)

    print(f"\nResults saved to: {output_file}")
    print("="*80)

if __name__ == '__main__':
    main()
