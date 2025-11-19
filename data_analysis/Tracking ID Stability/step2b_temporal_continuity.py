#!/usr/bin/env python3
"""
Step 2b - Analyze Tracking ID Temporal Continuity
Analyzes track persistence, lengths, and temporal gaps
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

def analyze_tracks(xml_path):
    """Extract temporal information about each track"""
    tree = ET.parse(xml_path)
    root = tree.getroot()

    # Get total frames
    meta = root.find('meta')
    task = meta.find('task')
    total_frames = int(task.find('size').text)

    track_info = []

    for track in root.findall('track'):
        track_id = track.get('id')
        label = track.get('label')

        # Get all frame numbers for this track
        frames = []
        positions = []

        for box in track.findall('box'):
            frame_num = int(box.get('frame'))
            frames.append(frame_num)

            # Get center position
            xtl = float(box.get('xtl'))
            ytl = float(box.get('ytl'))
            xbr = float(box.get('xbr'))
            ybr = float(box.get('ybr'))
            center_x = (xtl + xbr) / 2
            center_y = (ytl + ybr) / 2
            positions.append((center_x, center_y))

        if not frames:
            continue

        frames = sorted(frames)
        frame_start = frames[0]
        frame_end = frames[-1]
        num_appearances = len(frames)
        span = frame_end - frame_start + 1

        # Check for gaps (missing frames in the sequence)
        expected_frames = set(range(frame_start, frame_end + 1))
        actual_frames = set(frames)
        gaps = expected_frames - actual_frames
        num_gaps = len(gaps)

        # Calculate continuity ratio
        continuity = num_appearances / span if span > 0 else 0

        # Get team for players
        team = None
        if label == 'player':
            first_box = track.findall('box')[0]
            team_attr = first_box.find("attribute[@name='team']")
            if team_attr is not None:
                team = team_attr.text

        # Calculate displacement (if multiple frames)
        displacement = 0
        if len(positions) > 1:
            for i in range(len(positions) - 1):
                dx = positions[i+1][0] - positions[i][0]
                dy = positions[i+1][1] - positions[i][1]
                displacement += np.sqrt(dx**2 + dy**2)

        track_info.append({
            'track_id': track_id,
            'label': label,
            'team': team,
            'frame_start': frame_start,
            'frame_end': frame_end,
            'span': span,
            'appearances': num_appearances,
            'gaps': num_gaps,
            'continuity': continuity,
            'displacement': displacement,
            'frames': frames,
            'positions': positions
        })

    return track_info, total_frames

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
    }

def analyze_dataset(dataset_name, xml_path):
    """Analyze temporal continuity for one dataset"""
    print(f"\n{'='*80}")
    print(f"Dataset: {dataset_name}")
    print('='*80)

    xml_file = Path(xml_path)
    if not xml_file.exists():
        print(f"File not found: {xml_path}")
        return None

    tracks, total_frames = analyze_tracks(xml_path)

    print(f"\nTotal frames: {total_frames}")
    print(f"Total tracks: {len(tracks)}")

    # Group by class
    tracks_by_class = defaultdict(list)
    for track in tracks:
        class_name = track['team'] if track['team'] else track['label']
        tracks_by_class[class_name].append(track)

    # Analyze each class
    class_stats = {}

    for class_name, class_tracks in sorted(tracks_by_class.items()):
        print(f"\n{class_name.upper()}:")
        print("-" * 40)

        # Track length statistics
        spans = [t['span'] for t in class_tracks]
        appearances = [t['appearances'] for t in class_tracks]
        continuities = [t['continuity'] for t in class_tracks]
        gaps_list = [t['gaps'] for t in class_tracks]

        span_stats = compute_statistics(spans)
        continuity_stats = compute_statistics(continuities)

        print(f"  Tracks: {len(class_tracks)}")
        print(f"  Avg span: {span_stats['mean']:.0f} frames (min: {span_stats['min']:.0f}, max: {span_stats['max']:.0f})")
        print(f"  Avg continuity: {continuity_stats['mean']:.2%}")

        # Find tracks with significant gaps
        tracks_with_gaps = [t for t in class_tracks if t['gaps'] > 0]
        if tracks_with_gaps:
            print(f"  Tracks with gaps: {len(tracks_with_gaps)} ({len(tracks_with_gaps)/len(class_tracks)*100:.1f}%)")
            total_gaps = sum(t['gaps'] for t in tracks_with_gaps)
            print(f"  Total gap frames: {total_gaps}")

        class_stats[class_name] = {
            'count': len(class_tracks),
            'span': span_stats,
            'continuity': continuity_stats,
            'tracks_with_gaps': len(tracks_with_gaps),
            'total_gaps': sum(gaps_list)
        }

    return {
        'dataset': dataset_name,
        'total_frames': total_frames,
        'total_tracks': len(tracks),
        'tracks': tracks,
        'class_stats': class_stats
    }

def main():
    print("="*80)
    print("STEP 2b: TRACKING ID TEMPORAL CONTINUITY ANALYSIS")
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

    total_tracks_all = sum(r['total_tracks'] for r in all_results.values())
    print(f"\nTotal tracks across all datasets: {total_tracks_all}")

    # Aggregate continuity statistics
    all_continuities = []
    all_gaps_count = 0

    for result in all_results.values():
        for track in result['tracks']:
            all_continuities.append(track['continuity'])
            if track['gaps'] > 0:
                all_gaps_count += 1

    continuity_stats = compute_statistics(all_continuities)
    print(f"\nOverall continuity: {continuity_stats['mean']:.2%} (median: {continuity_stats['median']:.2%})")
    print(f"Tracks with temporal gaps: {all_gaps_count} ({all_gaps_count/total_tracks_all*100:.1f}%)")

    # Save results
    output_file = Path('/cluster/work/tmstorma/Football2025/data_analysis/Tracking ID Stability/temporal_continuity.json')

    # Prepare for JSON (remove non-serializable data)
    output_data = {}
    for dataset_name, result in all_results.items():
        output_data[dataset_name] = {
            'total_frames': result['total_frames'],
            'total_tracks': result['total_tracks'],
            'class_stats': result['class_stats'],
            'tracks_summary': [
                {
                    'track_id': t['track_id'],
                    'label': t['label'],
                    'team': t['team'],
                    'span': t['span'],
                    'appearances': t['appearances'],
                    'gaps': t['gaps'],
                    'continuity': t['continuity']
                }
                for t in result['tracks']
            ]
        }

    with open(output_file, 'w') as f:
        json.dump(output_data, f, indent=2)

    print(f"\nResults saved to: {output_file}")
    print("="*80)

if __name__ == '__main__':
    main()
