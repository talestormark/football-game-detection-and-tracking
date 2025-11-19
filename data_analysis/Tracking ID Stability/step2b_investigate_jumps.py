#!/usr/bin/env python3
"""
Step 2b - Investigate Player Position Jumps
Creates detailed visualizations of player jumps in RBK-BODO-part3
"""

import xml.etree.ElementTree as ET
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import cv2

# Image dimensions
IMG_WIDTH = 1920
IMG_HEIGHT = 1080

def get_track_with_jumps(xml_path):
    """Extract tracks and identify position jumps"""
    tree = ET.parse(xml_path)
    root = tree.getroot()

    tracks_with_jumps = []

    for track in root.findall('track'):
        track_id = track.get('id')
        label = track.get('label')

        if label != 'player':
            continue

        frames = []
        positions = []
        boxes = []

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
            boxes.append((xtl, ytl, xbr, ybr))

        if len(frames) < 2:
            continue

        # Check for jumps
        jumps = []
        for i in range(len(positions) - 1):
            dx = positions[i+1][0] - positions[i][0]
            dy = positions[i+1][1] - positions[i][1]
            distance = np.sqrt(dx**2 + dy**2)
            frame_gap = frames[i+1] - frames[i]
            speed = distance / frame_gap if frame_gap > 0 else 0

            if speed > 300:  # Jump threshold
                jumps.append({
                    'index': i,
                    'from_frame': frames[i],
                    'to_frame': frames[i+1],
                    'from_pos': positions[i],
                    'to_pos': positions[i+1],
                    'from_box': boxes[i],
                    'to_box': boxes[i+1],
                    'distance': distance,
                    'speed': speed
                })

        if jumps:
            # Get team
            team = 'unknown'
            first_box = track.findall('box')[0]
            team_attr = first_box.find("attribute[@name='team']")
            if team_attr is not None:
                team = team_attr.text

            tracks_with_jumps.append({
                'track_id': track_id,
                'team': team,
                'frames': frames,
                'positions': positions,
                'boxes': boxes,
                'jumps': jumps
            })

    return tracks_with_jumps

def visualize_jump(track, jump, output_path, images_dir):
    """Create simple before/after visualization of a jump"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))

    images_path = Path(images_dir)

    # Before frame
    before_img_path = images_path / f'frame_{jump["from_frame"]+1:06d}.png'  # +1 for offset
    if before_img_path.exists():
        img = cv2.imread(str(before_img_path))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Draw box
        xtl, ytl, xbr, ybr = [int(c) for c in jump['from_box']]
        cv2.rectangle(img, (xtl, ytl), (xbr, ybr), (0, 255, 0), 5)
        cv2.putText(img, f"Track {track['track_id']} - Frame {jump['from_frame']}", (50, 80),
                   cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 4)

        ax1.imshow(img)
    ax1.axis('off')
    ax1.set_title(f'Before: Frame {jump["from_frame"]}', fontsize=14, fontweight='bold')

    # After frame
    after_img_path = images_path / f'frame_{jump["to_frame"]+1:06d}.png'  # +1 for offset
    if after_img_path.exists():
        img = cv2.imread(str(after_img_path))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Draw box
        xtl, ytl, xbr, ybr = [int(c) for c in jump['to_box']]
        cv2.rectangle(img, (xtl, ytl), (xbr, ybr), (255, 0, 0), 5)
        cv2.putText(img, f"Track {track['track_id']} - Frame {jump['to_frame']}", (50, 80),
                   cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 0), 4)

        ax2.imshow(img)
    ax2.axis('off')
    ax2.set_title(f'After: Frame {jump["to_frame"]}', fontsize=14, fontweight='bold')

    plt.suptitle(f'Track {track["track_id"]} ({track["team"]}) - ID Switch at Frame {jump["from_frame"]}→{jump["to_frame"]}',
                fontsize=16, fontweight='bold')

    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"  Saved: {output_path}")

def main():
    print("="*80)
    print("INVESTIGATING PLAYER POSITION JUMPS - RBK-BODO-part3")
    print("="*80)

    xml_path = '/cluster/projects/vc/courses/TDT17/other/Football2025/RBK-BODO/part3/RBK_BODO_PART3/annotations.xml'
    images_dir = '/cluster/projects/vc/courses/TDT17/other/Football2025/RBK-BODO/part3/RBK_BODO_PART3/data/images/train'

    output_dir = Path('/cluster/work/tmstorma/Football2025/data_analysis/Tracking ID Stability/jump_investigation')
    output_dir.mkdir(exist_ok=True)

    tracks_with_jumps = get_track_with_jumps(xml_path)

    print(f"\nFound {len(tracks_with_jumps)} player tracks with position jumps")

    for i, track in enumerate(tracks_with_jumps):
        print(f"\nTrack {track['track_id']} ({track['team']}):")
        print(f"  Total frames: {len(track['frames'])}")
        print(f"  Number of jumps: {len(track['jumps'])}")

        for j, jump in enumerate(track['jumps']):
            print(f"    Jump {j+1}: {jump['distance']:.0f} px in {jump['to_frame']-jump['from_frame']} frames")
            print(f"            From frame {jump['from_frame']} to {jump['to_frame']}")

            # Create visualization
            output_path = output_dir / f"jump_track{track['track_id']}_#{j+1}.png"
            visualize_jump(track, jump, output_path, images_dir)

    print("\n" + "="*80)
    print("INVESTIGATION COMPLETE")
    print(f"Output directory: {output_dir}")
    print("="*80)
    print("\nWHY THIS IS NOT (necessarily) AN ID SWITCH:")
    print("- ID switch = DIFFERENT objects get same ID")
    print("- Position jump = SAME ID moves far (could be annotation error)")
    print("- Check the frame images to see if it's the same player or not")
    print("="*80)

if __name__ == '__main__':
    main()
