#!/usr/bin/env python3
"""
Step 2b - Visualize Track Trajectories
Creates visualizations of object paths over time
"""

import xml.etree.ElementTree as ET
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import random
from matplotlib.colors import LinearSegmentedColormap

# Dataset paths
DATASETS = {
    'RBK-AALESUND': '/cluster/projects/vc/courses/TDT17/other/Football2025/RBK-AALESUND/annotations.xml',
    'RBK-FREDRIKSTAD': '/cluster/projects/vc/courses/TDT17/other/Football2025/RBK-FREDRIKSTAD/annotations.xml',
}

# Image dimensions for reference
IMG_WIDTH = 1920
IMG_HEIGHT = 1080

# Colors
COLORS = {
    'home': '#3498db',
    'away': '#e74c3c',
    'referee': '#f39c12',
    'ball': '#2ecc71'
}

def get_track_data(xml_path):
    """Extract track trajectories"""
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

        # Get team for players
        team = None
        if label == 'player':
            first_box = track.findall('box')[0]
            team_attr = first_box.find("attribute[@name='team']")
            if team_attr is not None:
                team = team_attr.text

        class_name = team if team else label

        tracks.append({
            'track_id': track_id,
            'label': label,
            'class': class_name,
            'frames': frames,
            'positions': positions
        })

    return tracks

def visualize_dataset_tracks(dataset_name, xml_path, output_dir):
    """Create trajectory visualizations for one dataset"""
    print(f"\nProcessing {dataset_name}...")

    tracks = get_track_data(xml_path)

    # Group by class
    tracks_by_class = {}
    for track in tracks:
        class_name = track['class']
        if class_name not in ['event_labels', 'event']:  # Skip event markers
            if class_name not in tracks_by_class:
                tracks_by_class[class_name] = []
            tracks_by_class[class_name].append(track)

    # 1. All player tracks on field
    print(f"  Creating full trajectory visualization...")
    fig, ax = plt.subplots(figsize=(16, 9))

    # Draw field boundary
    ax.add_patch(plt.Rectangle((0, 0), IMG_WIDTH, IMG_HEIGHT,
                               fill=False, edgecolor='gray', linewidth=2))

    for class_name, class_tracks in tracks_by_class.items():
        if class_name in COLORS:
            color = COLORS[class_name]

            for track in class_tracks:
                positions = track['positions']
                x_coords = [p[0] for p in positions]
                y_coords = [p[1] for p in positions]

                # Plot trajectory with semi-transparent line
                ax.plot(x_coords, y_coords, color=color, alpha=0.3, linewidth=1)

                # Mark start and end
                ax.plot(x_coords[0], y_coords[0], 'o', color=color,
                       markersize=4, alpha=0.5)

    ax.set_xlim(0, IMG_WIDTH)
    ax.set_ylim(IMG_HEIGHT, 0)  # Invert y-axis
    ax.set_xlabel('X Position (pixels)', fontsize=12)
    ax.set_ylabel('Y Position (pixels)', fontsize=12)
    ax.set_title(f'{dataset_name}: All Object Trajectories', fontsize=14, fontweight='bold')
    ax.set_aspect('equal')
    ax.grid(alpha=0.3)

    # Legend
    legend_elements = [plt.Line2D([0], [0], color=COLORS[c], lw=2, label=c.replace('_', ' ').title())
                      for c in tracks_by_class.keys() if c in COLORS]
    ax.legend(handles=legend_elements, loc='upper right')

    plt.tight_layout()
    output_path = output_dir / f'{dataset_name}_all_trajectories.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"    Saved: {output_path}")
    plt.close()

    # 2. Individual class examples
    print(f"  Creating individual track examples...")

    for class_name, class_tracks in tracks_by_class.items():
        if class_name in COLORS:
            # Select 2 example tracks (or 1 for ball)
            num_examples = min(2, len(class_tracks)) if class_name != 'ball' else min(1, len(class_tracks))
            example_tracks = random.sample(class_tracks, num_examples)

            fig, ax = plt.subplots(figsize=(16, 9))

            # Draw field boundary
            ax.add_patch(plt.Rectangle((0, 0), IMG_WIDTH, IMG_HEIGHT,
                                       fill=False, edgecolor='gray', linewidth=2))

            # Special handling for ball to show speed variations
            if class_name == 'ball':
                for track in example_tracks:
                    positions = track['positions']
                    frames = track['frames']
                    x_coords = np.array([p[0] for p in positions])
                    y_coords = np.array([p[1] for p in positions])

                    # Calculate speeds (distance per frame)
                    speeds = []
                    for i in range(len(positions) - 1):
                        dx = x_coords[i+1] - x_coords[i]
                        dy = y_coords[i+1] - y_coords[i]
                        distance = np.sqrt(dx**2 + dy**2)
                        frame_gap = frames[i+1] - frames[i]
                        speed = distance / frame_gap if frame_gap > 0 else 0
                        speeds.append(speed)

                    # Normalize speeds for color mapping
                    speeds = np.array(speeds)
                    if len(speeds) > 0 and speeds.max() > 0:
                        norm_speeds = speeds / speeds.max()
                    else:
                        norm_speeds = speeds

                    # Create color map (blue = slow, red = fast)
                    cmap = plt.cm.coolwarm

                    # Plot segments with colors based on speed
                    for i in range(len(x_coords) - 1):
                        ax.plot(x_coords[i:i+2], y_coords[i:i+2],
                               color=cmap(norm_speeds[i]), linewidth=3, alpha=0.8)

                    # Mark start (circle) and end (star)
                    ax.plot(x_coords[0], y_coords[0], 'o', color='green',
                           markersize=10, label=f"Start (Track {track['track_id']})")
                    ax.plot(x_coords[-1], y_coords[-1], '*', color='black',
                           markersize=14, label='End')

                    # Add colorbar
                    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=0, vmax=speeds.max()))
                    sm.set_array([])
                    cbar = plt.colorbar(sm, ax=ax, fraction=0.046, pad=0.04)
                    cbar.set_label('Speed (pixels/frame)', fontsize=12)

            else:
                # Regular plotting for players and referee
                # Use different colors for each track
                track_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']

                for idx, track in enumerate(example_tracks):
                    positions = track['positions']
                    x_coords = [p[0] for p in positions]
                    y_coords = [p[1] for p in positions]

                    track_color = track_colors[idx % len(track_colors)]

                    # Plot trajectory
                    ax.plot(x_coords, y_coords, color=track_color,
                           linewidth=2, alpha=0.7, label=f"Track {track['track_id']}")

                    # Mark start (circle) and end (star)
                    ax.plot(x_coords[0], y_coords[0], 'o', color=track_color,
                           markersize=8)
                    ax.plot(x_coords[-1], y_coords[-1], '*', color=track_color,
                           markersize=12)

            ax.set_xlim(0, IMG_WIDTH)
            ax.set_ylim(IMG_HEIGHT, 0)
            ax.set_xlabel('X Position (pixels)', fontsize=12)
            ax.set_ylabel('Y Position (pixels)', fontsize=12)
            ax.set_title(f'{dataset_name}: {class_name.title()} Track Examples',
                        fontsize=14, fontweight='bold')
            ax.set_aspect('equal')
            ax.grid(alpha=0.3)
            ax.legend(loc='upper right')

            plt.tight_layout()
            output_path = output_dir / f'{dataset_name}_{class_name}_examples.png'
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"    Saved: {output_path}")
            plt.close()

def main():
    print("="*80)
    print("STEP 2b: VISUALIZING TRACK TRAJECTORIES")
    print("="*80)

    # Create output directory
    output_dir = Path('/cluster/work/tmstorma/Football2025/data_analysis/Tracking ID Stability/visualizations')
    output_dir.mkdir(exist_ok=True)

    # Set random seed for reproducibility
    random.seed(42)

    # Process selected datasets (not all to save time)
    for dataset_name, xml_path in DATASETS.items():
        xml_file = Path(xml_path)
        if xml_file.exists():
            visualize_dataset_tracks(dataset_name, xml_path, output_dir)

    print("\n" + "="*80)
    print("VISUALIZATION COMPLETE")
    print(f"Output directory: {output_dir}")
    print("="*80)

if __name__ == '__main__':
    main()
