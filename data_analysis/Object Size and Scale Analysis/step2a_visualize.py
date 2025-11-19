#!/usr/bin/env python3
"""
Step 2a - Visualize Object Size and Scale Distributions
Creates box plots, scatter plots, and histograms
"""

import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Load statistics
stats_file = Path('/cluster/work/tmstorma/Football2025/data_analysis/Object Size and Scale Analysis/size_statistics.json')
with open(stats_file, 'r') as f:
    data = json.load(f)

raw_measurements = data['raw_measurements']

# Output directory
output_dir = Path('/cluster/work/tmstorma/Football2025/data_analysis/Object Size and Scale Analysis/visualizations')
output_dir.mkdir(exist_ok=True)

# Define colors for each class
colors = {
    'home_player': '#3498db',
    'away_player': '#e74c3c',
    'referee': '#f39c12',
    'ball': '#2ecc71'
}

# Class order and labels
classes = ['home_player', 'away_player', 'referee', 'ball']
class_labels = ['Home Player', 'Away Player', 'Referee', 'Ball']

print("="*80)
print("STEP 2a: GENERATING SIZE VISUALIZATIONS")
print("="*80)
print("Generating 2 essential visualizations:\n")

# 1. BOX PLOT - Area Distribution
print("1. Creating box plot for area distribution...")

fig, ax = plt.subplots(figsize=(12, 7))

areas_data = [raw_measurements[cls]['areas'] for cls in classes if cls in raw_measurements]
positions = range(len(areas_data))

bp = ax.boxplot(areas_data, positions=positions, widths=0.6, patch_artist=True,
                showfliers=False)  # Hide outliers for clarity

# Color the boxes
for patch, cls in zip(bp['boxes'], classes):
    patch.set_facecolor(colors[cls])
    patch.set_alpha(0.7)

ax.set_xlabel('Object Class', fontsize=12, fontweight='bold')
ax.set_ylabel('Area (pixels²)', fontsize=12, fontweight='bold')
ax.set_title('Bounding Box Area Distribution by Class', fontsize=14, fontweight='bold')
ax.set_xticks(positions)
ax.set_xticklabels(class_labels)
ax.grid(axis='y', alpha=0.3, linestyle='--')

plt.tight_layout()
box_plot_path = output_dir / 'area_distribution_boxplot.png'
plt.savefig(box_plot_path, dpi=300, bbox_inches='tight')
print(f"   Saved: {box_plot_path}")
plt.close()

# 2. HISTOGRAM - Aspect Ratio Distribution
print("\n2. Creating aspect ratio histograms...")

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
axes = axes.flatten()

for i, (cls, label) in enumerate(zip(classes, class_labels)):
    if cls in raw_measurements:
        aspect_ratios = raw_measurements[cls]['aspect_ratios']

        # Filter extreme outliers for better visualization
        ar_array = np.array(aspect_ratios)
        ar_filtered = ar_array[(ar_array > 0.1) & (ar_array < 3.0)]

        axes[i].hist(ar_filtered, bins=50, color=colors[cls], alpha=0.7, edgecolor='black')
        axes[i].axvline(np.median(ar_filtered), color='red', linestyle='--',
                       linewidth=2, label=f'Median: {np.median(ar_filtered):.2f}')
        axes[i].set_xlabel('Aspect Ratio (W/H)', fontsize=10, fontweight='bold')
        axes[i].set_ylabel('Frequency', fontsize=10, fontweight='bold')
        axes[i].set_title(f'{label}', fontsize=12, fontweight='bold')
        axes[i].legend()
        axes[i].grid(axis='y', alpha=0.3, linestyle='--')

plt.suptitle('Aspect Ratio Distribution by Class', fontsize=14, fontweight='bold')
plt.tight_layout()
hist_path = output_dir / 'aspect_ratio_histograms.png'
plt.savefig(hist_path, dpi=300, bbox_inches='tight')
print(f"   Saved: {hist_path}")
plt.close()

print("\n" + "="*80)
print("VISUALIZATION COMPLETE")
print(f"Output directory: {output_dir}")
print("="*80)
