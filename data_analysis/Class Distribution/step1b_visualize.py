#!/usr/bin/env python3
"""
Step 1b - Visualize Class Distribution
Creates stacked bar chart and pie chart
"""

import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Load results
results_file = Path('/cluster/work/tmstorma/Football2025/data_analysis/Class Distribution/class_counts.json')
with open(results_file, 'r') as f:
    data = json.load(f)

datasets = data['datasets']
totals = data['totals']

# Output directory
output_dir = Path('/cluster/work/tmstorma/Football2025/data_analysis/Class Distribution/visualizations')
output_dir.mkdir(exist_ok=True)

# Define colors for each class
colors = {
    'home_player': '#3498db',  # Blue
    'away_player': '#e74c3c',  # Red
    'referee': '#f39c12',      # Orange
    'ball': '#2ecc71'          # Green
}

print("="*80)
print("STEP 1b: GENERATING VISUALIZATIONS")
print("="*80)

# 1. STACKED BAR CHART - Per Dataset
print("\n1. Creating stacked bar chart...")

fig, ax = plt.subplots(figsize=(12, 6))

dataset_names = list(datasets.keys())
classes = ['home_player', 'away_player', 'referee', 'ball']

# Prepare data
data_matrix = []
for class_name in classes:
    data_matrix.append([datasets[ds][class_name] for ds in dataset_names])

# Create stacked bars
x = np.arange(len(dataset_names))
width = 0.6

bottom = np.zeros(len(dataset_names))
for i, class_name in enumerate(classes):
    ax.bar(x, data_matrix[i], width, label=class_name.replace('_', ' ').title(),
           bottom=bottom, color=colors[class_name])
    bottom += data_matrix[i]

ax.set_xlabel('Dataset', fontsize=12)
ax.set_ylabel('Number of Instances', fontsize=12)
ax.set_title('Class Distribution by Dataset', fontsize=14, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(dataset_names, rotation=45, ha='right')
ax.legend(loc='upper right')
ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
stacked_bar_path = output_dir / 'class_distribution_by_dataset.png'
plt.savefig(stacked_bar_path, dpi=300, bbox_inches='tight')
print(f"   Saved: {stacked_bar_path}")
plt.close()

# 2. PIE CHART - Overall Distribution
print("\n2. Creating pie chart...")

fig, ax = plt.subplots(figsize=(10, 8))

# Prepare data
labels = [cls.replace('_', ' ').title() for cls in classes]
sizes = [totals[cls] for cls in classes]
pie_colors = [colors[cls] for cls in classes]

# Create pie chart
wedges, texts, autotexts = ax.pie(sizes, labels=labels, colors=pie_colors,
                                    autopct='%1.1f%%', startangle=90,
                                    textprops={'fontsize': 12})

# Make percentage text bold
for autotext in autotexts:
    autotext.set_color('white')
    autotext.set_fontweight('bold')
    autotext.set_fontsize(11)

ax.set_title('Overall Class Distribution Across All Datasets',
             fontsize=14, fontweight='bold', pad=20)

# Add count information
total_count = sum(sizes)
info_text = f'Total Instances: {total_count:,}'
plt.figtext(0.5, 0.02, info_text, ha='center', fontsize=10, style='italic')

plt.tight_layout()
pie_chart_path = output_dir / 'overall_class_distribution.png'
plt.savefig(pie_chart_path, dpi=300, bbox_inches='tight')
print(f"   Saved: {pie_chart_path}")
plt.close()

# 3. PERCENTAGE STACKED BAR CHART - Shows relative proportions
print("\n3. Creating percentage stacked bar chart...")

fig, ax = plt.subplots(figsize=(12, 6))

# Calculate percentages
data_pct_matrix = []
for class_name in classes:
    row = []
    for ds in dataset_names:
        total = sum(datasets[ds].values())
        pct = (datasets[ds][class_name] / total * 100) if total > 0 else 0
        row.append(pct)
    data_pct_matrix.append(row)

# Create stacked bars
bottom = np.zeros(len(dataset_names))
for i, class_name in enumerate(classes):
    ax.bar(x, data_pct_matrix[i], width, label=class_name.replace('_', ' ').title(),
           bottom=bottom, color=colors[class_name])
    bottom += data_pct_matrix[i]

ax.set_xlabel('Dataset', fontsize=12)
ax.set_ylabel('Percentage (%)', fontsize=12)
ax.set_title('Class Distribution by Dataset (Percentage)', fontsize=14, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(dataset_names, rotation=45, ha='right')
ax.set_ylim(0, 100)
ax.legend(loc='upper right')
ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
pct_bar_path = output_dir / 'class_distribution_percentage.png'
plt.savefig(pct_bar_path, dpi=300, bbox_inches='tight')
print(f"   Saved: {pct_bar_path}")
plt.close()

# 4. GROUPED BAR CHART - Side-by-side comparison
print("\n4. Creating grouped bar chart...")

fig, ax = plt.subplots(figsize=(14, 7))

# Number of datasets and classes
n_datasets = len(dataset_names)
n_classes = len(classes)

# Bar width and positions
bar_width = 0.2
x = np.arange(n_datasets)

# Create bars for each class
for i, class_name in enumerate(classes):
    values = [datasets[ds][class_name] for ds in dataset_names]
    offset = (i - n_classes/2 + 0.5) * bar_width
    ax.bar(x + offset, values, bar_width,
           label=class_name.replace('_', ' ').title(),
           color=colors[class_name])

ax.set_xlabel('Dataset', fontsize=12, fontweight='bold')
ax.set_ylabel('Number of Instances', fontsize=12, fontweight='bold')
ax.set_title('Class Distribution Comparison Across Datasets', fontsize=14, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(dataset_names, rotation=45, ha='right')
ax.legend(loc='upper right', fontsize=10)
ax.grid(axis='y', alpha=0.3, linestyle='--')

plt.tight_layout()
grouped_bar_path = output_dir / 'class_distribution_grouped.png'
plt.savefig(grouped_bar_path, dpi=300, bbox_inches='tight')
print(f"   Saved: {grouped_bar_path}")
plt.close()

print("\n" + "="*80)
print("VISUALIZATION COMPLETE")
print(f"Output directory: {output_dir}")
print("="*80)
