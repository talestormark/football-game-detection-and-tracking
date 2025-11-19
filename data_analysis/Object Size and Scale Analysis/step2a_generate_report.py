#!/usr/bin/env python3
"""
Step 2a - Generate Object Size and Scale Analysis Report
Creates comprehensive markdown report with visualizations and recommendations
"""

import json
from pathlib import Path
from datetime import datetime

# Load statistics
stats_file = Path('/cluster/work/tmstorma/Football2025/data_analysis/Object Size and Scale Analysis/size_statistics.json')
with open(stats_file, 'r') as f:
    data = json.load(f)

statistics = data['statistics']

OUTPUT_FILE = Path('/cluster/work/tmstorma/Football2025/data_analysis/Object Size and Scale Analysis/STEP2A_REPORT.md')

def main():
    print("Generating Step 2a Object Size and Scale Analysis Report...")

    report_lines = []

    # Header
    report_lines.append("# Step 2a: Object Size and Scale Analysis")
    report_lines.append(f"\n**Report Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    report_lines.append("---\n")

    # Overview
    report_lines.append("## Overview\n")
    report_lines.append("This analysis examines bounding box sizes and aspect ratios to inform model configuration:")
    report_lines.append("- **Anchor box sizes** for detection models")
    report_lines.append("- **Input resolution** requirements")
    report_lines.append("- **Data augmentation** strategies\n")
    report_lines.append("---\n")

    # Summary Statistics Table
    report_lines.append("## Summary Statistics\n")

    report_lines.append("| Class | Mean Area (px²) | Median Area (px²) | Mean Aspect Ratio | Median Aspect Ratio | Sample Count |")
    report_lines.append("|-------|----------------|------------------|-------------------|---------------------|--------------|")

    classes_order = ['home_player', 'away_player', 'referee', 'ball']
    class_labels = {
        'home_player': 'Home Player',
        'away_player': 'Away Player',
        'referee': 'Referee',
        'ball': 'Ball'
    }

    for cls in classes_order:
        if cls in statistics:
            stats = statistics[cls]
            report_lines.append(
                f"| {class_labels[cls]} | "
                f"{stats['area']['mean']:.0f} | "
                f"{stats['area']['median']:.0f} | "
                f"{stats['aspect_ratio']['mean']:.2f} | "
                f"{stats['aspect_ratio']['median']:.2f} | "
                f"{stats['area']['count']:,} |"
            )

    report_lines.append("\n---\n")

    # Visualizations
    report_lines.append("## Visual Analysis\n")

    report_lines.append("### Bounding Box Area Distribution\n")
    report_lines.append("![Area Distribution](./visualizations/area_distribution_boxplot.png)\n")
    report_lines.append("*Box plots showing the distribution of bounding box areas for each object class.*\n")

    report_lines.append("### Aspect Ratio Distribution\n")
    report_lines.append("![Aspect Ratio](./visualizations/aspect_ratio_histograms.png)\n")
    report_lines.append("*Histograms showing aspect ratio (width/height) distributions. Critical for anchor box design.*\n")
    report_lines.append("---\n")

    # Key Findings
    report_lines.append("## Key Findings\n")

    # Size hierarchy
    report_lines.append("### 1. Object Size Hierarchy\n")
    for cls in classes_order:
        if cls in statistics:
            area = statistics[cls]['area']['mean']
            rel_size = statistics[cls]['relative_area']['mean']
            report_lines.append(f"- **{class_labels[cls]}**: {area:.0f} px² (avg), {rel_size:.3f}% of image area")
    report_lines.append("")

    # Aspect ratios
    report_lines.append("### 2. Aspect Ratios\n")
    report_lines.append("- **Players & Referee**: ~0.45-0.47 (tall, narrow - approximately 2:1 height:width ratio)")
    report_lines.append("- **Ball**: ~1.0 (square/circular)\n")

    # Size variation
    report_lines.append("### 3. Size Variation\n")
    for cls in classes_order:
        if cls in statistics:
            min_area = statistics[cls]['area']['min']
            max_area = statistics[cls]['area']['max']
            ratio = max_area / min_area if min_area > 0 else 0
            report_lines.append(f"- **{class_labels[cls]}**: {ratio:.1f}x range (min: {min_area:.0f} px², max: {max_area:.0f} px²)")
    report_lines.append("")

    # Ball detection challenge
    if 'ball' in statistics:
        ball_area = statistics['ball']['area']['mean']
        player_area = statistics['home_player']['area']['mean']
        size_diff = player_area / ball_area
        report_lines.append("### 4. Ball Detection Challenge\n")
        report_lines.append(f"- Ball is **~{size_diff:.0f}x smaller** than players on average")
        report_lines.append(f"- Minimum ball area: {statistics['ball']['area']['min']:.0f} px² (very small object)")
        report_lines.append("- Requires **multi-scale detection** capabilities\n")

    report_lines.append("---\n")

    # Recommendations
    report_lines.append("## Recommendations for Model Configuration\n")

    report_lines.append("### 1. Anchor Box Sizes\n")
    report_lines.append("Based on aspect ratio analysis, recommended anchor configurations:\n")
    report_lines.append("**Players (Home/Away/Referee):**")
    report_lines.append("- Aspect ratios: **0.4, 0.45, 0.5**")
    report_lines.append("- Sizes (width × height): ")

    # Calculate recommended anchor sizes based on statistics
    if 'home_player' in statistics:
        median_w = statistics['home_player']['width']['median']
        median_h = statistics['home_player']['height']['median']
        report_lines.append(f"  - Small: ~{median_w*0.7:.0f} × {median_h*0.7:.0f}")
        report_lines.append(f"  - Medium: ~{median_w:.0f} × {median_h:.0f}")
        report_lines.append(f"  - Large: ~{median_w*1.5:.0f} × {median_h*1.5:.0f}\n")

    report_lines.append("**Ball:**")
    report_lines.append("- Aspect ratio: **1.0** (square)")
    if 'ball' in statistics:
        median_w = statistics['ball']['width']['median']
        median_h = statistics['ball']['height']['median']
        report_lines.append(f"- Size: ~{median_w:.0f} × {median_h:.0f}\n")

    report_lines.append("### 2. Input Resolution\n")
    report_lines.append("- **Recommended**: 1280×720 or higher")
    report_lines.append("- **Rationale**: Ball detection requires preserving small object details")
    report_lines.append(f"  - Minimum ball size is {statistics['ball']['area']['min']:.0f} px² at 1920×1080")
    report_lines.append("  - Lower resolutions may lose critical ball information\n")

    report_lines.append("### 3. Data Augmentation\n")
    report_lines.append("- **Scale augmentation**: ±30-50% (based on observed size variation)")
    report_lines.append("- **Aspect ratio**: Keep within 0.3-0.6 for players, 0.8-1.2 for ball")
    report_lines.append("- **Multi-scale training**: Essential for handling ball vs player size difference\n")

    report_lines.append("### 4. Model Architecture Considerations\n")
    report_lines.append("- Use **Feature Pyramid Network (FPN)** or similar multi-scale architecture")
    report_lines.append("- Enable **small object detection** layers")
    report_lines.append("- Consider separate detection heads for ball vs players due to size disparity\n")

    report_lines.append("---\n")

    # Summary
    report_lines.append("## Summary\n")
    report_lines.append("**Key Insights:**")
    report_lines.append("- Players have consistent aspect ratios (~0.45-0.47) suitable for tall anchor boxes")
    report_lines.append("- Ball is significantly smaller, requiring dedicated small-object detection strategies")
    report_lines.append("- Large size variation within classes (up to 450x for ball) necessitates multi-scale approach\n")
    report_lines.append("**Next Steps:**")
    report_lines.append("- Apply recommended anchor configurations in detection model")
    report_lines.append("- Use input resolution ≥720p to preserve small object details")
    report_lines.append("- Implement multi-scale data augmentation during training\n")

    report_lines.append("---\n")

    # Write report
    OUTPUT_FILE.write_text('\n'.join(report_lines))
    print(f"Report saved to: {OUTPUT_FILE}")

if __name__ == '__main__':
    main()
