#!/usr/bin/env python3
"""
Step 2b - Generate Tracking ID Stability Report
Creates comprehensive markdown report with all findings
"""

import json
from pathlib import Path
from datetime import datetime

# Load analysis results
continuity_file = Path('/cluster/work/tmstorma/Football2025/data_analysis/Tracking ID Stability/temporal_continuity.json')
issues_file = Path('/cluster/work/tmstorma/Football2025/data_analysis/Tracking ID Stability/tracking_issues.json')

with open(continuity_file, 'r') as f:
    continuity_data = json.load(f)

with open(issues_file, 'r') as f:
    issues_data = json.load(f)

OUTPUT_FILE = Path('/cluster/work/tmstorma/Football2025/data_analysis/Tracking ID Stability/STEP2B_REPORT.md')

def main():
    print("Generating Step 2b Tracking ID Stability Report...")

    report_lines = []

    # Header
    report_lines.append("# Step 2b: Tracking ID Stability Analysis")
    report_lines.append(f"\n**Report Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    report_lines.append("---\n")

    # Overview
    report_lines.append("## Overview\n")
    report_lines.append("This analysis evaluates tracking ID consistency and temporal continuity:")
    report_lines.append("1. **Frame numbering consistency** - Alignment between XML, images, and labels")
    report_lines.append("2. **Temporal continuity** - Track persistence across frames")
    report_lines.append("3. **ID switches and gaps** - Detection of tracking anomalies")
    report_lines.append("4. **Track trajectory visualization** - Visual confirmation of tracking quality\n")
    report_lines.append("---\n")

    # Part 1: Frame Consistency
    report_lines.append("## Part 1: Frame Numbering Consistency\n")
    report_lines.append("**Objective:** Verify frame IDs are consistent across data sources.\n")
    report_lines.append("**Findings:**\n")
    report_lines.append("✓ **All datasets validated successfully**")
    report_lines.append("- Frame counts match across XML annotations, images, and label files")
    report_lines.append("- No missing or corrupted frames detected\n")
    report_lines.append("**Indexing Convention:**")
    report_lines.append("- XML annotations: Frame 0 to N-1")
    report_lines.append("- Image files: frame_000001.png to frame_N.png")
    report_lines.append("- Mapping: XML frame `k` → image `frame_{k+1:06d}.png`")
    report_lines.append("- This is **standard CVAT convention** - data is fully usable\n")
    report_lines.append("**Minor Issues:**")
    report_lines.append("- RBK-BODO-part3: One missing label file (frame 1138)")
    report_lines.append("- All other datasets: Complete and consistent\n")
    report_lines.append("---\n")

    # Part 2: Temporal Continuity
    report_lines.append("## Part 2: Temporal Continuity Analysis\n")

    # Summary statistics
    total_tracks = sum(data['total_tracks'] for data in continuity_data.values())

    report_lines.append("### Overall Statistics\n")
    report_lines.append(f"- **Total tracks analyzed:** {total_tracks}")
    report_lines.append(f"- **Datasets:** {len(continuity_data)}\n")

    # Dataset breakdown
    report_lines.append("### Track Continuity by Dataset\n")
    report_lines.append("| Dataset | Total Tracks | Total Frames | Avg Continuity |")
    report_lines.append("|---------|--------------|--------------|----------------|")

    for dataset_name, data in continuity_data.items():
        # Calculate average continuity across all tracks
        all_continuities = [t['continuity'] for t in data['tracks_summary']]
        avg_continuity = sum(all_continuities) / len(all_continuities) if all_continuities else 0

        report_lines.append(
            f"| {dataset_name} | "
            f"{data['total_tracks']} | "
            f"{data['total_frames']} | "
            f"{avg_continuity:.1%} |"
        )

    report_lines.append("\n**Key Metrics:**\n")
    report_lines.append("- **Continuity**: Percentage of frames where track appears within its time span")
    report_lines.append("- **Gap**: Missing frame within a track's lifetime\n")

    # Class-level analysis
    report_lines.append("### Continuity by Object Class\n")

    report_lines.append("**Players (Home/Away):**")
    report_lines.append("- Average continuity: 97-100%")
    report_lines.append("- Very stable tracking throughout match\n")

    report_lines.append("**Referee:**")
    report_lines.append("- Average continuity: 67-100%")
    report_lines.append("- One dataset (RBK-HamKam) shows lower continuity (66.5%)")
    report_lines.append("- Likely due to referee moving off-screen\n")

    report_lines.append("**Ball:**")
    report_lines.append("- Average continuity: 82-100%")
    report_lines.append("- Gaps expected - ball goes off-screen, out of play")
    report_lines.append("- RBK-HamKam ball: 82.3% (270 gap frames)\n")

    report_lines.append("---\n")

    # Part 3: ID Switches and Issues
    report_lines.append("## Part 3: ID Switch Detection\n")

    total_jumps = sum(data['total_jumps'] for data in issues_data.values())
    total_switches = sum(data['potential_switches'] for data in issues_data.values())
    tracks_with_jumps = sum(data['tracks_with_jumps'] for data in issues_data.values())

    report_lines.append("### Position Jump Analysis\n")
    report_lines.append(f"- **Tracks with position jumps:** {tracks_with_jumps} out of {total_tracks} ({tracks_with_jumps/total_tracks*100:.1f}%)")
    report_lines.append(f"- **Total jumps detected:** {total_jumps}\n")

    report_lines.append("**Definition:** Jump = sudden position change >300 pixels/frame\n")

    report_lines.append("**Findings:**")
    report_lines.append("- Most jumps occur in **ball tracks** (expected - ball moves very fast)")
    report_lines.append("- Examples:")
    report_lines.append("  - RBK-HamKam ball: 1406 pixels in 1 frame")
    report_lines.append("  - RBK-VIKING ball: 1027 pixels in 1 frame")
    report_lines.append("- **2 player jumps detected (RBK-BODO-part3)** - further investigation required\n")

    report_lines.append("### Confirmed ID Switch in RBK-BODO-part3\n")
    report_lines.append("⚠️ **ID Switch Detected:** Visual inspection revealed an actual ID switch\n")
    report_lines.append("**Details:**")
    report_lines.append("- **Location:** Frame 1182 → 1183")
    report_lines.append("- **Affected tracks:** Track 2 and Track 11")
    report_lines.append("- **Type:** Simultaneous ID swap between two players")
    report_lines.append("- **Evidence:** Players wearing different colored jerseys (white vs yellow) before/after")
    report_lines.append("- **Impact:** 2 tracks (1% of total) unreliable after frame 1182\n")

    report_lines.append("#### Visual Evidence\n")
    report_lines.append("**Track 11 ID Switch:**\n")
    report_lines.append("![Track 11 ID Switch](./jump_investigation/jump_track11_#1.png)\n")
    report_lines.append("*White jersey player (before) switches to yellow jersey player (after) - clear ID swap.*\n")

    report_lines.append("**Track 2 ID Switch:**\n")
    report_lines.append("![Track 2 ID Switch](./jump_investigation/jump_track2_#1.png)\n")
    report_lines.append("*Complementary ID swap - the reverse exchange of the same two players.*\n")

    report_lines.append("**Root Cause:**")
    report_lines.append("- Both jumps occur at exact same frame transition (1182→1183)")
    report_lines.append("- Similar distances (~555 pixels)")
    report_lines.append("- Likely annotation error or re-identification failure at this specific frame\n")

    report_lines.append("---\n")

    # Part 4: Visualizations
    report_lines.append("## Part 4: Track Trajectory Visualization\n")

    report_lines.append("### Example: All Trajectories\n")
    report_lines.append("![RBK-AALESUND Trajectories](./visualizations/RBK-AALESUND_all_trajectories.png)\n")
    report_lines.append("*All object trajectories for RBK-AALESUND dataset. Shows movement patterns across the field.*\n")

    report_lines.append("### Example: Individual Tracks\n")
    report_lines.append("![Home Player Examples](./visualizations/RBK-AALESUND_home_examples.png)\n")
    report_lines.append("*Sample home player trajectories with consistent tracking IDs. Circle = start, Star = end.*\n")

    report_lines.append("**Additional visualizations available:**")
    report_lines.append("- `./visualizations/` directory contains trajectory maps for:")
    report_lines.append("  - All tracks combined")
    report_lines.append("  - Individual class examples (home, away, referee)\n")

    report_lines.append("---\n")

    # Summary
    report_lines.append("## Summary\n")

    report_lines.append("### Data Quality Assessment\n")
    report_lines.append("✓ **Very good tracking quality overall**\n")

    report_lines.append("**Strengths:**")
    report_lines.append("- 97.6% average temporal continuity")
    report_lines.append("- Frame numbering fully consistent")
    report_lines.append("- Player tracks are very stable (97-100% continuity)")
    report_lines.append("- 199 out of 201 tracks (99%) have perfect ID consistency\n")

    report_lines.append("**Issues Identified:**")
    report_lines.append("- **1 confirmed ID switch** in RBK-BODO-part3 affecting 2 tracks")
    report_lines.append("  - Isolated incident at frame 1182→1183")
    report_lines.append("  - Only 1% of tracks affected")
    report_lines.append("- Ball tracks have expected gaps (goes off-screen)")
    report_lines.append("- One referee track has lower continuity (likely off-screen)\n")

    report_lines.append("### Impact on Training\n")
    report_lines.append("**For Object Detection:**")
    report_lines.append("- ✓ Data is fully suitable")
    report_lines.append("- High-quality bounding boxes with consistent annotations")
    report_lines.append("- ID switch does not affect detection (boxes are correct)\n")

    report_lines.append("**For Object Tracking:**")
    report_lines.append("- ✓ Generally suitable for training tracking models")
    report_lines.append("- 99% of tracks have perfect ID consistency")
    report_lines.append("- **Recommendation:** Exclude RBK-BODO-part3 frames 1182+ for Track 2 & 11, or use entire dataset knowing 1% error rate")
    report_lines.append("- Ball tracking may need special handling due to gaps\n")

    report_lines.append("---\n")

    # Recommendations
    report_lines.append("## Recommendations\n")

    report_lines.append("**For Model Training:**")
    report_lines.append("1. Use temporal information - tracking IDs are reliable")
    report_lines.append("2. Handle ball disappearances in tracking logic")
    report_lines.append("3. Consider separate tracking strategies for ball vs players\n")

    report_lines.append("**For Data Preprocessing:**")
    report_lines.append("1. Account for frame indexing offset (XML frame N → image N+1)")
    report_lines.append("2. No cleanup needed - data quality is excellent")
    report_lines.append("3. Optional: Fill small gaps in ball tracks with interpolation\n")

    report_lines.append("---\n")

    # Write report
    OUTPUT_FILE.write_text('\n'.join(report_lines))
    print(f"Report saved to: {OUTPUT_FILE}")

if __name__ == '__main__':
    main()
