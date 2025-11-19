#!/usr/bin/env python3
"""
Step 1b - Generate Class Distribution Report
Creates comprehensive markdown report with visualizations
"""

import json
from pathlib import Path
from datetime import datetime

# Load analysis results
results_file = Path('/cluster/work/tmstorma/Football2025/data_analysis/Class Distribution/class_counts.json')
analysis_file = Path('/cluster/work/tmstorma/Football2025/data_analysis/Class Distribution/balance_analysis.json')

with open(results_file, 'r') as f:
    data = json.load(f)

with open(analysis_file, 'r') as f:
    analysis = json.load(f)

datasets = data['datasets']
totals = data['totals']
issues = analysis['issues']

OUTPUT_FILE = Path('/cluster/work/tmstorma/Football2025/data_analysis/Class Distribution/STEP1B_REPORT.md')

def main():
    print("Generating Step 1b Class Distribution Report...")

    report_lines = []

    # Header
    report_lines.append("# Step 1b: Class Distribution Analysis")
    report_lines.append(f"\n**Report Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    report_lines.append("---\n")

    # Overview
    report_lines.append("## Overview\n")
    report_lines.append("This report analyzes the distribution of annotated instances across object classes:")
    report_lines.append("- `home_player` - Home team players")
    report_lines.append("- `away_player` - Away team players")
    report_lines.append("- `referee` - Match referees")
    report_lines.append("- `ball` - Football\n")
    report_lines.append("---\n")

    # Class Distribution by Dataset
    report_lines.append("## Class Distribution by Dataset\n")

    # Table
    report_lines.append("| Dataset | Home Player | Away Player | Referee | Ball | Total |")
    report_lines.append("|---------|-------------|-------------|---------|------|-------|")

    for dataset_name, counts in datasets.items():
        total = sum(counts.values())
        report_lines.append(
            f"| {dataset_name} | "
            f"{counts['home_player']:,} | "
            f"{counts['away_player']:,} | "
            f"{counts['referee']:,} | "
            f"{counts['ball']:,} | "
            f"{total:,} |"
        )

    # Totals row
    grand_total = sum(totals.values())
    report_lines.append(
        f"| **TOTAL** | "
        f"**{totals['home_player']:,}** | "
        f"**{totals['away_player']:,}** | "
        f"**{totals['referee']:,}** | "
        f"**{totals['ball']:,}** | "
        f"**{grand_total:,}** |"
    )
    report_lines.append("\n---\n")

    # Visualizations
    report_lines.append("## Visual Analysis\n")

    report_lines.append("### Class Distribution Comparison\n")
    report_lines.append("![Class Distribution Grouped](./visualizations/class_distribution_grouped.png)\n")
    report_lines.append("*Grouped bar chart showing class counts per dataset. Missing bars indicate absent classes.*\n")

    report_lines.append("### Overall Distribution\n")
    report_lines.append("![Overall Distribution](./visualizations/overall_class_distribution.png)\n")
    report_lines.append("*Pie chart showing overall class distribution across all datasets.*\n")
    report_lines.append("---\n")

    # Key Findings
    report_lines.append("## Key Findings\n")

    # Overall percentages
    report_lines.append("### Overall Class Distribution\n")
    for class_name in ['home_player', 'away_player', 'referee', 'ball']:
        count = totals[class_name]
        pct = (count / grand_total * 100) if grand_total > 0 else 0
        report_lines.append(f"- **{class_name.replace('_', ' ').title()}**: {count:,} instances ({pct:.2f}%)")
    report_lines.append("")

    # Player balance
    total_players = totals['home_player'] + totals['away_player']
    if total_players > 0:
        home_pct = (totals['home_player'] / total_players) * 100
        away_pct = (totals['away_player'] / total_players) * 100
        report_lines.append("### Player Team Balance\n")
        report_lines.append(f"- **Home players**: {home_pct:.1f}% (expected ~50%)")
        report_lines.append(f"- **Away players**: {away_pct:.1f}% (expected ~50%)")

        if abs(home_pct - 50) > 15:
            report_lines.append(f"- ⚠️ **Significant imbalance**: {abs(home_pct - 50):.1f}% deviation from expected 50/50 split\n")
        else:
            report_lines.append(f"- ✓ **Balanced**: Within acceptable range\n")

    report_lines.append("---\n")

    # Issues
    report_lines.append("## Issues Identified\n")

    if issues:
        report_lines.append(f"**Found {len(issues)} issue(s):**\n")

        for i, issue in enumerate(issues, 1):
            if issue['issue'] == 'missing_classes':
                report_lines.append(f"### {i}. {issue['dataset']}: Missing Class Labels\n")
                report_lines.append(f"**Missing classes**: {', '.join(issue['classes'])}\n")
                report_lines.append("**Impact**: All players are labeled as 'home' team. This dataset lacks team distinction.\n")
                report_lines.append("**Recommendation**: ")
                report_lines.append("- Option 1: Re-annotate with proper team labels")
                report_lines.append("- Option 2: Train a separate team classifier")
                report_lines.append("- Option 3: Use only for player detection (not team classification)\n")
    else:
        report_lines.append("✓ No significant issues found.\n")

    report_lines.append("---\n")

    # Summary
    report_lines.append("## Summary\n")

    # Count datasets with issues
    datasets_with_issues = len([i for i in issues if i['issue'] == 'missing_classes'])
    datasets_ok = len(datasets) - datasets_with_issues

    if datasets_with_issues > 0:
        report_lines.append(f"**Status**: ⚠️ Issues detected in {datasets_with_issues} out of {len(datasets)} datasets\n")
        report_lines.append(f"**Datasets with complete labels**: {datasets_ok}")
        report_lines.append(f"**Datasets missing team labels**: {datasets_with_issues}\n")
        report_lines.append("**Critical Issue**: RBK-VIKING and RBK-BODO-part3 lack team classification (away_player, referee).\n")
        report_lines.append("**Impact on Training**:")
        report_lines.append("- These datasets can be used for generic player detection")
        report_lines.append("- Cannot be used for team classification without additional processing")
        report_lines.append("- Overall dataset has 69% home vs 31% away imbalance\n")
    else:
        report_lines.append("**Status**: ✓ All datasets have balanced class distribution\n")

    # Next Steps
    report_lines.append("---\n")
    report_lines.append("## Recommendations\n")

    if datasets_with_issues > 0:
        report_lines.append("1. **For complete team classification**: Re-annotate RBK-VIKING and RBK-BODO-part3 with team labels")
        report_lines.append("2. **For object detection only**: Current annotations are sufficient for detecting players, referees, and ball")
        report_lines.append("3. **Hybrid approach**: Use well-labeled datasets (AALESUND, FREDRIKSTAD, HamKam) for team classification, all datasets for object detection\n")
    else:
        report_lines.append("Dataset is ready for training with full team classification capability.\n")

    report_lines.append("---\n")
    report_lines.append("\n## Next Steps\n")
    report_lines.append("Proceed to model training or optional analyses (object size, tracking stability).\n")

    # Write report
    OUTPUT_FILE.write_text('\n'.join(report_lines))
    print(f"Report saved to: {OUTPUT_FILE}")

if __name__ == '__main__':
    main()
