#!/usr/bin/env python3
"""
Step 1b - Analyze Class Distribution and Balance
Identifies imbalances and underrepresented classes
"""

import json
from pathlib import Path

# Load results from counting step
results_file = Path('/cluster/work/tmstorma/Football2025/data_analysis/Class Distribution/class_counts.json')

with open(results_file, 'r') as f:
    data = json.load(f)

datasets = data['datasets']
totals = data['totals']

print("="*80)
print("STEP 1b: CLASS DISTRIBUTION ANALYSIS")
print("="*80)

# Analyze each dataset
print("\n" + "="*80)
print("DATASET-LEVEL ANALYSIS")
print("="*80)

issues = []

for dataset_name, counts in datasets.items():
    print(f"\n{dataset_name}:")

    dataset_total = sum(counts.values())

    # Check for missing classes
    missing_classes = [cls for cls, count in counts.items() if count == 0]

    if missing_classes:
        print(f"  ⚠️  WARNING: Missing classes: {', '.join(missing_classes)}")
        issues.append({
            'dataset': dataset_name,
            'issue': 'missing_classes',
            'classes': missing_classes
        })
    else:
        print(f"  ✓ All classes present")

        # Check balance among player classes
        home = counts['home_player']
        away = counts['away_player']
        referee = counts['referee']

        # Calculate ratios
        if home > 0 and away > 0:
            ratio = max(home, away) / min(home, away)
            if ratio > 1.5:
                print(f"  ⚠️  Imbalance: home/away ratio = {ratio:.2f}")
                issues.append({
                    'dataset': dataset_name,
                    'issue': 'player_imbalance',
                    'ratio': ratio
                })

# Overall analysis
print("\n" + "="*80)
print("OVERALL DISTRIBUTION")
print("="*80)

grand_total = sum(totals.values())

print(f"\nClass distribution across all datasets:")
for class_name, count in totals.items():
    pct = (count / grand_total * 100) if grand_total > 0 else 0
    print(f"  {class_name:15s}: {count:7d} ({pct:5.2f}%)")

# Identify underrepresented classes
print("\n" + "="*80)
print("UNDERREPRESENTATION ANALYSIS")
print("="*80)

# Expected ratios (approximate)
# In a typical match: ~22 players total, 2-3 referees, 1 ball
# Expected distribution should be roughly equal between home/away players
print("\nExpected vs Actual:")

total_players = totals['home_player'] + totals['away_player']
if total_players > 0:
    home_pct = (totals['home_player'] / total_players) * 100
    away_pct = (totals['away_player'] / total_players) * 100
    print(f"  Home players: {home_pct:.1f}% (expected ~50%)")
    print(f"  Away players: {away_pct:.1f}% (expected ~50%)")

    if abs(home_pct - 50) > 15:
        print("  ⚠️  Significant imbalance between home and away players")

# Ball should be ~3-5% (1 ball vs ~20-25 people)
ball_pct = (totals['ball'] / grand_total) * 100
print(f"\n  Ball: {ball_pct:.1f}% (expected ~3-5%)")
if ball_pct < 2 or ball_pct > 8:
    print("  ⚠️  Ball representation outside expected range")

# Summary of issues
print("\n" + "="*80)
print("SUMMARY OF ISSUES")
print("="*80)

if issues:
    print(f"\nFound {len(issues)} issue(s):\n")
    for i, issue in enumerate(issues, 1):
        if issue['issue'] == 'missing_classes':
            print(f"{i}. {issue['dataset']}: Missing {', '.join(issue['classes'])}")
            print(f"   → All players labeled as 'home' - needs team classification")
        elif issue['issue'] == 'player_imbalance':
            print(f"{i}. {issue['dataset']}: Home/away ratio = {issue['ratio']:.2f}")
else:
    print("\nNo significant issues found - dataset is well balanced")

# Recommendations
print("\n" + "="*80)
print("RECOMMENDATIONS")
print("="*80)

if any(issue['issue'] == 'missing_classes' for issue in issues):
    print("\n⚠️  CRITICAL: Some datasets lack team classification")
    print("   → RBK-VIKING and RBK-BODO-part3 need team labels")
    print("   → Options:")
    print("      1. Re-annotate with team labels")
    print("      2. Use a separate team classifier")
    print("      3. Exclude from training (not recommended)")
else:
    print("\n✓ Dataset is ready for training")

# Save analysis results
analysis_file = Path('/cluster/work/tmstorma/Football2025/data_analysis/Class Distribution/balance_analysis.json')
with open(analysis_file, 'w') as f:
    json.dump({
        'issues': issues,
        'overall_distribution': totals,
        'total_instances': grand_total
    }, f, indent=2)

print(f"\nAnalysis saved to: {analysis_file}")
print("="*80)
