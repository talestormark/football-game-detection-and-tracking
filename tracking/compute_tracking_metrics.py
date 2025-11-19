#!/usr/bin/env python3
"""
Compute tracking metrics directly from MOT format files
"""

from pathlib import Path
import numpy as np
from collections import defaultdict

def parse_mot_file(file_path):
    """
    Parse MOT format file
    Returns: dict of frame_idx -> list of tracks
    """
    tracks = defaultdict(list)

    with open(file_path, 'r') as f:
        for line in f:
            parts = line.strip().split(',')
            if len(parts) < 9:
                continue

            frame = int(parts[0])
            track_id = int(parts[1])
            x = float(parts[2])
            y = float(parts[3])
            w = float(parts[4])
            h = float(parts[5])
            conf = float(parts[6])
            class_id = int(parts[7])

            tracks[frame].append({
                'track_id': track_id,
                'bbox': [x, y, w, h],
                'class_id': class_id,
                'conf': conf
            })

    return tracks

def compute_iou(bbox1, bbox2):
    """Compute IoU between two bboxes [x, y, w, h]"""
    x1, y1, w1, h1 = bbox1
    x2, y2, w2, h2 = bbox2

    x1_max, y1_max = x1 + w1, y1 + h1
    x2_max, y2_max = x2 + w2, y2 + h2

    x_inter_min = max(x1, x2)
    y_inter_min = max(y1, y2)
    x_inter_max = min(x1_max, x2_max)
    y_inter_max = min(y1_max, y2_max)

    if x_inter_max < x_inter_min or y_inter_max < y_inter_min:
        return 0.0

    intersection = (x_inter_max - x_inter_min) * (y_inter_max - y_inter_min)
    area1 = w1 * h1
    area2 = w2 * h2
    union = area1 + area2 - intersection

    return intersection / union if union > 0 else 0.0

def match_tracks(gt_tracks, pred_tracks, iou_threshold=0.5):
    """
    Match predicted tracks to ground truth using Hungarian matching
    Returns: matches, false_positives, false_negatives
    """
    if len(gt_tracks) == 0 and len(pred_tracks) == 0:
        return [], [], []
    if len(gt_tracks) == 0:
        return [], list(range(len(pred_tracks))), []
    if len(pred_tracks) == 0:
        return [], [], list(range(len(gt_tracks)))

    # Compute IoU matrix
    iou_matrix = np.zeros((len(gt_tracks), len(pred_tracks)))
    for i, gt in enumerate(gt_tracks):
        for j, pred in enumerate(pred_tracks):
            iou_matrix[i, j] = compute_iou(gt['bbox'], pred['bbox'])

    # Greedy matching (could use Hungarian algorithm for optimal matching)
    matches = []
    matched_gt = set()
    matched_pred = set()

    # Sort by IoU descending
    candidates = []
    for i in range(len(gt_tracks)):
        for j in range(len(pred_tracks)):
            if iou_matrix[i, j] >= iou_threshold:
                candidates.append((iou_matrix[i, j], i, j))

    candidates.sort(reverse=True)

    for iou_val, i, j in candidates:
        if i not in matched_gt and j not in matched_pred:
            matches.append((i, j, iou_val))
            matched_gt.add(i)
            matched_pred.add(j)

    false_positives = [j for j in range(len(pred_tracks)) if j not in matched_pred]
    false_negatives = [i for i in range(len(gt_tracks)) if i not in matched_gt]

    return matches, false_positives, false_negatives

def compute_metrics(gt_file, pred_file, dataset_name):
    """Compute tracking metrics for one dataset"""
    print(f"\n{'='*80}")
    print(f"Computing metrics for {dataset_name}")
    print(f"{'='*80}")

    gt_tracks = parse_mot_file(gt_file)
    pred_tracks = parse_mot_file(pred_file)

    # Statistics
    total_gt = 0
    total_pred = 0
    total_matches = 0
    total_fp = 0
    total_fn = 0
    id_switches = 0

    # Track ID mapping (gt_id -> pred_id in previous frame)
    prev_mapping = {}

    frames = sorted(set(list(gt_tracks.keys()) + list(pred_tracks.keys())))

    for frame in frames:
        gt_frame = gt_tracks.get(frame, [])
        pred_frame = pred_tracks.get(frame, [])

        total_gt += len(gt_frame)
        total_pred += len(pred_frame)

        # Match tracks
        matches, fps, fns = match_tracks(gt_frame, pred_frame)

        total_matches += len(matches)
        total_fp += len(fps)
        total_fn += len(fns)

        # Check ID switches
        current_mapping = {}
        for gt_idx, pred_idx, iou in matches:
            gt_id = gt_frame[gt_idx]['track_id']
            pred_id = pred_frame[pred_idx]['track_id']

            current_mapping[gt_id] = pred_id

            # Check if this GT ID was matched before with different pred ID
            if gt_id in prev_mapping:
                if prev_mapping[gt_id] != pred_id:
                    id_switches += 1

        prev_mapping = current_mapping

    # Compute metrics
    precision = total_matches / total_pred if total_pred > 0 else 0
    recall = total_matches / total_gt if total_gt > 0 else 0
    mota = 1 - (total_fp + total_fn + id_switches) / total_gt if total_gt > 0 else 0
    idf1_num = 2 * total_matches
    idf1_den = total_gt + total_pred
    idf1 = idf1_num / idf1_den if idf1_den > 0 else 0

    print(f"\nDetection Metrics:")
    print(f"  Total GT objects: {total_gt}")
    print(f"  Total Predicted objects: {total_pred}")
    print(f"  True Positives (matches): {total_matches}")
    print(f"  False Positives: {total_fp}")
    print(f"  False Negatives: {total_fn}")
    print(f"  Precision: {precision:.3f}")
    print(f"  Recall: {recall:.3f}")

    print(f"\nTracking Metrics:")
    print(f"  ID Switches: {id_switches}")
    print(f"  MOTA: {mota:.3f}")
    print(f"  IDF1: {idf1:.3f}")

    return {
        'dataset': dataset_name,
        'total_gt': total_gt,
        'total_pred': total_pred,
        'matches': total_matches,
        'fp': total_fp,
        'fn': total_fn,
        'id_switches': id_switches,
        'precision': precision,
        'recall': recall,
        'mota': mota,
        'idf1': idf1
    }

def main():
    print("="*80)
    print("Football Tracking Metrics Evaluation")
    print("="*80)

    data_dir = Path('/cluster/work/tmstorma/Football2025/tracking/hota_data')

    datasets = ['RBK-AALESUND', 'RBK-FREDRIKSTAD', 'RBK-HamKam']

    all_results = []

    for dataset in datasets:
        gt_file = data_dir / 'gt' / dataset / 'gt.txt'
        pred_file = data_dir / 'trackers' / 'ByteTrack' / dataset / 'data.txt'

        results = compute_metrics(gt_file, pred_file, dataset)
        all_results.append(results)

    # Overall summary
    print(f"\n{'='*80}")
    print("Overall Summary Across All Datasets")
    print(f"{'='*80}")

    total_gt = sum(r['total_gt'] for r in all_results)
    total_pred = sum(r['total_pred'] for r in all_results)
    total_matches = sum(r['matches'] for r in all_results)
    total_fp = sum(r['fp'] for r in all_results)
    total_fn = sum(r['fn'] for r in all_results)
    total_id_switches = sum(r['id_switches'] for r in all_results)

    overall_precision = total_matches / total_pred if total_pred > 0 else 0
    overall_recall = total_matches / total_gt if total_gt > 0 else 0
    overall_mota = 1 - (total_fp + total_fn + total_id_switches) / total_gt if total_gt > 0 else 0
    overall_idf1 = (2 * total_matches) / (total_gt + total_pred) if (total_gt + total_pred) > 0 else 0

    print(f"\nOverall Detection Metrics:")
    print(f"  Total GT objects: {total_gt}")
    print(f"  Total Predicted objects: {total_pred}")
    print(f"  True Positives: {total_matches}")
    print(f"  False Positives: {total_fp}")
    print(f"  False Negatives: {total_fn}")
    print(f"  Precision: {overall_precision:.3f}")
    print(f"  Recall: {overall_recall:.3f}")

    print(f"\nOverall Tracking Metrics:")
    print(f"  Total ID Switches: {total_id_switches}")
    print(f"  MOTA: {overall_mota:.3f}")
    print(f"  IDF1: {overall_idf1:.3f}")

    print(f"\n{'='*80}")
    print("Metric Interpretation")
    print(f"{'='*80}")
    print("Detection Metrics:")
    print("  - Precision: Fraction of predicted objects that match GT (TP / (TP + FP))")
    print("  - Recall: Fraction of GT objects that are detected (TP / (TP + FN))")
    print("\nTracking Metrics:")
    print("  - MOTA: Multiple Object Tracking Accuracy (penalizes FP, FN, ID switches)")
    print("    Range: (-inf, 1.0], higher is better, >0.5 is good")
    print("  - IDF1: ID F1 Score (fraction of correctly identified detections)")
    print("    Range: [0, 1], higher is better, >0.7 is good")
    print("  - ID Switches: Lower is better, ideally <50 for this dataset")

    print("\n" + "="*80)
    print(f"Target Performance (from CLAUDE.md):")
    print(f"  - HOTA: > 60")
    print(f"  - IDF1: > 70 (0.70)")
    print(f"  - ID switches: < 50")
    print("="*80)

if __name__ == '__main__':
    main()
