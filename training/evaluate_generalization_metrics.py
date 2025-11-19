#!/usr/bin/env python3
"""
Compute detection metrics (Precision, Recall, mAP) for generalization datasets
with merged player class (home+away+referee → player)
"""

from ultralytics import YOLO
from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np
import json

def parse_xml_annotations(xml_path, frame_number):
    """Parse XML to get ground truth boxes for a specific frame"""
    tree = ET.parse(xml_path)
    root = tree.getroot()

    boxes = []
    for track in root.findall('.//track'):
        label = track.get('label')
        for box in track.findall('box'):
            if int(box.get('frame')) == frame_number:
                xtl = float(box.get('xtl'))
                ytl = float(box.get('ytl'))
                xbr = float(box.get('xbr'))
                ybr = float(box.get('ybr'))

                # Map label to merged class
                # For VIKING/BODO: label is "player" or "ball"
                # All players (home/away/referee) are labeled as "player"
                if label == 'player':
                    merged_label = 'player'
                elif label == 'ball':
                    merged_label = 'ball'
                else:
                    # For other datasets with separate labels
                    if label in ['home', 'away', 'referee', 'home_player']:
                        merged_label = 'player'
                    else:
                        continue

                boxes.append({
                    'label': merged_label,
                    'bbox': [xtl, ytl, xbr, ybr]
                })

    return boxes

def bbox_iou(box1, box2):
    """Calculate IoU between two boxes [xtl, ytl, xbr, ybr]"""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - intersection

    return intersection / union if union > 0 else 0

def compute_metrics(predictions, ground_truths, iou_threshold=0.5):
    """Compute precision, recall for a set of predictions"""
    if len(ground_truths) == 0:
        return 0.0, 0.0, 0, 0, 0

    if len(predictions) == 0:
        return 0.0, 0.0, 0, len(ground_truths), 0

    # Match predictions to ground truth
    matched_gt = set()
    true_positives = 0

    for pred in predictions:
        best_iou = 0
        best_gt_idx = -1

        for gt_idx, gt in enumerate(ground_truths):
            if gt_idx in matched_gt:
                continue
            if gt['label'] != pred['label']:
                continue

            iou = bbox_iou(pred['bbox'], gt['bbox'])
            if iou > best_iou:
                best_iou = iou
                best_gt_idx = gt_idx

        if best_iou >= iou_threshold and best_gt_idx >= 0:
            true_positives += 1
            matched_gt.add(best_gt_idx)

    false_positives = len(predictions) - true_positives
    false_negatives = len(ground_truths) - true_positives

    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0

    return precision, recall, true_positives, false_positives, false_negatives

def main():
    print("="*80)
    print("Generalization Dataset Evaluation - Detection Metrics")
    print("="*80)

    # Paths
    model_path = Path('/cluster/work/tmstorma/Football2025/training/runs/yolov8s_4class2/weights/best.pt')

    datasets = [
        {
            'name': 'RBK-VIKING',
            'xml': '/cluster/projects/vc/courses/TDT17/other/Football2025/RBK-VIKING/annotations.xml',
            'img_dir': '/cluster/projects/vc/courses/TDT17/other/Football2025/RBK-VIKING/data/images/train',
            'frames': [99, 499]  # XML frame numbers (0-indexed)
        },
        {
            'name': 'RBK-BODO-part3',
            'xml': '/cluster/projects/vc/courses/TDT17/other/Football2025/RBK-BODO/part3/RBK_BODO_PART3/annotations.xml',
            'img_dir': '/cluster/projects/vc/courses/TDT17/other/Football2025/RBK-BODO/part3/RBK_BODO_PART3/data/images/train',
            'frames': [99, 499]
        }
    ]

    # Load model
    print("\nLoading model...")
    model = YOLO(str(model_path))

    all_results = []

    for dataset in datasets:
        print(f"\n{'='*80}")
        print(f"Processing {dataset['name']}")
        print(f"{'='*80}")

        for frame_idx in dataset['frames']:
            img_path = Path(dataset['img_dir']) / f"frame_{frame_idx+1:06d}.png"

            print(f"\nFrame {frame_idx+1:06d}:")

            # Get ground truth
            gt_boxes = parse_xml_annotations(dataset['xml'], frame_idx)
            gt_player = [b for b in gt_boxes if b['label'] == 'player']
            gt_ball = [b for b in gt_boxes if b['label'] == 'ball']

            print(f"  Ground truth: {len(gt_player)} players, {len(gt_ball)} balls")

            # Run inference
            results = model.predict(
                source=str(img_path),
                conf=0.25,
                iou=0.7,
                verbose=False
            )

            result = results[0]
            boxes = result.boxes

            # Convert predictions to merged classes
            pred_boxes = []
            if len(boxes) > 0:
                for i in range(len(boxes)):
                    cls_id = int(boxes.cls[i].cpu().numpy())
                    conf = float(boxes.conf[i].cpu().numpy())
                    xyxy = boxes.xyxy[i].cpu().numpy()

                    # Merge classes: 0=home, 1=away, 2=referee → player; 3=ball → ball
                    if cls_id in [0, 1, 2]:
                        label = 'player'
                    elif cls_id == 3:
                        label = 'ball'
                    else:
                        continue

                    pred_boxes.append({
                        'label': label,
                        'bbox': [float(xyxy[0]), float(xyxy[1]), float(xyxy[2]), float(xyxy[3])],
                        'conf': conf
                    })

            pred_player = [b for b in pred_boxes if b['label'] == 'player']
            pred_ball = [b for b in pred_boxes if b['label'] == 'ball']

            print(f"  Predictions: {len(pred_player)} players, {len(pred_ball)} balls")

            # Compute metrics for mAP@0.5
            prec_player_50, rec_player_50, tp_p_50, fp_p_50, fn_p_50 = compute_metrics(pred_player, gt_player, iou_threshold=0.5)
            prec_ball_50, rec_ball_50, tp_b_50, fp_b_50, fn_b_50 = compute_metrics(pred_ball, gt_ball, iou_threshold=0.5)

            # Compute metrics for mAP@0.75 (for 0.5:0.95 approximation)
            prec_player_75, rec_player_75, _, _, _ = compute_metrics(pred_player, gt_player, iou_threshold=0.75)
            prec_ball_75, rec_ball_75, _, _, _ = compute_metrics(pred_ball, gt_ball, iou_threshold=0.75)

            # Approximate mAP@0.5:0.95 as average of mAP@0.5 and mAP@0.75
            # (True mAP@0.5:0.95 requires computing at 0.5, 0.55, 0.6, ..., 0.95)
            map50_player = prec_player_50 if rec_player_50 > 0 else 0
            map50_ball = prec_ball_50 if rec_ball_50 > 0 else 0
            map50_95_player = (prec_player_50 + prec_player_75) / 2 if (rec_player_50 > 0 or rec_player_75 > 0) else 0
            map50_95_ball = (prec_ball_50 + prec_ball_75) / 2 if (rec_ball_50 > 0 or rec_ball_75 > 0) else 0

            result_data = {
                'dataset': dataset['name'],
                'frame': f"frame_{frame_idx+1:06d}",
                'gt_player': len(gt_player),
                'gt_ball': len(gt_ball),
                'pred_player': len(pred_player),
                'pred_ball': len(pred_ball),
                'player': {
                    'precision': round(prec_player_50, 3),
                    'recall': round(rec_player_50, 3),
                    'mAP50': round(map50_player, 3),
                    'mAP50_95': round(map50_95_player, 3),
                    'TP': tp_p_50,
                    'FP': fp_p_50,
                    'FN': fn_p_50
                },
                'ball': {
                    'precision': round(prec_ball_50, 3),
                    'recall': round(rec_ball_50, 3),
                    'mAP50': round(map50_ball, 3),
                    'mAP50_95': round(map50_95_ball, 3),
                    'TP': tp_b_50,
                    'FP': fp_b_50,
                    'FN': fn_b_50
                }
            }

            all_results.append(result_data)

            print(f"  Player - Precision: {prec_player_50:.3f}, Recall: {rec_player_50:.3f}")
            print(f"  Ball   - Precision: {prec_ball_50:.3f}, Recall: {rec_ball_50:.3f}")

    # Print summary table
    print("\n" + "="*80)
    print("\nDETECTION METRICS SUMMARY (Merged Classes)")
    print("="*80)
    print("\n## Player Detection Metrics\n")
    print(f"{'Dataset':<20} {'Frame':<18} {'Precision':<12} {'Recall':<10} {'mAP@0.5':<10} {'mAP@0.5:0.95':<12}")
    print("-" * 80)

    for r in all_results:
        print(f"{r['dataset']:<20} {r['frame']:<18} {r['player']['precision']:<12.3f} {r['player']['recall']:<10.3f} {r['player']['mAP50']:<10.3f} {r['player']['mAP50_95']:<12.3f}")

    print("\n## Ball Detection Metrics\n")
    print(f"{'Dataset':<20} {'Frame':<18} {'Precision':<12} {'Recall':<10} {'mAP@0.5':<10} {'mAP@0.5:0.95':<12}")
    print("-" * 80)

    for r in all_results:
        print(f"{r['dataset']:<20} {r['frame']:<18} {r['ball']['precision']:<12.3f} {r['ball']['recall']:<10.3f} {r['ball']['mAP50']:<10.3f} {r['ball']['mAP50_95']:<12.3f}")

    # Calculate averages
    avg_player_prec = np.mean([r['player']['precision'] for r in all_results])
    avg_player_rec = np.mean([r['player']['recall'] for r in all_results])
    avg_player_map50 = np.mean([r['player']['mAP50'] for r in all_results])
    avg_player_map50_95 = np.mean([r['player']['mAP50_95'] for r in all_results])

    avg_ball_prec = np.mean([r['ball']['precision'] for r in all_results])
    avg_ball_rec = np.mean([r['ball']['recall'] for r in all_results])
    avg_ball_map50 = np.mean([r['ball']['mAP50'] for r in all_results])
    avg_ball_map50_95 = np.mean([r['ball']['mAP50_95'] for r in all_results])

    print("\n" + "="*80)
    print("\n## Average Metrics (All Frames)\n")
    print(f"{'Class':<10} {'Precision':<12} {'Recall':<10} {'mAP@0.5':<10} {'mAP@0.5:0.95':<12}")
    print("-" * 80)
    print(f"{'Player':<10} {avg_player_prec:<12.3f} {avg_player_rec:<10.3f} {avg_player_map50:<10.3f} {avg_player_map50_95:<12.3f}")
    print(f"{'Ball':<10} {avg_ball_prec:<12.3f} {avg_ball_rec:<10.3f} {avg_ball_map50:<10.3f} {avg_ball_map50_95:<12.3f}")

    # Save to JSON
    summary = {
        'detailed_results': all_results,
        'average_metrics': {
            'player': {
                'precision': round(avg_player_prec, 3),
                'recall': round(avg_player_rec, 3),
                'mAP50': round(avg_player_map50, 3),
                'mAP50_95': round(avg_player_map50_95, 3)
            },
            'ball': {
                'precision': round(avg_ball_prec, 3),
                'recall': round(avg_ball_rec, 3),
                'mAP50': round(avg_ball_map50, 3),
                'mAP50_95': round(avg_ball_map50_95, 3)
            }
        }
    }

    output_file = Path('/cluster/work/tmstorma/Football2025/training/inference_generalization/metrics.json')
    with open(output_file, 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\n{'='*80}")
    print(f"\nMetrics saved to: {output_file}")
    print("\n" + "="*80)
    print("\nNote: mAP@0.5:0.95 is approximated as average of mAP@0.5 and mAP@0.75")
    print("For precise mAP@0.5:0.95, would need to compute at all IoU thresholds 0.5-0.95")

if __name__ == '__main__':
    main()
