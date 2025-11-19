#!/usr/bin/env python3
"""
Run HOTA evaluation using TrackEval library
"""

import sys
from pathlib import Path

# Add TrackEval to path
trackeval_path = Path('/cluster/work/tmstorma/Football2025/tracking/TrackEval')
sys.path.insert(0, str(trackeval_path))

import trackeval

def main():
    print("="*80)
    print("HOTA Evaluation - ByteTrack on Football Validation Set")
    print("="*80)

    # Configuration
    eval_config = {
        'USE_PARALLEL': False,
        'NUM_PARALLEL_CORES': 1,
        'BREAK_ON_ERROR': True,
        'RETURN_ON_ERROR': False,
        'LOG_ON_ERROR': '/cluster/work/tmstorma/Football2025/tracking/hota_data/error_log.txt',
        'PRINT_RESULTS': True,
        'PRINT_ONLY_COMBINED': False,
        'PRINT_CONFIG': True,
        'TIME_PROGRESS': True,
        'DISPLAY_LESS_PROGRESS': False,
        'OUTPUT_SUMMARY': True,
        'OUTPUT_EMPTY_CLASSES': True,
        'OUTPUT_DETAILED': True,
        'PLOT_CURVES': True,
    }

    # Dataset configuration
    dataset_config = {
        'GT_FOLDER': '/cluster/work/tmstorma/Football2025/tracking/hota_data/gt',
        'TRACKERS_FOLDER': '/cluster/work/tmstorma/Football2025/tracking/hota_data/trackers',
        'OUTPUT_FOLDER': '/cluster/work/tmstorma/Football2025/tracking/hota_results',
        'TRACKERS_TO_EVAL': ['ByteTrack'],
        'BENCHMARK': 'football',
        'SPLIT_TO_EVAL': 'val',
        'INPUT_AS_ZIP': False,
        'PRINT_CONFIG': True,
        'TRACKER_SUB_FOLDER': '',
        'OUTPUT_SUB_FOLDER': '',
        'TRACKER_DISPLAY_NAMES': None,
        'SEQMAP_FOLDER': None,
        'SEQMAP_FILE': None,
        'SEQ_INFO': {
            'RBK-AALESUND': {'name': 'RBK-AALESUND', 'fps': 25, 'seq_length': 180},
            'RBK-FREDRIKSTAD': {'name': 'RBK-FREDRIKSTAD', 'fps': 25, 'seq_length': 181},
            'RBK-HamKam': {'name': 'RBK-HamKam', 'fps': 25, 'seq_length': 152}
        },
        'GT_LOC_FORMAT': '{gt_folder}/{seq}/gt.txt',
        'SKIP_SPLIT_FOL': True,
    }

    # Metrics configuration
    metrics_config = {
        'METRICS': ['HOTA', 'CLEAR', 'Identity'],
        'THRESHOLD': 0.5,
        'PRINT_CONFIG': True,
    }

    print("\nRunning evaluation...")
    print(f"  Ground truth: {dataset_config['GT_FOLDER']}")
    print(f"  Predictions: {dataset_config['TRACKERS_FOLDER']}")
    print(f"  Output: {dataset_config['OUTPUT_FOLDER']}")
    print()

    # Create evaluator
    evaluator = trackeval.Evaluator(eval_config)

    # Use custom dataset that doesn't restrict to pedestrian class
    # We'll process as if all objects are "pedestrian" for TrackEval compatibility
    from trackeval.datasets._base_dataset import _BaseDataset

    class FootballDataset(_BaseDataset):
        """Custom dataset for football tracking evaluation"""

        @staticmethod
        def get_default_dataset_config():
            code_path = trackeval.utils.get_code_path()
            default_config = {
                'GT_FOLDER': None,
                'TRACKERS_FOLDER': None,
                'OUTPUT_FOLDER': None,
                'TRACKERS_TO_EVAL': None,
            }
            return default_config

        def __init__(self, config=None):
            super().__init__()
            self.config = {**self.get_default_dataset_config(), **config}
            self.gt_fol = self.config['GT_FOLDER']
            self.tracker_fol = self.config['TRACKERS_FOLDER']
            self.output_fol = self.config['OUTPUT_FOLDER']
            self.tracker_list = self.config['TRACKERS_TO_EVAL']
            self.output_sub_fol = ''
            self.seq_list = ['RBK-AALESUND', 'RBK-FREDRIKSTAD', 'RBK-HamKam']
            self.seq_lengths = {'RBK-AALESUND': 180, 'RBK-FREDRIKSTAD': 181, 'RBK-HamKam': 152}
            # Frame offsets: MOT frames start at these values, need to remap to 0-indexed
            self.seq_frame_offsets = {'RBK-AALESUND': 1623, 'RBK-FREDRIKSTAD': 1636, 'RBK-HamKam': 1372}
            self.class_list = ['all']  # Treat all objects as single class

        def _load_raw_file(self, tracker, seq, is_gt):
            import numpy as np

            if is_gt:
                file_path = Path(self.gt_fol) / seq / 'gt.txt'
            else:
                file_path = Path(self.tracker_fol) / tracker / seq / 'data.txt'

            num_timesteps = self.seq_lengths[seq]
            frame_offset = self.seq_frame_offsets[seq]

            # Initialize data structures for all timesteps
            if is_gt:
                data = {
                    'gt_ids': [np.array([], dtype=int) for _ in range(num_timesteps)],
                    'gt_dets': [[] for _ in range(num_timesteps)],
                    'gt_classes': [np.array([], dtype=int) for _ in range(num_timesteps)],
                    'gt_crowd_ignore_regions': [[] for _ in range(num_timesteps)],
                    'gt_extras': {},
                    'num_timesteps': num_timesteps,
                    'seq': seq
                }
            else:
                data = {
                    'tracker_ids': [np.array([], dtype=int) for _ in range(num_timesteps)],
                    'tracker_dets': [[] for _ in range(num_timesteps)],
                    'tracker_classes': [np.array([], dtype=int) for _ in range(num_timesteps)],
                    'tracker_confidences': [np.array([], dtype=float) for _ in range(num_timesteps)],
                    'num_timesteps': num_timesteps,
                    'seq': seq
                }

            if not file_path.exists():
                return data

            # Parse MOT format file
            with open(file_path, 'r') as f:
                for line in f:
                    parts = line.strip().split(',')
                    if len(parts) < 9:
                        continue

                    mot_frame = int(parts[0])
                    # Remap from MOT frame number to 0-indexed timestep
                    frame = mot_frame - frame_offset

                    if frame < 0 or frame >= num_timesteps:
                        continue

                    track_id = int(parts[1])
                    x = float(parts[2])
                    y = float(parts[3])
                    w = float(parts[4])
                    h = float(parts[5])
                    conf = float(parts[6])
                    class_id = int(parts[7])

                    bbox = [x, y, x+w, y+h]

                    if is_gt:
                        data['gt_ids'][frame] = np.append(data['gt_ids'][frame], track_id)
                        data['gt_dets'][frame].append(bbox)
                        data['gt_classes'][frame] = np.append(data['gt_classes'][frame], 1)  # All class 1
                    else:
                        data['tracker_ids'][frame] = np.append(data['tracker_ids'][frame], track_id)
                        data['tracker_dets'][frame].append(bbox)
                        data['tracker_classes'][frame] = np.append(data['tracker_classes'][frame], 1)  # All class 1
                        data['tracker_confidences'][frame] = np.append(data['tracker_confidences'][frame], conf)

            return data

        def get_display_name(self, tracker):
            return tracker

        def get_preprocessed_seq_data(self, raw_data, cls):
            """Preprocess data for evaluation - simplified for football tracking"""
            import numpy as np

            # Build ID remapping to make IDs contiguous from 0
            all_gt_ids = set()
            all_tracker_ids = set()
            for t in range(raw_data['num_timesteps']):
                all_gt_ids.update(raw_data['gt_ids'][t])
                all_tracker_ids.update(raw_data['tracker_ids'][t])

            # Create mapping: original_id -> new_id (0-indexed)
            gt_id_map = {old_id: new_id for new_id, old_id in enumerate(sorted(all_gt_ids))}
            tracker_id_map = {old_id: new_id for new_id, old_id in enumerate(sorted(all_tracker_ids))}

            # Since we treat all objects as one class, no filtering needed
            data_keys = ['gt_ids', 'tracker_ids', 'gt_dets', 'tracker_dets', 'tracker_confidences', 'similarity_scores']
            data = {key: [None] * raw_data['num_timesteps'] for key in data_keys}
            unique_gt_ids = []
            unique_tracker_ids = []
            num_gt_dets = 0
            num_tracker_dets = 0

            for t in range(raw_data['num_timesteps']):
                # Get data for this timestep
                gt_ids_raw = raw_data['gt_ids'][t]
                gt_dets = raw_data['gt_dets'][t]
                tracker_ids_raw = raw_data['tracker_ids'][t]
                tracker_dets = raw_data['tracker_dets'][t]
                tracker_confidences = raw_data['tracker_confidences'][t]
                similarity_scores = raw_data['similarity_scores'][t]

                # Remap IDs to be 0-indexed and contiguous
                gt_ids = np.array([gt_id_map[old_id] for old_id in gt_ids_raw], dtype=int)
                tracker_ids = np.array([tracker_id_map[old_id] for old_id in tracker_ids_raw], dtype=int)

                # Store data
                data['gt_ids'][t] = gt_ids
                data['gt_dets'][t] = gt_dets
                data['tracker_ids'][t] = tracker_ids
                data['tracker_dets'][t] = tracker_dets
                data['tracker_confidences'][t] = tracker_confidences
                data['similarity_scores'][t] = similarity_scores

                # Track unique IDs
                unique_gt_ids += list(np.unique(gt_ids))
                unique_tracker_ids += list(np.unique(tracker_ids))
                num_gt_dets += len(gt_ids)
                num_tracker_dets += len(tracker_ids)

            # Calculate summary statistics
            data['num_tracker_dets'] = num_tracker_dets
            data['num_gt_dets'] = num_gt_dets
            data['num_tracker_ids'] = len(set(unique_tracker_ids))
            data['num_gt_ids'] = len(set(unique_gt_ids))
            data['num_timesteps'] = raw_data['num_timesteps']
            data['seq'] = raw_data['seq']

            # Ensure unique IDs per timestep
            self._check_unique_ids(data, after_preproc=True)

            return data

        def _calculate_similarities(self, gt_dets_t, tracker_dets_t):
            import numpy as np
            similarity_scores = np.zeros((len(gt_dets_t), len(tracker_dets_t)))
            for g, gt_det in enumerate(gt_dets_t):
                for t, tr_det in enumerate(tracker_dets_t):
                    similarity_scores[g, t] = self._calculate_box_iou(gt_det, tr_det)
            return similarity_scores

        @staticmethod
        def _calculate_box_iou(box1, box2, box_format='x0y0x1y1'):
            import numpy as np
            if box_format == 'x0y0x1y1':
                x1_min, y1_min, x1_max, y1_max = box1
                x2_min, y2_min, x2_max, y2_max = box2
            else:
                raise ValueError('box_format not understood')

            x_min_inter = max(x1_min, x2_min)
            y_min_inter = max(y1_min, y2_min)
            x_max_inter = min(x1_max, x2_max)
            y_max_inter = min(y1_max, y2_max)

            intersection = max(0.0, x_max_inter - x_min_inter) * max(0.0, y_max_inter - y_min_inter)
            box1_area = (x1_max - x1_min) * (y1_max - y1_min)
            box2_area = (x2_max - x2_min) * (y2_max - y2_min)
            union = box1_area + box2_area - intersection

            return intersection / union if union > 0 else 0.0

    # Create dataset
    dataset_list = [FootballDataset(dataset_config)]

    # Create metrics
    metrics_list = []
    for metric in metrics_config['METRICS']:
        if metric == 'HOTA':
            metrics_list.append(trackeval.metrics.HOTA(metrics_config))
        elif metric == 'CLEAR':
            metrics_list.append(trackeval.metrics.CLEAR(metrics_config))
        elif metric == 'Identity':
            metrics_list.append(trackeval.metrics.Identity(metrics_config))

    # Run evaluation
    output_res, output_msg = evaluator.evaluate(dataset_list, metrics_list)

    print("\n" + "="*80)
    print("HOTA Evaluation Complete!")
    print("="*80)
    print(f"\nResults saved to: {dataset_config['OUTPUT_FOLDER']}")
    print("\nKey Metrics:")
    print("  - HOTA: Higher Order Tracking Accuracy (overall tracking quality)")
    print("  - DetA: Detection Accuracy")
    print("  - AssA: Association Accuracy")
    print("  - MOTA: Multiple Object Tracking Accuracy")
    print("  - IDF1: ID F1 Score")
    print("  - ID switches: Number of identity switches")

if __name__ == '__main__':
    main()
