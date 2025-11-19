# Object Tracking - Step 3.3

## Overview

This directory contains the implementation of ByteTrack for assigning consistent tracking IDs to detected objects across video frames.

**Status:** ✅ **COMPLETED**

## Final Results

### HOTA Evaluation (All Datasets Combined)

**Primary Metrics:**
- **HOTA: 81.45%** ✅ (Target: > 60%) - **SIGNIFICANTLY EXCEEDED**
- **DetA (Detection Accuracy): 81.48%**
- **AssA (Association Accuracy): 81.48%**
- **Localization Accuracy: 87.40%**

**CLEAR/MOT Metrics:**
- **MOTA: 96.26%** - Excellent tracking accuracy
- **Detection Recall: 97.02%**
- **Detection Precision: 99.42%**

**Identity Metrics:**
- **IDF1: 95.47%** ✅ (Target: > 70%) - **SIGNIFICANTLY EXCEEDED**
- **ID Switches: 23** ✅ (Target: < 50) - **WELL BELOW TARGET**

### Per-Dataset Performance

All three validation datasets evaluate correctly:
- **RBK-AALESUND** (180 frames, 1920×1080)
- **RBK-FREDRIKSTAD** (181 frames, 1280×720) - Resolution fix applied
- **RBK-HamKam** (152 frames, 1920×1080)

**Total:** 11,742 ground truth objects, 11,458 predictions, 11,392 matches (97% recall)

## Critical Fix Applied

**Issue:** RBK-FREDRIKSTAD dataset had 1280×720 resolution, but evaluation assumed 1920×1080 for all datasets.

**Solution:** Modified `prepare_hota_data.py` to handle dataset-specific resolutions when converting normalized YOLO coordinates to pixel coordinates.

**Location:** `prepare_hota_data.py:173-177` - Added resolution parameters per dataset.

## Directory Structure

```
tracking/
├── README.md                           # This file
├── run_tracking_validation.py          # Run tracking on validation set
├── run_tracking_generalization.py      # Run tracking on test sets
├── run_tracking_slurm.sh               # SLURM job for tracking
├── prepare_hota_data.py                # Convert tracking outputs to MOT format
├── run_hota_evaluation.py              # HOTA evaluation script
├── run_hota_slurm.sh                   # SLURM job for HOTA evaluation
├── compute_tracking_metrics.py         # Simplified metrics (MOTA, IDF1)
├── runs/
│   └── val_tracking/                   # Final tracking outputs
│       ├── RBK-AALESUND/
│       ├── RBK-FREDRIKSTAD/
│       └── RBK-HamKam/
├── hota_data/                          # MOT format data for evaluation
│   ├── gt/                             # Ground truth annotations
│   └── trackers/
│       └── ByteTrack/                  # Tracking predictions
├── hota_results/                       # Final HOTA evaluation results
│   └── ByteTrack/
│       ├── all_summary.txt             # Summary metrics
│       ├── all_detailed.csv            # Detailed per-threshold results
│       └── all_plot.png                # Visualization
├── logs/                               # SLURM job logs
│   ├── 23300310_hota_output.txt        # Final successful HOTA run
│   └── 23300310_hota_error.err
└── TrackEval/                          # TrackEval library (modified)
```

## ByteTrack Configuration

Based on Step 2b temporal continuity analysis:

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| `track_high_thresh` | 0.5 | High confidence for starting/matching tracks |
| `track_low_thresh` | 0.3 | Low confidence for recovering occluded tracks |
| `track_buffer` | 30 | Keep lost tracks for 30 frames (1.2s @ 25fps) |
| `match_thresh` | 0.8 | High IoU threshold for precise matching |

## Usage

### 1. Run Tracking on Validation Set

```bash
cd /cluster/work/tmstorma/Football2025/tracking
sbatch run_tracking_slurm.sh
```

**Output:** `runs/val_tracking/` with tracked images and MOT format labels

### 2. Evaluate Tracking with HOTA

```bash
sbatch run_hota_slurm.sh
```

**Output:** `hota_results/ByteTrack/` with comprehensive evaluation metrics

### 3. Quick Metrics Check (No SLURM)

```bash
python compute_tracking_metrics.py
```

Computes MOTA, IDF1, precision, recall directly from MOT files.

## Key Scripts

### `run_tracking_validation.py`
Runs ByteTrack on validation frames using trained YOLOv8 model.
- Input: Trained model weights
- Output: Tracked images + MOT format labels
- Processes: RBK-AALESUND, RBK-FREDRIKSTAD, RBK-HamKam

### `prepare_hota_data.py` ⚠️ **IMPORTANT**
Converts tracking outputs to MOT format for evaluation.
- **Handles different resolutions per dataset** (critical fix)
- Converts normalized YOLO coordinates to absolute pixels
- Generates ground truth from XML annotations
- Output: `hota_data/` directory structure

### `run_hota_evaluation.py`
Custom FootballDataset class for TrackEval library.
- Implements required abstract methods
- Remaps track IDs to be contiguous from 0
- Handles frame offset mapping (frames don't start at 0)
- Fixed numpy deprecation issues (`np.float`, `np.int`)

### `compute_tracking_metrics.py`
Simplified metric computation without full HOTA library.
- Useful for quick validation
- Computes: MOTA, IDF1, precision, recall, ID switches

## TrackEval Library Modifications

The following fixes were applied to TrackEval library for compatibility:

1. **NumPy deprecation fixes:**
   - `np.float` → `np.float64` in `hota.py`, `track_map.py`
   - `np.int` → `np.int64` in `identity.py`

2. **Single-class support:**
   - Modified `_compute_final_fields()` in `hota.py:176-183`
   - Handles both array and scalar cases for single-class evaluation

## Common Issues & Solutions

### Issue: "Resolution mismatch" or "Low recall on one dataset"
**Solution:** Check that `prepare_hota_data.py` has correct resolution for each dataset.
- RBK-AALESUND: 1920×1080
- **RBK-FREDRIKSTAD: 1280×720** ⚠️
- RBK-HamKam: 1920×1080

### Issue: "IndexError: invalid index to scalar variable"
**Solution:** Track IDs must be remapped to contiguous 0-indexed values (implemented in `get_preprocessed_seq_data()`).

### Issue: "Frame numbering mismatch"
**Solution:** MOT frames start at different offsets (1623, 1636, 1372) - must remap to 0-indexed timesteps.

## For Future Developers

### To add a new dataset:

1. **Update `run_tracking_validation.py`:**
   - Add dataset to validation list with frame range

2. **Update `prepare_hota_data.py`:**
   - Add dataset to `datasets` list (line 174-178)
   - **Include correct resolution** (width, height)
   - **Include correct frame offset** for MOT conversion

3. **Update `run_hota_evaluation.py`:**
   - Add dataset to `seq_list` (line 104)
   - Add sequence length to `seq_lengths` (line 105)
   - Add frame offset to `seq_frame_offsets` (line 107)

### To modify tracking parameters:

Edit ByteTrack configuration in `run_tracking_validation.py:40-50` or create custom YAML config.

## References

- TrackEval: https://github.com/JonathonLuiten/TrackEval
- ByteTrack Paper: https://arxiv.org/abs/2110.06864
- HOTA Metrics: https://arxiv.org/abs/2009.07736
