# Tracking Visualizations

This document describes the visualization deliverables for the tracking project.

## Overview

After completing HOTA evaluation with excellent results (HOTA: 81.45%, IDF1: 95.47%, ID switches: 23), we are creating visual deliverables to showcase the tracking performance.

## Generated Visualizations

### 1. Side-by-Side Comparison Videos

Ground truth vs. predictions for each validation dataset:
- `visualizations/RBK-AALESUND_comparison.mp4` (180 frames)
- `visualizations/RBK-FREDRIKSTAD_comparison.mp4` (181 frames)
- `visualizations/RBK-HamKam_comparison.mp4` (152 frames)

Each frame shows:
- Left: Ground truth annotations
- Right: Tracking predictions
- Color-coded by class (home=green, away=blue, referee=yellow, ball=white)
- Track IDs displayed on each bounding box

### 2. Trajectory Visualizations

Movement path visualizations showing object trajectories:
- `visualizations/RBK-AALESUND_trajectories.png`
- `visualizations/RBK-FREDRIKSTAD_trajectories.png`
- `visualizations/RBK-HamKam_trajectories.png`

Shows:
- 150 frames of tracking data
- Colored lines showing movement paths
- Current positions marked
- Track IDs labeled

### 3. Validation Highlights Video

Combined highlights from all three datasets:
- `visualizations/validation_highlights.mp4` (~150 frames total)
- 50 evenly-sampled frames from each dataset
- Shows tracking predictions with bounding boxes and IDs
- Dataset name labeled on each frame

### 4. Metrics Summary

Visual summary of evaluation results:
- `visualizations/metrics_summary.png`
- Shows HOTA, IDF1, MOTA scores with targets
- Additional metrics (recall, precision, ID switches, etc.)
- Dataset information

## Running the Visualization Pipeline

### Option 1: SLURM Job (Recommended)

```bash
cd /cluster/work/tmstorma/Football2025/tracking
sbatch run_visualizations_slurm.sh
```

This will:
1. Activate the `football_analysis` conda environment
2. Install opencv-python if needed
3. Generate all visualizations
4. Save outputs to `visualizations/` directory

### Option 2: Interactive

```bash
module load Anaconda3/2024.02-1
source ${EBROOTANACONDA3}/etc/profile.d/conda.sh
conda activate football_analysis
pip install opencv-python

cd /cluster/work/tmstorma/Football2025/tracking
python create_visualizations.py
python create_metrics_overlay.py
```

## File Descriptions

### Scripts

- **create_visualizations.py**: Main script to generate videos and trajectory plots
  - Parses XML ground truth annotations
  - Parses YOLO tracking predictions
  - Creates side-by-side comparison videos
  - Creates trajectory visualizations
  - Creates highlights video

- **create_metrics_overlay.py**: Generate metrics summary visualization
  - Reads HOTA evaluation results
  - Creates formatted metrics overlay image

- **run_visualizations_slurm.sh**: SLURM batch script
  - Activates football_analysis environment
  - Runs visualization scripts
  - Logs output to `logs/`

## Expected Runtime

- Side-by-side videos: ~5-10 minutes per dataset
- Trajectory visualizations: ~1-2 minutes per dataset
- Highlights video: ~3-5 minutes
- Metrics overlay: <1 minute

**Total: ~20-30 minutes**

## Output Size

- Each comparison video: ~50-100 MB
- Highlights video: ~30-50 MB
- Trajectory images: ~5-10 MB each
- Metrics summary: ~1 MB

**Total: ~200-300 MB**

## Troubleshooting

### OpenCV Import Error

If you get `ModuleNotFoundError: No module named 'cv2'`:
```bash
conda activate football_analysis
pip install opencv-python
```

### Missing Frames

Ensure validation frames exist at:
- `/cluster/projects/vc/courses/TDT17/other/Football2025/RBK-AALESUND/data/images/train/`
- `/cluster/projects/vc/courses/TDT17/other/Football2025/RBK-FREDRIKSTAD/data/images/train/`
- `/cluster/projects/vc/courses/TDT17/other/Football2025/RBK-HamKam/data/images/train/`

### Tracking Labels Missing

Ensure tracking outputs exist at:
- `/cluster/work/tmstorma/Football2025/tracking/runs/val_tracking/RBK-AALESUND/labels/`
- `/cluster/work/tmstorma/Football2025/tracking/runs/val_tracking/RBK-FREDRIKSTAD/labels/`
- `/cluster/work/tmstorma/Football2025/tracking/runs/val_tracking/RBK-HamKam/labels/`

## Next Steps

Once visualizations are generated:
1. Review videos for quality
2. Check that tracking IDs are consistent
3. Verify ground truth vs predictions alignment
4. Use in presentation/documentation
5. Optional: Create generalization test videos for RBK-VIKING and RBK-BODO-part3
