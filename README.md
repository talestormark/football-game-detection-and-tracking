# Football Match Analysis: Object Detection and Tracking

Automated detection and tracking system for football match footage, identifying players, referees, and the ball across video frames.

## Overview

This project analyzes football match footage to extract insights for **RBK** (Rosenborg BK) and their opponents through:

- **Object Detection**: Detecting home players, away players, referees, and the ball in each frame
- **Object Tracking**: Maintaining consistent tracking IDs across frames for temporal analysis

**Model**: YOLOv8-s with ByteTrack
**Dataset**: 3 fully-labeled matches (5,141 frames total)
**Performance**: HOTA 81.45%, IDF1 95.47%, 23 ID switches

## Project Structure

```
Football2025/
├── data_analysis/              # Pre-training dataset analysis
│   ├── Dataset Quality & Annotation/     # Step 1a: Bbox and ID validation
│   ├── Class Distribution/               # Step 1b: Class balance analysis
│   ├── Object Size and Scale Analysis/   # Step 2a: Size variation analysis
│   └── Tracking ID Stability/            # Step 2b: Temporal continuity checks
│
├── dataset_preparation/        # Label conversion and dataset splits
│   ├── xml_to_yolo_converter.py         # XML → 4-class YOLO conversion
│   ├── prepare_dataset.py               # Train/val split generation
│   └── validate_conversion.py           # Conversion validation
│
├── dataset/                    # Training data (ignored in git)
│   ├── images/train/                    # Training images
│   ├── images/val/                      # Validation images
│   ├── labels/train/                    # 4-class YOLO labels
│   ├── labels/val/
│   └── data.yaml                        # Dataset configuration
│
├── training/                   # Model training scripts
│   └── train_yolov8.py                  # YOLOv8 training pipeline
│
├── tracking/                   # Tracking evaluation and visualization
│   ├── run_tracking.py                  # ByteTrack inference
│   ├── prepare_hota_data.py             # HOTA evaluation prep
│   ├── evaluate_tracking.py             # TrackEval metrics
│   ├── create_visualizations.py         # Video/trajectory generation
│   └── visualizations/                  # Output videos and plots
│
├── check_all_teams.py          # Dataset availability checker
```

## Quick Start

### 1. Environment Setup

The base dataset is located at (read-only):
```
/cluster/projects/vc/courses/TDT17/other/Football2025
```

All code and outputs are in this workspace:
```
/cluster/work/tmstorma/Football2025
```

### 2. Data Analysis

Explore pre-computed analysis reports:
- **Dataset Quality**: `data_analysis/Dataset Quality & Annotation/STEP1A_REPORT.md`
- **Class Distribution**: `data_analysis/Class Distribution/STEP1B_REPORT.md`
- **Tracking Stability**: `data_analysis/Tracking ID Stability/STEP2B_REPORT.md`

### 3. Dataset Preparation

Convert XML annotations to 4-class YOLO format:
```bash
cd dataset_preparation
python xml_to_yolo_converter.py
python prepare_dataset.py
python validate_conversion.py
```

See `dataset/data.yaml` for configuration.

### 4. Model Training

Train YOLOv8-s detector:
```bash
cd training
python train_yolov8.py
```

Trained weights saved to: `runs/detect/yolov8s_4class/weights/best.pt`

### 5. Tracking Evaluation

Run ByteTrack and evaluate performance:
```bash
cd tracking
python run_tracking.py
python prepare_hota_data.py
python evaluate_tracking.py
```

### 6. Visualizations

Generate output videos and trajectory plots:
```bash
cd tracking
python create_visualizations.py
```

Outputs in: `tracking/visualizations/`

## Results

### Detection Performance
- **Overall mAP@0.5**: 0.89
- **Home/Away Players**: mAP@0.5 > 0.90
- **Referee**: mAP@0.5 = 0.85
- **Ball**: mAP@0.5 = 0.74

### Tracking Performance
- **HOTA**: 81.45% (target: >60%)
- **IDF1**: 95.47% (target: >70%)
- **MOTA**: 96.26%
- **ID Switches**: 23 (target: <50)
- **Detection Precision**: 99.42%

## Deliverables

All deliverables located in `tracking/visualizations/`:

1. **Validation Highlights** (`validation_highlights.mp4`)
   150-frame compilation from all 3 validation datasets

2. **Side-by-Side Comparisons**
   Ground truth vs predictions for each match:
   - `RBK-AALESUND_comparison.mp4` (180 frames)
   - `RBK-FREDRIKSTAD_comparison.mp4` (181 frames)
   - `RBK-HamKam_comparison.mp4` (152 frames)

3. **Trajectory Visualizations**
   Movement path plots for each dataset:
   - `RBK-AALESUND_trajectories.png`
   - `RBK-FREDRIKSTAD_trajectories.png`
   - `RBK-HamKam_trajectories.png`

4. **Metrics Summary** (`metrics_summary.png`)
   Performance dashboard with all tracking metrics

## Datasets Used

**Training datasets** (3 matches with 4-class labels):
- RBK-AALESUND: 1,802 frames (1,622 train / 180 val)
- RBK-FREDRIKSTAD: 1,816 frames (1,635 train / 181 val)
- RBK-HamKam: 1,523 frames (1,371 train / 152 val)

**Total**: 4,628 training frames, 513 validation frames

## Class Definitions

- **Class 0 (home)**: RBK players (home team)
- **Class 1 (away)**: Opponent players
- **Class 2 (referee)**: Match officials
- **Class 3 (ball)**: Football

## Implementation Details

For complete implementation instructions, dataset preparation workflows, and training strategies, see:
- **CLAUDE.md**: Comprehensive project guide with step-by-step instructions
- **Individual reports**: Each `data_analysis/` subdirectory contains detailed findings

## Code Style

- Minimal and necessary code only
- Clean, readable, well-structured
- Clear variable names and concise comments
- No auto-generated modifications without approval

## References

Inspiration from previous student project:
https://github.com/roboflow/notebooks/blob/main/notebooks/how-to-track-football-players.ipynb

**Note**: The reference repo uses different dataset structure and labeling conventions.
