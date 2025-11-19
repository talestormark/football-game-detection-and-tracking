# Tracking Visualizations - Completed Deliverables

This directory contains all visualization deliverables for the ByteTrack object tracking implementation.

## Summary

**Project:** Football Object Tracking (Step 3.3 - 3.4)
**Status:** ✅ COMPLETED
**Date:** November 6, 2025

## Performance Highlights

- **HOTA:** 81.45% (target: >60%) - **36% above target**
- **IDF1:** 95.47% (target: >70%) - **36% above target**
- **MOTA:** 96.26% - Excellent tracking accuracy
- **ID Switches:** 23 (target: <50) - **54% below target**

## Visualization Files

### 1. Highlights Video
**File:** `validation_highlights.mp4` (8.6 MB)

- **Duration:** 6 seconds at 25 fps (150 frames)
- **Content:** 50 sampled frames from each validation dataset
- **Displays:** Bounding boxes, class labels, track IDs
- **Use case:** Quick overview of tracking performance

**Datasets included:**
- RBK-AALESUND (50 frames)
- RBK-FREDRIKSTAD (50 frames)
- RBK-HamKam (50 frames)

---

### 2. Side-by-Side Comparison Videos

Direct comparison of ground truth annotations vs. tracking predictions.

#### RBK-AALESUND
**File:** `RBK-AALESUND_comparison.mp4` (22 MB)
- **Frames:** 180 (frames 1622-1801)
- **Resolution:** 1920×1080
- **Duration:** 7.2 seconds at 25 fps
- **Left:** Ground truth from XML annotations
- **Right:** ByteTrack predictions

#### RBK-FREDRIKSTAD
**File:** `RBK-FREDRIKSTAD_comparison.mp4` (20 MB)
- **Frames:** 181 (frames 1635-1815)
- **Resolution:** 1280×720 (lower than others)
- **Duration:** 7.2 seconds at 25 fps
- **Note:** Required resolution-specific handling in evaluation

#### RBK-HamKam
**File:** `RBK-HamKam_comparison.mp4` (16 MB)
- **Frames:** 152 (frames 1371-1522)
- **Resolution:** 1920×1080
- **Duration:** 6.1 seconds at 25 fps

**Color coding:**
- 🟢 **Green:** Home team players
- 🔵 **Blue:** Away team players
- 🟡 **Yellow:** Referee
- ⚪ **White:** Ball

---

### 3. Trajectory Visualizations

Static images showing movement paths over time (first 150 frames of each dataset).

#### RBK-AALESUND
**File:** `RBK-AALESUND_trajectories.png` (1.6 MB)
- Shows player movement patterns across 150 frames
- Tracks with <5 frames excluded for clarity
- Track IDs labeled at final positions

#### RBK-FREDRIKSTAD
**File:** `RBK-FREDRIKSTAD_trajectories.png` (802 KB)
- Lower resolution (1280×720) results in smaller file size
- Same color coding as other visualizations

#### RBK-HamKam
**File:** `RBK-HamKam_trajectories.png` (1.7 MB)
- Full 1920×1080 resolution
- Clear visualization of player movements

**Use cases:**
- Analyzing player movement patterns
- Identifying tracking consistency
- Presentation material for spatial analysis

---

### 4. Metrics Summary

**File:** `metrics_summary.png` (72 KB)

Visual summary of HOTA evaluation results:
- **Main metrics:** HOTA, IDF1, MOTA with targets and status
- **Additional metrics:** Detection recall/precision, ID switches, fragmentations
- **Dataset info:** All three validation datasets listed
- **Format:** 1920×1080 infographic on dark background

**Use cases:**
- Presentation slides
- Project documentation
- Quick reference for performance metrics

---

## File Size Summary

| File | Size | Type |
|------|------|------|
| validation_highlights.mp4 | 8.6 MB | Video |
| RBK-AALESUND_comparison.mp4 | 22 MB | Video |
| RBK-FREDRIKSTAD_comparison.mp4 | 20 MB | Video |
| RBK-HamKam_comparison.mp4 | 16 MB | Video |
| RBK-AALESUND_trajectories.png | 1.6 MB | Image |
| RBK-FREDRIKSTAD_trajectories.png | 802 KB | Image |
| RBK-HamKam_trajectories.png | 1.7 MB | Image |
| metrics_summary.png | 72 KB | Image |
| **TOTAL** | **~70 MB** | - |

---

## How Visualizations Were Created

### Scripts Used

1. **create_visualizations.py**
   - Main visualization script
   - Parses XML ground truth and YOLO tracking predictions
   - Generates comparison videos, trajectories, and highlights

2. **create_metrics_overlay.py**
   - Creates metrics summary infographic
   - Reads HOTA evaluation results
   - Formats metrics with targets and status indicators

3. **run_visualizations_slurm.sh**
   - SLURM batch script for cluster execution
   - Activates `football_analysis` conda environment
   - Runs visualization pipeline

### Environment Requirements

- **Conda environment:** `football_analysis`
- **Key packages:** opencv-python, numpy
- **Execution:** SLURM job on CPUQ partition
- **Runtime:** ~1 minute total

---

## Usage in Presentations

### Recommended Order

1. **Start with:** `metrics_summary.png`
   - Show overall performance achievements
   - Highlight that all targets exceeded

2. **Demo with:** `validation_highlights.mp4`
   - Quick visual demonstration of tracking
   - Shows diversity across datasets

3. **Deep dive:** Side-by-side comparison videos
   - Pick one dataset (e.g., RBK-AALESUND)
   - Show ground truth vs predictions alignment
   - Highlight ID consistency

4. **Analysis:** Trajectory visualizations
   - Show movement patterns
   - Discuss spatial coverage
   - Demonstrate tracking continuity

### Key Talking Points

- **Excellent HOTA score (81.45%):** Significantly exceeds target, indicates balanced detection and association
- **High IDF1 (95.47%):** 95% of detections have correct IDs
- **Low ID switches (23):** Only 23 ID changes across 513 frames
- **Multi-dataset validation:** Consistent performance across different resolutions and conditions
- **Resolution handling:** Successfully addressed FREDRIKSTAD's 1280×720 resolution

---

## Technical Details

### Color Coding
All visualizations use consistent colors:
```python
CLASS_COLORS = {
    0: (0, 255, 0),      # home - green (BGR)
    1: (255, 0, 0),      # away - blue (BGR)
    2: (0, 255, 255),    # referee - yellow (BGR)
    3: (255, 255, 255)   # ball - white (BGR)
}
```

### Video Encoding
- **Codec:** mp4v (MPEG-4)
- **Frame rate:** 25 fps (matches source footage)
- **Quality:** High quality, no compression artifacts

### Trajectory Parameters
- **Frames shown:** First 150 frames per dataset
- **Minimum track length:** 5 frames (shorter tracks excluded)
- **Line width:** 2 pixels
- **ID label font:** cv2.FONT_HERSHEY_SIMPLEX, 0.4 scale

---

## Validation Data Sources

### Ground Truth
- **Format:** XML annotations from CVAT
- **Location:** `/cluster/projects/vc/courses/TDT17/other/Football2025/{DATASET}/annotations.xml`
- **Content:** Bounding boxes, track IDs, class labels

### Predictions
- **Format:** YOLO normalized coordinates with track IDs
- **Location:** `/cluster/work/tmstorma/Football2025/tracking/runs/val_tracking/{DATASET}/labels/`
- **Content:** `class_id x_center y_center width height track_id`

### Source Images
- **Location:** `/cluster/projects/vc/courses/TDT17/other/Football2025/{DATASET}/data/images/train/`
- **Format:** PNG images
- **Naming:** `frame_XXXXXX.png`

---

## Project Completion Status

✅ **Step 3.1:** Dataset preparation - COMPLETED
✅ **Step 3.2:** Object detection training - COMPLETED
✅ **Step 3.3:** Object tracking - COMPLETED
✅ **Step 3.4:** Visualizations - COMPLETED

**Project Status:** 🎉 **ALL DELIVERABLES COMPLETED**

---

## Contact & Documentation

For more information, see:
- `/cluster/work/tmstorma/Football2025/tracking/README.md` - Tracking implementation details
- `/cluster/work/tmstorma/Football2025/tracking/PRESENTATION_SCRIPT.md` - Presentation talking points
- `/cluster/work/tmstorma/Football2025/CLAUDE.md` - Full project documentation
