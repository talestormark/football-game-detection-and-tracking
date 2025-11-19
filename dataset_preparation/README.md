# Dataset Preparation Scripts

This folder contains scripts for converting CVAT XML annotations to YOLO format and validating the conversion.

## Scripts

### 1. xml_to_yolo_converter.py
Converts CVAT XML annotations to YOLO format (4 classes: home, away, referee, ball).

**Features:**
- Reads image dimensions from XML to handle different resolutions
- Converts bounding boxes from pixel coordinates to normalized YOLO format
- Preserves tracking IDs for HOTA evaluation
- Generates train/val splits

**Usage:**
```bash
python xml_to_yolo_converter.py
```

**Output:**
- `/cluster/work/tmstorma/Football2025/dataset/labels/` - YOLO format labels
- `/cluster/work/tmstorma/Football2025/dataset/gt_tracking.json` - Ground truth tracking

---

### 2. create_dataset_structure.py
Creates the final dataset structure with symlinks to images and generates data.yaml.

**Usage:**
```bash
python create_dataset_structure.py
```

**Output:**
- `/cluster/work/tmstorma/Football2025/dataset/images/` - Symlinked images
- `/cluster/work/tmstorma/Football2025/dataset/train.txt` - Training file list
- `/cluster/work/tmstorma/Football2025/dataset/val.txt` - Validation file list
- `/cluster/work/tmstorma/Football2025/dataset/data.yaml` - YOLO config

---

### 3. validate_conversion.py
Validates the conversion with multiple checks and generates visualizations.

**Checks:**
- File counts match expected values
- All coordinate values in valid range [0, 1]
- Class distribution matches Step 1b analysis
- Visual samples with bounding boxes

**Usage:**
```bash
python validate_conversion.py
```

**Output:**
- `validation_visualizations/` - Random sample visualizations

---

### 4. compare_original_vs_converted.py
Debugging tool to compare original 2-class labels with converted 4-class labels.

**Usage:**
```bash
python compare_original_vs_converted.py
```

**Output:**
- `comparison_visuals/` - Side-by-side comparisons

---

### 5. debug_specific_frames.py
Debugging tool to investigate specific problematic frames.

**Usage:**
```bash
python debug_specific_frames.py
```

**Output:**
- `debug_visuals/` - Frame-specific debug visualizations

---

## Dataset Statistics

**Training:** 4,628 frames, 108,422 boxes
- RBK-AALESUND: 1,622 frames (1920x1080)
- RBK-FREDRIKSTAD: 1,635 frames (1280x720)
- RBK-HamKam: 1,371 frames (1920x1080)

**Validation:** 513 frames, 11,738 boxes
- RBK-AALESUND: 180 frames
- RBK-FREDRIKSTAD: 181 frames
- RBK-HamKam: 152 frames

**Class Distribution:**
- home: 43.3%
- away: 44.5%
- referee: 8.5%
- ball: 3.7%

---

## Important Notes

1. **Different Resolutions:** RBK-FREDRIKSTAD uses 1280x720, others use 1920x1080
2. **Frame Indexing:** XML frame N corresponds to image frame_{N+1:06d}.png
3. **Validation Set:** Uses same frames for both detection and tracking evaluation
