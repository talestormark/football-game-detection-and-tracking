#!/usr/bin/env python3
"""
Step 2b - Validate Frame Numbering Consistency
Checks that XML annotations, images, labels, and train.txt are aligned
"""

import xml.etree.ElementTree as ET
from pathlib import Path
from collections import defaultdict

# Dataset paths
DATASETS = {
    'RBK-VIKING': {
        'xml': '/cluster/projects/vc/courses/TDT17/other/Football2025/RBK-VIKING/annotations.xml',
        'images': '/cluster/projects/vc/courses/TDT17/other/Football2025/RBK-VIKING/data/images/train',
        'labels': '/cluster/projects/vc/courses/TDT17/other/Football2025/RBK-VIKING/labels/train',
        'train_txt': '/cluster/projects/vc/courses/TDT17/other/Football2025/RBK-VIKING/train.txt'
    },
    'RBK-AALESUND': {
        'xml': '/cluster/projects/vc/courses/TDT17/other/Football2025/RBK-AALESUND/annotations.xml',
        'images': '/cluster/projects/vc/courses/TDT17/other/Football2025/RBK-AALESUND/data/images/train',
        'labels': '/cluster/projects/vc/courses/TDT17/other/Football2025/RBK-AALESUND/labels/train',
        'train_txt': '/cluster/projects/vc/courses/TDT17/other/Football2025/RBK-AALESUND/train.txt'
    },
    'RBK-FREDRIKSTAD': {
        'xml': '/cluster/projects/vc/courses/TDT17/other/Football2025/RBK-FREDRIKSTAD/annotations.xml',
        'images': '/cluster/projects/vc/courses/TDT17/other/Football2025/RBK-FREDRIKSTAD/data/images/train',
        'labels': '/cluster/projects/vc/courses/TDT17/other/Football2025/RBK-FREDRIKSTAD/labels/train',
        'train_txt': '/cluster/projects/vc/courses/TDT17/other/Football2025/RBK-FREDRIKSTAD/train.txt'
    },
    'RBK-HamKam': {
        'xml': '/cluster/projects/vc/courses/TDT17/other/Football2025/RBK-HamKam/annotations.xml',
        'images': '/cluster/projects/vc/courses/TDT17/other/Football2025/RBK-HamKam/data/images/train',
        'labels': '/cluster/projects/vc/courses/TDT17/other/Football2025/RBK-HamKam/labels/train',
        'train_txt': '/cluster/projects/vc/courses/TDT17/other/Football2025/RBK-HamKam/train.txt'
    },
    'RBK-BODO-part3': {
        'xml': '/cluster/projects/vc/courses/TDT17/other/Football2025/RBK-BODO/part3/RBK_BODO_PART3/annotations.xml',
        'images': '/cluster/projects/vc/courses/TDT17/other/Football2025/RBK-BODO/part3/RBK_BODO_PART3/data/images/train',
        'labels': '/cluster/projects/vc/courses/TDT17/other/Football2025/RBK-BODO/part3/RBK_BODO_PART3/labels/train',
        'train_txt': '/cluster/projects/vc/courses/TDT17/other/Football2025/RBK-BODO/part3/RBK_BODO_PART3/train.txt'
    },
}

def extract_frame_number(filename):
    """Extract frame number from filename like 'frame_000123.png'"""
    try:
        # Handle both 'frame_000123.png' and 'data/images/train/frame_000123.png'
        basename = Path(filename).stem
        if basename.startswith('frame_'):
            return int(basename.split('_')[1])
    except:
        pass
    return None

def get_xml_frames(xml_path):
    """Get all frame numbers referenced in XML annotations"""
    tree = ET.parse(xml_path)
    root = tree.getroot()

    frames = set()
    for track in root.findall('track'):
        for box in track.findall('box'):
            frame_num = int(box.get('frame'))
            frames.add(frame_num)

    return frames

def get_image_frames(images_dir):
    """Get all frame numbers from image files"""
    images_path = Path(images_dir)
    frames = set()

    if images_path.exists():
        for img_file in images_path.glob('frame_*.png'):
            frame_num = extract_frame_number(img_file.name)
            if frame_num is not None:
                frames.add(frame_num)

    return frames

def get_label_frames(labels_dir):
    """Get all frame numbers from label files"""
    labels_path = Path(labels_dir)
    frames = set()

    if labels_path.exists():
        for label_file in labels_path.glob('frame_*.txt'):
            frame_num = extract_frame_number(label_file.name)
            if frame_num is not None:
                frames.add(frame_num)

    return frames

def get_train_txt_frames(train_txt_path):
    """Get all frame numbers from train.txt"""
    frames = set()
    train_txt = Path(train_txt_path)

    if train_txt.exists():
        with open(train_txt, 'r') as f:
            for line in f:
                frame_num = extract_frame_number(line.strip())
                if frame_num is not None:
                    frames.add(frame_num)

    return frames

def validate_dataset(dataset_name, paths):
    """Validate frame consistency for one dataset"""
    print(f"\n{'='*80}")
    print(f"Dataset: {dataset_name}")
    print('='*80)

    # Check if paths exist
    xml_path = Path(paths['xml'])
    if not xml_path.exists():
        print(f"XML not found: {paths['xml']}")
        return

    # Get frame numbers from each source
    print("\nCollecting frame numbers from sources...")
    xml_frames = get_xml_frames(paths['xml'])
    image_frames = get_image_frames(paths['images'])
    label_frames = get_label_frames(paths['labels'])
    train_txt_frames = get_train_txt_frames(paths['train_txt'])

    print(f"  XML annotations: {len(xml_frames)} unique frames")
    print(f"  Image files: {len(image_frames)} frames")
    print(f"  Label files: {len(label_frames)} frames")
    print(f"  train.txt: {len(train_txt_frames)} frames")

    # Get expected range from XML
    if xml_frames:
        min_frame = min(xml_frames)
        max_frame = max(xml_frames)
        print(f"\nExpected frame range: {min_frame} to {max_frame}")

    # Check for issues
    issues = []

    # 1. XML frames without images
    xml_no_image = xml_frames - image_frames
    if xml_no_image:
        issues.append(f"XML frames missing images: {len(xml_no_image)}")
        print(f"\n  WARNING: {len(xml_no_image)} XML frames have no corresponding image")
        if len(xml_no_image) <= 5:
            print(f"    Frames: {sorted(list(xml_no_image))}")

    # 2. Images without XML annotations
    image_no_xml = image_frames - xml_frames
    if image_no_xml:
        issues.append(f"Images without XML annotations: {len(image_no_xml)}")
        print(f"\n  WARNING: {len(image_no_xml)} images have no XML annotations")
        if len(image_no_xml) <= 5:
            print(f"    Frames: {sorted(list(image_no_xml))}")

    # 3. XML frames without labels
    xml_no_label = xml_frames - label_frames
    if xml_no_label:
        issues.append(f"XML frames missing label files: {len(xml_no_label)}")
        print(f"\n  WARNING: {len(xml_no_label)} XML frames have no label file")
        if len(xml_no_label) <= 5:
            print(f"    Frames: {sorted(list(xml_no_label))}")

    # 4. Labels without XML
    label_no_xml = label_frames - xml_frames
    if label_no_xml:
        issues.append(f"Label files without XML: {len(label_no_xml)}")
        print(f"\n  WARNING: {len(label_no_xml)} label files have no XML annotations")
        if len(label_no_xml) <= 5:
            print(f"    Frames: {sorted(list(label_no_xml))}")

    # 5. train.txt discrepancies
    train_no_image = train_txt_frames - image_frames
    if train_no_image:
        issues.append(f"train.txt references missing images: {len(train_no_image)}")
        print(f"\n  WARNING: {len(train_no_image)} train.txt entries have no image file")

    image_not_in_train = image_frames - train_txt_frames
    if image_not_in_train:
        issues.append(f"Images not listed in train.txt: {len(image_not_in_train)}")
        print(f"\n  WARNING: {len(image_not_in_train)} images not listed in train.txt")

    # 6. Check for sequential gaps in XML frames
    if xml_frames:
        sorted_frames = sorted(xml_frames)
        expected_frames = set(range(min_frame, max_frame + 1))
        missing_frames = expected_frames - xml_frames
        if missing_frames:
            issues.append(f"Gaps in frame sequence: {len(missing_frames)} missing")
            print(f"\n  INFO: {len(missing_frames)} frames missing in sequence")
            if len(missing_frames) <= 10:
                print(f"    Missing frames: {sorted(list(missing_frames))}")

    # Summary
    if not issues:
        print(f"\n  ✓ All frame numbering is consistent")
    else:
        print(f"\n  Found {len(issues)} consistency issue(s)")

    return {
        'dataset': dataset_name,
        'xml_frames': len(xml_frames),
        'image_frames': len(image_frames),
        'label_frames': len(label_frames),
        'train_txt_frames': len(train_txt_frames),
        'issues': issues
    }

def main():
    print("="*80)
    print("STEP 2b: FRAME NUMBERING CONSISTENCY VALIDATION")
    print("="*80)

    results = []

    for dataset_name, paths in DATASETS.items():
        result = validate_dataset(dataset_name, paths)
        if result:
            results.append(result)

    # Overall summary
    print("\n" + "="*80)
    print("OVERALL SUMMARY")
    print("="*80)

    datasets_with_issues = sum(1 for r in results if r.get('issues'))
    print(f"\nDatasets analyzed: {len(results)}")
    print(f"Datasets with issues: {datasets_with_issues}")
    print(f"Datasets clean: {len(results) - datasets_with_issues}")

    if datasets_with_issues > 0:
        print(f"\nRecommendation: Review datasets with issues for potential data quality problems")
    else:
        print(f"\n✓ All datasets have consistent frame numbering")

    print("="*80)

if __name__ == '__main__':
    main()
