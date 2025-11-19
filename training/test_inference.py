#!/usr/bin/env python3
"""
Run inference on sample validation images from each dataset
"""

from ultralytics import YOLO
from pathlib import Path

def main():
    # Load trained model
    model_path = Path('/cluster/work/tmstorma/Football2025/training/runs/yolov8s_4class2/weights/best.pt')
    print(f"Loading model: {model_path}")
    model = YOLO(str(model_path))

    # Select 2 images from each validation dataset
    val_dir = Path('/cluster/work/tmstorma/Football2025/dataset/images/val')

    test_images = [
        # AALESUND - 2 images
        val_dir / 'RBK-AALESUND_frame_001623.png',
        val_dir / 'RBK-AALESUND_frame_001650.png',

        # FREDRIKSTAD - 2 images
        val_dir / 'RBK-FREDRIKSTAD_frame_001636.png',
        val_dir / 'RBK-FREDRIKSTAD_frame_001670.png',

        # HamKam - 2 images
        val_dir / 'RBK-HamKam_frame_001372.png',
        val_dir / 'RBK-HamKam_frame_001400.png',
    ]

    # Verify images exist
    print("\nVerifying images exist:")
    for img in test_images:
        if img.exists():
            print(f"  ✓ {img.name}")
        else:
            print(f"  ✗ {img.name} - NOT FOUND")

    # Run inference
    output_dir = Path('/cluster/work/tmstorma/Football2025/training/inference_results')
    output_dir.mkdir(exist_ok=True)

    print(f"\nRunning inference...")
    print(f"Output directory: {output_dir}")

    results = model.predict(
        source=[str(img) for img in test_images if img.exists()],
        conf=0.25,           # Confidence threshold
        iou=0.7,             # NMS IoU threshold
        save=True,           # Save annotated images
        project=str(output_dir.parent),
        name='inference_results',
        exist_ok=True,
        show_labels=True,    # Show class labels
        show_conf=True,      # Show confidence scores
        line_width=2,
        verbose=True
    )

    print(f"\n{'='*60}")
    print("Inference complete!")
    print(f"{'='*60}")
    print(f"\nResults saved to: {output_dir}")
    print(f"\nProcessed {len(results)} images:")

    # Print detection summary for each image
    for i, (result, img_path) in enumerate(zip(results, test_images)):
        if img_path.exists():
            boxes = result.boxes
            print(f"\n{i+1}. {img_path.name}")
            print(f"   Detections: {len(boxes)}")

            # Count by class
            class_names = ['home', 'away', 'referee', 'ball']
            if len(boxes) > 0:
                classes = boxes.cls.cpu().numpy()
                for cls_id, cls_name in enumerate(class_names):
                    count = (classes == cls_id).sum()
                    if count > 0:
                        print(f"     - {cls_name}: {count}")

if __name__ == '__main__':
    main()
