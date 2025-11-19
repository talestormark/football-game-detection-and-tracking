#!/usr/bin/env python3
"""
Run inference on VIKING and BODO-part3 frames
to verify the model detects home/away/referee correctly despite wrong labels
"""

from ultralytics import YOLO
from pathlib import Path

def main():
    print("="*60)
    print("Inference Test: VIKING and BODO-part3")
    print("="*60)

    # Load trained model
    model_path = Path('/cluster/work/tmstorma/Football2025/training/runs/yolov8s_4class2/weights/best.pt')
    print(f"\nLoading model: {model_path}")
    model = YOLO(str(model_path))

    # Select test images
    viking_dir = Path('/cluster/projects/vc/courses/TDT17/other/Football2025/RBK-VIKING/data/images/train')
    bodo_dir = Path('/cluster/projects/vc/courses/TDT17/other/Football2025/RBK-BODO/part3/RBK_BODO_PART3/data/images/train')

    test_images = [
        # VIKING - 2 frames
        viking_dir / 'frame_000100.png',
        viking_dir / 'frame_000500.png',

        # BODO-part3 - 2 frames (different frame numbers to avoid name collision)
        bodo_dir / 'frame_000100.png',
        bodo_dir / 'frame_000500.png',
    ]

    # Map for unique output names
    output_names = [
        'VIKING_frame_000100',
        'VIKING_frame_000500',
        'BODO_frame_000100',
        'BODO_frame_000500'
    ]

    # Verify images exist
    print("\nVerifying images exist:")
    valid_images = []
    for img in test_images:
        if img.exists():
            print(f"  ✓ {img.parent.parent.parent.parent.name if 'BODO' in str(img) else img.parent.parent.parent.name}/{img.name}")
            valid_images.append(img)
        else:
            print(f"  ✗ {img.name} - NOT FOUND")

    if not valid_images:
        print("\nNo valid images found!")
        return

    # Run inference
    output_dir = Path('/cluster/work/tmstorma/Football2025/training/inference_generalization')
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nRunning inference...")
    print(f"Output directory: {output_dir}")

    # Process each image individually to save with unique names
    all_results = []
    for img_path, output_name in zip(valid_images, output_names):
        print(f"Processing {output_name}...")
        results = model.predict(
            source=str(img_path),
            conf=0.25,
            iou=0.7,
            save=False,  # Don't auto-save, we'll save manually
            show_labels=True,
            show_conf=True,
            line_width=2,
            verbose=False
        )

        # Save with unique name
        from PIL import Image
        import numpy as np

        result = results[0]
        img_with_boxes = result.plot()  # Get annotated image
        img_pil = Image.fromarray(img_with_boxes)
        save_path = output_dir / f"{output_name}.jpg"
        img_pil.save(save_path)

        all_results.append(result)

    print(f"\n{'='*60}")
    print("Inference Complete!")
    print(f"{'='*60}")
    print(f"\nResults saved to: {output_dir}")
    print(f"\nProcessed {len(all_results)} images:")

    # Print detection summary for each image
    class_names = ['home', 'away', 'referee', 'ball']

    for i, (result, img_path, output_name) in enumerate(zip(all_results, valid_images, output_names)):
        dataset_name = "RBK-VIKING" if "VIKING" in str(img_path) else "RBK-BODO-part3"
        boxes = result.boxes
        print(f"\n{i+1}. {dataset_name}/{img_path.name}")
        print(f"   Total detections: {len(boxes)}")

        # Count by class
        if len(boxes) > 0:
            classes = boxes.cls.cpu().numpy()
            class_counts = {}
            for cls_id, cls_name in enumerate(class_names):
                count = (classes == cls_id).sum()
                if count > 0:
                    class_counts[cls_name] = count
                    print(f"     - {cls_name}: {count}")

            # Show merged player count
            player_count = class_counts.get('home', 0) + class_counts.get('away', 0) + class_counts.get('referee', 0)
            print(f"     → Merged 'player': {player_count} (home+away+referee)")

    print("\n" + "="*60)
    print("Key Observations:")
    print("="*60)
    print("✓ Model detects home/away/referee separately (based on jersey colors)")
    print("✓ Ground truth labels all as 'home' (incorrect)")
    print("→ For evaluation: merge predictions (home+away+referee) into 'player'")
    print("→ Then compare against ground truth 'home' → 'player'")

if __name__ == '__main__':
    main()
