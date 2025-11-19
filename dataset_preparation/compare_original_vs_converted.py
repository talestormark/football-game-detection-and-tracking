import cv2
import numpy as np
from pathlib import Path

def visualize_comparison(dataset_name, frame_number):
    """
    Compare original 2-class YOLO labels vs our 4-class converted labels.

    Args:
        dataset_name: e.g., 'RBK-AALESUND'
        frame_number: The frame number in the image filename (e.g., 616 for frame_000616.png)
    """
    source_dir = Path('/cluster/projects/vc/courses/TDT17/other/Football2025')
    dataset_dir = Path('/cluster/work/tmstorma/Football2025/dataset')

    # Image path
    image_path = source_dir / dataset_name / 'data' / 'images' / 'train' / f'frame_{frame_number:06d}.png'

    # Original label path (2-class: player=0, ball=1)
    original_label_path = source_dir / dataset_name / 'labels' / 'train' / f'frame_{frame_number:06d}.txt'

    # Our converted label path (4-class: home=0, away=1, referee=2, ball=3)
    # First check train, then val
    converted_label_path = dataset_dir / 'labels' / 'train' / f'{dataset_name}_frame_{frame_number:06d}.txt'
    if not converted_label_path.exists():
        converted_label_path = dataset_dir / 'labels' / 'val' / f'{dataset_name}_frame_{frame_number:06d}.txt'

    # Load image
    img = cv2.imread(str(image_path))
    if img is None:
        print(f"Failed to load image: {image_path}")
        return

    h, w = img.shape[:2]
    print(f"\nAnalyzing: {dataset_name} frame_{frame_number:06d}.png")
    print(f"Image dimensions: {w}x{h}")

    # Create two copies for visualization
    img_original = img.copy()
    img_converted = img.copy()

    # Parse ORIGINAL labels
    print(f"\n=== ORIGINAL 2-CLASS LABELS ===")
    print(f"Reading from: {original_label_path}")

    if original_label_path.exists():
        original_boxes = []
        with open(original_label_path, 'r') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                parts = line.split()
                if len(parts) == 5:
                    class_id, x_c, y_c, width, height = map(float, parts)
                    original_boxes.append({
                        'class_id': int(class_id),
                        'x_c': x_c, 'y_c': y_c, 'w': width, 'h': height
                    })

        print(f"Found {len(original_boxes)} boxes")
        print("First 5 boxes (class_id x_c y_c w h):")
        for i, box in enumerate(original_boxes[:5]):
            print(f"  {i+1}. class={box['class_id']} ({box['x_c']:.4f}, {box['y_c']:.4f}) {box['w']:.4f}x{box['h']:.4f}")

        # Draw original labels (player=green, ball=cyan)
        for box in original_boxes:
            x_c_px = box['x_c'] * w
            y_c_px = box['y_c'] * h
            w_px = box['w'] * w
            h_px = box['h'] * h

            x1 = int(x_c_px - w_px / 2)
            y1 = int(y_c_px - h_px / 2)
            x2 = int(x_c_px + w_px / 2)
            y2 = int(y_c_px + h_px / 2)

            # class 0=player (green), 1=ball (cyan)
            color = (0, 255, 0) if box['class_id'] == 0 else (255, 255, 0)
            label = "player" if box['class_id'] == 0 else "ball"

            cv2.rectangle(img_original, (x1, y1), (x2, y2), color, 2)
            cv2.putText(img_original, label, (x1, y1-5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    else:
        print(f"Original label file not found!")

    # Parse CONVERTED labels
    print(f"\n=== CONVERTED 4-CLASS LABELS ===")
    print(f"Reading from: {converted_label_path}")

    if converted_label_path.exists():
        converted_boxes = []
        with open(converted_label_path, 'r') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                parts = line.split()
                if len(parts) == 5:
                    class_id, x_c, y_c, width, height = map(float, parts)
                    converted_boxes.append({
                        'class_id': int(class_id),
                        'x_c': x_c, 'y_c': y_c, 'w': width, 'h': height
                    })

        print(f"Found {len(converted_boxes)} boxes")
        print("First 5 boxes (class_id x_c y_c w h):")
        for i, box in enumerate(converted_boxes[:5]):
            print(f"  {i+1}. class={box['class_id']} ({box['x_c']:.4f}, {box['y_c']:.4f}) {box['w']:.4f}x{box['h']:.4f}")

        # Draw converted labels (home=blue, away=red, referee=yellow, ball=green)
        class_colors = [
            (255, 0, 0),    # home - blue
            (0, 0, 255),    # away - red
            (0, 255, 255),  # referee - yellow
            (0, 255, 0)     # ball - green
        ]
        class_names = ['home', 'away', 'referee', 'ball']

        for box in converted_boxes:
            x_c_px = box['x_c'] * w
            y_c_px = box['y_c'] * h
            w_px = box['w'] * w
            h_px = box['h'] * h

            x1 = int(x_c_px - w_px / 2)
            y1 = int(y_c_px - h_px / 2)
            x2 = int(x_c_px + w_px / 2)
            y2 = int(y_c_px + h_px / 2)

            color = class_colors[box['class_id']]
            label = class_names[box['class_id']]

            cv2.rectangle(img_converted, (x1, y1), (x2, y2), color, 2)
            cv2.putText(img_converted, label, (x1, y1-5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    else:
        print(f"Converted label file not found!")

    # Save side-by-side comparison
    output_dir = Path('/cluster/work/tmstorma/Football2025/dataset_preparation/comparison_visuals')
    output_dir.mkdir(exist_ok=True)

    # Create side-by-side image
    combined = np.hstack([img_original, img_converted])

    # Add text labels
    cv2.putText(combined, "ORIGINAL (2-class)", (50, 50),
               cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 3)
    cv2.putText(combined, "CONVERTED (4-class)", (w + 50, 50),
               cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 3)

    output_path = output_dir / f'{dataset_name}_frame{frame_number:06d}_comparison.png'
    cv2.imwrite(str(output_path), combined)

    # Also save individual images
    cv2.imwrite(str(output_dir / f'{dataset_name}_frame{frame_number:06d}_original.png'), img_original)
    cv2.imwrite(str(output_dir / f'{dataset_name}_frame{frame_number:06d}_converted.png'), img_converted)

    print(f"\n✓ Saved comparison to: {output_path}")
    print(f"✓ Individual images saved to: {output_dir}")


if __name__ == '__main__':
    print("="*80)
    print("COMPARING ORIGINAL vs CONVERTED LABELS")
    print("="*80)

    # The image you showed appears to be from the validation visualizations
    # Let's check several frames to identify which one matches your screenshot

    # Based on the validation files generated, let's check a few
    frames_to_check = [
        ('RBK-AALESUND', 616),   # train
        ('RBK-HamKam', 545),     # train
        ('RBK-AALESUND', 1787),  # val
        ('RBK-HamKam', 1445),    # val
    ]

    for dataset_name, frame_num in frames_to_check:
        visualize_comparison(dataset_name, frame_num)
        print("\n" + "-"*80 + "\n")
