import cv2
import xml.etree.ElementTree as ET
from pathlib import Path

def visualize_from_xml(dataset_name, frame_id):
    """
    Visualize a specific frame directly from XML to compare with our conversion.
    """
    # Paths
    source_dir = Path('/cluster/projects/vc/courses/TDT17/other/Football2025')
    xml_path = source_dir / dataset_name / 'annotations.xml'

    # Note: XML frame N corresponds to image frame_{N+1:06d}.png
    image_path = source_dir / dataset_name / 'data' / 'images' / 'train' / f'frame_{frame_id+1:06d}.png'

    # Load image
    img = cv2.imread(str(image_path))
    if img is None:
        print(f"Failed to load image: {image_path}")
        return

    print(f"\nAnalyzing {dataset_name} frame {frame_id} (image: frame_{frame_id+1:06d}.png)")
    print(f"Image dimensions: {img.shape[1]}x{img.shape[0]}")

    # Parse XML
    tree = ET.parse(xml_path)
    root = tree.getroot()

    # Find all boxes for this frame
    boxes_found = []
    for track in root.findall('track'):
        track_id = int(track.get('id'))
        label = track.get('label')

        for box in track.findall('box'):
            if int(box.get('frame')) != frame_id:
                continue

            outside = int(box.get('outside', '0'))
            if outside == 1:
                continue

            # Get coordinates
            xtl = float(box.get('xtl'))
            ytl = float(box.get('ytl'))
            xbr = float(box.get('xbr'))
            ybr = float(box.get('ybr'))

            # Get team
            team = None
            if label == 'player':
                team_attr = box.find("attribute[@name='team']")
                if team_attr is not None:
                    team = team_attr.text

            if label == 'event_labels':
                continue

            boxes_found.append({
                'track_id': track_id,
                'label': label,
                'team': team,
                'xtl': xtl, 'ytl': ytl, 'xbr': xbr, 'ybr': ybr
            })

    print(f"Found {len(boxes_found)} boxes in XML")

    # Now check our converted label
    dataset_dir = Path('/cluster/work/tmstorma/Football2025/dataset')
    label_file = dataset_dir / 'labels' / 'train' / f'{dataset_name}_frame_{frame_id+1:06d}.txt'

    if not label_file.exists():
        # Try val
        label_file = dataset_dir / 'labels' / 'val' / f'{dataset_name}_frame_{frame_id+1:06d}.txt'

    converted_boxes = []
    with open(label_file, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 5:
                class_id, x_c, y_c, w, h = map(float, parts)
                converted_boxes.append({
                    'class_id': int(class_id),
                    'x_c': x_c, 'y_c': y_c, 'w': w, 'h': h
                })

    print(f"Found {len(converted_boxes)} boxes in converted label")

    if len(boxes_found) != len(converted_boxes):
        print(f"⚠️ BOX COUNT MISMATCH: XML has {len(boxes_found)}, converted has {len(converted_boxes)}")

    # Compare a few boxes
    print("\nFirst 3 boxes comparison:")
    print("XML boxes (pixel coords):")
    for i, box in enumerate(boxes_found[:3]):
        print(f"  {i+1}. {box['label']}/{box['team']}: ({box['xtl']:.1f}, {box['ytl']:.1f}) to ({box['xbr']:.1f}, {box['ybr']:.1f})")

    print("\nConverted boxes (YOLO normalized):")
    class_names = ['home', 'away', 'referee', 'ball']
    for i, box in enumerate(converted_boxes[:3]):
        # Convert back to pixels for comparison
        x_c_px = box['x_c'] * 1920
        y_c_px = box['y_c'] * 1080
        w_px = box['w'] * 1920
        h_px = box['h'] * 1080
        xtl_px = x_c_px - w_px/2
        ytl_px = y_c_px - h_px/2
        xbr_px = x_c_px + w_px/2
        ybr_px = y_c_px + h_px/2
        print(f"  {i+1}. {class_names[box['class_id']]}: ({xtl_px:.1f}, {ytl_px:.1f}) to ({xbr_px:.1f}, {ybr_px:.1f})")

    # Visualize both
    img_xml = img.copy()
    img_converted = img.copy()

    class_colors = {
        'home': (255, 0, 0),
        'away': (0, 0, 255),
        'referee': (0, 255, 255),
        'ball': (0, 255, 0)
    }

    # Draw XML boxes
    for box in boxes_found:
        class_key = box['team'] if box['label'] == 'player' else 'ball'
        color = class_colors.get(class_key, (128, 128, 128))
        cv2.rectangle(img_xml,
                     (int(box['xtl']), int(box['ytl'])),
                     (int(box['xbr']), int(box['ybr'])),
                     color, 2)

    # Draw converted boxes
    for box in converted_boxes:
        x_c_px = box['x_c'] * 1920
        y_c_px = box['y_c'] * 1080
        w_px = box['w'] * 1920
        h_px = box['h'] * 1080
        x1 = int(x_c_px - w_px/2)
        y1 = int(y_c_px - h_px/2)
        x2 = int(x_c_px + w_px/2)
        y2 = int(y_c_px + h_px/2)

        color = list(class_colors.values())[box['class_id']]
        cv2.rectangle(img_converted, (x1, y1), (x2, y2), color, 2)

    # Save comparison
    output_dir = Path('/cluster/work/tmstorma/Football2025/dataset_preparation/debug_visuals')
    output_dir.mkdir(exist_ok=True)

    cv2.imwrite(str(output_dir / f'{dataset_name}_frame{frame_id:06d}_xml.png'), img_xml)
    cv2.imwrite(str(output_dir / f'{dataset_name}_frame{frame_id:06d}_converted.png'), img_converted)

    print(f"\nSaved comparison images to {output_dir}")


if __name__ == '__main__':
    # Debug the specific frames mentioned
    print("="*60)
    print("DEBUGGING SPECIFIC FRAMES")
    print("="*60)

    # Check a few samples
    visualize_from_xml('RBK-AALESUND', 615)  # frame_000616.png
    visualize_from_xml('RBK-HamKam', 544)     # frame_000545.png
    visualize_from_xml('RBK-AALESUND', 1786)  # frame_001787.png (val)
