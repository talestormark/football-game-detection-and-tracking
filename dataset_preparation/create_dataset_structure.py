import os
from pathlib import Path

def create_symlinks_and_file_lists():
    """
    Create symlinks to images and generate train.txt and val.txt files.
    """
    # Base directories
    source_dir = Path('/cluster/projects/vc/courses/TDT17/other/Football2025')
    dataset_dir = Path('/cluster/work/tmstorma/Football2025/dataset')

    # Create image directories
    train_images_dir = dataset_dir / 'images' / 'train'
    val_images_dir = dataset_dir / 'images' / 'val'
    train_images_dir.mkdir(parents=True, exist_ok=True)
    val_images_dir.mkdir(parents=True, exist_ok=True)

    # Dataset configurations
    datasets = {
        'RBK-AALESUND': {
            'source_images': source_dir / 'RBK-AALESUND' / 'data' / 'images' / 'train',
            'total_frames': 1802,
            'train_end': 1621,
            'val_start': 1622,
        },
        'RBK-FREDRIKSTAD': {
            'source_images': source_dir / 'RBK-FREDRIKSTAD' / 'data' / 'images' / 'train',
            'total_frames': 1816,
            'train_end': 1634,
            'val_start': 1635,
        },
        'RBK-HamKam': {
            'source_images': source_dir / 'RBK-HamKam' / 'data' / 'images' / 'train',
            'total_frames': 1523,
            'train_end': 1370,
            'val_start': 1371,
        }
    }

    train_file_list = []
    val_file_list = []

    # Process each dataset
    for dataset_name, config in datasets.items():
        print(f"Processing {dataset_name}...")

        source_images = config['source_images']

        # Training frames
        for frame_id in range(0, config['train_end'] + 1):
            # Note: XML frame N corresponds to image frame_{N+1:06d}.png
            image_name = f'frame_{frame_id+1:06d}.png'
            source_image = source_images / image_name

            # Create unique name with dataset prefix
            target_name = f'{dataset_name}_{image_name}'
            target_path = train_images_dir / target_name

            # Create symlink if it doesn't exist
            if not target_path.exists():
                os.symlink(source_image, target_path)

            # Add to train list (use absolute path)
            train_file_list.append(str(target_path))

        # Validation frames
        for frame_id in range(config['val_start'], config['total_frames']):
            image_name = f'frame_{frame_id+1:06d}.png'
            source_image = source_images / image_name

            target_name = f'{dataset_name}_{image_name}'
            target_path = val_images_dir / target_name

            if not target_path.exists():
                os.symlink(source_image, target_path)

            val_file_list.append(str(target_path))

        print(f"  Created {config['train_end'] + 1} train and "
              f"{config['total_frames'] - config['val_start']} val symlinks")

    # Write train.txt
    train_txt = dataset_dir / 'train.txt'
    with open(train_txt, 'w') as f:
        for path in sorted(train_file_list):
            f.write(f"{path}\n")
    print(f"\nCreated {train_txt} with {len(train_file_list)} images")

    # Write val.txt
    val_txt = dataset_dir / 'val.txt'
    with open(val_txt, 'w') as f:
        for path in sorted(val_file_list):
            f.write(f"{path}\n")
    print(f"Created {val_txt} with {len(val_file_list)} images")


def rename_label_files():
    """
    Rename label files to match the image naming convention with dataset prefix.
    """
    dataset_dir = Path('/cluster/work/tmstorma/Football2025/dataset')

    datasets = {
        'RBK-AALESUND': {
            'total_frames': 1802,
            'train_end': 1621,
            'val_start': 1622,
        },
        'RBK-FREDRIKSTAD': {
            'total_frames': 1816,
            'train_end': 1634,
            'val_start': 1635,
        },
        'RBK-HamKam': {
            'total_frames': 1523,
            'train_end': 1370,
            'val_start': 1371,
        }
    }

    train_labels_dir = dataset_dir / 'labels' / 'train'
    val_labels_dir = dataset_dir / 'labels' / 'val'

    for dataset_name, config in datasets.items():
        print(f"Renaming labels for {dataset_name}...")

        # Training labels
        for frame_id in range(0, config['train_end'] + 1):
            old_name = f'frame_{frame_id+1:06d}.txt'
            new_name = f'{dataset_name}_frame_{frame_id+1:06d}.txt'
            old_path = train_labels_dir / old_name
            new_path = train_labels_dir / new_name

            if old_path.exists() and not new_path.exists():
                old_path.rename(new_path)

        # Validation labels
        for frame_id in range(config['val_start'], config['total_frames']):
            old_name = f'frame_{frame_id+1:06d}.txt'
            new_name = f'{dataset_name}_frame_{frame_id+1:06d}.txt'
            old_path = val_labels_dir / old_name
            new_path = val_labels_dir / new_name

            if old_path.exists() and not new_path.exists():
                old_path.rename(new_path)


def create_data_yaml():
    """
    Create data.yaml configuration file for YOLO training.
    """
    dataset_dir = Path('/cluster/work/tmstorma/Football2025/dataset')

    yaml_content = f"""# Football object detection dataset
# 4-class configuration: home, away, referee, ball

path: {dataset_dir}
train: train.txt
val: val.txt

# Classes
names:
  0: home
  1: away
  2: referee
  3: ball

# Number of classes
nc: 4
"""

    yaml_path = dataset_dir / 'data.yaml'
    with open(yaml_path, 'w') as f:
        f.write(yaml_content)

    print(f"\nCreated {yaml_path}")


def main():
    print("Creating dataset structure...")
    print("="*60)

    # Step 1: Create symlinks and file lists
    print("\n1. Creating symlinks and file lists...")
    create_symlinks_and_file_lists()

    # Step 2: Create data.yaml
    print("\n2. Creating data.yaml...")
    create_data_yaml()

    print("\n" + "="*60)
    print("Dataset structure created successfully!")
    print("\nDirectory structure:")
    print("/cluster/work/tmstorma/Football2025/dataset/")
    print("├── images/")
    print("│   ├── train/  (4,628 images)")
    print("│   └── val/    (513 images)")
    print("├── labels/")
    print("│   ├── train/  (4,628 label files)")
    print("│   └── val/    (513 label files)")
    print("├── train.txt")
    print("├── val.txt")
    print("├── data.yaml")
    print("└── gt_tracking.json")


if __name__ == '__main__':
    main()
