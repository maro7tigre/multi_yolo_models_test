import os
import shutil
import random

def prepare_dataset(complete_path='complete_dataset', output_path='dataset', val_ratio=0.2):
    # Step 1: Clean up dataset folder if it exists
    if os.path.exists(output_path):
        shutil.rmtree(output_path)
    os.makedirs(os.path.join(output_path, 'train'))
    os.makedirs(os.path.join(output_path, 'val'))

    # Step 2: For each class in complete_dataset
    for class_name in os.listdir(complete_path):
        class_dir = os.path.join(complete_path, class_name)
        if not os.path.isdir(class_dir):
            continue

        # Create class folders in train and val
        train_class_path = os.path.join(output_path, 'train', class_name)
        val_class_path = os.path.join(output_path, 'val', class_name)
        os.makedirs(train_class_path, exist_ok=True)
        os.makedirs(val_class_path, exist_ok=True)

        # List images
        images = [
            f for f in os.listdir(class_dir)
            if os.path.isfile(os.path.join(class_dir, f)) and f.lower().endswith(('.jpg', '.jpeg', '.png'))
        ]
        random.shuffle(images)

        split_idx = int(len(images) * (1 - val_ratio))
        train_images = images[:split_idx]
        val_images = images[split_idx:]

        # Copy train images
        for img in train_images:
            src = os.path.join(class_dir, img)
            dst = os.path.join(train_class_path, img)
            shutil.copy2(src, dst)

        # Copy val images
        for img in val_images:
            src = os.path.join(class_dir, img)
            dst = os.path.join(val_class_path, img)
            shutil.copy2(src, dst)

        print(f"{class_name}: {len(train_images)} train, {len(val_images)} val")


if __name__ == '__main__':
    prepare_dataset()
