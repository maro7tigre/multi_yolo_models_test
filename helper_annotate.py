import os
import csv

def annotate_classification_dataset(root_dir, output_csv='annotations.csv'):
    rows = []
    for split in ['train', 'val']:
        split_dir = os.path.join(root_dir, split)
        if not os.path.exists(split_dir):
            continue

        for class_name in os.listdir(split_dir):
            class_path = os.path.join(split_dir, class_name)
            if not os.path.isdir(class_path):
                continue

            for img_file in os.listdir(class_path):
                if img_file.lower().endswith(('.jpg', '.jpeg', '.png')):
                    image_path = os.path.join(split, class_name, img_file)  # relative path
                    rows.append([image_path, class_name])

    # Save as CSV
    with open(output_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['image_path', 'label'])
        writer.writerows(rows)

    print(f"Annotated {len(rows)} images and saved to {output_csv}")

# Example usage:
annotate_classification_dataset('dataset')
