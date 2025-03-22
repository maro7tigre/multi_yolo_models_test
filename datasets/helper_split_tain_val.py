import os
import shutil
import random
import cv2
import numpy as np

def prepare_dataset(complete_path='complete_dataset', output_path='images', 
                   val_ratio=0.2, img_width=640, img_height=640, 
                   normalize=True, norm_range=(-1, 1), class_dirs=None):
    """
    Prepares a dataset by splitting images into train and validation sets
    and resizing them to the specified dimensions.
    
    Args:
        complete_path: Path to the source dataset folder
        output_path: Path where the processed dataset will be saved
        val_ratio: Proportion of images to use for validation
        img_width: Target width for resized images
        img_height: Target height for resized images
        normalize: Whether to normalize pixel values
        norm_range: Target range for normalization, either (0,1) or (-1,1)
        class_dirs: List of specific class directories to process (e.g., ['cardboard', 'glass']),
                   if None, all directories in complete_path will be processed
    """
    # Step 1: Clean up dataset folder if it exists
    if os.path.exists(output_path):
        shutil.rmtree(output_path)
    os.makedirs(os.path.join(output_path, 'train'))
    os.makedirs(os.path.join(output_path, 'val'))

    # Step 2: Get list of class directories to process
    all_dirs = [d for d in os.listdir(complete_path) if os.path.isdir(os.path.join(complete_path, d))]
    
    # If class_dirs is provided, filter to only those directories that exist
    if class_dirs is not None:
        # Verify each requested directory exists
        dirs_to_process = []
        for class_name in class_dirs:
            if class_name in all_dirs:
                dirs_to_process.append(class_name)
            else:
                print(f"Warning: Requested class '{class_name}' not found in {complete_path}")
    else:
        # Process all directories
        dirs_to_process = all_dirs
    
    # Exit if no valid directories found
    if not dirs_to_process:
        print(f"Error: No valid class directories found to process")
        return

    # Step 3: Process each class directory
    for class_name in dirs_to_process:
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

        # Process and copy train images
        for img in train_images:
            process_and_save_image(
                src_path=os.path.join(class_dir, img),
                dst_path=os.path.join(train_class_path, img),
                width=img_width,
                height=img_height,
                normalize=normalize,
                norm_range=norm_range
            )

        # Process and copy val images
        for img in val_images:
            process_and_save_image(
                src_path=os.path.join(class_dir, img),
                dst_path=os.path.join(val_class_path, img),
                width=img_width,
                height=img_height,
                normalize=normalize,
                norm_range=norm_range
            )

        print(f"{class_name}: {len(train_images)} train, {len(val_images)} val")


def process_and_save_image(src_path, dst_path, width, height, normalize=True, norm_range=(-1, 1)):
    """
    Reads an image, resizes it to the specified dimensions, normalizes pixel values,
    and saves it.
    
    Args:
        src_path: Path to the source image
        dst_path: Path where the processed image will be saved
        width: Target width for the image
        height: Target height for the image
        normalize: Whether to normalize pixel values
        norm_range: Target range for normalization, either (0,1) or (-1,1)
    """
    try:
        # Read the image
        img = cv2.imread(src_path)
        if img is None:
            print(f"Warning: Could not read image {src_path}")
            return
            
        # Resize the image
        resized_img = cv2.resize(img, (width, height), interpolation=cv2.INTER_AREA)
        
        # Normalize pixel values if requested
        if normalize:
            # Convert to float32 for normalization
            resized_img = resized_img.astype(np.float32)
            
            # Standard normalization to [0,1]
            resized_img = resized_img / 255.0
            
            # Adjust to target range if needed
            if norm_range == (-1, 1):
                # Scale from [0,1] to [-1,1]
                resized_img = resized_img * 2.0 - 1.0
            
            # For saving normalized images, we need to convert back to uint8 [0,255]
            if dst_path.endswith(('.jpg', '.jpeg', '.png')):
                # Rescale based on the normalization used
                if norm_range == (-1, 1):
                    save_img = ((resized_img + 1.0) / 2.0 * 255.0).astype(np.uint8)
                else:
                    save_img = (resized_img * 255.0).astype(np.uint8)
                cv2.imwrite(dst_path, save_img)
                
                # Also save the normalized data in .npy format for direct loading
                npy_path = os.path.splitext(dst_path)[0] + '.npy'
                np.save(npy_path, resized_img)
            else:
                # If the destination isn't a standard image format, save as .npy
                np.save(dst_path, resized_img)
        else:
            # Save the non-normalized resized image
            cv2.imwrite(dst_path, resized_img)
    except Exception as e:
        print(f"Error processing {src_path}: {str(e)}")


if __name__ == '__main__':
    # Example usage with default parameters (process all class directories)
    prepare_dataset(class_dirs=['cardboard', 'glass', 'metal', 'paper', 'plastic'])
    
    # Example usage with specific class directories
    # prepare_dataset(class_dirs=['cardboard', 'glass', 'metal'])