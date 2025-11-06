import os
import glob
import random
import shutil

# Config
DATA_PATH = "output/"
IMAGE_PATH = os.path.join(DATA_PATH, "images/")
LABEL_PATH = os.path.join(DATA_PATH, "labels/")
VAL_SPLIT_PERCENT = 0.20  # 20% of the data will be used for validation


def create_split():
    print("Creating train/val split...")

    # Create destination folders
    train_img_dir = os.path.join(IMAGE_PATH, "train")
    val_img_dir = os.path.join(IMAGE_PATH, "val")
    train_label_dir = os.path.join(LABEL_PATH, "train")
    val_label_dir = os.path.join(LABEL_PATH, "val")
    
    os.makedirs(train_img_dir, exist_ok=True)
    os.makedirs(val_img_dir, exist_ok=True)
    os.makedirs(train_label_dir, exist_ok=True)
    os.makedirs(val_label_dir, exist_ok=True)

    # Find all generated images (assuming they are .jpg)
    all_images = glob.glob(os.path.join(IMAGE_PATH, "*.jpg"))
    if not all_images:
        print(f"Error: No .jpg images found in {IMAGE_PATH}. Did you run data generation?")
        return
        
    # Shuffle and split
    random.shuffle(all_images)
    split_index = int(len(all_images) * VAL_SPLIT_PERCENT)
    val_images = all_images[:split_index]
    train_images = all_images[split_index:]

    # Function to move files
    def move_files(file_list, img_dest_dir, label_dest_dir):
        moved_count = 0
        for img_path in file_list:
            try:
                # Get the base filename (e.g., "marker_synth_00001")
                base_filename = os.path.basename(img_path)
                file_root, _ = os.path.splitext(base_filename)
                
                # Define the corresponding label path
                label_filename = f"{file_root}.txt"
                label_path = os.path.join(LABEL_PATH, label_filename)

                # Define destination paths
                new_img_path = os.path.join(img_dest_dir, base_filename)
                new_label_path = os.path.join(label_dest_dir, label_filename)

                # Move the files
                if os.path.exists(label_path):
                    shutil.move(img_path, new_img_path)
                    shutil.move(label_path, new_label_path)
                    moved_count += 1
                else:
                    print(f"Warning: Label not found for {img_path}. Skipping.")
                    
            except Exception as e:
                print(f"Error moving {img_path}: {e}")
        return moved_count

    # Move the files
    print(f"Moving {len(val_images)} files to validation set...")
    val_moved = move_files(val_images, val_img_dir, val_label_dir)
    
    print(f"Moving {len(train_images)} files to training set...")
    train_moved = move_files(train_images, train_img_dir, train_label_dir)
    
    print("\nSplit complete.")
    print(f"  Total images: {len(all_images)}")
    print(f"  Training set: {train_moved} images")
    print(f"  Validation set: {val_moved} images")

if __name__ == "__main__":
    create_split()