import cv2
import numpy as np
import os
import random

# Config

# Input Paths
MARKER_PATH = "marker.png"
BACKGROUND_FOLDER = "backgrounds/"

# Output Paths
OUTPUT_IMAGE_FOLDER = "output/images/"
OUTPUT_LABEL_FOLDER = "output/labels/"

# Generation Settings
NUM_IMAGES_TO_GENERATE = 1000
MARKER_CLASS_ID = 0

# Augmentation Settings
# Scale of the marker relative to the background's smallest dimension
SCALE_RANGE = (0.05, 0.20)  # 5% to 20% of the background's height/width
ROTATION_RANGE = (0, 360)   # Min and max rotation angle in degrees

# End of Config


def create_output_folders():
    """Ensures the output directories for images and labels exist."""
    os.makedirs(OUTPUT_IMAGE_FOLDER, exist_ok=True)
    os.makedirs(OUTPUT_LABEL_FOLDER, exist_ok=True)


def get_background_files(folder):
    """Gets a list of all valid image files in the background folder."""
    valid_extensions = {".jpg", ".jpeg", ".png"}
    background_files = []
    for f in os.listdir(folder):
        if os.path.splitext(f)[1].lower() in valid_extensions:
            background_files.append(os.path.join(folder, f))
    if not background_files:
        print(f"Error: No background images found in {folder}.")
        exit()
    return background_files


def alpha_blend(foreground, background, alpha):
    """
    Performs alpha blending to paste a foreground (with alpha) onto a background.
    The alpha channel is used as a mask.
    """
    # Convert alpha mask to 3 channels for broadcasting
    alpha_3channel = cv2.cvtColor(alpha, cv2.COLOR_GRAY2BGR).astype(float) / 255.0
    
    # Split foreground into RGB and Alpha
    foreground_rgb = foreground[:, :, :3].astype(float)
    
    # Blend
    blended_portion = (alpha_3channel * foreground_rgb) + \
                      ((1 - alpha_3channel) * background.astype(float))
    
    return blended_portion.astype(np.uint8)

def paste_with_transparency(background, foreground, x_offset, y_offset):
    """
    Pastes a foreground image (with alpha channel) onto a background image
    at the specified x, y offset.
    """
    bg_h, bg_w = background.shape[:2]
    fg_h, fg_w = foreground.shape[:2]

    # Ensure the foreground fits within the background
    if y_offset + fg_h > bg_h or x_offset + fg_w > bg_w:
        print("Warning: Marker is partially out of bounds. Skipping.")
        return None, None  # Indicate failure

    # Get the region of interest (ROI) from the background
    roi = background[y_offset : y_offset + fg_h, x_offset : x_offset + fg_w]

    # Split the foreground into RGB and Alpha channels
    # The [:, :, 3] gets the 4th channel (alpha)
    foreground_rgb = foreground[:, :, :3]
    alpha_mask = foreground[:, :, 3]
    
    # Use the alpha_blend function
    blended_roi = alpha_blend(foreground, roi, alpha_mask)

    # Create a copy of the background to modify
    background_copy = background.copy()
    background_copy[y_offset : y_offset + fg_h, x_offset : x_offset + fg_w] = blended_roi
    
    return background_copy, (x_offset, y_offset, fg_w, fg_h)


def rotate_marker(image, angle):
    """
    Rotates an image (with alpha) by a given angle, resizing the canvas
    to fit the new bounding box perfectly.
    """
    # Get image dimensions
    (h, w) = image.shape[:2]
    # Get the center
    (cX, cY) = (w // 2, h // 2)

    # Get the rotation matrix
    M = cv2.getRotationMatrix2D((cX, cY), angle, 1.0)
    
    # --- Calculate new bounding box ---
    # We need to rotate the 4 corners of the image to find the new bounds
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])

    # New width and height
    new_w = int((h * sin) + (w * cos))
    new_h = int((h * cos) + (w * sin))

    # Adjust the rotation matrix to account for translation (to keep it centered)
    M[0, 2] += (new_w / 2) - cX
    M[1, 2] += (new_h / 2) - cY

    # Perform the actual rotation and return the image
    # cv2.INTER_LINEAR is good for quality
    # borderValue=(0,0,0,0) makes the new background transparent
    rotated_image = cv2.warpAffine(
        image, M, (new_w, new_h), 
        flags=cv2.INTER_LINEAR, 
        borderValue=(0, 0, 0, 0)
    )
    return rotated_image


def generate_training_data():
    """Main function to generate the dataset."""
    
    print("🚀 Starting data generation...")
    create_output_folders()
    
    # 1. Load the marker image (with alpha channel)
    marker_orig = cv2.imread(MARKER_PATH, cv2.IMREAD_UNCHANGED)
    if marker_orig is None:
        print(f"Error: Could not load marker image from {MARKER_PATH}.")
        return
    if marker_orig.shape[2] != 4:
        print(f"Error: Marker image at {MARKER_PATH} must have an alpha channel (4 channels).")
        return
        
    print(f"✅ Marker image loaded successfully ({marker_orig.shape[1]}x{marker_orig.shape[0]}).")

    # 2. Get list of background images
    background_files = get_background_files(BACKGROUND_FOLDER)
    print(f"✅ Found {len(background_files)} background images.")
    
    # 3. Start generation loop
    for i in range(NUM_IMAGES_TO_GENERATE):
        try:
            # --- Load Background ---
            bg_path = random.choice(background_files)
            bg_img = cv2.imread(bg_path)
            if bg_img is None:
                print(f"Warning: Could not read {bg_path}. Skipping.")
                continue
            bg_h, bg_w = bg_img.shape[:2]

            # --- Prepare Marker (Scale & Rotate) ---
            
            # 1. Scale
            min_bg_dim = min(bg_h, bg_w)
            scale = random.uniform(SCALE_RANGE[0], SCALE_RANGE[1])
            new_size = int(min_bg_dim * scale)
            # Ensure marker is at least 1x1
            new_size = max(1, new_size) 
            
            marker_scaled = cv2.resize(
                marker_orig, (new_size, new_size), 
                interpolation=cv2.INTER_AREA
            )

            # 2. Rotate
            angle = random.uniform(ROTATION_RANGE[0], ROTATION_RANGE[1])
            marker_final = rotate_marker(marker_scaled, angle)
            final_h, final_w = marker_final.shape[:2]

            # --- Find Paste Location ---
            # Ensure the marker fits on the background
            if final_h >= bg_h or final_w >= bg_w:
                print(f"Warning: Scaled marker ({final_w}x{final_h}) is larger than background ({bg_w}x{bg_h}). Rescaling...")
                # Fallback: resize to fit
                scale_factor = min((bg_h - 1) / final_h, (bg_w - 1) / final_w)
                final_w = int(final_w * scale_factor)
                final_h = int(final_h * scale_factor)
                marker_final = cv2.resize(marker_final, (final_w, final_h), interpolation=cv2.INTER_AREA)

            max_x = bg_w - final_w
            max_y = bg_h - final_h
            
            paste_x = random.randint(0, max_x)
            paste_y = random.randint(0, max_y)

            # --- Paste Marker onto Background ---
            # This function handles the alpha blending
            result_img, box = paste_with_transparency(
                bg_img, marker_final, paste_x, paste_y
            )
            
            if result_img is None:
                continue # Pasting failed, skip this iteration

            # --- Convert to YOLO Format ---
            # box = (x_min, y_min, w, h) in pixels
            x_min, y_min, box_w, box_h = box

            # YOLO format (class_id, x_center_norm, y_center_norm, w_norm, h_norm)
            x_center = x_min + box_w / 2
            y_center = y_min + box_h / 2
            
            x_center_norm = x_center / bg_w
            y_center_norm = y_center / bg_h
            w_norm = box_w / bg_w
            h_norm = box_h / bg_h
            
            yolo_string = f"{MARKER_CLASS_ID} {x_center_norm} {y_center_norm} {w_norm} {h_norm}"

            # --- Save Image and Label ---
            base_filename = f"marker_synth_{i+1:05d}"
            img_path = os.path.join(OUTPUT_IMAGE_FOLDER, f"{base_filename}.jpg")
            label_path = os.path.join(OUTPUT_LABEL_FOLDER, f"{base_filename}.txt")
            
            # Save image (use .jpg for smaller size, or .png to preserve quality)
            cv2.imwrite(img_path, result_img)
            
            # Save label
            with open(label_path, 'w') as f:
                f.write(yolo_string)
            
            if (i + 1) % 10 == 0:
                print(f"   ... Generated {i+1}/{NUM_IMAGES_TO_GENERATE} images")

        except Exception as e:
            print(f"An error occurred during generation: {e}. Skipping image {i+1}.")
            
    print(f"\n🎉 Generation complete! {NUM_IMAGES_TO_GENERATE} images and labels saved to 'output/'.")


# --- Run the generator ---
if __name__ == "__main__":
    generate_training_data()