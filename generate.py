import cv2
import numpy as np
import os
import random

# Config

# Input Paths
MARKER_PATH = "marker.png"
CONTROL_PATH = "control.png"
BACKGROUND_FOLDER = "backgrounds/"

# Output Paths
OUTPUT_IMAGE_FOLDER = "output/images/"
OUTPUT_LABEL_FOLDER = "output/labels/"

# Generation Settings
NUM_IMAGES_TO_GENERATE = 1000
MARKER_CLASS_ID = 0
MAX_CONTROL_PLACEMENT_ATTEMPTS = 50 # Max tries to place the control image

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
        # This can happen if the paste location is at the very edge
        # We should rather check this *before* calling the function
        # But as a safety, we can clip
        print(f"Warning: Foreground at ({x_offset}, {y_offset}) with size ({fg_w}, {fg_h}) is partially out of bounds on background ({bg_w}, {bg_h}). Clipping.")
        # Calculate valid region
        y_end = min(y_offset + fg_h, bg_h)
        x_end = min(x_offset + fg_w, bg_w)
        
        fg_h = y_end - y_offset
        fg_w = x_end - x_offset
        
        # If it's completely out, return failure
        if fg_h <= 0 or fg_w <= 0:
            return None, None

        # Clip the foreground image
        foreground = foreground[0:fg_h, 0:fg_w]
        
    # Get the region of interest (ROI) from the background
    roi = background[y_offset : y_offset + fg_h, x_offset : x_offset + fg_w]

    # Split the foreground into RGB and Alpha channels
    foreground_rgb = foreground[:, :, :3]
    alpha_mask = foreground[:, :, 3]
    
    # Use the alpha_blend function
    blended_roi = alpha_blend(foreground, roi, alpha_mask)

    # Create a copy of the background to modify
    background_copy = background.copy()
    background_copy[y_offset : y_offset + fg_h, x_offset : x_offset + fg_w] = blended_roi
    
    return background_copy, (x_offset, y_offset, fg_w, fg_h)

# --- New Helper Function ---
def is_overlapping(box1, box2):
    """
    Checks if two bounding boxes (x, y, w, h) overlap.
    """
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2
    
    # Convert to (x_min, y_min, x_max, y_max)
    box1_x_min, box1_y_min = x1, y1
    box1_x_max, box1_y_max = x1 + w1, y1 + h1
    
    box2_x_min, box2_y_min = x2, y2
    box2_x_max, box2_y_max = x2 + w2, y2 + h2

    # Check for non-overlap
    if (box1_x_max <= box2_x_min or  # box1 is left of box2
        box1_x_min >= box2_x_max or  # box1 is right of box2
        box1_y_max <= box2_y_min or  # box1 is above box2
        box1_y_min >= box2_y_max): # box1 is below box2
        return False
    
    # If none of the non-overlap conditions are met, they overlap
    return True
# --- End of New Helper Function ---


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
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])

    # New width and height
    new_w = int((h * sin) + (w * cos))
    new_h = int((h * cos) + (w * sin))

    # Adjust the rotation matrix to account for translation (to keep it centered)
    M[0, 2] += (new_w / 2) - cX
    M[1, 2] += (new_h / 2) - cY

    # Perform the actual rotation and return the image
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

    # --- New: Load Control Image ---
    control_orig = cv2.imread(CONTROL_PATH, cv2.IMREAD_UNCHANGED)
    if control_orig is None:
        print(f"Error: Could not load control image from {CONTROL_PATH}.")
        return
    if control_orig.shape[2] != 4:
        print(f"Error: Control image at {CONTROL_PATH} must have an alpha channel (4 channels).")
        return
    print(f"✅ Control image loaded successfully ({control_orig.shape[1]}x{control_orig.shape[0]}).")
    # --- End New ---

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
            min_bg_dim = min(bg_h, bg_w)
            scale = random.uniform(SCALE_RANGE[0], SCALE_RANGE[1])
            new_size = int(min_bg_dim * scale)
            new_size = max(1, new_size) 
            
            marker_scaled = cv2.resize(
                marker_orig, (new_size, new_size), 
                interpolation=cv2.INTER_AREA
            )

            angle = random.uniform(ROTATION_RANGE[0], ROTATION_RANGE[1])
            marker_final = rotate_marker(marker_scaled, angle)
            final_h, final_w = marker_final.shape[:2]

            # --- Find Paste Location for Marker ---
            if final_h >= bg_h or final_w >= bg_w:
                print(f"Warning: Scaled marker ({final_w}x{final_h}) is larger than background ({bg_w}x{bg_h}). Rescaling...")
                scale_factor = min((bg_h - 1) / final_h, (bg_w - 1) / final_w)
                final_w = int(final_w * scale_factor)
                final_h = int(final_h * scale_factor)
                # Ensure at least 1x1
                final_w = max(1, final_w)
                final_h = max(1, final_h)
                marker_final = cv2.resize(marker_final, (final_w, final_h), interpolation=cv2.INTER_AREA)

            max_x = bg_w - final_w
            max_y = bg_h - final_h
            
            paste_x = random.randint(0, max_x)
            paste_y = random.randint(0, max_y)

            # --- Paste Marker onto Background ---
            result_img, marker_box = paste_with_transparency(
                bg_img, marker_final, paste_x, paste_y
            )
            
            if result_img is None:
                continue # Pasting failed, skip this iteration

            # --- New: Prepare and Paste Control Image ---
            
            # 1. Prepare Control (Scale & Rotate)
            # We'll reuse the same scale and rotation ranges
            scale_c = random.uniform(SCALE_RANGE[0], SCALE_RANGE[1])
            new_size_c = int(min_bg_dim * scale_c)
            new_size_c = max(1, new_size_c)
            
            control_scaled = cv2.resize(
                control_orig, (new_size_c, new_size_c), 
                interpolation=cv2.INTER_AREA
            )
            
            angle_c = random.uniform(ROTATION_RANGE[0], ROTATION_RANGE[1])
            control_final = rotate_marker(control_scaled, angle_c)
            control_h, control_w = control_final.shape[:2]

            # 2. Find Non-Overlapping Paste Location for Control
            control_pasted = False
            for _ in range(MAX_CONTROL_PLACEMENT_ATTEMPTS):
                # Ensure control fits
                if control_h >= bg_h or control_w >= bg_w:
                    break # Control is too big, can't place it
                
                max_x_c = bg_w - control_w
                max_y_c = bg_h - control_h
                
                paste_x_c = random.randint(0, max_x_c)
                paste_y_c = random.randint(0, max_y_c)
                
                control_box = (paste_x_c, paste_y_c, control_w, control_h)
                
                # Check for overlap with the marker
                if not is_overlapping(marker_box, control_box):
                    # Found a valid spot!
                    # Paste it onto result_img (which already has the marker)
                    result_img_with_control, _ = paste_with_transparency(
                        result_img, control_final, paste_x_c, paste_y_c
                    )
                    
                    if result_img_with_control is not None:
                        result_img = result_img_with_control # Update image
                        control_pasted = True
                    
                    break # Exit the attempt loop

            # if not control_pasted:
            #     print(f"Warning: Could not place control for image {i+1}")
            
            # --- End of New Control Logic ---


            # --- Convert Marker to YOLO Format (Unchanged) ---
            # This logic only concerns the marker, as requested
            x_min, y_min, box_w, box_h = marker_box

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
            
            # Save image (which now has marker + control)
            cv2.imwrite(img_path, result_img)
            
            # Save label (which only has marker)
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