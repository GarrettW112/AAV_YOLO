import cv2
import os
import glob
from ultralytics import YOLO

# --- Configuration ---

# 1. ⚠️ UPDATE THIS PATH ⚠️
# Find your 'best.pt' file. It will be in a folder like:
# 'runs/detect/yolov8n_marker_model/weights/best.pt'
MODEL_PATH = "runs/detect/yolov8n_marker_model3/weights/best.pt"

# 2. Folder with your validation images
VALIDATION_IMAGE_FOLDER = "output/images/val/"

# 3. Folder to save results to
RESULTS_FOLDER = "output/validation_results/"

# 4. Confidence threshold (don't draw boxes with confidence below this)
CONF_THRESHOLD = 0.50

# --- End of Configuration ---


def run_predictions():
    """
    Loads the trained model, runs prediction on validation images,
    and saves the images with bounding boxes.
    """
    
    # 1. Ensure the output folder exists
    os.makedirs(RESULTS_FOLDER, exist_ok=True)
    print(f"Results will be saved to: {RESULTS_FOLDER}")

    # 2. Load the trained model
    try:
        model = YOLO(MODEL_PATH)
        print(f"Successfully loaded model from: {MODEL_PATH}")
    except Exception as e:
        print(f"Error: Could not load model from {MODEL_PATH}")
        print(f"Details: {e}")
        print("\nPlease make sure the MODEL_PATH variable is set correctly.")
        return

    # 3. Find all images in the validation folder
    image_paths = glob.glob(os.path.join(VALIDATION_IMAGE_FOLDER, "*.jpg"))
    image_paths.extend(glob.glob(os.path.join(VALIDATION_IMAGE_FOLDER, "*.png")))
    
    if not image_paths:
        print(f"Error: No images found in {VALIDATION_IMAGE_FOLDER}.")
        print("Please check the VALIDATION_IMAGE_FOLDER path.")
        return
        
    print(f"Found {len(image_paths)} images to process...")

    # 4. Loop, predict, plot, and save
    for image_path in image_paths:
        try:
            # Run prediction
            # We set 'device=0' to ensure it uses the GPU
            results = model(image_path, conf=CONF_THRESHOLD, device=0)
            
            # results is a list, get the first (and only) result
            result = results[0]
            
            # Use the built-in .plot() method
            # This returns a NumPy array of the image with boxes/labels drawn
            plotted_image = result.plot()
            
            # Define the save path
            base_filename = os.path.basename(image_path)
            save_path = os.path.join(RESULTS_FOLDER, base_filename)
            
            # Save the plotted image using OpenCV
            cv2.imwrite(save_path, plotted_image)
            
            print(f"  ... processed and saved {base_filename}")
            
        except Exception as e:
            print(f"Error processing {image_path}: {e}")

    print("\n🎉 Prediction complete!")
    print(f"All {len(image_paths)} result images saved to {RESULTS_FOLDER}.")


# --- Run the script ---
if __name__ == "__main__":
    run_predictions()