import cv2
import os
import glob
from ultralytics import YOLO

# Config
MODEL_PATH = "runs/detect/yolov8n_custom3/weights/best.pt"

# Folder with Val Images
VALIDATION_IMAGE_FOLDER = "output/images/val/"

# Folder to save results to
RESULTS_FOLDER = "output/validation_results/"

# Confidence threshold
CONF_THRESHOLD = 0.50



def run_predictions():
    """
    Loads the trained model, runs prediction on validation images,
    and saves the images with bounding boxes.
    """
    
    os.makedirs(RESULTS_FOLDER, exist_ok=True)
    print(f"Results will be saved to: {RESULTS_FOLDER}")

    # Load the trained model
    try:
        model = YOLO(MODEL_PATH)
        print(f"Successfully loaded model from: {MODEL_PATH}")
    except Exception as e:
        print(f"Error: Could not load model from {MODEL_PATH}")
        print(f"Details: {e}")
        print("\nPlease make sure the MODEL_PATH variable is set correctly.")
        return

    # Find all images in the validation folder
    image_paths = glob.glob(os.path.join(VALIDATION_IMAGE_FOLDER, "*.jpg"))
    image_paths.extend(glob.glob(os.path.join(VALIDATION_IMAGE_FOLDER, "*.png")))
    
    if not image_paths:
        print(f"Error: No images found in {VALIDATION_IMAGE_FOLDER}.")
        print("Please check the VALIDATION_IMAGE_FOLDER path.")
        return
        
    print(f"Found {len(image_paths)} images to process...")

    # Loop, predict, plot, and save
    for image_path in image_paths:
        try:
            results = model(image_path, conf=CONF_THRESHOLD, device=0)

            result = results[0]
            
            plotted_image = result.plot()

            base_filename = os.path.basename(image_path)
            save_path = os.path.join(RESULTS_FOLDER, base_filename)

            cv2.imwrite(save_path, plotted_image)
            
            print(f"  ... processed and saved {base_filename}")
            
        except Exception as e:
            print(f"Error processing {image_path}: {e}")

    print("\nPrediction complete")
    print(f"All {len(image_paths)} result images saved to {RESULTS_FOLDER}.")

if __name__ == "__main__":
    run_predictions()