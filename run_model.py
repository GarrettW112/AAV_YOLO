import cv2
from ultralytics import YOLO

# --- Configuration ---

# 1. ⚠️ UPDATE THIS PATH ⚠️
# Find your 'best.pt' file. It will be in a folder like:
# 'runs/detect/yolov8n_marker_model/weights/best.pt'
MODEL_PATH = "runs/detect/yolov8n_marker_model3/weights/best.pt"

# 2. Confidence threshold (don't draw boxes with confidence below this)
CONF_THRESHOLD = 0.50

# 3. Webcam Index (0 is usually the default built-in webcam)
CAMERA_INDEX = 0

# --- End of Configuration ---


def main():
    # 1. Load your custom-trained model
    print("Loading model...")
    try:
        model = YOLO(MODEL_PATH)
        print(f"Successfully loaded model from: {MODEL_PATH}")
    except Exception as e:
        print(f"Error: Could not load model from {MODEL_PATH}")
        print(f"Details: {e}")
        print("\nPlease make sure the MODEL_PATH variable is set correctly.")
        return

    # 2. Initialize the webcam
    print(f"Starting webcam (Index {CAMERA_INDEX})...")
    cap = cv2.VideoCapture(CAMERA_INDEX)
    
    if not cap.isOpened():
        print(f"Error: Could not open webcam at index {CAMERA_INDEX}.")
        print("Please check if another application is using the camera.")
        return

    print("Webcam started. Press 'q' to quit.")

    # 3. Start the real-time detection loop
    while True:
        # Read a frame from the camera
        ret, frame = cap.read()
        
        # If the frame was not read correctly, break the loop
        if not ret:
            print("Error: Could not read frame from webcam.")
            break
            
        # 4. Run YOLO prediction on the frame
        # We specify device=0 to use the 4070 Super GPU
        # We set verbose=False to avoid spamming the console
        results = model(
            frame, 
            conf=CONF_THRESHOLD, 
            device=0,
            verbose=False 
        )
        
        # results is a list, get the first (and only) result
        result = results[0]
        
        # 5. Get the plotted image
        # .plot() draws all boxes, labels, and confidences on the frame
        plotted_frame = result.plot()
        
        # 6. Display the frame
        cv2.imshow("YOLO Real-Time Detection", plotted_frame)

        # 7. Check for exit key
        # Wait 1ms for a key press. If 'q' is pressed, exit.
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # 8. Clean up
    print("Shutting down...")
    cap.release()
    cv2.destroyAllWindows()


# --- Run the script ---
if __name__ == "__main__":
    main()