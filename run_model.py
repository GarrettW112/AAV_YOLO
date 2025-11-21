import cv2
from ultralytics import YOLO

# Config
MODEL_PATH = "runs/detect/yolov8n_marker_model3/weights/best.pt"

# Confidence threshold
CONF_THRESHOLD = 0.50

# Webcam Index
CAMERA_INDEX = 0



def main():
    print("Loading model...")
    try:
        model = YOLO(MODEL_PATH)
        print(f"Successfully loaded model from: {MODEL_PATH}")
    except Exception as e:
        print(f"Error: Could not load model from {MODEL_PATH}")
        print(f"Details: {e}")
        print("\nPlease make sure the MODEL_PATH variable is set correctly.")
        return

    # Initialize the webcam
    print(f"Starting webcam (Index {CAMERA_INDEX})...")
    cap = cv2.VideoCapture(CAMERA_INDEX)
    
    if not cap.isOpened():
        print(f"Error: Could not open webcam at index {CAMERA_INDEX}.")
        print("Please check if another application is using the camera.")
        return

    print("Webcam started. Press 'q' to quit.")

    # Start the real-time detection loop
    while True:
        # Read a frame from the camera
        ret, frame = cap.read()
        
        # If the frame was not read correctly, break the loop
        if not ret:
            print("Error: Could not read frame from webcam.")
            break
            
        # Run YOLO prediction on the frame
        results = model(
            frame, 
            conf=CONF_THRESHOLD, 
            device=0,
            verbose=False 
        )
        
        result = results[0]
        
        # Get the plotted image
        # .plot() draws all boxes, labels, and confidences on the frame
        plotted_frame = result.plot()
        
        # Display the frame
        cv2.imshow("YOLO Real-Time Detection", plotted_frame)

        # Check for exit key
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Clean up
    print("Shutting down...")
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()