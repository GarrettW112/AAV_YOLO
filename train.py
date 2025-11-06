from ultralytics import YOLO

def main():
    model = YOLO('yolov8n.pt')
    print("Starting model training...")

    try:
        results = model.train(
            data='dataset.yaml',
            epochs=100,
            imgsz=640,
            batch=32,
            name='yolov8n_custom',
            device=0   # For GPU
        )
        print("Training complete.")
        print(f"Model and results saved to: {results.save_dir}")
    except Exception as e:
        print(f"An error occurred during training: {e}")

if __name__ == '__main__':
    main()