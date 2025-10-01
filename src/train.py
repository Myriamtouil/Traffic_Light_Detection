from ultralytics import YOLO

def train_model():
    model = YOLO("yolov8n.pt")  # base model
    
    # Train model (parameters tuned internally)
    model.train(
        data="data/data.yaml",  # dataset YAML file
        epochs=100,
        imgsz=512,
        batch=32,
        project="./output",
        name="traffic_light"
        # Other parameters were tuned but are hidden for clarity
    )

if __name__ == "__main__":
    train_model()
