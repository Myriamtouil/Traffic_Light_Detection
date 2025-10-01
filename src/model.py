from ultralytics import YOLO

def load_model(model_path="yolov8n.pt"):
    # Load pretrained YOLO model (can be fine-tuned)
    model = YOLO(model_path)
    return model
