from ultralytics import YOLO

def predict(image_path):
    model = YOLO("output/traffic_light/weights/best.pt")
    results = model(image_path)
    results.show()

if __name__ == "__main__":
    predict("data/sample/test.jpg")
