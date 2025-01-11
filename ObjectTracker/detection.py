from ultralytics import YOLO

class YOLODetector:
    def __init__(self, model_path="yolov8n-seg.pt"):
        self.model = YOLO(model_path)

    def detect(self, frame, target_class):
        results = self.model(frame)[0]
        for result in results:
            if target_class in result.names[result.boxes.cls[0]]:
                return result.masks.data[0], result.boxes.xyxy[0]
        return None, None
