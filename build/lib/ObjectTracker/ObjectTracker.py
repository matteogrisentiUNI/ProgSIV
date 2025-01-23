from ultralytics import YOLO

class ObjectTracker:
    def __init__(self, model_path=None, target_class=None, conf_threshold=0.5):
        self.model = YOLO(model_path) if model_path else YOLO('yolov8n-seg.pt')
        self.target_class = target_class
        self.conf_threshold = conf_threshold

    def detect(self, frame):
        results = self.model(frame)[0]
        detections = []

        for result in results:
            # Check if masks are available
            if result.masks is None:
                continue  # Skip if no mask is detected

            for box, mask, conf, cls_id in zip(result.boxes.xyxy, result.masks.data, result.boxes.conf, result.boxes.cls):
                class_name = self.model.names[int(cls_id)]
                if class_name == self.target_class and conf >= self.conf_threshold:
                    detections.append({
                        'box': box.cpu().numpy().astype(int),
                        'mask': mask.cpu().numpy(),
                        'confidence': float(conf),
                        'class_name': class_name
                    })
        return detections

