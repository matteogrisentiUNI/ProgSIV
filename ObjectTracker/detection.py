import cv2
import os
from ultralytics import YOLO
from ObjectTracker import utils

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

def detection(image, target_class, output_folder=None):

    print(f"YOLO OBJECT DETECTION")
    
    if image is None:
        raise FileNotFoundError(f"\tError: Image problem")

    # Initialize Object Tracker
    tracker = ObjectTracker(target_class=target_class, conf_threshold=0.1)
    # Perform YOLO detection
    detections = tracker.detect(image)

    # Extract detection data
    boxes = [d['box'] for d in detections]
    masks = [d['mask'] for d in detections]
    confidence = [d['confidence'] for d in detections]
    class_names = [d['class_name'] for d in detections]
    # Draw masks and bounding boxes on the image
    masked_image = utils.draw_mask(image, boxes, masks, class_names)
    
    if output_folder is not None:        # Save the processed image
        processed_image_path = os.path.join(output_folder, "YOLOMasked.jpg")
        cv2.imwrite(processed_image_path, masked_image)

    return masks, boxes

