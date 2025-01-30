import cv2
import os
from ultralytics import YOLO
import numpy as np

def draw_mask(frame, boxes, masks, class_names):
    for box, mask, class_name in zip(boxes, masks, class_names):

        #Check if the mask is valid
        if mask is not None:

            # Resize the mask to match the frame's size
            resized_mask = cv2.resize(mask, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_NEAREST)

            # Convert mask to boolean for indexing
            boolean_mask = resized_mask.astype(bool)

            # Create a pink overlay with the same shape as the frame
            pink_overlay = np.zeros_like(frame, dtype=np.uint8)
            pink_overlay[:] = (255, 105, 180)  # Pink in BGR

            # Apply the translucent pink overlay only on the masked area
            frame = np.where(boolean_mask[:, :, None], cv2.addWeighted(frame, 0.5, pink_overlay, 0.5, 0), frame)

        #Check if the bounding box is valid
        if box is not None:
            # Draw the bounding box in blue
            x1, y1, x2, y2 = box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)

        # Check if the name is valid
        if class_name is not None:
            # Put the class name on the top-right corner of the box
            cv2.putText(frame, class_name, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (255, 255, 255), 2, cv2.LINE_AA)
    return frame

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
    class_names = [d['class_name'] for d in detections]

    # Draw masks and bounding boxes on the image
    masked_image = draw_mask(image, boxes, masks, class_names)

    if output_folder is not None:        # Save the processed image
        processed_image_path = os.path.join(output_folder, "YOLOMasked.jpg")
        cv2.imwrite(processed_image_path, masked_image)

    return masks,boxes

