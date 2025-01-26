import cv2
import numpy as np
import os
from ObjectTracker import ObjectTracker, draw_mask, mask_motion_estimation, motion_estimation

def process_image(image, target_class, output_folder=None):

    print(f"YOLO PROCESS IMAGE")
    
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

def test_motion_estimantion_video():
    
    cap = cv2.VideoCapture('Demo/Video/Car3.mp4')
    output_folder = 'Demo/MotionEstimation/Car3'
    os.makedirs(output_folder, exist_ok=True)
    
    # to elaborate the first and second frame we have to extract the poi  
    ret, previus_frame = cap.read()            # read first frame 
    ret, next_frame = cap.read()               # read second frame

    # Finds the mask of the object we want to tracj using YOLO
    masks,boxes = process_image(previus_frame, 'car', output_folder=output_folder)
    mask = masks[0].astype(np.uint8)
    box = boxes[0]
    box = box.astype(np.float32)

    good_next_poi, good_previus_poi, Hgl = mask_motion_estimation(previus_frame, next_frame, mask=mask, output_folder=output_folder)
    box = cv2.perspectiveTransform(box.reshape(-1, 1, 2), Hgl).reshape(-1, 4)
    x1, y1, x2, y2 = box[0]

    color_poi = (0, 255, 0)         # color point of interest
    color_box = (255, 0, 255)       # color box
    
    # Draws the optical flow tracks
    for i, (new, old) in enumerate(zip(good_next_poi, good_previus_poi)):
        a, b = new.ravel()      # Returns a contiguous flattened array as (x, y) coordinates for new point
        c, d = old.ravel()      # Returns a contiguous flattened array as (x, y) coordinates for old point

        frame = cv2.circle(next_frame, (int(a), int(b)), 3, color_poi, -1)                      # Draws filled circle (thickness of -1) at new position with green color and radius of 3

    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)) , color_box, 2)  # draw the box
    
    resized_frame = cv2.resize(frame, (int(frame.shape[1] / 2), int(frame.shape[0] / 2)), interpolation=cv2.INTER_AREA)
    cv2.imshow("sparse optical flow", resized_frame)
    if cv2.waitKey(10) & 0xFF == ord('q'):
        cap.release()
        cv2.destroyAllWindows()
        return 
    
    previus_frame = cv2.cvtColor(next_frame, cv2.COLOR_BGR2GRAY) 
    previus_poi = good_next_poi.reshape(-1, 1, 2)

    
    while(cap.isOpened()):
        
        ret, next_frame_color = cap.read()
        next_frame = cv2.cvtColor(next_frame_color, cv2.COLOR_BGR2GRAY)

        good_previus_poi, good_next_poi, Hgl = motion_estimation(previus_frame, next_frame, previus_poi)
        
        box = box.reshape(-1, 2) 
        box = cv2.perspectiveTransform(box.reshape(-1, 1, 2), Hgl).reshape(-1, 4)
        x1, y1, x2, y2 = box[0]

        for i, (new, old) in enumerate(zip(good_next_poi, good_previus_poi)):
            a, b = new.ravel()      # Returns a contiguous flattened array as (x, y) coordinates for new point
            c, d = old.ravel()      # Returns a contiguous flattened array as (x, y) coordinates for old point

            frame = cv2.circle(next_frame_color, (int(a), int(b)), 3, color_poi, -1)                      # Draws filled circle (thickness of -1) at new position with green color and radius of 3

        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)) , color_box, 2)  # draw the box
    
        previus_frame = next_frame
        previus_poi = good_next_poi.reshape(-1, 1, 2)
        
        resized_frame = cv2.resize(frame, (int(frame.shape[1] / 2), int(frame.shape[0] / 2)), interpolation=cv2.INTER_AREA)
        cv2.imshow("sparse optical flow", resized_frame)
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break
   
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    test_motion_estimantion_video()