import cv2
import numpy as np
import os
from ObjectTracker import ObjectTracker, draw_mask

# GLOBAL MOTION PART
# Combina più immagini in un'unica immagine come collage.
def stack_images(scale, img_array):
    rows = len(img_array)
    cols = len(img_array[0])
    row_images = [len(row) == cols for row in img_array]
    if not all(row_images):
        print("Errore: Tutte le righe devono avere lo stesso numero di immagini.")
        return None
    height, width, _ = img_array[0][0].shape
    width = int(width * scale)
    height = int(height * scale)
    collage = []
    for row in img_array:
        resized_row = [
            cv2.resize(img, (width, height)) if img.shape[:2] != (height, width) else img
            for img in row
        ]
        collage.append(np.hstack(resized_row))
    return np.vstack(collage)

def compute_motion_global(frame1, frame2, output_folder=False):
    # Convert the frames in grey scale
    frame1_gray = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    frame2_gray = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

    # Detect points of interest (Shi-Tomasi Corner Detection)
    points = cv2.goodFeaturesToTrack(frame1_gray, maxCorners=200, qualityLevel=0.3, minDistance=7, blockSize=7)

    # Parameters for Lucas-Kanade Optical Flow
    lk_params = dict(winSize=(15, 15), maxLevel=2,
                criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

    # Calculate optical flow with Lucas-Kanade
    new_points, status, error = cv2.calcOpticalFlowPyrLK(frame1_gray, frame2_gray, points, None, **lk_params)

    # Select valid points
    good_new = new_points[status == 1]
    good_old = points[status == 1]

    # Calculate the homography between the first and second frames
    if len(good_old) >= 4:  # It need at least 4 points to compute homography
        H, _ = cv2.findHomography(good_old, good_new, cv2.RANSAC, 5.0)
        
    if output_folder: 
        # Draw points of interest on the frames for visualization
        frame1_points = frame1.copy()
        for point in points:
            x, y = point.ravel()
            cv2.circle(frame1_points, (int(x), int(y)), 5, (255, 0, 0), -1)
        frame2_points = frame2.copy()
        for point in new_points:
            x, y = point.ravel()
            cv2.circle(frame2_points, (int(x), int(y)), 5, (0, 255, 0), -1)

        # Create images to visualize global motion
        global_motion = np.ones_like(frame1) * 255          # White background
        frame1_good_point = np.ones_like(frame1) * 255      # White background
        frame2_good_point = np.ones_like(frame2) * 255      # White background

        for (new, old) in zip(good_new, good_old):          # Draw dots and arrows representing global motion
            x_new, y_new = new.ravel()
            x_old, y_old = old.ravel()
            cv2.circle(frame1_good_point, (int(x_old), int(y_old)), 5, (255, 0, 0), -1) 
            cv2.circle(global_motion, (int(x_old), int(y_old)), 5, (255, 0, 0), -1)  
            cv2.circle(frame2_good_point, (int(x_new), int(y_new)), 5, (0, 255, 0), -1)  
            cv2.circle(global_motion, (int(x_new), int(y_new)), 5, (0, 255, 0), -1)  
            cv2.arrowedLine(global_motion, (int(x_old), int(y_old)), (int(x_new), int(y_new)), (0, 0, 255), 2)

        # Combina le immagini in collages
        frames_points_collage = stack_images(0.25, [
            [frame1, frame1_points, frame1_good_point],
            [frame2, frame2_points, frame2_good_point],
        ])
        global_motion_collage = stack_images(0.5,[
            [frame1_good_point, frame2_good_point],
            [global_motion,  np.ones_like(global_motion) * 255 ]
        ])

        # Salva il collage come immagine in un file
        frames_points_collage_path = os.path.join(output_folder, "frames-points.png")
        global_motion_collage_path = os.path.join(output_folder, "global-motion.png")
        cv2.imwrite(frames_points_collage_path, frames_points_collage)
        cv2.imwrite(global_motion_collage_path, global_motion_collage)

    return H

def main2():
    # Carica il video
    cap = cv2.VideoCapture('Demo/Video/Africa.mp4')
    output_folder = 'Demo/MotionEstimation/Africa'
    os.makedirs(output_folder, exist_ok=True)

    # Leggi il primo frame
    ret, old_frame = cap.read()
    if not ret:
        print("Errore nel caricamento del video")
        exit()

    # Leggi il secondo frame
    ret, frame = cap.read()
    if not ret:
        print("Errore nel caricamento del secondo frame")
        cap.release()
        exit()

    H = compute_motion_global(old_frame, frame, output_folder)

    cap.release()
    cv2.destroyAllWindows()

# SPARSE OPTICAL FLOW PART
# Performs object detection of the target class on an image, Returns the boxes founded.
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

def mask_motion_estimation(frame1, frame2, mask=None, output_folder=None):

    if mask is not None: 
        if len(mask.shape) == 2 and len(frame1.shape) == 3:  # we have to add a dimension becouse the image has 3 dimension (width, hight, and color dimesnion)
            mask = np.expand_dims(mask, axis=-1)             # the mask has only two dimension width and height, so for compability we have to add a third dimension

        if mask.shape[:2] != frame1.shape[:2]:
            print("\tResizing mask to match image dimensions.")
            mask = cv2.resize(mask, (frame1.shape[1], frame1.shape[0] ), interpolation=cv2.INTER_NEAREST)

        if mask.shape[:2] != frame1.shape[:2]:              # Validate dimensions
            raise ValueError("\tError: The mask dimensions still do not match the image dimensions.", mask.shape, image.shape)

    #print("FRAME 1 DIM: ", frame1.shape)
    #print("MASK DIM: ", mask.shape)

    frame1_gray = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    frame2_gray = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
    points = cv2.goodFeaturesToTrack(frame1_gray, maxCorners=200, qualityLevel=0.3, minDistance=7, mask=mask, blockSize=7)

    # Parameters for Lucas-Kanade Optical Flow
    lk_params = dict(winSize=(15, 15), maxLevel=2,
                criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

    # Calculate optical flow with Lucas-Kanade
    new_points, status, error = cv2.calcOpticalFlowPyrLK(frame1_gray, frame2_gray, points, None, **lk_params)
    
    good_new = new_points[status == 1]
    good_old = points[status == 1]

    if output_folder is not None:
        # Apply the mask to extract the region of interest
        roi = cv2.bitwise_and(frame1, frame1, mask=mask)
        roi_path = os.path.join(output_folder, "ROI.jpg")   
        cv2.imwrite(roi_path, roi)

        # Draw points of interest on the frames for visualization
        frame1_points = frame1.copy()
        for point in points:
            x, y = point.ravel()
            cv2.circle(frame1_points, (int(x), int(y)), 5, (255, 0, 0), -1)
        frame1_points_path = os.path.join(output_folder, "frame1-points.jpg")
        cv2.imwrite(frame1_points_path, frame1_points)
    
    return good_new, good_old
def main1():
    # Carica il video
    cap = cv2.VideoCapture('Demo/Video/Car2.mp4')
    output_folder = 'Demo/MotionEstimation/Car2'
    os.makedirs(output_folder, exist_ok=True)

    # Leggi il primo frame
    ret, frame1 = cap.read()
    if not ret:
        print("Errore nel caricamento del video")
        exit()

    # Leggi il secondo frame
    ret, frame2 = cap.read()
    if not ret:
        print("Errore nel caricamento del secondo frame")
        cap.release()
        exit()
    
    # Estract mask and box from the first frame 
    masks,boxes = process_image(frame1, 'car', output_folder)
    mask = masks[0].astype(np.uint8)
    box = boxes[0]

    good_new, good_old = mask_motion_estimation(frame1, frame2, mask, output_folder)

    motion_mask = np.zeros_like(frame1)
    frame1_motion = frame1.copy()
    for (new, old) in zip(good_new, good_old):          # Draw dots and arrows representing global motion
            x_new, y_new = new.ravel()
            x_old, y_old = old.ravel()
            motion_mask = cv2.line(motion_mask, (int(x_new), int(y_new)), (int(x_old), int(y_old)), (255, 0, 0), 2)
            frame1_motion = cv2.circle(frame1_motion, (int(x_new), int(y_new)), 3, (255, 0, 0), -1)
            
    frame1_motion = cv2.add(frame1_motion, motion_mask)
    frame1_motion_path = os.path.join(output_folder, "frame1-motion.jpg")
    cv2.imwrite(frame1_motion_path, frame1_motion)

# VIDEO SPARE OPTICAL FLOW
def mainVideo():
    # Parameters for Shi-Tomasi corner detection & Parameters for Lucas-Kanade optical flow
    feature_params = dict(maxCorners = 300, qualityLevel = 0.2, minDistance = 2, blockSize = 7)
    lk_params = dict(winSize = (15,15), maxLevel = 2, criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
    
    cap = cv2.VideoCapture('Demo/Video/Car2.mp4')
    color = (0, 255, 0)                             # Variable for color to draw optical flow track
    
    ret, first_frame = cap.read()
    prev_gray = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)

    # Finds the mask of the object we want to tracj using YOLO
    masks,boxes = process_image(first_frame, 'car')
    mask = masks[0].astype(np.uint8)

    if len(mask.shape) == 2 and len(first_frame.shape) == 3:    # we have to add a dimension becouse the image has 3 dimension (width, hight, and color dimesnion)
        mask = np.expand_dims(mask, axis=-1)                    # the mask has only two dimension width and height, so for compability we have to add a third dimension

    if mask.shape[:2] != first_frame.shape[:2]:
        print("\tResizing mask to match image dimensions.")
        mask = cv2.resize(mask, (first_frame.shape[1], first_frame.shape[0] ), interpolation=cv2.INTER_NEAREST)

    if mask.shape[:2] != first_frame.shape[:2]:              # Validate dimensions
        raise ValueError("\tError: The mask dimensions still do not match the image dimensions.", mask.shape, image.shape)
    
    # Shi-Tomasi corner detection inside the mask
    prev = cv2.goodFeaturesToTrack(prev_gray, mask = mask, **feature_params)
   
    mask_motion = np.zeros_like(first_frame)     # Creates an image filled with zero intensities with the same dimensions as the frame - for later drawing purposes
    
    while(cap.isOpened()):
        
        ret, frame = cap.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Calculates sparse optical flow by Lucas-Kanade method
        next, status, error = cv2.calcOpticalFlowPyrLK(prev_gray, gray, prev, None, **lk_params)
        
        good_old = prev[status == 1]         # Selects good feature points for previous position
        good_new = next[status == 1]         # Selects good feature points for next position

        # Draws the optical flow tracks
        for i, (new, old) in enumerate(zip(good_new, good_old)):
            a, b = new.ravel()      # Returns a contiguous flattened array as (x, y) coordinates for new point
            c, d = old.ravel()      # Returns a contiguous flattened array as (x, y) coordinates for old point
    
            mask_motion = cv2.line(mask_motion, (int(a), int(b)), (int(c), int(d)), color, 2)   # Draws line between new and old position with green color and 2 thickness
            frame = cv2.circle(frame, (int(a), int(b)), 3, color, -1)                           # Draws filled circle (thickness of -1) at new position with green color and radius of 3
        
        output = cv2.add(frame, mask_motion)    # Overlays the optical flow tracks on the original frame
       
        prev_gray = gray.copy()                 # Updates previous frame
        prev = good_new.reshape(-1, 1, 2)       # Updates previous good feature points
        
        cv2.imshow("sparse optical flow", output)
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break
   
    cap.release()
    cv2.destroyAllWindows()

    

if __name__ == "__main__":
    mainVideo()