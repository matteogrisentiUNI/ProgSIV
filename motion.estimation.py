import cv2
import numpy as np
import os
from ObjectTracker import ObjectTracker, draw_mask

# GLOBAL MOTION PART
# Combina più immagini in un'unica immagine come un collage.
def stack_images(scale, img_array, rows, cols):
    
    min_cells = rows * cols
    
    total_images = len(img_array)
    if total_images > min_cells:
        rows = -(-total_images // cols)  
    
    img_h, img_w, _ = img_array[0].shape
    blank_image = np.ones((img_h, img_w, 3), dtype=np.uint8) * 255  
    
    total_cells = rows * cols
    if total_images < total_cells:
        img_array += [blank_image] * (total_cells - total_images)
    
    img_h = int(img_h * scale)
    img_w = int(img_w * scale)
    resized_images = [cv2.resize(img, (img_w, img_h)) for img in img_array]
    
    grid = []
    for r in range(rows):
        row_images = resized_images[r * cols:(r + 1) * cols]
        grid.append(np.hstack(row_images))
    
    return np.vstack(grid)

def indirect_compute_motion_global(frame1, frame2, output_folder=None, maxPoints=200, maxIteration=5):
    """
    Parameters:
        frame1: first frame
        frame2: second frame
        output_folder: folder for debug images
        maxPoints: number of point on which is computed the global motion
        maxIteration: number of itration to remove the outliers points
    Returns:
        numpy.ndarray: The final homography matrix describe the global motion.
    """

    print('INDIRECT GLOBAL MOTION COMPUTATION')

    # Convert the frames in grey scale
    frame1_gray = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    frame2_gray = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

    # Parameters for Lucas-Kanade Optical Flow
    lk_params = dict(winSize=(15, 15), maxLevel=2,
                criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
    
   
    discarded_mask = np.ones((frame1.shape[0], frame1.shape[1]), dtype=np.uint8) * 255  # Mask to track discarded points, initially, no points are discarded
    missing_points = maxPoints                                        # Number of missing poitn to reach 200 good point on wich compute Global Motion
    valid_old_points = np.empty((0, 2), dtype=np.float32)       # Array to accumulate valid points across iterations
    valid_new_points = np.empty((0, 2), dtype=np.float32)
    
    if output_folder is not None: 
        frame1_valid_points = []

    # Iteration loop to refine the points and keep the total number constant
    for iteration in range(maxIteration): 
        if missing_points == 0:
            print(f" - Iteration {iteration}: No missing points, skip this iteration.")
            continue

        # Detect points of interest (Shi-Tomasi Corner Detection)
        points = cv2.goodFeaturesToTrack(frame1_gray, maxCorners=missing_points, mask=discarded_mask, qualityLevel=0.1, minDistance=5, blockSize=5)
        if points is None: 
            print(f" - Iteration {iteration}: No points detected, skip this iteration.")
            continue
        
        # Calculate optical flow with Lucas-Kanade
        new_points, status, error = cv2.calcOpticalFlowPyrLK(frame1_gray, frame2_gray, points, None, **lk_params)

        # Select valid points
        good_new = new_points[status == 1]
        good_old = points[status == 1]

        # Remove outliers good points
        displacements = good_new - good_old                                 # Calculate displacement (motion vector)
        moduli = np.linalg.norm(displacements, axis=1)                      # Calculate the magnitude (modulo)
        directions = np.arctan2(displacements[:, 1], displacements[:, 0])   # Calculate the direction in radians

        median_magnitude = np.median(moduli)                                 # Calculate medians
        median_direction = np.median(directions)

        # Define thresholds for outliers (based on the median and some factor)
        magnitude_threshold = median_magnitude * 4  # Points with magnitude greater than 2 times the median are outliers
        direction_threshold = np.pi / 2             # Points whose direction deviates by more than 45 degrees are outliers

        # Filter out points that are outliers in terms of magnitude or direction
        filtered_old_points = []
        filtered_new_points = []

        for i in range(len(moduli)):
            if abs(moduli[i] - median_magnitude) <= magnitude_threshold and abs(directions[i] - median_direction) <= direction_threshold:
                filtered_old_points.append(good_old[i])
                filtered_new_points.append(good_new[i])
            else:   # Mark the discarded points in the mask, in this way the next iteration will not pick the same points
                x, y = good_old[i].astype(int)      # get coordinates of the discarted poitn
                discarded_mask[y, x] = 0            # update the mask

        filtered_old_points = np.array(filtered_old_points)     # Convert lists to numpy arrays
        filtered_new_points = np.array(filtered_new_points)

        if output_folder is not None: 
            frame1_valid_points_t = frame1.copy()

            for point in valid_old_points:
                x, y = point.ravel()
                cv2.circle(frame1_valid_points_t, (int(x), int(y)), 5, (255, 0, 0), -1) #blue
            
            for point in filtered_old_points:
                x, y = point.ravel()
                cv2.circle(frame1_valid_points_t, (int(x), int(y)), 5, (0, 255, 0), -1) #green

            discarded_indices = np.argwhere(discarded_mask == 0) 
            for idx in discarded_indices:
                y, x = idx 
                cv2.circle(frame1_valid_points_t, (int(x), int(y)), 5, (0, 0, 255), -1) #reed
            frame1_valid_points.append(frame1_valid_points_t)


        # Merge the valid points from the previous iteration with the filtered points of the current iteration
        valid_old_points = np.vstack((valid_old_points, filtered_old_points)) if valid_old_points.size else filtered_old_points
        valid_new_points = np.vstack((valid_new_points, filtered_new_points)) if valid_new_points.size else filtered_new_points

        missing_points = maxPoints - valid_old_points.shape[0]

        print(f" - Iteration {iteration}, start points: {points.shape[0]}, good points: {good_old.shape[0]}, filtered points: {filtered_old_points.shape[0]}, missing points: {missing_points}, valid points:{valid_old_points.shape[0]}")


    # Calculate the homography between the first and second frames
    if len(valid_old_points) >= 4:  # It need at least 4 points to compute homography
        H, _ = cv2.findHomography(valid_old_points, valid_new_points, cv2.RANSAC, 5.0)
    else:
        raise ValueError("\tError: Unsufficient good points to compute Global Motion")

    if output_folder is not None: 
        os.makedirs(output_folder, exist_ok=True)

        # Draw points of interest on the frames for visualization
        frame1_points = frame1.copy()
        for point in valid_old_points:
            x, y = point.ravel()
            cv2.circle(frame1_points, (int(x), int(y)), 5, (255, 0, 0), -1)
        frame2_points = frame2.copy()
        for point in valid_new_points:
            x, y = point.ravel()
            cv2.circle(frame2_points, (int(x), int(y)), 5, (0, 255, 0), -1)

        # Create images to visualize global motion
        global_motion = np.ones_like(frame1) * 255          # White background
        white_frame1_point = np.ones_like(frame1) * 255      # White background
        white_frame2_point = np.ones_like(frame2) * 255      # White background

        for (new, old) in zip(valid_new_points, valid_old_points):          # Draw dots and arrows representing global motion
            x_new, y_new = new.ravel()
            x_old, y_old = old.ravel()
            cv2.circle(white_frame1_point, (int(x_old), int(y_old)), 5, (255, 0, 0), -1) 
            cv2.circle(global_motion, (int(x_old), int(y_old)), 5, (255, 0, 0), -1)  
            cv2.circle(white_frame2_point, (int(x_new), int(y_new)), 5, (0, 255, 0), -1)  
            cv2.circle(global_motion, (int(x_new), int(y_new)), 5, (0, 255, 0), -1)  
            cv2.arrowedLine(global_motion, (int(x_old), int(y_old)), (int(x_new), int(y_new)), (0, 0, 255), 2)

        aligned_frame2 = cv2.warpPerspective(frame2, H, (frame1.shape[1], frame1.shape[0]))


        # Combina le immagini in collages
        frames_points_collage = stack_images(0.25, 
            [frame1, frame1_points, white_frame1_point, frame2, frame2_points, white_frame2_point],
            2,3 )
        frames1_valid_points_collage = stack_images(0.25, frame1_valid_points, 1, 3)
        global_motion_collage = stack_images(0.5,
            [white_frame1_point, white_frame2_point, global_motion,], 2, 2
        )

        # Salva il collage come immagine in un file
        aligned_frame2_path = os.path.join(output_folder, "aligned_frame2.png")
        frames_points_collage_path = os.path.join(output_folder, "frames-points.png")
        frames1_valid_points_collage_path = os.path.join(output_folder, "iteration_frame1-points.png") 
        global_motion_collage_path = os.path.join(output_folder, "global-motion.png")
        cv2.imwrite(aligned_frame2_path, aligned_frame2)
        cv2.imwrite(frames_points_collage_path, frames_points_collage)
        cv2.imwrite(frames1_valid_points_collage_path, frames1_valid_points_collage)
        cv2.imwrite(global_motion_collage_path, global_motion_collage)

    return H
def test_indirect_compute_global_motion():
    # Carica il video
    cap = cv2.VideoCapture('Demo/Video/Car2.mp4')
    output_folder = 'Demo/MotionEstimation/Car2/IndirectGM'
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

    H = indirect_compute_motion_global(old_frame, frame, output_folder=output_folder, maxPoints=300 )
    print("Final Homography Matrix:")
    print(H)

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
def test_mask_motion_estimation():
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



# VIDEO SPARSE OPTICAL FLOW
def test_sparse_motion_estimantion_video():
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
    test_indirect_compute_global_motion()