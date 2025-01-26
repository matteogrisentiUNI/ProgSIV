import cv2
import numpy as np
import os
from .ObjectTracker import ObjectTracker
from .mask_drawer import draw_mask

# Combien more images in an unique collage
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
def ensure_grayscale(frame):
    if len(frame.shape) == 2 or frame.shape[2] == 1:         # Check if the frame is already grayscale (single channel)
        return frame
    return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

# GLOBAL MOTION PART: Use Sparse Optical Flow on a set of point if interest ( poi ), and from that compute indirectly the global motion
def indirect_compute_motion_global(frame1, frame2, output_folder=None, maxPoints=200, maxIteration=5):
    """
    Parameters:
        frame1: first frame
        frame2: second frame
        output_folder: folder for debug images
        maxPoints: number of point on which is computed the global motion
        maxIteration: number of itration to remove the outliers points
    Returns:
        Hg: The final homography matrix describe the global motion.
    Workflow: 
        The idea is to compute the optical flow on a set of corresponding points of interest ( poi ). 
        Since in the video can be different element that can move indipendently we want to discart the poi
        that belong to that elements. The idea is an iterativly process where at each iteration:
            - find poi in order to reach maxPoints valid poi (Shi-Tomasi Corner Detection & Lucas-Kanade)
            - compute the median of the valid poi set
            - discard from the valid poi the poi that hase direction and magnitude too different from the median values
        After we found maxPoints valid poi or we reach maxIteration we compute from the valid poi set the 
        Homography matrix that rappresent the global motion.
    """

    print('INDIRECT GLOBAL MOTION COMPUTATION')

    # Convert the frames in grey scale
    frame1_gray = ensure_grayscale(frame1)
    frame2_gray = ensure_grayscale(frame2)

    # Parameters for Lucas-Kanade Optical Flow
    lk_params = dict(winSize=(15, 15), maxLevel=2, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
    
    discarded_mask = np.ones((frame1.shape[0], frame1.shape[1]), dtype=np.uint8) * 255  # Mask to track discarded points, initially, no points are discarded
    missing_poi = maxPoints                                                             # Number of missing poitn to reach 200 good point on wich compute Global Motion                
    valid_frame1_poi = np.empty((0, 2), dtype=np.float32)           # Array to accumulate valid points across iterations
    valid_frame2_poi = np.empty((0, 2), dtype=np.float32)             
    
    if output_folder is not None: 
        output_folder = output_folder + "/GlobalMotion"
        os.makedirs(output_folder, exist_ok=True)
        img_valid_frame1_poi = [ ]
        

    # Iteration loop to refine the points and keep the total number constant
    for iteration in range(maxIteration): 
        missing_poi = maxPoints - valid_frame1_poi.shape[0]
        if missing_poi < 10:
            print(f" - Iteration {iteration}: No missing points, skip this iteration.")
            continue

        # Detect poi (Shi-Tomasi Corner Detection) of the i-th iteration
        frame1_poi_i = cv2.goodFeaturesToTrack(frame1_gray, maxCorners=missing_poi, mask=discarded_mask, qualityLevel=0.1, minDistance=5, blockSize=5)
        if frame1_poi_i is None: 
            print(f" - Iteration {iteration}: No points detected, skip this iteration.")
            continue
        
        # Calculate optical flow with Lucas-Kanade
        frame2_poi_i, status, error = cv2.calcOpticalFlowPyrLK(frame1_gray, frame2_gray, frame1_poi_i, None, **lk_params)

        # Select poi for wich is founded a corrisponding poi in the other frame
        good_frame1_poi_i = frame1_poi_i[status == 1]
        good_frame2_poi_i = frame2_poi_i[status == 1]

        # Merge the valid points from the previous iteration with the points of the current iteration
        valid_frame1_poi = np.vstack((valid_frame1_poi, good_frame1_poi_i)) if valid_frame1_poi.size else good_frame1_poi_i
        valid_frame2_poi = np.vstack((valid_frame2_poi, good_frame2_poi_i)) if valid_frame2_poi.size else good_frame2_poi_i

        # Remove outliers good points
        displacements = valid_frame2_poi - valid_frame1_poi                 # Calculate displacement (motion vector)
        moduli = np.linalg.norm(displacements, axis=1)                      # Calculate the magnitude (modulo)
        directions = np.arctan2(displacements[:, 1], displacements[:, 0])   # Calculate the direction in radians

        median_magnitude = np.median(moduli)                                 # Calculate medians
        median_direction = np.median(directions)

        # Define thresholds for outliers (based on the median and some factor)
        magnitude_threshold = median_magnitude * 2  # Points with magnitude greater than 2 times the median are outliers
        direction_threshold = np.pi / 2             # Points whose direction deviates by more than 45 degrees are outliers

        # Filter out points that are outliers in terms of magnitude or direction
        filtered_frame1_poi = []
        filtered_frame2_poi = []

        for i in range(len(moduli)):
            if abs(moduli[i] - median_magnitude) <= magnitude_threshold and abs(directions[i] - median_direction) <= direction_threshold:
                filtered_frame1_poi.append(valid_frame1_poi[i])
                filtered_frame2_poi.append(valid_frame2_poi[i])
            else:   # Mark the discarded points in the mask, in this way the next iteration will not pick the same points
                x, y = valid_frame1_poi[i].astype(int)      # get coordinates of the discarted poitn
                discarded_mask[y, x] = 0                      # update the mask

        valid_frame1_poi = np.array(filtered_frame1_poi)     # Convert lists to numpy arrays
        valid_frame2_poi = np.array(filtered_frame2_poi)

        if output_folder is not None: 
            if iteration == 0: img_valid_frame1_points_t = frame1.copy()
            else: img_valid_frame1_points_t = img_valid_frame1_poi[iteration].copy()
            img_valid_frame1_points_t_new_points = img_valid_frame1_points_t.copy()

            for point in good_frame1_poi_i:     #The new poi founded in the iteration
                x, y = point.ravel()
                cv2.circle(img_valid_frame1_points_t_new_points, (int(x), int(y)), 5, (255, 0, 0), -1)     #blue
            

            for point in valid_frame1_poi:
                x, y = point.ravel()
                cv2.circle(img_valid_frame1_points_t, (int(x), int(y)), 5, (0, 255, 0), -1)     #green

            discarded_indices = np.argwhere(discarded_mask == 0) 
            for idx in discarded_indices:
                y, x = idx 
                cv2.circle(img_valid_frame1_points_t, (int(x), int(y)), 5, (0, 0, 255), -1) #reed
            
            img_valid_frame1_poi.append(img_valid_frame1_points_t_new_points )
            img_valid_frame1_poi.append(img_valid_frame1_points_t )

        print(f" - Iteration {iteration},  missing points: {missing_poi}, founded points: {frame1_poi_i.shape[0]}, good points: {good_frame1_poi_i.shape[0]}, valid points: {valid_frame1_poi.shape[0]}")


    # Calculate the homography between the first and second frames
    if len(valid_frame1_poi) >= 4:  # It need at least 4 points to compute homography
        H, _ = cv2.findHomography(valid_frame1_poi, valid_frame2_poi, cv2.RANSAC, 5.0)
    else:
        raise ValueError("\tError: Unsufficient good points to compute Global Motion")

    if output_folder is not None: 
        os.makedirs(output_folder, exist_ok=True)

        # Draw points of interest on the frames for visualization
        img_frame1_poi = frame1.copy()
        for point in valid_frame1_poi:
            x, y = point.ravel()
            cv2.circle(img_frame1_poi, (int(x), int(y)), 5, (255, 0, 0), -1)
        img_frame2_poi = frame2.copy()
        for point in valid_frame2_poi:
            x, y = point.ravel()
            cv2.circle(img_frame2_poi, (int(x), int(y)), 5, (0, 255, 0), -1)

        # Create images to visualize global motion
        img_global_motion = np.ones_like(frame1) * 255          # White background
        img_white_frame1_poi = np.ones_like(frame1) * 255      # White background
        img_white_frame2_poi = np.ones_like(frame2) * 255      # White background

        for (new, old) in zip(valid_frame2_poi, valid_frame1_poi):          # Draw dots and arrows representing global motion
            x_new, y_new = new.ravel()
            x_old, y_old = old.ravel()
            cv2.circle(img_white_frame1_poi, (int(x_old), int(y_old)), 5, (255, 0, 0), -1) 
            cv2.circle(img_global_motion, (int(x_old), int(y_old)), 5, (255, 0, 0), -1)  
            cv2.circle(img_white_frame2_poi, (int(x_new), int(y_new)), 5, (0, 255, 0), -1)  
            cv2.circle(img_global_motion, (int(x_new), int(y_new)), 5, (0, 255, 0), -1)  
            cv2.arrowedLine(img_global_motion, (int(x_old), int(y_old)), (int(x_new), int(y_new)), (0, 0, 255), 2)

        aligned_frame2 = cv2.warpPerspective(frame2, H, (frame1.shape[1], frame1.shape[0]))
        
        gray_frame2 = ensure_grayscale(frame2)
        gray_aligned_frame2 = ensure_grayscale(aligned_frame2)

        alignment_visualization = np.zeros((frame1.shape[0], frame1.shape[1], 3), dtype=np.uint8)

        _, binary_frame2 = cv2.threshold(gray_frame2, 50, 255, cv2.THRESH_BINARY)
        _, binary_aligned_frame2 = cv2.threshold(gray_aligned_frame2, 50, 255, cv2.THRESH_BINARY)

        white_overlap = (binary_frame2 == 255) & (binary_aligned_frame2 == 255)     # where both are white keep white
        alignment_visualization[white_overlap] = [255, 255, 255] 

        red_areas = (binary_frame2 == 0)                  # Where the original frame is black put it red
        alignment_visualization[red_areas] = [0, 0, 255] 

        green_areas = (binary_aligned_frame2 == 0)        # Where the alligment frame is black put it green
        alignment_visualization[green_areas] = [0, 255, 0]  

        border_color = [255, 0, 0]  # Blu
        border_thickness = 5
        alignment_visualization = cv2.copyMakeBorder(
            alignment_visualization, 
            border_thickness, border_thickness, border_thickness, border_thickness, 
            cv2.BORDER_CONSTANT, value=border_color
        )

        # Disegna una freccia che rappresenta la homography matrix
        t_x = H[0, 2]  # Traslazione lungo X
        t_y = H[1, 2]  # Traslazione lungo Y
        start_point = (frame1.shape[1] // 2, frame1.shape[0] // 2)  # Centro dell'immagine
        end_point = (int(start_point[0] + t_x * 10), int(start_point[1] + t_y * 10))  # Scala il movimento per visibilità
        arrow_color = (255, 0, 0)  # Colore giallo
        thickness = 3
        cv2.arrowedLine(alignment_visualization, start_point, end_point, arrow_color, thickness, tipLength=0.1)

        # Combina le immagini in collages
        frames_poi_collage = stack_images(0.25, 
            [frame1, img_frame1_poi, img_white_frame1_poi, frame2, img_frame2_poi, img_white_frame2_poi],
            2,3 )
        frames1_valid_poi_collage = stack_images(0.25, img_valid_frame1_poi, 1, 2)
        global_motion_collage = stack_images(0.5,
            [img_white_frame1_poi, img_white_frame2_poi, img_global_motion,], 2, 2
        )

        # Salva il collage come immagine in un file
        alignment_visualization_path = os.path.join(output_folder, "aligned_frame2.png")
        frames_poi_collage_path = os.path.join(output_folder, "frames-poi.png")
        frames1_valid_poi_collage_path = os.path.join(output_folder, "iteration_frame1-poi.png") 
        global_motion_collage_path = os.path.join(output_folder, "global-motion.png")
        cv2.imwrite(alignment_visualization_path, alignment_visualization)
        cv2.imwrite(frames_poi_collage_path, frames_poi_collage)
        cv2.imwrite(frames1_valid_poi_collage_path, frames1_valid_poi_collage)
        cv2.imwrite(global_motion_collage_path, global_motion_collage)

    return H

def mask_motion_estimation_fixedCamera(previus_frame, next_frame, mask=None, output_folder=None):
    """
    Parameters:
        previus_frame: first frame
        next_frame: second frame
        mask: a binary mask of the first frame containing the object to track
        output_folder: folder for debug images
    Returns:
        good_previus_poi: the good point of interest in the first frame
        good_next_poi: the good point of interest in the second frame
        motion: The final transition vector describe the local motion.
    Workflow: 
        Find (Shi-Tomasi Corner Detection) in the first frame the point of interest inside the mask, 
        so the poi of the object to track.
        Using Lucas-Kanade Optical Flow found the corrisponding points in the second frame. 
        From the couple of poi derive the transition vector to describe the local motion
    """

    if mask is not None: 
        if len(mask.shape) == 2 and len(previus_frame.shape) == 3:  # we have to add a dimension becouse the image has 3 dimension (width, hight, and color dimesnion)
            mask = np.expand_dims(mask, axis=-1)             # the mask has only two dimension width and height, so for compability we have to add a third dimension

        if mask.shape[:2] != previus_frame.shape[:2]:
            print("\tResizing mask to match image dimensions.")
            mask = cv2.resize(mask, (previus_frame.shape[1], previus_frame.shape[0] ), interpolation=cv2.INTER_NEAREST)

        if mask.shape[:2] != previus_frame.shape[:2]:              # Validate dimensions
            raise ValueError("\tError: The mask dimensions still do not match the image dimensions.", mask.shape, image.shape)

    # Shi-Tomasi Corner Detection
    previus_frame_gray = ensure_grayscale(previus_frame)
    next_frame_gray = ensure_grayscale(next_frame)
    previus_poi = cv2.goodFeaturesToTrack(previus_frame_gray, maxCorners=200, qualityLevel=0.3, minDistance=7, mask=mask, blockSize=7)

    # Parameters for Lucas-Kanade Optical Flow
    lk_params = dict(winSize=(15, 15), maxLevel=2, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

    # Calculate optical flow with Lucas-Kanade
    next_poi, status, error = cv2.calcOpticalFlowPyrLK(previus_frame_gray, next_frame_gray, previus_poi, None, **lk_params)
    
    good_next_poi = next_poi[status == 1]
    good_previus_poi = previus_poi[status == 1]
    motion = np.mean(good_next_poi - good_previus_poi, axis=0)

    if output_folder is not None:
        # Apply the mask to extract the region of interest
        roi = cv2.bitwise_and(previus_frame, previus_frame, mask=mask)
        roi_path = os.path.join(output_folder, "ROI.jpg")   
        cv2.imwrite(roi_path, roi)

        # Draw points of interest on the frames for visualization
        previus_frame_points = previus_frame.copy()
        for point in previus_poi:
            x, y = point.ravel()
            cv2.circle(previus_frame_points, (int(x), int(y)), 5, (255, 0, 0), -1)
        previus_frame_points_path = os.path.join(output_folder, "frame1-points.jpg")
        cv2.imwrite(previus_frame_points_path, previus_frame_points)
    
    return good_previus_poi, good_next_poi, motion
def motion_estimation_fixedCamera(previus_frame, next_frame, previus_poi):
    """
    Parameters:
        previus_frame: first frame
        next_frame: second frame
        previus_poi: the point of interest of the first frame
    Returns:
        good_previus_poi: the good point of interest in the first frame
        good_next_poi: the good point of interest in the second frame
        motion: The final transition vector describe the local motion.
    Workflow: 
        It is the streamlined version of motion estimation using the Lucas-Kanade optical flow method. 
        It computes optical flow between two frames and using the previus poi to estimate the 
        local motion with a transition vector
    """

    # Parameters for Lucas-Kanade Optical Flow
    lk_params = dict(winSize=(15, 15), maxLevel=2, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
     
    previus_frame = ensure_grayscale(previus_frame)
    next_frame = ensure_grayscale(next_frame)

    # Calculates sparse optical flow by Lucas-Kanade method
    next_poi, status, error = cv2.calcOpticalFlowPyrLK(previus_frame, next_frame, previus_poi, None, **lk_params)

    good_previus_poi = previus_poi[status == 1]     # Selects good poi of the previous frame
    good_next_poi = next_poi[status == 1]           # Selects good poi of the sucessor frame
    motion = np.mean(good_next_poi - good_previus_poi, axis=0)      # compute the avarage motion

    return good_previus_poi, good_next_poi, motion


def mask_motion_estimation(previus_frame, next_frame, mask=None, output_folder=None):
    """
    Parameters:
        previus_frame: first frame
        next_frame: second frame
        mask: a binary mask of the first frame containing the object to track
        output_folder: folder for debug images
    Returns:
        good_previus_poi: the good point of interest in the first frame
        good_next_poi: the good point of interest in the second frame
        motion: The final transition vector describe the local motion.
    Workflow: 
        Compiute the global motion estimation; remove the motion from the second frame.
        With a fixed camera version of the second frame compute the local motion.
        Combine the global motion and local motion to derive the complete motion of the object between the 2 frame
    """
    # Compute the global motion of the video
    Hg = indirect_compute_motion_global(previus_frame, next_frame, output_folder=output_folder)

    if np.allclose(Hg, np.eye(3), atol=1e-4):   # if the Hg is close to identity matrix ( global motion = 0, camera stationary)
        Hg = np.eye(3)                          # round it to the identity matrix avoid to propagate error with inverse
        aligned_next_frame = next_frame.copy()
    else: 
        # Try to remove the camera motion from the second frame, in order to be able to aply sparce optical flow
        aligned_next_frame = cv2.warpPerspective(next_frame, Hg, (previus_frame.shape[1], previus_frame.shape[0]))
    
    # compute optical flow on the rearrenged second frame
    good_new, good_old, motion = mask_motion_estimation_fixedCamera(previus_frame, aligned_next_frame, mask=mask, output_folder=output_folder)

    Hg_inv = np.linalg.inv(Hg)
    good_new_h = cv2.perspectiveTransform(good_new.reshape(-1, 1, 2), Hg_inv).reshape(-1, 2)
    good_old_h = cv2.perspectiveTransform(good_old.reshape(-1, 1, 2), Hg_inv).reshape(-1, 2)
    
    # Create the affine matrix of transition  3x3
    tx, ty = motion
    T = np.array([[1, 0, tx],[0, 1, ty],[0, 0, 1]], dtype=np.float64)

    # Combina il moto globale e locale
    if Hg.shape == (3, 3) and T.shape == (3, 3):
        # Matrix multiplication using numpy
        Hgl = np.dot(Hg_inv, T)  # Or equivalently H @ Hl in Python (for newer versions of Python)
    else:
        raise ValueError("Both H and Hl must be 3x3 homography matrices.")
    
    #print("Hg_inv: ", Hg_inv)
    #print("Hl: ", Hl)
    #print("Hgl: ", Hgl)

    return good_new_h, good_old_h, Hgl
def motion_estimation(previus_frame, next_frame, previus_poi):
    """
    Parameters:
        previus_frame: first frame
        next_frame: second frame
        previus_poi: the point of interest of the first frame
    Returns:
        good_previus_poi: the good point of interest in the first frame
        good_next_poi: the good point of interest in the second frame
        motion: The final transition vector describe the local motion.
    Workflow: 
        It is the streamlined version of motion estimation 
    """
    # Compute the global motion of the video
    Hg = indirect_compute_motion_global(previus_frame, next_frame, output_folder=None)

    if np.allclose(Hg, np.eye(3), atol=1):   # if the Hg is close to identity matrix ( global motion = 0, camera stationary)
        Hg = np.eye(3)                          # round it to the identity matrix avoid to propagate error with inverse
        aligned_next_frame = next_frame.copy()
    else: 
        # Try to remove the camera motion from the second frame, in order to be able to aply sparce optical flow
        aligned_next_frame = cv2.warpPerspective(next_frame, Hg, (previus_frame.shape[1], previus_frame.shape[0]))
    
    # compute optical flow on the rearrenged second frame
    good_new, good_old, motion = motion_estimation_fixedCamera(previus_frame, aligned_next_frame, previus_poi)

    Hg_inv = np.linalg.inv(Hg)
    good_new_h = cv2.perspectiveTransform(good_new.reshape(-1, 1, 2), Hg_inv).reshape(-1, 2)
    good_old_h = cv2.perspectiveTransform(good_old.reshape(-1, 1, 2), Hg_inv).reshape(-1, 2)
    
    # Create the affine matrix of transition  3x3
    tx, ty = motion
    T = np.array([[1, 0, tx],[0, 1, ty],[0, 0, 1]], dtype=np.float64)

    # Combina il moto globale e locale
    if Hg.shape == (3, 3) and T.shape == (3, 3):
        # Matrix multiplication using numpy
        Hgl = np.dot(Hg_inv, T)  # Or equivalently H @ Hl in Python (for newer versions of Python)
    else:
        raise ValueError("Both H and Hl must be 3x3 homography matrices.")
    
    #print("Hg_inv: ", Hg_inv)
    #print("Hl: ", T)
    #print("Hgl: ", Hgl)

    
    return good_new_h, good_old_h, Hgl

