import cv2
import numpy as np
import os

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
        A: The final affine matrix describe the object motion.
    Workflow: 
        Find (Shi-Tomasi Corner Detection) in the first frame the point of interest inside the mask, 
        so the poi of the object to track.
        Using Lucas-Kanade Optical Flow found the corrisponding points in the second frame. 
        From the couple of poi derive the object motion with a affine matrix
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
    if len(previus_frame) == 2 or previus_frame.shape[2] == 1:          # Check if the frame is already grayscale (single channel)
        previus_frame_gray = previus_frame
    else: previus_frame_gray = cv2.cvtColor(previus_frame, cv2.COLOR_BGR2GRAY)
    
    if len(next_frame) == 2 or next_frame.shape[2] == 1:                # Check if the frame is already grayscale (single channel)
        next_frame_gray = next_frame
    else: next_frame_gray = cv2.cvtColor(next_frame, cv2.COLOR_BGR2GRAY)
   
    previus_poi = cv2.goodFeaturesToTrack(previus_frame_gray, maxCorners=200, qualityLevel=0.3, minDistance=7, mask=mask, blockSize=7)

    # Parameters for Lucas-Kanade Optical Flow
    lk_params = dict(winSize=(15, 15), maxLevel=2, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

    # Calculate optical flow with Lucas-Kanade
    next_poi, status, error = cv2.calcOpticalFlowPyrLK(previus_frame_gray, next_frame_gray, previus_poi, None, **lk_params)
    
    good_next_poi = next_poi[status == 1]
    good_previus_poi = previus_poi[status == 1]

    if len(good_previus_poi) < 1:
        raise ValueError("No valid points detected for motion estimation.")
    # If there are not enough point to compute affine compute the transition 
    elif len(good_previus_poi) < 2:
         
        transition = np.mean(good_next_poi - good_previus_poi, axis=0)
        tx, ty = transition
        
        A = np.array([[1, 0, tx],         # Create a pure translation affine matrix (2x3)
                      [0, 1, ty]])
        #print(f"Warning: insufficient point for the affine, compute transition")
    else :                               
        A, _ = cv2.estimateAffinePartial2D(good_previus_poi, good_next_poi, method=cv2.RANSAC)
        

    if A is None:
        raise ValueError("Homography estimation failed.")

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
    
    return good_previus_poi, good_next_poi, A

def motion_estimation(previus_frame, next_frame, previus_poi):
    """
    Parameters:
        previus_frame: first frame
        next_frame: second frame
        previus_poi: the point of interest of the first frame
    Returns:
        good_previus_poi: the good point of interest in the first frame
        good_next_poi: the good point of interest in the second frame
        A: The final affine matrix describe the object motion.
    Workflow: 
        It is the streamlined version of motion estimation using the Lucas-Kanade optical flow method. 
        It computes optical flow between two frames and using the previus poi to estimate the 
        local motion with a affine matrix
    """

    # Parameters for Lucas-Kanade Optical Flow
    lk_params = dict(winSize=(15, 15), maxLevel=2, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
    
    if len(previus_frame.shape) == 3:  # L'immagine è a colori (ha 3 canali)
        previus_frame = cv2.cvtColor(previus_frame, cv2.COLOR_BGR2GRAY)

    if len(next_frame.shape) == 3:  # L'immagine è a colori (ha 3 canali)
        next_frame = cv2.cvtColor(next_frame, cv2.COLOR_BGR2GRAY)

    # Calculates sparse optical flow by Lucas-Kanade method
    next_poi, status, error = cv2.calcOpticalFlowPyrLK(previus_frame, next_frame, previus_poi, None, **lk_params)

    good_previus_poi = previus_poi[status == 1]     # Selects good poi of the previous frame
    good_next_poi = next_poi[status == 1]           # Selects good poi of the sucessor frame

    if len(good_previus_poi) < 1:
        raise ValueError("No valid points detected for motion estimation.")
    # If there are not enough point to compute affine compute the transition 
    elif len(good_previus_poi) < 2:
         
        transition = np.mean(good_next_poi - good_previus_poi, axis=0)
        tx, ty = transition
        
        A = np.array([[1, 0, tx],         # Create a pure translation affine matrix (2x3)
                      [0, 1, ty]])
        #print(f"Warning: insufficient point for the affine, compute transition")
    else :                               
        A, _ = cv2.estimateAffinePartial2D(good_previus_poi, good_next_poi, method=cv2.RANSAC)
        

    if A is None:
        raise ValueError("motion estimation failed.")

    return good_previus_poi, good_next_poi, A


