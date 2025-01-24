import cv2
import numpy as np
import os


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
