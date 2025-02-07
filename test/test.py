import cv2
import time
import psutil
import glob
import numpy as np
import os
from ObjectTracker import detection, mask_motion_estimation, motion_estimation, segmentation, feature_extraction, utils, mask_refinement

def test_discreto_video(video_path, object_detected, output_folder=None, saveVideo=False, debugPrint=False):
    cap = cv2.VideoCapture(video_path)
    os.makedirs(output_folder, exist_ok=True)
    performance_log_path = os.path.join(output_folder, 'performance.txt')
    os.makedirs(os.path.dirname(performance_log_path), exist_ok=True)

    if saveVideo:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        out = cv2.VideoWriter(os.path.join(output_folder, 'tracked_video.mp4'), fourcc, fps, (frame_width, frame_height))

    cpu_usage_list = []
    memory_usage_list = []
    frame_times = []
    start_time = time.time()
    frame_count = 0

    ret, previus_frame = cap.read()
    ret, next_frame = cap.read()
    frame_count = 2

    cpu_before = psutil.cpu_percent(interval=None)
    memory_before = psutil.virtual_memory().percent


# - DETECTION
    print('- DETECTION ')
    try:
        masks, boxes = detection(previus_frame, object_detected)
    except Exception as e:
        print(f"\tERROR: {e}")
        return
    
    # Print for debug
    if debugPrint: 
       debugPrintDetection(previus_frame, boxes, masks, frame_width, frame_height, output_folder=output_folder)

    if not masks:
        print(f'\tERROR:No {object_detected} detected in the first frame')
        return
    
    # Take only one object ( mask-box ), so the sistem is single object tracker
    mask = masks[0].astype(np.uint8)
    box = boxes[0].astype(np.float32)


# - MASK REFINEMENT 
    print('- MASK REFINEMENT ')
    try:
        refined_mask = mask_refinement.refine_mask(previus_frame, mask)
    except Exception as e:
        print(f"\tERROR: {e}")
        return
    
    # Print for debug
    if debugPrint: 
        debugPrintMaskRefinement(refined_mask, box, previus_frame, frame_width, frame_height, output_folder=output_folder)
        
    mask = refined_mask

# - MOTION
    print('- MOTION ESTIMATION')
    try:
        good_previus_poi, good_next_poi, A = mask_motion_estimation( previus_frame, next_frame, mask=mask )
    except Exception as e:
        print(f"\tERROR: {e}")
        return
    
    # translate the box
    box_r = box.reshape(-1, 1, 2)
    box_t = cv2.transform(box_r, A).reshape(4)
    
    # Print for debug
    if debugPrint: 
       debugPrintMotionEstimation(previus_frame, good_previus_poi, good_next_poi, A, box, box_t, frame_width, frame_height, mask, output_folder=output_folder)


# - CROP FRAME
    print('- CROP FRAME')# Resize the frame and the mask
    try:
        cropped_previus_frame, resized_mask = utils.resize(previus_frame, box, 150000, mask=mask)
        cropped_next_frame, _ = utils.resize(next_frame, box_t, 150000)
    except Exception as e:
        print(f"\tERROR: {e}")
        return
    
    # Print for debug
    if debugPrint:
        debugPrintFrameCrop(cropped_previus_frame, resized_mask, cropped_next_frame, output_folder=output_folder)


# - SEGMENTATION
    print('- SEGMENTATION')# Resize the frame and the mask
    
    # Apply Gaussian blur
    blurred_previus_frame = cv2.GaussianBlur(cropped_previus_frame, (3, 3), sigmaX=0)
    blurred_next_frame = cv2.GaussianBlur(cropped_next_frame, (3, 3), sigmaX=0)

    # Extract the histogram of the ROI
    predicted_histogram = feature_extraction.histogram_extraction(blurred_previus_frame, resized_mask)

    # Extract the region of interest
    next_mask = segmentation.segmentation(blurred_next_frame, predicted_histogram)
   
    # Print for debug
    if debugPrint:
        debugPrintSegmentation(blurred_next_frame, predicted_histogram, next_mask, output_folder=output_folder)



# Save next frame with bounding box an mask in the output video
    if saveVideo:
        output_frame = utils.draw_mask(next_frame, box_t, mask, object_detected, color_mask=(255, 0, 0))
        out.write(output_frame)
    
 
    previus_frame = cv2.cvtColor(next_frame, cv2.COLOR_BGR2GRAY)
    previus_poi = good_next_poi.reshape(-1, 1, 2)

    cpu_after = psutil.cpu_percent(interval=None)
    memory_after = psutil.virtual_memory().percent
    
    cpu_usage_list.append((cpu_before + cpu_after) / 2)
    memory_usage_list.append((memory_before + memory_after) / 2)
    
    cap.release()
    if saveVideo:
        out.release()
    cv2.destroyAllWindows()

    return 

    while cap.isOpened():
        print("Premi 'n' per il prossimo frame o 'q' per uscire.")
        key = cv2.waitKey(0) & 0xFF
        if key == ord('q'):
            break
        elif key != ord('n'):
            continue
        
        frame_start_time = time.time()
        cpu_before = psutil.cpu_percent(interval=None)
        memory_before = psutil.virtual_memory().percent

        ret, next_frame_color = cap.read()
        if not ret:
            print("Video Ended")
            break

        next_frame = cv2.cvtColor(next_frame_color, cv2.COLOR_BGR2GRAY)

        try:
            good_previus_poi, good_next_poi, A = motion_estimation(previus_frame, next_frame, previus_poi)
        except Exception as e:
            print(f"ERROR - motion estimation: {e}")
            return

        box = box.reshape(-1, 1, 2)
        box = cv2.transform(box, A).reshape(-1, 4)
        x1, y1, x2, y2 = box[0]

        frame = next_frame_color.copy()
        for new in good_next_poi:
            a, b = new.ravel()
            frame = cv2.circle(frame, (int(a), int(b)), 3, color_poi, -1)
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color_box, 2)

        if saveVideo:
            out.write(frame)
        if showVideo:
            resized_frame = cv2.resize(frame, (frame_width // 2, frame_height // 2), interpolation=cv2.INTER_AREA)
            cv2.imshow("Sparse Optical Flow", resized_frame)

        previus_frame = next_frame
        previus_poi = good_next_poi.reshape(-1, 1, 2)
        
        cpu_after = psutil.cpu_percent(interval=None)
        memory_after = psutil.virtual_memory().percent
        cpu_usage_list.append((cpu_before + cpu_after) / 2)
        memory_usage_list.append((memory_before + memory_after) / 2)
        frame_times.append(time.time() - frame_start_time)
        frame_count += 1

    total_processing_time = time.time() - start_time

    avg_cpu_usage = np.mean(cpu_usage_list)
    avg_memory_usage = np.mean(memory_usage_list)
    avg_time_per_frame = np.mean(frame_times)

    with open(performance_log_path, 'a') as f:
        f.write("\n--- New Run ---\n")
        f.write(f"Total frames processed: {frame_count}\n")
        f.write(f"Average CPU usage: {avg_cpu_usage:.2f}%\n")
        f.write(f"Average Memory usage: {avg_memory_usage:.2f}%\n")
        f.write(f"Average time per frame: {avg_time_per_frame:.4f} seconds\n")
        f.write(f"Total processing time: {total_processing_time:.2f} seconds\n")

    cap.release()
    if saveVideo:
        out.release()
    cv2.destroyAllWindows()


# DEBUG PRINT FUNCTIONS
def debugPrintDetection(previus_frame, boxes, masks, frame_width, frame_height, output_folder=None):

    print(output_folder)

    masked_previus_frame = utils.draw_mask(previus_frame, boxes, masks[0], object_detected)
    resized_masked_previus_frame = cv2.resize(masked_previus_frame, (frame_width // 2, frame_height // 2), interpolation=cv2.INTER_AREA)
    
    cv2.imshow("Masked Image", resized_masked_previus_frame)
    if output_folder is not None:        # Save the processed image
        processed_image_path = os.path.join(output_folder, "1_Detection_YOLO_Mask.jpg")
        cv2.imwrite(processed_image_path, resized_masked_previus_frame)
    
    # block the code in order to analyse the behavior frame to frame
    while True:
        key = cv2.waitKey(0) & 0xFF
        if key == ord('w'): break
    cv2.destroyAllWindows()

def debugPrintMaskRefinement(refined_mask, box, previus_frame, frame_width, frame_height, output_folder=None):
    
    masked_previus_frame = utils.draw_mask(previus_frame, box, refined_mask, object_detected, color_mask=(255, 0, 0))
    resized_masked_previus_frame = cv2.resize(masked_previus_frame, (frame_width // 2, frame_height // 2), interpolation=cv2.INTER_AREA)
    
    cv2.imshow("Refined Masked Image", resized_masked_previus_frame)
    if output_folder is not None:        # Save the processed image
        processed_image_path = os.path.join(output_folder, "2_MaskRefinment.jpg")
        cv2.imwrite(processed_image_path, resized_masked_previus_frame)
    
    # block the code in order to analyse the behavior frame to frame
    while True:
        key = cv2.waitKey(0) & 0xFF
        if key == ord('w'): break
    cv2.destroyAllWindows()

def debugPrintBoxResiz(box, box_resized, previus_frame, frame_width, frame_height, output_folder=None):
    # Draw the original box and the resized box
    previus_frame_br = previus_frame.copy()
    
    x1, y1, x2, y2 = map(int, box) 
    cv2.rectangle(previus_frame_br, (x1, y1), (x2, y2), (255, 0, 0), 1)

    x1_r, y1_r, x2_r, y2_r = map(int, box_resized) 
    cv2.rectangle(previus_frame_br, (x1_r, y1_r), (x2_r, y2_r), (255, 0, 0), 3)

    resized_previus_frame_br = cv2.resize(previus_frame_br, (frame_width // 2, frame_height // 2), interpolation=cv2.INTER_AREA)

    cv2.imshow("Resized Box", resized_previus_frame_br)

    if output_folder is not None:        
        previus_frame_br_path = os.path.join(output_folder, "3_ResizedBox.jpg")
        cv2.imwrite(previus_frame_br_path, resized_previus_frame_br)

    
    # block the code in order to analyse the behavior frame to frame
    while True:
        key = cv2.waitKey(0) & 0xFF
        if key == ord('w'): break
    cv2.destroyAllWindows()

def debugPrintMotionEstimation(previus_frame, previus_poi, next_poi, A, box, box_t, frame_width, frame_height, mask, output_folder=None):
    
    # Apply the mask to extract the region of interest
    # roi = cv2.bitwise_and(cropped_previus_frame, cropped_previus_frame, mask=mask)
    # roi_path = os.path.join(output_folder, "4_1_ROI.jpg")   
    # cv2.imwrite(roi_path, roi)

    # Draw points of interest on the current frame and the next frame 
    frame_points = previus_frame.copy()

    for point in previus_poi:
        x, y = point.ravel()
        cv2.circle(frame_points, (int(x), int(y)), 5, (255, 0, 0), -1) #blu

    for point in next_poi:
        x, y = point.ravel()
        cv2.circle(frame_points, (int(x), int(y)), 4, (0, 255, 0), -1)


    # Rappresent the computed motion A with an arrow
    center_x, center_y = previus_frame.shape[1] // 2, previus_frame.shape[0] // 2  # Center of the image
    
    start_point = np.array([center_x, center_y, 1])         # Start of the arrow in the center of the image
    end_point = A @ np.array([center_x + 50, center_y, 1])  # End of the arrow dipent on the direction an intensiti of the motion; applied a offset of 50

    start = (int(start_point[0]), int(start_point[1]))      # convert to int coordinates
    end = (int(end_point[0]), int(end_point[1]))

    cv2.arrowedLine(frame_points, start, end, (0, 0, 255), 3, tipLength=0.3)  # Draw to arrow in red
    
    # Draw the original box and the translated box
    x1, y1, x2, y2 = map(int, box) 
    cv2.rectangle(frame_points, (x1, y1), (x2, y2), (255, 0, 0), 3)

    x1_t, y1_t, x2_t, y2_t = map(int, box_t) 
    cv2.rectangle(frame_points, (x1_t, y1_t), (x2_t, y2_t), (0, 255, 0), 2, lineType=4)
    

    resized_frame_points = cv2.resize(frame_points, (frame_width // 2, frame_height // 2), interpolation=cv2.INTER_AREA)

    cv2.imshow("Motion Image", resized_frame_points)
    if output_folder is not None:        # Save the processed image
        frame_points_path = os.path.join(output_folder, "4_MotionEstimation.jpg")
        cv2.imwrite(frame_points_path, resized_frame_points)
    
    # block the code in order to analyse the behavior frame to frame
    while True:
        key = cv2.waitKey(0) & 0xFF
        if key == ord('w'): break
    cv2.destroyAllWindows()

def debugPrintFrameCrop(cropped_previus_frame, resized_mask, cropped_next_frame, output_folder=None):
    cropped_previus_frame_mask = utils.draw_mask(cropped_previus_frame, masks=resized_mask)
    cropped_frames = np.vstack((cropped_previus_frame_mask, cropped_next_frame))

    cv2.imshow("Cropped Frames", cropped_frames)
    if output_folder is not None:        
        cropped_frames_path = os.path.join(output_folder, "5_CroppedFrames.jpg")
        cv2.imwrite(cropped_frames_path, cropped_frames)
    
    # block the code in order to analyse the behavior frame to frame
    while True:
        key = cv2.waitKey(0) & 0xFF
        if key == ord('w'): break
    cv2.destroyAllWindows()

def debugPrintSegmentation(blurred, predicted_histogram, mask, output_folder=None):
     
    histogram_img = utils.draw_histogtams(predicted_histogram, blurred.shape[0], blurred.shape[1])
    combined_image = np.hstack((blurred, histogram_img))

    cv2.imshow("Blured Frame & Histogram", combined_image)
    if output_folder is not None:        
        combined_image_path = os.path.join(output_folder, "6_1_Segmentation_PreProcessing.jpg")
        cv2.imwrite(combined_image_path, combined_image)
    
    # block the code in order to analyse the behavior frame to frame
    while True:
        key = cv2.waitKey(0) & 0xFF
        if key == ord('w'): break
    cv2.destroyAllWindows()

if __name__ == "__main__":

    video_path = 'Demo/Video/Ship.mp4'
    output_folder = os.path.join('test/Global/Ship')
    object_detected = 'boat'

    test_discreto_video(video_path, object_detected, output_folder=output_folder,  saveVideo=True, debugPrint=True)


