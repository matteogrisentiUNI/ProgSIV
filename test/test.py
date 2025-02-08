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
    frame_count = 1

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


# Save the Output 
    if saveVideo:
        output_frame = utils.draw_mask(previus_frame, box, mask, object_detected)
        resized_output_frame = cv2.resize(output_frame, (frame_width, frame_height), interpolation=cv2.INTER_AREA)
        out.write(resized_output_frame)


    cpu_after = psutil.cpu_percent(interval=None)
    memory_after = psutil.virtual_memory().percent
    
    cpu_usage_list.append((cpu_before + cpu_after) / 2)
    memory_usage_list.append((memory_before + memory_after) / 2)

    # Activate all the debug print ( A LOT !! )
    debugPrint = False

    while cap.isOpened():
        #Discrete Flow of the Video
        print("FRAME N.", frame_count)
        #cv2.waitKey(0)
        cv2.destroyAllWindows()
        
        # Start counter for the performance
        frame_start_time = time.time()
        cpu_before = psutil.cpu_percent(interval=None)
        memory_before = psutil.virtual_memory().percent

        # Take the next frame to analize
        ret, next_frame = cap.read()
        if not ret:
            print("Video Ended")
            break
        
        # TRUCK OBJECT
        #if frame_count >= 180:
        #    next_mask, next_box = trucking(previus_frame, mask, box, next_frame, output_folder=output_folder, debugPrint=True)
        #else: 
        next_mask, next_box = trucking(previus_frame, mask, box, next_frame, output_folder=output_folder, debugPrint=True)
        
        # Save next frame with bounding box an mask in the output video
        if saveVideo:
            output_frame = utils.draw_mask(next_frame, next_box, next_mask, object_detected, color_mask=(255, 0, 0))
            resized_output_frame = cv2.resize(output_frame, (frame_width, frame_height), interpolation=cv2.INTER_AREA)
            
            if debugPrint:
                cv2.imshow('Final Frame', resized_output_frame)
                cv2.waitKey(0)
                cv2.destroyAllWindows()

            out.write(resized_output_frame)
    
        # Update the variable for the next cycle 
        previus_frame = next_frame
        mask = next_mask
        box = next_box

        # Perfoormance Misure of the cingle cycle
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



# TRUCKING FUNCTION
def trucking(previus_frame, previus_mask, previus_box, next_frame, output_folder=None, debugPrint=False):
    
# - MOTION
    print('- MOTION ESTIMATION')
    try:
        good_previus_poi, good_next_poi, A = mask_motion_estimation( previus_frame, next_frame, mask=previus_mask )
    except Exception as e:
        print(f"\tERROR: {e}")
        return
    
    # translate the box
    box_r = previus_box.reshape(-1, 1, 2)
    next_box = cv2.transform(box_r, A).reshape(4)

    # Print for debug
    if debugPrint: 
       debugPrintMotionEstimation(previus_frame, good_previus_poi, good_next_poi, A, previus_box, next_box, output_folder=output_folder)


# - CROP FRAME
    print('- CROP FRAME')# Resize the frame and the mask
    try:
        cropped_previus_frame, resized_mask, _ , _ = utils.resize(previus_frame, previus_box, 150000, A, mask=previus_mask)
        cropped_next_frame, _ , resized_box, scaling_factors = utils.resize(next_frame, next_box, 150000, A)
        
    except Exception as e:
        print(f"\tERROR: {e}")
        return
    
    # Print for debug
    if debugPrint:
        debugPrintFrameCrop(cropped_previus_frame, resized_mask, cropped_next_frame, resized_box, next_box, next_frame, output_folder=output_folder)


# - SEGMENTATION
    print('- SEGMENTATION')# Resize the frame and the mask
    
    # Apply Gaussian blur
    blurred_previus_frame = cv2.GaussianBlur(cropped_previus_frame, (3, 3), sigmaX=0)
    blurred_next_frame = cv2.GaussianBlur(cropped_next_frame, (3, 3), sigmaX=0)

    # Extract the histogram of the ROI
    predicted_histogram = feature_extraction.histogram_extraction(blurred_previus_frame, resized_mask)

    # Extract the region of interest
    next_mask = segmentation.segmentation(blurred_next_frame, predicted_histogram, output_folder=output_folder, debugPrint=debugPrint)

    # Print for debug
    if debugPrint:
        debugPrintSegmentation(blurred_next_frame, next_mask, output_folder=output_folder)

#   RESCALING: we have to rescale the mask to the original frame dimension
    # 1: Resize the mask to the origina pixel quantity
    original_width = int(next_mask.shape[1] / scaling_factors)
    original_height = int(next_mask.shape[0] / scaling_factors)
    next_mask = cv2.resize(next_mask, (original_width, original_height), interpolation=cv2.INTER_LINEAR)

    # 2: Rescale to the dimension of the resized_box
    x1, y1, x2, y2 = resized_box
    next_mask = cv2.resize(next_mask, ( int(y2-y1), int(x2-x1)), interpolation=cv2.INTER_NEAREST)

    # 3: Adding black padding in order the mask match the dimension of the original frame
    next_mask = utils.resize_mask_with_padding(next_mask, resized_box, next_frame.shape[0], next_frame.shape[1])
    
    # 4: shrink the box in order to be closer to the final mask
    next_box = utils.shrink_box_to_mask(resized_box, next_mask, threshold=5)


    return next_mask, next_box


# DEBUG PRINT FUNCTIONS
def debugPrintDetection(previus_frame, boxes, masks, frame_width, frame_height, output_folder=None):
    masked_previus_frame = utils.draw_mask(previus_frame, boxes, masks[0], object_detected)
    resized_masked_previus_frame = cv2.resize(masked_previus_frame, (frame_width // 2, frame_height // 2), interpolation=cv2.INTER_AREA)
    
    cv2.imshow("Masked Image", resized_masked_previus_frame)
    if output_folder is not None:        # Save the processed image
        processed_image_path = os.path.join(output_folder, "1_Detection_YOLO_Mask.jpg")
        cv2.imwrite(processed_image_path, resized_masked_previus_frame)
    
    # block the code in order to analyse the behavior frame to frame
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def debugPrintMaskRefinement(refined_mask, box, previus_frame, frame_width, frame_height, output_folder=None):
    
    masked_previus_frame = utils.draw_mask(previus_frame, box, refined_mask, object_detected, color_mask=(255, 0, 0))
    resized_masked_previus_frame = cv2.resize(masked_previus_frame, (frame_width // 2, frame_height // 2), interpolation=cv2.INTER_AREA)
    
    cv2.imshow("Refined Masked Image", resized_masked_previus_frame)
    if output_folder is not None:        # Save the processed image
        processed_image_path = os.path.join(output_folder, "2_MaskRefinment.jpg")
        cv2.imwrite(processed_image_path, resized_masked_previus_frame)
    
    # block the code in order to analyse the behavior frame to frame
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def debugPrintMotionEstimation(previus_frame, previus_poi, next_poi, A, box, box_t, output_folder=None):
    
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

    cv2.imshow("Motion Image", frame_points)
    if output_folder is not None:        # Save the processed image
        frame_points_path = os.path.join(output_folder, "3_MotionEstimation.jpg")
        cv2.imwrite(frame_points_path, frame_points)
    
    # block the code in order to analyse the behavior frame to frame
    cv2.waitKey(0)
    cv2.destroyAllWindows() 

def debugPrintFrameCrop(cropped_previus_frame, resized_mask, cropped_next_frame, resized_box, box, frame, output_folder=None):
    frame_width = 1000
    frame_height = 600
    
    # Draw the original box and the resized box
    frame_br = frame.copy()
    x1, y1, x2, y2 = map(int, box) 
    cv2.rectangle(frame_br, (x1, y1), (x2, y2), (255, 0, 0), 1)

    x1_r, y1_r, x2_r, y2_r = map(int, resized_box) 
    cv2.rectangle(frame_br, (x1_r, y1_r), (x2_r, y2_r), (255, 0, 0), 3)

    cropped_previus_frame_mask = utils.draw_mask(cropped_previus_frame, masks=resized_mask)

    resized_frame_br = cv2.resize(frame_br, (frame_width//2, frame_height//2), interpolation=cv2.INTER_AREA)
    
    h, w = cropped_next_frame.shape[:2]          # Get the originale dimension 
    aspect_ratio = int( w / h )                  # compute the ratio to manintain the same proporsion
    cropped_previus_frame_mask = cv2.resize(cropped_previus_frame_mask, (int(frame_height//4)*aspect_ratio, frame_height//4), interpolation=cv2.INTER_AREA)
    
    h, w = cropped_next_frame.shape[:2]     # Get the originale dimension 
    aspect_ratio = int(w / h )                   # compute the ratio to manintain the same proporsion
    cropped_next_frame = cv2.resize(cropped_next_frame, (int(frame_height//4)*aspect_ratio, frame_height//4), interpolation=cv2.INTER_AREA)

    cropped_frames = np.vstack((cropped_previus_frame_mask, cropped_next_frame))
    final_image = np.hstack((resized_frame_br,cropped_frames))

    cv2.imshow("Resize", final_image)
    if output_folder is not None:        
        final_images_path = os.path.join(output_folder, "4_Resize&Crop.jpg")
        cv2.imwrite(final_images_path, final_image)
    
    # block the code in order to analyse the behavior frame to frame
    cv2.waitKey(0) 
    cv2.destroyAllWindows()

def debugPrintSegmentation(blurred, mask, output_folder=None):

    # overlay the mask to the image as a translucent layer
    masked_image = utils.draw_mask(blurred, masks=mask)

    cv2.imshow("Masked Frame & Histogram", masked_image)
    if output_folder is not None:        
        combined_image_path = os.path.join(output_folder, "6_Segmentation.jpg")
        cv2.imwrite(combined_image_path, masked_image)
    
    cv2.waitKey(0) 
    cv2.destroyAllWindows()



if __name__ == "__main__":

    video_path = 'Demo/Video/Car6.mp4'
    output_folder = os.path.join('test/Global/Car6')
    object_detected = 'car'

    test_discreto_video(video_path, object_detected, output_folder=output_folder,  saveVideo=True, debugPrint=True)


