import cv2
import time
import psutil
import numpy as np
import os
from LOBES import detection, mask_motion_estimation, segmentation, feature_extraction, utils, mask_refinement


# DETECTION PHASE: Yolo Detection + Mask Refinment
def detection_complete(frame, object_class):
    #print('- DETECTION ')
    try:
        masks, boxes = detection(frame, object_class)
    except Exception as e:
        print(f"\tERROR: {e}")
        return
    
    if not masks:
        print(f'\tERROR:No {object_class} detected in the first frame')
        return
    
    # Take only one object ( mask-box ), so the sistem is single object tracker
    mask = masks[0].astype(np.uint8)
    box = boxes[0].astype(np.float32)

# - MASK REFINEMENT 
    #print('- MASK REFINEMENT ')
    try:
        refined_mask = mask_refinement.refine_mask(frame, mask)
    except Exception as e:
        print(f"\tERROR: {e}")
        return
    
    mask = refined_mask

    histogram = feature_extraction.histogram_extraction(frame, mask)

    return mask, box, histogram

# TRACKING PHASE: Motion + Segmentation
def tracking(prev_frame, prev_histogram, prev_mask, prev_box, prev_A, next_frame, output_folder=None, debugPrint=False):
# - MOTION
    #print('- MOTION ESTIMATION')
    sf=0.85
    try:
        _, _, A = mask_motion_estimation( prev_frame, next_frame, mask=prev_mask )
    except Exception as e:
        print(f"\tERROR: {e}")
        A = prev_A
    scale_x = np.sqrt(A[0, 0]**2 + A[0, 1]**2)
    scale_y = np.sqrt(A[1, 0]**2 + A[1, 1]**2)
    if scale_x!=0 and scale_y!=0:
        prev_scale_x = np.sqrt(A[0, 0]**2 + prev_A[0, 1]**2)
        prev_scale_y = np.sqrt(A[1, 0]**2 + prev_A[1, 1]**2)      
        if  scale_x < prev_scale_x*sf or scale_x > prev_scale_x*(1/sf) or scale_y < prev_scale_y*sf or scale_y > prev_scale_y*(1/sf):
            print("kept previous scale matrix")
            A = prev_A

    '''t_prev = np.array([prev_A[0, 2], prev_A[1, 2]])
    t_curr = np.array([A[0, 2], A[1, 2]])
    mag_prev = np.linalg.norm(t_prev)
    mag_curr = np.linalg.norm(t_curr)
    #print("translation proportion: ",mag_curr/mag_prev)
    if mag_prev > 0 and ((mag_curr / mag_prev) < 0.5 or (mag_curr / mag_prev) > (1/0.5)):
        print("kept previous translation matrix")
        A = prev_A'''
            

    
    next_box = utils.predict_bounding_box_final(next_frame, prev_box, A)
    bb_x1 = int(next_box[0])
    bb_y1 = int(next_box[1])
    bb_x2 = int(next_box[2])
    bb_y2 = int(next_box[3])

    bounded_next_frame = next_frame[bb_y1:bb_y2, bb_x1:bb_x2]


# - SEGMENTATION
    try:
        # Apply Gaussian blur
        blurred_next_frame = cv2.GaussianBlur(bounded_next_frame, (3, 3), sigmaX=0)

        # Decide if the output is valid or not
        next_mask, next_histogram, similarity = segmentation.segmentation(blurred_next_frame, prev_histogram)
        #print(similarity)
        if similarity > 0.29:
            rows, cols = prev_mask.shape
            next_mask = cv2.warpAffine(prev_mask, A, (cols, rows))
            next_histogram = prev_histogram

    except Exception as e:
        print(f"\tSEGMENTATION ERROR: {e}")
        return    
    

    # 3: ensure the mask is of the same size of the image, otherwise add a padding to fill the missing pixels
    next_mask = utils.resize_mask_with_padding(next_mask, next_box, next_frame.shape[0], next_frame.shape[1])

    # 4: shrink the box in order to be closer to the final mask
    next_box = utils.shrink_box_to_mask(next_box, next_mask, threshold=5)

    return next_mask, next_box, next_histogram, A


# LOB&S 
def LOBES(video_path, object_detected, vertical=False, output_folder=None, saveVideo=False, debugPrint=False):
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

    ret, previous_frame = cap.read()
    if vertical:  # if the video is in portrait mode rotate it of 90 degrees
        previous_frame = cv2.rotate(previous_frame, cv2.ROTATE_90_CLOCKWISE)
    frame_count = 1

    cpu_before = psutil.cpu_percent(interval=None)
    memory_before = psutil.virtual_memory().percent


## --- DETECTION --- ##
    mask, box, histogram = detection_complete(previous_frame, object_detected)
    # Save the Output 
    if saveVideo:
        output_frame = utils.draw_mask(previous_frame, box, mask, object_detected)
        resized_output_frame = cv2.resize(output_frame, (frame_width, frame_height), interpolation=cv2.INTER_AREA)
        out.write(resized_output_frame)

    cpu_after = psutil.cpu_percent(interval=None)
    memory_after = psutil.virtual_memory().percent
    
    cpu_usage_list.append((cpu_before + cpu_after) / 2)
    memory_usage_list.append((memory_before + memory_after) / 2)

    # Activate all the debug print ( A LOT !! )
    debugPrint = False
    i=1
    # set prev_A as an identity affine matrix
    prev_A = np.array([[1, 0, 0], [0, 1, 0]], dtype=np.float32)
    while cap.isOpened():
        #print(i)
        #i += 1

        # Start counter for the performance
        frame_start_time = time.time()
        cpu_before = psutil.cpu_percent(interval=None)
        memory_before = psutil.virtual_memory().percent

        # Take the next frame to analize
        ret, next_frame = cap.read()
        if vertical:  # if the video is in portrait mode rotate it of 90 degrees
            next_frame = cv2.rotate(next_frame, cv2.ROTATE_90_CLOCKWISE)
        if not ret:
            print("Video Ended")
            break

## --- TRAKING --- ##
        next_mask, next_box, next_histogram, next_A = tracking(previous_frame, histogram, mask, box, prev_A, next_frame, output_folder=output_folder, debugPrint=debugPrint)
        
        # Save next frame with bounding box an mask in the output video
        if saveVideo:
            output_frame = utils.draw_mask(next_frame, next_box, next_mask, object_detected, color_mask=(255, 0, 0))
            resized_output_frame = cv2.resize(output_frame, (frame_width, frame_height), interpolation=cv2.INTER_AREA)
            out.write(resized_output_frame)
    
        # Update the variable for the next cycle 
        previous_frame = next_frame
        mask = next_mask
        box = next_box
        histogram = next_histogram
        prev_A = next_A

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

def LOBES_debugVersion(video_path, object_detected, output_folder=None, saveVideo=False, startFrameDebug=None, endFrameDebug=None):
    cap = cv2.VideoCapture(video_path)
    os.makedirs(output_folder, exist_ok=True)

    if saveVideo:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        out = cv2.VideoWriter(os.path.join(output_folder, 'tracked_video_v3.mp4'), fourcc, fps, (frame_width, frame_height))


    ret, previous_frame = cap.read()


## --- DETECTION --- ##
    try:
        mask, box, histogram = detection_complete(previous_frame, object_detected)
    except:
        return

    # Save the Output 
    if saveVideo:
        output_frame = utils.draw_mask(previous_frame, box, mask, object_detected)
        resized_output_frame = cv2.resize(output_frame, (frame_width, frame_height), interpolation=cv2.INTER_AREA)
        out.write(resized_output_frame)


## --- TRAKING --- ##
    
    debugPrint = False      # Activate all the debug print ( A LOT !! )
    i=0

    while cap.isOpened():
        i += 1
        print(i)

        ret, next_frame = cap.read()        # Take the next frame to analize
        if not ret:
            print("Video Ended")
            break
        
        if(startFrameDebug is not None):
            if(endFrameDebug is not None):
                if i>= startFrameDebug and i<=endFrameDebug:   debugPrint = True
                else:                                          debugPrint = False
            else:
                if i>= startFrameDebug:                        debugPrint = True
                else:                                          debugPrint = False
        
        
        try: 
            next_mask, next_box, next_histogram = tracking(previous_frame, histogram, mask, box, next_frame, output_folder=output_folder, debugPrint=debugPrint)

            # Save next frame with bounding box an mask in the output video
            if saveVideo:
                output_frame = utils.draw_mask(next_frame, next_box, next_mask, object_detected, color_mask=(255, 0, 0))
                resized_output_frame = cv2.resize(output_frame, (frame_width, frame_height), interpolation=cv2.INTER_AREA)
                
                if debugPrint:
                    cv2.imshow('FINAL FRAME', resized_output_frame)
                    cv2.waitKey(0)
                    cv2.destroyAllWindows()
                
                out.write(resized_output_frame)

            # Update the variable for the next cycle 
            previous_frame = next_frame
            mask = next_mask
            box = next_box
            histogram = segmentation.update_histogram(histogram, next_histogram)
        
        except:
            if saveVideo:
                resized_output_frame = cv2.resize(next_frame, (frame_width, frame_height), interpolation=cv2.INTER_AREA) 
                out.write(resized_output_frame)

    cap.release()
    if saveVideo:
        out.release()
    cv2.destroyAllWindows()
