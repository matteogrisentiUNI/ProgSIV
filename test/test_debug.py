import cv2
import time
import psutil
import glob
import numpy as np
import os
from LOBES import detection, mask_motion_estimation, segmentation, feature_extraction, utils, mask_refinement



# DETECTION FUNCTION ###################################################################################
def detection_complete(frame, object_class, target_pixels=150000):
    print('- DETECTION ')
    try:
        masks, boxes = detection(frame, object_class)
    except Exception as e:
        print(f"\tERROR: {e}")
        return
    
    if not masks:
        print(f'\tERROR: No {object_detected} detected in the first frame')
        return
    
    # Take only one object ( mask-box ), so the sistem is single object tracker
    mask = masks[0].astype(np.uint8)
    box = boxes[0].astype(np.float32)

# - MASK REFINEMENT 
    print('- MASK REFINEMENT ')
    try:
        refined_mask = mask_refinement.refine_mask(frame, mask)
    except Exception as e:
        print(f"\tERROR: {e}")
        return
    
    mask = refined_mask

    histogram = feature_extraction.histogram_extraction(frame, mask)
    # Compute the total number of pixels (iterate over one channel)
    total_pixels = np.sum(histogram['blue'])
    
    # Compute the scaling factor
    scaling_factor = target_pixels / total_pixels
    
    # Rescale each channel
    for channel in histogram:
        for i in range(len(histogram[channel])):
            histogram[channel][i] = int(histogram[channel][i] * scaling_factor)

    return mask, box, histogram




# TRACKING FUNCTION ####################################################################################
def tracking(prev_frame, prev_histogram, prev_mask, prev_box, next_frame, output_folder=None, debugPrint=False):
# - MOTION
    #print('- MOTION ESTIMATION')
    try:
        pp, np, A = mask_motion_estimation( prev_frame, next_frame, mask=prev_mask )
    except Exception as e:
        print(f"\tMOTION ERROR: {e}")
        return
    
    next_box = utils.predict_bounding_box_3(next_frame, prev_box, A)
    bb_x1 = int(next_box[0])
    bb_y1 = int(next_box[1])
    bb_x2 = int(next_box[2])
    bb_y2 = int(next_box[3])

    bounded_next_frame = next_frame[bb_y1:bb_y2, bb_x1:bb_x2]

    if debugPrint:
        debugPrintMotionEstimation(prev_frame, pp, np, A, prev_box, next_box)

# - CROP FRAME
    try:
        cropped_next_frame, scaling_factors = utils.rescale(bounded_next_frame, 150000)

    except Exception as e:
        print(f"\tCROP FRAME ERROR: {e}")
        return
    
    if debugPrint:
        debugPrintFrameCrop(cropped_next_frame)

# - SEGMENTATION
    try:
        # Apply Gaussian blur
        blurred_next_frame = cv2.GaussianBlur(cropped_next_frame, (3, 3), sigmaX=0)

        # Extract the region of interest
        next_mask = segmentation.segmentation(blurred_next_frame, prev_histogram, output_folder=output_folder, debugPrint=debugPrint)

    except Exception as e:
        print(f"\tSEGMENTATION ERROR: {e}")
        return

    if debugPrint:
        debugPrintSegmentation(blurred_next_frame, next_mask)


#   RESCALING: we have to rescale the mask to the original frame dimension
    #print('- RESCALING')
    # 1: Resize the mask to the origina pixel quantity
    original_width = int(next_mask.shape[1] / scaling_factors)
    original_height = int(next_mask.shape[0] / scaling_factors)
    next_mask = cv2.resize(next_mask, (original_width, original_height), interpolation=cv2.INTER_LINEAR)
    
    # Count the number of pixels of the original mask
    prev_pixel_count = cv2.countNonZero(prev_mask)
    new_pixel_count = cv2.countNonZero(next_mask)
    #print(f"Number of pixels in the mask: {prev_pixel_count}")
    #print(f"Number of pixels in the rescaled mask: {new_pixel_count}")
    if new_pixel_count < prev_pixel_count*0.9 or new_pixel_count > prev_pixel_count*1.1:
        rows, cols = prev_mask.shape
        # Apply the affine matrix to the previous mask
        next_mask = cv2.warpAffine(prev_mask, A, (cols, rows), flags=cv2.INTER_NEAREST)
    else:
        # Adding black padding in order the mask match the dimension of the original frame
        next_mask = utils.resize_mask_with_padding(next_mask, next_box, next_frame.shape[0], next_frame.shape[1])

    # 4: shrink the box in order to be closer to the final mask
    next_box = utils.shrink_box_to_mask(next_box, next_mask, threshold=5)

    next_histogram = feature_extraction.histogram_extraction(next_frame, next_mask)

    return next_mask, next_box, next_histogram



#######################################################################################################
def test_discreto_video(video_path, object_detected, output_folder=None, saveVideo=False, debugPrint=False):
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
        
        #if i>= 200 and i<=210:   debugPrint = True
        #else:                    debugPrint = False
        
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
    np.reshape(box, 4)
    x1, y1, x2, y2 = map(int, box) 
    cv2.rectangle(frame_points, (x1, y1), (x2, y2), (255, 0, 0), 3)

    np.reshape(box_t, 4)
    x1_t, y1_t, x2_t, y2_t = map(int, box_t) 
    cv2.rectangle(frame_points, (x1_t, y1_t), (x2_t, y2_t), (0, 255, 0), 2, lineType=4)

    cv2.imshow("Motion Image", frame_points)
    if output_folder is not None:        # Save the processed image
        frame_points_path = os.path.join(output_folder, "3_MotionEstimation.jpg")
        cv2.imwrite(frame_points_path, frame_points)
    
    # block the code in order to analyse the behavior frame to frame
    cv2.waitKey(0)
    #cv2.destroyAllWindows() 

def debugPrintFrameCrop(cropped_next_frame, output_folder=None):
    frame_height = 600
  
    h, w = cropped_next_frame.shape[:2]          # Get the originale dimension 
    aspect_ratio = int( w / h )                  # compute the ratio to manintain the same proporsion
    cropped_previus_frame_mask = cv2.resize(cropped_next_frame, (int(frame_height)*aspect_ratio, frame_height), interpolation=cv2.INTER_AREA)
    
    cv2.imshow("Cropped Next Frame", cropped_previus_frame_mask)
    if output_folder is not None:        
        final_images_path = os.path.join(output_folder, "4_Resize&Crop.jpg")
        cv2.imwrite(final_images_path, cropped_previus_frame_mask)
    
    # block the code in order to analyse the behavior frame to frame
    cv2.waitKey(0) 
    #cv2.destroyAllWindows()

def debugPrintSegmentation(blurred, mask, output_folder=None):

    # overlay the mask to the image as a translucent layer
    masked_image = utils.draw_mask(blurred, masks=mask)

    cv2.imshow("New Mask", masked_image)
    if output_folder is not None:        
        combined_image_path = os.path.join(output_folder, "6_Segmentation.jpg")
        cv2.imwrite(combined_image_path, masked_image)
    
    cv2.waitKey(0) 
    #cv2.destroyAllWindows()



if __name__ == "__main__":

    video_path = 'test/Video/Car2.mp4'
    output_folder = os.path.join('test/Global_Test/Car2')
    object_detected = 'car'

    test_discreto_video(video_path, object_detected, output_folder=output_folder,  saveVideo=True, debugPrint=True)


