import os
import cv2 
import numpy as np
from ObjectTracker import detection, mask_refinement, feature_extraction, region_extraction

'''
# Performs object detection of the target class on an image, 
# Returns the boxes founded.
def process_image(image_path, target_class, output_folder):

    print(f"YOLO PROCESS IMAGE")
    
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"\tError: Unable to load the image from {image_path}")

    # Initialize Object Tracker
    tracker = ObjectTracker(target_class=target_class, conf_threshold=0.1)
    # print("\tTracker initialized. Starting detection...")

    # Perform YOLO detection
    detections = tracker.detect(image)
    # print(f"\tDetection completed: {len(detections)} objects found")

    # Extract detection data
    boxes = [d['box'] for d in detections]
    masks = [d['mask'] for d in detections]
    class_names = [d['class_name'] for d in detections]

    # Draw masks and bounding boxes on the image
    masked_image = draw_mask(image, boxes, masks, class_names)

    # Save the processed image
    processed_image_path = os.path.join(output_folder, "YOLOMasked.jpg")
    cv2.imwrite(processed_image_path, masked_image)
    # print(f"\tProcessed image saved at {processed_image_path}")

    return masks,boxes
'''

def main():
    # Paths
    image_path = 'Demo/Images/person.jpg'  
    output_folder = 'Demo/YOLO_Image/person'
    os.makedirs(output_folder, exist_ok=True)

    try:
        # Call the function to process the image and extract the box of roi
        mask, boxes = detection(image_path, 'person', output_folder)

        # Load an image
        image = cv2.imread(image_path)  # Input image (BGR format)
        im_x, im_y = image.shape[:2]  # Shape of the image (height, width)
        x1, y1, x2, y2 = boxes[0]
        # Crop image and mask to get the segmented ROI
        bounded_image = image[y1:y2, x1:x2] # Cut the region of interest from the image
        resized_mask = cv2.resize(mask[0], (im_y, im_x))  # Resize mask to image dimensions
        bounded_mask = resized_mask[y1:y2, x1:x2] # Extract the corresponding mask region using integer indices
        refined_mask = mask_refinement.refine_mask(bounded_image, bounded_mask) # Refine the mask

        # Find centroid
        centroid = region_extraction.find_centroid(refined_mask)

        # Extract the roi of the first box
        color_histogram = feature_extraction.histogram_extraction(image, refined_mask)

        '''# Create a pink mask (BGR color: (255, 105, 180) for pink)
        pink_mask = np.zeros_like(bounded_image, dtype=np.uint8)
        pink_mask[:, :] = (180, 105, 255)  # Pink color in BGR

        # Apply the mask to the pink overlay
        pink_translucent_mask = cv2.bitwise_and(pink_mask, pink_mask, mask=refined_mask)

        # Blend the pink mask with the bounded image
        alpha = 0.5  # Transparency factor
        overlay = cv2.addWeighted(bounded_image, 1 - alpha, pink_translucent_mask, alpha, 0)

        # Draw the centroid as a red dot
        if centroid is not None:
            # Centroid coordinates (rounded to integers)
            centroid_x, centroid_y = int(centroid[0]), int(centroid[1])
            
            # Draw a red dot at the centroid location
            cv2.circle(overlay, (centroid_x, centroid_y), 5, (0, 0, 255), -1)  # Red color in BGR, radius 5

        # Show the resulting image
        cv2.imshow("Bounded Image with Pink Mask Overlay", overlay)
        cv2.waitKey(0)
        cv2.destroyAllWindows()'''

        blurred = cv2.GaussianBlur(bounded_image, (5, 5), sigmaX=0)

        new_mask = region_extraction.histogram_based_region_growing(bounded_image, centroid)

        #print("Processing completed successfully.")

        # draw the new mask over the original image and show it
        #new_mask_image = cv2.cvtColor(new_mask, cv2.COLOR_GRAY2BGR)
        #cv2.imshow("New Mask", new_mask_image)
        #cv2.waitKey(0)
        #cv2.destroyAllWindows()

    except Exception as e:
        print(f"Error during processing: {e}")

if __name__ == "__main__":
    main()