import os
import cv2 
import numpy as np
from ObjectTracker import ObjectTracker, draw_mask, mask_refinement, region_extraction

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


def main():
    # Paths
    image_path = 'Demo/Images/person.jpg'  
    output_folder = 'Demo/YOLO_Image/person'
    os.makedirs(output_folder, exist_ok=True)

    try:
        # Call the function to process the image and extract the box of roi
        mask, boxes = process_image(image_path, 'person', output_folder)
        # Load an image and its corresponding mask
        image = cv2.imread(image_path)  # Input image (BGR format)

        # Refine the mask
        refined_mask = mask_refinement.refine_mask(image, mask[0])
        # Call the function
        centroid = region_extraction.find_centroid(refined_mask)

        # make the original mask into a 3 channel image, where the region inside the mask is bright red
        mask_image = cv2.cvtColor(mask[0], cv2.COLOR_GRAY2BGR)
        mask_image = mask_image.astype(np.float32) / 255.0
        mask_image = np.where(mask_image > 0, [0, 0, 255], [0, 0, 0]).astype(np.uint8)

        # make the refined mask into a 3 channel image, where the region inside the mask is bright blue
        refined_mask_image = cv2.cvtColor(refined_mask, cv2.COLOR_GRAY2BGR)
        refined_mask_image = refined_mask_image.astype(np.float32) / 255.0
        refined_mask_image = np.where(refined_mask_image > 0, [255, 0, 0], [0, 0, 0]).astype(np.uint8)

        # resize the original mask, the refined mask and the image to the same size
        mask_image = cv2.resize(mask_image, (image.shape[1], image.shape[0]))
        refined_mask_image = cv2.resize(refined_mask_image, (image.shape[1], image.shape[0]))

        # draw the original mask and the refined mask over the original image as translucent masks
        combined_image = cv2.addWeighted(image, 0.5, mask_image, 0.5, 0)
        combined_image = cv2.addWeighted(image, 0.5, refined_mask_image, 0.5, 0)

        # draw the centroid on the combined image
        cv2.circle(combined_image, centroid, 5, (0, 0, 255), -1)

        # Display the combined image
        cv2.imshow("Mask Refinement", combined_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        # create a png image with the original image keeping only the points inside the refined mask
        final_image = cv2.bitwise_and(image, image, mask=refined_mask)
        cv2.imwrite(os.path.join(output_folder, "7_refined_mask_image.png"), final_image)

    except Exception as e:
        print(f"Error during processing: {e}")

if __name__ == "__main__":
    main()