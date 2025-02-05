import os
import cv2 
import random
import numpy as np
from ObjectTracker import ObjectTracker, draw_mask, mask_refinement, feature_extraction, region_extraction, contours

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
def visualize_superpixels_with_random_colors(image, labels):
    # Create a blank image to hold the colored superpixels
    height, width, _ = image.shape
    colored_superpixels = np.zeros((height, width, 3), dtype=np.uint8)
    
    # Assign a random color to each superpixel
    unique_labels = np.unique(labels)
    random_colors = {label: [random.randint(0, 255) for _ in range(3)] for label in unique_labels}
    
    for label in unique_labels:
        mask = labels == label
        colored_superpixels[mask] = random_colors[label]
    
    # Show the visualization
    cv2.imshow("Superpixels Visualization", colored_superpixels)

def main():
    # Paths
    image_path = 'Demo/Images/person.jpg'  
    output_folder = 'Demo/YOLO_Image/person'
    os.makedirs(output_folder, exist_ok=True)

    try:
        # Call the function to process the image and extract the box of roi
        mask, boxes = process_image(image_path, 'person', output_folder)

        # Load an image
        image = cv2.imread(image_path)  # Input image (BGR format)
        im_x, im_y = image.shape[:2]  # Shape of the image (height, width)
        x1, y1, x2, y2 = boxes[0]
        x1 = max(0,int(x1-(x2-x1)*0.1))
        x2 = min(im_y,int(x2+(x2-x1)*0.1))
        y1 = max(0, int(y1-(y2-y1)*0.1))
        y2 = min(im_x,int(y2+(y2-y1)*0.1))
        # Crop image and mask to get the segmented ROI
        bounded_image = image[y1:y2, x1:x2] # Cut the region of interest from the image
        blurred = cv2.GaussianBlur(bounded_image, (3, 3), sigmaX=0)
        resized_mask = cv2.resize(mask[0], (im_y, im_x))  # Resize mask to image dimensions
        bounded_mask = resized_mask[y1:y2, x1:x2] # Extract the corresponding mask region using integer indices
        refined_mask = mask_refinement.refine_mask(bounded_image, bounded_mask) # Refine the mask

        # Extract the histogram of the ROI
        color_histogram = feature_extraction.histogram_extraction(blurred, refined_mask)

        # Perform SLIC segmentation and compute average BGR values
        labels, mask, result, cluster_info = region_extraction.slic_segmentation(blurred)
        # Visualize superpixels with random colors
        #visualize_superpixels_with_random_colors(blurred, labels)

        #cv2.imshow("SLIC", result)
        #cv2.imshow("Result Mask", mask)
        #cv2.waitKey(0)
        #cv2.destroyAllWindows()

        # Assuming you have slic_results from previous processing
        result_mask = region_extraction.histogram_based_refinement(
            image=blurred,
            initial_labels=labels,
            pred_hist=color_histogram,
            tolerance=10
        )

    except Exception as e:
        print(f"Error during processing: {e}")

if __name__ == "__main__":
    main()