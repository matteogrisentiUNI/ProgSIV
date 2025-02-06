import os
import cv2
import numpy as np
from ObjectTracker import detection, mask_refinement, feature_extraction, segmentation, utils      

def main():
    # Paths
    image_path = 'Demo/Images/person.jpg'  
    output_folder = 'Demo/YOLO_Image/person'
    os.makedirs(output_folder, exist_ok=True)

    try:        
        # Load an image
        image = cv2.imread(image_path)  # Input image (BGR format)

        # Call the function to detect the box of roi
        masks, boxes = detection(image, 'person', output_folder)

        # Resize the image and the mask
        resized_image, resized_mask = utils.resize(image, boxes[0], masks[0], 150000)

        # Apply Gaussian blur
        blurred = cv2.GaussianBlur(resized_image, (3, 3), sigmaX=0)

        # Refine the mask
        refined_mask = mask_refinement.refine_mask(resized_image, resized_mask)

        # Extract the histogram of the ROI
        predicted_histogram = feature_extraction.histogram_extraction(blurred, refined_mask)

        # Extract the region of interest
        mask = segmentation.segmentation(blurred, predicted_histogram)

        # Save the mask
        mask_path = os.path.join(output_folder, "mask.jpg")
        cv2.imwrite(mask_path, mask)
        #print(f"\tMask saved as {mask_path}")

    except Exception as e:
        print(f"Error during processing: {e}")

if __name__ == "__main__":
    main()