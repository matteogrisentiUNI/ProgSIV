import os
import cv2
import numpy as np
from ObjectTracker import detection, mask_refinement, feature_extraction, region_extraction

def main():
    # Paths
    image_path = 'Demo/Images/person.jpg'  
    output_folder = 'Demo/YOLO_Image/person'
    os.makedirs(output_folder, exist_ok=True)

    try:        
        # Load an image
        image = cv2.imread(image_path)  # Input image (BGR format)
        im_x, im_y = image.shape[:2]  # Shape of the image (height, width)
        # Call the function to detect the box of roi
        masks, boxes = detection(image, 'person', output_folder)
        x1, y1, x2, y2 = boxes[0]
        x1 = max(0,int(x1-(x2-x1)*0.1))
        x2 = min(im_y,int(x2+(x2-x1)*0.1))
        y1 = max(0, int(y1-(y2-y1)*0.1))
        y2 = min(im_x,int(y2+(y2-y1)*0.1))
        # Crop image and mask to get the segmented ROI
        bounded_image = image[y1:y2, x1:x2] # Cut the region of interest from the image
        blurred = cv2.GaussianBlur(bounded_image, (3, 3), sigmaX=0)
        resized_mask = cv2.resize(masks[0], (im_y, im_x))  # Resize mask to image dimensions
        bounded_mask = resized_mask[y1:y2, x1:x2] # Extract the corresponding mask region using integer indices
        refined_mask = mask_refinement.refine_mask(bounded_image, bounded_mask) # Refine the mask

        # Extract the histogram of the ROI
        predicted_histogram = feature_extraction.histogram_extraction(blurred, refined_mask)

        # Extract the region of interest
        mask = region_extraction.segmentation(blurred, predicted_histogram)

        # Save the mask
        mask_path = os.path.join(output_folder, "mask.jpg")
        cv2.imwrite(mask_path, mask)
        #print(f"\tMask saved as {mask_path}")

    except Exception as e:
        print(f"Error during processing: {e}")

if __name__ == "__main__":
    main()