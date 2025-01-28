import cv2
import os
import numpy as np
import matplotlib.pyplot as plt


#Extracts the first set of feature from the YOLO Segmentation, it return:
# - Contours of the object
# - Region of the object ( mask )
# - Histogram of the object
def extract_region_of_interest_with_mask(image_path, mask, box, output_folder):
   
    print(f"FIRST FEATURES EXTRACTION")

    image = cv2.imread(image_path)                      # Load the image
    if image is None:
        raise FileNotFoundError(f"\tError: Unable to load the image from {image_path}")
    
    # Ensure mask dimensions match the image dimensions
    if len(mask.shape) == 2 and len(image.shape) == 3:  # we have to add a dimension becouse the image has 3 dimension (width, hight, and color dimesnion)
        mask = np.expand_dims(mask, axis=-1)            # the mask has only two dimension width and height, so for compability we have to add a third dimension

    # mask coming from YOLO has a different scale respect to the image so we have to rescale it. 
    # in the resizing we increse the dimension, to define the new pixels we use the INTER_NEAREST
    # so we associate at each new pixel the value of the nearest pixel in the original image
    # in this way we don't introduce greies and shades
    if mask.shape[:2] != image.shape[:2]:
        print("\tResizing mask to match image dimensions.")
        mask = cv2.resize(mask, (image.shape[1], image.shape[0] ), interpolation=cv2.INTER_NEAREST)

    # Validate dimensions
    if mask.shape[:2] != image.shape[:2]:
        raise ValueError("\tError: The mask dimensions still do not match the image dimensions.", mask.shape, image.shape)


    # Since in the nextr fram we will work with the box, extract from image and mask the box part
    x, y, w, h = box
    image = image[y:y+h, x:x+w]
    mask = mask[y:y+h, x:x+w]

    box_path = os.path.join(output_folder, "1 Box.jpg")
    cv2.imwrite(box_path, image)

    # Ensure the mask is uint8 before saving
    if mask.max() <= 1.0:                       # if the mask is in float
        mask = (mask * 255).astype(np.uint8)    # convert it in to value between 0-255
    else:                                       # if it is already expresed trought integer
        mask = mask.astype(np.uint8)            # convert directly
    
    mask_path = os.path.join(output_folder, "2 Region.jpg") 
    cv2.imwrite(mask_path, mask)

    # Find contours in the mask
    contours = contourns_extraction(mask, output_folder)
    
    # Apply the mask to extract the region of interest
    roi = cv2.bitwise_and(image, image, mask=mask)

    roi_path = os.path.join(output_folder, "4 ROI.jpg")   
    cv2.imwrite(roi_path, roi)

    #Calculate the color histogram of the ROI 
    color_histogram = histogram_extraction(roi, mask, output_folder)
    
    return contours, mask, color_histogram


# Extract the contourns from a mask 
def contourns_extraction(mask, output_folder=False):
   
    log = "\tCONTOURNS EXTRACTION: "

    try:
        # Find contours in the mask
        contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        #NB: cv2.CHAIN_APPROX_SIMPLE allows not to store all the edges ( point of contours ), 
        #    only the edges that are at the ends of straight lines

        # Create an image with the mask margins
        if output_folder:        
            margins_image = np.zeros_like(mask, dtype=np.uint8)
            cv2.drawContours(margins_image, contours, -1, (255), 1)  
            margins_path = os.path.join(output_folder, "3.2 Contours.jpg")
            cv2.imwrite(margins_path, margins_image)

        print(log, "OK ")

    except ValueError as err:
        print(log, "ERROR \n\t ", err)

    return contours

# Extract the contourns from an image and a mask
def histogram_extraction(image, mask, output_folder=False):
   
    log = "\tHISTOGRAM EXTRACTION "

    try:
               
        # Ensure mask dimensions match the image dimensions
        if len(mask.shape) == 2 and len(image.shape) == 3:  # we have to add a dimension becouse the image has 3 dimension (width, hight, and color dimesnion)
            mask = np.expand_dims(mask, axis=-1)            # the mask has only two dimension width and height, so for compability we have to add a third dimension

        # Mask and Image need to have the same shape, otherwise the mask will extract a wrong part of the image 
        if mask.shape[:2] != image.shape[:2]:
            log = log + " -> Resising mask: "
            mask = cv2.resize(mask, (image.shape[1], image.shape[0] ), interpolation=cv2.INTER_NEAREST)

        # Validate dimensions
        if mask.shape[:2] != image.shape[:2]:
            raise ValueError("The mask dimensions still do not match the image dimensions.", mask.shape, image.shape)

        # Calculate the color histogram of the mask part of the image
        bgr_image = cv2.split(image)                # Separate the image in its three R,G and B planes
        histSize = 256                              # Each scale of color is a bin
        histRange = (0, 256)                        # Define the range for each plane
        accumulate = False                          # We want that each bin has the same dimension
        
        b_hist = cv2.calcHist(bgr_image, [0], mask, [histSize], histRange, accumulate=accumulate)
        g_hist = cv2.calcHist(bgr_image, [1], mask, [histSize], histRange, accumulate=accumulate)
        r_hist = cv2.calcHist(bgr_image, [2], mask, [histSize], histRange, accumulate=accumulate)   
        
        color_histogram = {
            "blue": b_hist,     
            "green": g_hist,        
            "red": r_hist
        }

        
        # Normalize histograms for better comparability
        for color in color_histogram:               
            color_histogram[color] = color_histogram[color] / color_histogram[color].sum()


        if output_folder:                   # Plot and save the histograms
            plt.figure(figsize=(10, 6))
            plt.title("Color Histogram ")
            plt.xlabel("Pixel Intensity")
            plt.ylabel("Normalized Frequency")
            plt.plot(color_histogram["blue"], color="blue", label="Blue")
            plt.plot(color_histogram["green"], color="green", label="Green")
            plt.plot(color_histogram["red"], color="red", label="Red")
            plt.legend()
            histogram_path = os.path.join(output_folder, "5.2 Color_Histogram.png")
            plt.savefig(histogram_path)
            plt.close()

        print(log, "OK ")

    except ValueError as err:
        print(log, "ERROR \n\t ", err)


    return color_histogram

#Extracts the box regions of the image where objects were detected
def extract_region_of_interest(image_path, box, output_folder):

    print(f"ROI BOX ESTRACTION")
    x, y, w, h = box

    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"\tError: Unable to load the image from {image_path}")

    print("\tExtracting detected region: ", box)
    roi = image[y:y+h, x:x+w]
    print("\tRoi shape:", roi.shape)

     # Save each ROI
    roi_path = os.path.join(output_folder, "ROI_Box.jpg")
    cv2.imwrite(roi_path, roi)
    print(f"\tROI saved as {roi_path}")
   
    return roi