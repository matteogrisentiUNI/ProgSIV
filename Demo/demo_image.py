import os
import cv2 
from ObjectTracker import ObjectTracker, draw_mask, extract_region_of_interest_with_mask


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
    image_path = 'Demo/Images/Car.jpg'  
    output_folder = 'Demo/YOLO_Image/Car'
    os.makedirs(output_folder, exist_ok=True)

    try:
        # Call the function to process the image and extract the box of roi
        masks,boxes = process_image(image_path, 'car', output_folder)

        # Extract the roi of the first box
        contours, region, color_histogram = extract_region_of_interest_with_mask(image_path, masks[0], boxes[0], output_folder)

        print("Processing completed successfully.")

    except Exception as e:
        print(f"Error during processing: {e}")



if __name__ == "__main__":
    main()

