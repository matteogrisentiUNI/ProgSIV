import cv2
from ObjectTracker import region_extraction

'''if __name__ == "__main__":
    # Load a grayscale image
    image = imread("Demo/Images/PersonNBG.png", as_gray=True)
    # convert image to 8-bit grayscale
    image = (image * 255).astype(np.uint8)  # Normalize to 8-bit if needed

    # Perform adaptive region growing
    segmented_image, num_clusters = adaptive_region_growing(image, threshold=20)

    print(f"Number of clusters detected: {num_clusters}")'''

if __name__ == "__main__":
    image_path = "Demo/YOLO_Image/person/1_Box.jpg"  # Replace with your PNG image path
    output_path = "Demo/YOLO_Image/person/6_segmented.png"   # Path to save the processed image
    try:
        # Process the image
        output_image, num_regions, clusters, best_channel = region_extraction.preprocess_and_segment(image_path,1)
        print(f"Number of regions detected: {num_regions}")
        # Save the output image
        cv2.imwrite(output_path, output_image)
        print(f"Segmented image saved to {output_path}")
        # Display the output image
        cv2.imshow("Segmented Image", output_image)
        cv2.waitKey(0)  # Wait for a key press to close the window
        cv2.destroyAllWindows()
    except Exception as e:
        print(f"Error: {e}")
