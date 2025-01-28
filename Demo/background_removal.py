import cv2
from ObjectTracker import contour_descriptors

# Example usage:
if __name__ == "__main__":
    # Load the cropped bounding box region
    cropped_image = cv2.imread("Demo/YOLO_Image/person/1_Box.jpg")

    # Get major contours with forced closure
    major_contours, closed_edges = contour_descriptors.get_major_contours(
        cropped_image, min_contour_area=79, closing_kernel_size=(5, 5)
    )

    # Draw the contours on the image for visualization
    output_image = cropped_image.copy()
    cv2.drawContours(output_image, major_contours, -1, (0, 255, 0), 2)

    # Display the result
    cv2.imshow("Closed Edges", closed_edges)  # Debug: Closed edges image
    cv2.imshow("Major Contours with Closure", output_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
