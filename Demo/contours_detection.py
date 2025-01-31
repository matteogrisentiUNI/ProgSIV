import cv2
import random
import numpy as np
from ObjectTracker import contours

# Example usage:
if __name__ == "__main__":
    # Load the cropped bounding box region
    cropped_image = cv2.imread("Demo/YOLO_Image/person/1_Box.jpg")

    # Get major contours with forced closure
    major_contours, closed_edges = contours.get_major_contours(
        cropped_image, min_contour_area=150, closing_kernel_size=(5, 5)
    )

    # Draw the contours on the image for visualization
    output_image = cropped_image.copy()
    contours_image = np.zeros_like(output_image)
    for contour in major_contours:
        # Generate a random color
        color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
        # Draw the contour with the random color and fill it with the same translucent color
        cv2.drawContours(output_image, [contour], -1, color, 2)
        cv2.fillPoly(output_image, [contour], color)
        
        # Draw the contour on the contours image
        cv2.drawContours(contours_image, [contour], -1, (255, 255, 255), 2)

    # Display the result
    cv2.imshow("Closed Edges", closed_edges)  # Debug: Closed edges image
    cv2.imshow("Major Contours", contours_image)
    cv2.imshow("Major Contours with Closure", output_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    