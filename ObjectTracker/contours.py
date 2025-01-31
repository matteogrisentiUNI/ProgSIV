import cv2
import numpy as np

def get_major_contours(image, min_contour_area=150, closing_kernel_size=(5, 5)):
    """
    Detect and return the major contours in the input image, forcing closure on almost-closed contours.

    Args:
        image (np.ndarray): Input image (grayscale or BGR).
        min_contour_area (int): Minimum area to consider a contour as "major".
        closing_kernel_size (tuple): Kernel size for morphological closing.

    Returns:
        list: A list of major contours.
    """
    # Convert the image to grayscale if it's in color
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        print("Converted image to grayscale.")
    else:
        gray = image
        print("Image is already grayscale.")

    brightness = max(100-np.mean(gray), 0)
    print("Brightness adj", brightness)

    # Apply Gaussian Blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (7, 7), 0)
    print("Applied Gaussian Blur.")

    # Increase contrast
    blurred = cv2.addWeighted(blurred, 1.08, np.zeros_like(blurred), 0, brightness)
    print("Increased contrast.")

    # Perform edge detection using Canny
    edges = cv2.Canny(blurred, 50, 150)
    print("Performed Canny edge detection.")

    # Morphological closing to force contour closure
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, closing_kernel_size)
    closed_edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
    dilation_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))  # Adjust kernel size for thickness
    thicker_edges = cv2.dilate(closed_edges, dilation_kernel, iterations=1)
    found_edges = thicker_edges #cv2.GaussianBlur(thicker_edges, (5, 5), 0)
    print("Applied morphological closing to connect gaps in edges.")

    # Find contours
    contours, _ = cv2.findContours(found_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    print(f"Found {len(contours)} contours in the image.")

    # Filter major contours by area
    major_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > min_contour_area]
    print(f"Filtered {len(major_contours)} major contours (area > {min_contour_area}).")
    
    return major_contours, found_edges  # Returning the modified edge image for debugging

def calculate_contour_descriptors(contour):
    """
    Calculate shape descriptors and curvature/edge features for a given contour.
    """
    if len(contour.shape) == 3:
        contour = contour[:, 0, :]  # Flatten to Nx2

    # Compute Moments
    M = cv2.moments(contour)
    if M["m00"] != 0:
        centroid = (M["m10"] / M["m00"], M["m01"] / M["m00"])
    else:
        centroid = (0, 0)

    # Compactness
    area = cv2.contourArea(contour)
    perimeter = cv2.arcLength(contour, True)
    compactness = (perimeter ** 2) / (4 * np.pi * area) if area > 0 else 0

    # Eccentricity
    if len(contour) >= 5:  # Minimum 5 points needed to fit an ellipse
        _, (major_axis, minor_axis), _ = cv2.fitEllipse(contour)
        if major_axis < minor_axis:
            major_axis, minor_axis = minor_axis, major_axis
        eccentricity = np.sqrt(1 - (minor_axis ** 2 / major_axis ** 2))
    else:
        eccentricity = 0

    # Convexity
    hull = cv2.convexHull(contour)
    hull_perimeter = cv2.arcLength(hull, True)
    convexity = perimeter / hull_perimeter if hull_perimeter > 0 else 0

    # Contour Curvature
    curvature = []
    k = 3
    for i in range(len(contour)):
        prev_point = contour[i - k]
        curr_point = contour[i]
        next_point = contour[(i + k) % len(contour)]

        area = 0.5 * np.abs(
            prev_point[0] * (curr_point[1] - next_point[1]) +
            curr_point[0] * (next_point[1] - prev_point[1]) +
            next_point[0] * (prev_point[1] - curr_point[1])
        )
        edge1 = np.linalg.norm(prev_point - curr_point)
        edge2 = np.linalg.norm(curr_point - next_point)
        edge3 = np.linalg.norm(next_point - prev_point)
        curvature_value = (4 * area) / (edge1 * edge2 * edge3 + 1e-10)
        curvature.append(curvature_value)

    # Simplify the contour as a polygon, approximating corners as straight lines (epsilon=0.02*perimeter)
    epsilon = 0.005 * perimeter
    approx = cv2.approxPolyDP(contour, epsilon, True)
    num_corners = len(approx)

    # Get the number of curves (starting from a corner, count how many times does the curve change direction)
    num_curves = 0
    prev_angle = None
    for i in range(len(approx)):
        prev_point = approx[i - 1][0]
        curr_point = approx[i][0]
        next_point = approx[(i + 1) % len(approx)][0]

        # Compute the angle between the edges
        edge1 = prev_point - curr_point
        edge2 = next_point - curr_point
        angle = np.arccos(np.dot(edge1, edge2) / (np.linalg.norm(edge1) * np.linalg.norm(edge2) + 1e-10))
        if prev_angle is not None:
            if angle > prev_angle:
                num_curves += 1
        prev_angle = angle

    # draw and show the white contour over black background
    '''img = np.zeros((512, 512, 3), np.uint8)
    cv2.drawContours(img, [approx], -1, (255, 255, 255), 3)
    cv2.imshow('image', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()'''

    return {
        "Centroid": centroid,
        "Compactness": compactness,
        "Eccentricity": eccentricity,
        "Convexity": convexity,
        "Contour Curvature": curvature,
        "Number of Corners": num_corners,
        "Number of Curves": num_curves,
    }
