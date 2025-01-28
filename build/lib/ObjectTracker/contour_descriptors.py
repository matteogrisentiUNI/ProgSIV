import cv2
import numpy as np

def calculate_contour_descriptors(contour):
    """
    Calculate shape descriptors and curvature/edge features for a given contour.
    
    Args:
        contour (ndarray): Contour from OpenCV's findContours (Nx1x2 array of points).
    
    Returns:
        dict: Dictionary containing calculated descriptors:
              - Centroid
              - Compactness
              - Eccentricity
              - Convexity
              - Contour Curvature
              - Number of Corners/Edges
              - Number of Curves (convexity changes)
    """
    # Ensure the contour is in the correct format
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
        eccentricity = np.sqrt(1 - (minor_axis ** 2 / major_axis ** 2))
    else:
        eccentricity = 0  # Cannot calculate if there are fewer than 5 points

    # Convexity
    hull = cv2.convexHull(contour)
    hull_perimeter = cv2.arcLength(hull, True)
    convexity = perimeter / hull_perimeter if hull_perimeter > 0 else 0

    # Contour Curvature
    curvature = []
    k = 3  # Use a small window of 3 points for curvature calculation
    for i in range(len(contour)):
        prev_point = contour[i - k]
        curr_point = contour[i]
        next_point = contour[(i + k) % len(contour)]
        
        # Triangle area-based curvature calculation
        area = 0.5 * np.abs(
            prev_point[0] * (curr_point[1] - next_point[1]) +
            curr_point[0] * (next_point[1] - prev_point[1]) +
            next_point[0] * (prev_point[1] - curr_point[1])
        )
        edge1 = np.linalg.norm(prev_point - curr_point)
        edge2 = np.linalg.norm(curr_point - next_point)
        edge3 = np.linalg.norm(next_point - prev_point)
        curvature_value = (4 * area) / (edge1 * edge2 * edge3 + 1e-10)  # Avoid division by zero
        curvature.append(curvature_value)

    # Number of Corners/Edges
    corners = cv2.goodFeaturesToTrack(np.float32(contour), maxCorners=50, qualityLevel=0.01, minDistance=10)
    num_corners = len(corners) if corners is not None else 0

    # Number of Curves (changes in convexity)
    is_convex = [cv2.pointPolygonTest(hull, tuple(pt), False) >= 0 for pt in contour]
    num_curves = sum(is_convex[i] != is_convex[i - 1] for i in range(1, len(is_convex)))

    # Return descriptors
    return {
        "Centroid": centroid,
        "Compactness": compactness,
        "Eccentricity": eccentricity,
        "Convexity": convexity,
        "Contour Curvature": curvature,
        "Number of Corners": num_corners,
        "Number of Curves": num_curves,
    }
