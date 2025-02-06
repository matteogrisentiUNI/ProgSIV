import cv2
import numpy as np
from collections import defaultdict
from collections import defaultdict
from scipy.ndimage import gaussian_filter1d
from ObjectTracker import utils

# --- Contour Operations ---
def simplify_contours(contours, epsilon_factor=0.001):
    simplified = []
    for cnt in contours:
        peri = cv2.arcLength(cnt, True)
        epsilon = epsilon_factor * peri
        approx = cv2.approxPolyDP(cnt, epsilon, True)
        simplified.append(approx)
    return simplified

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

def process_contour(contour, distance_threshold=15):
    """
    Process a contour to remove noisy "blob-like" structures based on the given algorithm.
    
    Args:
        contour (numpy.ndarray): The input contour (Nx1x2 array from OpenCV).
        distance_threshold (float): Maximum distance to consider points as neighbors.

    Returns:
        numpy.ndarray: The processed contour.
    """
    # Flatten the contour for easier processing
    contour = contour[:, 0, :]  # Convert Nx1x2 to Nx2
    n = len(contour)

    #print(f"Initial number of points in the contour: {n}")
    
    # Initialize the index manually for the while loop
    i = 0
    while i < len(contour):
        point = contour[i]
        neighbors = []
        #print("Processing point:", i)
        
        # Find neighbors within the distance threshold
        for j, other_point in enumerate(contour):
            if i == j:
                continue
            euclidean_dist = np.linalg.norm(point - other_point)
            direction = True
            
            if euclidean_dist < distance_threshold:
                #print("Adding neighbor:", j)
                # Compute contour-following distance (clockwise) - corrected
                if i < j:
                    clockwise_distance = np.sum([
                        np.linalg.norm(contour[(k + 1) % n] - contour[k]) 
                        for k in range(i, j)
                    ])
                else:
                    clockwise_distance = np.sum([
                        np.linalg.norm(contour[(k + 1) % n] - contour[k]) 
                        for k in range(i, n)
                    ]) + np.sum([
                        np.linalg.norm(contour[(k + 1) % n] - contour[k]) 
                        for k in range(0, j)
                    ])
                
                # Compute contour-following distance (anticlockwise)
                if i > j:
                    anticlockwise_distance = np.sum([
                        np.linalg.norm(contour[(k + 1) % n] - contour[k]) 
                        for k in range(j, i)
                    ])
                else:
                    anticlockwise_distance = np.sum([
                        np.linalg.norm(contour[(k + 1) % n] - contour[k]) 
                        for k in range(j, n)
                    ]) + np.sum([
                        np.linalg.norm(contour[(k + 1) % n] - contour[k]) 
                        for k in range(0, i)
                    ])


                if clockwise_distance > anticlockwise_distance:
                    contour_distance = anticlockwise_distance
                    direction = False
                else:
                    contour_distance = clockwise_distance
                #print(f"Euclidean distance: {euclidean_dist:.2f}, Contour distance: {contour_distance:.2f}")
                
                neighbors.append((j, euclidean_dist, contour_distance, direction))
        
        # If there are neighbors, process them
        if len(neighbors) > 0:
            #print("Total number of neighbors:", len(neighbors))
            # Sort neighbors by descending Contour distance
            neighbors = sorted(neighbors, key=lambda x: x[2], reverse=True)
            # Check for valid connections and remove intermediate points
            for neighbor in neighbors:
                index, euclidean_dist, contour_dist, direction = neighbor
                #print(f"Point {i} -> Neighbor {index}: Euclidean={euclidean_dist:.2f}, Contour={contour_dist:.2f}")
                
                if euclidean_dist < contour_dist/1.05:
                    # Connect the points directly
                    #print(f"Connecting Point {i} and Point {index}, removing intermediate points.")
                    if direction:
                        #print(f"Connecting Point {i} and Point {index}, removing intermediate points clockwise.")
                        # Remove from the original contour all the points that connect the two points along the clockwise direction
                        contour = np.delete(contour, range(i + 1, index), axis=0)
                    else:
                        #print(f"Connecting Point {i} and Point {index}, removing intermediate points anticlockwise.")
                        # Remove from the original contour all the points that connect the two points along the anticlockwise direction
                        contour = np.delete(contour, range(index + 1, n), axis=0)
                        contour = np.delete(contour, range(0, i), axis=0)
                    n = len(contour)
                    #print(f"New number of points in the contour: {n}")
                    break
                #print(f"Point {i} is valid, skipping to the next point.")
        #else: 
            #print(f"Point {i} has no neighbors, skipping to the next point.")
        i += 1
        #print(f"Processed point {i} of {len(contour)}")
    #print(f"Final number of points in the processed contour: {len(contour)}")

    # Apply operation to smoothen the contour
    #contour = smooth_contour(contour, sigma=0.0)

    return contour

def smooth_contour(contour, sigma=1.0):
    # Apply Gaussian filter to x and y coordinates separately
    contour[:, 0] = gaussian_filter1d(contour[:, 0], sigma=sigma)
    contour[:, 1] = gaussian_filter1d(contour[:, 1], sigma=sigma)
    return contour


# --- Histogram Operations ---
def add_histograms(current_histogram, added_histogram):
    """
    Adds two histograms represented as dictionaries ('blue', 'green', 'red').

    Args:
        hist1: First histogram (dictionary with 'blue', 'green', 'red' keys).
        hist2: Second histogram (dictionary with 'blue', 'green', 'red' keys).

    Returns:
        A new histogram where each channel is the sum of the corresponding channels in hist1 and hist2.
    """
    combined_histogram = {}
    for channel in ['blue', 'green', 'red']:
        combined_histogram[channel] = current_histogram[channel] + added_histogram[channel]
    return combined_histogram

def remove_histograms(current_histogram, removed_histogram):
    """
    Adds two histograms represented as dictionaries ('blue', 'green', 'red').

    Args:
        hist1: First histogram (dictionary with 'blue', 'green', 'red' keys).
        hist2: Second histogram (dictionary with 'blue', 'green', 'red' keys).

    Returns:
        A new histogram where each channel is the sum of the corresponding channels in hist1 and hist2.
    """
    refined_histogram = {}
    for channel in ['blue', 'green', 'red']:
        refined_histogram[channel] = current_histogram[channel] - removed_histogram[channel]
    return refined_histogram

def get_superpixel_histogram(image, labels, superpixel_id, bins=256):
    """
    Calculate histogram for a specific superpixel

    Args:
        image: Input image (grayscale or color)
        labels: Superpixel label map from SLIC
        superpixel_id: ID of the superpixel to analyze
        bins: Number of histogram bins (default 256)

    Returns:
        A dictionary with histograms for 'blue', 'green', and 'red' channels.
    """
    # Create mask for the specific superpixel
    mask = (labels == superpixel_id)

    hist_dict = {'blue': np.zeros(bins), 'green': np.zeros(bins), 'red': np.zeros(bins)}

    if len(image.shape) == 2:  # Grayscale
        # Extract pixels belonging to this superpixel
        pixels = image[mask]
        # Calculate histogram
        hist = cv2.calcHist([pixels], [0], None, [bins], [0, 256]).flatten()
        hist_dict['blue'] = hist  # Treat grayscale as blue channel
        hist_dict['green'] = hist
        hist_dict['red'] = hist
    else:  # Color image
        for idx, color in enumerate(['blue', 'green', 'red']):
            pixels = image[:, :, idx][mask]
            hist = cv2.calcHist([pixels], [0], None, [bins], [0, 256]).flatten()
            hist_dict[color] = hist

    return hist_dict

def check_constraints(hist1, hist2, tolerance=10):
    """
    Compares two histograms to ensure the second histogram never exceeds the first.
    
    Args:
        hist1: A dictionary containing histograms for 'blue', 'green', and 'red' channels (reference histogram).
        hist2: A dictionary containing histograms for 'blue', 'green', and 'red' channels (compared histogram).
        tolerance: A tolerance value for comparing histograms (default 1.1).

    Returns:
        True if hist2 is always less than or equal to hist1 for all values in all channels, False otherwise.
    """
    unsatisfied_up = 0
    unsatisfied_down = 0
    for channel in ['blue', 'green', 'red']:
        for i in range(len(hist1[channel])):
            if hist2[channel][i] > (hist1[channel][i]+tolerance):
                unsatisfied_up += 1
            if hist2[channel][i] < (hist1[channel][i]-tolerance):
                unsatisfied_down += 1
    return unsatisfied_up, unsatisfied_down


# --- Superpixel Operations ---
def remove_border_superpixels(labels, min_border_pixels=5):
    """
    Removes superpixels that touch the image borders with at least `min_border_pixels` pixels.

    Args:
        labels (np.ndarray): 2D array of superpixel labels.
        image (np.ndarray): Input image (used to get dimensions).
        min_border_pixels (int): Minimum number of pixels on a border to consider removal.

    Returns:
        np.ndarray: Updated labels where border-touching superpixels are removed.
    """
    height, width = labels.shape
    unique_labels = np.unique(labels)
    
    # Create a set to store labels to be removed
    labels_to_remove = set()
    
    # Check each border
    for label in unique_labels:
        # Create a mask for the current label
        label_mask = labels == label
        
        # Check the borders
        top_border = np.sum(label_mask[0, :])
        bottom_border = np.sum(label_mask[height - 1, :])
        left_border = np.sum(label_mask[:, 0])
        right_border = np.sum(label_mask[:, width - 1])
        
        # If the label has >= `min_border_pixels` on any border, mark it for removal
        if (top_border >= min_border_pixels or
            bottom_border >= min_border_pixels or
            left_border >= min_border_pixels or
            right_border >= min_border_pixels):
            labels_to_remove.add(label)
    
    # Create a mask for valid labels (not in the removal set)
    updated_labels = labels.copy()
    for label in labels_to_remove:
        updated_labels[updated_labels == label] = -1  # Mark as -1 (or background)
    
    return updated_labels

def create_mask(labels, mask_labels):
    """
    Create an overlay of the biggest contour over a black image of the same size as the input image.

    Args:
        image (np.ndarray): Input image (only used for size reference).
        labels (np.ndarray): Label array with superpixel labels.
        mask_labels (list or set): List of labels to include in the mask.
    
    Returns:
        mask
    """
    # Create the union mask
    union_mask = np.zeros(labels.shape, dtype=np.uint8)
    for sp in mask_labels:
        union_mask[labels == sp] = 255  # Set mask pixels to white (255)
    # Find the biggest contour
    contours, _ = cv2.findContours(union_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = simplify_contours(contours)
    mask = np.zeros(labels.shape, dtype=np.uint8)
    if contours:
        # Get the largest contour based on area
        max_contour = max(contours, key=cv2.contourArea)
        #print (f"Contour area: {cv2.contourArea(max_contour)}")
        '''# show contour
        cv2.drawContours(mask, [max_contour], 0, 255, -1)
        cv2.imshow("Contour", mask)
        cv2.waitKey(0)
        cv2.destroyAllWindows()'''
        processed_contour = process_contour(max_contour)
        # Get bounding box of the processed contour
        x, y, w, h = cv2.boundingRect(processed_contour)
        # Create a new black image of the size of the bounding box
        mask = np.zeros(union_mask.shape, dtype=np.uint8)
        # Draw the processed contour on the black image
        cv2.drawContours(mask, [processed_contour], 0, 255, -1)
        # Crop the union mask to the bounding box
        mask = mask[y:y+h, x:x+w]
    return mask

def slic_segmentation(image, num_superpixels=300, merge_threshold=20, slic_type=cv2.ximgproc.SLIC, compactness=5):
    """
    Performs SLIC (Simple Linear Iterative Clustering) superpixel segmentation on the input image.

    Parameters:
    - image (numpy.ndarray): Input image in BGR format.
    - num_superpixels (int): Approximate number of superpixels to generate. Default is 300.
    - merge_threshold (float): Threshold for merging adjacent superpixels based on color similarity.
                               If <= 0, no merging is performed. Default is 20.
    - slic_type (int): Type of SLIC algorithm to use (e.g., cv2.ximgproc.SLIC, SLICO, or MSLIC). Default is cv2.ximgproc.SLIC.
    - compactness (float): Compactness factor for superpixel shape regularity. Higher values lead to more compact shapes. Default is 5.

    Returns:
    - merged_labels (numpy.ndarray): 2D array of final superpixel labels after merging, with each pixel assigned a superpixel ID.
    - merged_mask (numpy.ndarray): 2D binary mask with thick boundaries for the final merged superpixels.
    - merged_result (numpy.ndarray): Input image with the final merged superpixel boundaries highlighted in green.
    - cluster_info (dict): Information about the final clusters, including:
        - "avg_bgr": Dictionary mapping each superpixel ID to its average BGR color.
        - "size": Dictionary mapping each superpixel ID to its pixel count.
    """
    # --- SLIC Segmentation ---
    lab_image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    region_size = int(np.sqrt(image.size / num_superpixels))
    slic = cv2.ximgproc.createSuperpixelSLIC(lab_image, slic_type, region_size, compactness)
    slic.iterate(10)
    slic.enforceLabelConnectivity(10)
    labels = slic.getLabels()
    original_mask = slic.getLabelContourMask(True)
    
    if merge_threshold <= 0:
        avg_bgr = {label: np.mean(image[labels == label], axis=0) for label in np.unique(labels)}
        result = image.copy()
        result[original_mask == 255] = (0, 255, 0)
        return labels, original_mask, result, {"avg_bgr": avg_bgr, "size": dict(zip(*np.unique(labels, return_counts=True)))}

    # --- Superpixel Adjacency via Boundary Pixels ---
    height, width = labels.shape
    adjacent = defaultdict(set)
    
    # Horizontal boundaries
    for y in range(height):
        for x in range(width-1):
            left = labels[y, x]
            right = labels[y, x+1]
            if left != right:
                adjacent[left].add(right)
                adjacent[right].add(left)
    
    # Vertical boundaries
    for y in range(height-1):
        for x in range(width):
            top = labels[y, x]
            bottom = labels[y+1, x]
            if top != bottom:
                adjacent[top].add(bottom)
                adjacent[bottom].add(top)

    # --- Union-Find with Iterative Merging ---
    parent = list(range(np.max(labels) + 1))
    bgr_sum = {label: np.sum(image[labels == label], axis=0) for label in np.unique(labels)}
    cluster_size = {label: np.sum(labels == label) for label in np.unique(labels)}
    
    def find(u):
        while parent[u] != u:
            parent[u] = parent[parent[u]]
            u = parent[u]
        return u
    
    # Iterate until no more merges occur
    changed = True
    while changed:
        changed = False
        for label in list(adjacent.keys()):  # Iterate copy to avoid runtime errors
            root_label = find(label)
            current_adjacent = adjacent[root_label].copy()
            
            for neighbor in current_adjacent:
                root_neighbor = find(neighbor)
                if root_label == root_neighbor:
                    continue
                
                avg_label = bgr_sum[root_label] / cluster_size[root_label]
                avg_neighbor = bgr_sum[root_neighbor] / cluster_size[root_neighbor]
                
                if all(abs(avg_label - avg_neighbor) <= merge_threshold):
                    # Merge smaller into larger
                    if cluster_size[root_label] < cluster_size[root_neighbor]:
                        root_label, root_neighbor = root_neighbor, root_label
                    
                    parent[root_neighbor] = root_label
                    bgr_sum[root_label] += bgr_sum[root_neighbor]
                    cluster_size[root_label] += cluster_size[root_neighbor]
                    del bgr_sum[root_neighbor]
                    del cluster_size[root_neighbor]
                    
                    # Update adjacency: merge neighbor's adjacents into root
                    adjacent[root_label].update(adjacent[root_neighbor])
                    adjacent[root_label].discard(root_label)  # Remove self-reference
                    del adjacent[root_neighbor]
                    
                    changed = True

    # --- Final Labeling ---
    unique_roots = {find(label) for label in np.unique(labels)}
    root_to_id = {root: i for i, root in enumerate(unique_roots)}
    merged_labels = np.vectorize(lambda x: root_to_id[find(x)])(labels)
    
    # --- Create Thick Boundary Mask ---
    merged_mask = np.zeros_like(merged_labels, dtype=np.uint8)
    for label in np.unique(merged_labels):
        label_mask = (merged_labels == label).astype(np.uint8)
        contours, _ = cv2.findContours(label_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(merged_mask, contours, -1, 255, 1)
        
        merged_result = image.copy()
        merged_result[merged_mask == 255] = (0, 255, 0)
    
    # --- Cluster Info ---
    cluster_info = {
        "avg_bgr": {i: bgr_sum[root]/cluster_size[root] for i, root in enumerate(unique_roots)},
        "size": {i: cluster_size[root] for i, root in enumerate(unique_roots)}
    }
    
    '''# show results
    cv2.imshow("Original", image)
    cv2.imshow("SLIC Segmentation", merged_result)
    cv2.imshow("SLIC Mask", merged_mask)
    cv2.waitKey(0)
    cv2.destroyAllWindows()'''

    return merged_labels, merged_mask, merged_result, cluster_info

def histogram_based_refinement(image, initial_labels, pred_hist, tolerance=15):
    """
    Implements histogram-based superpixel selection
    
    Args:
        image: Input BGR image.
        labels: Superpixel label map from SLIC.
        pred_hist: Dictionary of predicted histograms {'blue': [...], 'green': [...], 'red': [...]}
        tolerance: Allowed overflow.
        
    Returns:
        Final segmentation mask (list of superpixel labels).
    """
    # Debug counters
    debug = {
        'checked': 0,
        'removed': 0,
        'underflow_rejected': 0,
    }

    # Assign border superpixels to background (label == -1)
    labels = remove_border_superpixels(initial_labels)
    #print("final number of labels: ", len(np.unique(labels)))
    # Initialize current histogram as sum of all superpixels
    current_histogram = {
        'blue': np.zeros(256, dtype=int),
        'green': np.zeros(256, dtype=int),
        'red': np.zeros(256, dtype=int)
    }
    valid_labels = [l for l in np.unique(labels) if l != -1]
    #print(f"Valid labels length: {len(valid_labels)}")
    for label_i in valid_labels:
        sp_hist = get_superpixel_histogram(image, initial_labels, label_i)
        current_histogram = add_histograms(current_histogram, sp_hist)
    #print(f"Initialized histogram")

    # Check constraints
    unsatisfied_up, unsatisfied_down = check_constraints(pred_hist, current_histogram, tolerance)
    #print(f"Unsatisfied up: {unsatisfied_up}, unsatisfied down: {unsatisfied_down}")
    #utils.plot_histograms(pred_hist, current_histogram)

    final_labels = []

    # --- Region Refining Loop ---
    while valid_labels:
        current_label = valid_labels.pop(0)
        #print(f"Current label: {current_label}")
        debug['checked'] += 1

        # Get superpixel properties
        sp_hist = get_superpixel_histogram(image, np.array(initial_labels), current_label)
        temp_hist = remove_histograms(current_histogram, sp_hist)
        #plot_histograms(pred_hist, temp_hist)
        
        # Check new constraints
        new_up, new_down = check_constraints(pred_hist, temp_hist, tolerance)
        #print(f"New up: {new_up}, new down: {new_down}")
        
        # Decision criteria
        if new_up < unsatisfied_up and new_down <= unsatisfied_down:
            current_histogram = temp_hist
            unsatisfied_up = new_up
            unsatisfied_down = new_down
            debug['removed'] += 1
            #print(f"Removed {current_label} | New Up: {new_up}, Down: {new_down}")
            
        else:
            #print("current label is valid")
            final_labels.append(current_label)
            #plot_histograms(pred_hist, current_histogram)
            #show_translucent_mask(image, initial_labels, final_labels)
            debug['underflow_rejected'] += 1
    #utils.plot_histograms(pred_hist, current_histogram)
    #print("Debug info:", debug)
    return final_labels


# --- Main Function ---
def segmentation (cropped_image, pred_hist, tolerance=10, output_folder=None):
    print("Image Shape: ", cropped_image.shape)

    # --- SLIC Segmentation ---
    slic_labels, slic_mask, slic_result, slic_cluster_info = slic_segmentation(cropped_image)
    #print("SLIC Segmentation Completed, total number of labels: ", len(np.unique(slic_labels)))
    '''# show results
    cv2.imshow("Original", cropped_image)
    cv2.imshow("SLIC Segmentation", slic_result)
    cv2.imshow("SLIC Mask", slic_mask)
    cv2.waitKey(0)
    cv2.destroyAllWindows()'''

    # --- Region Refinement ---
    final_labels = histogram_based_refinement(cropped_image, slic_labels, pred_hist, tolerance=tolerance)
    #print("Region Refinement Completed, final number of labels: ", len(np.unique(final_labels)))
    # show the final labels
    #utils.show_translucent_mask(cropped_image, slic_labels, final_labels)

    # --- Final Visualization ---
    mask = create_mask(slic_labels, final_labels)

    print("Final Mask Shape: ", mask.shape)

    if output_folder is not None:
        # Save the final mask
        cv2.imwrite(f"{output_folder}/final_mask.png", mask)

    return mask
