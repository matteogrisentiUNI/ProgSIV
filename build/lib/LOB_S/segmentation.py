import cv2
import numpy as np
from collections import defaultdict, deque
from scipy.ndimage import gaussian_filter1d
from LOB_S import utils

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
                        # Remove points clockwise between `i` and `index`
                        if index > i:
                            contour = np.delete(contour, range(i + 1, index), axis=0)
                        else:
                            contour = np.delete(contour, range(i + 1, len(contour)), axis=0)  # From `i+1` to the end (including n)
                            contour = np.delete(contour, range(0, index), axis=0)            # From start to `index`
                    else:
                        #print(f"Connecting Point {i} and Point {index}, removing intermediate points anticlockwise.")
                        # Remove points anticlockwise between `i` and `index`
                        if index < i:
                            contour = np.delete(contour, range(index + 1, i), axis=0)
                        else:
                            contour = np.delete(contour, range(index + 1, len(contour)), axis=0)  # From `index+1` to the end (including n)
                            contour = np.delete(contour, range(0, i), axis=0)                    # From start to `i`

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

def update_histogram(current_histogram, new_histogram, weight=0.7):
    """
    Updates a histogram represented as a dictionary ('blue', 'green', 'red').

    Args:
        hist1: Current histogram (dictionary with 'blue', 'green', 'red' keys).
        hist2: Updated histogram (dictionary with 'blue', 'green', 'red' keys).
        weight: Weight for the updated histogram (default 0.5).

    Returns:
        A new histogram where each channel is a weighted average of the corresponding channels in hist1 and hist2.
    """
    updated_histogram = {}
    for channel in ['blue', 'green', 'red']:
        updated_histogram[channel] = weight * current_histogram[channel] + (1 - weight) * new_histogram[channel]
    unsatisfied_up, usatisfied_down = check_constraints(current_histogram, updated_histogram)
    if usatisfied_down+unsatisfied_up > 500:
        return current_histogram
    return updated_histogram

def get_superpixel_histogram(image, labels, superpixel_id, bins=256):
    """
    Compute histogram for a given superpixel using np.bincount.
    """
    mask = (labels == superpixel_id)
    hist_dict = {}
    if image.ndim == 2:  # Grayscale: treat as single-channel repeated across colors
        pixels = image[mask].ravel()
        hist = np.bincount(pixels, minlength=bins)
        for channel in ['blue', 'green', 'red']:
            hist_dict[channel] = hist.copy()
    else:
        for idx, channel in enumerate(['blue', 'green', 'red']):
            channel_pixels = image[:, :, idx][mask].ravel()
            hist_dict[channel] = np.bincount(channel_pixels, minlength=bins)
    return hist_dict

def check_constraints(hist_ref, hist_current, tolerance=10):
    """
    Vectorized comparison of two histogram dictionaries.
    Returns counts of bins where the current histogram is too high or too low.
    """
    unsatisfied_up = 0
    unsatisfied_down = 0
    for ch in ['blue', 'green', 'red']:
        diff = hist_current[ch] - hist_ref[ch]
        unsatisfied_up += np.count_nonzero(diff > tolerance)
        unsatisfied_down += np.count_nonzero(diff < -tolerance)
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
        processed_contour = process_contour(max_contour)
        # Create a new black image of the size of the bounding box
        mask = np.zeros(union_mask.shape, dtype=np.uint8)
        # Draw the processed contour on the black image
        cv2.drawContours(mask, [processed_contour], 0, 255, -1)
    return mask

def slic_segmentation(image, num_superpixels=250, merge_threshold=20,
                      slic_type=cv2.ximgproc.SLIC, compactness=5):
    """
    Performs SLIC superpixel segmentation with optional merging based on color similarity.
    (See original docstring for full details.)
    """
    # --- SLIC Segmentation ---
    lab_image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    region_size = int(np.sqrt(image.size / num_superpixels))
    slic = cv2.ximgproc.createSuperpixelSLIC(lab_image, slic_type, region_size, compactness)
    slic.iterate(10)
    slic.enforceLabelConnectivity(10)
    labels = slic.getLabels()
    original_mask = slic.getLabelContourMask(True)
    
    # If merging is not desired, compute cluster info and return early.
    if merge_threshold <= 0:
        unique, counts = np.unique(labels, return_counts=True)
        # Compute per-cluster BGR sums using np.bincount
        b_sum = np.bincount(labels.ravel(), weights=image[..., 0].ravel())
        g_sum = np.bincount(labels.ravel(), weights=image[..., 1].ravel())
        r_sum = np.bincount(labels.ravel(), weights=image[..., 2].ravel())
        avg_bgr = {i: np.array([b_sum[i], g_sum[i], r_sum[i]]) / counts[i]
                   for i in unique}
        result = image.copy()
        result[original_mask == 255] = (0, 255, 0)
        cluster_info = {"avg_bgr": avg_bgr, "size": dict(zip(unique, counts))}
        return labels, original_mask, result, cluster_info

    # --- Compute Adjacency Map via Vectorized Boundary Extraction ---
    height, width = labels.shape
    adjacent = defaultdict(set)
    
    # Horizontal neighbors: compare labels[:, :-1] with labels[:, 1:]
    left = labels[:, :-1]
    right = labels[:, 1:]
    mask = left != right
    if np.any(mask):
        a = left[mask]
        b = right[mask]
        # Always store the smaller label first to avoid duplicates.
        pairs = np.stack([np.minimum(a, b), np.maximum(a, b)], axis=-1)
        for p in np.unique(pairs, axis=0):
            adjacent[p[0]].add(p[1])
            adjacent[p[1]].add(p[0])
    
    # Vertical neighbors: compare labels[:-1, :] with labels[1:, :]
    top = labels[:-1, :]
    bottom = labels[1:, :]
    mask = top != bottom
    if np.any(mask):
        a = top[mask]
        b = bottom[mask]
        pairs = np.stack([np.minimum(a, b), np.maximum(a, b)], axis=-1)
        for p in np.unique(pairs, axis=0):
            adjacent[p[0]].add(p[1])
            adjacent[p[1]].add(p[0])
    
    # --- Precompute Cluster Statistics Using np.bincount ---
    num_labels = np.max(labels) + 1
    sizes = np.bincount(labels.ravel(), minlength=num_labels)
    b_sum = np.bincount(labels.ravel(), weights=image[..., 0].ravel(), minlength=num_labels)
    g_sum = np.bincount(labels.ravel(), weights=image[..., 1].ravel(), minlength=num_labels)
    r_sum = np.bincount(labels.ravel(), weights=image[..., 2].ravel(), minlength=num_labels)
    bgr_sum = {i: np.array([b_sum[i], g_sum[i], r_sum[i]])
               for i in range(num_labels)}
    cluster_size = {i: sizes[i] for i in range(num_labels)}
    
    # --- Union-Find for Iterative Merging ---
    parent = list(range(num_labels))
    
    def find(u):
        while parent[u] != u:
            parent[u] = parent[parent[u]]
            u = parent[u]
        return u
    
    changed = True
    while changed:
        changed = False
        # Work on a copy of keys to avoid modification issues during iteration.
        for label in list(adjacent.keys()):
            root_label = find(label)
            # Copy current neighbors to avoid concurrent modification.
            for neighbor in list(adjacent[root_label]):
                root_neighbor = find(neighbor)
                if root_label == root_neighbor:
                    continue
                # Compute average colors.
                avg_label = bgr_sum[root_label] / cluster_size[root_label]
                avg_neighbor = bgr_sum[root_neighbor] / cluster_size[root_neighbor]
                # If all channels are within the merge threshold...
                if np.all(np.abs(avg_label - avg_neighbor) <= merge_threshold):
                    # Merge the smaller cluster into the larger.
                    if cluster_size[root_label] < cluster_size[root_neighbor]:
                        root_label, root_neighbor = root_neighbor, root_label
                    parent[root_neighbor] = root_label
                    bgr_sum[root_label] += bgr_sum[root_neighbor]
                    cluster_size[root_label] += cluster_size[root_neighbor]
                    # Remove merged neighbor's data.
                    del bgr_sum[root_neighbor]
                    del cluster_size[root_neighbor]
                    # Merge adjacency: update neighbor sets.
                    adjacent[root_label].update(adjacent[root_neighbor])
                    adjacent[root_label].discard(root_label)
                    del adjacent[root_neighbor]
                    changed = True

    # --- Final Label Mapping ---
    # Build a mapping from original label to new label based on union-find results.
    mapping = np.empty(num_labels, dtype=np.int32)
    root_to_new = {}
    new_label = 0
    for i in range(num_labels):
        root = find(i)
        if root not in root_to_new:
            root_to_new[root] = new_label
            new_label += 1
        mapping[i] = root_to_new[root]
    merged_labels = mapping[labels]

    return merged_labels

def histogram_based_refinement(image, initial_labels, pred_hist, tolerance=10, debugPrint=False):
    """
    Refines superpixel segmentation based on histogram constraints.
    
    Args:
      image: Input BGR image.
      initial_labels: Superpixel label map from SLIC.
      pred_hist: Target histograms as a dict {'blue': [...], 'green': [...], 'red': [...]}.
      tolerance: Allowed per-bin deviation.
      debugPrint: If True, plot histograms.
      
    Returns:
      final_labels: List of superpixel IDs retained after refinement.
    """
    # Remove border superpixels (assume remove_border_superpixels returns a np.array)
    labels = np.array(remove_border_superpixels(initial_labels))
    
    # Get list of valid superpixel labels (ignore -1)
    valid_labels = [lbl for lbl in np.unique(labels) if lbl != -1]

    # Precompute each superpixel’s histogram once.
    sp_hist_dict = {lbl: get_superpixel_histogram(image, labels, lbl) for lbl in valid_labels}

    # Build the overall histogram by summing all superpixel histograms.
    current_histogram = {ch: np.zeros(256, dtype=int) for ch in ['blue', 'green', 'red']}
    for lbl in valid_labels:
        current_histogram = add_histograms(current_histogram, sp_hist_dict[lbl])

    # Compute initial constraint violations.
    unsatisfied_up, unsatisfied_down = check_constraints(pred_hist, current_histogram, tolerance)
    if debugPrint:
        utils.plot_histograms(pred_hist, current_histogram, width=800, height=600)

    final_labels = []
    # Use deque for efficient pop from left.
    valid_queue = deque(valid_labels)

    while valid_queue:
        current_label = valid_queue.popleft()
        sp_hist = sp_hist_dict[current_label]
        # Simulate removal by subtracting this superpixel’s histogram.
        temp_hist = remove_histograms(current_histogram, sp_hist)
        new_up, new_down = check_constraints(pred_hist, temp_hist, tolerance)
        
        # Remove superpixel if it improves (reduces) the "overflow" without worsening underflow.
        if new_up < unsatisfied_up and new_down <= unsatisfied_down:
            current_histogram = temp_hist
            unsatisfied_up, unsatisfied_down = new_up, new_down
        else:
            final_labels.append(current_label)
    
    return final_labels

# --- Main Function ---
def segmentation (cropped_image, pred_hist, tolerance=15, output_folder=None, debugPrint=False):
    #print("Image Shape: ", cropped_image.shape)

    # --- SLIC Segmentation ---
    slic_labels = slic_segmentation(cropped_image)

    # --- Region Refinement ---
    final_labels = histogram_based_refinement(cropped_image, slic_labels, pred_hist, tolerance=tolerance, debugPrint=debugPrint)
 
    # --- Final Visualization ---
    mask = create_mask(slic_labels, final_labels)

    #print("Final Mask Shape: ", mask.shape)

    return mask
