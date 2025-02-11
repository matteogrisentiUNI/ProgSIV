import cv2
import numpy as np
from collections import defaultdict, deque
from scipy.ndimage import gaussian_filter1d
from LOBES import utils

# --- Contour Operations ---
def simplify_contours(contours, epsilon_factor=0.001):
    simplified = []
    for cnt in contours:
        peri = cv2.arcLength(cnt, True)
        epsilon = epsilon_factor * peri
        approx = cv2.approxPolyDP(cnt, epsilon, True)
        simplified.append(approx)
    return simplified

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

def update_histogram(hist1, hist2, weight=0.7):
    """
    Creates an updated histogram by combining two input histograms using a weighted sum.
    
    For each bin in each channel, the updated value is computed as:
        weight * hist1 + (1 - weight) * hist2

    Parameters:
        hist1 (dict): First histogram with keys 'blue', 'green', 'red' (numpy arrays).
        hist2 (dict): Second histogram with the same structure as hist1.
        weight (float): Weight factor for hist1 (should be between 0 and 1). 
                        The contribution of hist2 will be (1 - weight).

    Returns:
        dict: A new histogram with the same keys, where each bin is the weighted sum of the two inputs.
    """
    if not (0 <= weight <= 1):
        raise ValueError("Weight must be between 0 and 1.")

    updated_hist = {ch: np.zeros(256, dtype=int) for ch in ['blue', 'green', 'red']}
    for channel in ['blue', 'green', 'red']:
        for i in range(256):
            updated_hist[channel][i] = weight * hist1[channel][i] + (1 - weight) * hist2[channel][i]

    #utils.plot_histograms(hist1, updated_hist, 800, 600)

    return updated_hist

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

def check_constraints(h1, h2):
    """
    Computes a similarity measure between two histograms (with 'blue', 'green', 'red' channels)
    using the normalized histogram intersection method.
    
    The measure returns 1 if the histograms are identical and 0 if they are completely disjoint.
    
    Parameters:
        h1 (dict): Reference histogram with keys 'blue', 'green', 'red'. Each value should be a numpy array.
        h2 (dict): Histogram to compare, with the same structure as h1.
        tolerance (int, optional): Unused in this implementation.
    
    Returns:
        float: A similarity score in the range [0, 1].
    """
    total_min = 0.0
    total_max = 0.0
    for channel in ['blue', 'green', 'red']:
        total_min += np.sum(np.minimum(h1[channel], h2[channel]))
        total_max += np.sum(np.maximum(h1[channel], h2[channel]))
    # Prevent division by zero: if both histograms are empty, consider them identical.
    #if total_max != 0 and total_min != 0:
        #print(total_min/total_max)
    return 1.0 if total_max == 0 else total_min / total_max


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

def histogram_based_refinement(image, initial_labels, pred_hist):
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
    old_similarity_measure = check_constraints(pred_hist, current_histogram)
    #print("Initial Similarity Measure: ", old_similarity_measure)
    #utils.plot_histograms(pred_hist, current_histogram, width=800, height=600)

    final_labels = []
    # Use deque for efficient pop from left.
    valid_queue = deque(valid_labels)

    while valid_queue:
        current_label = valid_queue.popleft()
        sp_hist = sp_hist_dict[current_label]
        # Simulate removal by subtracting this superpixel’s histogram.
        temp_hist = remove_histograms(current_histogram, sp_hist)
        temp_similarity_measure = check_constraints(pred_hist, temp_hist)
        
        # Remove superpixel if it improves (reduces) the "overflow" without worsening underflow.
        if temp_similarity_measure > (old_similarity_measure+0.05):
            current_histogram = temp_hist
            old_similarity_measure = temp_similarity_measure
        else:
            final_labels.append(current_label)
    #print("Final Similarity Measure: ", old_similarity_measure)
    #utils.plot_histograms(pred_hist, current_histogram, width=800, height=600)

    return final_labels, current_histogram

# --- Main Function ---
def segmentation (cropped_image, pred_hist):
    #print("Image Shape: ", cropped_image.shape)

    # --- SLIC Segmentation ---
    slic_labels = slic_segmentation(cropped_image)
    #utils.visualize_superpixels_with_random_colors(cropped_image, slic_labels)

    # --- Region Refinement ---
    final_labels, next_histogram = histogram_based_refinement(cropped_image, slic_labels, pred_hist)
    #utils.plot_histograms(pred_hist, next_histogram, width=800, height=600)
    final_histogram = update_histogram(pred_hist, next_histogram)
    #utils.show_translucent_mask(cropped_image, slic_labels, final_labels)
 
    # --- Final Visualization ---
    final_mask = create_mask(slic_labels, final_labels)

    return final_mask, final_histogram
