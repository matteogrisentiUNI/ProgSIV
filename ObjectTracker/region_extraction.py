import cv2
import numpy as np
from collections import defaultdict
from collections import defaultdict
from matplotlib import pyplot as plt
from ObjectTracker import contours as cs

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

def plot_histograms(hist1, hist2):
    """
    Plot two histograms on the same graph.

    Args:
        hist1: A dictionary containing histograms for 'blue', 'green', and 'red' channels (line graph).
        hist2: A dictionary containing histograms for 'blue', 'green', and 'red' channels (bar chart).
    """
    # Create a black image to display the plot
    img = np.zeros((400, 512, 3), dtype=np.uint8)

    # Create a Matplotlib figure
    plt.figure(figsize=(10, 6))

    # Loop through each color channel
    for color in ['blue', 'green', 'red']:
        # Get the histogram data for the current color
        hist_values1 = hist1[color]
        hist_values2 = hist2[color]

        # Generate x-axis values (0-255)
        x_values = np.arange(256)

        # Plot the first histogram as a line graph
        plt.plot(x_values, hist_values1, color=color, label=f'{color} (line)')

        # Plot the second histogram as a bar chart
        plt.bar(x_values, hist_values2, color=color, alpha=0.5, label=f'{color} (bar)')

    # Add labels, legend, and title
    plt.xlabel('Pixel Value')
    plt.ylabel('Number of Pixels')
    plt.title('Histograms for RGB Channels')
    plt.legend()

    # Save the plot as an image to show in OpenCV
    plt.savefig('histogram_plot.png')
    plt.close()

    # Load the saved plot as an image
    plot_img = cv2.imread('histogram_plot.png')

    # Resize for better display
    plot_img = cv2.resize(plot_img, (800, 600))

    # Display the image using OpenCV
    cv2.imshow('Histograms', plot_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

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

def visualize_superpixel_histogram(image, labels, superpixel_id, bins=256):
    """
    Calculate and visualize histogram for a specific superpixel using OpenCV.
    The image and histogram are displayed without deformation, with empty spaces filled with black.
    
    Args:
        image: Input image (grayscale or color).
        labels: Superpixel label map from SLIC.
        superpixel_id: ID of the superpixel to analyze.
        bins: Number of histogram bins (default 256).
    """
    # Create mask for the specific superpixel
    mask = (labels == superpixel_id).astype(np.uint8)
    
    # Create visualization of the superpixel mask
    mask_vis = image.copy()
    mask_vis[mask == 1] = 0  # Set superpixel pixels to black

    # Dimensions for the histogram image
    hist_height = 400
    hist_width = 512
    hist_image = np.zeros((hist_height, hist_width, 3), np.uint8)

    if len(image.shape) == 2:  # Grayscale
        pixels = image[mask == 1]
        hist = cv2.calcHist([pixels], [0], None, [bins], [0, 256])

        # Normalize histogram for visualization
        cv2.normalize(hist, hist, 0, hist_height, cv2.NORM_MINMAX)

        # Draw histogram
        for i in range(bins - 1):
            cv2.line(hist_image, 
                     (int(i * hist_width / bins), hist_height - int(hist[i])),
                     (int((i + 1) * hist_width / bins), hist_height - int(hist[i + 1])),
                     (255, 255, 255), 2)

    else:  # Color image
        colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]  # BGR format
        hist = {}

        for i, color in enumerate(['blue', 'green', 'red']):
            pixels = image[:, :, i][mask == 1]
            channel_hist = cv2.calcHist([pixels], [0], None, [bins], [0, 256]).flatten()
            hist[color] = channel_hist

            # Normalize histogram for visualization
            cv2.normalize(channel_hist, channel_hist, 0, hist_height, cv2.NORM_MINMAX)

            # Draw histogram
            for j in range(bins - 1):
                cv2.line(hist_image,
                         (int(j * hist_width / bins), hist_height - int(channel_hist[j])),
                         (int((j + 1) * hist_width / bins), hist_height - int(channel_hist[j + 1])),
                         colors[i], 2)

    # Add padding to maintain aspect ratio
    mask_height, mask_width = mask_vis.shape[:2]
    combined_width = hist_width + mask_width
    combined_height = max(hist_height, mask_height)

    # Create a black canvas for combined image
    combined = np.zeros((combined_height, combined_width, 3), dtype=np.uint8)

    # Place mask visualization
    combined[:mask_height, :mask_width, :] = cv2.cvtColor(mask_vis, cv2.COLOR_GRAY2BGR) if len(image.shape) == 2 else mask_vis

    # Place histogram
    combined[:hist_height, mask_width:mask_width + hist_width, :] = hist_image

    # Show the combined image
    cv2.imshow('Superpixel Analysis', combined)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return hist, mask_vis

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

def show_translucent_mask(image, labels, mask_labels):
    """
    Compose a mask from the union of superpixels (mask_labels) and overlay it translucently on the image.
    """
    import cv2
    import numpy as np
    union_mask = np.zeros(labels.shape, dtype=np.uint8)
    for sp in mask_labels:
        union_mask[labels == sp] = 255
    # Create a BGR version of the union mask
    union_mask_bgr = cv2.cvtColor(union_mask, cv2.COLOR_GRAY2BGR)
    # Overlay: 70% original image, 30% mask (here mask is white)
    overlay = cv2.addWeighted(image, 0.7, union_mask_bgr, 0.3, 0)
    cv2.imshow("Growing - Current Segmentation", overlay)
    cv2.waitKey(1)  # short delay to update window

def create_overlay(image, labels, mask_labels):
    """
    Create an overlay of the biggest contour over a black image of the same size as the input image.

    Args:
        image (np.ndarray): Input image (only used for size reference).
        labels (np.ndarray): Label array with superpixel labels.
        mask_labels (list or set): List of labels to include in the mask.
    
    Returns:
        np.ndarray: Black image with the biggest contour overlay.
    """
    # Create a black image of the same size as the input image
    black_img = np.zeros_like(image, dtype=np.uint8)
    
    # Create the union mask
    union_mask = np.zeros(labels.shape, dtype=np.uint8)
    for sp in mask_labels:
        union_mask[labels == sp] = 255  # Set mask pixels to white (255)

    # Find the biggest contour
    contours, _ = cv2.findContours(union_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = cs.simplify_contours(contours)
    if contours:
        # Get the largest contour based on area
        max_contour = max(contours, key=cv2.contourArea)
        processed_contour = cs.process_contour(max_contour)
        
        # Draw the largest contour on the black image
        cv2.drawContours(black_img, [processed_contour], -1, (255, 255, 255), thickness=1)
    
    #final_image = cs.refine_contour(black_img, 2)

    return black_img

def find_centroid(mask):
    """
    Finds the centroid inside a given mask.
    
    Args:
        mask (numpy.ndarray): The binary mask (same size as the image).
    
    Returns:
        entroid
    """
    # Calculate the centroid
    moments = cv2.moments(mask)
    if moments["m00"] != 0:
        cx = int(moments["m10"] / moments["m00"])
        cy = int(moments["m01"] / moments["m00"])
    else:
        raise ValueError("The mask is empty or has no area.")
    centroid = (cx, cy)

    # Check if the centroid is inside the mask
    if mask[cy, cx] == 0:
        centroidBackup = centroid
        #find the nearest point inside the mask
        while mask[cy, cx] == 0:
            if cy > 0:
                cy -= 1
            elif cx > 0:
                cx -= 1
            else:
                break
        centroidBackup = (cx, cy)
        # get the vector from the original centroid to the new centroid
        vector = (cx - centroidBackup[0], cy - centroidBackup[1])
        # move the centroid of 1. times the vector
        centroid = (int(centroidBackup[0] + vector[0] * 1.5), int(centroidBackup[1] + vector[1] * 1.5)) 
        # print(f"Centroid moved from {centroidBackup} to {centroid}")
        
    return centroid

def slic_segmentation(image, num_superpixels=300, merge_threshold=20, slic_type=cv2.ximgproc.SLIC, compactness=5):
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
    
    return merged_labels, merged_mask, merged_result, cluster_info

def histogram_based_refinement(image, initial_labels, pred_hist, tolerance=10):
    """
    Implements histogram-based region growing from centroid with shape priors.
    
    Args:
        image: Input BGR image.
        labels: Superpixel label map from SLIC.
        pred_hist: Dictionary of predicted histograms {'blue': [...], 'green': [...], 'red': [...]}
        shape_info: Dictionary of shape properties.
        centroid: (y, x) tuple of object center.
        tolerance: Allowed overflow percentage (0-1).
        
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
    #labels = label_isolated_superpixels(filtered_labels)
    # Initialize current histogram as sum of all superpixels
    current_histogram = {
        'blue': np.zeros(256, dtype=int),
        'green': np.zeros(256, dtype=int),
        'red': np.zeros(256, dtype=int)
    }
    valid_labels = [l for l in np.unique(labels) if l != -1]
    for label_i in valid_labels:
        sp_hist = get_superpixel_histogram(image, initial_labels, label_i)
        current_histogram = add_histograms(current_histogram, sp_hist)
    print(f"Initialized histogram")

    # Check constraints
    unsatisfied_up, unsatisfied_down = check_constraints(pred_hist, current_histogram, tolerance)
    print(f"Unsatisfied up: {unsatisfied_up}, unsatisfied down: {unsatisfied_down}")
    #plot_histograms(pred_hist, current_histogram)

    final_labels = []

    # --- Region Refining Loop ---
    while valid_labels and unsatisfied_up>100 and unsatisfied_down<200:
        current_label = valid_labels.pop(0)
        print(f"Current label: {current_label}")
        debug['checked'] += 1

        # Get superpixel properties
        sp_hist = get_superpixel_histogram(image, np.array(initial_labels), current_label)
        temp_hist = remove_histograms(current_histogram, sp_hist)
        #plot_histograms(pred_hist, temp_hist)
        
        # Check new constraints
        new_up, new_down = check_constraints(pred_hist, temp_hist, tolerance)
        print(f"New up: {new_up}, new down: {new_down}")
        
        # Decision criteria
        if new_up < unsatisfied_up and new_down <= (unsatisfied_down):
            current_histogram = temp_hist
            unsatisfied_up = new_up
            unsatisfied_down = new_down
            debug['removed'] += 1
            print(f"Removed {current_label} | New Up: {new_up}, Down: {new_down}")
            
        else:
            print("current label is valid")
            final_labels.append(current_label)
            #plot_histograms(pred_hist, current_histogram)
            #show_translucent_mask(image, initial_labels, final_labels)
            debug['underflow_rejected'] += 1

    # --- Final Visualization ---
    mask = create_overlay(image, initial_labels, final_labels)
    plot_histograms(pred_hist, current_histogram)
    cv2.imshow("Final Segmentation", mask)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    print("Debug info:", debug)
    return valid_labels
