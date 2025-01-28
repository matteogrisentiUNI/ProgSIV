import cv2
import numpy as np
import random

def apply_custom_threshold(channel):
    """
    Apply custom thresholding to each channel.
    Values from 0 to 75 are set to 0, from 76 to 175 are set to 125, and from 176 to 255 are set to 255.
    """
    thresholded = np.zeros_like(channel)
    thresholded[(channel >= 0) & (channel <= 50)] = 0
    thresholded[(channel > 50) & (channel <= 130)] = 80
    thresholded[(channel > 130) & (channel <= 200)] = 165
    thresholded[(channel > 200) & (channel <= 255)] = 255
    return thresholded

def adaptive_region_growing(image_channel, alpha_channel):
    """
    Perform adaptive region growing on a single-channel image with transparent background.

    Parameters:
        image_channel (np.ndarray): Single-channel input image as a NumPy array.
        alpha_channel (np.ndarray): Alpha channel of the image for transparency.

    Returns:
        np.ndarray: Clustered image with each region assigned a unique cluster ID.
        int: Number of clusters detected in the image.
    """
    # Exclude transparent regions
    non_transparent_mask = alpha_channel > 0
    image_channel = np.where(non_transparent_mask, image_channel, 0)

    # Initialize variables
    rows, cols = image_channel.shape
    clusters = np.zeros((rows, cols), dtype=np.int32)
    cluster_id = 1
    stack = []
    
    # A dictionary to track cluster sizes
    cluster_sizes = {}

    # Define 8-connectivity offsets
    offsets = [
        (-1, -1), (-1, 0), (-1, 1),
        (0, -1), (0, 1),
        (1, -1), (1, 0), (1, 1)
    ]

    # Region growing: Assign initial clusters based on intensity similarity
    for i in range(rows):
        for j in range(cols):
            if clusters[i, j] == 0 and non_transparent_mask[i, j]:  # Start a new cluster for non-transparent pixels
                stack.append((i, j))
                clusters[i, j] = cluster_id
                cluster_sizes[cluster_id] = 1  # Initialize size of the new cluster

                while stack:
                    current = stack.pop()
                    for offset in offsets:
                        x, y = current[0] + offset[0], current[1] + offset[1]
                        if (
                            0 <= x < rows and 0 <= y < cols and
                            clusters[x, y] == 0 and
                            non_transparent_mask[x, y] and
                            abs(int(image_channel[x, y]) - int(image_channel[current])) <= 50  # Tolerance for merging
                        ):
                            clusters[x, y] = cluster_id
                            cluster_sizes[cluster_id] += 1  # Increase cluster size
                            stack.append((x, y))

                cluster_id += 1

    # Merge the smallest clusters until we have 10 clusters
    num_clusters = 14
    while len(cluster_sizes) > num_clusters:  # Stop when we have 10 or fewer clusters
        # Find the smallest cluster by size
        sorted_cluster_sizes = sorted(cluster_sizes.items(), key=lambda x: x[1])
        smallest_cluster_id, smallest_cluster_size = sorted_cluster_sizes[0]
        smallest_cluster_mask = clusters == smallest_cluster_id

        # Find neighboring clusters for the smallest cluster
        # Take a random point from the smallest cluster
        x, y = np.where(smallest_cluster_mask)
        x, y = x[0], y[0]  # Take the first point in the smallest cluster

        neighboring_cluster_id = -1  # Initialize as invalid

        # Go right until we find a different cluster
        while y < cols and clusters[x, y] == smallest_cluster_id:
            y += 1
        if y < cols and non_transparent_mask[x, y]:  # If found a neighboring cluster
            neighboring_cluster_id = clusters[x, y]
        
        # If not found, go left from the initial point until we find a different cluster
        elif y == cols or not non_transparent_mask[x, y]:
            y = y - 1
            while y >= 0 and clusters[x, y] == smallest_cluster_id:
                y -= 1
            if y >= 0 and non_transparent_mask[x, y]:  # If found a neighboring cluster
                neighboring_cluster_id = clusters[x, y]

        # If not found, go down from the initial point until we find a different cluster
        elif neighboring_cluster_id == -1:  # Only check if still no neighbor found
            y = y + 1
            while x < rows and clusters[x, y] == smallest_cluster_id:
                x += 1
            if x < rows and non_transparent_mask[x, y]:  # If found a neighboring cluster
                neighboring_cluster_id = clusters[x, y]

        # If not found, go up from the initial point until we find a different cluster
        elif neighboring_cluster_id == -1:  # Only check if still no neighbor found
            x = x - 1
            while x >= 0 and clusters[x, y] == smallest_cluster_id:
                x -= 1
            if x >= 0 and non_transparent_mask[x, y]:  # If found a neighboring cluster
                neighboring_cluster_id = clusters[x, y]

        # If we still have not found a neighboring cluster, skip to next cluster merge attempt
        if neighboring_cluster_id == -1:
            cluster_sizes[smallest_cluster_id] = 1000000000  # Invalidate the smallest cluster
            num_clusters += 1  # Skip to the next cluster merge attempt
            continue

        # Now assign all pixels from the smallest cluster to the neighboring cluster
        clusters[smallest_cluster_mask] = neighboring_cluster_id
        cluster_sizes[neighboring_cluster_id] += cluster_sizes[smallest_cluster_id]  # Update size
        del cluster_sizes[smallest_cluster_id]  # Remove the merged cluster

    '''# show one at a time a colored mask corresponding to each cluster
    for cluster_id in cluster_sizes:
        cluster_mask = clusters == cluster_id
        cluster_color = [random.randint(0, 255) for _ in range(3)]
        cluster_image = np.zeros((rows, cols, 3), dtype=np.uint8)
        cluster_image[cluster_mask] = cluster_color
        cv2.imshow(f"Cluster {cluster_id}", cluster_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    # compose the final image with the remaining clusters
    final_image = np.zeros((rows, cols, 3), dtype=np.uint8)
    for cluster_id in cluster_sizes:
        cluster_mask = clusters == cluster_id
        cluster_color = [random.randint(0, 255) for _ in range(3)]
        final_image[cluster_mask] = cluster_color'''
    
    # Initialize the final image with 4 channels (RGBA)
    final_image = np.zeros((rows, cols, 4), dtype=np.uint8)

    # Iterate over all cluster IDs (from 1 to max cluster_id)
    for cluster_id in cluster_sizes:
        # Create a mask for the current cluster
        cluster_mask = clusters == cluster_id

        # Assign each cluster it's average color

        
        # Set the RGB channels to the custom color
        final_image[cluster_mask, 0] = cluster_color[0]  # Red
        final_image[cluster_mask, 1] = cluster_color[1]  # Green
        final_image[cluster_mask, 2] = cluster_color[2]  # Blue

        # Set the alpha channel to the original alpha channel value
        final_image[cluster_mask, 3] = alpha_channel[cluster_mask]  # Using the original alpha channel for transparency

    '''# Now `final_image` contains the clustered image with transparency applied
    cv2.imshow("Final Image", final_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()'''

    return clusters, len(cluster_sizes), final_image

def preprocess_and_segment(image_path, 
                            best_channel = None, 
                            blur_ksize=7, 
                            canny_low=50, 
                            canny_high=150):
    """
    Preprocess and segment a PNG image with Canny edge detection and adaptive region growing.

    Parameters:
        image_path (str): Path to the PNG image with a transparent background.
        threshold (int): Intensity difference threshold for region growing (default: 50).
        blur_ksize (int): Kernel size for Gaussian blur (default: 5).
        canny_low (int): Lower threshold for Canny edge detection (default: 50).
        canny_high (int): Upper threshold for Canny edge detection (default: 150).
        alpha (float): Transparency factor for overlay masks (0 = transparent, 1 = opaque) (default: 0.5).

    Returns:
        np.ndarray: Processed image with translucent masks over regions and no background.
    """
    # Load the image without alpha channel
    image = cv2.imread(image_path, cv2.IMREAD_COLOR_BGR)

# Preprocess:
    # Apply histogram equalization on the intensity (grayscale) channel
    gray_equalized = cv2.equalizeHist(image[:, :, 2])
    image[:, :, 2] = gray_equalized 
    # Increase contrast
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    image[:, :, 2] = clahe.apply(image[:, :, 2])
    #Apply Gaussian Blur
    blurred = cv2.GaussianBlur(image, (blur_ksize, blur_ksize), 0)
    
    '''# Show the thresholded images of the channels with cv2.imshow
    cv2.imshow("TBlue Channel", thresholded_blue)
    cv2.imshow("TGreen Channel", thresholded_green)
    cv2.imshow("TRed Channel", thresholded_red)
    cv2.imshow("TIntensity Channel", thresholded_intensity)
    cv2.waitKey(0)
    cv2.destroyAllWindows()'''
    best_thresholded = None
    if best_channel is None:
        # Extract the blue, green and red channels and intensity channel as separate images, create an image with all the cannels to 0 exept the current color channel
        blue_channel = blurred[:, :, 0]
        green_channel = blurred[:, :, 1]
        red_channel = blurred[:, :, 2]
        gray = cv2.cvtColor(blurred, cv2.COLOR_BGR2GRAY)
        # Apply custom thresholding to each channel
        thresholded_red = apply_custom_threshold(red_channel)
        thresholded_green = apply_custom_threshold(green_channel)
        thresholded_blue = apply_custom_threshold(blue_channel)
        thresholded_intensity = apply_custom_threshold(gray)
        thresholded_channels = [thresholded_red, thresholded_green, thresholded_blue, thresholded_intensity]
        # Apply Canny edge detection for each channel
        edges_r = cv2.Canny(thresholded_red, canny_low, canny_high)  # Red channel
        edges_g = cv2.Canny(thresholded_green, canny_low, canny_high)  # Green channel
        edges_b = cv2.Canny(thresholded_blue, canny_low, canny_high)  # Blue channel
        edges_intensity = cv2.Canny(thresholded_intensity, canny_low, canny_high)  # Intensity channel
        edges = [edges_r, edges_g, edges_b, edges_intensity]
        best_density = np.sum(edges_r) / edges_r.size
        for i, edgei in enumerate(edges):
            density = np.sum(edgei) / edgei.size  # Calculate edge density as the ratio of edge pixels to total pixels
            if density < best_density:  # Looking for the smallest density (less clutter)
                best_density = density
                best_edge_index = i  # Update index of the best edge
        best_channel = best_edge_index
        # Get the best edge and best thresholded image based on density
        best_thresholded = thresholded_channels[best_channel]
        #print("Best edge index: ", best_edge_index)
    else: 
        if best_channel < 4:
            best_thresholded = apply_custom_threshold(blurred[:, :, best_channel])
        elif best_channel == 4:
            best_thresholded = apply_custom_threshold(cv2.cvtColor(blurred, cv2.COLOR_BGR2GRAY))
    print("Using best channel:", best_channel)

    # Perform region growing
    best_thresholded = np.array(best_thresholded, dtype=np.uint8)
    print("pre-processing completed")
    clusters, num_clusters, image = adaptive_region_growing(best_thresholded)
    
    return image, num_clusters, clusters, best_channel