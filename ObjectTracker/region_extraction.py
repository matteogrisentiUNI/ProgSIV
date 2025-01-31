import cv2
import numpy as np
import matplotlib.pyplot as plt

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

def adaptive_region_growing(image):
    """
    Perform adaptive region growing on a single-channel image with transparent background.

    Parameters:
        image_channel (np.ndarray): Single-channel input image as a NumPy array.
        alpha_channel (np.ndarray): Alpha channel of the image for transparency.

    Returns:
        np.ndarray: Clustered image with each region assigned a unique cluster ID.
        int: Number of clusters detected in the image.
    """

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


from collections import deque

def histogram_based_region_growing(image, seed_point, threshold=97, connectivity=8):
    """
    Perform region growing based on histogram matching.
    
    Parameters:
    - image: Input image (grayscale or color).
    - seed_point: (x, y) coordinates of the seed pixel.
    - histogram: A dictionary of histograms for each color channel (keys: "blue", "green", "red") or a single histogram for grayscale images.
    - threshold: A threshold for histogram similarity to consider a pixel part of the region.
    - connectivity: 4 or 8 connectivity for neighborhood exploration.
    
    Returns:
    - mask: A binary mask where the region of interest has been grown.
    """
    
    height, width = image.shape[:2] # Get image dimensions
    
    mask = np.zeros((height, width), dtype=np.uint8) # Initialize the binary mask
    print("Initialized mask with dimensions:", mask.shape)

    clusters = [] # Initialize cluster list

    # Function to get neighbors based on connectivity
    def get_neighbors(y, x):
        neighbors = []
        if connectivity == 4:
            directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # 4-connected
        elif connectivity == 8:
            directions = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]  # 8-connected
        for dx, dy in directions:
            ny, nx = y + dy, x + dx
            if 0 <= ny < height and 0 <= nx < width:
                neighbors.append((ny, nx))
        return neighbors
    
    # Initialize the queue with the seed point (convert seed to (y, x) format)
    seed_x, seed_y = seed_point  # Convert seed to (y, x)
    inv_seed_point = seed_y, seed_x
    print("x: ",seed_x, "y: ", seed_y)
    queue = deque([(inv_seed_point)])  # Use (y, x) coordinates

    print(queue)
    mask[inv_seed_point] = 1  # Mark seed as part of the region
    cluster_size = 0 # Cluster size

    # Initialize a histogram for the seed point
    b, g, r = image[inv_seed_point]
    cluster_blue = b  # Initialize blue histogram
    cluster_green = g  # Initialize green histogram
    cluster_red = r  # Initialize red histogram

    print(f"Seed point initialized at ({seed_y}, {seed_x}) with initial mask set.")

    # Region growing
    while queue:
        y, x = queue.popleft()  # Pop (y, x) from the queue
        #print(f"Processing pixel at ({y}, {x})")

        # Get pixel value
        pixel_value = image[y, x] if len(image.shape) == 2 else image[y, x, :]
        #print(f"Pixel value: {pixel_value}")

        # Check histogram similarity
        b_value, g_value, r_value = pixel_value
        blue_difference = int(b_value) - int(cluster_blue)
        green_difference = int(g_value) - int(cluster_green)
        red_difference = int(r_value) - int(cluster_red)
        is_similar = abs(blue_difference) < threshold and abs(green_difference) < threshold and abs(red_difference) < threshold

        if is_similar:
            mask[y, x] = 1  # Mark as part of the region
            cluster_size += 1  # Update cluster size
            # Update cluster histogram
            cluster_blue = int(cluster_blue + (blue_difference / cluster_size))
            cluster_green = int(cluster_green + (green_difference / cluster_size))
            cluster_red = int(cluster_red + (red_difference / cluster_size))
            # Add neighbors to the queue if they are unvisited
            for ny, nx in get_neighbors(y, x):
                if mask[ny, nx] == 0:  # If not yet visited
                    queue.append((ny, nx))
                    mask[ny, nx] = 3  # Mark as "in queue"
        else:
            mask[y, x] = 2  # Mark as visited but NOT part of the region
            # print(f"Pixel at ({x}, {y}) is NOT similar to histogram. Skipping.")
            
    # After the region growing is done, add the final cluster mask to the clusters list
    clusters.append(mask)

    # Create masks for different values
    mask_0 = (mask == 0).astype(np.uint8)  # Pixels with value = 0
    mask_1 = (mask == 1).astype(np.uint8)  # Pixels with value = 1
    mask_2 = (mask == 2).astype(np.uint8)  # Pixels with value = 2
    mask_3 = (mask == 3).astype(np.uint8)  # Pixels with value = 3

    # Define colors for each mask (BGR format)
    color_0 = (180, 105, 255)  # Pink for mask_0
    color_1 = (255, 0, 0)      # Blue for mask_1
    color_2 = (0, 255, 0)      # Green for mask_2
    color_3 = (0, 0, 255)      # Red for mask_3

    # Create color overlays for each mask
    pink_mask = np.zeros_like(image, dtype=np.uint8)
    pink_mask[:, :] = color_0
    blue_mask = np.zeros_like(image, dtype=np.uint8)
    blue_mask[:, :] = color_1
    green_mask = np.zeros_like(image, dtype=np.uint8)
    green_mask[:, :] = color_2
    red_mask = np.zeros_like(image, dtype=np.uint8)
    red_mask[:, :] = color_3

    # Apply the individual masks to their respective colors
    pink_overlay = cv2.bitwise_and(pink_mask, pink_mask, mask=mask_0)
    blue_overlay = cv2.bitwise_and(blue_mask, blue_mask, mask=mask_1)
    green_overlay = cv2.bitwise_and(green_mask, green_mask, mask=mask_2)
    red_overlay = cv2.bitwise_and(red_mask, red_mask, mask=mask_3)

    # Combine all overlays into one
    combined_overlay = cv2.addWeighted(pink_overlay, 1, blue_overlay, 1, 0)
    combined_overlay = cv2.addWeighted(combined_overlay, 1, green_overlay, 1, 0)
    combined_overlay = cv2.addWeighted(combined_overlay, 1, red_overlay, 1, 0)

    # Blend the combined overlay with the original image
    alpha = 0.5  # Transparency factor
    final_image = cv2.addWeighted(image, 1 - alpha, green_overlay, alpha, 0)

    # Draw a red dot at the seed point location
    cv2.circle(final_image, (seed_point), 5, (0, 0, 255), -1)  # Red color in BGR, radius 5

    # Show the resulting image
    cv2.imshow("Bounded Image with Colored Masks Overlay", final_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return clusters
