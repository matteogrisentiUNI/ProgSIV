import numpy as np
import cv2
import matplotlib.pyplot as plt
import random

def adaptive_region_growing(image, threshold=50):
    """
    Adaptive region-growing segmentation algorithm.

    Parameters:
        image (ndarray): Grayscale input image (2D array).
        threshold (int): Intensity difference threshold for splitting clusters.

    Returns:
        segmented_image (ndarray): Image with regions colored by cluster.
        num_clusters (int): Total number of detected clusters.
    """
    # Ensure input is grayscale
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Preprocess image
    image = cv2.GaussianBlur(image, (15, 15), 0)  # Apply Gaussian filter
    image = cv2.resize(image, (0, 0), fx=0.4, fy=0.4) # Downscale the image to 50% of its original size
    #image = cv2.convertScaleAbs(image, alpha=1.6, beta=0)  # Increase brightness
    #clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(18, 18))  # Increase contrast
    #image = clahe.apply(image)
    image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX).astype(np.float32)

    # save the image to file if in a non-interactive environment
    plt.figure(figsize=(10, 10))
    plt.imshow(image, cmap='gray')
    plt.axis('off')
    plt.title("Preprocessed Image")
    plt.savefig("Demo/preprocessed_image.png")
    plt.close()  # Ensure no interactive mode issues

    # Initialize variables
    height, width = image.shape
    visited = np.zeros((height, width), dtype=bool)  # Tracks visited pixels
    clusters = np.zeros((height, width), dtype=int)  # Cluster labels
    cluster_means = []  # Stores mean intensity of each cluster
    current_cluster_id = 0

    # Helper function for growing a cluster
    def grow_cluster(seed_point, cluster_id):
        cluster_pixels = [seed_point]  # Pixels belonging to the cluster
        cluster_mean = image[seed_point]  # Initial mean is the seed's intensity
        cluster_sum = float(cluster_mean)  # Use float to avoid overflow
        cluster_size = 1
        queue = [seed_point]  # Queue for region-growing

        while queue:
            x, y = queue.pop(0)

            # Check 8-connected neighbors
            for dx, dy in [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]:
                nx, ny = x + dx, y + dy
                if 0 <= nx < height and 0 <= ny < width and not visited[nx, ny]:
                    intensity_diff = abs(image[nx, ny] - cluster_mean)

                    # Add pixel to cluster if within threshold
                    if intensity_diff <= threshold:
                        visited[nx, ny] = True
                        clusters[nx, ny] = cluster_id
                        queue.append((nx, ny))
                        cluster_pixels.append((nx, ny))
                        cluster_sum += image[nx, ny]
                        cluster_size += 1
                        cluster_mean = cluster_sum / cluster_size

        # Save the mean intensity of the cluster
        cluster_means.append(cluster_size)

    # Main region-growing loop
    for i in range(height):
        for j in range(width):
            if not visited[i, j]:
                current_cluster_id += 1  # New cluster
                visited[i, j] = True
                clusters[i, j] = current_cluster_id
                grow_cluster((i, j), current_cluster_id)

    # Convert cluster_means to numpy array
    cluster_means = np.array(cluster_means)

    # Sort and find the mean size of the 15 largest clusters
    if len(cluster_means) >= 5:
        largest_clusters = np.sort(cluster_means)[-5:-1]
        mean_of_largest = np.mean(largest_clusters)
    else:
        mean_of_largest = np.mean(cluster_means)

    # Merge clusters smaller than 70% of the mean size of the largest clusters
    for i in range(height):
        for j in range(width):
            cluster_id = clusters[i, j]
            if cluster_id > 0 and cluster_means[cluster_id - 1] < 0.5 * mean_of_largest:
                # Find the largest neighboring cluster to merge with
                largest_neighbor = cluster_id
                largest_size = cluster_means[cluster_id - 1]

                for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    ni, nj = i + dx, j + dy
                    if 0 <= ni < height and 0 <= nj < width:
                        neighbor_id = clusters[ni, nj]
                        if neighbor_id > 0 and cluster_means[neighbor_id - 1] > largest_size:
                            largest_neighbor = neighbor_id
                            largest_size = cluster_means[neighbor_id - 1]

                clusters[i, j] = largest_neighbor

    # Create a segmented image with random colors
    segmented_image = np.zeros((height, width, 3), dtype=np.uint8)
    random.seed(42)  # Fix seed for reproducibility
    cluster_colors = {
        cluster_id: [random.randint(0, 255) for _ in range(3)]
        for cluster_id in range(1, current_cluster_id + 1)
    }

    for i in range(height):
        for j in range(width):
            if clusters[i, j] > 0:
                segmented_image[i, j] = cluster_colors[clusters[i, j]]

    # Save the image to file if in a non-interactive environment
    plt.figure(figsize=(10, 10))
    plt.imshow(segmented_image)
    plt.axis('off')
    plt.title(f"Segmented Image with {current_cluster_id} Clusters")
    plt.savefig("Demo/segmented_output.png")
    plt.close()  # Ensure no interactive mode issues

    return segmented_image, current_cluster_id


    
# Example usage
if __name__ == "__main__":
    from skimage.io import imread

    # Load a grayscale image
    image = imread("Demo/Images/PersonNBG.png", as_gray=True)
    # convert image to 8-bit grayscale
    image = (image * 255).astype(np.uint8)  # Normalize to 8-bit if needed

    # Perform adaptive region growing
    segmented_image, num_clusters = adaptive_region_growing(image, threshold=20)

    print(f"Number of clusters detected: {num_clusters}")
