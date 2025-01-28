import numpy as np
import cv2
import matplotlib.pyplot as plt
import random
from ObjectTracker import region_extraction

def adaptive_region_growing(image, alpha_channel):
    """
    Perform adaptive region growing on an image.

    Parameters:
        image (np.ndarray): Input image as a NumPy array.

    Returns:
        np.ndarray: Clustered image with each region assigned a unique cluster ID.
        int: Number of clusters detected in the image.
    """
    # Exclude transparent regions
    non_transparent_mask = alpha_channel > 0
    image_channel = np.where(non_transparent_mask, image_channel, 0)

    # Initialize variables
    rows, cols = image.shape[:2]
    clusters = np.zeros((rows, cols), dtype=np.int32)
    cluster_id = 1
    stack = []
    
    # Define 8-connectivity offsets
    offsets = [
        (-1, -1), (-1, 0), (-1, 1),
        (0, -1), (0, 1),
        (1, -1), (1, 0), (1, 1)
    ]
    
    # Iterate over the image startinf from the top-left corner
    for i in range(rows):
        for j in range(cols):
            if clusters[i, j] == 0 and non_transparent_mask[i, j]:
                stack.append((i, j))
                while stack:
                    current = stack.pop()
                    for offset in offsets:
                        x, y = current[0] + offset[0], current[1] + offset[1]
                        if (
                            0 <= x < rows and 0 <= y < cols and
                            clusters[x, y] == 0 and
                            non_transparent_mask[x, y] and
                            abs(int(image_channel[x, y]) - int(image_channel[current])) <= 50
                        ):
                            clusters[x, y] = cluster_id
                            stack.append((x, y))
                cluster_id += 1
                
    # Starting from the smaller cluster, merge it with its biggest neighbor until 10 clusters are left
    while cluster_id > 10:
        cluster_sizes = np.bincount(clusters.flatten())
        cluster_sizes[0] = 0
        smallest_cluster = np.argmin(cluster_sizes)
        if smallest_cluster == 0:
            break
        cluster_sizes[smallest_cluster] = 0
        biggest_neighbor = np.argmax(cluster_sizes)
        clusters[clusters == smallest_cluster] = biggest_neighbor
        cluster_id -= 1
    
    return clusters, cluster_id - 1

# Example usage
'''if __name__ == "__main__":
    # Load a grayscale image
    image = imread("Demo/Images/PersonNBG.png", as_gray=True)
    # convert image to 8-bit grayscale
    image = (image * 255).astype(np.uint8)  # Normalize to 8-bit if needed

    # Perform adaptive region growing
    segmented_image, num_clusters = adaptive_region_growing(image, threshold=20)

    print(f"Number of clusters detected: {num_clusters}")'''

if __name__ == "__main__":
    image_path = "Demo/YOLO_Image/person/4_ROI.png"  # Replace with your PNG image path
    output_path = "Demo/segmented_image.png"   # Path to save the processed image
    try:
        # Process the image
        output_image, num_regions, clusters = region_extraction.preprocess_and_segment(image_path)
        print(f"Number of regions detected: {num_regions}")
        # Save the output image
        cv2.imwrite(output_path, output_image)
        print(f"Segmented image saved to {output_path}")
        # Display the output image
        cv2.imshow("Segmented Image", output_image)
        cv2.waitKey(0)  # Wait for a key press to close the window
        cv2.destroyAllWindows()
    except Exception as e:
        print(f"Error: {e}")
