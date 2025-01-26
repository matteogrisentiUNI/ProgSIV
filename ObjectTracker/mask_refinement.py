import cv2
import numpy as np

def refine_mask(image, mask, kernel_size=3, guided_radius=5, guided_eps=1e-2):
    """
    Refines a segmentation mask using morphological operations and guided filtering.

    Args:
        image (numpy.ndarray): The input image (H x W x C) in BGR format.
        mask (numpy.ndarray): The binary segmentation mask (H x W) with values 0 or 255.
        kernel_size (int): Kernel size for morphological operations (default: 3).
        guided_radius (int): Radius for the guided filter (default: 5).
        guided_eps (float): Regularization parameter for the guided filter (default: 1e-2).

    Returns:
        numpy.ndarray: The refined mask (H x W) with values 0 or 255.
    """
    # Ensure the mask is binary
    mask = (mask > 0).astype(np.uint8) * 255

    # Morphological operations to reduce noise
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)  # Fill small gaps
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)   # Remove noise

    # Normalize the mask for guided filtering
    normalized_mask = mask.astype(np.float32) / 255.0

    # Convert the guide image to grayscale to match the mask's single channel
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Resize the mask to match the guide image's size
    resized_mask = cv2.resize(normalized_mask, (gray_image.shape[1], gray_image.shape[0]), interpolation=cv2.INTER_NEAREST)

    # Apply guided filtering to refine the mask
    refined_mask = cv2.ximgproc.guidedFilter(gray_image, resized_mask, guided_radius, guided_eps)

    # Threshold the refined mask to binary (0 or 255)
    refined_mask = (refined_mask > 0.5).astype(np.uint8) * 255

    return refined_mask
