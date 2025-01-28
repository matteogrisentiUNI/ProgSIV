import cv2
import numpy as np

def remove_background(image_path):
    """
    Removes the background from an input image using contour detection.
    
    Args:
        image_path (str): Path to the input image.
    
    Returns:
        np.ndarray: Binary mask of the foreground object.
    """
    # Step 1: Load the image
    print("[INFO] Loading image...")
    image = cv2.imread(image_path)
    if image is None:
        print("[ERROR] Could not load image. Check the file path.")
        return None
    print("[INFO] Image loaded with shape:", image.shape)

    # Step 2: Resize for faster processing (optional)
    max_dim = 500  # Resize based on the largest dimension
    height, width = image.shape[:2]
    scale = max_dim / max(height, width) if max(height, width) > max_dim else 1
    image = cv2.resize(image, (int(width * scale), int(height * scale)))
    print("[INFO] Image resized to:", image.shape)

    # Step 3: Convert to grayscale
    print("[INFO] Converting image to grayscale...")
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Step 4: Apply Gaussian Blur
    print("[INFO] Applying Gaussian Blur...")
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Step 5: Apply binary thresholding
    print("[INFO] Applying binary thresholding...")
    _, binary_mask = cv2.threshold(blurred, 128, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Step 6: Find contours
    print("[INFO] Finding contours...")
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Step 7: Identify the largest contour (assuming it's the foreground object)
    print("[INFO] Identifying largest contour...")
    if len(contours) == 0:
        print("[ERROR] No contours found.")
        return None

    largest_contour = max(contours, key=cv2.contourArea)

    # Step 8: Create a mask from the largest contour
    print("[INFO] Creating foreground mask...")
    mask = np.zeros_like(binary_mask)
    cv2.drawContours(mask, [largest_contour], -1, (255), thickness=cv2.FILLED)

    # Optional: Debug visualization of intermediate steps
    print("[DEBUG] Contour area of largest object:", cv2.contourArea(largest_contour))
    # cv2.imshow("Original Image", image)
    # cv2.imshow("Binary Threshold", binary_mask)
    # cv2.imshow("Foreground Mask", mask)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    # Step 9: Return the mask
    print("[INFO] Background removal complete.")
    return mask

# Example usage:
if __name__ == "__main__":
    mask = remove_background("Demo/YOLO_Image/Car/1_Box.jpg")
    if mask is not None:
        # Save the mask as an output image
        cv2.imwrite("Demo/YOLO_Image/person/X_mask.png", mask)
        print("[INFO] Foreground mask saved as 'foreground_mask.png'")
