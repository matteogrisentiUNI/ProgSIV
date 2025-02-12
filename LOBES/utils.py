import cv2
from matplotlib import pyplot as plt
import numpy as np
import random
import os

def draw_histogram(hist, height, width):
    # Plot a instogram on a graph
    plt.figure(figsize=(width / 100, height / 100))
    plt.title("Color Histogram ")
    plt.xlabel("Pixel Intensity")
    plt.ylabel("Normalized Frequency")
    plt.plot(hist["blue"], color="blue", label="Blue")
    plt.plot(hist["green"], color="green", label="Green")
    plt.plot(hist["red"], color="red", label="Red")
    plt.legend()

    # Save the plot as an image to show in OpenCV
    temp_filename = 'histogram_plot.png'
    plt.savefig(temp_filename)
    plt.close()

    # Load the saved plot as an image
    plot_img = cv2.imread(temp_filename)

    # Resize for better display
    plot_img = cv2.resize(plot_img, (width, height))

    # Remove the temporary file
    os.remove(temp_filename)

    return plot_img

def plot_histograms(hist1, hist2, width, height):
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
    temp_filename = 'histogram_plot.png'
    plt.savefig(temp_filename)
    plt.close()

    # Load the saved plot as an image
    plot_img = cv2.imread('histogram_plot.png')


    # Resize for better display
    plot_img = cv2.resize(plot_img, (width, height))

    cv2.imshow('Histogram Plot', plot_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    # Remove the temporary file
    os.remove(temp_filename)

    return plot_img

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

def visualize_superpixels_with_random_colors(image, labels):
    # Create a blank image to hold the colored superpixels
    height, width, _ = image.shape
    colored_superpixels = np.zeros((height, width, 3), dtype=np.uint8)
    
    # Assign a random color to each superpixel
    unique_labels = np.unique(labels)
    random_colors = {label: [random.randint(0, 255) for _ in range(3)] for label in unique_labels}
    
    for label in unique_labels:
        mask = labels == label
        colored_superpixels[mask] = random_colors[label]
    
    # Show the visualization
    cv2.imshow("Superpixels Visualization", colored_superpixels)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

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
    cv2.waitKey(0)  # short delay to update window
    cv2.destroyAllWindows()

def draw_mask(frame, boxes=None, masks=None, class_names=None, color_mask=(255, 105, 180)):
    
    # Converti in liste se non lo sono
    if not isinstance(boxes, (list, np.ndarray)):
        boxes = [boxes]
    elif isinstance(boxes, (list, np.ndarray)) and not all(isinstance(b, (list, np.ndarray)) for b in boxes):
        boxes = [boxes]
    if not isinstance(masks, (list, np.ndarray)):
        masks = [masks]
    elif masks.ndim == 2:
        masks = [masks]
    if not isinstance(class_names, (list, np.ndarray)):
        class_names = [class_names]
    
    # Trova la lunghezza massima
    max_len = max(len(boxes), len(masks), len(class_names))
    
    # Normalizza le liste alla stessa lunghezza
    boxes.extend([None] * (max_len - len(boxes)))
    masks.extend([None] * (max_len - len(masks)))
    class_names.extend([None] * (max_len - len(class_names)))

    for box, mask, class_name in zip(boxes, masks, class_names):

        #Check if the mask is valid
        if mask is not None:

            # Resize the mask to match the frame's size
            resized_mask = cv2.resize(mask, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_NEAREST)

            # Convert mask to boolean for indexing
            boolean_mask = resized_mask.astype(bool)

            # Create a pink overlay with the same shape as the frame
            pink_overlay = np.zeros_like(frame, dtype=np.uint8)
            pink_overlay[:] = color_mask# Pink in BGR

            # Apply the translucent pink overlay only on the masked area
            frame = np.where(boolean_mask[:, :, None], cv2.addWeighted(frame, 0.5, pink_overlay, 0.5, 0), frame)

        #Check if the bounding box is valid
        if box is not None:
            # Draw the bounding box in blue
            x1, y1, x2, y2 = map(int, box) 
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)

        # Check if the name is valid
        if class_name is not None:
            # Put the class name on the top-right corner of the box
            cv2.putText(frame, class_name, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (255, 255, 255), 2, cv2.LINE_AA)
    return frame

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

def rescale(image, target_pixels):
    # Get the current image dimensions
    im_y, im_x = image.shape[:2]  # Shape of the image (height, width)
    #print(f"Image size: {im_x}x{im_y}")
    actual_number_of_pixels = im_x * im_y

    # Compute the scaling factor
    scaling_factor = (target_pixels / actual_number_of_pixels) ** 0.5

    # Compute new width and height while preserving aspect ratio
    new_width = int(im_x * scaling_factor)
    new_height = int(im_y * scaling_factor)

    # Resize the image
    resized_image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_LINEAR)

    return resized_image, scaling_factor


def resize_mask_with_padding(mask, box, target_height, target_width):
    """
    Resize the mask and apply it to the original frame, handling out-of-bounds cases.
    """
    # Extract box coordinates
    x1, y1, x2, y2 = map(int, box)

    # Ensure the box is within the image boundaries
    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(target_width, x2)
    y2 = min(target_height, y2)

    # Recalculate box dimensions after clipping
    box_width = x2 - x1
    box_height = y2 - y1

    if box_width <= 0 or box_height <= 0:
        # If the box is completely out of bounds, return an empty mask
        return np.zeros((target_height, target_width), dtype=np.uint8)

    # Resize the mask to the (clipped) box dimensions
    resized_mask = cv2.resize(mask, (box_width, box_height))
    # Create a black image of the target frame size
    padded_mask = np.zeros((target_height, target_width), dtype=np.uint8)

    # Insert the resized mask into the padded mask
    padded_mask[y1:y2, x1:x2] = resized_mask

    return padded_mask

def shrink_box_to_mask(resized_box, mask, threshold=5):

    # Trova i contorni della maschera per determinare l'area occupata
    mask_indices = np.argwhere(mask > 0)  # Trova i pixel non neri
    if mask_indices.size == 0:
        return resized_box  # Se la maschera è vuota, mantieni il box originale
    
    # Calcola il centro del box attuale
    centroid = find_centroid(mask)

    # Trova i limiti della maschera
    min_y, min_x = mask_indices.min(axis=0)
    max_y, max_x = mask_indices.max(axis=0)

    # Calcola larghezza e altezza della maschera, con un margine di sicurezza (threshold)
    width = (max_x - min_x) + 2 * threshold
    height = (max_y - min_y) + 2 * threshold

    # Definisci il nuovo bounding box
    x1 = max(centroid[0] - width*1.05 // 2, 0)
    x2 = min(centroid[0] + width*1.05 // 2, mask.shape[1])
    y1 = max(centroid[1] - height*1.05 // 2, 0)
    y2 = min(centroid[1] + height*1.05 // 2, mask.shape[0])

    return np.array([ int(x1), int(y1), int(x2), int(y2) ])

def predict_bounding_box_final(image, box, affine_matrix):
    """
    Predicts the transformed bounding box using an affine matrix, 
    doubles its width and height while keeping the object centered, 
    and adds padding based on the displacement ratio in each direction.

    Parameters:
      image         : Current frame (BGR) for clipping.
      box           : Previous bounding box [x_min, y_min, x_max, y_max].
      affine_matrix : 2x3 affine transformation matrix.

    Returns:
      final_box     : Predicted, expanded, and padded bounding box as an int32 NumPy array.
    """
    # Convert original box to float and get image size
    box = np.array(box, dtype=np.float32)
    im_h, im_w = image.shape[:2]
    orig_width = box[2] - box[0]
    orig_height = box[3] - box[1]

    # Compute predicted bounding box via affine transformation
    corners = np.array([
        [box[0], box[1]],
        [box[2], box[1]],
        [box[2], box[3]],
        [box[0], box[3]]
    ], dtype=np.float32)
    transformed = cv2.transform(corners.reshape(-1, 1, 2), affine_matrix).reshape(-1, 2)

    # Get min/max coordinates of transformed box
    x_min_pred, y_min_pred = np.min(transformed, axis=0)
    x_max_pred, y_max_pred = np.max(transformed, axis=0)
    pred_width = x_max_pred - x_min_pred
    pred_height = y_max_pred - y_min_pred

    # Compute max absolute displacement in each direction
    max_disp_x = max(abs(x_max_pred - box[2]), abs(box[0] - x_min_pred))
    max_disp_y = max(abs(y_max_pred - box[3]), abs(box[1] - y_min_pred))

    # Center the box and double its width and height
    x_center = (x_min_pred + x_max_pred) / 2
    y_center = (y_min_pred + y_max_pred) / 2
    expanded_width = 2 * pred_width
    expanded_height = 2 * pred_height

    x_min_exp = x_center - expanded_width / 2
    x_max_exp = x_center + expanded_width / 2
    y_min_exp = y_center - expanded_height / 2
    y_max_exp = y_center + expanded_height / 2

    # Compute padding based on displacement ratio
    padding_x = (max_disp_x / orig_width) * 0.5 * orig_width
    padding_y = (max_disp_y / orig_height) * 0.5 * orig_height

    # Apply padding
    x_min_final = max(0, int(x_min_exp - padding_x))
    y_min_final = max(0, int(y_min_exp - padding_y))
    x_max_final = min(im_w - 1, int(x_max_exp + padding_x))
    y_max_final = min(im_h - 1, int(y_max_exp + padding_y))

    return np.array([x_min_final, y_min_final, x_max_final, y_max_final], dtype=np.int32)

def compute_motion_scaling_factor(A, base_scale=1.5, min_scale=1, max_scale=2):
    """
    Calcola il motion scaling factor basato sulla matrice di trasformazione affine.
    
    :param A: Matrice affine (2x3) numpy array
    :param base_scale: Fattore base di scaling
    :param min_scale: Valore minimo di scaling
    :param max_scale: Valore massimo di scaling
    :return: Motion scaling factor adattivo
    """
    if A.shape != (2, 3):
        raise ValueError("The matrix should be 2x3")

    # Estrai i coefficienti di scala e shear
    a, b, tx = A[0]
    c, d, ty = A[1]

    # Calcola il fattore di scala basato sulla norma euclidea
    S_x = np.sqrt(a**2 + c**2)
    S_y = np.sqrt(b**2 + d**2)
    motion_scaling_factor = (S_x + S_y) / 2  # Media tra le due scale

    # Normalizza e scala nel range desiderato
    motion_scaling_factor = np.clip(motion_scaling_factor * base_scale, min_scale, max_scale)

    return motion_scaling_factor



# DEBUG PRINT FUNCTIONS
def debugPrintDetection(previus_frame, boxes, masks, object_detected, frame_width, frame_height, output_folder=None):
    masked_previus_frame = draw_mask(previus_frame, boxes, masks[0], object_detected)
    resized_masked_previus_frame = cv2.resize(masked_previus_frame, (frame_width // 2, frame_height // 2), interpolation=cv2.INTER_AREA)
    
    cv2.imshow("Masked Image", resized_masked_previus_frame)
    if output_folder is not None:        # Save the processed image
        processed_image_path = os.path.join(output_folder, "1_Detection_YOLO_Mask.jpg")
        cv2.imwrite(processed_image_path, resized_masked_previus_frame)
    
    # block the code in order to analyse the behavior frame to frame
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def debugPrintMaskRefinement(refined_mask, box, previus_frame, object_detected, frame_width, frame_height, output_folder=None):
    
    masked_previus_frame = draw_mask(previus_frame, box, refined_mask, object_detected, color_mask=(255, 0, 0))
    resized_masked_previus_frame = cv2.resize(masked_previus_frame, (frame_width // 2, frame_height // 2), interpolation=cv2.INTER_AREA)
    
    cv2.imshow("Refined Masked Image", resized_masked_previus_frame)
    if output_folder is not None:        # Save the processed image
        processed_image_path = os.path.join(output_folder, "2_MaskRefinment.jpg")
        cv2.imwrite(processed_image_path, resized_masked_previus_frame)
    
    # block the code in order to analyse the behavior frame to frame
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def debugPrintMotionEstimation(previus_frame, previus_poi, next_poi, A, box, box_t, output_folder=None):
    
    # Apply the mask to extract the region of interest
    # roi = cv2.bitwise_and(cropped_previus_frame, cropped_previus_frame, mask=mask)
    # roi_path = os.path.join(output_folder, "4_1_ROI.jpg")   
    # cv2.imwrite(roi_path, roi)

    # Draw points of interest on the current frame and the next frame 
    frame_points = previus_frame.copy()

    for point in previus_poi:
        x, y = point.ravel()
        cv2.circle(frame_points, (int(x), int(y)), 5, (255, 0, 0), -1) #blu

    for point in next_poi:
        x, y = point.ravel()
        cv2.circle(frame_points, (int(x), int(y)), 4, (0, 255, 0), -1)


    # Rappresent the computed motion A with an arrow
    center_x, center_y = previus_frame.shape[1] // 2, previus_frame.shape[0] // 2  # Center of the image
    
    start_point = np.array([center_x, center_y, 1])         # Start of the arrow in the center of the image
    end_point = A @ np.array([center_x + 50, center_y, 1])  # End of the arrow dipent on the direction an intensiti of the motion; applied a offset of 50

    start = (int(start_point[0]), int(start_point[1]))      # convert to int coordinates
    end = (int(end_point[0]), int(end_point[1]))

    cv2.arrowedLine(frame_points, start, end, (0, 0, 255), 3, tipLength=0.3)  # Draw to arrow in red
    
    # Draw the original box and the translated box
    np.reshape(box, 4)
    x1, y1, x2, y2 = map(int, box) 
    cv2.rectangle(frame_points, (x1, y1), (x2, y2), (255, 0, 0), 3)

    np.reshape(box_t, 4)
    x1_t, y1_t, x2_t, y2_t = map(int, box_t) 
    cv2.rectangle(frame_points, (x1_t, y1_t), (x2_t, y2_t), (0, 255, 0), 2, lineType=4)

    cv2.imshow("Motion Image", frame_points)
    if output_folder is not None:        # Save the processed image
        frame_points_path = os.path.join(output_folder, "3_MotionEstimation.jpg")
        cv2.imwrite(frame_points_path, frame_points)
    
    # block the code in order to analyse the behavior frame to frame
    cv2.waitKey(0)
    #cv2.destroyAllWindows() 

def debugPrintFrameCrop(cropped_next_frame, output_folder=None):
    frame_height = 600
  
    h, w = cropped_next_frame.shape[:2]          # Get the originale dimension 
    aspect_ratio = int( w / h )                  # compute the ratio to manintain the same proporsion
    cropped_previus_frame_mask = cv2.resize(cropped_next_frame, (int(frame_height)*aspect_ratio, frame_height), interpolation=cv2.INTER_AREA)
    
    cv2.imshow("Cropped Next Frame", cropped_previus_frame_mask)
    if output_folder is not None:        
        final_images_path = os.path.join(output_folder, "4_Resize&Crop.jpg")
        cv2.imwrite(final_images_path, cropped_previus_frame_mask)
    
    # block the code in order to analyse the behavior frame to frame
    cv2.waitKey(0) 
    #cv2.destroyAllWindows()

def debugPrintSegmentation(blurred, mask, output_folder=None):

    # overlay the mask to the image as a translucent layer
    masked_image = draw_mask(blurred, masks=mask)

    cv2.imshow("New Mask", masked_image)
    if output_folder is not None:        
        combined_image_path = os.path.join(output_folder, "6_Segmentation.jpg")
        cv2.imwrite(combined_image_path, masked_image)
    
    cv2.waitKey(0) 
    #cv2.destroyAllWindows()

