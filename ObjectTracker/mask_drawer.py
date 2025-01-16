import cv2
import numpy as np

def draw_mask(frame, boxes, masks, class_names):
    for box, mask, class_name in zip(boxes, masks, class_names):

        #Check if the mask is valid
        if mask is not None:

            # Resize the mask to match the frame's size
            resized_mask = cv2.resize(mask, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_NEAREST)

            # Convert mask to boolean for indexing
            boolean_mask = resized_mask.astype(bool)

            # Create a pink overlay with the same shape as the frame
            pink_overlay = np.zeros_like(frame, dtype=np.uint8)
            pink_overlay[:] = (255, 105, 180)  # Pink in BGR

            # Apply the translucent pink overlay only on the masked area
            frame = np.where(boolean_mask[:, :, None], cv2.addWeighted(frame, 0.5, pink_overlay, 0.5, 0), frame)

        #Check if the bounding box is valid
        if box is not None:
            # Draw the bounding box in blue
            x1, y1, x2, y2 = box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)

        # Check if the name is valid
        if class_name is not None:
            # Put the class name on the top-right corner of the box
            cv2.putText(frame, class_name, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (255, 255, 255), 2, cv2.LINE_AA)
    return frame
