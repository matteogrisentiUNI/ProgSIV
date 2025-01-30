import cv2
import time
import psutil
import numpy as np
import os
from ObjectTracker import ObjectTracker, draw_mask

def main():
    # Initialize Object Tracker for 'dog' class
    tracker = ObjectTracker(target_class='boat', conf_threshold=0.1)

    # Load the input video
    video_path = 'Demo/Video/Ship.mp4'
    output_video_path = 'Demo/YOLO_Video/ShipYOLOMasked.mp4'

    # Define and create the performance log directory
    performance_dir = 'Demo/YOLO_Performance'
    performance_log_path = os.path.join(performance_dir, 'ShipYOLOPerformance.txt')
    os.makedirs(performance_dir, exist_ok=True)

    cap = cv2.VideoCapture(video_path)

    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    # get video orientation
    # orientation = int(cap.get(cv2.CAP_PROP_ORIENTATION))
    # if the video is in portrait mode rotate it of 90 degrees
    # if orientation == 90:
    #    width, height = height, width

    # Define video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    # Performance tracking variables
    cpu_usage_list = []
    memory_usage_list = []
    frame_times = []

    start_time = time.time()

    frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_start_time = time.time()

        # Monitor CPU and Memory usage before detection
        cpu_before = psutil.cpu_percent(interval=None)
        memory_before = psutil.virtual_memory().percent

        # YOLO Detection
        detections = tracker.detect(frame)

        # Extract detection data
        boxes = [d['box'] for d in detections]
        masks = [d['mask'] for d in detections]
        class_names = [d['class_name'] for d in detections]

        # Draw masks and bounding boxes
        masked_frame = draw_mask(frame, boxes, masks, class_names)

        # Write the frame to output video
        out.write(masked_frame)

        # Monitor CPU and Memory usage after detection
        cpu_after = psutil.cpu_percent(interval=None)
        memory_after = psutil.virtual_memory().percent

        # Record performance metrics
        cpu_usage_list.append((cpu_before + cpu_after) / 2)
        memory_usage_list.append((memory_before + memory_after) / 2)
        frame_time = time.time() - frame_start_time
        frame_times.append(frame_time)

        frame_count += 1

    total_processing_time = time.time() - start_time

    # Compute average performance metrics
    avg_cpu_usage = np.mean(cpu_usage_list)
    avg_memory_usage = np.mean(memory_usage_list)
    avg_time_per_frame = np.mean(frame_times)

    # Write performance log
    with open(performance_log_path, 'x') as f:
        f.write(f"Total frames processed: {frame_count}\n")
        f.write(f"Average CPU usage: {avg_cpu_usage:.2f}%\n")
        f.write(f"Average Memory usage: {avg_memory_usage:.2f}%\n")
        f.write(f"Average time per frame: {avg_time_per_frame:.4f} seconds\n")
        f.write(f"Total processing time: {total_processing_time:.2f} seconds\n")

    # Release resources
    cap.release()
    out.release()
    print(f"Processing complete. Video saved as {output_video_path}")
    print(f"Performance log saved as {performance_log_path}")

if __name__ == "__main__":
    main()
