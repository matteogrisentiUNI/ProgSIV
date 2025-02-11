import cv2
import time
import psutil
import glob
import numpy as np
import os
from LOB_S import detection, mask_motion_estimation, motion_estimation

def calculate_mean_run(file_path):
    
    total_cpu = 0
    total_memory = 0
    total_time_per_frame = 0
    total_processing_time = 0
    run_count = 0

    with open(file_path, 'r') as file:
        lines = file.readlines()

    for line in lines:
        if "Average CPU usage" in line:
            cpu_usage = float(line.split(":")[1].strip().replace('%', ''))
            total_cpu += cpu_usage
        elif "Average Memory usage" in line:
            memory_usage = float(line.split(":")[1].strip().replace('%', ''))
            total_memory += memory_usage
        elif "Average time per frame" in line:
            time_per_frame = float(line.split(":")[1].strip().split()[0])  
            total_time_per_frame += time_per_frame
        elif "Total processing time" in line:
            processing_time = float(line.split(":")[1].strip().split()[0])  
            total_processing_time += processing_time
            run_count += 1

    # Calcola le medie
    mean_cpu = total_cpu / run_count
    mean_memory = total_memory / run_count
    mean_time_per_frame = total_time_per_frame / run_count
    mean_processing_time = total_processing_time / run_count

    # Scrivi le medie nel file
    with open(file_path, 'a') as file:
        file.write(f"\n--- Mean ({run_count} runs)---\n")
        file.write(f"Average CPU usage: {mean_cpu:.2f}%\n")
        file.write(f"Average Memory usage: {mean_memory:.2f}%\n")
        file.write(f"Average time per frame: {mean_time_per_frame:.4f} seconds\n")
        file.write(f"Total processing time: {mean_processing_time:.2f} seconds\n")


def test_motion_estimantion_video(video_path, object_detected,  output_folder=None, saveVideo=False, showVideo = False):
    
    cap = cv2.VideoCapture(video_path)
    os.makedirs(output_folder, exist_ok=True)
    performance_log_path = os.path.join(output_folder, 'performance.txt')
    os.makedirs(os.path.dirname(performance_log_path), exist_ok=True)


    # Setup video writer if saving is enabled
    if saveVideo:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        out = cv2.VideoWriter(os.path.join(output_folder, 'tracked_video.mp4'), fourcc, fps, (frame_width, frame_height))

    # Performance tracking variables
    cpu_usage_list = []
    memory_usage_list = []
    frame_times = []
    start_time = time.time()
    frame_start_time = time.time()
    frame_count = 0


    # to elaborate the first and second frame we have to extract the poi  
    ret, previus_frame = cap.read()            # read first frame 
    ret, next_frame = cap.read()               # read second frame
    frame_count = 2 

    
    # Monitor CPU and Memory usage before detection
    cpu_before = psutil.cpu_percent(interval=None)
    memory_before = psutil.virtual_memory().percent


    # Finds the mask of the object we want to tracj using YOLO
    masks,boxes = detection(previus_frame, object_detected , output_folder=output_folder)
    if not masks: 
        print('No', object_detected, ' detected in the first frame')
        return
    
    mask = masks[0].astype(np.uint8)
    box = boxes[0]
    box = box.astype(np.float32)

    try:
        good_previus_poi, good_next_poi, A = mask_motion_estimation(previus_frame, next_frame, mask=mask, output_folder=output_folder)
    except Exception as e:
        print(f"ERROR - mask motion estimation: {e}")
        return  # Termina la funzione

    # Apply affine transformation
    box = box.reshape(-1, 1, 2)
    box = cv2.transform(box, A).reshape(-1, 4)
    x1, y1, x2, y2 = box[0]
    

    color_poi = (0, 255, 0)         # color point of interest
    color_box = (255, 0, 255)       # color box
    

    # Draws the point of interest and the box on the next frame
    if saveVideo or showVideo:
        frame = next_frame.copy()
        for new in good_next_poi:
            a, b = new.ravel()      # Returns a contiguous flattened array as (x, y) coordinates for new point
            frame = cv2.circle(frame, (int(a), int(b)), 3, color_poi, -1)                      # Draws circle (thickness of -1) at new position with green color and radius of 3
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)) , color_box, 2)            # Draw the box
        
        # Save or show the frame with the box
        if saveVideo:
            out.write(frame)
        
        if showVideo:
            resized_frame = cv2.resize(frame, (frame_width // 2, frame_height // 2), interpolation=cv2.INTER_AREA)
            cv2.imshow("Sparse Optical Flow", resized_frame)
            if cv2.waitKey(10) & 0xFF == ord('q'):
                cap.release()
                cv2.destroyAllWindows()
                return

    
    previus_frame = cv2.cvtColor(next_frame, cv2.COLOR_BGR2GRAY) 
    previus_poi = good_next_poi.reshape(-1, 1, 2)


    # Monitor CPU and Memory usage after detection
    cpu_after = psutil.cpu_percent(interval=None)
    memory_after = psutil.virtual_memory().percent

    # Record performance metrics
    cpu_usage_list.append((cpu_before + cpu_after) / 2)
    memory_usage_list.append((memory_before + memory_after) / 2)
    frame_time = time.time() - frame_start_time
    frame_times.append(frame_time)


    while(cap.isOpened()):

        # Monitor CPU and Memory usage before detection
        frame_start_time = time.time()
        cpu_before = psutil.cpu_percent(interval=None)
        memory_before = psutil.virtual_memory().percent
        
        ret, next_frame_color = cap.read()
        if not ret:  
            print("Video Ended")
            break  
        next_frame = cv2.cvtColor(next_frame_color, cv2.COLOR_BGR2GRAY)

        try:
            good_previus_poi, good_next_poi, A = motion_estimation(previus_frame, next_frame, previus_poi)
        except Exception as e:
            print(f"ERROR - motion estimation: {e}")
            return  # Termina la funzione

        box = box.reshape(-1, 1, 2) 
        box = cv2.transform(box, A).reshape(-1, 4)
        x1, y1, x2, y2 = box[0]
        # print(x1, y1, x2, y2, box)

        # Draws the point of interest and the box on the next frame
        if saveVideo or showVideo:
            frame = next_frame_color.copy()
            for new in good_next_poi:
                a, b = new.ravel()      # Returns a contiguous flattened array as (x, y) coordinates for new point
                frame = cv2.circle(frame, (int(a), int(b)), 3, color_poi, -1)                      # Draws circle (thickness of -1) at new position with green color and radius of 3
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)) , color_box, 2)            # Draw the box
            
            # Save or show the frame with the box
            if saveVideo:
                out.write(frame)
            
            if showVideo:
                resized_frame = cv2.resize(frame, (frame_width // 2, frame_height // 2), interpolation=cv2.INTER_AREA)
                cv2.imshow("Sparse Optical Flow", resized_frame)
                if cv2.waitKey(10) & 0xFF == ord('q'):
                    cap.release()
                    cv2.destroyAllWindows()
                    return
        
        previus_frame = next_frame
        previus_poi = good_next_poi.reshape(-1, 1, 2)
        

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
    # Open the file in append mode instead of exclusive creation
    with open(performance_log_path, 'a') as f:
        f.write("\n--- New Run ---\n")
        f.write(f"Total frames processed: {frame_count}\n")
        f.write(f"Average CPU usage: {avg_cpu_usage:.2f}%\n")
        f.write(f"Average Memory usage: {avg_memory_usage:.2f}%\n")
        f.write(f"Average time per frame: {avg_time_per_frame:.4f} seconds\n")
        f.write(f"Total processing time: {total_processing_time:.2f} seconds\n")

    # Release resources
    cap.release()
    if saveVideo:
        out.release()
    cv2.destroyAllWindows()

# Run K tests on the same video and derive a mean of the performance 
def k_test(video_path, object_detected,  output_folder):

    performance_log_path = os.path.join(output_folder, 'performance.txt')   
    if os.path.exists(performance_log_path):        # clean the old performance reports
        os.remove(performance_log_path)

    print('Iteration 1')
    test_motion_estimantion_video(video_path, object_detected, output_folder,  saveVideo=True, showVideo=False)

    for i in range(9): 
        print('Iteration ', i+2)
        test_motion_estimantion_video(video_path, object_detected, output_folder,  saveVideo=False, showVideo=False)

    calculate_mean_run(performance_log_path)
def calculate_mean_run(file_path):
    
    total_cpu = 0
    total_memory = 0
    total_time_per_frame = 0
    total_processing_time = 0
    run_count = 0

    with open(file_path, 'r') as file:
        lines = file.readlines()

    for line in lines:
        if "Average CPU usage" in line:
            cpu_usage = float(line.split(":")[1].strip().replace('%', ''))
            total_cpu += cpu_usage
        elif "Average Memory usage" in line:
            memory_usage = float(line.split(":")[1].strip().replace('%', ''))
            total_memory += memory_usage
        elif "Average time per frame" in line:
            time_per_frame = float(line.split(":")[1].strip().split()[0])  
            total_time_per_frame += time_per_frame
        elif "Total processing time" in line:
            processing_time = float(line.split(":")[1].strip().split()[0])  
            total_processing_time += processing_time
            run_count += 1

    # Calcola le medie
    mean_cpu = total_cpu / run_count
    mean_memory = total_memory / run_count
    mean_time_per_frame = total_time_per_frame / run_count
    mean_processing_time = total_processing_time / run_count

    # Scrivi le medie nel file
    with open(file_path, 'a') as file:
        file.write(f"\n--- Mean ({run_count} runs)---\n")
        file.write(f"Average CPU usage: {mean_cpu:.2f}%\n")
        file.write(f"Average Memory usage: {mean_memory:.2f}%\n")
        file.write(f"Average time per frame: {mean_time_per_frame:.4f} seconds\n")
        file.write(f"Total processing time: {mean_processing_time:.2f} seconds\n")

# Run the test on a set of videos inside the folder path
def process_videos_in_folder(folder_path, object_detected, output_base_folder, saveVideo=True, showVideo=False):
    # Found all the videos in the folder
    video_files = glob.glob(os.path.join(folder_path, "*.mp4"))  

    if not video_files:
        print("No viedo founded in:", folder_path)
        return

    for video_path in video_files:
        video_name = os.path.splitext(os.path.basename(video_path))[0]  # Nome of the file without the extension
        output_folder = os.path.join(output_base_folder, f"{video_name}")

        print(f"Elaboration of Video: {video_name}")

        test_motion_estimantion_video(
            video_path, object_detected, output_folder, 
            saveVideo=saveVideo, showVideo=showVideo
        )


if __name__ == "__main__":

    video_path = 'Demo/Video/Ship.mp4'
    output_folder = os.path.join('test/MotionEstimation/Ship')
    object_detected = 'boat'

    test_motion_estimantion_video(video_path, object_detected, output_folder,  saveVideo=True, showVideo=True)
    # process_videos_in_folder(video_path, object_detected, output_folder,  saveVideo=True, showVideo=True)
    # k_test(video_path, object_detected, output_folder)