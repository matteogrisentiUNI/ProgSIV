# Low-Level Object-Tracking & Segmentation (LOB&S)
A Python library for efficient object tracking and segmentation using low-level video and image processing techniques.

# Features
- Mask refinement using morphological operations and guided filters.
- Motion estimation with Sparse Optical Flow and Affine transformation.
- Superpixel-based segmentation with histogram-based refinement.
- Dynamic cropping for computational efficiency.

# Installation
```bash
pip install git+https://github.com/matteogrisenti/ProgSIV.git
```

# Pipeline Overview
- **Initial Object Detection**: Uses a YOLO v8_seg model, pre-trained on COCO, to detect the bounding box and mask of the object in the first frame.
- **Mask Refinement**: Applies morphological operations (opening/closure) and a guided filter for denoising and smoothing.
- **Motion Estimation**: Predicts the next bounding box using Sparse Optical Flow and Affine transformation.
- **Segmentation**: Performs superpixel-based segmentation followed by histogram refinement and mask creation.

# Supported Algorithms and Techniques
- **Object Detection**: YOLO v8_seg model for the first frame.
- **Mask Refinement**: Morphological operations and guided filtering.
- **Motion Estimation**:
  - Sparse Optical Flow using Shi-Tomasi Corner Detection and Lucas-Kanade Optical Flow.
  - Affine transformation for object motion prediction.
- **Segmentation**:
  - Simple Linear Iterative Clustering (SLIC) for superpixels.
  - Histogram-based refinement for background removal.
  - Contour-based refinement for final mask creation.

# Example use
```python
import os
from LOBES import LOBES

if __name__ == "__main__":

    video_path = 'test/Video/Car4.mp4'
    output_folder = os.path.join('test/Global/Car4')
    object_detected = 'car'

    LOBES(video_path, object_detected, vertical=False, output_folder=output_folder,  saveVideo=True, debugPrint=False)
```

# Core Functions
```python
# main function that performes Object Tracking and Segmentation on the input video
def LOBES(video_path, object_detected, vertical=False, output_folder=None, saveVideo=False, debugPrint=False):
```
```python
# Performs YOLO v8_seg detection and segmentation on an image
def detection(frame, object_class): return masks, boxes
```
```python
# motion estimation function, performs targeted motion estimation using Lucas Kanade optical flow to predict the position of a specific object on the next frame
def mask_motion_estimation(previus_frame, next_frame, mask): return previous_points, next_points, affine_matrix
```
```python
# segmentation function, performs slic, histogram based refinement and contour refinement to extract the subject from an image
def segmentation (cropped_image, pred_hist, debugPrint=False): return mask
```
*The library also includes more than 30 helper functions*

# Performance
The main function performance, on a 300$ laptop, is the following:
```
Average CPU usage: 14%
Average Memory usage: 79%
Average time per frame: 0.24 seconds
Motion estimation approximate accuracy: 95%
Segmentation approximate accuracy: 55-60%
```

# Limitations
- YOLO v8_seg detection is pretrained with COCO dateset and may not generalize to all scenarios.
- Mask refinement and segmentation depend on the quality of initial detection and input video.
- Motion estimation may struggle with rapid or non-linear object movements.
- Segmentation is highly dependent on the accuracy of the predicted bounding box and may thus diverge for higly dynamic and complex scenarios.

