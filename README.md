# Low Level Object Tracking & Segmentation (LOB&S)
A Python library for efficient object tracking and segmentation using low-level video and image processing techniques.

# Features
- Mask refinement using morphological operations and guided filters.
- Motion estimation with Sparse Optical Flow and Affine transformation.
- Superpixel-based segmentation with histogram-based refinement.
- Dynamic cropping for computational efficiency.

# Installation
I do not have enough information to write this.

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

# Examples
I do not have enough information to write this.

# Performance
I do not have enough information to write this.

# Limitations
- YOLO v8_seg detection is pretrained with COCO dateset and may not generalize to all scenarios.
- Mask refinement and segmentation depend on the quality of initial detection and input video.
- Motion estimation may struggle with rapid or non-linear object movements.
- Segmentation may diverge for higly dynamic and complex scenarios.

# Contact
GitHub Repository: [ProgSIV](https://github.com/matteogrisenti/ProgSIV)
