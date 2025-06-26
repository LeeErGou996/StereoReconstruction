# Stereo Vision, Depth, and Mesh Reconstruction Project

This project implements a complete stereo vision pipeline using OpenCV, supporting color image processing, disparity and depth map computation, point cloud generation, and 3D mesh reconstruction with color. It is designed for research and engineering applications in 3D vision, robotics, and photogrammetry.

## Quick Start with Docker

You can use the pre-built Docker image to run this project without manual environment setup:

```bash
# Pull the image
docker pull leeergou/3dsmc-with-project:latest

# Run the container (with code-server, port 8443)
docker run -it -p 8443:8443 leeergou/3dsmc-with-project:latest
```

Then open your browser and visit `http://localhost:8443` to access the code-server environment with all project files and dependencies ready.

---

## Features

- **Flexible Image Input**: Supports both grayscale and color stereo images.
- **Feature Detection & Matching**: SIFT, SURF, ORB feature extraction and robust matching.
- **Relative Pose Estimation**: 8-point algorithm for essential matrix and camera pose.
- **Stereo Rectification**: Aligns epipolar lines for accurate disparity computation.
- **Dense Disparity Map**: Computes dense disparity using StereoBM/StereoSGBM.
- **Depth Map Calculation**: Converts disparity to metric depth using camera parameters.
- **Color Fusion**: Generates colorized disparity/depth maps and blended visualizations.
- **Point Cloud Generation**: Reprojects depth to 3D colored point clouds.
- **Mesh Reconstruction**: Exports 3D mesh/point cloud in PLY format, with color.
- **Comprehensive Output**: Saves all intermediate and final results for analysis and visualization.

## Directory Structure

```
workspace/rootba/
├── CMakeLists.txt         # Project build configuration
├── README.md              # Project documentation (this file)
├── data/                  # Example input images (left.png, right.png)
├── include/               # (Optional) Extra headers
│   └── stereo_reconstruction.hpp
├── lib/                   # External libraries (OpenCV, etc.)
│   ├── opencv/
│   └── opencv_contrib/
├── src/                   # Source code
│   ├── main.cpp                 # Main pipeline entry
│   ├── disparity.h/.cpp         # Disparity computation & feature matching
│   ├── depth.h/.cpp             # Depth map computation
│   ├── meshReconstruction.h/.cpp# Point cloud & mesh export
│   ├── denseMatching.h/.cpp     # Dense stereo rectification & matching
│   ├── 8point.h/.cpp            # 8-point pose estimation
│   ├── colorUtils.cpp           # Color fusion utilities
│   └── CMakeLists.txt           # Source build config
├── test/                  # Output results (auto-generated)
│   ├── disparity.png, disparity_color_jet.png, ...
│   ├── depth.png, depth_color.png, depth_raw.exr
│   ├── reconstructed_mesh.ply, reconstructed_mesh_simplified.ply
│   └── ...
```

## Build Instructions

### Prerequisites
- CMake >= 3.10
- GCC/G++
- OpenCV 4.7.0 (with contrib modules)
- Git

### Steps
1. **Clone OpenCV and Contrib**
   ```bash
   cd workspace/rootba/lib/
   git clone https://github.com/opencv/opencv.git -b 4.7.0
   git clone https://github.com/opencv/opencv_contrib.git -b 4.7.0
   ```
2. **Configure and Build**
   ```bash
   cd workspace/rootba
   mkdir build && cd build
   cmake ..
   make
   ```
3. **Run the Pipeline**
   ```bash
   ./main
   ```
   The program will process `data/left.png` and `data/right.png` and output results to the `test/` directory.

## Output Files (in `test/`)
- `left_original.png`, `right_original.png`: Original color images
- `left_rectified.png`, `right_rectified.png`: Rectified color images
- `feature_matches_color.png`: Visualized feature matches
- `disparity.png`: Grayscale disparity map
- `disparity_color_jet.png`, `disparity_color_hot.png`: Colorized disparity maps
- `disparity_blended.png`, `disparity_blended_strong.png`: Blended color/disparity overlays
- `depth.png`: Normalized depth map
- `depth_color.png`: Colorized depth map
- `depth_raw.exr`: Raw float depth map
- `reconstructed_mesh.ply`: 3D colored point cloud/mesh (PLY)
- `reconstructed_mesh_simplified.ply`: Downsampled mesh/point cloud

## Source Code Overview
- **main.cpp**: Orchestrates the full pipeline: image loading, feature matching, rectification, disparity/depth/mesh computation, and result saving.
- **disparity.cpp/h**: Feature detection, matching, pose estimation, and dense disparity computation.
- **depth.cpp/h**: Converts disparity to depth, handles normalization and quality evaluation.
- **meshReconstruction.cpp/h**: Generates colored point clouds and exports PLY mesh files.
- **denseMatching.cpp/h**: Handles dense stereo rectification and matching.
- **8point.cpp/h**: Implements the 8-point algorithm for pose estimation.
- **colorUtils.cpp**: Utility functions for colorizing and blending images.

## Customization & Tips
- Change input/output paths and algorithm options in `src/main.cpp`.
- Adjust stereo/feature parameters for your dataset.
- The code is modular and can be extended for new algorithms or output formats.

## License
This project is for academic and research use. Please cite appropriately if used in publications.

---

## Update Log

### 6.26 ergou update
- **Updated README file**: Enhanced documentation with comprehensive project information and usage instructions
- **Implemented new Docker deployment**: 
  - Created Docker image with pre-configured environment
  - Added code-server integration for web-based development
  - Simplified setup process with one-command deployment
- **Implemented new input and output save format**:
  - Organized input images into separate left and right folders under data directory
  - Created output subfolders named after input filenames for better organization
  - Added batch processing capability for multiple image pairs
  - Implemented comprehensive logging system with log.txt files in each output subfolder
  - Enhanced file organization and result management 