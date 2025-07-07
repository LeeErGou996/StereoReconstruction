# Stereo Vision, Depth, and Mesh Reconstruction Project

This project implements a complete stereo vision pipeline using OpenCV, supporting color image processing, disparity and depth map computation, point cloud generation, and 3D mesh reconstruction with color. It is designed for research and engineering applications in 3D vision, robotics, and photogrammetry.

## Quick Start with Docker

You can use the pre-built Docker image to run this project without manual environment setup:

```bash
# Pull the image
docker pull leeergou/3dproj

# Run the container (with code-server, port 8443)
docker run -it -p 8443:8443 leeergou/3dproj
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
/config/workspace/StereoReconstruction/
├── CMakeLists.txt         # Project build configuration
├── README.md              # Project documentation (this file)
├── data/                  # Example input images (left.png, right.png)
├── include/               # (Optional) Extra headers
│   └── stereo_reconstruction.hpp
├── lib/                   # External libraries (OpenCV, etc.)
│   ├── opencv/            # OpenCV source code
│   │   ├── build/         # OpenCV build directory
│   │   └── install/       # OpenCV installation directory
│   └── opencv_contrib/    # OpenCV contrib modules
│       └── modules/       # Contrib modules (xfeatures2d, etc.)
├── src/                   # Source code
│   ├── main.cpp                 # Main pipeline entry
│   ├── disparity.h/.cpp         # Disparity computation & feature matching
│   ├── depth.h/.cpp             # Depth map computation
│   ├── meshReconstruction.h/.cpp# Point cloud & mesh export
│   ├── denseMatching.h/.cpp     # Dense stereo rectification & matching
│   ├── 8point.h/.cpp            # 8-point pose estimation
│   ├── colorUtils.cpp           # Color fusion utilities
│   └── CMakeLists.txt           # Source build config
├── build/                 # Project build directory
└── test/                  # Output results (auto-generated)
    ├── disparity.png, disparity_color_jet.png, ...
    ├── depth.png, depth_color.png, depth_raw.exr
    ├── reconstructed_mesh.ply, reconstructed_mesh_simplified.ply
    └── ...
```

## Build Instructions

### Prerequisites
- CMake >= 3.10
- GCC/G++ (GNU 9.5.0 or later)
- Git
- Basic development tools (make, pkg-config, etc.)

### Step 1: Clone and Setup OpenCV with Contrib Modules

```bash
# Navigate to the project directory
cd /config/workspace/StereoReconstruction/

# Create lib directory if it doesn't exist
mkdir -p lib
cd lib

# Clone OpenCV (version 4.7.0)
git clone https://github.com/opencv/opencv.git -b 4.7.0

# Clone OpenCV contrib modules (version 4.7.0)
git clone https://github.com/opencv/opencv_contrib.git -b 4.7.0
```

### Step 2: Build OpenCV with Contrib Modules

```bash
# Navigate to OpenCV directory
cd /config/workspace/StereoReconstruction/lib/opencv

# Create build directory
mkdir -p build
cd build

# Configure CMake with contrib modules (disabled HDF to avoid MPI dependency)
cmake -DOPENCV_EXTRA_MODULES_PATH=/config/workspace/StereoReconstruction/lib/opencv_contrib/modules \
      -DCMAKE_INSTALL_PREFIX=/config/workspace/StereoReconstruction/lib/opencv/install \
      -DBUILD_opencv_hdf=OFF \
      ..

# Build OpenCV (using half of available CPU cores)
make -j$(($(nproc)/2))

# Install OpenCV
make install
```

**Important Notes:**
- We disable the HDF module (`-DBUILD_opencv_hdf=OFF`) to avoid MPI dependency issues
- The build process may take 30-60 minutes depending on your system
- Make sure you have sufficient disk space (at least 5GB free)

### Step 3: Verify OpenCV Installation

```bash
# Check if xfeatures2d module is available (required for SIFT/SURF)
ls /config/workspace/StereoReconstruction/lib/opencv/install/include/opencv2/xfeatures2d.hpp

# Check OpenCV version
/config/workspace/StereoReconstruction/lib/opencv/install/bin/opencv_version
```

### Step 4: Build the Project

```bash
# Navigate back to project root
cd /config/workspace/StereoReconstruction

# Create build directory
mkdir -p build
cd build

# Configure the project
cmake ..

# Build the project
make -j$(($(nproc)/2))
```

### Step 5: Run the Pipeline

```bash
# Run the stereo reconstruction pipeline
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
- `depth_raw.exr`: Raw float depth map (requires OpenEXR support)
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
- For better performance, adjust the number of CPU cores used in make commands based on your system.

## License
This project is for academic and research use. Please cite appropriately if used in publications. 