## log
### ../test/testtripletshow.cpp
We added a helper function `toGrayscale(const MyImage& src, MyImage& gray)` to convert RGBA images to grayscale before colorization. This was needed to correctly process ELAS output in `testtripletshow.cpp`.

### ../test/testdisparityBM.cpp
We attempted to improve edge accuracy in disparity estimation and sdded a custom `medianFilter3x3()` to reduce noise. However, the visual quality did not improve. Retained `medianFilter3x3()` in codebase for potential future use.

### ../src/pointcloud.cpp & pointcloud.h + ../test/testpointcloud.cpp

```bash
Mesh triangulateFromDepth(const DepthImage& depth,
                          const ColorImage& color,
                          const CameraIntrinsics& K,
                          float depthThreshold,
                          int stepSize);
```

### ICP Alignment Pipeline

#### Added files:
```bash
../src/utils/kdtree.h
../src/utils/nanoflann.hpp
../src/icp.cpp
../src/icp.h
../test/testicp.cpp
```

#### Workflow:

Start from the existing pipeline (e.g., disparity estimation using ELAS).

Generate two point clouds separately:

From `disparity_ELAS_left.png`

From `disparity_ELAS_right.png`

Note: You must manually change the input path in testpointcloud.cpp to switch between the left and right disparity maps.

After generating both point clouds, run:

```bash
./testicp
```
This aligns the right point cloud to the left using rigid ICP.

The aligned and color-coded result will be saved as:

```bash
../output/aligned_colored.ply
```
Left point cloud: green
Aligned right point cloud: red

This visualization helps verify whether the ICP alignment succeeded by checking how well the two point clouds overlap.