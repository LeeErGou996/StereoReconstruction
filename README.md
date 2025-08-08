# MysterroReconstr - 3D Reconstruction Project

## Usage

### Quick Start Example
Workflow from stereo matching to point cloud generation:

```bash
cd build
./testdisparityELAS test8                    # Generate disparity map using ELAS
./testdepth test8                            # Compute depth from disparity
./testpointcloud test8 left normal ../output/test8/disparity_ELAS_left_original_no_fill.png  # Generate point cloud

./testpointcloud test8 right normal ../output/test8/disparity_ELAS_right_original_no_fill.png  # Generate point cloud 

./testpointcloud test8 left normal ../output/test8/disparity_ELAS_left_original_no_fill.png  # Generate point cloud 
```

### 1. Build Project
First, use `install_and_build.sh` to build the project:

```bash
chmod +x install_and_build.sh
./install_and_build.sh
```

### 2. Run Test Programs
Enter the build folder to test programs starting with "test":

```bash
cd build
```

Available test programs:
- `./testimage` - Image processing test
- `./testfeature_matching` - Feature matching test
- `./test8point` - 8-point algorithm test
- `./testRectification` - Image rectification test
- `./testdisparity` - Stereo matching test
- `./testdisparityELAS` - ELAS stereo matching test
- `./testdisparitySGM` - SGM stereo matching test
- `./testdisparityADCE` - AD-Census stereo matching test
- `./testdisparityBM` - BM stereo matching test
- `./testdepth` - Depth computation test
- `./testRectificationVis` - Rectification visualization test
- `./testpointcloud` - Point cloud generation test
- `./testicp` - ICP registration test
- `./testicp_plane` - Point-to-plane ICP test
- `./testtripletshow` - Triplet view display test

### 3. Output Results
All results are saved in the `output` folder.

## Source Code
Original code is saved in the `src` directory.

## Function Modules
Includes the following functions:
- feature_matching - Feature matching
- 8point - 8-point algorithm
- disparityELAS - ELAS stereo matching
- BM - Block matching algorithm
- SGM - Semi-global matching
- ADCE - AD-Census algorithm
- pointcloud - Point cloud processing
- depth - Depth map

## MiddEval3 Program Execution

download testQ and traningQ from this link:
https://vision.middlebury.edu/stereo/submit3/zip/MiddEval3-data-Q.zip

### Generate Disparity Maps
Use the `runalg` command to generate disparity maps for the training set:

```bash
cd MiddEval3
./runalg Q training ADCE    # Use ADCE algorithm
./runalg Q training SGM     # Use SGM algorithm
./runalg Q training ELAS    # Use ELAS algorithm
./runalg Q training BM      # Use BM algorithm
```

### Evaluate Algorithm Performance
Use the `runeval` command to calculate bad2.0 metrics:

```bash
# Evaluate a single algorithm
./runeval Q training 2 ADCE    # Evaluate bad2.0 metrics for ADCE algorithm

# Evaluate all algorithms
./runeval Q training 2 MyELAS SGM ADCE BM
```

### Visualize Results
Use the `runviz` command to convert PFM format to PNG format:

```bash
./runviz Q    # Convert all PFM images to PNG format
```

### Parameter Description
- `Q` - Quarter resolution
- `H` - Half resolution  
- `F` - Full resolution
- `training` - Training dataset
- `test` - Test dataset
- `2` - bad2.0 threshold