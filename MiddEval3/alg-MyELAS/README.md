# MyELAS Algorithm for MiddEval3

This is a custom ELAS (Efficient Large-scale Stereo) algorithm implementation adapted for the Middlebury Stereo Evaluation v.3 framework.

## Features

- **ELAS Algorithm**: Efficient Large-scale Stereo matching
- **SGM-style Hole Filling**: Post-processing to fill invalid disparity regions
- **PFM Output**: Standard PFM format output for MiddEval3 compatibility
- **Custom Image Processing**: Uses custom PNG reading/writing functions
- **No External Dependencies**: Pure C++ implementation with libpng only

## Build Instructions

1. **Prerequisites**:
   - libpng development library
   - CMake 3.10+
   - C++14 compiler

2. **Install libpng**:
   ```bash
   sudo apt install libpng-dev
   ```

3. **Build**:
   ```bash
   chmod +x build.sh
   ./build.sh
   ```

4. **Manual Build**:
   ```bash
   mkdir -p build
   cd build
   cmake ..
   make
   ```

## Usage

The algorithm is designed to work with MiddEval3 framework:

```bash
# Run on single dataset
./runalg Q Motorcycle MyELAS

# Run on all training sets
./runalg Q training MyELAS

# Evaluate results
./runeval Q Motorcycle 1.0 MyELAS
```

## Algorithm Parameters

- **Disparity Range**: [0, maxdisp] (from calib.txt)
- **Support Threshold**: 0.95
- **Support Texture**: 10
- **Grid Size**: 20
- **Filtering**: Median filter
- **Post-processing**: SGM-style hole filling

## Output

- `disp0MyELAS.pfm`: Dense disparity map in PFM format
- `timeMyELAS.txt`: Runtime in seconds

## Integration with MiddEval3

This algorithm follows the MiddEval3 interface:
- Input: PNG stereo images (im0.png, im1.png)
- Output: PFM disparity map (disp0.pfm)
- Timing: Runtime in seconds (time.txt)

## Notes

- Uses SGM-style hole filling for better completeness
- Converts ELAS negative disparities to INFINITY for PFM format
- Uses custom PNG reading functions (no ImageMagick dependency)
- Supports both RGB and grayscale PNG images
- Optimized for quarter-resolution datasets 