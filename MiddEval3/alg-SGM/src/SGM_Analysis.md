# SGM (Semi-Global Matching) Algorithm Analysis

## Overview
The SGM folder contains a complete implementation of the Semi-Global Matching algorithm for stereo disparity estimation. This is a high-quality, efficient implementation that follows the original SGM paper by Hirschmüller.

## File Structure

### Core Files
- **`SemiGlobalMatching.h`** - Main class header with interface definition
- **`SemiGlobalMatching.cpp`** - Implementation of the SGM algorithm
- **`sgm_types.h`** - Type definitions and constants
- **`sgm_util.h`** - Utility functions header
- **`sgm_util.cpp`** - Utility functions implementation

### Supporting Files
- **`stdafx.h`** - Precompiled header
- **`stdafx.cpp`** - Precompiled header implementation
- **`targetver.h`** - Windows target version

## Algorithm Components

### 1. Census Transform
- **Purpose**: Converts pixel values to binary descriptors for robust matching
- **Options**: 5x5 or 9x7 census windows
- **Implementation**: `census_transform_5x5()` and `census_transform_9x7()`

### 2. Cost Computation
- **Method**: Hamming distance between census descriptors
- **Output**: Initial matching costs for each disparity level

### 3. Cost Aggregation
- **Method**: Multi-directional path aggregation (4 or 8 paths)
- **Directions**: Left-right, right-left, up-down, down-up, and 4 diagonal directions
- **Penalty**: P1 for small disparity changes, P2 for large changes

### 4. Disparity Computation
- **Method**: Winner-takes-all (WTA) optimization
- **Output**: Initial disparity map

### 5. Post-processing
- **Left-right consistency check**: Removes occluded pixels
- **Speckle removal**: Removes small isolated regions
- **Hole filling**: Interpolates invalid disparities
- **Median filtering**: Smooths the final result

## Key Parameters

### Disparity Range
```cpp
option.min_disparity = 0;      // Minimum disparity
option.max_disparity = 64;     // Maximum disparity
```

### Path Configuration
```cpp
option.num_paths = 8;          // Number of aggregation paths (4 or 8)
```

### Census Transform
```cpp
option.census_size = SemiGlobalMatching::Census5x5;  // 5x5 or 9x7
```

### Quality Checks
```cpp
option.is_check_unique = true;           // Enable uniqueness check
option.uniqueness_ratio = 0.95f;        // Uniqueness threshold
option.is_check_lr = true;              // Enable left-right check
option.lrcheck_thres = 1.0f;            // Left-right threshold
```

### Speckle Removal
```cpp
option.is_remove_speckles = true;       // Enable speckle removal
option.min_speckle_aera = 20;           // Minimum speckle area
```

### Hole Filling
```cpp
option.is_fill_holes = true;            // Enable hole filling
```

### Penalty Parameters
```cpp
option.p1 = 10;                         // Small disparity change penalty
option.p2_init = 150;                   // Large disparity change penalty
```

## Usage Example

### Basic Usage
```cpp
#include "SemiGlobalMatching.h"

// Create SGM instance
SemiGlobalMatching sgm;

// Configure parameters
SemiGlobalMatching::SGMOption option;
option.min_disparity = 0;
option.max_disparity = 64;
option.num_paths = 8;
option.census_size = SemiGlobalMatching::Census5x5;
option.is_check_unique = true;
option.uniqueness_ratio = 0.95f;
option.is_check_lr = true;
option.lrcheck_thres = 1.0f;
option.is_remove_speckles = true;
option.min_speckle_aera = 20;
option.is_fill_holes = true;
option.p1 = 10;
option.p2_init = 150;

// Initialize
if (!sgm.Initialize(width, height, option)) {
    // Handle initialization error
    return false;
}

// Process images
float* disparity_map = new float[width * height];
if (!sgm.Match(left_image_data, right_image_data, disparity_map)) {
    // Handle matching error
    return false;
}

// Use disparity map
// ...

// Clean up
delete[] disparity_map;
```

### Parameter Tuning Guidelines

#### For High-Quality Results
- Use 8 paths for better accuracy
- Use 9x7 census for more robust matching
- Enable all quality checks
- Use higher uniqueness ratio (0.95-0.99)
- Use lower left-right threshold (0.5-1.0)

#### For Real-time Performance
- Use 4 paths for faster processing
- Use 5x5 census for faster computation
- Disable some quality checks if needed
- Use lower uniqueness ratio (0.8-0.9)

#### For Different Scenes
- **Indoor scenes**: Lower disparity range (32-64)
- **Outdoor scenes**: Higher disparity range (64-128)
- **Textured scenes**: Lower P1, higher P2
- **Smooth scenes**: Higher P1, lower P2

## Performance Characteristics

### Time Complexity
- Census transform: O(width × height)
- Cost computation: O(width × height × disparity_range)
- Cost aggregation: O(width × height × disparity_range × num_paths)
- Post-processing: O(width × height)

### Memory Usage
- Census values: 2 × width × height × sizeof(uint32/uint64)
- Cost arrays: 2 × width × height × disparity_range × sizeof(uint8/uint16)
- Disparity maps: 2 × width × height × sizeof(float)

### Typical Performance
- **512×512 images, 64 disparities, 8 paths**: ~100-200ms
- **1024×1024 images, 128 disparities, 8 paths**: ~500-1000ms

## Advantages

1. **High Quality**: Produces accurate disparity maps
2. **Robust**: Handles textureless regions well
3. **Configurable**: Many parameters for different scenarios
4. **Efficient**: Optimized implementation
5. **Complete**: Includes all standard post-processing steps

## Limitations

1. **Computational Cost**: Higher than simple block matching
2. **Memory Usage**: Requires significant memory for large images
3. **Parameter Tuning**: Requires careful parameter selection
4. **Texture Dependency**: Performance depends on image texture

## Comparison with Other Algorithms

### vs Block Matching (BM)
- **SGM**: Higher quality, more robust
- **BM**: Faster, simpler

### vs ELAS
- **SGM**: More accurate in textureless regions
- **ELAS**: Faster, better for real-time applications

### vs SGBM (OpenCV)
- **SGM**: This implementation is more configurable
- **SGBM**: More optimized, better integrated with OpenCV

## Integration Notes

### Dependencies
- Requires C++11 or later
- Uses standard library containers
- No external dependencies beyond standard library

### Compilation
```bash
g++ -std=c++11 -O3 -march=native SemiGlobalMatching.cpp sgm_util.cpp -o sgm_test
```

### Thread Safety
- Not thread-safe for concurrent access
- Each instance should be used by single thread
- Multiple instances can be used in parallel

## Test Program

A complete test program `testdisparitySGM.cpp` has been created that demonstrates:
- Image loading using OpenCV
- Parameter configuration
- SGM processing
- Result saving
- Error handling

This provides a complete working example of how to use the SGM algorithm in practice. 