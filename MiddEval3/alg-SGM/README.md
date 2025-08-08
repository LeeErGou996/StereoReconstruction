# SGM Algorithm for MiddEval3

This directory contains the Semi-Global Matching (SGM) algorithm implementation for the Middlebury Stereo Evaluation v.3 framework.

## Overview

SGM is a popular stereo matching algorithm that uses semi-global optimization to compute dense disparity maps. This implementation is based on the SemiGlobalMatching library and adapted for the MiddEval3 evaluation framework.

## Files

- `run` - Main execution script for MiddEval3
- `CMakeLists.txt` - CMake configuration file
- `build.sh` - Build script
- `src/main.cpp` - Main SGM implementation
- `README.md` - This documentation

## Building

To build the SGM algorithm:

```bash
cd alg-SGM
chmod +x build.sh
./build.sh
```

This will create the `sgm` executable in the `build/` directory.

## Usage

The algorithm is designed to be called by the MiddEval3 framework:

```bash
./runalg Q Motorcycle SGM
```

Or manually:

```bash
./run <im0.png> <im1.png> <ndisp> <outdir>
```

## Parameters

The SGM algorithm uses the following default parameters:

- **Disparity Range**: [0, maxdisp-1]
- **Number of Paths**: 8 (8-path aggregation)
- **Census Window**: 5x5 pixels
- **P1 Penalty**: 10
- **P2 Penalty**: 150
- **Uniqueness Ratio**: 0.95
- **LR Check Threshold**: 1.0
- **Min Speckle Area**: 20 pixels
- **Fill Holes**: Enabled

## Output

The algorithm produces:
- `disp0.pfm` - Dense disparity map in PFM format
- `time.txt` - Execution time in seconds

## Dependencies

- CMake 3.10+
- C++14 compiler
- SemiGlobalMatching library (included)
- Image processing utilities (included)

## Performance

SGM typically provides good accuracy with reasonable computational cost. The algorithm is particularly effective for structured scenes and provides dense disparity maps with hole filling.

## Notes

- The algorithm processes PGM format images (converted from PNG by the run script)
- Invalid disparities are marked as INFINITY in the PFM output
- The implementation includes speckle removal and hole filling for better results 