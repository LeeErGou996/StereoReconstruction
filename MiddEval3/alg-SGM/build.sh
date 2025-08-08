#!/bin/bash

# SGM算法构建脚本

echo "Building SGM algorithm for MiddEval3..."
echo "======================================"

# 检查当前目录
if [ ! -f "CMakeLists.txt" ]; then
    echo "Error: CMakeLists.txt not found in current directory"
    exit 1
fi

# 创建构建目录
echo "Creating build directory..."
mkdir -p build
cd build

# 检查依赖文件
echo "Checking dependencies..."
if [ ! -f "../../../MysterroReconstr/src/SGM/SemiGlobalMatching_linux.cpp" ]; then
    echo "Error: SemiGlobalMatching_linux.cpp not found!"
    exit 1
fi

if [ ! -f "../../../MysterroReconstr/src/SGM/sgm_util_linux.cpp" ]; then
    echo "Error: sgm_util_linux.cpp not found!"
    exit 1
fi

if [ ! -f "../../../MysterroReconstr/src/utils/imageutils.cpp" ]; then
    echo "Error: imageutils.cpp not found!"
    exit 1
fi

echo "All dependencies found."

# 运行cmake
echo "Running cmake..."
cmake ..

if [ $? -ne 0 ]; then
    echo "Error: cmake failed!"
    exit 1
fi

# 编译
echo "Compiling..."
make -j$(nproc)

if [ $? -ne 0 ]; then
    echo "Error: compilation failed!"
    exit 1
fi

# 检查可执行文件
if [ -f "sgm" ]; then
    echo "Build successful! Executable: build/sgm"
    ls -la sgm
else
    echo "Error: Executable not found!"
    exit 1
fi

echo "SGM build completed successfully!" 