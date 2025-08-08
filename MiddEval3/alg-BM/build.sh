#!/bin/bash

# Build script for BM algorithm

echo "Building BM algorithm for MiddEval3..."

# 检查是否在正确的目录
if [ ! -f "CMakeLists.txt" ]; then
    echo "Error: CMakeLists.txt not found. Please run this script from the alg-BM directory."
    exit 1
fi

# 创建build目录
if [ ! -d "build" ]; then
    echo "Creating build directory..."
    mkdir -p build
fi

# 进入build目录
cd build

# 运行cmake
echo "Running cmake..."
cmake .. -DCMAKE_BUILD_TYPE=Release

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

# 检查可执行文件是否创建成功
if [ ! -f "bm" ]; then
    echo "Error: executable 'bm' not found after compilation!"
    exit 1
fi

echo "Build completed successfully!"
echo "Executable: $(pwd)/bm"

# 显示可执行文件信息
echo "Executable details:"
ls -la bm

cd ..

echo "BM algorithm is ready to use!"
echo "Usage: ./run <im0.png> <im1.png> <ndisp> <outdir>" 