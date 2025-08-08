#!/bin/bash

# 构建MyELAS算法

echo "Building MyELAS algorithm..."

# 创建build目录
mkdir -p build
cd build

# 检查ImageMagick是否安装
if ! command -v convert &> /dev/null; then
    echo "Warning: ImageMagick not found!"
    echo "Please install ImageMagick: sudo apt install imagemagick"
    echo "Continuing build anyway..."
fi

# 运行cmake
echo "Running cmake..."
cmake ..

# 编译
echo "Compiling..."
make

if [ $? -eq 0 ]; then
    echo "Build completed successfully!"
    echo "Executable: build/my_elas"
    echo "You can now test with: ./runalg Q Motorcycle MyELAS"
else
    echo "Build failed!"
    exit 1
fi 