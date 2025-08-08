#!/bin/bash

# 自动化构建脚本：安装依赖并在 build 目录下执行 cmake 和 make
set -e

echo "[INFO] install dependence: libpng, PCL, ImageMagick ..."
sudo apt update
sudo apt install -y libpng-dev libpcl-dev imagemagick

# 创建 build 目录（如果不存在）
mkdir -p build
cd build

# 运行 CMake 生成 Makefile
cmake ..

# 编译
make -j$(nproc)

echo "[INFO] Build finished successfully!"