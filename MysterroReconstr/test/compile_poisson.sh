#!/bin/bash

echo "=== 编译Poisson重建测试程序 ==="

# 设置编译器标志
CXX_FLAGS="-std=c++14 -O2 -Wall -Wextra"
INCLUDE_DIRS="-I../src -I../src/utils"

# 检查PCL支持
PCL_FLAGS=""
if pkg-config --exists pcl_common; then
    PCL_FLAGS="$(pkg-config --cflags --libs pcl_common pcl_surface pcl_io)"
    CXX_FLAGS="$CXX_FLAGS -DHAVE_PCL"
    echo "✓ PCL支持已启用"
else
    echo "⚠ PCL不可用，将使用简化方法"
fi

# 检查OpenMP支持
if command -v g++ &> /dev/null; then
    # 尝试编译OpenMP测试
    echo "int main() { return 0; }" > /tmp/omp_test.cpp
    if g++ -fopenmp /tmp/omp_test.cpp -o /tmp/omp_test 2>/dev/null; then
        CXX_FLAGS="$CXX_FLAGS -fopenmp"
        echo "✓ OpenMP支持已启用"
    else
        echo "⚠ OpenMP不可用，继续编译"
    fi
    rm -f /tmp/omp_test.cpp /tmp/omp_test
fi

# 编译Poisson重建测试程序
echo "编译testpoisson..."
g++ $CXX_FLAGS $INCLUDE_DIRS \
    testpoisson.cpp \
    ../src/poissonReconstruction.cpp \
    ../src/depth.cpp \
    ../src/utils/imageutils.cpp \
    -lpng $PCL_FLAGS \
    -o testpoisson

if [ $? -eq 0 ]; then
    echo "✓ testpoisson编译成功!"
    echo "运行测试: ./testpoisson"
else
    echo "✗ testpoisson编译失败!"
    exit 1
fi

# 编译深度图测试程序（如果需要）
echo "编译testdepth..."
g++ $CXX_FLAGS $INCLUDE_DIRS \
    testdepth.cpp \
    ../src/depth.cpp \
    ../src/utils/imageutils.cpp \
    -lpng \
    -o testdepth

if [ $? -eq 0 ]; then
    echo "✓ testdepth编译成功!"
    echo "运行测试: ./testdepth"
else
    echo "✗ testdepth编译失败!"
fi

echo "=== 编译完成 ===" 