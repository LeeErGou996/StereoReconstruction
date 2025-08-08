#!/bin/bash

echo "=== 编译AD-Census测试程序 ==="

# 编译命令
g++ -std=c++14 -I../src -o testdisparityADCE \
    testdisparityADCE.cpp \
    ../src/ADCE/ADCensusStereo_linux.cpp \
    ../src/ADCE/cost_computor.cpp \
    ../src/ADCE/cross_aggregator.cpp \
    ../src/ADCE/scanline_optimizer.cpp \
    ../src/ADCE/multistep_refiner.cpp \
    ../src/ADCE/adcensus_util.cpp \
    ../src/ELAS/src/elas.cpp \
    ../src/ELAS/src/descriptor.cpp \
    ../src/ELAS/src/matrix.cpp \
    ../src/ELAS/src/triangle.cpp \
    ../src/ELAS/src/filter.cpp \
    ../src/ELAS/src/rectangle.cpp \
    ../src/ELAS/src/image.cpp \
    -lpng -lm

if [ $? -eq 0 ]; then
    echo "编译成功!"
    echo "运行测试程序..."
    echo "使用方法: ./testdisparityADCE <left_image> <right_image> <camera_file> [output_dir]"
    echo "示例: ./testdisparityADCE ../data/left/test1.png ../data/right/test1.png ../data/camera/test1.txt"
else
    echo "编译失败!"
    exit 1
fi 