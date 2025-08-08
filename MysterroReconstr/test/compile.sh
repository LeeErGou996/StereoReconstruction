#!/bin/bash

echo "=== 编译深度计算测试程序 ==="

# 编译命令
g++ -std=c++14 -I../src -o testdepth \
    testdepth.cpp \
    ../src/depth.cpp \
    ../src/utils/imageutils.cpp \
    -lpng

if [ $? -eq 0 ]; then
    echo "编译成功!"
    echo "运行测试程序..."
    ./testdepth
else
    echo "编译失败!"
    exit 1
fi 