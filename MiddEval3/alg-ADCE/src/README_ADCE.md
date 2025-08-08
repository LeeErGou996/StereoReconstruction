# AD-Census 立体匹配算法 (Linux版本)

## 概述

AD-Census是一种结合了AD（Absolute Difference）和Census变换的立体匹配算法。本实现是从Windows版本移植到Linux版本，并移除了OpenCV依赖，使用自定义的图像处理方法。

## 文件结构

```
ADCE/
├── ADCensusStereo_linux.h          # AD-Census主类头文件（Linux版本）
├── ADCensusStereo_linux.cpp        # AD-Census主类实现（Linux版本）
├── adcensus_types.h                # 数据类型定义
├── adcensus_util.h                 # 工具函数头文件
├── adcensus_util.cpp               # 工具函数实现
├── cost_computor.h                 # 代价计算器头文件
├── cost_computor.cpp               # 代价计算器实现
├── cross_aggregator.h              # 十字臂聚合器头文件
├── cross_aggregator.cpp            # 十字臂聚合器实现
├── scanline_optimizer.h            # 扫描线优化器头文件
├── scanline_optimizer.cpp          # 扫描线优化器实现
├── multistep_refiner.h             # 多步优化器头文件
├── multistep_refiner.cpp           # 多步优化器实现
└── README_ADCE.md                  # 本说明文件
```

## 主要修改

### 1. 移除OpenCV依赖
- 使用自定义的图像加载和保存函数
- 使用ELAS库的图像处理功能
- 实现了自己的Census变换和Hamming距离计算

### 2. Linux兼容性
- 使用标准C++库
- 添加了Linux版本的头文件和实现文件
- 修改了内存管理和文件操作

### 3. 接口统一
- 参考SGM算法的接口设计
- 添加了获取代价数据和视差数据的方法
- 统一了参数设置接口

## 算法流程

1. **代价计算 (Cost Computation)**
   - 将彩色图像转换为灰度图像
   - 计算AD代价（绝对差值）
   - 计算Census变换
   - 计算Census代价（Hamming距离）
   - 组合AD和Census代价

2. **代价聚合 (Cost Aggregation)**
   - 构建十字臂结构
   - 在十字臂内进行代价聚合
   - 支持多次迭代聚合

3. **扫描线优化 (Scanline Optimization)**
   - 4方向扫描线优化
   - 使用SGM的惩罚机制
   - 计算最终视差图

4. **多步优化 (Multi-step Refinement)**
   - 异常值检测
   - 迭代区域投票
   - 适当插值
   - 深度不连续性调整

## 参数说明

```cpp
struct ADCensusOption {
    sint32  min_disparity;          // 最小视差
    sint32  max_disparity;          // 最大视差
    sint32  lambda_ad;              // AD代价权重
    sint32  lambda_census;          // Census代价权重
    sint32  cross_L1;               // 十字臂空间约束L1
    sint32  cross_L2;               // 十字臂空间约束L2
    sint32  cross_t1;               // 十字臂颜色约束t1
    sint32  cross_t2;               // 十字臂颜色约束t2
    float32 so_p1;                  // 扫描线优化惩罚p1
    float32 so_p2;                  // 扫描线优化惩罚p2
    sint32  so_tso;                 // 扫描线优化阈值tso
    sint32  irv_ts;                 // 迭代区域投票阈值ts
    float32 irv_th;                 // 迭代区域投票阈值th
    float32 lrcheck_thres;          // 左右一致性检查阈值
    bool    do_lr_check;            // 是否进行左右一致性检查
    bool    do_filling;             // 是否进行视差填充
    bool    do_discontinuity_adjustment; // 是否进行不连续性调整
};
```

## 使用方法

### 1. 编译

```bash
cd test/
chmod +x compile_ADCE.sh
./compile_ADCE.sh
```

### 2. 运行

```bash
./testdisparityADCE <left_image> <right_image> <camera_file> [output_dir]
```

示例：
```bash
./testdisparityADCE ../data/left/test1.png ../data/right/test1.png ../data/camera/test1.txt
```

### 3. 编程接口

```cpp
#include "ADCE/ADCensusStereo_linux.h"

// 创建AD-Census对象
ADCensusStereo adcensus;

// 设置参数
ADCensusOption option;
option.min_disparity = 0;
option.max_disparity = 64;
// ... 设置其他参数

// 初始化
adcensus.Initialize(width, height, option);

// 执行匹配
float32* disparity_map = new float32[width * height];
adcensus.Match(img_left, img_right, disparity_map);

// 清理
delete[] disparity_map;
```

## 输出结果

算法会生成以下文件：
- `adcensus_*.raw`: 原始视差数据
- `adcensus_*.png`: 可视化的视差图
- `adcensus_*_depth.raw`: 深度图数据（如果有相机参数）
- `adcensus_*_depth.png`: 可视化的深度图

## 性能特点

- **精度**: AD-Census算法在Middlebury数据集上表现良好
- **速度**: 相比SGM算法，计算复杂度较低
- **内存**: 内存使用量适中，适合实时应用
- **鲁棒性**: 对光照变化和纹理变化有较好的鲁棒性

## 注意事项

1. 输入图像必须是3通道彩色图像
2. 左右图像尺寸必须相同
3. 视差范围应根据场景设置合适的值
4. 参数调优对结果质量有重要影响

## 参考

- 原始AD-Census算法论文
- SGM算法的实现参考
- ELAS库的图像处理功能 