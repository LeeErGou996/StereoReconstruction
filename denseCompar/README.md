# DenseCompar - 立体视觉深度估计算法比较系统

## 项目概述

DenseCompar是一个基于OpenCV的立体视觉深度估计算法比较系统，支持多种密集匹配算法，并采用创新的色差图质量评估方式。系统专注于评估生成的色差图的视觉质量，而不是传统的与真实深度图的绝对误差比较。

## 支持的算法

### 1. 密集匹配算法
- **SGBM (Semi-Global Block Matching)**: OpenCV实现的半全局块匹配算法
- **BM (Block Matching)**: 传统的块匹配算法
- **ELAS (Efficient Large-Scale Stereo)**: 高效的立体匹配算法，支持多种预设配置

### 2. 稀疏匹配算法
- **8-Point Algorithm**: 用于相机位姿估计的8点算法
- **ORB/SIFT/SURF**: 多种特征检测器支持

## 核心特性

### 1. 色差图质量评估系统
系统采用创新的评估方式，专注于色差图的视觉质量：

- **基本质量指标**: 对比度、饱和度、亮度、锐度
- **连续性指标**: 平滑度、边缘保持、噪声水平
- **结构指标**: 梯度幅值、纹理丰富度
- **视觉质量指标**: 颜色一致性、深度感知质量
- **综合视觉质量分数**: 所有指标的加权综合

### 2. 配置管理
- 支持热重载的配置文件系统
- 详细的参数说明和调优建议
- 算法参数的灵活配置

### 3. 输出格式
- 原始视差图和深度图
- 彩色视差图（Jet和Hot色彩映射）
- 详细的评估报告
- 算法比较结果

## 文件结构

```
denseCompar/
├── main.cpp                              # 主程序入口
├── color_disparity_quality_metrics.h     # 色差图质量评估头文件
├── color_disparity_quality_metrics.cpp   # 色差图质量评估实现
├── generate_color_quality_report.cpp     # 独立的质量评估报告生成器
├── config.h                              # 配置管理头文件
├── config.cpp                            # 配置管理实现
├── config.txt                            # 配置文件
├── denseMatching.h/.cpp                  # 密集匹配算法接口
├── depth.h/.cpp                          # 深度计算模块
├── sparseMatching.h/.cpp                 # 稀疏匹配算法
├── bmMatcher.h/.cpp                      # BM算法实现
├── sgbmMatcher.h/.cpp                    # SGBM算法实现
├── elasMatcher.h/.cpp                    # ELAS算法实现
├── 8point.h/.cpp                         # 8点算法实现
├── elas/                                 # ELAS算法库
├── data/                                 # 输入数据目录
│   ├── left/                             # 左图像
│   ├── right/                            # 右图像
│   ├── camera/                           # 相机参数
│   └── ground/                           # 真实深度图
├── output/                               # 输出结果目录
└── build/                                # 编译输出目录
```

## 编译和运行

### 1. 环境要求
- OpenCV 4.0+
- C++17 编译器
- CMake 3.10+
- X11库（Linux）

### 2. 编译步骤
```bash
cd denseCompar
mkdir build && cd build
cmake ..
make -j4
```

### 3. 运行主程序
```bash
./denseCompar
```

### 4. 运行独立的质量评估
```bash
./color_quality_report ../output/test1/
```

## 配置说明

### 主要参数
- `numDisparities`: 最大视差范围（16的倍数）
- `blockSize`: 匹配窗口大小（奇数）
- `sgbmP1/P2`: SGBM平滑项参数
- `elasPreset`: ELAS预设配置

### 路径配置
- `leftDir/rightDir`: 左右图像目录
- `cameraDir`: 相机参数目录
- `outputDir`: 输出结果目录

## 评估指标

### 色差图质量指标（主要评估方式）
1. **对比度 (0-1)**: 明暗对比程度，越高越好
2. **锐度 (0-1)**: 图像清晰度，越高越好
3. **边缘保持 (0-1)**: 边缘保持能力，越高越好
4. **平滑度 (0-1)**: 图像平滑程度，越高越好
5. **噪声水平 (0-1)**: 噪声程度，越低越好
6. **综合视觉质量分数 (0-1)**: 所有指标的加权综合

### 传统误差指标（参考）
- MSE/RMSE/MAE: 与真实深度图的误差
- Middlebury评估指标: 视差误差百分比等

## 使用示例

### 1. 基本使用
```bash
# 编译
cd build && make

# 运行（自动处理data目录下的所有图像对）
./denseCompar

# 查看结果
ls ../output/
```

### 2. 质量评估
```bash
# 生成详细的质量报告
./color_quality_report ../output/test1/

# 查看报告
cat ../output/test1/color_disparity_quality_report.txt
```

### 3. 参数调优
```bash
# 编辑配置文件
vim ../config.txt

# 重新运行（支持热重载）
./denseCompar
```

## 输出文件说明

### 主程序输出
- `disparity_*.png`: 原始视差图
- `disparity_*_color_jet.png`: Jet色彩映射视差图
- `disparity_*_color_hot.png`: Hot色彩映射视差图
- `depth_*.png`: 深度图
- `depth_*_color.png`: 彩色深度图
- `depth_*_raw.exr`: 原始深度数据
- `error_comparison.txt`: 算法比较报告

### 质量评估输出
- `color_disparity_quality_report.txt`: 详细质量报告
- `color_quality_comparison.txt`: 算法质量比较

## 算法选择建议

### 视觉质量优先
- 选择综合视觉质量分数最高的算法
- 通常ELAS在视觉质量方面表现较好

### 对比度优先
- 选择对比度最高的算法
- SGBM通常具有较高的对比度

### 锐度优先
- 选择锐度最高的算法
- BM算法通常具有较高的锐度

## 注意事项

1. **图像格式**: 支持PNG、JPG、JPEG、BMP、TIFF格式
2. **相机参数**: 需要提供相机内参文件（txt格式）
3. **真实深度图**: 可选，用于传统误差评估
4. **内存使用**: 大图像可能需要较多内存
5. **计算时间**: ELAS算法相对较慢，但质量较高

## 故障排除

### 常见问题
1. **编译错误**: 检查OpenCV版本和依赖库
2. **运行错误**: 检查输入路径和文件格式
3. **内存不足**: 减小图像尺寸或增加系统内存
4. **参数调优**: 参考config.txt中的参数说明

### 性能优化
1. 使用较小的`numDisparities`值
2. 调整`blockSize`参数
3. 启用ELAS的`elasSubsampling`
4. 使用多线程编译

## 版本历史

- **v2.0**: 引入色差图质量评估系统
- **v1.0**: 基础算法比较功能

## 许可证

本项目基于MIT许可证开源。

## 贡献

欢迎提交Issue和Pull Request来改进项目。 