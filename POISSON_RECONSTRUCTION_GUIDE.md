# Poisson表面重建指南

## 概述

本项目提供了多种Poisson表面重建的实现方案，包括自定义实现和基于成熟库的实现。

## 可用的Poisson重建实现

### 1. PCL (Point Cloud Library) - 推荐

PCL是最成熟的开源点云处理库，提供真正的Poisson重建算法。

#### 安装PCL

**Ubuntu/Debian:**
```bash
sudo apt-get update
sudo apt-get install libpcl-dev
```

**Windows (使用vcpkg):**
```bash
vcpkg install pcl
```

**macOS:**
```bash
brew install pcl
```

#### 使用PCL Poisson重建

```cpp
#include "pcl_poisson_reconstruction.h"

// 检查PCL是否可用
if (MeshReconstruction::isPCLAvailable()) {
    // 设置参数
    MeshReconstruction::setPCLPoissonParams(8, 8, 1.5f);
    
    // 从深度图重建
    Mesh mesh = MeshReconstruction::generatePCLPoissonMesh(
        depthMap, colorImage, K, 8, 8, 1.5f);
    
    // 保存结果
    MeshReconstruction::saveMeshFile(mesh, "result.ply", "ply");
}
```

#### PCL参数说明

- `depth`: 八叉树深度 (默认8)
- `solverDivide`: 求解器分割深度 (默认8)
- `samplesPerNode`: 每个节点的样本数 (默认1.5)
- `confidence`: 是否使用置信度权重 (默认false)
- `manifold`: 是否保持流形 (默认false)

### 2. Open3D - Python接口

Open3D提供简单易用的Python接口。

#### 安装Open3D

```bash
pip install open3d
# 或
conda install -c open3d-admin open3d
```

#### 使用Open3D Poisson重建

```python
import open3d as o3d
import numpy as np

# 读取点云
pcd = o3d.io.read_point_cloud("input.ply")

# 估计法线
pcd.estimate_normals()

# Poisson重建
mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
    pcd, depth=8, width=0, scale=1.1, linear_fit=False)

# 保存结果
o3d.io.write_triangle_mesh("output.ply", mesh)
```

### 3. CGAL - 高质量几何算法

CGAL提供高质量的几何算法实现。

#### 安装CGAL

**Ubuntu/Debian:**
```bash
sudo apt-get install libcgal-dev
```

**Windows (vcpkg):**
```bash
vcpkg install cgal
```

#### 使用CGAL Poisson重建

```cpp
#include <CGAL/poisson_surface_reconstruction.h>
#include <CGAL/IO/read_xyz_points.h>

// 读取点云和法线
std::vector<Point> points;
std::vector<Vector> normals;

// 计算平均间距
double average_spacing = CGAL::compute_average_spacing<CGAL::Sequential_tag>(
    points, 6, CGAL::parameters::point_map(CGAL::make_property_map(points)));

// Poisson重建
Polyhedron output_mesh;
bool ok = CGAL::poisson_surface_reconstruction_delaunay_2(
    points.begin(), points.end(),
    CGAL::make_property_map(points),
    CGAL::make_property_map(normals),
    output_mesh, average_spacing);
```

## 项目集成

### 编译配置

项目已配置为可选地支持PCL。如果PCL不可用，相关功能会被禁用但不会影响编译。

```cmake
# CMakeLists.txt 已包含PCL支持
find_package(PCL QUIET)
if(PCL_FOUND)
    add_definitions(-DHAVE_PCL)
    target_link_libraries(main ${PCL_LIBRARIES})
endif()
```

### 使用示例

```cpp
// 在你的代码中
#include "pcl_poisson_reconstruction.h"

void processDepthMap(const cv::Mat& depthMap, const cv::Mat& colorImage, const cv::Mat& K) {
    // 检查是否有PCL支持
    if (MeshReconstruction::isPCLAvailable()) {
        std::cout << "Using PCL Poisson reconstruction..." << std::endl;
        
        Mesh mesh = MeshReconstruction::generatePCLPoissonMesh(
            depthMap, colorImage, K, 8, 8, 1.5f);
        
        if (!mesh.vertices.empty()) {
            MeshReconstruction::saveMeshFile(mesh, "pcl_result.ply", "ply");
        }
    } else {
        std::cout << "PCL not available, using custom implementation..." << std::endl;
        
        // 使用自定义实现（不是真正的Poisson重建）
        Mesh mesh = MeshReconstruction::generatePoissonMesh(depthMap, colorImage, K);
        
        if (!mesh.vertices.empty()) {
            MeshReconstruction::saveMeshFile(mesh, "custom_result.ply", "ply");
        }
    }
}
```

## 性能比较

| 实现 | 质量 | 速度 | 内存使用 | 易用性 |
|------|------|------|----------|--------|
| PCL | 高 | 中等 | 中等 | 高 |
| Open3D | 高 | 快 | 低 | 很高 |
| CGAL | 很高 | 慢 | 高 | 中等 |
| 自定义实现 | 低 | 快 | 低 | 高 |

## 推荐使用场景

1. **生产环境**: 使用PCL或Open3D
2. **快速原型**: 使用Open3D Python接口
3. **高质量要求**: 使用CGAL
4. **学习目的**: 使用自定义实现

## 故障排除

### PCL安装问题

**Ubuntu/Debian:**
```bash
# 如果找不到PCL
sudo apt-get install libpcl-dev libpcl-all-dev
```

**Windows:**
```bash
# 使用vcpkg
vcpkg install pcl:x64-windows
```

### 编译错误

如果遇到编译错误，检查：
1. PCL是否正确安装
2. CMake是否正确找到PCL
3. 编译器版本是否兼容

### 运行时错误

常见问题：
1. 点云数量不足（需要至少100个点）
2. 法线估计失败
3. 内存不足（对于大型点云）

## 总结

推荐使用PCL进行Poisson重建，因为它：
- 实现了真正的Poisson算法
- 有良好的文档和社区支持
- 性能和质量都很好
- 易于集成到现有项目中

如果PCL不可用，Open3D是一个很好的替代选择，特别是对于Python用户。 