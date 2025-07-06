# 立体视觉系统流程图 - 程序依赖关系与变量传递

## 系统架构概览

```mermaid
graph TB
    subgraph "输入层"
        A[图像文件] --> B[相机参数文件]
        A --> C[配置文件]
    end
    
    subgraph "核心处理流程"
        D[图像读取] --> E[特征检测匹配]
        E --> F[姿态估计]
        F --> G[立体校正]
        G --> H[视差计算]
        H --> I[深度计算]
        I --> J[网格重建]
    end
    
    subgraph "输出层"
        K[视差图] --> L[深度图]
        L --> M[3D网格]
        M --> N[点云文件]
    end
    
    B --> E
    C --> E
    C --> G
    C --> H
    C --> J
```

## 详细程序依赖关系图

```mermaid
graph TD
    subgraph "配置管理"
        CONFIG[config.cpp] --> CONFIG_H[config.h]
        CONFIG --> CONFIG_TXT[config.txt]
    end
    
    subgraph "图像输入"
        MAIN[main.cpp] --> IMG_L[左图像]
        MAIN --> IMG_R[右图像]
        MAIN --> CAMERA[相机参数]
    end
    
    subgraph "特征处理模块"
        DISPARITY[disparity.cpp] --> DISPARITY_H[disparity.h]
        DISPARITY --> FEATURE_DETECTOR[特征检测器]
        DISPARITY --> FEATURE_MATCHER[特征匹配器]
    end
    
    subgraph "几何计算模块"
        EIGHTPOINT[8point.cpp] --> EIGHTPOINT_H[8point.h]
        DENSE_MATCHING[denseMatching.cpp] --> DENSE_MATCHING_H[denseMatching.h]
        SPARSE_MATCHING[sparseMatching.cpp] --> SPARSE_MATCHING_H[sparseMatching.h]
    end
    
    subgraph "深度处理模块"
        DEPTH[depth.cpp] --> DEPTH_H[depth.h]
    end
    
    subgraph "重建模块"
        MESH_RECON[meshReconstruction.cpp] --> MESH_RECON_H[meshReconstruction.h]
        POISSON[poissonReconstruction.cpp] --> POISSON_H[poissonReconstruction.h]
    end
    
    subgraph "辅助模块"
        COLOR_UTILS[colorUtils.cpp]
        POINT_CLOUD_FILTER[pointCloudFilter.cpp] --> POINT_CLOUD_FILTER_H[pointCloudFilter.h]
    end
    
    %% 主要数据流
    MAIN --> DISPARITY
    MAIN --> EIGHTPOINT
    MAIN --> DENSE_MATCHING
    MAIN --> SPARSE_MATCHING
    MAIN --> DEPTH
    MAIN --> MESH_RECON
    
    %% 配置依赖
    CONFIG --> MAIN
    CONFIG --> DISPARITY
    CONFIG --> DENSE_MATCHING
    CONFIG --> SPARSE_MATCHING
    CONFIG --> MESH_RECON
    
    %% 模块间依赖
    DISPARITY --> EIGHTPOINT
    EIGHTPOINT --> DENSE_MATCHING
    DENSE_MATCHING --> DEPTH
    DEPTH --> MESH_RECON
    MESH_RECON --> POISSON
    
    %% 样式定义
    classDef activeModule fill:#90EE90,stroke:#333,stroke-width:2px
    classDef inactiveModule fill:#FFB6C1,stroke:#333,stroke-width:1px
    classDef configModule fill:#87CEEB,stroke:#333,stroke-width:2px
    classDef dataModule fill:#F0E68C,stroke:#333,stroke-width:1px
    
    class MAIN,DISPARITY,EIGHTPOINT,DENSE_MATCHING,DEPTH,MESH_RECON activeModule
    class COLOR_UTILS,POINT_CLOUD_FILTER inactiveModule
    class CONFIG,CONFIG_H,CONFIG_TXT configModule
    class IMG_L,IMG_R,CAMERA dataModule
```

## 变量传递关系图

```mermaid
graph LR
    subgraph "输入变量"
        INPUT_IMG_L[imgL: cv::Mat] 
        INPUT_IMG_R[imgR: cv::Mat]
        INPUT_IMG_L_COLOR[imgL_color: cv::Mat]
        INPUT_IMG_R_COLOR[imgR_color: cv::Mat]
        CAMERA_K[K: cv::Mat]
        CAMERA_DIST[distCoeffs: cv::Mat]
        CONFIG_PARAMS[Config::instance()]
    end
    
    subgraph "特征处理阶段"
        FEATURE_PTS_L[ptsL: vector<Point2f>]
        FEATURE_PTS_R[ptsR: vector<Point2f>]
        FEATURE_DETECTOR[detector: Ptr<Feature2D>]
    end
    
    subgraph "几何计算阶段"
        POSE_R[R: cv::Mat]
        POSE_T[t: cv::Mat]
        RECT_L[rectL: cv::Mat]
        RECT_R[rectR: cv::Mat]
        RECT_L_COLOR[rectL_color: cv::Mat]
        RECT_R_COLOR[rectR_color: cv::Mat]
    end
    
    subgraph "视差深度阶段"
        DISPARITY_MAP[disparity: cv::Mat]
        Q_MATRIX[Q_matrix: cv::Mat]
        DEPTH_MAP[depthMap: cv::Mat]
    end
    
    subgraph "重建阶段"
        MESH_RESULT[mesh: MeshReconstruction::Mesh]
    end
    
    %% 变量传递关系
    INPUT_IMG_L --> FEATURE_DETECTOR
    INPUT_IMG_R --> FEATURE_DETECTOR
    FEATURE_DETECTOR --> FEATURE_PTS_L
    FEATURE_DETECTOR --> FEATURE_PTS_R
    
    FEATURE_PTS_L --> POSE_R
    FEATURE_PTS_R --> POSE_R
    CAMERA_K --> POSE_R
    POSE_R --> POSE_T
    
    INPUT_IMG_L --> RECT_L
    INPUT_IMG_R --> RECT_R
    POSE_R --> RECT_L
    POSE_T --> RECT_L
    POSE_R --> RECT_R
    POSE_T --> RECT_R
    CAMERA_K --> RECT_L
    CAMERA_K --> RECT_R
    
    INPUT_IMG_L_COLOR --> RECT_L_COLOR
    INPUT_IMG_R_COLOR --> RECT_R_COLOR
    POSE_R --> RECT_L_COLOR
    POSE_T --> RECT_L_COLOR
    POSE_R --> RECT_R_COLOR
    POSE_T --> RECT_R_COLOR
    
    RECT_L --> DISPARITY_MAP
    RECT_R --> DISPARITY_MAP
    CONFIG_PARAMS --> DISPARITY_MAP
    
    DISPARITY_MAP --> DEPTH_MAP
    Q_MATRIX --> DEPTH_MAP
    
    DEPTH_MAP --> MESH_RESULT
    RECT_L_COLOR --> MESH_RESULT
    CAMERA_K --> MESH_RESULT
    CONFIG_PARAMS --> MESH_RESULT
```

## 详细处理流程时序图

```mermaid
sequenceDiagram
    participant Main as main.cpp
    participant Config as config.cpp
    participant Disparity as disparity.cpp
    participant EightPoint as 8point.cpp
    participant DenseMatch as denseMatching.cpp
    participant SparseMatch as sparseMatching.cpp
    participant Depth as depth.cpp
    participant MeshRec as meshReconstruction.cpp
    participant Poisson as poissonReconstruction.cpp
    
    Main->>Config: Config::load("config.txt")
    Config-->>Main: 配置参数加载完成
    
    Main->>Main: 读取左右图像 (imgL, imgR, imgL_color, imgR_color)
    Main->>Main: 读取相机参数 (K, distCoeffs)
    
    Main->>Disparity: createDetector(Config::instance().algorithm)
    Disparity-->>Main: detector (Ptr<Feature2D>)
    
    Main->>Disparity: detectAndMatch(imgL, imgR, detector, ptsL, ptsR)
    Disparity-->>Main: 特征点对 (ptsL, ptsR)
    
    Main->>EightPoint: estimatePose(ptsL, ptsR, K, R, t)
    EightPoint-->>Main: 相对姿态 (R, t)
    
    Main->>DenseMatch: rectifyImages(imgL, imgR, R, t, rectL, rectR)
    DenseMatch-->>Main: 校正图像 (rectL, rectR)
    
    alt 匹配模式选择
        Main->>SparseMatch: computeDisparityMap(rectL, rectR, disparity)
        SparseMatch-->>Main: 稀疏视差图
    else
        Main->>DenseMatch: computeDisparityMap(rectL, rectR, disparity)
        DenseMatch-->>Main: 密集视差图
    end
    
    Main->>DenseMatch: getQMatrix()
    DenseMatch-->>Main: Q_matrix
    
    Main->>Depth: computeDepthMap(disparity, Q_matrix, depthMap)
    Depth-->>Main: 深度图 (depthMap)
    
    Main->>MeshRec: setReconstructionParams(Config::instance().meshParams)
    Main->>Config: generatePoissonMeshes(depthMap, rectL_color, K)
    Config->>MeshRec: generatePoissonMesh(depthMap, colorImage, K)
    MeshRec->>Poisson: 内部调用Poisson重建
    Poisson-->>MeshRec: 重建网格
    MeshRec-->>Config: Mesh对象
    Config-->>Main: 最终网格 (mesh)
    
    Main->>MeshRec: saveMeshFile(mesh, "mesh.ply", "ply")
    MeshRec-->>Main: 文件保存完成
```

## 模块功能与依赖关系表

| 模块 | 主要功能 | 输入变量 | 输出变量 | 依赖模块 | 状态 |
|------|----------|----------|----------|----------|------|
| **config.cpp** | 配置管理 | config.txt | Config::instance() | 无 | ✅ 活跃 |
| **disparity.cpp** | 特征检测匹配 | imgL, imgR, algorithm | ptsL, ptsR, detector | config | ✅ 活跃 |
| **8point.cpp** | 姿态估计 | ptsL, ptsR, K | R, t | disparity | ✅ 活跃 |
| **denseMatching.cpp** | 立体校正+密集匹配 | imgL, imgR, R, t, K | rectL, rectR, disparity, Q | 8point, config | ✅ 活跃 |
| **sparseMatching.cpp** | 稀疏匹配 | rectL, rectR | disparity | denseMatching | ✅ 活跃 |
| **depth.cpp** | 深度计算 | disparity, Q_matrix | depthMap | denseMatching | ✅ 活跃 |
| **meshReconstruction.cpp** | 网格重建 | depthMap, colorImage, K | mesh | depth, config | ✅ 活跃 |
| **poissonReconstruction.cpp** | Poisson重建 | depthMap, colorImage, K | mesh | meshReconstruction | ⚠️ 间接 |
| **pointCloudFilter.cpp** | 点云滤波 | pointCloud | filteredPointCloud | meshReconstruction | ❌ 未使用 |
| **colorUtils.cpp** | 颜色处理 | disparity, colorImage | colorDisparity | 无 | ❌ 未使用 |

## 数据流总结

### 主要数据流路径
1. **图像输入** → **特征检测** → **姿态估计** → **立体校正** → **视差计算** → **深度计算** → **网格重建**

### 关键变量传递链
- `imgL, imgR` → `ptsL, ptsR` → `R, t` → `rectL, rectR` → `disparity` → `depthMap` → `mesh`

### 配置参数影响
- `Config::instance()` 影响所有主要模块的参数设置
- 通过配置文件统一管理算法参数

### 未充分利用的模块
- `pointCloudFilter.cpp`: 可在深度计算后添加滤波步骤
- `colorUtils.cpp`: 可替代main.cpp中的重复实现 