#include "config.h"
#include "poissonReconstruction.h"
#include <iostream>

// ==================== 配置函数实现 ====================

void resetPoissonReconstructionParams() {
    Config::instance().meshParams.voxelSize = 0.005f;
    Config::instance().meshParams.triangulationStep = 2;
    Config::instance().meshParams.depthThreshold = 5.0f;
    Config::instance().meshParams.depthDiffThreshold = 0.1f;
    Config::instance().meshParams.normalNeighbors = 10;
    Config::instance().meshParams.sdfNeighbors = 8;
    
    Config::instance().meshParams.poissonDepth = 8;
    Config::instance().meshParams.poissonSolverDivide = 8;
    Config::instance().meshParams.poissonSamplesPerNode = 1.5f;
    Config::instance().meshParams.poissonUseConfidence = false;
    Config::instance().meshParams.poissonManifold = false;
    Config::instance().meshParams.poissonOutputPolygons = false;
    
    std::cout << "[CONFIG] 重置为默认Poisson重建参数" << std::endl;
}

void setDebugPoissonParams() {
    Config::instance().meshParams.voxelSize = 0.01f;           // 更大的体素
    Config::instance().meshParams.triangulationStep = 4;       // 更大的步长
    Config::instance().meshParams.depthThreshold = 10.0f;      // 更宽松的深度阈值
    Config::instance().meshParams.depthDiffThreshold = 0.2f;   // 更宽松的深度差异
    Config::instance().meshParams.normalNeighbors = 5;         // 更少的邻居
    Config::instance().meshParams.sdfNeighbors = 4;
    
    Config::instance().meshParams.poissonDepth = 6;            // 更浅的深度
    Config::instance().meshParams.poissonSolverDivide = 6;
    Config::instance().meshParams.poissonSamplesPerNode = 1.0f;
    Config::instance().meshParams.poissonUseConfidence = false;
    Config::instance().meshParams.poissonManifold = false;
    Config::instance().meshParams.poissonOutputPolygons = false;
    
    std::cout << "[CONFIG] 设置调试模式参数（快速但质量较低）" << std::endl;
}

void setFastPoissonParams() {
    Config::instance().meshParams.voxelSize = 0.008f;
    Config::instance().meshParams.triangulationStep = 3;
    Config::instance().meshParams.depthThreshold = 8.0f;
    Config::instance().meshParams.depthDiffThreshold = 0.15f;
    Config::instance().meshParams.normalNeighbors = 8;
    Config::instance().meshParams.sdfNeighbors = 6;
    
    Config::instance().meshParams.poissonDepth = 7;
    Config::instance().meshParams.poissonSolverDivide = 7;
    Config::instance().meshParams.poissonSamplesPerNode = 1.2f;
    Config::instance().meshParams.poissonUseConfidence = false;
    Config::instance().meshParams.poissonManifold = false;
    Config::instance().meshParams.poissonOutputPolygons = false;
    
    std::cout << "[CONFIG] 设置快速重建参数" << std::endl;
}

void setHighQualityPoissonParams() {
    Config::instance().meshParams.voxelSize = 0.003f;          // 更小的体素
    Config::instance().meshParams.triangulationStep = 1;       // 更小的步长
    Config::instance().meshParams.depthThreshold = 3.0f;       // 更严格的深度阈值
    Config::instance().meshParams.depthDiffThreshold = 0.05f;  // 更严格的深度差异
    Config::instance().meshParams.normalNeighbors = 15;        // 更多的邻居
    Config::instance().meshParams.sdfNeighbors = 12;
    
    Config::instance().meshParams.poissonDepth = 10;           // 更深的深度
    Config::instance().meshParams.poissonSolverDivide = 10;
    Config::instance().meshParams.poissonSamplesPerNode = 2.0f;
    Config::instance().meshParams.poissonUseConfidence = true;
    Config::instance().meshParams.poissonManifold = true;
    Config::instance().meshParams.poissonOutputPolygons = false;
    
    std::cout << "[CONFIG] 设置高质量重建参数（慢但质量较高）" << std::endl;
}

void setPoissonReconstructionParams(float voxelSize, int triangulationStep, float depthThreshold) {
    Config::instance().meshParams.voxelSize = voxelSize;
    Config::instance().meshParams.triangulationStep = triangulationStep;
    Config::instance().meshParams.depthThreshold = depthThreshold;
    
    std::cout << "[CONFIG] 设置基本Poisson重建参数:" << std::endl;
    std::cout << "  体素大小: " << voxelSize << std::endl;
    std::cout << "  三角化步长: " << triangulationStep << std::endl;
    std::cout << "  深度阈值: " << depthThreshold << std::endl;
}

void setAdvancedPoissonParams(int maxGridSize, int normalNeighbors, int sdfNeighbors) {
    Config::instance().meshParams.normalNeighbors = normalNeighbors;
    Config::instance().meshParams.sdfNeighbors = sdfNeighbors;
    
    std::cout << "[CONFIG] 设置高级Poisson参数:" << std::endl;
    std::cout << "  法向量邻居数: " << normalNeighbors << std::endl;
    std::cout << "  SDF邻居数: " << sdfNeighbors << std::endl;
}

#ifdef HAVE_PCL
void setPCLPoissonParams(int depth, int solverDivide, float samplesPerNode,
                        bool confidence, bool manifold, bool outputPolygons) {
    Config::instance().meshParams.poissonDepth = depth;
    Config::instance().meshParams.poissonSolverDivide = solverDivide;
    Config::instance().meshParams.poissonSamplesPerNode = samplesPerNode;
    Config::instance().meshParams.poissonUseConfidence = confidence;
    Config::instance().meshParams.poissonManifold = manifold;
    Config::instance().meshParams.poissonOutputPolygons = outputPolygons;
    
    std::cout << "[CONFIG] 设置PCL Poisson参数:" << std::endl;
    std::cout << "  深度: " << depth << std::endl;
    std::cout << "  求解器分割: " << solverDivide << std::endl;
    std::cout << "  每节点样本数: " << samplesPerNode << std::endl;
    std::cout << "  使用置信度: " << (confidence ? "是" : "否") << std::endl;
    std::cout << "  强制流形: " << (manifold ? "是" : "否") << std::endl;
    std::cout << "  输出多边形: " << (outputPolygons ? "是" : "否") << std::endl;
}
#endif

// ==================== 工具函数实现 ====================

void validateDepthMap(const MeshReconstruction::ImageData& depthMap, const std::string& context) {
    if (depthMap.rows <= 0 || depthMap.cols <= 0) {
        std::cerr << "[VALIDATION ERROR] " << context << ": 深度图尺寸无效 " 
                  << depthMap.rows << "x" << depthMap.cols << std::endl;
        return;
    }
    
    double minVal, maxVal;
    depthMap.minMaxLoc(&minVal, &maxVal);
    
    std::cout << "[VALIDATION] " << context << ":" << std::endl;
    std::cout << "  尺寸: " << depthMap.rows << "x" << depthMap.cols << std::endl;
    std::cout << "  类型: " << depthMap.type << std::endl;
    std::cout << "  深度范围: [" << minVal << ", " << maxVal << "]" << std::endl;
    
    if (maxVal <= 0) {
        std::cerr << "[VALIDATION WARNING] " << context << ": 没有有效的深度值" << std::endl;
    }
    
    int validPixels = 0;
    size_t totalPixels = depthMap.rows * depthMap.cols;
    
    if (depthMap.type == MeshReconstruction::ImageData::TYPE_32FC1) {
        for (size_t i = 0; i < totalPixels; i++) {
            if (depthMap.data_float[i] > 0 && std::isfinite(depthMap.data_float[i])) {
                validPixels++;
            }
        }
    }
    
    std::cout << "  有效像素: " << validPixels << "/" << totalPixels 
              << " (" << (100.0 * validPixels / totalPixels) << "%)" << std::endl;
}

void validateNormals(const std::vector<PointWithNormal>& normals, const std::string& context) {
    if (normals.empty()) {
        std::cerr << "[VALIDATION ERROR] " << context << ": 法向量为空" << std::endl;
        return;
    }
    
    int validNormals = 0;
    float avgLength = 0.0f;
    
    for (const auto& pwn : normals) {
        float length = sqrt(pwn.normal.nx * pwn.normal.nx + 
                           pwn.normal.ny * pwn.normal.ny + 
                           pwn.normal.nz * pwn.normal.nz);
        if (length > 0.1f && length < 2.0f) {  // 合理的法向量长度
            validNormals++;
        }
        avgLength += length;
    }
    
    avgLength /= normals.size();
    
    std::cout << "[VALIDATION] " << context << ":" << std::endl;
    std::cout << "  法向量数量: " << normals.size() << std::endl;
    std::cout << "  有效法向量: " << validNormals << " (" 
              << (100.0 * validNormals / normals.size()) << "%)" << std::endl;
    std::cout << "  平均长度: " << avgLength << std::endl;
}

void printImageDataInfo(const MeshReconstruction::ImageData& img, const std::string& name) {
    std::cout << "[INFO] " << name << ":" << std::endl;
    std::cout << "  尺寸: " << img.rows << "x" << img.cols << std::endl;
    std::cout << "  通道数: " << img.channels << std::endl;
    std::cout << "  类型: " << img.type << std::endl;
    
    size_t dataSize = 0;
    switch(img.type) {
        case MeshReconstruction::ImageData::TYPE_8UC1:
        case MeshReconstruction::ImageData::TYPE_8UC3:
            dataSize = img.data_uint8.size();
            break;
        case MeshReconstruction::ImageData::TYPE_16UC1:
        case MeshReconstruction::ImageData::TYPE_16SC1:
            dataSize = img.data_uint16.size() * 2;
            break;
        case MeshReconstruction::ImageData::TYPE_32FC1:
            dataSize = img.data_float.size() * 4;
            break;
        case MeshReconstruction::ImageData::TYPE_64FC1:
            dataSize = img.data_double.size() * 8;
            break;
    }
    
    std::cout << "  数据大小: " << (dataSize / 1024.0 / 1024.0) << " MB" << std::endl;
}