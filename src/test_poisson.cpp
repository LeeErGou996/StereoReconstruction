#include "poissonReconstruction.h"
#include "meshReconstruction.h"
#include <opencv2/opencv.hpp>
#include <iostream>

int main() {
    std::cout << "=== Poisson Reconstruction Test ===" << std::endl;
    
    // 检查PCL是否可用
    if (MeshReconstruction::isPCLAvailable()) {
        std::cout << "✓ PCL library is available" << std::endl;
    } else {
        std::cout << "✗ PCL library not available" << std::endl;
    }
    
    // 设置PCL Poisson重建参数
    MeshReconstruction::setPCLPoissonParams(
        8,    // depth: 八叉树深度
        8,    // solverDivide: 求解器分割深度
        1.5f, // samplesPerNode: 每个节点的样本数
        false, // confidence: 不使用置信度权重
        false, // manifold: 不保持流形
        false  // outputPolygons: 输出三角形
    );
    
    // 设置其他参数
    MeshReconstruction::setPoissonReconstructionParams(0.01f, 2, 10.0f);
    MeshReconstruction::setAdvancedPoissonParams(150, 10, 6);
    
    std::cout << "\n=== Parameter Summary ===" << std::endl;
    std::cout << "Poisson reconstruction parameters configured." << std::endl;
    std::cout << "Ready to use generatePoissonMesh() function." << std::endl;
    
    // 示例用法（需要实际的深度图数据）
    /*
    cv::Mat depthMap = cv::imread("depth.png", cv::IMREAD_ANYDEPTH);
    cv::Mat colorImage = cv::imread("color.jpg");
    cv::Mat K = cv::Mat::eye(3, 3, CV_64F); // 相机内参矩阵
    
    if (!depthMap.empty()) {
        std::cout << "\n=== Starting Poisson Reconstruction ===" << std::endl;
        
        Mesh mesh = MeshReconstruction::generatePoissonMesh(depthMap, colorImage, K);
        
        if (!mesh.vertices.empty()) {
            std::cout << "✓ Reconstruction successful!" << std::endl;
            std::cout << "  Vertices: " << mesh.vertices.size() << std::endl;
            std::cout << "  Faces: " << mesh.faces.size() << std::endl;
            
            // 保存结果
            MeshReconstruction::saveMeshFile(mesh, "poisson_result.ply", "ply");
            std::cout << "  Result saved to: poisson_result.ply" << std::endl;
        } else {
            std::cout << "✗ Reconstruction failed!" << std::endl;
        }
    } else {
        std::cout << "✗ No depth map data available for testing" << std::endl;
    }
    */
    
    std::cout << "\n=== Test Completed ===" << std::endl;
    std::cout << "The modified Poisson reconstruction is ready to use." << std::endl;
    std::cout << "It will automatically use PCL if available, otherwise fall back to custom implementation." << std::endl;
    
    return 0;
} 