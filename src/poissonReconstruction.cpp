#include "config.h"
#include "poissonReconstruction.h"
#include <opencv2/flann.hpp>
#include <omp.h>
#include <iostream>
#include <limits>
#include <cmath>
#include <algorithm>
#include <vector>
#include <map>
#include <chrono>
#include <numeric>

// PCL includes (需要安装PCL库)
#ifdef HAVE_PCL
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl/surface/poisson.h>
#include <pcl/io/ply_io.h>
#include <pcl/io/obj_io.h>
#include <pcl/common/io.h>
#include <pcl/common/transforms.h>
#endif

namespace MeshReconstruction {

// ==================== 参数配置接口 ====================
// void setPoissonReconstructionParams(float voxelSize, int triangulationStep, float depthThreshold) { ... }
// void setAdvancedPoissonParams(int maxGridSize, int normalNeighbors, int sdfNeighbors) { ... }
// void setPCLPoissonParams(int depth, int solverDivide, float samplesPerNode,
//                         bool confidence, bool manifold, bool outputPolygons) { ... }
// void resetPoissonReconstructionParams() { ... }
// void setDebugPoissonParams() { ... }
// void setFastPoissonParams() { ... }

// ==================== PCL辅助函数 ====================

#ifdef HAVE_PCL

// 检查PCL是否可用
bool isPCLAvailable() {
    return true;
}

// 转换OpenCV Mat到PCL点云
pcl::PointCloud<pcl::PointNormal>::Ptr convertToPCLPointCloud(
    const std::vector<PointWithNormal>& pointsWithNormals) {
    pcl::PointCloud<pcl::PointNormal>::Ptr cloud(new pcl::PointCloud<pcl::PointNormal>);
    cloud->reserve(pointsWithNormals.size());
    for (const auto& pwn : pointsWithNormals) {
        pcl::PointNormal point;
        point.x = pwn.point.x;
        point.y = pwn.point.y;
        point.z = pwn.point.z;
        point.normal_x = pwn.normal.nx;
        point.normal_y = pwn.normal.ny;
        point.normal_z = pwn.normal.nz;
        cloud->push_back(point);
    }
    return cloud;
}

// 转换PCL网格到我们的Mesh格式，并采样颜色
Mesh convertFromPCLMesh(const pcl::PolygonMesh& pclMesh, const cv::Mat& colorImage, const cv::Mat& K) {
    Mesh mesh;
    pcl::PointCloud<pcl::PointXYZ> vertices;
    pcl::fromPCLPointCloud2(pclMesh.cloud, vertices);
    // 相机内参
    double fx = K.at<double>(0, 0);
    double fy = K.at<double>(1, 1);
    double cx = K.at<double>(0, 2);
    double cy = K.at<double>(1, 2);
    for (const auto& vertex : vertices) {
        // 投影到像素坐标
        int u = static_cast<int>(std::round(vertex.x * fx / vertex.z + cx));
        int v = static_cast<int>(std::round(vertex.y * fy / vertex.z + cy));
        uint8_t r = 128, g = 128, b = 128;
        if (vertex.z > 0 && u >= 0 && v >= 0 && u < colorImage.cols && v < colorImage.rows) {
            cv::Vec3b color = colorImage.at<cv::Vec3b>(v, u);
            b = color[0]; g = color[1]; r = color[2];
        }
        mesh.vertices.emplace_back(vertex.x, vertex.y, vertex.z, r, g, b, u, v);
    }
    for (const auto& polygon : pclMesh.polygons) {
        if (polygon.vertices.size() >= 3) {
            Triangle triangle(polygon.vertices[0], polygon.vertices[1], polygon.vertices[2]);
            mesh.faces.push_back(triangle);
            if (polygon.vertices.size() == 4) {
                Triangle triangle2(polygon.vertices[0], polygon.vertices[2], polygon.vertices[3]);
                mesh.faces.push_back(triangle2);
            }
        }
    }
    return mesh;
}

#else

// 如果PCL不可用，提供空实现
bool isPCLAvailable() {
    return false;
}

#endif

// ==================== 辅助函数实现 ====================

cv::Point3f interpolateVertex(cv::Point3f p1, cv::Point3f p2, float val1, float val2, float isoValue = 0.0f) {
    if (std::abs(val1 - val2) < 1e-6f) return p1;
    float mu = (isoValue - val1) / (val2 - val1);
    mu = std::max(0.0f, std::min(1.0f, mu));
    return cv::Point3f(
        p1.x + mu * (p2.x - p1.x),
        p1.y + mu * (p2.y - p1.y),
        p1.z + mu * (p2.z - p1.z)
    );
}

// 计算加权SDF的辅助函数 - 优化版本
float calculateWeightedSDF(float worldX, float worldY, float worldZ, 
                          const std::vector<PointWithNormal>& pointsWithNormals,
                          int numNeighbors = 4) {
    
    float weightedSDF = 0.0f;
    float totalWeight = 0.0f;
    float sigma = Config::instance().meshParams.voxelSize * 2.0f;
    
    // 使用简化的最近邻搜索 - 只检查距离阈值内的点
    float searchRadius = Config::instance().meshParams.voxelSize * 5.0f; // 搜索半径
    float searchRadius2 = searchRadius * searchRadius;
    
    std::vector<std::pair<float, int>> candidates;
    candidates.reserve(numNeighbors * 4);
    
    // 快速预筛选：只检查可能在搜索半径内的点
    for (size_t i = 0; i < pointsWithNormals.size(); i += 1) { // 跳跃采样减少计算
        const auto& point = pointsWithNormals[i];
        float dx = worldX - point.point.x;
        float dy = worldY - point.point.y;
        float dz = worldZ - point.point.z;
        float distance2 = dx*dx + dy*dy + dz*dz;
        
        if (distance2 < searchRadius2) {
            candidates.push_back({sqrt(distance2), i});
        }
        
        if (candidates.size() >= numNeighbors * 3) break; // 早期终止
    }
    
    // 如果候选点不够，扩大搜索范围
    if (candidates.size() < numNeighbors) {
        for (size_t i = 0; i < pointsWithNormals.size(); i += 5) {
            const auto& point = pointsWithNormals[i];
            float dx = worldX - point.point.x;
            float dy = worldY - point.point.y;
            float dz = worldZ - point.point.z;
            float distance2 = dx*dx + dy*dy + dz*dz;
            
            candidates.push_back({sqrt(distance2), i});
            
            if (candidates.size() >= numNeighbors * 2) break;
        }
    }
    
    if (candidates.empty()) return 0.0f;
    
    // 排序并选择最近的邻居
    std::sort(candidates.begin(), candidates.end());
    int useNeighbors = std::min((int)candidates.size(), numNeighbors);
    
    for (int i = 0; i < useNeighbors; i++) {
        const auto& point = pointsWithNormals[candidates[i].second];
        
        float dx = worldX - point.point.x;
        float dy = worldY - point.point.y;
        float dz = worldZ - point.point.z;
        
        float distance = candidates[i].first;
        float signedDist = dx * point.normal.nx + dy * point.normal.ny + dz * point.normal.nz;
        
        // 使用高斯权重
        float weight = exp(-distance * distance / (2.0f * sigma * sigma));
        
        weightedSDF += weight * signedDist;
        totalWeight += weight;
    }
    
    return totalWeight > 1e-6f ? weightedSDF / totalWeight : 0.0f;
}

// ==================== 接口实现：从深度图估计法线 ====================
std::vector<PointWithNormal> estimateNormalsFromDepth(const cv::Mat& depthMap, const cv::Mat& K) {
    std::vector<PointWithNormal> pointsWithNormals;
    
    // Get camera intrinsics
    double fx = K.at<double>(0, 0);
    double fy = K.at<double>(1, 1);
    double cx = K.at<double>(0, 2);
    double cy = K.at<double>(1, 2);
    
    // Check depth map type and convert
    cv::Mat processedDepth;
    if (depthMap.type() == CV_16U) {
        depthMap.convertTo(processedDepth, CV_32F, 1.0/1000.0);
    } else if (depthMap.type() == CV_16S) {
        depthMap.convertTo(processedDepth, CV_32F, 1.0/16.0);
    } else if (depthMap.type() == CV_32F) {
        processedDepth = depthMap.clone();
    } else {
        std::cerr << "Warning: Unsupported depth map type " << depthMap.type() 
                  << " for normal estimation, trying direct conversion" << std::endl;
        depthMap.convertTo(processedDepth, CV_32F);
    }
    
    std::cout << "[INFO] Estimating normals from depth map..." << std::endl;
    std::cout << "[DEBUG] Input depth map: " << depthMap.rows << "x" << depthMap.cols 
              << ", type=" << depthMap.type() << std::endl;
    std::cout << "[DEBUG] Processed depth map type: " << processedDepth.type() << std::endl;
    
    // 检查深度值范围
    double minVal, maxVal;
    cv::minMaxLoc(processedDepth, &minVal, &maxVal);
    std::cout << "[DEBUG] Depth value range: [" << minVal << ", " << maxVal << "]" << std::endl;
    std::cout << "[DEBUG] Depth threshold: " << Config::instance().meshParams.depthThreshold << std::endl;
    
    // Use step size for efficiency
    int step = Config::instance().meshParams.triangulationStep;
    
    // 预分配内存
    int expectedPoints = ((processedDepth.rows - 2*step) / step) * ((processedDepth.cols - 2*step) / step);
    pointsWithNormals.reserve(expectedPoints);
    std::cout << "[DEBUG] Expected points: " << expectedPoints 
              << " (step=" << step << ")" << std::endl;
    
    int validDepthCount = 0;
    int validNeighborCount = 0;
    int totalProcessed = 0;
    
    #pragma omp parallel for reduction(+:validDepthCount,validNeighborCount,totalProcessed)
    for (int y = step; y < processedDepth.rows - step; y += step) {
        std::vector<PointWithNormal> localPoints; // 线程局部存储
        
        for (int x = step; x < processedDepth.cols - step; x += step) {
            totalProcessed++;
            float centerDepth = processedDepth.at<float>(y, x);
            
            // Filter invalid depth values
            if (centerDepth <= 0 || centerDepth > Config::instance().meshParams.depthThreshold || 
                std::isnan(centerDepth) || std::isinf(centerDepth)) {
                continue;
            }
            validDepthCount++;
            
            // Get neighboring depths with bounds checking
            float leftDepth = processedDepth.at<float>(y, x - step);
            float rightDepth = processedDepth.at<float>(y, x + step);
            float topDepth = processedDepth.at<float>(y - step, x);
            float bottomDepth = processedDepth.at<float>(y + step, x);
            
            // Check if neighbors are valid
            if (leftDepth <= 0 || rightDepth <= 0 || topDepth <= 0 || bottomDepth <= 0 ||
                std::isnan(leftDepth) || std::isnan(rightDepth) || std::isnan(topDepth) || std::isnan(bottomDepth)) {
                continue;
            }
            validNeighborCount++;
            
            // Additional neighbor validation for better normal estimation
            float depthDiffThreshold = centerDepth * Config::instance().meshParams.depthDiffThreshold;
            if (std::abs(leftDepth - centerDepth) > depthDiffThreshold ||
                std::abs(rightDepth - centerDepth) > depthDiffThreshold ||
                std::abs(topDepth - centerDepth) > depthDiffThreshold ||
                std::abs(bottomDepth - centerDepth) > depthDiffThreshold) {
                continue; // Skip points with large depth discontinuities
            }
            
            // Back-project center point to 3D
            float worldX = (x - cx) * centerDepth / fx;
            float worldY = (y - cy) * centerDepth / fy;
            float worldZ = centerDepth;
            
            Point3D point(worldX, worldY, worldZ, 0, 0, 0, x, y);
            
            // Improved normal calculation using cross product method
            float leftX = (x - step - cx) * leftDepth / fx;
            float leftY = (y - cy) * leftDepth / fy;
            
            float rightX = (x + step - cx) * rightDepth / fx;
            float rightY = (y - cy) * rightDepth / fy;
            
            float topX = (x - cx) * topDepth / fx;
            float topY = (y - step - cy) * topDepth / fy;
            
            float bottomX = (x - cx) * bottomDepth / fx;
            float bottomY = (y + step - cy) * bottomDepth / fy;
            
            // Calculate tangent vectors
            cv::Point3f tangentU(rightX - leftX, rightY - leftY, rightDepth - leftDepth);
            cv::Point3f tangentV(bottomX - topX, bottomY - topY, bottomDepth - topDepth);
            
            // Cross product to get normal
            cv::Point3f normal = tangentU.cross(tangentV);
            
            // Normalize
            float length = sqrt(normal.x*normal.x + normal.y*normal.y + normal.z*normal.z);
            if (length > 1e-6f) {
                normal.x /= length;
                normal.y /= length;
                normal.z /= length;
            } else {
                normal = cv::Point3f(0, 0, 1); // Default normal
            }
            
            // Ensure normal points towards camera
            if (normal.z < 0) {
                normal.x = -normal.x;
                normal.y = -normal.y;
                normal.z = -normal.z;
            }
            
            Normal3D normalVec(normal.x, normal.y, normal.z);
            localPoints.push_back(PointWithNormal(point, normalVec));
        }
        
        // 合并线程局部结果
        #pragma omp critical
        {
            pointsWithNormals.insert(pointsWithNormals.end(), localPoints.begin(), localPoints.end());
        }
    }
    
    std::cout << "[DEBUG] Processing statistics:" << std::endl;
    std::cout << "  Total pixels processed: " << totalProcessed << std::endl;
    std::cout << "  Valid depth pixels: " << validDepthCount << std::endl;
    std::cout << "  Valid neighbor pixels: " << validNeighborCount << std::endl;
    std::cout << "[INFO] Estimated normals for " << pointsWithNormals.size() << " points" << std::endl;
    
    return pointsWithNormals;
}

// ==================== 接口实现：从3D点估计法线 ====================
std::vector<PointWithNormal> estimateNormals(const std::vector<Point3D>& points, int kNeighbors) {
    std::vector<PointWithNormal> pointsWithNormals;
    
    if (points.size() < kNeighbors + 1) {
        std::cerr << "Error: Not enough points for normal estimation" << std::endl;
        return pointsWithNormals;
    }
    
    std::cout << "[INFO] Estimating normals using optimized PCA (k=" << kNeighbors << ")..." << std::endl;
    
    // 构建KD-tree
    cv::Mat pointsMat(points.size(), 3, CV_32F);
    for (size_t i = 0; i < points.size(); i++) {
        pointsMat.at<float>(i, 0) = points[i].x;
        pointsMat.at<float>(i, 1) = points[i].y;
        pointsMat.at<float>(i, 2) = points[i].z;
    }
    
    cv::flann::KDTreeIndexParams indexParams(5);
    cv::flann::Index kdtree(pointsMat, indexParams);
    
    pointsWithNormals.resize(points.size());
    
    #pragma omp parallel for
    for (int i = 0; i < points.size(); i++) {
        const Point3D& centerPoint = points[i];
        
        // K最近邻搜索
        float queryArr[3] = {centerPoint.x, centerPoint.y, centerPoint.z};
        cv::Mat queryMat(1, 3, CV_32F, queryArr);
        std::vector<int> indices(kNeighbors + 1);
        std::vector<float> dists(kNeighbors + 1);
        kdtree.knnSearch(queryMat, indices, dists, kNeighbors + 1);
        
        // 检查有效邻居数量
        int validIndices = 0;
        for (int j = 0; j < indices.size(); j++) {
            if (indices[j] >= 0 && indices[j] < points.size()) {
                validIndices++;
            }
        }
        
        if (validIndices < 4) {
            Normal3D normal(0, 0, 1);
            pointsWithNormals[i] = PointWithNormal(centerPoint, normal);
            continue;
        }
        
        // 计算质心（跳过第一个点，因为是查询点本身）
        cv::Point3d centroid(0, 0, 0);
        int validNeighbors = 0;
        for (int j = 1; j < indices.size() && validNeighbors < kNeighbors; j++) {
            if (indices[j] >= 0 && indices[j] < points.size()) {
                centroid.x += points[indices[j]].x;
                centroid.y += points[indices[j]].y;
                centroid.z += points[indices[j]].z;
                validNeighbors++;
            }
        }
        
        if (validNeighbors == 0) {
            Normal3D normal(0, 0, 1);
            pointsWithNormals[i] = PointWithNormal(centerPoint, normal);
            continue;
        }
        
        centroid.x /= validNeighbors;
        centroid.y /= validNeighbors;
        centroid.z /= validNeighbors;
        
        // 构建协方差矩阵
        cv::Mat covariance = cv::Mat::zeros(3, 3, CV_64F);
        for (int j = 1; j < indices.size() && (j-1) < validNeighbors; j++) {
            if (indices[j] >= 0 && indices[j] < points.size()) {
                double dx = points[indices[j]].x - centroid.x;
                double dy = points[indices[j]].y - centroid.y;
                double dz = points[indices[j]].z - centroid.z;
                
                covariance.at<double>(0, 0) += dx * dx;
                covariance.at<double>(0, 1) += dx * dy;
                covariance.at<double>(0, 2) += dx * dz;
                covariance.at<double>(1, 0) += dy * dx;
                covariance.at<double>(1, 1) += dy * dy;
                covariance.at<double>(1, 2) += dy * dz;
                covariance.at<double>(2, 0) += dz * dx;
                covariance.at<double>(2, 1) += dz * dy;
                covariance.at<double>(2, 2) += dz * dz;
            }
        }
        covariance /= validNeighbors;
        
        // 特征值分解
        cv::Mat eigenValues, eigenVectors;
        bool success = cv::eigen(covariance, eigenValues, eigenVectors);
        
        if (!success || eigenVectors.rows != 3 || eigenVectors.cols != 3) {
            Normal3D normal(0, 0, 1);
            pointsWithNormals[i] = PointWithNormal(centerPoint, normal);
            continue;
        }
        
        // 最小特征值对应的特征向量（最后一行）
        double nx = eigenVectors.at<double>(2, 0);
        double ny = eigenVectors.at<double>(2, 1);
        double nz = eigenVectors.at<double>(2, 2);
        
        Normal3D normal(nx, ny, nz);
        normal.normalize();
        
        // 确保法向量朝向观察者
        if (normal.nz < 0) {
            normal.nx = -normal.nx;
            normal.ny = -normal.ny;
            normal.nz = -normal.nz;
        }
        
        pointsWithNormals[i] = PointWithNormal(centerPoint, normal);
    }
    
    std::cout << "[INFO] Estimated normals for " << pointsWithNormals.size() << " points" << std::endl;
    return pointsWithNormals;
}

// ==================== 接口实现：主Poisson重建函数 ====================
Mesh generatePoissonMesh(const cv::Mat& depthMap, const cv::Mat& colorImage, const cv::Mat& K) {
    Mesh mesh;
    std::cout << "[INFO] Starting Poisson surface reconstruction..." << std::endl;
    std::vector<PointWithNormal> pointsWithNormals = estimateNormalsFromDepth(depthMap, K);
    if (pointsWithNormals.size() < 100) {
        std::cerr << "Error: Not enough points with normals for Poisson reconstruction" << std::endl;
        return mesh;
    }
    if (pointsWithNormals.size() > 1000) {
        std::cout << "[INFO] Refining normals using 3D neighborhood analysis..." << std::endl;
        std::vector<Point3D> points3D;
        points3D.reserve(pointsWithNormals.size());
        for (const auto& pwn : pointsWithNormals) {
            points3D.push_back(pwn.point);
        }
        std::vector<PointWithNormal> refinedNormals = estimateNormals(points3D, Config::instance().meshParams.normalNeighbors);
        if (refinedNormals.size() == pointsWithNormals.size()) {
            pointsWithNormals = refinedNormals;
            std::cout << "[INFO] Normal refinement completed" << std::endl;
        }
    }
    std::cout << "[INFO] Using " << pointsWithNormals.size() << " points for reconstruction" << std::endl;
#ifdef HAVE_PCL
    if (isPCLAvailable()) {
        std::cout << "[INFO] Using PCL Poisson surface reconstruction..." << std::endl;
        try {
            pcl::PointCloud<pcl::PointNormal>::Ptr cloud_with_normals = 
                convertToPCLPointCloud(pointsWithNormals);
            pcl::Poisson<pcl::PointNormal> poisson;
            poisson.setInputCloud(cloud_with_normals);
            poisson.setDepth(Config::instance().meshParams.poissonDepth);
            poisson.setSolverDivide(Config::instance().meshParams.poissonSolverDivide);
            poisson.setSamplesPerNode(Config::instance().meshParams.poissonSamplesPerNode);
            poisson.setConfidence(Config::instance().meshParams.poissonUseConfidence);
            poisson.setManifold(Config::instance().meshParams.poissonManifold);
            poisson.setOutputPolygons(Config::instance().meshParams.poissonOutputPolygons);
            std::cout << "[INFO] PCL Poisson parameters:" << std::endl;
            std::cout << "  Depth: " << Config::instance().meshParams.poissonDepth << std::endl;
            std::cout << "  Solver divide: " << Config::instance().meshParams.poissonSolverDivide << std::endl;
            std::cout << "  Samples per node: " << Config::instance().meshParams.poissonSamplesPerNode << std::endl;
            pcl::PolygonMesh pclMesh;
            poisson.reconstruct(pclMesh);
            mesh = convertFromPCLMesh(pclMesh, colorImage, K);
            std::cout << "[INFO] PCL Poisson reconstruction completed" << std::endl;
            std::cout << "[INFO] Generated mesh with " << mesh.vertices.size() 
                      << " vertices and " << mesh.faces.size() << " faces" << std::endl;
        } catch (const std::exception& e) {
            std::cerr << "Error during PCL Poisson reconstruction: " << e.what() << std::endl;
        }
    } else {
        std::cout << "[INFO] PCL not available, cannot perform Poisson reconstruction." << std::endl;
    }
#else
    std::cout << "[INFO] PCL not compiled, cannot perform Poisson reconstruction." << std::endl;
#endif
    return mesh;
}

} // namespace MeshReconstruction