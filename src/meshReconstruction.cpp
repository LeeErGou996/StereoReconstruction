#include "meshReconstruction.h"
#include "poissonReconstruction.h"
#include <iostream>
#include <fstream>
#include <vector>
#include <algorithm>
#include <cmath>
#include <opencv2/imgproc.hpp>
#include <map>
#include <unordered_map>
#include <numeric>
#include <filesystem>

namespace MeshReconstruction {

// Global parameters
static ReconstructionParams g_params;

void setReconstructionParams(const ReconstructionParams& params) {
    g_params = params;
    std::cout << "[INFO] Mesh reconstruction parameters updated" << std::endl;
    std::cout << "[INFO] Reconstruction mode: " << (g_params.reconstructionMode == 0 ? "Point Cloud" : g_params.reconstructionMode == 1 ? "Triangulated Mesh" : "Poisson Surface Reconstruction") << std::endl;
}

bool reconstructAndSaveMesh(const cv::Mat& depthMap, 
                           const cv::Mat& colorImage, 
                           const std::string& outputPath) {
    
    if (depthMap.empty()) {
        std::cerr << "Error: Depth map is empty" << std::endl;
        return false;
    }
    
    // Create output directory
    std::filesystem::path outputDir(outputPath);
    std::filesystem::create_directories(outputDir);
    
    // Get camera intrinsics (assuming standard camera model)
    cv::Mat K = (cv::Mat_<double>(3,3) << 1758.23, 0, 953.34, 0, 1758.23, 552.29, 0, 0, 1);
    
    bool success = false;
    
    if (g_params.reconstructionMode == 0) {
        // Point cloud mode
        std::cout << "[INFO] Generating point cloud..." << std::endl;
        std::vector<Point3D> points = generatePointCloud(depthMap, colorImage, K);
        
        if (!points.empty()) {
            std::string pointCloudPath = outputPath + "/pointcloud.ply";
            success = savePointCloudPLY(points, pointCloudPath);
            if (success) {
                std::cout << "[INFO] Point cloud saved to: " << pointCloudPath << std::endl;
            }
        } else {
            std::cerr << "Error: Failed to generate point cloud" << std::endl;
        }
        
    } else if (g_params.reconstructionMode == 1) {
        // Triangulated mesh mode
        std::cout << "[INFO] Generating triangulated mesh..." << std::endl;
        Mesh mesh = generateTriangulatedMesh(depthMap, colorImage, K);
        
        if (!mesh.vertices.empty() && !mesh.faces.empty()) {
            std::string meshPath = outputPath + "/mesh." + g_params.meshFormat;
            success = saveMeshFile(mesh, meshPath, g_params.meshFormat);
            if (success) {
                std::cout << "[INFO] Triangulated mesh saved to: " << meshPath << std::endl;
            }
        } else {
            std::cerr << "Error: Failed to generate triangulated mesh" << std::endl;
        }
        
    } else if (g_params.reconstructionMode == 2) {
        // Poisson surface reconstruction mode
        std::cout << "[INFO] Generating Poisson surface reconstruction..." << std::endl;
        Mesh mesh = generatePoissonMesh(depthMap, colorImage, K);
        
        if (!mesh.vertices.empty() && !mesh.faces.empty()) {
            std::string meshPath = outputPath + "/poisson_mesh." + g_params.meshFormat;
            success = saveMeshFile(mesh, meshPath, g_params.meshFormat);
            if (success) {
                std::cout << "[INFO] Poisson mesh saved to: " << meshPath << std::endl;
            }
        } else {
            std::cerr << "Error: Failed to generate Poisson mesh" << std::endl;
        }
        
    } else {
        std::cerr << "Error: Unknown reconstruction mode: " << g_params.reconstructionMode << std::endl;
        return false;
    }
    
    return success;
}

bool reconstructFromDisparity(const cv::Mat& disparityMap,
                             const cv::Mat& Q,
                             const cv::Mat& colorImage,
                             const std::string& outputPath) {
    
    if (disparityMap.empty() || Q.empty()) {
        std::cerr << "Error: Disparity map or Q matrix is empty" << std::endl;
        return false;
    }
    
    // Convert disparity to depth
    cv::Mat xyz;
    cv::reprojectImageTo3D(disparityMap, xyz, Q, true);
    
    // Extract depth information
    std::vector<cv::Mat> xyzChannels;
    cv::split(xyz, xyzChannels);
    cv::Mat depthMap = xyzChannels[2];
    
    // Call depth map reconstruction method
    return reconstructAndSaveMesh(depthMap, colorImage, outputPath);
}

std::vector<Point3D> generatePointCloud(const cv::Mat& depthMap, 
                                       const cv::Mat& colorImage,
                                       const cv::Mat& K) {
    
    std::vector<Point3D> points;
    
    // Get camera intrinsics (pre-compute for efficiency)
    double fx = K.at<double>(0, 0);
    double fy = K.at<double>(1, 1);
    double cx = K.at<double>(0, 2);
    double cy = K.at<double>(1, 2);
    double inv_fx = 1.0 / fx;  // Pre-compute inverse
    double inv_fy = 1.0 / fy;
    
    // Check depth map type and convert
    cv::Mat processedDepth;
    if (depthMap.type() == CV_16U) {
        depthMap.convertTo(processedDepth, CV_32F, 1.0/1000.0);
    } else if (depthMap.type() == CV_32F) {
        processedDepth = depthMap.clone();
    } else {
        std::cerr << "Warning: Unsupported depth map type: " << depthMap.type() << std::endl;
        processedDepth = depthMap.clone();
    }
    
    bool hasColor = !colorImage.empty() && (colorImage.size() == depthMap.size());
    
    // Pre-calculate dimensions and estimate capacity
    int totalPixels = processedDepth.rows * processedDepth.cols;
    int totalIterations = totalPixels / (g_params.triangulationStep * g_params.triangulationStep);
    int estimatedPoints = totalIterations; // Estimate 25% valid points
    
    // Pre-allocate vector to avoid reallocation
    points.reserve(estimatedPoints);
    
    // Pre-compute color lookup if needed
    std::vector<uint8_t> depthColors;
    if (!hasColor || !g_params.useColor) {
        depthColors.reserve(256);
        for (int i = 0; i < 256; ++i) {
            float depth = i * 5000.0f / 255.0f;
            uint8_t colorValue = static_cast<uint8_t>(255 * (1.0f - std::min(depth / 5000.0f, 1.0f)));
            depthColors.push_back(colorValue);
        }
    }
    
    std::cout << "[INFO] Starting optimized point cloud generation..." << std::endl;
    std::cout << "[INFO] Total iterations: " << totalIterations << ", Estimated points: " << estimatedPoints << std::endl;
    
    // Generate points (optimized with progress)
    int validPointCount = 0;
    int currentIteration = 0;
    int progressStep = std::max(1, totalIterations / 20); // Update every 5%
    
    for (int y = 0; y < processedDepth.rows; y += g_params.triangulationStep) {
        for (int x = 0; x < processedDepth.cols; x += g_params.triangulationStep) {
            currentIteration++;
            
            // Progress update every 5%
            if (currentIteration % progressStep == 0) {
                int progress = (currentIteration * 100) / totalIterations;
                std::cout << "[PROGRESS] Point cloud generation: " << progress << "% (" << currentIteration << "/" << totalIterations << ")" << std::endl;
            }
            
            float depth = processedDepth.at<float>(y, x);
            
            // Fast filtering of invalid depth values
            if (depth <= 0 || depth > g_params.depthThreshold || std::isnan(depth) || std::isinf(depth)) {
                continue;
            }
            
            // Optimized back-projection (pre-computed inverses)
            float worldX = (x - cx) * depth * inv_fx;
            float worldY = (y - cy) * depth * inv_fy;
            float worldZ = depth;
            
            Point3D point(worldX, worldY, worldZ, 0, 0, 0, x, y);
            
            // Optimized color assignment
            if (hasColor && g_params.useColor) {
                if (colorImage.channels() == 3) {
                    const cv::Vec3b& color = colorImage.at<cv::Vec3b>(y, x);
                    point.b = color[0];
                    point.g = color[1];
                    point.r = color[2];
                } else if (colorImage.channels() == 1) {
                    uint8_t gray = colorImage.at<uint8_t>(y, x);
                    point.r = point.g = point.b = gray;
                }
            } else {
                // Use pre-computed depth colors
                int colorIndex = static_cast<int>(depth * 255.0f / 5000.0f);
                colorIndex = std::min(255, std::max(0, colorIndex));
                uint8_t colorValue = depthColors[colorIndex];
                point.r = colorValue;
                point.g = colorValue;
                point.b = 255 - colorValue;
            }
            
            points.push_back(point);
            validPointCount++;
        }
    }
    
    std::cout << "[PROGRESS] Point cloud generation: 100% (" << totalIterations << "/" << totalIterations << ")" << std::endl;
    std::cout << "[INFO] Generated " << points.size() << " points for point cloud" << std::endl;
    
    return points;
}

Mesh generateTriangulatedMesh(const cv::Mat& depthMap, 
                             const cv::Mat& colorImage,
                             const cv::Mat& K) {
    
    Mesh mesh;
    
    // Get camera intrinsics (pre-compute for efficiency)
    double fx = K.at<double>(0, 0);
    double fy = K.at<double>(1, 1);
    double cx = K.at<double>(0, 2);
    double cy = K.at<double>(1, 2);
    double inv_fx = 1.0 / fx;  // Pre-compute inverse
    double inv_fy = 1.0 / fy;
    
    // Check depth map type and convert
    cv::Mat processedDepth;
    if (depthMap.type() == CV_16U) {
        depthMap.convertTo(processedDepth, CV_32F, 1.0/1000.0);
    } else if (depthMap.type() == CV_32F) {
        processedDepth = depthMap.clone();
    } else {
        std::cerr << "Warning: Unsupported depth map type: " << depthMap.type() << std::endl;
        processedDepth = depthMap.clone();
    }
    
    bool hasColor = !colorImage.empty() && (colorImage.size() == depthMap.size());
    
    // Use larger step for triangulation to reduce computation
    int step = g_params.triangulationStep > 0 ? g_params.triangulationStep : 1;
    
    // Pre-calculate dimensions and estimate capacity
    int totalRows = processedDepth.rows / step;
    int totalCols = processedDepth.cols / step;
    int totalIterations = totalRows * totalCols;
    int estimatedPoints = totalIterations; // Estimate 50% valid points
    
    // Pre-allocate vectors to avoid reallocation
    std::vector<cv::Point2f> points2D;
    std::vector<int> validIndices;
    points2D.reserve(estimatedPoints);
    validIndices.reserve(estimatedPoints);
    mesh.vertices.reserve(estimatedPoints);
    
    std::cout << "[INFO] Starting optimized point generation..." << std::endl;
    std::cout << "[INFO] Total iterations: " << totalIterations << ", Estimated points: " << estimatedPoints << std::endl;
    
    // Pre-compute color lookup if needed
    std::vector<uint8_t> depthColors;
    if (!hasColor || !g_params.useColor) {
        depthColors.reserve(256);
        for (int i = 0; i < 256; ++i) {
            float depth = i * 5000.0f / 255.0f;
            uint8_t colorValue = static_cast<uint8_t>(255 * (1.0f - std::min(depth / 5000.0f, 1.0f)));
            depthColors.push_back(colorValue);
        }
    }
    
    // First pass: generate vertices and collect 2D points (optimized with progress)
    int validPointCount = 0;
    int currentIteration = 0;
    int progressStep = std::max(1, totalIterations / 20); // Update every 5%
    
    for (int y = 0; y < processedDepth.rows; y += step) {
        for (int x = 0; x < processedDepth.cols; x += step) {
            currentIteration++;
            
            // Progress update every 5%
            if (currentIteration % progressStep == 0) {
                int progress = (currentIteration * 100) / totalIterations;
                std::cout << "[PROGRESS] Point generation: " << progress << "% (" << currentIteration << "/" << totalIterations << ")" << std::endl;
            }
            
            float depth = processedDepth.at<float>(y, x);
            
            // Fast filtering of invalid depth values
            if (depth <= 0 || depth > g_params.depthThreshold || std::isnan(depth) || std::isinf(depth)) {
                continue;
            }
            
            // Optimized back-projection (pre-computed inverses)
            float worldX = (x - cx) * depth * inv_fx;
            float worldY = (y - cy) * depth * inv_fy;
            float worldZ = depth;
            
            Point3D point(worldX, worldY, worldZ, 0, 0, 0, x, y);
            
            // Optimized color assignment
            if (hasColor && g_params.useColor) {
                if (colorImage.channels() == 3) {
                    const cv::Vec3b& color = colorImage.at<cv::Vec3b>(y, x);
                    point.b = color[0];
                    point.g = color[1];
                    point.r = color[2];
                } else if (colorImage.channels() == 1) {
                    uint8_t gray = colorImage.at<uint8_t>(y, x);
                    point.r = point.g = point.b = gray;
                }
            } else {
                // Use pre-computed depth colors
                int colorIndex = static_cast<int>(depth * 255.0f / 5000.0f);
                colorIndex = std::min(255, std::max(0, colorIndex));
                uint8_t colorValue = depthColors[colorIndex];
                point.r = colorValue;
                point.g = colorValue;
                point.b = 255 - colorValue;
            }
            
            mesh.vertices.push_back(point);
            points2D.push_back(cv::Point2f(static_cast<float>(x), static_cast<float>(y)));
            validIndices.push_back(validPointCount++);
        }
    }
    
    std::cout << "[PROGRESS] Point generation: 100% (" << totalIterations << "/" << totalIterations << ")" << std::endl;
    std::cout << "[INFO] Generated " << points2D.size() << " valid points" << std::endl;
    
    if (points2D.size() < 3) {
        std::cerr << "Error: Not enough valid points for triangulation" << std::endl;
        return mesh;
    }
    
    // Adaptive point sampling based on density
    const int MAX_POINTS_FOR_TRIANGULATION = 100000; // 大幅增加点数限制
    if (points2D.size() > MAX_POINTS_FOR_TRIANGULATION) {
        std::cout << "[INFO] Sampling " << points2D.size() << " points to " << MAX_POINTS_FOR_TRIANGULATION << " points..." << std::endl;
        
        // Use adaptive sampling for better distribution
        std::vector<cv::Point2f> sampledPoints2D;
        std::vector<int> sampledValidIndices;
        std::vector<Point3D> sampledVertices;
        
        sampledPoints2D.reserve(MAX_POINTS_FOR_TRIANGULATION);
        sampledValidIndices.reserve(MAX_POINTS_FOR_TRIANGULATION);
        sampledVertices.reserve(MAX_POINTS_FOR_TRIANGULATION);
        
        // 使用更智能的采样策略：保留边缘和细节区域
        int sampleStep = points2D.size() / MAX_POINTS_FOR_TRIANGULATION;
        
        // 首先添加边界点（前10%和后10%）
        int boundaryCount = MAX_POINTS_FOR_TRIANGULATION / 10;
        for (int i = 0; i < boundaryCount && i < points2D.size(); i++) {
            sampledPoints2D.push_back(points2D[i]);
            sampledValidIndices.push_back(sampledVertices.size());
            sampledVertices.push_back(mesh.vertices[validIndices[i]]);
        }
        
        for (int i = points2D.size() - boundaryCount; i < points2D.size() && sampledPoints2D.size() < MAX_POINTS_FOR_TRIANGULATION; i++) {
            if (i >= 0) {
                sampledPoints2D.push_back(points2D[i]);
                sampledValidIndices.push_back(sampledVertices.size());
                sampledVertices.push_back(mesh.vertices[validIndices[i]]);
            }
        }
        
        // 然后均匀采样剩余点
        for (int i = boundaryCount; i < points2D.size() - boundaryCount && sampledPoints2D.size() < MAX_POINTS_FOR_TRIANGULATION; i += sampleStep) {
            sampledPoints2D.push_back(points2D[i]);
            sampledValidIndices.push_back(sampledVertices.size());
            sampledVertices.push_back(mesh.vertices[validIndices[i]]);
        }
        
        points2D = std::move(sampledPoints2D);
        validIndices = std::move(sampledValidIndices);
        mesh.vertices = std::move(sampledVertices);
        
        std::cout << "[INFO] Sampled to " << points2D.size() << " points for triangulation" << std::endl;
    }
    
    std::cout << "[INFO] Performing optimized Delaunay triangulation..." << std::endl;
    
    // Create optimized lookup map using unordered_map for better performance
    std::unordered_map<int64_t, int> pointToIndex;
    pointToIndex.reserve(points2D.size());
    
    for (size_t i = 0; i < points2D.size(); i++) {
        int x = static_cast<int>(std::round(points2D[i].x));
        int y = static_cast<int>(std::round(points2D[i].y));
        int64_t key = (static_cast<int64_t>(x) << 32) | static_cast<int64_t>(y);
        pointToIndex[key] = validIndices[i];
    }
    
    // Perform Delaunay triangulation with optimized boundary
    cv::Rect boundingRect = cv::boundingRect(points2D);
    // 增加边界扩展以获得更好的三角剖分
    boundingRect.x -= 20;
    boundingRect.y -= 20;
    boundingRect.width += 40;
    boundingRect.height += 40;
    
    cv::Subdiv2D subdiv(boundingRect);
    
    // Batch insert points with progress tracking
    std::cout << "[INFO] Inserting " << points2D.size() << " points into triangulation..." << std::endl;
    int insertProgressStep = std::max(1, (int)points2D.size() / 20); // Every 5%
    for (size_t i = 0; i < points2D.size(); i++) {
        subdiv.insert(points2D[i]);
        
        if ((i + 1) % insertProgressStep == 0) {
            int progress = ((i + 1) * 100) / points2D.size();
            std::cout << "[PROGRESS] Point insertion: " << progress << "% (" << (i + 1) << "/" << points2D.size() << ")" << std::endl;
        }
    }
    
    std::cout << "[PROGRESS] Point insertion: 100% (" << points2D.size() << "/" << points2D.size() << ")" << std::endl;
    
    // Get triangles
    std::vector<cv::Vec6f> triangleList;
    subdiv.getTriangleList(triangleList);
    
    std::cout << "[INFO] Found " << triangleList.size() << " triangles, processing..." << std::endl;
    
    // Pre-allocate faces vector
    mesh.faces.reserve(triangleList.size() / 2); // Estimate 50% valid triangles
    
    // Convert triangles to mesh faces (optimized with progress)
    int validTriangleCount = 0;
    int triangleProgressStep = std::max(1, (int)triangleList.size() / 20); // Every 5%
    
    for (size_t i = 0; i < triangleList.size(); i++) {
        const auto& triangle = triangleList[i];
        
        // Progress update every 5%
        if ((i + 1) % triangleProgressStep == 0) {
            int progress = ((i + 1) * 100) / triangleList.size();
            std::cout << "[PROGRESS] Triangle processing: " << progress << "% (" << (i + 1) << "/" << triangleList.size() << ")" << std::endl;
        }
        
        cv::Point2f pt1(triangle[0], triangle[1]);
        cv::Point2f pt2(triangle[2], triangle[3]);
        cv::Point2f pt3(triangle[4], triangle[5]);
        
        // Fast bounds checking
        if (pt1.x < 0 || pt1.x >= processedDepth.cols || pt1.y < 0 || pt1.y >= processedDepth.rows ||
            pt2.x < 0 || pt2.x >= processedDepth.cols || pt2.y < 0 || pt2.y >= processedDepth.rows ||
            pt3.x < 0 || pt3.x >= processedDepth.cols || pt3.y < 0 || pt3.y >= processedDepth.rows) {
            continue;
        }
        
        // Fast vertex index lookup
        int64_t key1 = (static_cast<int64_t>(std::round(pt1.x)) << 32) | static_cast<int64_t>(std::round(pt1.y));
        int64_t key2 = (static_cast<int64_t>(std::round(pt2.x)) << 32) | static_cast<int64_t>(std::round(pt2.y));
        int64_t key3 = (static_cast<int64_t>(std::round(pt3.x)) << 32) | static_cast<int64_t>(std::round(pt3.y));
        
        auto it1 = pointToIndex.find(key1);
        auto it2 = pointToIndex.find(key2);
        auto it3 = pointToIndex.find(key3);
        
        if (it1 == pointToIndex.end() || it2 == pointToIndex.end() || it3 == pointToIndex.end()) {
            continue;
        }
        
        int idx1 = it1->second;
        int idx2 = it2->second;
        int idx3 = it3->second;
        
        // Fast depth difference check
        float depth1 = mesh.vertices[idx1].z;
        float depth2 = mesh.vertices[idx2].z;
        float depth3 = mesh.vertices[idx3].z;
        
        float maxDepthDiff = std::max({std::abs(depth1 - depth2), 
                                      std::abs(depth2 - depth3), 
                                      std::abs(depth1 - depth3)});
        
        if (maxDepthDiff >= g_params.maxDepthDifference) {
            continue;
        }
        
        // Fast area check (avoid division)
        float area = std::abs((pt2.x - pt1.x) * (pt3.y - pt1.y) - (pt3.x - pt1.x) * (pt2.y - pt1.y));
        if (area > 800.0f) { // 降低面积阈值，允许更小的三角形
            continue;
        }
        
        mesh.faces.push_back(Triangle(idx1, idx2, idx3));
        validTriangleCount++;
    }
    
    std::cout << "[PROGRESS] Triangle processing: 100% (" << triangleList.size() << "/" << triangleList.size() << ")" << std::endl;
    std::cout << "[INFO] Generated optimized triangulated mesh with " << mesh.vertices.size() 
              << " vertices and " << mesh.faces.size() << " faces" << std::endl;
    
    return mesh;
}

bool savePointCloudPLY(const std::vector<Point3D>& points, 
                      const std::string& filename) {
    
    if (points.empty()) {
        std::cerr << "Error: Point cloud is empty, cannot save" << std::endl;
        return false;
    }
    
    std::ofstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error: Cannot create file " << filename << std::endl;
        return false;
    }
    
    // Write PLY header
    file << "ply\n";
    file << "format ascii 1.0\n";
    file << "element vertex " << points.size() << "\n";
    file << "property float x\n";
    file << "property float y\n";
    file << "property float z\n";
    file << "property uchar red\n";
    file << "property uchar green\n";
    file << "property uchar blue\n";
    file << "end_header\n";
    
    // Pre-allocate string buffer for better performance
    std::string buffer;
    buffer.reserve(points.size() * 50); // Estimate 50 chars per point
    
    // Write point data with buffering
    for (const auto& point : points) {
        buffer.clear();
        buffer += std::to_string(point.x) + " " + 
                  std::to_string(point.y) + " " + 
                  std::to_string(point.z) + " " +
                  std::to_string(static_cast<int>(point.r)) + " " +
                  std::to_string(static_cast<int>(point.g)) + " " +
                  std::to_string(static_cast<int>(point.b)) + "\n";
        file << buffer;
    }
    
    file.close();
    
    std::cout << "[INFO] Successfully saved " << points.size() << " points to " << filename << std::endl;
    return true;
}

bool saveMeshFile(const Mesh& mesh, const std::string& filename, const std::string& format) {
    
    if (mesh.vertices.empty() || mesh.faces.empty()) {
        std::cerr << "Error: Mesh is empty, cannot save" << std::endl;
        std::cerr << "  Vertices: " << mesh.vertices.size() << std::endl;
        std::cerr << "  Faces: " << mesh.faces.size() << std::endl;
        return false;
    }
    
    std::cout << "[INFO] Saving mesh to: " << filename << std::endl;
    std::cout << "[INFO] Format: " << format << std::endl;
    std::cout << "[INFO] Vertices: " << mesh.vertices.size() << ", Faces: " << mesh.faces.size() << std::endl;
    
    std::ofstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error: Cannot create file " << filename << std::endl;
        std::cerr << "  Check if directory exists and has write permissions" << std::endl;
        return false;
    }
    
    std::string format_lower = format;
    std::transform(format_lower.begin(), format_lower.end(), format_lower.begin(), ::tolower);
    
    // Pre-allocate string buffer for better performance
    std::string buffer;
    buffer.reserve(100); // Estimate buffer size
    
    if (format_lower == "ply") {
        // Write PLY header
        file << "ply\n";
        file << "format ascii 1.0\n";
        file << "element vertex " << mesh.vertices.size() << "\n";
        file << "property float x\n";
        file << "property float y\n";
        file << "property float z\n";
        file << "property uchar red\n";
        file << "property uchar green\n";
        file << "property uchar blue\n";
        file << "element face " << mesh.faces.size() << "\n";
        file << "property list uchar int vertex_indices\n";
        file << "end_header\n";
        
        // Write vertex data with buffering
        for (const auto& vertex : mesh.vertices) {
            buffer.clear();
            buffer += std::to_string(vertex.x) + " " + 
                      std::to_string(vertex.y) + " " + 
                      std::to_string(vertex.z) + " " +
                      std::to_string(static_cast<int>(vertex.r)) + " " +
                      std::to_string(static_cast<int>(vertex.g)) + " " +
                      std::to_string(static_cast<int>(vertex.b)) + "\n";
            file << buffer;
        }
        
        // Write face data with buffering
        for (const auto& face : mesh.faces) {
            buffer.clear();
            buffer += "3 " + std::to_string(face.v1) + " " + 
                      std::to_string(face.v2) + " " + 
                      std::to_string(face.v3) + "\n";
            file << buffer;
        }
        
    } else if (format_lower == "obj") {
        // Write OBJ header
        file << "# Mesh generated by StereoReconstruction\n";
        file << "# Vertices: " << mesh.vertices.size() << "\n";
        file << "# Faces: " << mesh.faces.size() << "\n\n";
        
        // Write vertex data (OBJ uses 1-based indexing) with buffering
        for (const auto& vertex : mesh.vertices) {
            buffer.clear();
            buffer += "v " + std::to_string(vertex.x) + " " + 
                      std::to_string(vertex.y) + " " + 
                      std::to_string(vertex.z) + "\n";
            file << buffer;
        }
        
        // Write face data (OBJ uses 1-based indexing) with buffering
        for (const auto& face : mesh.faces) {
            buffer.clear();
            buffer += "f " + std::to_string(face.v1 + 1) + " " + 
                      std::to_string(face.v2 + 1) + " " + 
                      std::to_string(face.v3 + 1) + "\n";
            file << buffer;
        }
        
    } else if (format_lower == "stl") {
        // Write STL header
        file << "solid mesh\n";
        
        // Write face data as triangles with buffering
        for (const auto& face : mesh.faces) {
            const Point3D& v1 = mesh.vertices[face.v1];
            const Point3D& v2 = mesh.vertices[face.v2];
            const Point3D& v3 = mesh.vertices[face.v3];
            
            // Calculate normal (optimized)
            float nx = (v2.y - v1.y) * (v3.z - v1.z) - (v2.z - v1.z) * (v3.y - v1.y);
            float ny = (v2.z - v1.z) * (v3.x - v1.x) - (v2.x - v1.x) * (v3.z - v1.z);
            float nz = (v2.x - v1.x) * (v3.y - v1.y) - (v2.y - v1.y) * (v3.x - v1.x);
            
            // Normalize (avoid division if possible)
            float length = sqrt(nx*nx + ny*ny + nz*nz);
            if (length > 1e-6f) {
                nx /= length;
                ny /= length;
                nz /= length;
            }
            
            buffer.clear();
            buffer += "  facet normal " + std::to_string(nx) + " " + 
                      std::to_string(ny) + " " + std::to_string(nz) + "\n";
            buffer += "    outer loop\n";
            buffer += "      vertex " + std::to_string(v1.x) + " " + 
                      std::to_string(v1.y) + " " + std::to_string(v1.z) + "\n";
            buffer += "      vertex " + std::to_string(v2.x) + " " + 
                      std::to_string(v2.y) + " " + std::to_string(v2.z) + "\n";
            buffer += "      vertex " + std::to_string(v3.x) + " " + 
                      std::to_string(v3.y) + " " + std::to_string(v3.z) + "\n";
            buffer += "    endloop\n";
            buffer += "  endfacet\n";
            file << buffer;
        }
        
        file << "endsolid mesh\n";
        
    } else {
        std::cerr << "Error: Unsupported format: " << format << std::endl;
        file.close();
        return false;
    }
    
    file.close();
    
    std::cout << "[INFO] Successfully saved mesh with " << mesh.vertices.size() 
              << " vertices and " << mesh.faces.size() << " faces to " << filename << std::endl;
    return true;
}

// ==================== Poisson Reconstruction Functions ====================
// 相关实现已迁移到poissonReconstruction.cpp

} // namespace MeshReconstruction