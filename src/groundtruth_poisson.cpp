#include <iostream>
#include <string>
#include <filesystem>
#include <vector>
#include <fstream>
#include <regex>

#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include "meshReconstruction.h"
#include "poissonReconstruction.h"
#include "config.h"

// Function to get base filename without extension
std::string getBaseFilename(const std::string& filepath) {
    std::filesystem::path path(filepath);
    return path.stem().string();
}

// Function to parse camera intrinsics from file
bool parseCameraIntrinsics(const std::string& cameraFilePath, cv::Mat& K) {
    std::ifstream file(cameraFilePath);
    if (!file.is_open()) {
        std::cerr << "Error: Cannot open camera file: " << cameraFilePath << std::endl;
        return false;
    }
    
    std::string line;
    std::getline(file, line);
    file.close();
    
    // Parse format: cam0=[1733.74 0 792.27; 0 1733.74 541.89; 0 0 1]
    // Support negative numbers for cx, cy values
    std::regex pattern(R"(cam0=\[([+-]?[\d.]+)\s+([+-]?[\d.]+)\s+([+-]?[\d.]+);\s+([+-]?[\d.]+)\s+([+-]?[\d.]+)\s+([+-]?[\d.]+);\s+([+-]?[\d.]+)\s+([+-]?[\d.]+)\s+([+-]?[\d.]+)\])");
    std::smatch matches;
    
    if (std::regex_search(line, matches, pattern) && matches.size() == 10) {
        K = (cv::Mat_<double>(3,3) << 
            std::stod(matches[1]), std::stod(matches[2]), std::stod(matches[3]),
            std::stod(matches[4]), std::stod(matches[5]), std::stod(matches[6]),
            std::stod(matches[7]), std::stod(matches[8]), std::stod(matches[9]));
        
        std::cout << "Successfully loaded camera intrinsics from: " << cameraFilePath << std::endl;
        std::cout << "Camera matrix K:" << std::endl << K << std::endl;
        return true;
    } else {
        std::cerr << "Error: Invalid camera intrinsics format in file: " << cameraFilePath << std::endl;
        std::cerr << "Expected format: cam0=[fx 0 cx; 0 fy cy; 0 0 1]" << std::endl;
        std::cerr << "Found: " << line << std::endl;
        return false;
    }
}

// 合并多个tile mesh为一个完整mesh
MeshReconstruction::Mesh mergeMeshes(const std::vector<MeshReconstruction::Mesh>& meshes) {
    MeshReconstruction::Mesh merged;
    int vertexOffset = 0;
    for (const auto& mesh : meshes) {
        merged.vertices.insert(merged.vertices.end(), mesh.vertices.begin(), mesh.vertices.end());
        for (const auto& face : mesh.faces) {
            MeshReconstruction::Triangle newFace(face.v1 + vertexOffset, face.v2 + vertexOffset, face.v3 + vertexOffset);
            merged.faces.push_back(newFace);
        }
        vertexOffset += mesh.vertices.size();
    }
    return merged;
}

// Function to process a single groundtruth depth map
bool processGroundtruthDepth(const std::string& depthMapPath, const std::string& colorImagePath, 
                            const std::string& cameraFilePath, const std::string& outputDir) {
    
    std::string baseFilename = getBaseFilename(depthMapPath);
    std::cout << "\n=== Processing groundtruth depth: " << baseFilename << " ===" << std::endl;
    
    // Create output directory for this pair
    std::string pairOutputDir = outputDir + "/" + baseFilename + "/";
    std::filesystem::create_directories(pairOutputDir);
    
    // === 新增：重定向cout/cerr到log文件 ===
    std::streambuf* orig_cout = std::cout.rdbuf();
    std::streambuf* orig_cerr = std::cerr.rdbuf();
    std::ofstream log_file(pairOutputDir + "log.txt");
    std::cout.rdbuf(log_file.rdbuf());
    std::cerr.rdbuf(log_file.rdbuf());
    
    std::cout << "Depth map: " << depthMapPath << std::endl;
    std::cout << "Color image: " << colorImagePath << std::endl;
    std::cout << "Camera file: " << cameraFilePath << std::endl;
    std::cout << "Output directory: " << pairOutputDir << std::endl;
    std::cout << "----------------------------------------" << std::endl;
    
    // Load camera intrinsics
    cv::Mat K;
    if (!parseCameraIntrinsics(cameraFilePath, K)) {
        std::cerr << "Error: Failed to load camera intrinsics" << std::endl;
        return false;
    }
    
    // Load depth map
    cv::Mat depthMap = cv::imread(depthMapPath, cv::IMREAD_ANYDEPTH);
    if (depthMap.empty()) {
        std::cerr << "Error: Cannot read depth map " << depthMapPath << std::endl;
        return false;
    }
    
    // Load color image
    cv::Mat colorImage = cv::imread(colorImagePath, cv::IMREAD_COLOR);
    if (colorImage.empty()) {
        std::cerr << "Error: Cannot read color image " << colorImagePath << std::endl;
        return false;
    }
    
    // === 新增：对彩色图像做 undistort/rectify 处理 ===
    cv::Mat distCoeffs = Config::instance().distCoeffs;
    cv::Mat R1 = cv::Mat::eye(3, 3, CV_64F); // 单目，直接用单位矩阵
    cv::Mat P1 = K.clone(); // 投影矩阵直接用K
    cv::Mat map1, map2;
    cv::initUndistortRectifyMap(K, distCoeffs, R1, P1, colorImage.size(), CV_32FC1, map1, map2);
    cv::Mat colorImageRectified;
    cv::remap(colorImage, colorImageRectified, map1, map2, cv::INTER_LINEAR);
    // 保存校正后的彩色图像
    cv::imwrite(pairOutputDir + "color_image_rectified.png", colorImageRectified);
    std::cout << "Rectified color image saved" << std::endl;

    // === 新增：自动对齐尺寸 ===
    if (colorImageRectified.size() != depthMap.size()) {
        cv::resize(colorImageRectified, colorImageRectified, depthMap.size());
        std::cout << "Resized color image to match depth map size." << std::endl;
    }
    
    std::cout << "Successfully loaded depth map, size: " << depthMap.size() << ", type: " << depthMap.type() << std::endl;
    std::cout << "Successfully loaded color image, size: " << colorImage.size() << ", channels: " << colorImage.channels() << std::endl;
    
    // Save original images
    cv::imwrite(pairOutputDir + "groundtruth_depth.png", depthMap);
    cv::imwrite(pairOutputDir + "color_image.png", colorImage);
    std::cout << "Original images saved" << std::endl;
    
    // Check if sizes match
    if (depthMap.size() != colorImage.size()) {
        std::cerr << "Warning: Depth map and color image sizes don't match!" << std::endl;
        std::cerr << "Depth: " << depthMap.size() << ", Color: " << colorImage.size() << std::endl;
    }
    
    std::cout << "\n1. Starting Poisson surface reconstruction..." << std::endl;
    
    // 一次性对全图Poisson重建
    MeshReconstruction::Mesh mesh = MeshReconstruction::generatePoissonMesh(depthMap, colorImageRectified, K);
    // 反转所有三角面顶点顺序
    for (auto& face : mesh.faces) {
        std::swap(face.v2, face.v3);
    }
    if (!mesh.vertices.empty() && !mesh.faces.empty()) {
        std::cout << "Successfully generated mesh with " << mesh.vertices.size() 
                  << " vertices and " << mesh.faces.size() << " faces" << std::endl;
        // Save mesh
        std::string meshPath = pairOutputDir + "groundtruth_poisson_mesh.ply";
        if (MeshReconstruction::saveMeshFile(mesh, meshPath, "ply")) {
            std::cout << "Mesh saved to: " << meshPath << std::endl;
        } else {
            std::cerr << "Error: Failed to save mesh" << std::endl;
            return false;
        }
        // Save point cloud (vertices only)
        std::string pointCloudPath = pairOutputDir + "groundtruth_pointcloud.ply";
        if (MeshReconstruction::savePointCloudPLY(mesh.vertices, pointCloudPath)) {
            std::cout << "Point cloud saved to: " << pointCloudPath << std::endl;
        } else {
            std::cerr << "Error: Failed to save point cloud" << std::endl;
        }
    } else {
        std::cerr << "Error: Failed to generate mesh" << std::endl;
        return false;
    }
    
    std::cout << "\n2. Processing completed for " << baseFilename << "!" << std::endl;
    std::cout << "Output files saved to: " << pairOutputDir << std::endl;

    // === 新增：恢复cout/cerr输出流 ===
    std::cout.rdbuf(orig_cout);
    std::cerr.rdbuf(orig_cerr);
    log_file.flush();
    log_file.close();
    
    return true;
}

int main() {
    // Load configuration
    Config::load("../src/config.txt");
    std::cout << "=== Groundtruth Poisson Reconstruction System ===" << std::endl;
    
    // Define paths
    std::string groundtruthDir = "../data/ground/";
    std::string colorDir = "../data/left/";  // Use left images as color images
    std::string cameraDir = "../data/camera/";
    std::string outputDir = "../test/groundtruth/";
    
    std::cout << "Groundtruth depth directory: " << groundtruthDir << std::endl;
    std::cout << "Color images directory: " << colorDir << std::endl;
    std::cout << "Camera files directory: " << cameraDir << std::endl;
    std::cout << "Output directory: " << outputDir << std::endl;
    
    // Create output directory
    std::filesystem::create_directories(outputDir);
    
    // Get all groundtruth depth files
    std::vector<std::string> depthFiles;
    for (const auto& entry : std::filesystem::directory_iterator(groundtruthDir)) {
        if (entry.is_regular_file()) {
            std::string ext = entry.path().extension().string();
            if (ext == ".png" || ext == ".jpg" || ext == ".jpeg" || ext == ".bmp" || ext == ".tiff" || ext == ".exr") {
                depthFiles.push_back(entry.path().string());
            }
        }
    }
    
    if (depthFiles.empty()) {
        std::cerr << "Error: No depth files found in groundtruth directory: " << groundtruthDir << std::endl;
        return -1;
    }
    
    std::cout << "Found " << depthFiles.size() << " groundtruth depth files" << std::endl;
    
    // Process each groundtruth depth file
    int successCount = 0;
    for (const auto& depthFile : depthFiles) {
        std::string baseFilename = getBaseFilename(depthFile);
        std::string colorFile = colorDir + baseFilename + ".png";
        std::string cameraFile = cameraDir + baseFilename + ".txt";
        
        bool success = true;
        
        // Check if corresponding files exist
        if (!std::filesystem::exists(colorFile)) {
            std::cout << "Warning: No corresponding color image found for " << baseFilename << std::endl;
            success = false;
        }
        
        if (!std::filesystem::exists(cameraFile)) {
            std::cout << "Warning: No corresponding camera file found for " << baseFilename << std::endl;
            success = false;
        }
        
        // Process the groundtruth depth
        if (success) {
            if (!processGroundtruthDepth(depthFile, colorFile, cameraFile, outputDir)) {
                std::cout << "[ERROR] Reconstruction failed for " << baseFilename << ", moving to next input." << std::endl;
                success = false;
            }
        }
        
        if (success) {
            successCount++;
        }
    }
    
    std::cout << "\n=== Groundtruth Processing Summary ===" << std::endl;
    std::cout << "Total groundtruth depth files found: " << depthFiles.size() << std::endl;
    std::cout << "Successfully processed: " << successCount << std::endl;
    std::cout << "Failed: " << (depthFiles.size() - successCount) << std::endl;
    
    return 0;
} 