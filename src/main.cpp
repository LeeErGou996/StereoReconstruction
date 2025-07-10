#include <iostream>
#include <string>
#include <algorithm>
#include <filesystem>
#include <vector>
#include <fstream>
#include <regex>

#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include "disparity.h"
#include "8point.h"
#include "denseMatching.h"
#include "depth.h"
#include "meshReconstruction.h"
#include "sparseMatching.h"
#include "pointCloudFilter.h" // 点云滤波模块
#include "poissonReconstruction.h" // 添加Poisson重建头文件
#include "config.h"

// Simplified color processing function (defined directly in main.cpp to avoid header conflicts)
cv::Mat createSimpleColorDisparity(const cv::Mat& disparity, int colormap = cv::COLORMAP_JET) {
    if (disparity.empty()) return cv::Mat();
    
    cv::Mat disparity8;
    double minVal, maxVal;
    cv::minMaxLoc(disparity, &minVal, &maxVal, nullptr, nullptr, disparity > 0);
    
    if (maxVal > 0) {
        cv::Mat validMask = disparity > 0;
        disparity.convertTo(disparity8, CV_8U, 255.0/maxVal);
        disparity8.setTo(0, ~validMask);
    } else {
        disparity8 = cv::Mat::zeros(disparity.size(), CV_8U);
    }
    
    cv::Mat colorDisparity;
    cv::applyColorMap(disparity8, colorDisparity, colormap);
    
    cv::Mat validMask = disparity > 0;
    colorDisparity.setTo(cv::Scalar(0, 0, 0), ~validMask);
    
    return colorDisparity;
}

cv::Mat createSimpleBlendedImage(const cv::Mat& disparity, const cv::Mat& colorImage, double alpha = 0.5) {
    if (disparity.empty() || colorImage.empty()) return cv::Mat();
    if (disparity.size() != colorImage.size()) return colorImage.clone();
    
    cv::Mat colorDisparity = createSimpleColorDisparity(disparity);
    if (colorDisparity.empty()) return colorImage.clone();
    
    cv::Mat colorImg3C;
    if (colorImage.channels() == 1) {
        cv::cvtColor(colorImage, colorImg3C, cv::COLOR_GRAY2BGR);
    } else {
        colorImg3C = colorImage.clone();
    }
    
    cv::Mat blended;
    cv::addWeighted(colorImg3C, 1.0 - alpha, colorDisparity, alpha, 0, blended);
    return blended;
}

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

// Function to process a single stereo pair
bool processStereoPair(const std::string& leftImagePath, const std::string& rightImagePath, 
                      const std::string& outputDir, const cv::Mat& K, const cv::Mat& distCoeffs,
                      std::string matchingMode) {
    
    std::string baseFilename = getBaseFilename(leftImagePath);
    std::cout << "\n=== Processing stereo pair: " << baseFilename << " ===" << std::endl;
    
    // Create output directory for this pair
    std::string pairOutputDir = outputDir + "/" + baseFilename + "/";
    std::filesystem::create_directories(pairOutputDir);
    
    std::cout << "Left image: " << leftImagePath << std::endl;
    std::cout << "Right image: " << rightImagePath << std::endl;
    std::cout << "Output directory: " << pairOutputDir << std::endl;
    std::cout << "----------------------------------------" << std::endl;
    
    // 保存原始缓冲区
    std::streambuf* orig_cout = std::cout.rdbuf();
    std::streambuf* orig_cerr = std::cerr.rdbuf();
    // 打开log文件
    std::ofstream log_file(pairOutputDir + "log.txt");
    std::cout.rdbuf(log_file.rdbuf());
    std::cerr.rdbuf(log_file.rdbuf());
    
    // Read images
    cv::Mat imgL_color = cv::imread(leftImagePath, cv::IMREAD_COLOR);
    cv::Mat imgR_color = cv::imread(rightImagePath, cv::IMREAD_COLOR);
    cv::Mat imgL = cv::imread(leftImagePath, cv::IMREAD_GRAYSCALE);
    cv::Mat imgR = cv::imread(rightImagePath, cv::IMREAD_GRAYSCALE);
    
    if (imgL.empty() || imgR.empty() || imgL_color.empty() || imgR_color.empty()) {
        std::cerr << "Error: Cannot read image " << leftImagePath << " or " << rightImagePath << std::endl;
        return false;
    }
    
    std::cout << "Successfully read images, size: " << imgL.size() << std::endl;
    std::cout << "Color image channels: " << imgL_color.channels() << std::endl;
    
    // Save original color images
    cv::imwrite(pairOutputDir + "left_original.png", imgL_color);
    cv::imwrite(pairOutputDir + "right_original.png", imgR_color);
    std::cout << "Original color images saved" << std::endl;
    
    // Create feature detector and matcher
    DisparityProcessor sparseMatcher;
    cv::Ptr<cv::Feature2D> detector = sparseMatcher.createDetector(Config::instance().algorithm);
    if (!detector) {
        std::cerr << "Error: Cannot create feature detector" << std::endl;
        return false;
    }
    
    std::cout << "\n2. Feature detection and matching..." << std::endl;
    
    // Feature detection and matching
    std::vector<cv::Point2f> ptsL, ptsR;
    if (!sparseMatcher.detectAndMatch(imgL, imgR, detector, ptsL, ptsR)) {
        std::cerr << "Error: Feature detection and matching failed" << std::endl;
        return false;
    }
    
    std::cout << "Successfully matched " << ptsL.size() << " feature points" << std::endl;
    
    // Visualize feature matching
    if (!imgL_color.empty() && !imgR_color.empty() && ptsL.size() > 0) {
        std::vector<cv::KeyPoint> kpL, kpR;
        for (const auto& pt : ptsL) {
            kpL.push_back(cv::KeyPoint(pt, 1.0f));
        }
        for (const auto& pt : ptsR) {
            kpR.push_back(cv::KeyPoint(pt, 1.0f));
        }
        
        std::vector<cv::DMatch> matches;
        for (size_t i = 0; i < std::min(ptsL.size(), ptsR.size()); ++i) {
            matches.push_back(cv::DMatch(i, i, 0));
        }
        
        cv::Mat matchImg;
        cv::drawMatches(imgL_color, kpL, imgR_color, kpR, matches, matchImg);
        cv::imwrite(pairOutputDir + "feature_matches_color.png", matchImg);
        std::cout << "Color feature matching image saved" << std::endl;
    }
    
    std::cout << "\n3. Pose estimation..." << std::endl;
    
    // Estimate relative pose
    cv::Mat R, t;
    if (!EightPoint::estimatePose(ptsL, ptsR, K, R, t)) {
        std::cerr << "Error: Pose estimation failed" << std::endl;
        return false;
    }
    
    std::cout << "Successfully estimated relative pose" << std::endl;
    
    std::cout << "\n4. Stereo rectification..." << std::endl;
    
    // Stereo rectification and disparity computation
    cv::Mat rectL, rectR;
    DenseMatcher denseMatcher(K, distCoeffs, Config::instance().numDisparities, Config::instance().blockSize);

    if (!denseMatcher.rectifyImages(imgL, imgR, R, t, rectL, rectR)) {
        std::cerr << "Error: Stereo rectification failed" << std::endl;
        return false;
    }
    
    std::cout << "Stereo rectification completed successfully" << std::endl;
    
    // 直接使用rectL和rectR，不做滤波
    cv::Mat rectL_filtered = rectL;
    cv::Mat rectR_filtered = rectR;
    cv::imwrite(pairOutputDir + "left_rectified_filtered.png", rectL_filtered);
    cv::imwrite(pairOutputDir + "right_rectified_filtered.png", rectR_filtered);
    
    // Rectify color images
    cv::Mat rectL_color, rectR_color;
    
    // Get rectification parameters
    cv::Mat R1, R2, P1, P2, Q;
    cv::stereoRectify(K, distCoeffs, K, distCoeffs, 
                      imgL.size(), R, t, R1, R2, P1, P2, Q);
    
    // Generate rectification maps
    cv::Mat mapLx, mapLy, mapRx, mapRy;
    cv::initUndistortRectifyMap(K, distCoeffs, R1, P1, 
                                imgL_color.size(), CV_32FC1, mapLx, mapLy);
    cv::initUndistortRectifyMap(K, distCoeffs, R2, P2, 
                                imgR_color.size(), CV_32FC1, mapRx, mapRy);
    
    // Apply rectification to color images
    cv::remap(imgL_color, rectL_color, mapLx, mapLy, cv::INTER_LINEAR);
    cv::remap(imgR_color, rectR_color, mapRx, mapRy, cv::INTER_LINEAR);
    // 直接使用rectL_color，不做滤波
    cv::Mat rectL_color_filtered = rectL_color;
    cv::imwrite(pairOutputDir + "left_rectified_color_filtered.png", rectL_color_filtered);
    // Save rectified color images
    cv::imwrite(pairOutputDir + "left_rectified.png", rectL_color);
    cv::imwrite(pairOutputDir + "right_rectified.png", rectR_color);
    std::cout << "Rectified color images saved" << std::endl;
    
    std::cout << "\n5. Disparity computation..." << std::endl;
    
    // Compute disparity map
    cv::Mat disparity;
    if (matchingMode == "sparse") {
        SparseMatcher sparseMatcher(K, distCoeffs, Config::instance().numDisparities, Config::instance().blockSize);
        if (!sparseMatcher.computeDisparityMap(rectL_filtered, rectR_filtered, disparity)) {
            std::cerr << "Error: Sparse disparity computation failed" << std::endl;
            return false;
        }
    } else {
        DenseMatcher denseMatcher(K, distCoeffs, Config::instance().numDisparities, Config::instance().blockSize);
        if (!denseMatcher.computeDisparityMap(rectL_filtered, rectR_filtered, disparity)) {
            std::cerr << "Error: Dense disparity computation failed" << std::endl;
            return false;
        }
    }
    
    // Save disparity map
    cv::Mat disparity8;
    disparity.convertTo(disparity8, CV_8U, 255.0/(Config::instance().numDisparities*16.0));
    cv::imwrite(pairOutputDir + "disparity.png", disparity8);
    
    // Save color disparity maps
    cv::Mat colorDisparityJet = createSimpleColorDisparity(disparity, cv::COLORMAP_JET);
    cv::Mat colorDisparityHot = createSimpleColorDisparity(disparity, cv::COLORMAP_HOT);
    cv::imwrite(pairOutputDir + "disparity_color_jet.png", colorDisparityJet);
    cv::imwrite(pairOutputDir + "disparity_color_hot.png", colorDisparityHot);
    
    // Save blended images
    cv::Mat blended = createSimpleBlendedImage(disparity, rectL_color, 0.5);
    cv::Mat blendedStrong = createSimpleBlendedImage(disparity, rectL_color, 0.7);
    cv::imwrite(pairOutputDir + "disparity_blended.png", blended);
    cv::imwrite(pairOutputDir + "disparity_blended_strong.png", blendedStrong);
    
    std::cout << "Successfully saved disparity map and related outputs" << std::endl;
    
    std::cout << "\n6. Depth map computation..." << std::endl;

    // Depth computation
    cv::Mat Q_matrix = denseMatcher.getQMatrix();
    cv::Mat depthMap;
    std::string depthImagePath = pairOutputDir + "depth.png";
    if (!Depth::computeDepthMap(disparity, Q_matrix, depthMap, depthImagePath)) {
        std::cerr << "Error: Depth computation failed" << std::endl;
        return false;
    }
    
    // Save depth maps
    cv::imwrite(pairOutputDir + "depth.png", depthMap);
    
    // Create color depth map（先归一化并转为8位）
    cv::Mat normDepth, colorDepthMap;
    double minVal, maxVal;
    cv::minMaxLoc(depthMap, &minVal, &maxVal, nullptr, nullptr);
    depthMap.convertTo(normDepth, CV_8UC1, 255.0 / (maxVal - minVal), -minVal * 255.0 / (maxVal - minVal));
    cv::applyColorMap(normDepth, colorDepthMap, cv::COLORMAP_JET);
    cv::imwrite(pairOutputDir + "depth_color.png", colorDepthMap);
    
    // Save raw depth data
    cv::Mat depthFloat;
    depthMap.convertTo(depthFloat, CV_32F);
    cv::imwrite(pairOutputDir + "depth_raw.exr", depthFloat);
    
    std::cout << "Successfully saved depth maps" << std::endl;
    
    std::cout << "\n7. Mesh reconstruction..." << std::endl;
    
    // Set mesh reconstruction parameters
    MeshReconstruction::setReconstructionParams(Config::instance().meshParams);
    
    // Mesh reconstruction (format will be determined by parameters)
    std::string meshPath = pairOutputDir;
    // Remove trailing slash to avoid double slashes
    if (!meshPath.empty() && (meshPath.back() == '/' || meshPath.back() == '\\')) {
        meshPath.pop_back();
    }
    
    // === 自动插入：点云保存分支 ===
    if (Config::instance().reconstructionMode == 0) {
        std::cout << "[INFO] Generating and saving point cloud..." << std::endl;
        auto points = MeshReconstruction::generatePointCloud(depthMap, rectL_color_filtered, K);
        if (!points.empty()) {
            std::string pointCloudPath = pairOutputDir + "pointcloud.ply";
            if (MeshReconstruction::savePointCloudPLY(points, pointCloudPath)) {
                std::cout << "[INFO] Point cloud saved to: " << pointCloudPath << std::endl;
            } else {
                std::cerr << "Error: Failed to save point cloud" << std::endl;
                return false;
            }
        } else {
            std::cerr << "Error: Point cloud is empty" << std::endl;
            return false;
        }
    }
    // Poisson重建参数设置和多模式重建全部迁移到Config，由Config管理和调用
    MeshReconstruction::Mesh mesh = Config::generatePoissonMeshes(depthMap, rectL_color_filtered, K);
    
    if (!mesh.vertices.empty() && !mesh.faces.empty()) {
        MeshReconstruction::saveMeshFile(mesh, meshPath + "_poisson_mesh.ply", "ply");
        std::cout << "[INFO] Poisson mesh saved to: " << meshPath + "_poisson_mesh.ply" << std::endl;
    } else {
        std::cerr << "Error: Failed to generate Poisson mesh" << std::endl;
        return false;
    }
    
    std::cout << "Successfully completed 3D reconstruction" << std::endl;
    
    std::cout << "\n8. Processing completed for " << baseFilename << "!" << std::endl;
    std::cout << "Output files saved to: " << pairOutputDir << std::endl;
    
    // 恢复输出
    std::cout.rdbuf(orig_cout);
    std::cerr.rdbuf(orig_cerr);
    log_file.flush();
    log_file.close();
    
    return true;
}

int main() {
    Config::load("../src/config.txt");
    std::cout << "=== Stereo Vision Processing System (Batch Processing) ===" << std::endl;
    
    // Interactive input for reconstruction method
    // int reconstructionMode = 1;  // Default: Triangulated Mesh
    
    std::cout << "\n=== Reconstruction Method Selection ===" << std::endl;
    std::cout << "Please select reconstruction method:" << std::endl;
    std::cout << "  0: Point Cloud" << std::endl;
    std::cout << "  1: Triangulated Mesh" << std::endl;
    std::cout << "  2: Poisson Surface Reconstruction" << std::endl;
    std::cout << "Enter your choice (0, 1, or 2): ";
    
    std::string input;
    std::getline(std::cin, input);
    
    // Parse user input
    if (input == "0") {
        Config::instance().reconstructionMode = 0;
        std::cout << "Selected: Point Cloud" << std::endl;
    } else if (input == "1") {
        Config::instance().reconstructionMode = 1;
        std::cout << "Selected: Triangulated Mesh" << std::endl;
    } else if (input == "2") {
        Config::instance().reconstructionMode = 2;
        std::cout << "Selected: Poisson Surface Reconstruction" << std::endl;
    } else {
        std::cout << "Invalid input. Using default: Triangulated Mesh" << std::endl;
        Config::instance().reconstructionMode = 1;
    }
    
    std::cout << "=============================" << std::endl;
    
    // Define camera intrinsics - will be loaded from files
    cv::Mat K;
    std::cout << "Left images directory: " << Config::instance().leftDir << std::endl;
    std::cout << "Right images directory: " << Config::instance().rightDir << std::endl;
    std::cout << "Output directory: " << Config::instance().outputDir << std::endl;
    // Create output directory
    std::filesystem::create_directories(Config::instance().outputDir);
    
    // Get all files from left directory
    std::vector<std::string> leftFiles;
    for (const auto& entry : std::filesystem::directory_iterator(Config::instance().leftDir)) {
        if (entry.is_regular_file()) {
            std::string ext = entry.path().extension().string();
            if (ext == ".png" || ext == ".jpg" || ext == ".jpeg" || ext == ".bmp" || ext == ".tiff") {
                leftFiles.push_back(entry.path().string());
            }
        }
    }
    
    if (leftFiles.empty()) {
        std::cerr << "Error: No image files found in left directory: " << Config::instance().leftDir << std::endl;
        return -1;
    }
    
    std::cout << "Found " << leftFiles.size() << " left images" << std::endl;
    
    // Interactive input for matching mode
    std::string matchingMode = "dense";
    std::cout << "Please select matching mode (dense/sparse): ";
    std::getline(std::cin, matchingMode);
    if (matchingMode != "sparse" && matchingMode != "dense") matchingMode = "dense";
    
    // Process each stereo pair
    int successCount = 0;
    for (const auto& leftFile : leftFiles) {
        // 清理上一次循环的变量（如K_pair等）
        cv::Mat K_pair; // 每次循环新建，避免残留
        // 其他如特征点、图像等变量都在各自作用域内自动释放

        std::string baseFilename = getBaseFilename(leftFile);
        std::string rightFile = Config::instance().rightDir + baseFilename + ".png"; // Assuming same extension
        std::string cameraFile = Config::instance().cameraDir + baseFilename + ".txt";
        std::string pairOutputDir = Config::instance().outputDir + "/" + baseFilename + "/";
        std::filesystem::create_directories(pairOutputDir);

        // --- 新增：重定向cout/cerr到log文件 ---
        std::streambuf* orig_cout = std::cout.rdbuf();
        std::streambuf* orig_cerr = std::cerr.rdbuf();
        std::ofstream log_file(pairOutputDir + "log.txt");
        std::cout.rdbuf(log_file.rdbuf());
        std::cerr.rdbuf(log_file.rdbuf());
        // -----------------------------------

        bool success = true;
        // Check if corresponding right image exists
        if (!std::filesystem::exists(rightFile)) {
            std::cout << "Warning: No corresponding right image found for " << baseFilename << std::endl;
            success = false;
        }

        // Load camera intrinsics for this stereo pair
        if (success && !parseCameraIntrinsics(cameraFile, K_pair)) {
            std::cout << "Warning: Cannot load camera intrinsics for " << baseFilename << ", skipping..." << std::endl;
            success = false;
        }

        // Process the stereo pair
        if (success) {
            if (!processStereoPair(leftFile, rightFile, Config::instance().outputDir, K_pair, Config::instance().distCoeffs, matchingMode)) {
                std::cout << "[ERROR] Reconstruction failed for " << baseFilename << ", moving to next input." << std::endl;
                success = false;
            }
        }

        if (success) {
            successCount++;
        }

        // --- 恢复输出流 ---
        std::cout.rdbuf(orig_cout);
        std::cerr.rdbuf(orig_cerr);
        log_file.flush();
        log_file.close();
    }
    
    std::cout << "\n=== Batch Processing Summary ===" << std::endl;
    std::cout << "Total stereo pairs found: " << leftFiles.size() << std::endl;
    std::cout << "Successfully processed: " << successCount << std::endl;
    std::cout << "Failed: " << (leftFiles.size() - successCount) << std::endl;
    
    return 0;
}