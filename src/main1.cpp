#include <iostream>
#include <string>
#include <algorithm>
#include <filesystem>
#include <vector>
#include <fstream>

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

// Function to process a single stereo pair
bool processStereoPair(const std::string& leftImagePath, const std::string& rightImagePath, 
                      const std::string& outputDir, const cv::Mat& K, const cv::Mat& distCoeffs,
                      int numDisparities, int blockSize, FeatureType algorithm, int reconstructionMode, std::string matchingMode) {
    
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
    cv::Ptr<cv::Feature2D> detector = sparseMatcher.createDetector(algorithm);
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
    DenseMatcher denseMatcher(K, distCoeffs, numDisparities, blockSize);

    if (!denseMatcher.rectifyImages(imgL, imgR, R, t, rectL, rectR)) {
        std::cerr << "Error: Stereo rectification failed" << std::endl;
        return false;
    }
    
    std::cout << "Stereo rectification completed successfully" << std::endl;
    
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
    
    // Save rectified color images
    cv::imwrite(pairOutputDir + "left_rectified.png", rectL_color);
    cv::imwrite(pairOutputDir + "right_rectified.png", rectR_color);
    std::cout << "Rectified color images saved" << std::endl;
    
    std::cout << "\n5. Disparity computation..." << std::endl;
    
    // Compute disparity map
    cv::Mat disparity;
    if (matchingMode == "sparse") {
        SparseMatcher sparseMatcher(K, distCoeffs, numDisparities, blockSize);
        if (!sparseMatcher.computeDisparityMap(rectL, rectR, disparity)) {
            std::cerr << "Error: Sparse disparity computation failed" << std::endl;
            return false;
        }
    } else {
        DenseMatcher denseMatcher(K, distCoeffs, numDisparities, blockSize);
        if (!denseMatcher.computeDisparityMap(rectL, rectR, disparity)) {
            std::cerr << "Error: Dense disparity computation failed" << std::endl;
            return false;
        }
    }
    
    // Save disparity map
    cv::Mat disparity8;
    disparity.convertTo(disparity8, CV_8U, 255.0/(numDisparities*16.0));
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
    
    // Mesh reconstruction (format will be determined by parameters)
    std::string meshPath = pairOutputDir;
    // Remove trailing slash to avoid double slashes
    if (!meshPath.empty() && (meshPath.back() == '/' || meshPath.back() == '\\')) {
        meshPath.pop_back();
    }
    
    // Set mesh reconstruction parameters
    MeshReconstruction::ReconstructionParams meshParams;
    meshParams.reconstructionMode = reconstructionMode;
    meshParams.meshFormat = "ply";      // Default PLY format
    meshParams.triangulationStep = 1;   // 改为1获得最高精度
    meshParams.maxDepthDifference = 500.0f; // 放宽深度差异限制
    meshParams.useColor = true;         // Use color information
    meshParams.depthThreshold = 8000.0f; // 降低深度阈值，专注于近处物体
    
    // Poisson reconstruction parameters
    meshParams.poissonDepth = 8.0f;           // Octree depth
    meshParams.poissonSolverDivide = 8.0f;    // Solver depth
    meshParams.poissonSamplesPerNode = 1.5f;  // Samples per node
    meshParams.poissonFullDepth = 5.0f;       // Full reconstruction depth
    meshParams.poissonTrim = 0.0f;            // Trimming parameter
    meshParams.poissonUseConfidence = false;  // Use confidence weights
    
    // Apply mesh reconstruction parameters
    MeshReconstruction::setReconstructionParams(meshParams);
    
    if (!MeshReconstruction::reconstructAndSaveMesh(depthMap, rectL_color, meshPath)) {
        std::cerr << "Error: Mesh reconstruction failed" << std::endl;
        return false;
    }
    
    std::cout << "Successfully completed 3D reconstruction" << std::endl;
    
    std::cout << "\n8. Processing completed for " << baseFilename << "!" << std::endl;
    std::cout << "Output files saved to: " << pairOutputDir << std::endl;
    
    // 恢复输出
    std::cout.rdbuf(orig_cout);
    std::cerr.rdbuf(orig_cerr);
    
    return true;
}

int main() {
    std::cout << "=== Stereo Vision Processing System (Batch Processing) ===" << std::endl;
    
    // Interactive input for reconstruction method
    int reconstructionMode = 1;  // Default: Triangulated Mesh
    
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
        reconstructionMode = 0;
        std::cout << "Selected: Point Cloud" << std::endl;
    } else if (input == "1") {
        reconstructionMode = 1;
        std::cout << "Selected: Triangulated Mesh" << std::endl;
    } else if (input == "2") {
        reconstructionMode = 2;
        std::cout << "Selected: Poisson Surface Reconstruction" << std::endl;
    } else {
        std::cout << "Invalid input. Using default: Triangulated Mesh" << std::endl;
        reconstructionMode = 1;
    }
    
    std::cout << "=============================" << std::endl;
    
    // Define camera intrinsics
    cv::Mat K = (cv::Mat_<double>(3,3) << 1758.23, 0, 953.34, 0, 1758.23, 552.29, 0, 0, 1);
    cv::Mat distCoeffs = cv::Mat::zeros(5, 1, CV_64F);
    
    // Define stereo matching parameters
    int numDisparities = 128;
    int blockSize = 5;
    
    // Change algorithm here: SIFT, SURF, ORB
    FeatureType algorithm = FeatureType::ORB;  // Default use ORB
    
    // Directory paths
    std::string leftDir = "../data/left/";
    std::string rightDir = "../data/right/";
    std::string outputDir = "../test/";
    
    std::cout << "Left images directory: " << leftDir << std::endl;
    std::cout << "Right images directory: " << rightDir << std::endl;
    std::cout << "Output directory: " << outputDir << std::endl;
    
    // Create output directory
    std::filesystem::create_directories(outputDir);
    
    // Get all files from left directory
    std::vector<std::string> leftFiles;
    for (const auto& entry : std::filesystem::directory_iterator(leftDir)) {
        if (entry.is_regular_file()) {
            std::string ext = entry.path().extension().string();
            if (ext == ".png" || ext == ".jpg" || ext == ".jpeg" || ext == ".bmp" || ext == ".tiff") {
                leftFiles.push_back(entry.path().string());
            }
        }
    }
    
    if (leftFiles.empty()) {
        std::cerr << "Error: No image files found in left directory: " << leftDir << std::endl;
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
        std::string baseFilename = getBaseFilename(leftFile);
        std::string rightFile = rightDir + baseFilename + ".png"; // Assuming same extension
        
        // Check if corresponding right image exists
        if (!std::filesystem::exists(rightFile)) {
            std::cout << "Warning: No corresponding right image found for " << baseFilename << std::endl;
            continue;
        }
        
        // Process the stereo pair
        if (processStereoPair(leftFile, rightFile, outputDir, K, distCoeffs, 
                             numDisparities, blockSize, algorithm, reconstructionMode, matchingMode)) {
            successCount++;
        }
    }
    
    std::cout << "\n=== Batch Processing Summary ===" << std::endl;
    std::cout << "Total stereo pairs found: " << leftFiles.size() << std::endl;
    std::cout << "Successfully processed: " << successCount << std::endl;
    std::cout << "Failed: " << (leftFiles.size() - successCount) << std::endl;
    
    return 0;
}