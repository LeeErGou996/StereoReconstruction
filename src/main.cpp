#include <iostream>
#include <string>
#include <algorithm>

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

int main() {
    std::cout << "=== Stereo Vision Processing System (with color support) ===" << std::endl;
    
    // Define camera intrinsics
    cv::Mat K = (cv::Mat_<double>(3,3) << 1758.23, 0, 953.34, 0, 1758.23, 552.29, 0, 0, 1);
    cv::Mat distCoeffs = cv::Mat::zeros(5, 1, CV_64F);
    
    // Define stereo matching parameters
    int numDisparities = 288;
    int blockSize = 9;
    
    // Change algorithm here: SIFT, SURF, ORB
    FeatureType algorithm = FeatureType::ORB;  // Default use ORB
    
    // Image paths
    std::string leftImagePath = "../data/left.png";
    std::string rightImagePath = "../data/right.png";
    std::string outputBasePath = "../test/";
    std::string outputPath = outputBasePath + "disparity.png";
    
    std::cout << "\n1. Start stereo vision processing..." << std::endl;
    std::cout << "Left image: " << leftImagePath << std::endl;
    std::cout << "Right image: " << rightImagePath << std::endl;
    std::cout << "Output path: " << outputBasePath << std::endl;
    std::cout << "----------------------------------------" << std::endl;
    
    // **Modification 1: Read both color and grayscale images**
    cv::Mat imgL_color = cv::imread(leftImagePath, cv::IMREAD_COLOR);   // Color image
    cv::Mat imgR_color = cv::imread(rightImagePath, cv::IMREAD_COLOR);  // Color image
    cv::Mat imgL = cv::imread(leftImagePath, cv::IMREAD_GRAYSCALE);     // Grayscale image (for feature detection)
    cv::Mat imgR = cv::imread(rightImagePath, cv::IMREAD_GRAYSCALE);    // Grayscale image (for feature detection)
    
    if (imgL.empty() || imgR.empty() || imgL_color.empty() || imgR_color.empty()) {
        std::cerr << "Error: Cannot read image " << leftImagePath << " or " << rightImagePath << std::endl;
        return -1;
    }
    
    std::cout << "Successfully read images, size: " << imgL.size() << std::endl;
    std::cout << "Color image channels: " << imgL_color.channels() << std::endl;
    
    // **Modification 2: Save original color images**
    cv::imwrite(outputBasePath + "left_original.png", imgL_color);
    cv::imwrite(outputBasePath + "right_original.png", imgR_color);
    std::cout << "Original color images saved" << std::endl;
    
    // Create feature detector and matcher (using grayscale images)
    DisparityProcessor sparseMatcher;
    cv::Ptr<cv::Feature2D> detector = sparseMatcher.createDetector(algorithm);
    if (!detector) {
        std::cerr << "Error: Cannot create feature detector" << std::endl;
        return -1;
    }
    
    std::cout << "\n2. Feature detection and matching..." << std::endl;
    
    // Feature detection and matching (using grayscale images)
    std::vector<cv::Point2f> ptsL, ptsR;
    if (!sparseMatcher.detectAndMatch(imgL, imgR, detector, ptsL, ptsR)) {
        std::cerr << "Error: Feature detection and matching failed" << std::endl;
        return -1;
    }
    
    std::cout << "Successfully matched " << ptsL.size() << " feature points" << std::endl;
    
    // **New: Visualize feature matching (color version)**
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
        cv::imwrite(outputBasePath + "feature_matches_color.png", matchImg);
        std::cout << "Color feature matching image saved" << std::endl;
    }
    
    std::cout << "\n3. Pose estimation..." << std::endl;
    
    // Estimate relative pose
    cv::Mat R, t;
    if (!EightPoint::estimatePose(ptsL, ptsR, K, R, t)) {
        std::cerr << "Error: Pose estimation failed" << std::endl;
        return -1;
    }
    
    std::cout << "Successfully estimated relative pose" << std::endl;
    
    std::cout << "\n4. Stereo rectification..." << std::endl;
    
    // Stereo rectification and disparity computation
    cv::Mat rectL, rectR;
    // DenseMatcher denseMatcher(K, distCoeffs, numDisparities, blockSize);
    DenseMatcher denseMatcher(K, distCoeffs, numDisparities, blockSize, /*useCustomMatcher=*/true);


    if (!denseMatcher.rectifyImages(imgL, imgR, R, t, rectL, rectR)) {
        std::cerr << "Error: Stereo rectification failed" << std::endl;
        return -1;
    }
    
    std::cout << "Stereo rectification completed successfully" << std::endl;
    
    // **New: Rectify color images**
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
    cv::imwrite(outputBasePath + "left_rectified.png", rectL_color);
    cv::imwrite(outputBasePath + "right_rectified.png", rectR_color);
    std::cout << "Rectified color images saved" << std::endl;
    
    std::cout << "\n5. Disparity computation..." << std::endl;
    
    // Compute disparity map
    cv::Mat disparity;
    if (!denseMatcher.computeDisparityMap(rectL, rectR, disparity)) {
        std::cerr << "Error: Disparity computation failed" << std::endl;
        return -1;
    }
    
    // Save traditional grayscale disparity map
    cv::Mat disparity8;
    disparity.convertTo(disparity8, CV_8U, 255.0/(numDisparities*16.0));
    
    if (!cv::imwrite(outputPath, disparity8)) {
        std::cerr << "Error: Cannot save disparity map to " << outputPath << std::endl;
        return -1;
    }
    
    std::cout << "Successfully saved grayscale disparity map to: " << outputPath << std::endl;
    
    // **New: Create and save color disparity map**
    cv::Mat colorDisparity = createSimpleColorDisparity(disparity, cv::COLORMAP_JET);
    cv::imwrite(outputBasePath + "disparity_color_jet.png", colorDisparity);
    
    cv::Mat colorDisparity2 = createSimpleColorDisparity(disparity, cv::COLORMAP_HOT);
    cv::imwrite(outputBasePath + "disparity_color_hot.png", colorDisparity2);
    
    // **New: Create and save blended image**
    if (!rectL_color.empty()) {
        cv::Mat blended = createSimpleBlendedImage(disparity, rectL_color, 0.4);
        cv::imwrite(outputBasePath + "disparity_blended.png", blended);
        
        cv::Mat blended2 = createSimpleBlendedImage(disparity, rectL_color, 0.6);
        cv::imwrite(outputBasePath + "disparity_blended_strong.png", blended2);
        
        std::cout << "Successfully saved color disparity map and blended image" << std::endl;
    }
    
    std::cout << "\n6. Depth map computation..." << std::endl;

    // Compute and get depth map
    std::string depthImagePath = outputBasePath + "depth.png";
    cv::Mat Q_matrix = denseMatcher.getQMatrix();
    cv::Mat depthMap; // Mat object for receiving depth map
    if (!Depth::computeDepthMap(disparity, Q_matrix, depthMap, depthImagePath)) {
        std::cerr << "Error: Cannot compute depth map!" << std::endl;
        return -1;
    }

    // Save depth map to file (normalized and original)
    if (depthMap.empty() || depthMap.type() != CV_32F) {
        std::cerr << "Error: Depth map is empty or incorrect type (needs CV_32F), cannot save." << std::endl;
        return -1;
    }

    cv::Mat validDepthMask;
    cv::bitwise_and(
        depthMap > 0,           // Depth is positive
        depthMap < 10000.0f,    // Depth within reasonable range
        validDepthMask
    );
    depthMap.setTo(0, ~validDepthMask); // Set invalid depth to 0

    cv::Mat normalizedDepthMap;
    if (cv::countNonZero(validDepthMask) > 0) {
        std::vector<float> validDepths;
        for (int r = 0; r < depthMap.rows; ++r) {
            for (int c = 0; c < depthMap.cols; ++c) {
                if (validDepthMask.at<uchar>(r, c)) {
                    validDepths.push_back(depthMap.at<float>(r, c));
                }
            }
        }
        
        if (!validDepths.empty()) {
            std::sort(validDepths.begin(), validDepths.end());
            float minDepth = validDepths[0];
            float maxDepth = validDepths[static_cast<int>(validDepths.size() * 0.98)]; // 98th percentile
            
            std::cout << "Depth range: " << minDepth << " - " << maxDepth << " units" << std::endl;
            
            normalizedDepthMap = cv::Mat::zeros(depthMap.size(), CV_8U);
            
            for (int r = 0; r < depthMap.rows; ++r) {
                for (int c = 0; c < depthMap.cols; ++c) {
                    if (validDepthMask.at<uchar>(r, c)) {
                        float depth = depthMap.at<float>(r, c);
                        depth = std::max(minDepth, std::min(maxDepth, depth)); // Clip to range
                        uchar normalizedValue = static_cast<uchar>(
                            255.0f * (depth - minDepth) / (maxDepth - minDepth)
                        );
                        normalizedDepthMap.at<uchar>(r, c) = normalizedValue;
                    }
                }
            }
        } else {
            normalizedDepthMap = cv::Mat::zeros(depthMap.size(), CV_8U);
        }
    } else {
        normalizedDepthMap = cv::Mat::zeros(depthMap.size(), CV_8U);
        std::cerr << "Warning: No valid depth values, creating empty depth map" << std::endl;
    }

    if (!cv::imwrite(depthImagePath, normalizedDepthMap)) {
        std::cerr << "Error: Cannot save normalized depth map to " << depthImagePath << std::endl;
        return -1;
    }
    std::cout << "Successfully saved normalized depth map to: " << depthImagePath << std::endl;

    // **New: Create color depth map**
    cv::Mat colorDepth;
    cv::applyColorMap(normalizedDepthMap, colorDepth, cv::COLORMAP_JET);
    cv::imwrite(outputBasePath + "depth_color.png", colorDepth);
    std::cout << "Successfully saved color depth map" << std::endl;

    // Save original depth map (float format)
    // std::string rawDepthPath = outputBasePath + "depth_raw.exr";
    
    // if (cv::imwrite(rawDepthPath, depthMap)) {
    //     std::cout << "Successfully saved original depth map to: " << rawDepthPath << std::endl;
    // } else {
    //     std::cerr << "Error: Cannot save original depth map to " << rawDepthPath << std::endl;
    //     return -1;
    // }

    std::cout << "\n7. Mesh reconstruction..." << std::endl;

    // **Modification: Use color information for mesh reconstruction**
    std::string meshOutputPath = outputBasePath + "reconstructed_mesh.ply";
    std::cout << "Start 3D mesh reconstruction (including color information)..." << std::endl;
    
    // If MeshReconstruction supports color information, pass rectified color images
    if (!MeshReconstruction::reconstructAndSaveMesh(depthMap, rectL_color, meshOutputPath)) {
        std::cerr << "Error: 3D mesh reconstruction failed!" << std::endl;
        return -1;
    }
    std::cout << "Successfully saved 3D mesh to: " << meshOutputPath << std::endl;
    
    std::cout << "\n8. Processing completed! Output file summary:" << std::endl;
    std::cout << "========================================" << std::endl;
    
    std::cout << "\n📷 Original images:" << std::endl;
    std::cout << "  - " << outputBasePath << "left_original.png" << std::endl;
    std::cout << "  - " << outputBasePath << "right_original.png" << std::endl;
    
    std::cout << "\n🔧 Rectified images:" << std::endl;
    std::cout << "  - " << outputBasePath << "left_rectified.png" << std::endl;
    std::cout << "  - " << outputBasePath << "right_rectified.png" << std::endl;
    
    std::cout << "\n🎯 Feature matching:" << std::endl;
    std::cout << "  - " << outputBasePath << "feature_matches_color.png" << std::endl;
    
    std::cout << "\n📊 Disparity map:" << std::endl;
    std::cout << "  - " << outputPath << " (Grayscale)" << std::endl;
    std::cout << "  - " << outputBasePath << "disparity_color_jet.png (Color-JET)" << std::endl;
    std::cout << "  - " << outputBasePath << "disparity_color_hot.png (Color-HOT)" << std::endl;
    std::cout << "  - " << outputBasePath << "disparity_blended.png (Blended image)" << std::endl;
    std::cout << "  - " << outputBasePath << "disparity_blended_strong.png (Strong blended image)" << std::endl;
    
    std::cout << "\n📏 Depth map:" << std::endl;
    std::cout << "  - " << depthImagePath << " (Normalized)" << std::endl;
    std::cout << "  - " << outputBasePath << "depth_color.png (Color)" << std::endl;
    //std::cout << "  - " << rawDepthPath << " (Original float)" << std::endl;
    
    std::cout << "\n🧊 3D mesh:" << std::endl;
    std::cout << "  - " << meshOutputPath << " (PLY format)" << std::endl;
    
    // **New: Quality assessment report**
    cv::Mat validMask = disparity > 0;
    int validPixels = cv::countNonZero(validMask);
    int totalPixels = disparity.rows * disparity.cols;
    float validRatio = static_cast<float>(validPixels) / totalPixels;
    
    std::cout << "\n📈 Quality assessment:" << std::endl;
    std::cout << "  - Image size: " << imgL.cols << "x" << imgL.rows << std::endl;
    std::cout << "  - Feature matching point count: " << ptsL.size() << std::endl;
    std::cout << "  - Valid disparity pixel ratio: " << (validRatio * 100) << "%" << std::endl;
    std::cout << "  - Disparity range: 0-" << numDisparities << " pixels" << std::endl;
    
    std::cout << "\n=== Stereo Vision Processing System completed ===" << std::endl;
    
    return 0;
}