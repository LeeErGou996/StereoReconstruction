#include "depth.h"
#include <iostream>

namespace Depth {
bool computeDepthMap(const cv::Mat& disparity,
                     const cv::Mat& Q,
                     cv::Mat& depthMapOut,
                     const std::string& outputPath,
                     bool useCustom) {

    // Debug: Print disparity and Q matrix info
    std::cout << "[DEBUG] Disparity map info: rows=" << disparity.rows 
              << ", cols=" << disparity.cols 
              << ", type=" << disparity.type() 
              << " (CV_16S=" << CV_16S << ", CV_32F=" << CV_32F << ")" << std::endl;
    std::cout << "[DEBUG] Q matrix info: rows=" << Q.rows 
              << ", cols=" << Q.cols 
              << ", type=" << Q.type() << std::endl;

    if (disparity.empty() || Q.empty()) {
        std::cerr << "Error: Disparity map or Q matrix is empty, cannot compute depth map." << std::endl;
        return false;
    }

    // **Fix 1**: Check and convert disparity map data type
    // StereoBM/StereoSGBM output disparity values are usually multiplied by 16
    // Need to convert to float and divide by 16
    // 8-bit disparity map also needs processing
    cv::Mat processedDisparity;
    if (disparity.type() == CV_16S) {
        // StereoBM/StereoSGBM输出的视差值通常乘以16存储
        // 需要转换为浮点数并除以16
        disparity.convertTo(processedDisparity, CV_32F, 1.0/16.0);
        std::cout << "[DEBUG] Converted CV_16S disparity to CV_32F, scale: 1/16" << std::endl;
    } else if (disparity.type() == CV_8U) {
        // 8位视差图也需要处理
        disparity.convertTo(processedDisparity, CV_32F);
        std::cout << "[DEBUG] Converted CV_8U disparity to CV_32F" << std::endl;
    } else if (disparity.type() == CV_32F) {
        processedDisparity = disparity.clone();
        std::cout << "[DEBUG] Disparity map is already CV_32F format" << std::endl;
    } else {
        std::cerr << "Error: Unsupported disparity map data type: " << disparity.type() << std::endl;
        return false;
    }

    // **Fix 2**: Handle invalid disparity values
    // Set negative and zero values to NaN, so reprojectImageTo3D will handle them correctly
    cv::Mat validMask = processedDisparity > 0;
    processedDisparity.setTo(std::numeric_limits<float>::quiet_NaN(), ~validMask);

    // Debug: Check statistics of valid disparity values
    double minDisp, maxDisp;
    cv::minMaxLoc(processedDisparity, &minDisp, &maxDisp, nullptr, nullptr, validMask);
    std::cout << "[DEBUG] Valid disparity value range: [" << minDisp << ", " << maxDisp << "]" << std::endl;
    std::cout << "[DEBUG] Number of valid disparity pixels: " << cv::countNonZero(validMask) 
              << " / " << (disparity.rows * disparity.cols) << std::endl;
    
    // reprojectImageTo3D
    cv::Mat xyz;
    if (useCustom && Q.rows == 4 && Q.cols == 4 && cv::checkRange(Q)) {
        std::cout << "[DEBUG] Using custom depth computation (manual triangulation from disparity)\n";

        float fx = Q.at<double>(2, 3);  // fx = Q[2][3]
        float baseline = -1.0f / Q.at<double>(3, 2); // baseline = -1 / Q[3][2]
        float cx = Q.at<double>(0, 3);  // cx = Q[0][3]
        float cy = Q.at<double>(1, 3);  // cy = Q[1][3]

        cv::Mat xyz(disparity.size(), CV_32FC3);
        depthMapOut = cv::Mat::zeros(disparity.size(), CV_32F);

        for (int y = 0; y < disparity.rows; ++y) {
            for (int x = 0; x < disparity.cols; ++x) {
                float d = processedDisparity.at<float>(y, x);
                if (d > 0.0f) {
                    float Z = fx * baseline / d;
                    float X = (x - cx) * Z / fx;
                    float Y = (y - cy) * Z / fx;
                    xyz.at<cv::Vec3f>(y, x) = cv::Vec3f(X, Y, Z);
                    depthMapOut.at<float>(y, x) = Z;
                } else {
                    xyz.at<cv::Vec3f>(y, x) = cv::Vec3f(0, 0, 0);
                    depthMapOut.at<float>(y, x) = 0;
                }
            }
        }

        std::cout << "[DEBUG] Custom depth map computed manually from disparity.\n";
    }
    else {
        cv::reprojectImageTo3D(processedDisparity, xyz, Q, true, CV_32F);
        std::cout << "[DEBUG] Used OpenCV reprojectImageTo3D\n";

        // Debug: Print some sample xyz values
        std::cout << "[DEBUG] XYZ map info: rows=" << xyz.rows 
                << ", cols=" << xyz.cols 
                << ", type=" << xyz.type() << std::endl;

        // **Fix 3**: More efficient extraction of depth information
        std::vector<cv::Mat> xyzChannels;
        cv::split(xyz, xyzChannels);
        depthMapOut = xyzChannels[2].clone();
    }

    // **Fix 4**: Smarter depth value handling
    // Create valid depth mask
    cv::Mat validDepthMask;
    cv::bitwise_and(
        depthMapOut > 0,           // Depth is positive
        depthMapOut < 10000.0f,    // Depth in reasonable range
        validDepthMask
    );
    
    // Set invalid depth to 0
    depthMapOut.setTo(0, ~validDepthMask);

    // Debug: Depth statistics
    if (cv::countNonZero(validDepthMask) > 0) {
        double minDepth, maxDepth;
        cv::minMaxLoc(depthMapOut, &minDepth, &maxDepth, nullptr, nullptr, validDepthMask);
        std::cout << "[DEBUG] Valid depth range: [" << minDepth << ", " << maxDepth << "] unit" << std::endl;
        std::cout << "[DEBUG] Number of valid depth pixels: " << cv::countNonZero(validDepthMask) << std::endl;
        
        // Show center depth
        int centerX = depthMapOut.cols / 2;
        int centerY = depthMapOut.rows / 2;
        float centerDepth = depthMapOut.at<float>(centerY, centerX);
        std::cout << "[DEBUG] Center depth: " << centerDepth << std::endl;
    } else {
        std::cerr << "Warning: No valid depth value!" << std::endl;
    }

    // **Fix 5**: Improved normalization strategy
    // Use percentile normalization, more robust
    cv::Mat normalizedDepthMap;
    
    if (cv::countNonZero(validDepthMask) > 0) {
        // 使用百分位数归一化，更稳健
        std::vector<float> validDepths;
        for (int r = 0; r < depthMapOut.rows; ++r) {
            for (int c = 0; c < depthMapOut.cols; ++c) {
                if (validDepthMask.at<uchar>(r, c)) {
                    validDepths.push_back(depthMapOut.at<float>(r, c));
                }
            }
        }
        
        if (!validDepths.empty()) {
            std::sort(validDepths.begin(), validDepths.end());
            float minDepth = validDepths[0];
            float maxDepth = validDepths[static_cast<int>(validDepths.size() * 0.98)]; // 98th percentile
            
            std::cout << "[DEBUG] Use 98th percentile normalization: [" << minDepth << ", " << maxDepth << "]" << std::endl;
            
            // Create normalized depth map
            normalizedDepthMap = cv::Mat::zeros(depthMapOut.size(), CV_8U);
            
            for (int r = 0; r < depthMapOut.rows; ++r) {
                for (int c = 0; c < depthMapOut.cols; ++c) {
                    if (validDepthMask.at<uchar>(r, c)) {
                        float depth = depthMapOut.at<float>(r, c);
                        depth = std::max(minDepth, std::min(maxDepth, depth)); // 裁剪到范围
                        uchar normalizedValue = static_cast<uchar>(
                            255.0f * (depth - minDepth) / (maxDepth - minDepth)
                        );
                        normalizedDepthMap.at<uchar>(r, c) = normalizedValue;
                    }
                }
            }
        } else {
            normalizedDepthMap = cv::Mat::zeros(depthMapOut.size(), CV_8U);
        }
    } else {
        normalizedDepthMap = cv::Mat::zeros(depthMapOut.size(), CV_8U);
        std::cerr << "Warning: No valid depth value, create empty depth map" << std::endl;
    }

    return true;
}

// **New feature**: Depth map quality evaluation
bool evaluateDepthQuality(const cv::Mat& depthMap, const cv::Mat& originalDisparity) {
    if (depthMap.empty() || originalDisparity.empty()) {
        std::cerr << "Error: Input image is empty" << std::endl;
        return false;
    }
    
    // Calculate valid pixel ratio
    cv::Mat validMask = depthMap > 0;
    int validPixels = cv::countNonZero(validMask);
    int totalPixels = depthMap.rows * depthMap.cols;
    float validRatio = static_cast<float>(validPixels) / totalPixels;
    
    std::cout << "\n=== Depth map quality evaluation ===" << std::endl;
    std::cout << "Valid pixel ratio: " << (validRatio * 100) << "%" << std::endl;
    
    if (validRatio < 0.3) {
        std::cout << "Warning: Valid depth pixel ratio is low, you may need to adjust stereo matching parameters" << std::endl;
    }
    
    // Calculate depth distribution
    if (validPixels > 0) {
        double minVal, maxVal;
        cv::minMaxLoc(depthMap, &minVal, &maxVal, nullptr, nullptr, validMask);
        
        cv::Scalar meanDepth = cv::mean(depthMap, validMask);
        std::cout << "Depth stats - min: " << minVal 
                  << ", max: " << maxVal 
                  << ", mean: " << meanDepth[0] << std::endl;
    }
    
    return true;
}

} // namespace Depth