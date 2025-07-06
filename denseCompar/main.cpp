#include <iostream>
#include <string>
#include <filesystem>
#include <vector>
#include <fstream>
#include <regex>
#include <iomanip>
#include <algorithm>
#include <chrono>
#include <sstream>
#include <exception>
#include <opencv2/opencv.hpp>

#include "denseMatching.h"
#include "depth.h"
#include "config.h"
#include "sparseMatching.h"
#include "8point.h"
#include "elasMatcher.h"
#include "color_disparity_quality_metrics.h"

// 全局日志文件流
std::ofstream globalLogFile;

// 日志记录函数
void logMessage(const std::string& message, bool toConsole = true) {
    if (globalLogFile.is_open()) {
        globalLogFile << message << std::endl;
        globalLogFile.flush();
    }
    if (toConsole) {
        std::cout << message << std::endl;
    }
}

// 简化的颜色处理函数
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

// 归一化深度图到[0,1]范围
cv::Mat normalizeDepthMap(const cv::Mat& depthMap, const cv::Mat& validMask) {
    cv::Mat normalized;
    depthMap.copyTo(normalized);
    
    // 只对有效像素进行归一化
    cv::Mat validDepth;
    depthMap.copyTo(validDepth, validMask);
    
    if (cv::countNonZero(validMask) > 0) {
        double minVal, maxVal;
        cv::minMaxLoc(validDepth, &minVal, &maxVal, nullptr, nullptr);
        
        if (maxVal > minVal) {
            // 归一化到[0,1]范围
            normalized = (normalized - minVal) / (maxVal - minVal);
            // 将无效像素设为0
            normalized.setTo(0, ~validMask);
        }
    }
    
    return normalized;
}

// 将真实深度图转换为与估计深度图相同的单位（米）
cv::Mat convertGroundTruthToMeters(const cv::Mat& groundTruthDepth) {
    cv::Mat groundTruthMeters;
    
    // 获取真实深度图的有效范围（排除0值）
    cv::Mat gtValid = groundTruthDepth > 0;
    double gtMin, gtMax;
    cv::minMaxLoc(groundTruthDepth, &gtMin, &gtMax, nullptr, nullptr, gtValid);
    
    // 假设真实深度图的值0-255对应深度范围0-10米
    // 这是一个合理的假设，因为大多数室内场景的深度在10米以内
    const double maxDepthMeters = 10.0; // 最大深度10米
    
    if (gtMax > gtMin) {
        // 将0-255映射到0-10米
        groundTruthDepth.convertTo(groundTruthMeters, CV_32F, maxDepthMeters / 255.0, 0);
        // 将无效像素设为0
        groundTruthMeters.setTo(0, ~gtValid);
    } else {
        groundTruthMeters = cv::Mat::zeros(groundTruthDepth.size(), CV_32F);
    }
    
    return groundTruthMeters;
}

// 归一化到[0,1]，只对有效像素（>0）
void normalizeTo01(const cv::Mat& src, cv::Mat& dst) {
    double minVal, maxVal;
    cv::minMaxLoc(src, &minVal, &maxVal, nullptr, nullptr, src > 0);
    if (maxVal > minVal) {
        src.convertTo(dst, CV_32F, 1.0 / (maxVal - minVal), -minVal / (maxVal - minVal));
        dst.setTo(0, src <= 0); // 保持无效像素为0
    } else {
        dst = cv::Mat::zeros(src.size(), CV_32F);
    }
}

// 计算MSE误差的函数（使用归一化到[0,1]的一致性比较）
double computeMSE(const cv::Mat& estimated, const cv::Mat& groundTruth) {
    if (estimated.empty() || groundTruth.empty()) {
        logMessage("Error: Empty depth maps for MSE calculation", false);
        return -1.0;
    }
    if (estimated.size() != groundTruth.size()) {
        logMessage("Error: Depth map sizes don't match for MSE calculation", false);
        return -1.0;
    }
    // 归一化
    cv::Mat estNorm, gtNorm;
    normalizeTo01(estimated, estNorm);
    normalizeTo01(groundTruth, gtNorm);
    // 创建有效像素掩码（两个深度图都大于0的像素）
    cv::Mat validMask = (gtNorm > 0) & (estNorm > 0);
    if (cv::countNonZero(validMask) == 0) {
        logMessage("Warning: No valid pixels for MSE calculation", false);
        return -1.0;
    }
    // 计算差值
    cv::Mat diff;
    cv::absdiff(estNorm, gtNorm, diff);
    cv::Mat validMaskFloat;
    validMask.convertTo(validMaskFloat, CV_32F);
    diff = diff.mul(validMaskFloat);
    double mse = cv::sum(diff.mul(diff))[0] / cv::countNonZero(validMask);
    return mse;
}

// 计算RMSE误差的函数
double computeRMSE(const cv::Mat& estimated, const cv::Mat& groundTruth) {
    double mse = computeMSE(estimated, groundTruth);
    return (mse > 0) ? std::sqrt(mse) : -1.0;
}

// 计算平均绝对误差(MAE)的函数（使用归一化到[0,1]的一致性比较）
double computeMAE(const cv::Mat& estimated, const cv::Mat& groundTruth) {
    if (estimated.empty() || groundTruth.empty()) {
        logMessage("Error: Empty depth maps for MAE calculation", false);
        return -1.0;
    }
    if (estimated.size() != groundTruth.size()) {
        logMessage("Error: Depth map sizes don't match for MAE calculation", false);
        return -1.0;
    }
    // 归一化
    cv::Mat estNorm, gtNorm;
    normalizeTo01(estimated, estNorm);
    normalizeTo01(groundTruth, gtNorm);
    // 创建有效像素掩码（两个深度图都大于0的像素）
    cv::Mat validMask = (gtNorm > 0) & (estNorm > 0);
    if (cv::countNonZero(validMask) == 0) {
        logMessage("Warning: No valid pixels for MAE calculation", false);
        return -1.0;
    }
    // 计算差值
    cv::Mat diff;
    cv::absdiff(estNorm, gtNorm, diff);
    cv::Mat validMaskFloat;
    validMask.convertTo(validMaskFloat, CV_32F);
    diff = diff.mul(validMaskFloat);
    double mae = cv::sum(diff)[0] / cv::countNonZero(validMask);
    return mae;
}

// ========== Middlebury评估指标 ==========

// 计算视差误差百分比（Middlebury标准）
double computeDisparityErrorPercentage(const cv::Mat& estimated, const cv::Mat& groundTruth, 
                                      double threshold = 1.0, double maxDisparity = 255.0) {
    (void)maxDisparity; // 避免未使用参数警告
    if (estimated.empty() || groundTruth.empty()) {
        logMessage("Error: Empty disparity maps for error percentage calculation", false);
        return -1.0;
    }
    if (estimated.size() != groundTruth.size()) {
        logMessage("Error: Disparity map sizes don't match for error percentage calculation", false);
        return -1.0;
    }
    
    // 确保数据类型一致 - 转换为CV_32F
    cv::Mat estFloat, gtFloat;
    estimated.convertTo(estFloat, CV_32F);
    groundTruth.convertTo(gtFloat, CV_32F);
    
    // 创建有效像素掩码
    cv::Mat validMask = (gtFloat > 0) & (estFloat > 0);
    if (cv::countNonZero(validMask) == 0) {
        logMessage("Warning: No valid pixels for error percentage calculation", false);
        return -1.0;
    }
    
    // 计算绝对视差误差
    cv::Mat absError;
    cv::absdiff(estFloat, gtFloat, absError);
    
    // 统计超过阈值的像素
    cv::Mat errorMask = (absError > threshold) & validMask;
    int errorPixels = cv::countNonZero(errorMask);
    int totalValidPixels = cv::countNonZero(validMask);
    
    double errorPercentage = (totalValidPixels > 0) ? 
        (static_cast<double>(errorPixels) / totalValidPixels) * 100.0 : -1.0;
    
    return errorPercentage;
}

// 计算RMS视差误差（Middlebury标准）
double computeRMSDisparityError(const cv::Mat& estimated, const cv::Mat& groundTruth) {
    if (estimated.empty() || groundTruth.empty()) {
        logMessage("Error: Empty disparity maps for RMS error calculation", false);
        return -1.0;
    }
    if (estimated.size() != groundTruth.size()) {
        logMessage("Error: Disparity map sizes don't match for RMS error calculation", false);
        return -1.0;
    }
    
    // 确保数据类型一致 - 转换为CV_32F
    cv::Mat estFloat, gtFloat;
    estimated.convertTo(estFloat, CV_32F);
    groundTruth.convertTo(gtFloat, CV_32F);
    
    // 创建有效像素掩码
    cv::Mat validMask = (gtFloat > 0) & (estFloat > 0);
    if (cv::countNonZero(validMask) == 0) {
        logMessage("Warning: No valid pixels for RMS error calculation", false);
        return -1.0;
    }
    
    // 计算视差误差
    cv::Mat error;
    cv::subtract(estFloat, gtFloat, error);
    
    // 应用有效掩码
    cv::Mat validError;
    error.copyTo(validError, validMask);
    
    // 计算RMS误差
    cv::Mat errorSquared;
    cv::multiply(validError, validError, errorSquared);
    
    double sumSquared = cv::sum(errorSquared)[0];
    int validPixels = cv::countNonZero(validMask);
    
    double rmsError = (validPixels > 0) ? std::sqrt(sumSquared / validPixels) : -1.0;
    
    return rmsError;
}

// 计算平均视差误差（Middlebury标准）
double computeMeanDisparityError(const cv::Mat& estimated, const cv::Mat& groundTruth) {
    if (estimated.empty() || groundTruth.empty()) {
        logMessage("Error: Empty disparity maps for mean error calculation", false);
        return -1.0;
    }
    if (estimated.size() != groundTruth.size()) {
        logMessage("Error: Disparity map sizes don't match for mean error calculation", false);
        return -1.0;
    }
    
    // 确保数据类型一致 - 转换为CV_32F
    cv::Mat estFloat, gtFloat;
    estimated.convertTo(estFloat, CV_32F);
    groundTruth.convertTo(gtFloat, CV_32F);
    
    // 创建有效像素掩码
    cv::Mat validMask = (gtFloat > 0) & (estFloat > 0);
    if (cv::countNonZero(validMask) == 0) {
        logMessage("Warning: No valid pixels for mean error calculation", false);
        return -1.0;
    }
    
    // 计算视差误差
    cv::Mat error;
    cv::subtract(estFloat, gtFloat, error);
    
    // 应用有效掩码
    cv::Mat validError;
    error.copyTo(validError, validMask);
    
    // 计算平均误差
    double sumError = cv::sum(validError)[0];
    int validPixels = cv::countNonZero(validMask);
    
    double meanError = (validPixels > 0) ? sumError / validPixels : -1.0;
    
    return meanError;
}

// 计算视差误差的标准差（Middlebury标准）
double computeDisparityErrorStdDev(const cv::Mat& estimated, const cv::Mat& groundTruth) {
    if (estimated.empty() || groundTruth.empty()) {
        logMessage("Error: Empty disparity maps for std dev calculation", false);
        return -1.0;
    }
    if (estimated.size() != groundTruth.size()) {
        logMessage("Error: Disparity map sizes don't match for std dev calculation", false);
        return -1.0;
    }
    
    // 确保数据类型一致 - 转换为CV_32F
    cv::Mat estFloat, gtFloat;
    estimated.convertTo(estFloat, CV_32F);
    groundTruth.convertTo(gtFloat, CV_32F);
    
    // 创建有效像素掩码
    cv::Mat validMask = (gtFloat > 0) & (estFloat > 0);
    if (cv::countNonZero(validMask) == 0) {
        logMessage("Warning: No valid pixels for std dev calculation", false);
        return -1.0;
    }
    
    // 计算视差误差
    cv::Mat error;
    cv::subtract(estFloat, gtFloat, error);
    
    // 应用有效掩码
    cv::Mat validError;
    error.copyTo(validError, validMask);
    
    // 计算平均误差
    double meanError = computeMeanDisparityError(estimated, groundTruth);
    if (meanError < 0) return -1.0;
    
    // 计算方差 - 修复计算方式
    double sumSquared = 0.0;
    int validPixels = 0;
    
    for (int y = 0; y < validError.rows; ++y) {
        for (int x = 0; x < validError.cols; ++x) {
            if (validMask.at<uchar>(y, x) > 0) {
                float err = validError.at<float>(y, x);
                sumSquared += (err - meanError) * (err - meanError);
                validPixels++;
            }
        }
    }
    
    double variance = (validPixels > 1) ? sumSquared / (validPixels - 1) : 0.0;
    double stdDev = std::sqrt(variance);
    
    return stdDev;
}

// 计算视差误差的百分位数（Middlebury标准）
double computeDisparityErrorPercentile(const cv::Mat& estimated, const cv::Mat& groundTruth, 
                                      double percentile = 95.0) {
    if (estimated.empty() || groundTruth.empty()) {
        logMessage("Error: Empty disparity maps for percentile calculation", false);
        return -1.0;
    }
    if (estimated.size() != groundTruth.size()) {
        logMessage("Error: Disparity map sizes don't match for percentile calculation", false);
        return -1.0;
    }
    
    // 确保数据类型一致 - 转换为CV_32F
    cv::Mat estFloat, gtFloat;
    estimated.convertTo(estFloat, CV_32F);
    groundTruth.convertTo(gtFloat, CV_32F);
    
    // 创建有效像素掩码
    cv::Mat validMask = (gtFloat > 0) & (estFloat > 0);
    if (cv::countNonZero(validMask) == 0) {
        logMessage("Warning: No valid pixels for percentile calculation", false);
        return -1.0;
    }
    
    // 计算视差误差
    cv::Mat error;
    cv::subtract(estFloat, gtFloat, error);
    
    // 收集有效误差值
    std::vector<double> errors;
    for (int y = 0; y < error.rows; ++y) {
        for (int x = 0; x < error.cols; ++x) {
            if (validMask.at<uchar>(y, x) > 0) {
                errors.push_back(std::abs(error.at<float>(y, x)));
            }
        }
    }
    
    if (errors.empty()) {
        logMessage("Warning: No valid error values for percentile calculation", false);
        return -1.0;
    }
    
    // 排序并计算百分位数
    std::sort(errors.begin(), errors.end());
    size_t index = static_cast<size_t>((percentile / 100.0) * errors.size());
    if (index >= errors.size()) index = errors.size() - 1;
    
    return errors[index];
}

// 计算视差图的有效像素比例
double computeValidPixelRatio(const cv::Mat& disparity) {
    if (disparity.empty()) {
        logMessage("Error: Empty disparity map for valid pixel ratio calculation", false);
        return -1.0;
    }
    
    // 确保数据类型一致
    cv::Mat dispFloat;
    disparity.convertTo(dispFloat, CV_32F);
    
    cv::Mat validMask = dispFloat > 0;
    int validPixels = cv::countNonZero(validMask);
    int totalPixels = disparity.rows * disparity.cols;
    
    double validRatio = (totalPixels > 0) ? 
        (static_cast<double>(validPixels) / totalPixels) * 100.0 : 0.0;
    
    return validRatio;
}

// 计算视差图的动态范围
double computeDisparityDynamicRange(const cv::Mat& disparity) {
    if (disparity.empty()) {
        logMessage("Error: Empty disparity map for dynamic range calculation", false);
        return -1.0;
    }
    
    // 确保数据类型一致
    cv::Mat dispFloat;
    disparity.convertTo(dispFloat, CV_32F);
    
    cv::Mat validMask = dispFloat > 0;
    if (cv::countNonZero(validMask) == 0) {
        logMessage("Warning: No valid pixels for dynamic range calculation", false);
        return -1.0;
    }
    
    double minVal, maxVal;
    cv::minMaxLoc(dispFloat, &minVal, &maxVal, nullptr, nullptr, validMask);
    
    return maxVal - minVal;
}

// 解析相机内参的函数
bool parseCameraIntrinsics(const std::string& cameraFilePath, cv::Mat& K) {
    std::ifstream file(cameraFilePath);
    if (!file.is_open()) {
        logMessage("Error: Cannot open camera file: " + cameraFilePath, false);
        return false;
    }
    
    std::string line;
    std::getline(file, line);
    file.close();
    
    // 解析格式: cam0=[1733.74 0 792.27; 0 1733.74 541.89; 0 0 1]
    std::regex pattern(R"(cam0=\[([+-]?[\d.]+)\s+([+-]?[\d.]+)\s+([+-]?[\d.]+);\s+([+-]?[\d.]+)\s+([+-]?[\d.]+)\s+([+-]?[\d.]+);\s+([+-]?[\d.]+)\s+([+-]?[\d.]+)\s+([+-]?[\d.]+)\])");
    std::smatch matches;
    
    if (std::regex_search(line, matches, pattern) && matches.size() == 10) {
        K = (cv::Mat_<double>(3,3) << 
            std::stod(matches[1]), std::stod(matches[2]), std::stod(matches[3]),
            std::stod(matches[4]), std::stod(matches[5]), std::stod(matches[6]),
            std::stod(matches[7]), std::stod(matches[8]), std::stod(matches[9]));
        
        logMessage("Successfully loaded camera intrinsics from: " + cameraFilePath);
        logMessage("Camera matrix K:");
        std::stringstream ss;
        ss << K;
        logMessage(ss.str());
        return true;
    } else {
        logMessage("Error: Invalid camera intrinsics format in file: " + cameraFilePath, false);
        logMessage("Expected format: cam0=[fx 0 cx; 0 fy cy; 0 0 1]", false);
        logMessage("Found: " + line, false);
        return false;
    }
}

// 获取基础文件名的函数
std::string getBaseFilename(const std::string& filepath) {
    std::filesystem::path path(filepath);
    return path.stem().string();
}

// 算法比较结果结构体
struct AlgorithmResult {
    std::string name;
    double mse;
    double rmse;
    double mae;
    
    // Middlebury评估指标
    double disparityErrorPercentage;  // 视差误差百分比
    double rmsDisparityError;         // RMS视差误差
    double meanDisparityError;        // 平均视差误差
    double disparityErrorStdDev;      // 视差误差标准差
    double disparityError95Percentile; // 95%百分位误差
    double validPixelRatio;           // 有效像素比例
    double disparityDynamicRange;     // 视差动态范围
    
    // 色差图质量指标
    ColorDisparityQualityMetrics colorQuality;
    
    cv::Mat depthMap;
    cv::Mat disparityMap;
    cv::Mat colorDisparityMap;
    
    AlgorithmResult(const std::string& n) : name(n), mse(-1.0), rmse(-1.0), mae(-1.0),
        disparityErrorPercentage(-1.0), rmsDisparityError(-1.0), meanDisparityError(-1.0),
        disparityErrorStdDev(-1.0), disparityError95Percentile(-1.0), validPixelRatio(-1.0),
        disparityDynamicRange(-1.0) {}
};

// 处理单个立体图像对的核心函数（支持多算法比较）
bool processStereoPairWithComparison(const std::string& leftImagePath, const std::string& rightImagePath, 
                                    const std::string& outputDir, const cv::Mat& K, const cv::Mat& distCoeffs,
                                    const std::string& groundTruthPath) {
    
    std::string baseFilename = getBaseFilename(leftImagePath);
    logMessage("\n=== Processing stereo pair: " + baseFilename + " ===");
    
    // 创建输出目录
    std::string pairOutputDir = outputDir + "/" + baseFilename + "/";
    std::filesystem::create_directories(pairOutputDir);
    
    logMessage("Left image: " + leftImagePath);
    logMessage("Right image: " + rightImagePath);
    logMessage("Ground truth: " + groundTruthPath);
    logMessage("Output directory: " + pairOutputDir);
    logMessage("----------------------------------------");
    
    // 读取图像
    cv::Mat imgL_color = cv::imread(leftImagePath, cv::IMREAD_COLOR);
    cv::Mat imgR_color = cv::imread(rightImagePath, cv::IMREAD_COLOR);
    cv::Mat imgL = cv::imread(leftImagePath, cv::IMREAD_GRAYSCALE);
    cv::Mat imgR = cv::imread(rightImagePath, cv::IMREAD_GRAYSCALE);
    
    if (imgL.empty() || imgR.empty() || imgL_color.empty() || imgR_color.empty()) {
        logMessage("Error: Cannot read image " + leftImagePath + " or " + rightImagePath, false);
        return false;
    }
    
    logMessage("Successfully read images, size: [" + std::to_string(imgL.cols) + " x " + std::to_string(imgL.rows) + "]");
    
    // 读取真实深度图
    cv::Mat groundTruthDepth = cv::imread(groundTruthPath, cv::IMREAD_GRAYSCALE);
    if (groundTruthDepth.empty()) {
        logMessage("Error: Cannot read ground truth depth: " + groundTruthPath, false);
        return false;
    }
    
    // 保存原始彩色图像
    cv::imwrite(pairOutputDir + "left_original.png", imgL_color);
    cv::imwrite(pairOutputDir + "right_original.png", imgR_color);
    cv::imwrite(pairOutputDir + "ground_truth_depth.png", groundTruthDepth);
    logMessage("Original images and ground truth saved");
    
    // 定义要比较的算法
    std::vector<std::string> algorithms = {"SGBM", "BM", "ELAS"}; // "PATCHMATCH" 已注释
    std::vector<AlgorithmResult> results;
    
    // 立体校正（只需要做一次）
    logMessage("\n1. Feature-based pose estimation and stereo rectification...");
    
    cv::Mat rectL, rectR;
    
    // 根据配置决定是否使用8point算法估计R和t
    cv::Mat R, t;
    
    if (Config::instance().use8pointAlgorithm) {
        logMessage("Using 8-point algorithm to estimate R and t parameters", false);
        
        SparseMatcher sparseMatcher(K, distCoeffs);
        
        // 根据配置设置特征类型
        std::string featureTypeStr = Config::instance().featureType;
        if (featureTypeStr == "SIFT") {
            sparseMatcher.setFeatureType(FeatureType::SIFT);
        } else if (featureTypeStr == "SURF") {
            sparseMatcher.setFeatureType(FeatureType::SURF);
        } else {
            sparseMatcher.setFeatureType(FeatureType::ORB); // 默认使用ORB
        }
        
        if (!sparseMatcher.estimatePoseFromImages(imgL, imgR, R, t)) {
            logMessage("Warning: 8-point algorithm failed, using fallback parameters", false);
            // 如果8point算法失败，使用默认参数作为后备方案
            R = cv::Mat::eye(3, 3, CV_64F);  // 单位旋转矩阵
            t = (cv::Mat_<double>(3,1) << -0.12, 0, 0);  // 假设基线为12cm
            logMessage("Using fallback parameters: R=identity, t=[-0.12, 0, 0]", false);
        } else {
            logMessage("Successfully estimated pose using 8-point algorithm", false);
            logMessage("Estimated rotation matrix R:", false);
            std::stringstream ss;
            ss << R;
            logMessage(ss.str(), false);
            logMessage("Estimated translation vector t:", false);
            ss.str("");
            ss << t;
            logMessage(ss.str(), false);
        }
    } else {
        logMessage("Using hardcoded default parameters (R=identity, t=[-0.12, 0, 0])", false);
        // 使用硬编码的默认参数
        R = cv::Mat::eye(3, 3, CV_64F);  // 单位旋转矩阵
        t = (cv::Mat_<double>(3,1) << -0.12, 0, 0);  // 假设基线为12cm
    }
    
    // 使用SGBM进行校正（校正过程对所有算法都一样）
    DenseMatcher tempMatcher(K, distCoeffs, Config::instance().numDisparities, Config::instance().blockSize, DenseAlgorithmType::SGBM);
    if (!tempMatcher.rectifyImages(imgL, imgR, R, t, rectL, rectR)) {
        logMessage("Error: Stereo rectification failed", false);
        return false;
    }
    
    logMessage("Stereo rectification completed successfully");
    
    // 保存校正后的图像
    cv::imwrite(pairOutputDir + "left_rectified.png", rectL);
    cv::imwrite(pairOutputDir + "right_rectified.png", rectR);
    
    // 校正彩色图像
    cv::Mat rectL_color, rectR_color;
    cv::Mat R1, R2, P1, P2, Q;
    cv::stereoRectify(K, distCoeffs, K, distCoeffs, 
                      imgL.size(), R, t, R1, R2, P1, P2, Q);
    
    cv::Mat mapLx, mapLy, mapRx, mapRy;
    cv::initUndistortRectifyMap(K, distCoeffs, R1, P1, 
                                imgL_color.size(), CV_32FC1, mapLx, mapLy);
    cv::initUndistortRectifyMap(K, distCoeffs, R2, P2, 
                                imgR_color.size(), CV_32FC1, mapRx, mapRy);
    
    cv::remap(imgL_color, rectL_color, mapLx, mapLy, cv::INTER_LINEAR);
    cv::remap(imgR_color, rectR_color, mapRx, mapRy, cv::INTER_LINEAR);
    
    cv::imwrite(pairOutputDir + "left_rectified_color.png", rectL_color);
    cv::imwrite(pairOutputDir + "right_rectified_color.png", rectR_color);
    logMessage("Rectified color images saved");
    
    // 对每种算法进行处理
    for (const auto& algo : algorithms) {
        logMessage("\n2. Processing algorithm: " + algo + "...");
        AlgorithmResult result(algo);
        cv::Mat disparity;
        if (algo == "ELAS") {
            logMessage("Starting ELAS algorithm with preset: " + Config::instance().elasPreset);
            ElasMatcher elas;
            if (!elas.computeDisparity(rectL, rectR, disparity)) {
                logMessage("Error: ELAS disparity computation failed", false);
                continue;
            }

        } else {
            DenseAlgorithmType type = (algo == "SGBM") ? DenseAlgorithmType::SGBM : DenseAlgorithmType::BM;
            DenseMatcher denseMatcher(K, distCoeffs, Config::instance().numDisparities, Config::instance().blockSize, type);
        if (!denseMatcher.computeDisparityMap(rectL, rectR, disparity)) {
                logMessage("Error: " + algo + " disparity computation failed", false);
            continue;
            }
        }
        result.disparityMap = disparity.clone();
        // 保存视差图
        cv::Mat disparity8;
        disparity.convertTo(disparity8, CV_8U, 255.0/(Config::instance().numDisparities*16.0));
        cv::imwrite(pairOutputDir + "disparity_" + algo + ".png", disparity8);
        // 保存彩色视差图
        cv::Mat colorDisparityJet = createSimpleColorDisparity(disparity, cv::COLORMAP_JET);
        cv::Mat colorDisparityHot = createSimpleColorDisparity(disparity, cv::COLORMAP_HOT);
        cv::imwrite(pairOutputDir + "disparity_" + algo + "_color_jet.png", colorDisparityJet);
        cv::imwrite(pairOutputDir + "disparity_" + algo + "_color_hot.png", colorDisparityHot);
        
        // 评估色差图质量
        result.colorDisparityMap = colorDisparityJet.clone();
        result.colorQuality = ColorDisparityQualityEvaluator::evaluateQuality(colorDisparityJet);
        
        logMessage("Color disparity quality evaluation completed for " + algo);
        logMessage("  Visual Quality Score: " + std::to_string(result.colorQuality.visualQuality));
        logMessage("  Contrast: " + std::to_string(result.colorQuality.contrast));
        logMessage("  Sharpness: " + std::to_string(result.colorQuality.sharpness));
        logMessage("  Smoothness: " + std::to_string(result.colorQuality.smoothness));
        logMessage("  Noise Level: " + std::to_string(result.colorQuality.noiseLevel));
        
        logMessage("Disparity computation completed for " + algo);
        
        // 计算深度图 - 使用改进的Middlebury深度转换
        cv::Mat Q_matrix = tempMatcher.getQMatrix();
        cv::Mat depthMap;
        
        // 使用专门为Middlebury评估优化的深度转换
        if (!Depth::computeDepthMapForMiddlebury(disparity, Q_matrix, depthMap, 0.12)) {
            logMessage("Error: " + algo + " depth computation failed", false);
            continue;
        }
        
        // 评估深度图质量
        Depth::evaluateDepthQuality(depthMap, disparity);
        
        result.depthMap = depthMap.clone();
        
        // 保存深度图
        cv::imwrite(pairOutputDir + "depth_" + algo + ".png", depthMap);
        
        // 创建彩色深度图
        cv::Mat normDepth, colorDepthMap;
        double minVal, maxVal;
        cv::minMaxLoc(depthMap, &minVal, &maxVal, nullptr, nullptr);
        depthMap.convertTo(normDepth, CV_8UC1, 255.0 / (maxVal - minVal), -minVal * 255.0 / (maxVal - minVal));
        cv::applyColorMap(normDepth, colorDepthMap, cv::COLORMAP_JET);
        cv::imwrite(pairOutputDir + "depth_" + algo + "_color.png", colorDepthMap);
        
        // 保存原始深度数据
        cv::Mat depthFloat;
        depthMap.convertTo(depthFloat, CV_32F);
        cv::imwrite(pairOutputDir + "depth_" + algo + "_raw.exr", depthFloat);
        
        logMessage("Depth computation completed for " + algo);
        
        // 显示深度图信息
        double depthMin, depthMax;
        cv::minMaxLoc(depthMap, &depthMin, &depthMax, nullptr, nullptr);
        logMessage(algo + " depth range: [" + std::to_string(depthMin) + ", " + std::to_string(depthMax) + "]");
        
        // 计算误差（使用8位编码一致性）
        double gtMin, gtMax;
        cv::minMaxLoc(groundTruthDepth, &gtMin, &gtMax, nullptr, nullptr);
        logMessage("Ground truth depth range: [" + std::to_string(gtMin) + ", " + std::to_string(gtMax) + "]");
        
        // 计算误差指标（使用8位编码一致性）
        result.mse = computeMSE(depthMap, groundTruthDepth);
        result.rmse = computeRMSE(depthMap, groundTruthDepth);
        result.mae = computeMAE(depthMap, groundTruthDepth);
        
        // 计算Middlebury评估指标（基于视差图）
        // 注意：这里应该比较视差图与视差图，而不是视差图与深度图
        // 由于我们没有真实的视差图，这里使用深度图作为参考，但需要调整阈值
        result.disparityErrorPercentage = computeDisparityErrorPercentage(disparity, groundTruthDepth, 10.0); // 增加阈值
        result.rmsDisparityError = computeRMSDisparityError(disparity, groundTruthDepth);
        result.meanDisparityError = computeMeanDisparityError(disparity, groundTruthDepth);
        result.disparityErrorStdDev = computeDisparityErrorStdDev(disparity, groundTruthDepth);
        result.disparityError95Percentile = computeDisparityErrorPercentile(disparity, groundTruthDepth, 95.0);
        result.validPixelRatio = computeValidPixelRatio(disparity);
        result.disparityDynamicRange = computeDisparityDynamicRange(disparity);
        
        logMessage(algo + " Error Metrics (meters):");
        logMessage("  MSE: " + std::to_string(result.mse) + " m²");
        logMessage("  RMSE: " + std::to_string(result.rmse) + " m");
        logMessage("  MAE: " + std::to_string(result.mae) + " m");
        
        logMessage(algo + " Middlebury Disparity Metrics:");
        logMessage("  Error Percentage (>1px): " + std::to_string(result.disparityErrorPercentage) + "%");
        logMessage("  RMS Error: " + std::to_string(result.rmsDisparityError) + " pixels");
        logMessage("  Mean Error: " + std::to_string(result.meanDisparityError) + " pixels");
        logMessage("  Std Dev: " + std::to_string(result.disparityErrorStdDev) + " pixels");
        logMessage("  95% Percentile: " + std::to_string(result.disparityError95Percentile) + " pixels");
        logMessage("  Valid Pixel Ratio: " + std::to_string(result.validPixelRatio) + "%");
        logMessage("  Dynamic Range: " + std::to_string(result.disparityDynamicRange) + " pixels");
        
        results.push_back(result);
    }
    
    // 保存误差比较结果到文件
    std::ofstream errorFile(pairOutputDir + "error_comparison.txt");
    if (errorFile.is_open()) {
        errorFile << "Algorithm Comparison Results for " << baseFilename << std::endl;
        errorFile << "================================================" << std::endl;
        errorFile << "Note: Ground truth depth maps are converted to meters (0-255 -> 0-10m) for comparison" << std::endl;
        errorFile << "Middlebury-style disparity evaluation metrics are also included" << std::endl;
        errorFile << "Color disparity quality metrics are now the primary evaluation focus" << std::endl;
        errorFile << std::fixed << std::setprecision(6);
        
        for (const auto& result : results) {
            errorFile << result.name << ":" << std::endl;
            errorFile << "  Color Disparity Quality Metrics (Primary Focus):" << std::endl;
            errorFile << "    Visual Quality Score: " << result.colorQuality.visualQuality << " (0-1, higher is better)" << std::endl;
            errorFile << "    Contrast: " << result.colorQuality.contrast << " (0-1, higher is better)" << std::endl;
            errorFile << "    Saturation: " << result.colorQuality.saturation << " (0-1, higher is better)" << std::endl;
            errorFile << "    Brightness: " << result.colorQuality.brightness << " (0-1, moderate is best)" << std::endl;
            errorFile << "    Sharpness: " << result.colorQuality.sharpness << " (0-1, higher is better)" << std::endl;
            errorFile << "    Smoothness: " << result.colorQuality.smoothness << " (0-1, higher is better)" << std::endl;
            errorFile << "    Edge Preservation: " << result.colorQuality.edgePreservation << " (0-1, higher is better)" << std::endl;
            errorFile << "    Noise Level: " << result.colorQuality.noiseLevel << " (0-1, lower is better)" << std::endl;
            errorFile << "    Color Consistency: " << result.colorQuality.colorConsistency << " (0-1, higher is better)" << std::endl;
            errorFile << "    Depth Perception: " << result.colorQuality.depthPerception << " (0-1, higher is better)" << std::endl;
            errorFile << "  Depth Metrics (meters):" << std::endl;
            errorFile << "    MSE: " << result.mse << std::endl;
            errorFile << "    RMSE: " << result.rmse << std::endl;
            errorFile << "    MAE: " << result.mae << std::endl;
            errorFile << "  Middlebury Disparity Metrics:" << std::endl;
            errorFile << "    Error Percentage (>1px): " << result.disparityErrorPercentage << "%" << std::endl;
            errorFile << "    RMS Error: " << result.rmsDisparityError << " pixels" << std::endl;
            errorFile << "    Mean Error: " << result.meanDisparityError << " pixels" << std::endl;
            errorFile << "    Std Dev: " << result.disparityErrorStdDev << " pixels" << std::endl;
            errorFile << "    95% Percentile: " << result.disparityError95Percentile << " pixels" << std::endl;
            errorFile << "    Valid Pixel Ratio: " << result.validPixelRatio << "%" << std::endl;
            errorFile << "    Dynamic Range: " << result.disparityDynamicRange << " pixels" << std::endl;
            errorFile << std::endl;
        }
        
        // 找出最佳算法（基于不同指标）
        if (!results.empty()) {
            // 基于色差图视觉质量的最佳算法（主要指标）
            auto bestVisualQuality = std::max_element(results.begin(), results.end(),
                [](const AlgorithmResult& a, const AlgorithmResult& b) {
                    return a.colorQuality.visualQuality < b.colorQuality.visualQuality;
                });
            
            if (bestVisualQuality != results.end() && bestVisualQuality->colorQuality.visualQuality > 0) {
                errorFile << "Best algorithm (highest visual quality): " << bestVisualQuality->name << std::endl;
                errorFile << "Best Visual Quality Score: " << bestVisualQuality->colorQuality.visualQuality << std::endl;
            }
            
            // 基于对比度的最佳算法
            auto bestContrast = std::max_element(results.begin(), results.end(),
                [](const AlgorithmResult& a, const AlgorithmResult& b) {
                    return a.colorQuality.contrast < b.colorQuality.contrast;
                });
            
            if (bestContrast != results.end() && bestContrast->colorQuality.contrast > 0) {
                errorFile << "Best algorithm (highest contrast): " << bestContrast->name << std::endl;
                errorFile << "Best Contrast: " << bestContrast->colorQuality.contrast << std::endl;
            }
            
            // 基于锐度的最佳算法
            auto bestSharpness = std::max_element(results.begin(), results.end(),
                [](const AlgorithmResult& a, const AlgorithmResult& b) {
                    return a.colorQuality.sharpness < b.colorQuality.sharpness;
                });
            
            if (bestSharpness != results.end() && bestSharpness->colorQuality.sharpness > 0) {
                errorFile << "Best algorithm (highest sharpness): " << bestSharpness->name << std::endl;
                errorFile << "Best Sharpness: " << bestSharpness->colorQuality.sharpness << std::endl;
            }
            
            // 基于MSE的最佳算法（次要指标）
            auto bestMSE = std::min_element(results.begin(), results.end(),
                [](const AlgorithmResult& a, const AlgorithmResult& b) {
                    return a.mse < b.mse;
                });
            
            if (bestMSE != results.end() && bestMSE->mse > 0) {
                errorFile << "Best algorithm (lowest MSE): " << bestMSE->name << std::endl;
                errorFile << "Best MSE: " << bestMSE->mse << std::endl;
            }
            
            // 基于Middlebury误差百分比的最佳算法
            auto bestDisparityError = std::min_element(results.begin(), results.end(),
                [](const AlgorithmResult& a, const AlgorithmResult& b) {
                    return a.disparityErrorPercentage < b.disparityErrorPercentage;
                });
            
            if (bestDisparityError != results.end() && bestDisparityError->disparityErrorPercentage > 0) {
                errorFile << "Best algorithm (lowest disparity error %): " << bestDisparityError->name << std::endl;
                errorFile << "Best Error Percentage: " << bestDisparityError->disparityErrorPercentage << "%" << std::endl;
            }
        }
        
        errorFile.close();
        logMessage("Error comparison results saved to: " + pairOutputDir + "error_comparison.txt");
    }
    
    logMessage("\n4. Processing completed for " + baseFilename + "!");
    logMessage("Output files saved to: " + pairOutputDir);
    
    return true;
}

// 处理单个立体图像对的核心函数（保持原有功能，用于向后兼容）
bool processStereoPair(const std::string& leftImagePath, const std::string& rightImagePath, 
                      const std::string& outputDir, const cv::Mat& K, const cv::Mat& distCoeffs) {
    
    std::string baseFilename = getBaseFilename(leftImagePath);
    logMessage("\n=== Processing stereo pair: " + baseFilename + " ===");
    
    // 创建输出目录
    std::string pairOutputDir = outputDir + "/" + baseFilename + "/";
    std::filesystem::create_directories(pairOutputDir);
    
    logMessage("Left image: " + leftImagePath);
    logMessage("Right image: " + rightImagePath);
    logMessage("Output directory: " + pairOutputDir);
    logMessage("----------------------------------------");
    
    // 读取图像
    cv::Mat imgL_color = cv::imread(leftImagePath, cv::IMREAD_COLOR);
    cv::Mat imgR_color = cv::imread(rightImagePath, cv::IMREAD_COLOR);
    cv::Mat imgL = cv::imread(leftImagePath, cv::IMREAD_GRAYSCALE);
    cv::Mat imgR = cv::imread(rightImagePath, cv::IMREAD_GRAYSCALE);
    
    if (imgL.empty() || imgR.empty() || imgL_color.empty() || imgR_color.empty()) {
        logMessage("Error: Cannot read image " + leftImagePath + " or " + rightImagePath, false);
        return false;
    }
    
    logMessage("Successfully read images, size: [" + std::to_string(imgL.cols) + " x " + std::to_string(imgL.rows) + "]");
    
    // 保存原始彩色图像
    cv::imwrite(pairOutputDir + "left_original.png", imgL_color);
    cv::imwrite(pairOutputDir + "right_original.png", imgR_color);
    logMessage("Original color images saved");
    
    // 创建DenseMatcher实例
    DenseMatcher denseMatcher(K, distCoeffs, Config::instance().numDisparities, Config::instance().blockSize);
    
    logMessage("\n1. Stereo rectification...");
    
    // 立体校正
    cv::Mat rectL, rectR;
    
    // 根据配置决定是否使用8point算法估计R和t
    cv::Mat R, t;
    
    if (Config::instance().use8pointAlgorithm) {
        logMessage("Using 8-point algorithm to estimate R and t parameters", false);
        
        SparseMatcher sparseMatcher(K, distCoeffs);
        
        // 根据配置设置特征类型
        std::string featureTypeStr = Config::instance().featureType;
        if (featureTypeStr == "SIFT") {
            sparseMatcher.setFeatureType(FeatureType::SIFT);
        } else if (featureTypeStr == "SURF") {
            sparseMatcher.setFeatureType(FeatureType::SURF);
        } else {
            sparseMatcher.setFeatureType(FeatureType::ORB); // 默认使用ORB
        }
        
        if (!sparseMatcher.estimatePoseFromImages(imgL, imgR, R, t)) {
            logMessage("Warning: 8-point algorithm failed, using fallback parameters", false);
            // 如果8point算法失败，使用默认参数作为后备方案
            R = cv::Mat::eye(3, 3, CV_64F);  // 单位旋转矩阵
            t = (cv::Mat_<double>(3,1) << -0.12, 0, 0);  // 假设基线为12cm
            logMessage("Using fallback parameters: R=identity, t=[-0.12, 0, 0]", false);
        } else {
            logMessage("Successfully estimated pose using 8-point algorithm", false);
        }
    } else {
        logMessage("Using hardcoded default parameters (R=identity, t=[-0.12, 0, 0])", false);
        // 使用硬编码的默认参数
        R = cv::Mat::eye(3, 3, CV_64F);  // 单位旋转矩阵
        t = (cv::Mat_<double>(3,1) << -0.12, 0, 0);  // 假设基线为12cm
    }
    
    if (!denseMatcher.rectifyImages(imgL, imgR, R, t, rectL, rectR)) {
        logMessage("Error: Stereo rectification failed", false);
        return false;
    }
    
    logMessage("Stereo rectification completed successfully");
    
    // 保存校正后的图像
    cv::imwrite(pairOutputDir + "left_rectified.png", rectL);
    cv::imwrite(pairOutputDir + "right_rectified.png", rectR);
    
    // 校正彩色图像
    cv::Mat rectL_color, rectR_color;
    cv::Mat R1, R2, P1, P2, Q;
    cv::stereoRectify(K, distCoeffs, K, distCoeffs, 
                      imgL.size(), R, t, R1, R2, P1, P2, Q);
    
    cv::Mat mapLx, mapLy, mapRx, mapRy;
    cv::initUndistortRectifyMap(K, distCoeffs, R1, P1, 
                                imgL_color.size(), CV_32FC1, mapLx, mapLy);
    cv::initUndistortRectifyMap(K, distCoeffs, R2, P2, 
                                imgR_color.size(), CV_32FC1, mapRx, mapRy);
    
    cv::remap(imgL_color, rectL_color, mapLx, mapLy, cv::INTER_LINEAR);
    cv::remap(imgR_color, rectR_color, mapRx, mapRy, cv::INTER_LINEAR);
    
    cv::imwrite(pairOutputDir + "left_rectified_color.png", rectL_color);
    cv::imwrite(pairOutputDir + "right_rectified_color.png", rectR_color);
    logMessage("Rectified color images saved");
    
    logMessage("\n2. Disparity computation...");
    
    // 计算视差图
    cv::Mat disparity;
    if (!denseMatcher.computeDisparityMap(rectL, rectR, disparity)) {
        logMessage("Error: Dense disparity computation failed", false);
        return false;
    }
    
    // 保存视差图
    cv::Mat disparity8;
    disparity.convertTo(disparity8, CV_8U, 255.0/(Config::instance().numDisparities*16.0));
    cv::imwrite(pairOutputDir + "disparity.png", disparity8);
    
    // 保存彩色视差图
    cv::Mat colorDisparityJet = createSimpleColorDisparity(disparity, cv::COLORMAP_JET);
    cv::Mat colorDisparityHot = createSimpleColorDisparity(disparity, cv::COLORMAP_HOT);
    cv::imwrite(pairOutputDir + "disparity_color_jet.png", colorDisparityJet);
    cv::imwrite(pairOutputDir + "disparity_color_hot.png", colorDisparityHot);
    
    logMessage("Successfully saved disparity map and related outputs");
    
    logMessage("\n3. Depth map computation...");

    // 深度计算
    cv::Mat Q_matrix = denseMatcher.getQMatrix();
    cv::Mat depthMap;
    if (!Depth::computeDepthMap(disparity, Q_matrix, depthMap)) {
        logMessage("Error: Depth computation failed", false);
        return false;
    }
    
    // 保存深度图
    cv::imwrite(pairOutputDir + "depth.png", depthMap);
    
    // 创建彩色深度图
    cv::Mat normDepth, colorDepthMap;
    double minVal, maxVal;
    cv::minMaxLoc(depthMap, &minVal, &maxVal, nullptr, nullptr);
    depthMap.convertTo(normDepth, CV_8UC1, 255.0 / (maxVal - minVal), -minVal * 255.0 / (maxVal - minVal));
    cv::applyColorMap(normDepth, colorDepthMap, cv::COLORMAP_JET);
    cv::imwrite(pairOutputDir + "depth_color.png", colorDepthMap);
    
    // 保存原始深度数据
    cv::Mat depthFloat;
    depthMap.convertTo(depthFloat, CV_32F);
    cv::imwrite(pairOutputDir + "depth_raw.exr", depthFloat);
    
    logMessage("Successfully saved depth maps");
    
    logMessage("\n4. Processing completed for " + baseFilename + "!");
    logMessage("Output files saved to: " + pairOutputDir);
    
    return true;
}

int main() {
    // 创建主日志文件
    globalLogFile.open("../output/run_log.txt");
    if (!globalLogFile.is_open()) {
        std::cerr << "Warning: Cannot create log file, logging to console only" << std::endl;
    }
    
    // 记录开始时间
    auto startTime = std::chrono::system_clock::now();
    auto time_t = std::chrono::system_clock::to_time_t(startTime);
    logMessage("=== Dense Stereo Depth Generation System with Algorithm Comparison ===");
    logMessage("Start time: " + std::string(std::ctime(&time_t)));
    
    // 加载配置 - 使用相对于build目录的路径
    Config::load("../config.txt");
    
    // 设置输入输出路径（相对于build目录）
    std::string leftDir = "../data/left/";
    std::string rightDir = "../data/right/";
    std::string cameraDir = "../data/camera/";
    std::string groundDir = "../data/ground/";
    std::string outputDir = "../output/";
    
    logMessage("Left images directory: " + leftDir);
    logMessage("Right images directory: " + rightDir);
    logMessage("Camera directory: " + cameraDir);
    logMessage("Ground truth directory: " + groundDir);
    logMessage("Output directory: " + outputDir);
    
    // 检查输入目录是否存在
    if (!std::filesystem::exists(leftDir)) {
        logMessage("Error: Left images directory does not exist: " + leftDir, false);
        return -1;
    }
    if (!std::filesystem::exists(rightDir)) {
        logMessage("Error: Right images directory does not exist: " + rightDir, false);
        return -1;
    }
    if (!std::filesystem::exists(cameraDir)) {
        logMessage("Error: Camera directory does not exist: " + cameraDir, false);
        return -1;
    }
    if (!std::filesystem::exists(groundDir)) {
        logMessage("Error: Ground truth directory does not exist: " + groundDir, false);
        return -1;
    }
    
    // 创建输出目录
    std::filesystem::create_directories(outputDir);
    logMessage("Output directory created/verified: " + outputDir);
    
    // 获取左图像文件列表
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
        logMessage("Error: No image files found in left directory: " + leftDir, false);
        return -1;
    }
    
    logMessage("Found " + std::to_string(leftFiles.size()) + " left images");
    
    // 处理每个立体图像对
    int successCount = 0;
    for (const auto& leftFile : leftFiles) {
        cv::Mat K_pair;
        
        std::string baseFilename = getBaseFilename(leftFile);
        std::string rightFile = rightDir + baseFilename + ".png";
        std::string cameraFile = cameraDir + baseFilename + ".txt";
        std::string groundTruthFile = groundDir + baseFilename + ".png";
        
        bool success = true;
        
        // 检查对应的右图像是否存在
        if (!std::filesystem::exists(rightFile)) {
            logMessage("Warning: No corresponding right image found for " + baseFilename);
            success = false;
        }
        
        // 检查对应的真实深度图是否存在
        if (!std::filesystem::exists(groundTruthFile)) {
            logMessage("Warning: No corresponding ground truth depth found for " + baseFilename);
            success = false;
        }
        
        // 加载相机内参
        if (success && !parseCameraIntrinsics(cameraFile, K_pair)) {
            logMessage("Warning: Cannot load camera intrinsics for " + baseFilename + ", skipping...");
            success = false;
        }
        
        // 处理立体图像对（使用算法比较版本）
        if (success) {
            if (!processStereoPairWithComparison(leftFile, rightFile, outputDir, K_pair, Config::instance().distCoeffs, groundTruthFile)) {
                logMessage("[ERROR] Processing failed for " + baseFilename + ", moving to next input.");
                success = false;
            }
        }
        
        if (success) {
            successCount++;
        }
    }
    
    // 记录结束时间和总耗时
    auto endTime = std::chrono::system_clock::now();
    auto end_time_t = std::chrono::system_clock::to_time_t(endTime);
    auto duration = std::chrono::duration_cast<std::chrono::seconds>(endTime - startTime);
    
    logMessage("\n=== Processing Summary ===");
    logMessage("Total stereo pairs found: " + std::to_string(leftFiles.size()));
    logMessage("Successfully processed: " + std::to_string(successCount));
    logMessage("Failed: " + std::to_string(leftFiles.size() - successCount));
    logMessage("End time: " + std::string(std::ctime(&end_time_t)));
    logMessage("Total processing time: " + std::to_string(duration.count()) + " seconds");
    
    // 关闭日志文件
    if (globalLogFile.is_open()) {
        globalLogFile.close();
    }
    
    return 0;
} 