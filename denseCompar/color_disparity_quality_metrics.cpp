#include "color_disparity_quality_metrics.h"
#include <opencv2/opencv.hpp>
#include <iostream>
#include <fstream>
#include <algorithm>
#include <cmath>

ColorDisparityQualityMetrics ColorDisparityQualityEvaluator::evaluateQuality(const cv::Mat& colorDisparity) {
    ColorDisparityQualityMetrics metrics;
    
    // 确保图像是彩色图像
    cv::Mat colorImage;
    if (colorDisparity.channels() == 1) {
        cv::cvtColor(colorDisparity, colorImage, cv::COLOR_GRAY2BGR);
    } else {
        colorImage = colorDisparity.clone();
    }
    
    // 计算基本质量指标
    metrics.contrast = computeContrast(colorImage);
    metrics.saturation = computeSaturation(colorImage);
    metrics.brightness = computeBrightness(colorImage);
    metrics.sharpness = computeSharpness(colorImage);
    
    // 计算连续性指标
    metrics.smoothness = computeSmoothness(colorImage);
    metrics.edgePreservation = computeEdgePreservation(colorImage);
    metrics.noiseLevel = computeNoiseLevel(colorImage);
    
    // 计算结构指标
    metrics.gradientMagnitude = computeGradientMagnitude(colorImage);
    metrics.textureRichness = computeTextureRichness(colorImage);
    
    // 计算视觉质量指标
    metrics.colorConsistency = computeColorConsistency(colorImage);
    metrics.depthPerception = computeDepthPerception(colorImage);
    
    // 计算综合视觉质量分数
    metrics.visualQuality = computeVisualQualityScore(metrics);
    
    return metrics;
}

double ColorDisparityQualityEvaluator::computeContrast(const cv::Mat& image) {
    cv::Mat gray;
    if (image.channels() == 3) {
        cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);
    } else {
        gray = image.clone();
    }
    
    cv::Scalar mean, stddev;
    cv::meanStdDev(gray, mean, stddev);
    
    // 对比度 = 标准差 / 均值
    double contrast = stddev[0] / (mean[0] + 1e-6);
    return std::min(contrast, 1.0); // 归一化到[0,1]
}

double ColorDisparityQualityEvaluator::computeSaturation(const cv::Mat& image) {
    if (image.channels() != 3) return 0.0;
    
    cv::Mat hsv;
    cv::cvtColor(image, hsv, cv::COLOR_BGR2HSV);
    
    std::vector<cv::Mat> channels;
    cv::split(hsv, channels);
    
    // 计算饱和度通道的平均值
    cv::Scalar mean = cv::mean(channels[1]);
    return mean[0] / 255.0; // 归一化到[0,1]
}

double ColorDisparityQualityEvaluator::computeBrightness(const cv::Mat& image) {
    cv::Mat gray;
    if (image.channels() == 3) {
        cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);
    } else {
        gray = image.clone();
    }
    
    cv::Scalar mean = cv::mean(gray);
    return mean[0] / 255.0; // 归一化到[0,1]
}

double ColorDisparityQualityEvaluator::computeSharpness(const cv::Mat& image) {
    cv::Mat gray;
    if (image.channels() == 3) {
        cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);
    } else {
        gray = image.clone();
    }
    
    // 使用拉普拉斯算子计算锐度
    cv::Mat laplacian;
    cv::Laplacian(gray, laplacian, CV_64F);
    
    cv::Scalar mean, stddev;
    cv::meanStdDev(laplacian, mean, stddev);
    
    // 锐度 = 拉普拉斯响应的方差
    double sharpness = stddev[0] * stddev[0];
    return std::min(sharpness / 1000.0, 1.0); // 归一化到[0,1]
}

double ColorDisparityQualityEvaluator::computeSmoothness(const cv::Mat& image) {
    cv::Mat gray;
    if (image.channels() == 3) {
        cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);
    } else {
        gray = image.clone();
    }
    
    // 计算梯度
    cv::Mat gradX, gradY;
    cv::Sobel(gray, gradX, CV_64F, 1, 0);
    cv::Sobel(gray, gradY, CV_64F, 0, 1);
    
    cv::Mat gradientMagnitude;
    cv::magnitude(gradX, gradY, gradientMagnitude);
    
    // 平滑度 = 1 - 平均梯度幅值
    cv::Scalar mean = cv::mean(gradientMagnitude);
    double smoothness = 1.0 - std::min(mean[0] / 100.0, 1.0);
    return std::max(smoothness, 0.0);
}

double ColorDisparityQualityEvaluator::computeEdgePreservation(const cv::Mat& image) {
    cv::Mat gray;
    if (image.channels() == 3) {
        cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);
    } else {
        gray = image.clone();
    }
    
    // 使用Canny边缘检测
    cv::Mat edges;
    cv::Canny(gray, edges, 50, 150);
    
    // 计算边缘像素比例
    int edgePixels = cv::countNonZero(edges);
    int totalPixels = edges.rows * edges.cols;
    
    double edgeRatio = static_cast<double>(edgePixels) / totalPixels;
    return std::min(edgeRatio * 10.0, 1.0); // 归一化到[0,1]
}

double ColorDisparityQualityEvaluator::computeNoiseLevel(const cv::Mat& image) {
    cv::Mat gray;
    if (image.channels() == 3) {
        cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);
    } else {
        gray = image.clone();
    }
    
    // 使用高斯滤波估计噪声
    cv::Mat blurred;
    cv::GaussianBlur(gray, blurred, cv::Size(5, 5), 0);
    
    cv::Mat noise = gray - blurred;
    cv::Scalar mean, stddev;
    cv::meanStdDev(noise, mean, stddev);
    
    // 噪声水平 = 噪声的标准差
    double noiseLevel = stddev[0] / 255.0;
    return std::min(noiseLevel, 1.0);
}

double ColorDisparityQualityEvaluator::computeStructuralSimilarity(const cv::Mat& image1, const cv::Mat& image2) {
    // 简化的结构相似性计算
    cv::Mat gray1, gray2;
    if (image1.channels() == 3) {
        cv::cvtColor(image1, gray1, cv::COLOR_BGR2GRAY);
    } else {
        gray1 = image1.clone();
    }
    
    if (image2.channels() == 3) {
        cv::cvtColor(image2, gray2, cv::COLOR_BGR2GRAY);
    } else {
        gray2 = image2.clone();
    }
    
    // 确保图像大小一致
    if (gray1.size() != gray2.size()) {
        cv::resize(gray2, gray2, gray1.size());
    }
    
    // 计算相关系数作为结构相似性的近似
    cv::Mat correlation;
    cv::matchTemplate(gray1, gray2, correlation, cv::TM_CCOEFF_NORMED);
    
    return correlation.at<double>(0, 0);
}

double ColorDisparityQualityEvaluator::computeGradientMagnitude(const cv::Mat& image) {
    cv::Mat gray;
    if (image.channels() == 3) {
        cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);
    } else {
        gray = image.clone();
    }
    
    cv::Mat gradX, gradY;
    cv::Sobel(gray, gradX, CV_64F, 1, 0);
    cv::Sobel(gray, gradY, CV_64F, 0, 1);
    
    cv::Mat gradientMagnitude;
    cv::magnitude(gradX, gradY, gradientMagnitude);
    
    cv::Scalar mean = cv::mean(gradientMagnitude);
    return std::min(mean[0] / 100.0, 1.0);
}

double ColorDisparityQualityEvaluator::computeTextureRichness(const cv::Mat& image) {
    cv::Mat gray;
    if (image.channels() == 3) {
        cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);
    } else {
        gray = image.clone();
    }
    
    // 使用局部二值模式(LBP)的简化版本
    cv::Mat lbp = cv::Mat::zeros(gray.size(), CV_8U);
    
    for (int i = 1; i < gray.rows - 1; i++) {
        for (int j = 1; j < gray.cols - 1; j++) {
            uchar center = gray.at<uchar>(i, j);
            uchar code = 0;
            
            // 简化的LBP计算
            if (gray.at<uchar>(i-1, j-1) > center) code |= 1;
            if (gray.at<uchar>(i-1, j) > center) code |= 2;
            if (gray.at<uchar>(i-1, j+1) > center) code |= 4;
            if (gray.at<uchar>(i, j+1) > center) code |= 8;
            if (gray.at<uchar>(i+1, j+1) > center) code |= 16;
            if (gray.at<uchar>(i+1, j) > center) code |= 32;
            if (gray.at<uchar>(i+1, j-1) > center) code |= 64;
            if (gray.at<uchar>(i, j-1) > center) code |= 128;
            
            lbp.at<uchar>(i, j) = code;
        }
    }
    
    // 计算LBP的直方图
    cv::Mat hist;
    int histSize[] = {256};
    float ranges[] = {0, 256};
    const int* channels = {0};
    const float* rangesPtr[] = {ranges};
    cv::calcHist(&lbp, 1, channels, cv::Mat(), hist, 1, histSize, rangesPtr);
    
    // 纹理丰富度 = 非零直方图bin的数量
    int nonZeroBins = cv::countNonZero(hist);
    return std::min(static_cast<double>(nonZeroBins) / 256.0, 1.0);
}

double ColorDisparityQualityEvaluator::computeColorConsistency(const cv::Mat& image) {
    if (image.channels() != 3) return 0.0;
    
    cv::Mat hsv;
    cv::cvtColor(image, hsv, cv::COLOR_BGR2HSV);
    
    std::vector<cv::Mat> channels;
    cv::split(hsv, channels);
    
    // 计算色调和饱和度的标准差
    cv::Scalar meanH, stddevH, meanS, stddevS;
    cv::meanStdDev(channels[0], meanH, stddevH);
    cv::meanStdDev(channels[1], meanS, stddevS);
    
    // 颜色一致性 = 1 - (色调标准差 + 饱和度标准差) / 255
    double consistency = 1.0 - (stddevH[0] + stddevS[0]) / 255.0;
    return std::max(consistency, 0.0);
}

double ColorDisparityQualityEvaluator::computeDepthPerception(const cv::Mat& image) {
    // 深度感知质量基于色差图的颜色分布
    if (image.channels() != 3) return 0.0;
    
    cv::Mat hsv;
    cv::cvtColor(image, hsv, cv::COLOR_BGR2HSV);
    
    std::vector<cv::Mat> channels;
    cv::split(hsv, channels);
    
    // 计算色调的分布
    cv::Mat hist;
    int histSize[] = {180}; // 色调范围0-179
    float ranges[] = {0, 180};
    const int* channels_hist = {0};
    const float* rangesPtr[] = {ranges};
    cv::calcHist(&channels[0], 1, channels_hist, cv::Mat(), hist, 1, histSize, rangesPtr);
    
    // 深度感知质量 = 色调分布的均匀性
    cv::Scalar mean, stddev;
    cv::meanStdDev(hist, mean, stddev);
    
    double uniformity = 1.0 - stddev[0] / (mean[0] + 1e-6);
    return std::max(uniformity, 0.0);
}

double ColorDisparityQualityEvaluator::computeVisualQualityScore(const ColorDisparityQualityMetrics& metrics) {
    // 综合视觉质量分数，权重可以根据需要调整
    double score = 0.0;
    
    // 基本质量指标权重
    score += 0.15 * metrics.contrast;
    score += 0.10 * metrics.saturation;
    score += 0.10 * metrics.brightness;
    score += 0.15 * metrics.sharpness;
    
    // 连续性指标权重
    score += 0.10 * metrics.smoothness;
    score += 0.15 * metrics.edgePreservation;
    score -= 0.10 * metrics.noiseLevel; // 噪声是负面指标
    
    // 结构指标权重
    score += 0.05 * metrics.gradientMagnitude;
    score += 0.05 * metrics.textureRichness;
    
    // 视觉质量指标权重
    score += 0.10 * metrics.colorConsistency;
    score += 0.05 * metrics.depthPerception;
    
    return std::max(std::min(score, 1.0), 0.0);
}

std::string ColorDisparityQualityEvaluator::generateQualityReport(const ColorDisparityQualityMetrics& metrics, 
                                                                 const std::string& algorithmName) {
    std::stringstream report;
    report << "=== Color Disparity Quality Report for " << algorithmName << " ===" << std::endl;
    report << std::fixed << std::setprecision(4);
    
    report << "\nBasic Quality Metrics:" << std::endl;
    report << "  Contrast: " << metrics.contrast << " (0-1, higher is better)" << std::endl;
    report << "  Saturation: " << metrics.saturation << " (0-1, higher is better)" << std::endl;
    report << "  Brightness: " << metrics.brightness << " (0-1, moderate is best)" << std::endl;
    report << "  Sharpness: " << metrics.sharpness << " (0-1, higher is better)" << std::endl;
    
    report << "\nContinuity Metrics:" << std::endl;
    report << "  Smoothness: " << metrics.smoothness << " (0-1, higher is better)" << std::endl;
    report << "  Edge Preservation: " << metrics.edgePreservation << " (0-1, higher is better)" << std::endl;
    report << "  Noise Level: " << metrics.noiseLevel << " (0-1, lower is better)" << std::endl;
    
    report << "\nStructural Metrics:" << std::endl;
    report << "  Gradient Magnitude: " << metrics.gradientMagnitude << " (0-1, moderate is best)" << std::endl;
    report << "  Texture Richness: " << metrics.textureRichness << " (0-1, higher is better)" << std::endl;
    
    report << "\nVisual Quality Metrics:" << std::endl;
    report << "  Color Consistency: " << metrics.colorConsistency << " (0-1, higher is better)" << std::endl;
    report << "  Depth Perception: " << metrics.depthPerception << " (0-1, higher is better)" << std::endl;
    
    report << "\nOverall Quality Score: " << metrics.visualQuality << " (0-1, higher is better)" << std::endl;
    
    return report.str();
}

std::string ColorDisparityQualityEvaluator::compareAlgorithmsQuality(
    const std::vector<std::pair<std::string, ColorDisparityQualityMetrics>>& results) {
    
    std::stringstream comparison;
    comparison << "=== Color Disparity Quality Comparison ===" << std::endl;
    comparison << std::fixed << std::setprecision(4);
    
    // 找出最佳算法
    auto bestAlgorithm = std::max_element(results.begin(), results.end(),
        [](const auto& a, const auto& b) {
            return a.second.visualQuality < b.second.visualQuality;
        });
    
    comparison << "\nBest Overall Quality: " << bestAlgorithm->first 
               << " (Score: " << bestAlgorithm->second.visualQuality << ")" << std::endl;
    
    comparison << "\nDetailed Comparison:" << std::endl;
    comparison << std::setw(15) << "Algorithm" 
               << std::setw(12) << "Quality" 
               << std::setw(12) << "Contrast" 
               << std::setw(12) << "Sharpness" 
               << std::setw(12) << "Smoothness" 
               << std::setw(12) << "Noise" << std::endl;
    comparison << std::string(75, '-') << std::endl;
    
    for (const auto& result : results) {
        comparison << std::setw(15) << result.first
                   << std::setw(12) << result.second.visualQuality
                   << std::setw(12) << result.second.contrast
                   << std::setw(12) << result.second.sharpness
                   << std::setw(12) << result.second.smoothness
                   << std::setw(12) << result.second.noiseLevel << std::endl;
    }
    
    return comparison.str();
}

bool ColorDisparityQualityEvaluator::saveQualityReport(const std::string& filename, 
    const std::vector<std::pair<std::string, ColorDisparityQualityMetrics>>& results) {
    
    std::ofstream file(filename);
    if (!file.is_open()) {
        return false;
    }
    
    file << "Color Disparity Quality Evaluation Report" << std::endl;
    file << "Generated on: " << std::chrono::system_clock::now().time_since_epoch().count() << std::endl;
    file << std::fixed << std::setprecision(6);
    
    for (const auto& result : results) {
        file << "\n" << generateQualityReport(result.second, result.first);
    }
    
    file << "\n" << compareAlgorithmsQuality(results);
    
    file.close();
    return true;
} 