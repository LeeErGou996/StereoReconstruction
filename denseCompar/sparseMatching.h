#ifndef SPARSE_MATCHING_H
#define SPARSE_MATCHING_H

#include <opencv2/opencv.hpp>
#include <vector>

// 特征类型枚举
enum class FeatureType {
    ORB,
    SIFT,
    SURF
};

class SparseMatcher {
public:
    SparseMatcher(const cv::Mat& K, const cv::Mat& distCoeffs);
    
    // 设置特征类型
    void setFeatureType(FeatureType type);
    
    // 创建特征检测器
    cv::Ptr<cv::Feature2D> createDetector(FeatureType type);
    
    // 特征检测和匹配
    bool detectAndMatch(const cv::Mat& imgL, const cv::Mat& imgR,
                       std::vector<cv::Point2f>& ptsL, 
                       std::vector<cv::Point2f>& ptsR);
    
    // 从图像直接估计位姿（完整流程）
    bool estimatePoseFromImages(const cv::Mat& imgL, const cv::Mat& imgR,
                               cv::Mat& R, cv::Mat& t);

private:
    cv::Mat K_;
    cv::Mat distCoeffs_;
    FeatureType featureType_;
    
    // 特征匹配
    bool matchFeatures(const cv::Mat& descL, const cv::Mat& descR,
                      const std::vector<cv::KeyPoint>& kpL,
                      const std::vector<cv::KeyPoint>& kpR,
                      std::vector<cv::Point2f>& ptsL,
                      std::vector<cv::Point2f>& ptsR);
};

#endif // SPARSE_MATCHING_H 