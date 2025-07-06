#pragma once
#include <opencv2/opencv.hpp>
#include <memory>
#include "sgbmMatcher.h"
#include "bmMatcher.h"

// 支持的密集匹配算法类型
enum class DenseAlgorithmType {
    SGBM,
    BM
    // 以后可扩展更多算法
};

// 前向声明
class SGBMMatcher;
class BMMatcher;

class DenseMatcher {
public:
    DenseMatcher(const cv::Mat& K, const cv::Mat& distCoeffs,
                 int numDisparities, int blockSize, DenseAlgorithmType algo = DenseAlgorithmType::SGBM);

    // 立体校正
    bool rectifyImages(const cv::Mat& imgL, const cv::Mat& imgR,
                      const cv::Mat& R, const cv::Mat& t,
                      cv::Mat& rectL, cv::Mat& rectR);

    // 视差计算（自动分发到具体算法）
    bool computeDisparityMap(const cv::Mat& rectL, const cv::Mat& rectR,
                             cv::Mat& disparity);

    // 获取Q矩阵
    const cv::Mat& getQMatrix() const;

private:
    cv::Mat K_, distCoeffs_;
    int numDisparities_, blockSize_;
    DenseAlgorithmType algo_;
    cv::Mat Q_;
    std::unique_ptr<SGBMMatcher> sgbm_;
    std::unique_ptr<BMMatcher> bm_;
}; 