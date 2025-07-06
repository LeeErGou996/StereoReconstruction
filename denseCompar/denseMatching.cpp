#include "denseMatching.h"
#include "sgbmMatcher.h"
#include "bmMatcher.h"
#include <iostream>

DenseMatcher::DenseMatcher(const cv::Mat& K, const cv::Mat& distCoeffs,
                           int numDisparities, int blockSize, DenseAlgorithmType algo)
    : K_(K.clone()), distCoeffs_(distCoeffs.clone()), numDisparities_(numDisparities), 
      blockSize_(blockSize), algo_(algo) {
    if (algo == DenseAlgorithmType::SGBM) {
        sgbm_ = std::make_unique<SGBMMatcher>(K_, distCoeffs_, numDisparities_, blockSize_);
    } else if (algo == DenseAlgorithmType::BM) {
        bm_ = std::make_unique<BMMatcher>(K_, distCoeffs_, numDisparities_, blockSize_);
    }
}

bool DenseMatcher::rectifyImages(const cv::Mat& imgL, const cv::Mat& imgR,
                                 const cv::Mat& R, const cv::Mat& t,
                                 cv::Mat& rectL, cv::Mat& rectR) {
    // 立体校正通用实现
    cv::Mat R1, R2, P1, P2, Qmat;
    cv::stereoRectify(K_, distCoeffs_, K_, distCoeffs_, imgL.size(), R, t, R1, R2, P1, P2, Qmat);
    Q_ = Qmat.clone();
    cv::Mat mapLx, mapLy, mapRx, mapRy;
    cv::initUndistortRectifyMap(K_, distCoeffs_, R1, P1, imgL.size(), CV_32FC1, mapLx, mapLy);
    cv::initUndistortRectifyMap(K_, distCoeffs_, R2, P2, imgR.size(), CV_32FC1, mapRx, mapRy);
    cv::remap(imgL, rectL, mapLx, mapLy, cv::INTER_LINEAR);
    cv::remap(imgR, rectR, mapRx, mapRy, cv::INTER_LINEAR);
    return true;
}

bool DenseMatcher::computeDisparityMap(const cv::Mat& rectL, const cv::Mat& rectR, cv::Mat& disparity) {
    if (algo_ == DenseAlgorithmType::SGBM && sgbm_) {
        return sgbm_->computeDisparityMap(rectL, rectR, disparity);
    } else if (algo_ == DenseAlgorithmType::BM && bm_) {
        return bm_->computeDisparityMap(rectL, rectR, disparity);
    } else {
        std::cerr << "[DenseMatcher] Unknown or uninitialized algorithm!" << std::endl;
        return false;
    }
}

const cv::Mat& DenseMatcher::getQMatrix() const { 
    return Q_; 
} 