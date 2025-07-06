#include "8point.h"
#include <iostream>

#include <opencv2/calib3d.hpp>

namespace EightPoint {
bool estimatePose(const std::vector<cv::Point2f>& ptsL,
                  const std::vector<cv::Point2f>& ptsR,
                  const cv::Mat& K,
                  cv::Mat& R,
                  cv::Mat& t) {

    if (ptsL.size() < 8 || ptsR.size() < 8) {
        std::cerr << "Error: Insufficient matching point count (at least 8 points are required)" << std::endl;
        return false;
    }

    // 8 point method to estimate the essential matrix
    cv::Mat E = cv::findEssentialMat(ptsL, ptsR, K, cv::RANSAC);

    if (E.empty()) {
        std::cerr << "Error: Failed to estimate essential matrix" << std::endl;
        return false;
    }

    // Recover pose from essential matrix
    int inliers = cv::recoverPose(E, ptsL, ptsR, K, R, t);

    if (inliers < 8) {
        std::cerr << "Warning: Fewer inliers (" << inliers << ")" << std::endl;
    }

    return true;
}
} // namespace EightPoint 