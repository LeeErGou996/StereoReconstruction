#pragma once
#include <opencv2/opencv.hpp>

class SGBMMatcher {
public:
    SGBMMatcher(const cv::Mat& K, const cv::Mat& distCoeffs, int numDisparities, int blockSize);
    bool computeDisparityMap(const cv::Mat& rectL, const cv::Mat& rectR, cv::Mat& disparity);
private:
    cv::Mat K_, distCoeffs_;
    int numDisparities_, blockSize_;
}; 