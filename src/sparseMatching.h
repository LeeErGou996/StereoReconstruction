#ifndef SPARSE_MATCHING_H
#define SPARSE_MATCHING_H

#include <opencv2/opencv.hpp>

class SparseMatcher {
public:
    SparseMatcher(const cv::Mat& K, const cv::Mat& distCoeffs,
                  int numDisparities, int blockSize);

    bool computeDisparityMap(const cv::Mat& rectL, const cv::Mat& rectR,
                             cv::Mat& disparity);

private:
    cv::Mat K_;
    cv::Mat distCoeffs_;
    int numDisparities_;
    int blockSize_;
};

#endif // SPARSE_MATCHING_H 