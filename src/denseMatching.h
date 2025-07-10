#ifndef DENSE_MATCHING_H
#define DENSE_MATCHING_H

#include <opencv2/opencv.hpp>
#include "elasMatcher.h"

class DenseMatcher {
public:
    DenseMatcher(const cv::Mat& K, const cv::Mat& distCoeffs,
                 int numDisparities, int blockSize);

    bool rectifyImages(const cv::Mat& imgL, const cv::Mat& imgR,
                       const cv::Mat& R, const cv::Mat& t,
                       cv::Mat& rectL, cv::Mat& rectR);

    bool computeDisparityMap(const cv::Mat& rectL, const cv::Mat& rectR,
                             cv::Mat& disparity);

    const cv::Mat& getQMatrix() const;

private:
    cv::Mat K_;
    cv::Mat distCoeffs_;
    int numDisparities_;
    int blockSize_;
    cv::Mat Q_;
};

#endif // DENSE_MATCHING_H 