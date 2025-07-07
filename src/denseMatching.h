#ifndef DENSE_MATCHING_H
#define DENSE_MATCHING_H

#include <opencv2/opencv.hpp>

class DenseMatcher {
public:
    DenseMatcher(const cv::Mat& K, const cv::Mat& distCoeffs,
                 int numDisparities, int blockSize,
                 bool useCustomMatcher = false);

    bool rectifyImages(const cv::Mat& imgL, const cv::Mat& imgR,
                       const cv::Mat& R, const cv::Mat& t,
                       cv::Mat& rectL, cv::Mat& rectR);

    bool computeDisparityMap(const cv::Mat& rectL, const cv::Mat& rectR,
                             cv::Mat& disparity);

    const cv::Mat& getQMatrix() const;

private:
    bool computeDisparityMapCustom(const cv::Mat& rectL, const cv::Mat& rectR,
                                   cv::Mat& disparity);

    cv::Mat K_;
    cv::Mat distCoeffs_;
    int numDisparities_;
    int blockSize_;
    bool useCustomMatcher_; 
    cv::Mat Q_;
};

#endif // DENSE_MATCHING_H
