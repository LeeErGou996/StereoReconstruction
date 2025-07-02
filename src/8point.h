#ifndef EIGHT_POINT_H
#define EIGHT_POINT_H

// ---- 新增：特征类型枚举 ----
enum class FeatureType {
    ORB,
    SIFT,
    SURF
};
// ---- 新增结束 ----

#include <opencv2/opencv.hpp>
#include <vector>

namespace EightPoint {
bool estimatePose(const std::vector<cv::Point2f>& ptsL,
                  const std::vector<cv::Point2f>& ptsR,
                  const cv::Mat& K,
                  cv::Mat& R,
                  cv::Mat& t);
} // namespace EightPoint

#endif // EIGHT_POINT_H 