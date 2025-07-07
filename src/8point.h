#ifndef EIGHT_POINT_H
#define EIGHT_POINT_H

#include <opencv2/opencv.hpp>
#include <vector>

namespace EightPoint {
bool estimatePose(const std::vector<cv::Point2f>& ptsL,
                  const std::vector<cv::Point2f>& ptsR,
                  const cv::Mat& K,
                  cv::Mat& R,
                  cv::Mat& t);

cv::Mat computeEssentialMatrix8Point(const std::vector<cv::Point2f>& ptsL,
                                     const std::vector<cv::Point2f>& ptsR,
                                     const cv::Mat& K);

} // namespace EightPoint

#endif // EIGHT_POINT_H 