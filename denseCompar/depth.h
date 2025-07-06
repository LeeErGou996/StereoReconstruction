#ifndef DEPTH_H
#define DEPTH_H

#include <opencv2/opencv.hpp>

namespace Depth {
bool computeDepthMap(const cv::Mat& disparity,
                     const cv::Mat& Q,
                     cv::Mat& depthMap);

// 专门用于Middlebury评估的深度转换
bool computeDepthMapForMiddlebury(const cv::Mat& disparity,
                                  const cv::Mat& Q,
                                  cv::Mat& depthMapOut,
                                  double baseline);

// 将深度图转换为与真实深度图相同的尺度
bool normalizeDepthToGroundTruth(const cv::Mat& depthMap, 
                                 const cv::Mat& groundTruth,
                                 cv::Mat& normalizedDepth);

// 深度图质量评估
bool evaluateDepthQuality(const cv::Mat& depthMap, const cv::Mat& originalDisparity);
} // namespace Depth

#endif // DEPTH_H 