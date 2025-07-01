#pragma once
#include "meshReconstruction.h"
#include <vector>
#include <opencv2/opencv.hpp>

namespace MeshReconstruction {
std::vector<Point3D> filterPointCloud(const std::vector<Point3D>& input, float maxNeighborDist);
// 图像去噪滤波器，method=0:中值滤波，1:高斯滤波，2:双边滤波
typedef enum { DENOISE_MEDIAN=0, DENOISE_GAUSSIAN=1, DENOISE_BILATERAL=2 } DenoiseMethod;
cv::Mat denoiseImage(const cv::Mat& input, int method = 0);
} 