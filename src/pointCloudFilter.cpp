#include "pointCloudFilter.h"
#include <vector>
#include <cmath>
#include <opencv2/opencv.hpp>

namespace MeshReconstruction {
// 空间距离过滤：只保留与最近邻距离小于阈值的点
std::vector<Point3D> filterPointCloud(const std::vector<Point3D>& input, float maxNeighborDist = 20.0f) {
    std::vector<Point3D> output;
    if (input.empty()) return output;
    for (size_t i = 0; i < input.size(); ++i) {
        const Point3D& p = input[i];
        float minDist = std::numeric_limits<float>::max();
        for (size_t j = 0; j < input.size(); ++j) {
            if (i == j) continue;
            float dx = p.x - input[j].x;
            float dy = p.y - input[j].y;
            float dz = p.z - input[j].z;
            float dist = std::sqrt(dx*dx + dy*dy + dz*dz);
            if (dist < minDist) minDist = dist;
        }
        if (minDist < maxNeighborDist) {
            output.push_back(p);
        }
    }
    return output;
}

// 图像去噪滤波器
// method=0:中值滤波，1:高斯滤波，2:双边滤波
cv::Mat denoiseImage(const cv::Mat& input, int method) {
    cv::Mat output;
    switch (method) {
        case 0: // 中值滤波
            cv::medianBlur(input, output, 5);
            break;
        case 1: // 高斯滤波
            cv::GaussianBlur(input, output, cv::Size(5,5), 1.5);
            break;
        case 2: // 双边滤波
            cv::bilateralFilter(input, output, 9, 75, 75);
            break;
        default:
            output = input.clone();
            break;
    }
    return output;
}
} 