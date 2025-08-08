#pragma once
#include <opencv2/core.hpp>
#include <string>

namespace ErrorVisualizer {

    /**
     * @brief 生成彩色误差图（JET colormap）
     * @param pred 预测图像（CV_32F）
     * @param gt   Ground Truth 图像（CV_32F）
     * @return 彩色误差图（CV_8UC3）
     */
    cv::Mat computeErrorMap(const cv::Mat& pred, const cv::Mat& gt);

    /**
     * @brief 拼接三张图（input / pred / gt）+ error map 形成展示图
     * @param inputImage 原图（CV_8UC3）
     * @param pred 预测图（CV_32F）
     * @param gt   Ground Truth（CV_32F）
     * @return 拼接好的大图（CV_8UC3）
     */
    cv::Mat createComparisonDisplay(const cv::Mat& inputImage, const cv::Mat& pred, const cv::Mat& gt);

}
