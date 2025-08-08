#include "error_visualizer.h"
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

namespace ErrorVisualizer {

    cv::Mat computeErrorMap(const cv::Mat& pred, const cv::Mat& gt) {
        cv::Mat error;
        cv::absdiff(pred, gt, error);

        cv::Mat error_norm, error_u8, error_color;
        cv::normalize(error, error_norm, 0, 255, cv::NORM_MINMAX);
        error_norm.convertTo(error_u8, CV_8U);
        cv::applyColorMap(error_u8, error_color, cv::COLORMAP_JET);
        return error_color;
    }

    cv::Mat applyColormapJet(const cv::Mat& src_float) {
        cv::Mat norm, u8, color;
        cv::normalize(src_float, norm, 0, 255, cv::NORM_MINMAX);
        norm.convertTo(u8, CV_8U);
        cv::applyColorMap(u8, color, cv::COLORMAP_JET);
        return color;
    }

    cv::Mat createComparisonDisplay(const cv::Mat& inputImage, const cv::Mat& pred, const cv::Mat& gt) {
        cv::Mat pred_color = applyColormapJet(pred);
        cv::Mat gt_color = applyColormapJet(gt);
        cv::Mat error_map = computeErrorMap(pred, gt);

        cv::Mat top_row, bottom_row;
        cv::hconcat(std::vector<cv::Mat>{inputImage, pred_color, gt_color}, top_row);

        int pad = (top_row.cols - error_map.cols) / 2;
        cv::Mat left_pad = cv::Mat::zeros(error_map.rows, pad, error_map.type());
        cv::Mat right_pad = cv::Mat::zeros(error_map.rows, pad, error_map.type());
        cv::hconcat(std::vector<cv::Mat>{left_pad, error_map, right_pad}, bottom_row);

        cv::Mat display;
        cv::vconcat(top_row, bottom_row, display);
        return display;
    }

}
