#include "bmMatcher.h"
#include "config.h"

BMMatcher::BMMatcher(const cv::Mat& K, const cv::Mat& distCoeffs, int numDisparities, int blockSize)
    : K_(K.clone()), distCoeffs_(distCoeffs.clone()), numDisparities_(numDisparities), blockSize_(blockSize) {}

bool BMMatcher::computeDisparityMap(const cv::Mat& rectL, const cv::Mat& rectR, cv::Mat& disparity) {
    cv::Ptr<cv::StereoBM> stereoBM = cv::StereoBM::create(numDisparities_, blockSize_);
    
    // 优化的BM参数设置 - 更宽松的参数以提高有效像素比例
    stereoBM->setPreFilterCap(31);           // 预滤波器上限，增强对比度
    stereoBM->setBlockSize(blockSize_);      // 匹配窗口大小
    stereoBM->setMinDisparity(0);            // 最小视差
    stereoBM->setNumDisparities(numDisparities_); // 最大视差
    stereoBM->setTextureThreshold(5);        // 纹理阈值，进一步降低以减少无效区域
    stereoBM->setUniquenessRatio(5);         // 唯一性比率，进一步降低以提高匹配密度
    stereoBM->setSpeckleWindowSize(100);     // 斑点窗口大小
    stereoBM->setSpeckleRange(16);           // 斑点视差范围，降低以减少过滤
    stereoBM->setDisp12MaxDiff(1);           // 左右一致性检查
    
    // 计算视差图
    stereoBM->compute(rectL, rectR, disparity);
    
    // 后处理：中值滤波去除噪声
    if (!disparity.empty()) {
        cv::Mat filteredDisparity;
        cv::medianBlur(disparity, filteredDisparity, 3);
        disparity = filteredDisparity;
    }
    
    return !disparity.empty();
} 