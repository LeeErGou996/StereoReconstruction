#include "denseMatching.h"
#include "config.h"
#include <iostream>

DenseMatcher::DenseMatcher(const cv::Mat& K, const cv::Mat& distCoeffs,
                           int numDisparities, int blockSize)
    : K_(K.clone()), distCoeffs_(distCoeffs.clone()),
      numDisparities_(numDisparities), blockSize_(blockSize) {
}

bool DenseMatcher::rectifyImages(const cv::Mat& imgL, const cv::Mat& imgR,
                                 const cv::Mat& R, const cv::Mat& t,
                                 cv::Mat& rectL, cv::Mat& rectR) {

    // Stereo rectification
    cv::Mat R1, R2, P1, P2, Q;
    cv::stereoRectify(K_, distCoeffs_, K_, distCoeffs_,
                      imgL.size(), R, t, R1, R2, P1, P2, Q);

    Q_ = Q.clone(); // Store the Q matrix

    // Generate rectification maps
    cv::Mat mapLx, mapLy, mapRx, mapRy;
    cv::initUndistortRectifyMap(K_, distCoeffs_, R1, P1,
                                imgL.size(), CV_32FC1, mapLx, mapLy);
    cv::initUndistortRectifyMap(K_, distCoeffs_, R2, P2,
                                imgR.size(), CV_32FC1, mapRx, mapRy);

    // Apply rectification
    cv::remap(imgL, rectL, mapLx, mapLy, cv::INTER_LINEAR);
    cv::remap(imgR, rectR, mapRx, mapRy, cv::INTER_LINEAR);

    return true;
}

bool DenseMatcher::computeDisparityMap(const cv::Mat& rectL, const cv::Mat& rectR,
                                        cv::Mat& disparity) {
    // --- StereoSGBM 小P1/P2参数测试 ---
    // Create stereo matcher
    cv::Ptr<cv::StereoSGBM> stereoSGBM = cv::StereoSGBM::create();
    
    // Set basic parameters
    stereoSGBM->setBlockSize(blockSize_);
    stereoSGBM->setMinDisparity(0);
    stereoSGBM->setNumDisparities(numDisparities_);
    
    // 参数全部从Config读取
    const auto& cfg = Config::instance();
    stereoSGBM->setP1(cfg.sgbmP1 * rectL.channels());
    stereoSGBM->setP2(cfg.sgbmP2 * rectL.channels());
    stereoSGBM->setPreFilterCap(cfg.preFilterCap);
    stereoSGBM->setUniquenessRatio(cfg.uniquenessRatio);
    stereoSGBM->setSpeckleWindowSize(cfg.speckleWindowSize);
    stereoSGBM->setSpeckleRange(cfg.speckleRange);
    stereoSGBM->setDisp12MaxDiff(cfg.disp12MaxDiff);
    
    // BM等效模式
    stereoSGBM->setMode(cv::StereoSGBM::MODE_SGBM);
    
    // Compute disparity map
    stereoSGBM->compute(rectL, rectR, disparity);
    
    // --- StereoBM 旧实现（已注释，仅供参考）---
    /*
    // Create stereo matcher
    cv::Ptr<cv::StereoBM> stereoBM = cv::StereoBM::create(numDisparities_, blockSize_);
    // Set parameters
    stereoBM->setPreFilterCap(31);
    stereoBM->setBlockSize(blockSize_);
    stereoBM->setMinDisparity(0);
    stereoBM->setNumDisparities(numDisparities_);
    stereoBM->setTextureThreshold(10);
    stereoBM->setUniquenessRatio(15);
    stereoBM->setSpeckleWindowSize(100);
    stereoBM->setSpeckleRange(32);
    stereoBM->setDisp12MaxDiff(1);
    // Compute disparity map
    stereoBM->compute(rectL, rectR, disparity);
    */

    return !disparity.empty();
}

const cv::Mat& DenseMatcher::getQMatrix() const {
    return Q_;
} 