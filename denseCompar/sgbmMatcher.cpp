#include "sgbmMatcher.h"
#include "config.h"

SGBMMatcher::SGBMMatcher(const cv::Mat& K, const cv::Mat& distCoeffs, int numDisparities, int blockSize)
    : K_(K.clone()), distCoeffs_(distCoeffs.clone()), numDisparities_(numDisparities), blockSize_(blockSize) {}

bool SGBMMatcher::computeDisparityMap(const cv::Mat& rectL, const cv::Mat& rectR, cv::Mat& disparity) {
    cv::Ptr<cv::StereoSGBM> stereoSGBM = cv::StereoSGBM::create();
    stereoSGBM->setBlockSize(blockSize_);
    stereoSGBM->setMinDisparity(0);
    stereoSGBM->setNumDisparities(numDisparities_);
    const auto& cfg = Config::instance();
    stereoSGBM->setP1(cfg.sgbmP1 * rectL.channels());
    stereoSGBM->setP2(cfg.sgbmP2 * rectL.channels());
    stereoSGBM->setPreFilterCap(cfg.preFilterCap);
    stereoSGBM->setUniquenessRatio(cfg.uniquenessRatio);
    stereoSGBM->setSpeckleWindowSize(cfg.speckleWindowSize);
    stereoSGBM->setSpeckleRange(cfg.speckleRange);
    stereoSGBM->setDisp12MaxDiff(cfg.disp12MaxDiff);
    stereoSGBM->setMode(cv::StereoSGBM::MODE_SGBM);
    stereoSGBM->compute(rectL, rectR, disparity);
    return !disparity.empty();
} 