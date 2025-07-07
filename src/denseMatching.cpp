#include "denseMatching.h"
#include <iostream>
#include <limits>

DenseMatcher::DenseMatcher(const cv::Mat& K, const cv::Mat& distCoeffs,
                           int numDisparities, int blockSize, bool useCustom)
    : K_(K.clone()), distCoeffs_(distCoeffs.clone()),
      numDisparities_(numDisparities), blockSize_(blockSize),
      useCustomMatcher_(useCustom) {}

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
    if (useCustomMatcher_) {
        return computeDisparityMapCustom(rectL, rectR, disparity);
    }

    // Create stereo matcher
    cv::Ptr<cv::StereoBM> stereo = cv::StereoBM::create(numDisparities_, blockSize_);

    // Set parameters
    stereo->setPreFilterCap(31);
    stereo->setBlockSize(blockSize_);
    stereo->setMinDisparity(0);
    stereo->setNumDisparities(numDisparities_);
    stereo->setTextureThreshold(10);
    stereo->setUniquenessRatio(15);
    stereo->setSpeckleWindowSize(100);
    stereo->setSpeckleRange(32);
    stereo->setDisp12MaxDiff(1);

    // Compute disparity map
    stereo->compute(rectL, rectR, disparity);

    return !disparity.empty();
}

bool DenseMatcher::computeDisparityMapCustom(const cv::Mat& rectL, const cv::Mat& rectR, cv::Mat& disparity) {
    int rows = rectL.rows;
    int cols = rectL.cols;
    int ws = blockSize_ / 2;

    cv::Mat grayL, grayR;
    if (rectL.channels() == 3) cv::cvtColor(rectL, grayL, cv::COLOR_BGR2GRAY);
    else grayL = rectL.clone();
    if (rectR.channels() == 3) cv::cvtColor(rectR, grayR, cv::COLOR_BGR2GRAY);
    else grayR = rectR.clone();

    // disparity = cv::Mat::zeros(rows, cols, CV_8U);
    disparity = cv::Mat::zeros(rows, cols, CV_32F);


    for (int y = ws; y < rows - ws; ++y) {
        if ((y - ws) % 20 == 0)
            std::cout << "[DEBUG] Processing row: " << y << " / " << (rows - ws) << std::endl;
        for (int x = ws + numDisparities_; x < cols - ws; ++x) {
            int minSSD = std::numeric_limits<int>::max();
            int bestDisp = 0;
            for (int d = 0; d < numDisparities_; ++d) {
                int ssd = 0;
                for (int v = -ws; v <= ws; ++v) {
                    for (int u = -ws; u <= ws; ++u) {
                        int pL = grayL.at<uchar>(y + v, x + u);
                        int pR = grayR.at<uchar>(y + v, x + u - d);
                        int diff = pL - pR;
                        ssd += diff * diff;
                    }
                }
                if (ssd < minSSD) {
                    minSSD = ssd;
                    bestDisp = d;
                }
            }
            // disparity.at<uchar>(y, x) = static_cast<uchar>(bestDisp * 255 / numDisparities_);
            disparity.at<float>(y, x) = static_cast<float>(bestDisp);

        }
    }

    return true;
}

const cv::Mat& DenseMatcher::getQMatrix() const {
    return Q_;
}