#include "sparseMatching.h"

SparseMatcher::SparseMatcher(const cv::Mat& K, const cv::Mat& distCoeffs,
                             int numDisparities, int blockSize)
    : K_(K.clone()), distCoeffs_(distCoeffs.clone()),
      numDisparities_(numDisparities), blockSize_(blockSize) {}

bool SparseMatcher::computeDisparityMap(const cv::Mat& rectL, const cv::Mat& rectR,
                                        cv::Mat& disparity) {
    // ORB特征点检测
    cv::Ptr<cv::ORB> orb = cv::ORB::create(2000);
    std::vector<cv::KeyPoint> kpL, kpR;
    cv::Mat descL, descR;
    orb->detectAndCompute(rectL, cv::noArray(), kpL, descL);
    orb->detectAndCompute(rectR, cv::noArray(), kpR, descR);

    // BFMatcher匹配
    cv::BFMatcher matcher(cv::NORM_HAMMING);
    std::vector<cv::DMatch> matches;
    matcher.match(descL, descR, matches);

    // 构建稀疏视差图
    disparity = cv::Mat::zeros(rectL.size(), CV_16S);
    for (const auto& m : matches) {
        cv::Point2f ptL = kpL[m.queryIdx].pt;
        cv::Point2f ptR = kpR[m.trainIdx].pt;
        int x = static_cast<int>(ptL.x + 0.5);
        int y = static_cast<int>(ptL.y + 0.5);
        float disp = ptL.x - ptR.x;
        if (x >= 0 && x < disparity.cols && y >= 0 && y < disparity.rows && disp > 0) {
            disparity.at<short>(y, x) = static_cast<short>(disp * 16); // 与dense一致，乘16
        }
    }
    return true;
} 