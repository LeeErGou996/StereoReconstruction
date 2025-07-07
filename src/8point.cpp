#include "8point.h"
#include <iostream>

#include <opencv2/calib3d.hpp>

namespace EightPoint {
bool estimatePose(const std::vector<cv::Point2f>& ptsL,
                  const std::vector<cv::Point2f>& ptsR,
                  const cv::Mat& K,
                  cv::Mat& R,
                  cv::Mat& t) {

    if (ptsL.size() < 8 || ptsR.size() < 8) {
        std::cerr << "Error: Insufficient matching point count (at least 8 points are required)" << std::endl;
        return false;
    }

    // 8 point method to estimate the essential matrix
    // cv::Mat E = cv::findEssentialMat(ptsL, ptsR, K, cv::RANSAC);
    cv::Mat E = computeEssentialMatrix8Point(ptsL, ptsR, K);


    if (E.empty()) {
        std::cerr << "Error: Failed to estimate essential matrix" << std::endl;
        return false;
    }

    // Recover pose from essential matrix
    int inliers = cv::recoverPose(E, ptsL, ptsR, K, R, t);

    if (inliers < 8) {
        std::cerr << "Warning: Fewer inliers (" << inliers << ")" << std::endl;
    }

    return true;
}
} // namespace EightPoint 


cv::Mat EightPoint::computeEssentialMatrix8Point(
    const std::vector<cv::Point2f>& ptsL,
    const std::vector<cv::Point2f>& ptsR,
    const cv::Mat& K
) {
    // 1. Normalize image points with K⁻¹
    std::vector<cv::Point2f> normL, normR;
    cv::Mat K_inv = K.inv();
    for (size_t i = 0; i < ptsL.size(); ++i) {
        cv::Mat p1 = (cv::Mat_<double>(3,1) << ptsL[i].x, ptsL[i].y, 1.0);
        cv::Mat p2 = (cv::Mat_<double>(3,1) << ptsR[i].x, ptsR[i].y, 1.0);
        cv::Mat np1 = K_inv * p1;
        cv::Mat np2 = K_inv * p2;
        normL.emplace_back(np1.at<double>(0), np1.at<double>(1));
        normR.emplace_back(np2.at<double>(0), np2.at<double>(1));
    }

    // 2. Hartley normalization (isotropic scaling to mean=0, std=√2)
    auto normalizePoints = [](const std::vector<cv::Point2f>& pts, cv::Mat& T) {
        cv::Point2f mean(0, 0);
        for (auto& pt : pts) mean += pt;
        mean *= (1.0f / pts.size());

        float scale = 0.0f;
        for (auto& pt : pts)
            scale += cv::norm(pt - mean);
        scale = sqrt(2.0f) * pts.size() / scale;

        T = (cv::Mat_<double>(3,3) <<
            scale, 0, -scale * mean.x,
            0, scale, -scale * mean.y,
            0, 0, 1);

        std::vector<cv::Point2f> normed;
        for (auto& pt : pts) {
            cv::Mat hp = (cv::Mat_<double>(3,1) << pt.x, pt.y, 1.0);
            cv::Mat tp = T * hp;
            normed.emplace_back(tp.at<double>(0), tp.at<double>(1));
        }
        return normed;
    };

    cv::Mat T1, T2;
    std::vector<cv::Point2f> pts1_normed = normalizePoints(normL, T1);
    std::vector<cv::Point2f> pts2_normed = normalizePoints(normR, T2);

    // 3. Construct A matrix
    cv::Mat A(ptsL.size(), 9, CV_64F);
    for (size_t i = 0; i < ptsL.size(); ++i) {
        double x1 = pts1_normed[i].x;
        double y1 = pts1_normed[i].y;
        double x2 = pts2_normed[i].x;
        double y2 = pts2_normed[i].y;

        A.at<double>(i, 0) = x2 * x1;
        A.at<double>(i, 1) = x2 * y1;
        A.at<double>(i, 2) = x2;
        A.at<double>(i, 3) = y2 * x1;
        A.at<double>(i, 4) = y2 * y1;
        A.at<double>(i, 5) = y2;
        A.at<double>(i, 6) = x1;
        A.at<double>(i, 7) = y1;
        A.at<double>(i, 8) = 1.0;
    }

    // 4. Solve A * e = 0 using SVD
    cv::Mat u, s, vt;
    cv::SVD::compute(A, s, u, vt);
    cv::Mat E_hat = vt.row(8).reshape(0, 3);

    // 5. Enforce rank 2 constraint
    cv::SVD::compute(E_hat, s, u, vt);
    s.at<double>(2) = 0.0;
    E_hat = u * cv::Mat::diag(s) * vt;

    // 6. Denormalize: E = T2^T * E_hat * T1
    cv::Mat E = T2.t() * E_hat * T1;

    return E;
}


// cv::Mat EightPoint::computeEssentialMatrix8Point(const std::vector<cv::Point2f>& ptsL,
//                                                   const std::vector<cv::Point2f>& ptsR,
//                                                   const cv::Mat& K) {
//     CV_Assert(ptsL.size() == ptsR.size() && ptsL.size() >= 8);

//     // Normalize points
//     std::vector<cv::Point2f> normL, normR;
//     cv::Mat K_inv = K.inv();

//     for (size_t i = 0; i < ptsL.size(); ++i) {
//         cv::Mat pL = (cv::Mat_<double>(3,1) << ptsL[i].x, ptsL[i].y, 1.0);
//         cv::Mat pR = (cv::Mat_<double>(3,1) << ptsR[i].x, ptsR[i].y, 1.0);

//         cv::Mat nL = K_inv * pL;
//         cv::Mat nR = K_inv * pR;

//         normL.emplace_back(nL.at<double>(0), nL.at<double>(1));
//         normR.emplace_back(nR.at<double>(0), nR.at<double>(1));
//     }

//     // Construct A matrix
//     cv::Mat A(static_cast<int>(normL.size()), 9, CV_64F);
//     for (size_t i = 0; i < normL.size(); ++i) {
//         double x1 = normL[i].x;
//         double y1 = normL[i].y;
//         double x2 = normR[i].x;
//         double y2 = normR[i].y;

//         A.at<double>(i,0) = x2 * x1;
//         A.at<double>(i,1) = x2 * y1;
//         A.at<double>(i,2) = x2;
//         A.at<double>(i,3) = y2 * x1;
//         A.at<double>(i,4) = y2 * y1;
//         A.at<double>(i,5) = y2;
//         A.at<double>(i,6) = x1;
//         A.at<double>(i,7) = y1;
//         A.at<double>(i,8) = 1.0;
//     }

//     // Solve Af = 0 using SVD
//     cv::Mat u, s, vt;
//     cv::SVD::compute(A, s, u, vt);
//     cv::Mat F = vt.row(8).reshape(0, 3); // reshape to 3x3

//     // Enforce rank-2 constraint
//     cv::SVD::compute(F, s, u, vt);
//     s.at<double>(2) = 0;
//     F = u * cv::Mat::diag(s) * vt;

//     // Return essential matrix
//     return F;
// }