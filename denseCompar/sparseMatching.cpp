#include "sparseMatching.h"
#include "8point.h"
#include <iostream>
#include <algorithm>
#include <opencv2/xfeatures2d.hpp>

SparseMatcher::SparseMatcher(const cv::Mat& K, const cv::Mat& distCoeffs)
    : K_(K.clone()), distCoeffs_(distCoeffs.clone()), featureType_(FeatureType::ORB) {
}

void SparseMatcher::setFeatureType(FeatureType type) {
    featureType_ = type;
}

cv::Ptr<cv::Feature2D> SparseMatcher::createDetector(FeatureType type) {
    switch (type) {
        case FeatureType::ORB:
            return cv::ORB::create(2000, 1.2f, 8, 31, 0, 2, cv::ORB::HARRIS_SCORE, 31, 20);
        case FeatureType::SIFT:
            return cv::SIFT::create(2000, 4, 0.04, 10, 1.6);
        case FeatureType::SURF:
            #ifdef OPENCV_ENABLE_NONFREE
            #ifdef HAVE_OPENCV_XFEATURES2D
            return cv::xfeatures2d::SURF::create(1000, 4, 3, false, false);
            #else
            std::cerr << "Warning: SURF not available, falling back to ORB" << std::endl;
            return cv::ORB::create(2000, 1.2f, 8, 31, 0, 2, cv::ORB::HARRIS_SCORE, 31, 20);
            #endif
            #else
            std::cerr << "Warning: SURF not available (non-free modules disabled), falling back to ORB" << std::endl;
            return cv::ORB::create(2000, 1.2f, 8, 31, 0, 2, cv::ORB::HARRIS_SCORE, 31, 20);
            #endif
        default:
            return cv::ORB::create(2000);
    }
}

bool SparseMatcher::detectAndMatch(const cv::Mat& imgL, const cv::Mat& imgR,
                                   std::vector<cv::Point2f>& ptsL, 
                                   std::vector<cv::Point2f>& ptsR) {
    
    if (imgL.empty() || imgR.empty()) {
        std::cerr << "Error: Input images are empty" << std::endl;
        return false;
    }
    
    // 创建特征检测器
    cv::Ptr<cv::Feature2D> detector = createDetector(featureType_);
    
    // 关键点检测与描述子计算
    std::vector<cv::KeyPoint> kpL, kpR;
    cv::Mat descL, descR;
    
    detector->detectAndCompute(imgL, cv::noArray(), kpL, descL);
    detector->detectAndCompute(imgR, cv::noArray(), kpR, descR);
    
    std::cout << "[INFO] Detected feature points: Left image=" << kpL.size() 
              << ", Right image=" << kpR.size() << std::endl;
    
    if (kpL.size() < 50 || kpR.size() < 50) {
        std::cerr << "Error: Insufficient detected feature points (at least 50 points are needed)" << std::endl;
        return false;
    }
    
    if (descL.empty() || descR.empty()) {
        std::cerr << "Error: Feature descriptor computation failed" << std::endl;
        return false;
    }
    
    // 特征匹配
    return matchFeatures(descL, descR, kpL, kpR, ptsL, ptsR);
}

bool SparseMatcher::matchFeatures(const cv::Mat& descL, const cv::Mat& descR,
                                  const std::vector<cv::KeyPoint>& kpL,
                                  const std::vector<cv::KeyPoint>& kpR,
                                  std::vector<cv::Point2f>& ptsL,
                                  std::vector<cv::Point2f>& ptsR) {
    
    // 根据特征类型选择正确的匹配器
    cv::Ptr<cv::DescriptorMatcher> matcher;
    
    // 检查描述子类型来选择匹配器
    if (descL.type() == CV_8U) {
        // 二进制描述子 (ORB, BRIEF等)
        matcher = cv::BFMatcher::create(cv::NORM_HAMMING);
        std::cout << "[INFO] Using HAMMING distance matcher (binary descriptor)" << std::endl;
    } else {
        // 浮点描述子 (SIFT, SURF等)
        matcher = cv::BFMatcher::create(cv::NORM_L2);
        std::cout << "[INFO] Using L2 distance matcher (float descriptor)" << std::endl;
    }
    
    // 使用KNN匹配获得更好的结果
    std::vector<std::vector<cv::DMatch>> knnMatches;
    matcher->knnMatch(descL, descR, knnMatches, 2);
    
    if (knnMatches.empty()) {
        std::cerr << "Error: KNN matching failed" << std::endl;
        return false;
    }
    
    // 使用Lowe's ratio test筛选好的匹配
    std::vector<cv::DMatch> goodMatches;
    const float ratio_thresh = 0.75f; // Lowe's ratio threshold
    
    for (const auto& match : knnMatches) {
        if (match.size() >= 2) {
            if (match[0].distance < ratio_thresh * match[1].distance) {
                goodMatches.push_back(match[0]);
            }
        }
    }
    
    std::cout << "[INFO] Number of matches after Lowe's ratio test: " << goodMatches.size() << std::endl;
    
    if (goodMatches.size() < 20) {
        std::cerr << "Error: Insufficient high-quality matches (at least 20 points are needed)" << std::endl;
        return false;
    }
    
    // 进一步筛选：按距离排序
    std::sort(goodMatches.begin(), goodMatches.end(), 
              [](const cv::DMatch &a, const cv::DMatch &b) { 
                  return a.distance < b.distance; 
              });
    
    // 动态调整匹配点数量
    int maxMatches = std::min(200, static_cast<int>(goodMatches.size()));
    int minMatches = std::max(50, maxMatches / 4);
    int numGoodMatches = std::max(minMatches, maxMatches);
    
    if (numGoodMatches > static_cast<int>(goodMatches.size())) {
        numGoodMatches = static_cast<int>(goodMatches.size());
    }
    
    std::cout << "[INFO] Final number of matches used: " << numGoodMatches << std::endl;
    
    // 提取匹配点坐标
    ptsL.clear();
    ptsR.clear();
    ptsL.reserve(numGoodMatches);
    ptsR.reserve(numGoodMatches);
    
    for (int i = 0; i < numGoodMatches; ++i) {
        const cv::DMatch& m = goodMatches[i];
        ptsL.push_back(kpL[m.queryIdx].pt);
        ptsR.push_back(kpR[m.trainIdx].pt);
    }
    
    // 几何验证 - 使用基础矩阵筛选outliers
    if (ptsL.size() >= 8) {
        std::vector<uchar> inlierMask;
        cv::Mat F = cv::findFundamentalMat(ptsL, ptsR, cv::FM_RANSAC, 3.0, 0.99, inlierMask);
        
        // 筛选内点
        std::vector<cv::Point2f> inlierPtsL, inlierPtsR;
        for (size_t i = 0; i < inlierMask.size(); ++i) {
            if (inlierMask[i]) {
                inlierPtsL.push_back(ptsL[i]);
                inlierPtsR.push_back(ptsR[i]);
            }
        }
        
        std::cout << "[INFO] Number of inliers after geometric verification: " 
                  << inlierPtsL.size() << " / " << ptsL.size() << std::endl;
        
        if (inlierPtsL.size() >= 8) {
            ptsL = inlierPtsL;
            ptsR = inlierPtsR;
        } else {
            std::cerr << "Warning: Insufficient inliers after geometric verification, continuing with original matches" << std::endl;
        }
    }
    
    return true;
}

bool SparseMatcher::estimatePoseFromImages(const cv::Mat& imgL, const cv::Mat& imgR,
                                           cv::Mat& R, cv::Mat& t) {
    
    std::vector<cv::Point2f> ptsL, ptsR;
    
    // 特征检测和匹配
    if (!detectAndMatch(imgL, imgR, ptsL, ptsR)) {
        std::cerr << "Error: Feature detection and matching failed" << std::endl;
        return false;
    }
    
    // 使用8point算法估计位姿
    if (!EightPoint::estimatePose(ptsL, ptsR, K_, R, t)) {
        std::cerr << "Error: Pose estimation failed" << std::endl;
        return false;
    }
    
    std::cout << "[INFO] Successfully estimated pose using 8-point algorithm" << std::endl;
    std::cout << "[INFO] Estimated rotation matrix R:" << std::endl << R << std::endl;
    std::cout << "[INFO] Estimated translation vector t:" << std::endl << t << std::endl;
    
    return true;
} 