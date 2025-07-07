#include "disparity.h"
#include <iostream>
#include <algorithm>
#include <opencv2/ximgproc/disparity_filter.hpp>

DisparityProcessor::DisparityProcessor() 
    : numDisparities_(64), blockSize_(9) {
    // **修复1**: 重新添加必要的成员变量初始化
    K_ = (cv::Mat_<double>(3,3) << 700, 0, 600, 0, 700, 400, 0, 0, 1);
    distCoeffs_ = cv::Mat::zeros(5, 1, CV_64F);
    
    std::cout << "[INFO] DisparityProcessor initialized with default parameters" << std::endl;
}

DisparityProcessor::~DisparityProcessor() {
    std::cout << "[INFO] DisparityProcessor destroyed" << std::endl;
}

void DisparityProcessor::setCameraMatrix(const cv::Mat& K, const cv::Mat& distCoeffs) {
    if (K.rows != 3 || K.cols != 3 || K.type() != CV_64F) {
        std::cerr << "Error: K matrix must be 3x3 of CV_64F type" << std::endl;
        return;
    }
    K_ = K.clone();
    distCoeffs_ = distCoeffs.clone();
    std::cout << "[INFO] Camera matrix updated" << std::endl;
}

void DisparityProcessor::setStereoParams(int numDisparities, int blockSize) {
    if (numDisparities <= 0 || numDisparities % 16 != 0) {
        std::cerr << "Error: numDisparities must be positive and a multiple of 16" << std::endl;
        return;
    }
    if (blockSize < 3 || blockSize % 2 == 0) {
        std::cerr << "Error: blockSize must be greater than or equal to 3 and odd" << std::endl;
        return;
    }
    
    numDisparities_ = numDisparities;
    blockSize_ = blockSize;
    std::cout << "[INFO] Stereo matching parameters updated: numDisparities=" << numDisparities_ 
              << ", blockSize=" << blockSize_ << std::endl;
}

cv::Ptr<cv::Feature2D> DisparityProcessor::createDetector(FeatureType type) {
    cv::Ptr<cv::Feature2D> detector;
    
    switch(type) {
        case FeatureType::SIFT:
            detector = cv::SIFT::create(1000); // Increase feature point count
            std::cout << "Using SIFT feature detector" << std::endl;
            break;
        case FeatureType::SURF:
            detector = cv::xfeatures2d::SURF::create(400); // Lower threshold to get more feature points
            std::cout << "Using SURF feature detector" << std::endl;
            break;
        case FeatureType::ORB:
            detector = cv::ORB::create(2000); // Increase feature point count
            std::cout << "Using ORB feature detector" << std::endl;
            break;
        default:
            std::cerr << "Invalid algorithm selection, using default ORB" << std::endl;
            detector = cv::ORB::create(2000);
    }
    
    return detector;
}

bool DisparityProcessor::detectAndMatch(const cv::Mat& imgL, const cv::Mat& imgR, 
                                      cv::Ptr<cv::Feature2D> detector,
                                      std::vector<cv::Point2f>& ptsL, 
                                      std::vector<cv::Point2f>& ptsR) {
    
    if (imgL.empty() || imgR.empty()) {
        std::cerr << "Error: Input images are empty" << std::endl;
        return false;
    }
    
    // 关键点检测与描述子计算
    std::vector<cv::KeyPoint> kpL, kpR;
    cv::Mat descL, descR;
    
    detector->detectAndCompute(imgL, cv::noArray(), kpL, descL);
    detector->detectAndCompute(imgR, cv::noArray(), kpR, descR);
    
    std::cout << "[INFO] Detected feature points: Left image=" << kpL.size() << ", Right image=" << kpR.size() << std::endl;
    
    if (kpL.size() < 50 || kpR.size() < 50) {
        std::cerr << "Error: Insufficient detected feature points (at least 50 points are needed)" << std::endl;
        return false;
    }
    
    if (descL.empty() || descR.empty()) {
        std::cerr << "Error: Feature descriptor computation failed" << std::endl;
        return false;
    }
    
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
        
        std::cout << "[INFO] Number of inliers after geometric verification: " << inlierPtsL.size() << " / " << ptsL.size() << std::endl;
        
        if (inlierPtsL.size() >= 8) {
            ptsL = inlierPtsL;
            ptsR = inlierPtsR;
        } else {
            std::cerr << "Warning: Insufficient inliers after geometric verification, continuing with original matches" << std::endl;
        }
    }
    
    return true;
}

bool DisparityProcessor::estimatePose(const std::vector<cv::Point2f>& ptsL, 
                                    const std::vector<cv::Point2f>& ptsR,
                                    cv::Mat& R, cv::Mat& t) {
    
    if (ptsL.size() < 8 || ptsR.size() < 8) {
        std::cerr << "Error: Insufficient matching points (at least 8 points are needed)" << std::endl;
        return false;
    }
    
    // 8点法估算本质矩阵
    std::vector<uchar> inlierMask;
    cv::Mat E = cv::findEssentialMat(ptsL, ptsR, K_, cv::RANSAC, 0.999, 1.0, inlierMask);
    
    if (E.empty()) {
        std::cerr << "Error: Essential matrix estimation failed" << std::endl;
        return false;
    }
    
    // 从本质矩阵恢复位姿
    int inliers = cv::recoverPose(E, ptsL, ptsR, K_, R, t, inlierMask);
    
    std::cout << "[INFO] Number of inliers after pose estimation: " << inliers << " / " << ptsL.size() << std::endl;
    
    if (inliers < 8) {
        std::cerr << "Warning: Few inliers after pose estimation (" << inliers << ")" << std::endl;
    }
    
    return true;
}

bool DisparityProcessor::rectifyImages(const cv::Mat& imgL, const cv::Mat& imgR,
                                     const cv::Mat& R, const cv::Mat& t,
                                     cv::Mat& rectL, cv::Mat& rectR, cv::Mat& Q) {
    
    // 立体校正
    cv::Mat R1, R2, P1, P2;
    cv::stereoRectify(K_, distCoeffs_, K_, distCoeffs_, 
                      imgL.size(), R, t, R1, R2, P1, P2, Q);
    
    // 生成校正映射表
    cv::Mat mapLx, mapLy, mapRx, mapRy;
    cv::initUndistortRectifyMap(K_, distCoeffs_, R1, P1, 
                                imgL.size(), CV_32FC1, mapLx, mapLy);
    cv::initUndistortRectifyMap(K_, distCoeffs_, R2, P2, 
                                imgR.size(), CV_32FC1, mapRx, mapRy);
    
    // 应用校正
    cv::remap(imgL, rectL, mapLx, mapLy, cv::INTER_LINEAR);
    cv::remap(imgR, rectR, mapRx, mapRy, cv::INTER_LINEAR);
    
    std::cout << "[INFO] Stereo rectification completed" << std::endl;
    return true;
}

bool DisparityProcessor::computeDisparityMap(const cv::Mat& rectL, const cv::Mat& rectR,
                                           cv::Mat& disparity) {
    
    // 创建立体匹配器 - 使用StereoSGBM获得更好的结果
    // 注释掉优化参数，恢复到原始参数
    cv::Ptr<cv::StereoSGBM> stereo = cv::StereoSGBM::create(
        0,               // minDisparity (恢复原始值)
        numDisparities_, // numDisparities (恢复使用成员变量)
        blockSize_       // blockSize (恢复使用成员变量)
    );
    
    // 设置参数以获得更好的效果
    stereo->setP1(8 * rectL.channels() * blockSize_ * blockSize_);
    stereo->setP2(32 * rectL.channels() * blockSize_ * blockSize_);
    stereo->setDisp12MaxDiff(1);
    stereo->setUniquenessRatio(10);
    stereo->setSpeckleWindowSize(100);
    stereo->setSpeckleRange(32);
    stereo->setPreFilterCap(63);
    stereo->setMode(cv::StereoSGBM::MODE_SGBM);
    
    // 计算视差图
    stereo->compute(rectL, rectR, disparity);

    // 注释掉WLS滤波代码
    /*
    // 计算右视差图（用于WLS滤波）
    cv::Ptr<cv::StereoSGBM> right_matcher = cv::StereoSGBM::create(
        16, 512, 7
    );
    right_matcher->setMode(cv::StereoSGBM::MODE_SGBM);
    cv::Mat disparity_right;
    right_matcher->compute(rectR, rectL, disparity_right);

    // 创建WLS滤波器
    cv::Ptr<cv::ximgproc::DisparityWLSFilter> wls_filter =
        cv::ximgproc::createDisparityWLSFilterGeneric(false);
    wls_filter->setLambda(8000.0);
    wls_filter->setSigmaColor(1.5);

    // 应用WLS滤波
    cv::Mat filtered_disparity;
    wls_filter->filter(disparity, rectL, filtered_disparity, disparity_right);

    // 替换原始disparity为filtered_disparity
    disparity = filtered_disparity.clone();
    */
    
    std::cout << "[INFO] Disparity map calculation completed, type: " << disparity.type() << std::endl;
    
    return !disparity.empty();
}

// **Major update: Read both color and grayscale images**
// Color image
// Grayscale image (for feature detection)
// Save color images for later use
// Feature detection and matching (using grayscale images)
// Estimate relative pose
// Stereo rectification (using grayscale images)
// **New: Rectify color images**
// Save rectified color images
// Compute disparity map
// Save disparity map and related data
// Save disparity map (8-bit)
// **New: Save additional color-related outputs**
// Save rectified color images
// Save color disparity map (pseudo color)
// **New: Method for rectifying color images**
// Stereo rectification (repeated calculation for consistency)
// Generate rectification maps
// Apply rectification to color images
// **New: Methods to get color information**
bool DisparityProcessor::computeDisparity(const std::string& leftImagePath, 
                                        const std::string& rightImagePath,
                                        const std::string& outputPath,
                                        FeatureType featureType) {
    
    std::cout << "\n=== Starting stereo vision processing ===" << std::endl;
    
    // **Major update: Read both color and grayscale images**
    cv::Mat imgL_color = cv::imread(leftImagePath, cv::IMREAD_COLOR);  // Color image
    cv::Mat imgR_color = cv::imread(rightImagePath, cv::IMREAD_COLOR); // Color image
    cv::Mat imgL = cv::imread(leftImagePath, cv::IMREAD_GRAYSCALE);    // Grayscale image (for feature detection)
    cv::Mat imgR = cv::imread(rightImagePath, cv::IMREAD_GRAYSCALE);   // Grayscale image (for feature detection)
    
    if (imgL.empty() || imgR.empty() || imgL_color.empty() || imgR_color.empty()) {
        std::cerr << "Error: Cannot read image " << leftImagePath << " or " << rightImagePath << std::endl;
        return false;
    }
    std::cout << "Successfully read images, size: " << imgL.size() << std::endl;
    std::cout << "Color image channels: " << imgL_color.channels() << std::endl;
    
    // Save color images for later use
    leftColorImage_ = imgL_color.clone();
    rightColorImage_ = imgR_color.clone();
    
    // Create feature detector
    cv::Ptr<cv::Feature2D> detector = createDetector(featureType);
    if (!detector) {
        std::cerr << "Error: Cannot create feature detector" << std::endl;
        return false;
    }
    
    // Feature detection and matching (using grayscale images)
    std::vector<cv::Point2f> ptsL, ptsR;
    if (!detectAndMatch(imgL, imgR, detector, ptsL, ptsR)) {
        std::cerr << "Error: Feature detection and matching failed" << std::endl;
        return false;
    }
    
    // Estimate relative pose
    cv::Mat R, t;
    if (!estimatePose(ptsL, ptsR, R, t)) {
        std::cerr << "Error: Pose estimation failed" << std::endl;
        return false;
    }
    
    // Stereo rectification (using grayscale images)
    cv::Mat rectL, rectR;
    if (!rectifyImages(imgL, imgR, R, t, rectL, rectR, Q_)) {
        std::cerr << "Error: Stereo rectification failed" << std::endl;
        return false;
    }

    // 注释掉图像预处理代码
    /*
    // 图像预处理：直方图均衡化和高斯去噪
    cv::equalizeHist(rectL, rectL);
    cv::equalizeHist(rectR, rectR);
    cv::GaussianBlur(rectL, rectL, cv::Size(3,3), 0);
    cv::GaussianBlur(rectR, rectR, cv::Size(3,3), 0);
    */

    // **New: Rectify color images**
    cv::Mat rectL_color, rectR_color;
    if (!rectifyColorImages(imgL_color, imgR_color, R, t, rectL_color, rectR_color)) {
        std::cerr << "Warning: Color image rectification failed, using grayscale rectification result" << std::endl;
        cv::cvtColor(rectL, rectL_color, cv::COLOR_GRAY2BGR);
        cv::cvtColor(rectR, rectR_color, cv::COLOR_GRAY2BGR);
    }
    
    // Save rectified color images
    rectifiedLeftColor_ = rectL_color.clone();
    rectifiedRightColor_ = rectR_color.clone();
    
    // Compute disparity map
    cv::Mat disparity;
    if (!computeDisparityMap(rectL, rectR, disparity)) {
        std::cerr << "Error: Disparity computation failed" << std::endl;
        return false;
    }
    
    // Save disparity map and related data
    disparity_ = disparity.clone();
    
    // Save disparity map (8-bit)
    cv::Mat disparity8;
    // 注释掉修正后的参数，恢复原始参数
    /*
    int actualNumDisparities = 512;  // 与SGBM中设置的numDisparities一致
    disparity.convertTo(disparity8, CV_8U, 255.0/(actualNumDisparities*16.0));
    */
    disparity.convertTo(disparity8, CV_8U, 255.0/(numDisparities_*16.0));
    
    if (!cv::imwrite(outputPath, disparity8)) {
        std::cerr << "Error: Cannot save disparity map to " << outputPath << std::endl;
        return false;
    }
    
    // **New: Save additional color-related outputs**
    std::string basePath = outputPath.substr(0, outputPath.find_last_of('.'));
    
    // Save rectified color images
    cv::imwrite(basePath + "_rectified_left_color.png", rectL_color);
    cv::imwrite(basePath + "_rectified_right_color.png", rectR_color);
    
    // Save color disparity map (pseudo color)
    cv::Mat colorDisparity;
    cv::applyColorMap(disparity8, colorDisparity, cv::COLORMAP_JET);
    cv::imwrite(basePath + "_disparity_color.png", colorDisparity);
    
    std::cout << "Successfully saved disparity map to: " << outputPath << std::endl;
    std::cout << "Additional saved files:" << std::endl;
    std::cout << "  - Rectified left color image: " << basePath + "_rectified_left_color.png" << std::endl;
    std::cout << "  - Rectified right color image: " << basePath + "_rectified_right_color.png" << std::endl;
    std::cout << "  - Color disparity map: " << basePath + "_disparity_color.png" << std::endl;
    std::cout << "=== Stereo vision processing completed ===" << std::endl;
    return true;
}

// **New: Method for rectifying color images**
bool DisparityProcessor::rectifyColorImages(const cv::Mat& imgL_color, const cv::Mat& imgR_color,
                                           const cv::Mat& R, const cv::Mat& t,
                                           cv::Mat& rectL_color, cv::Mat& rectR_color) {
    
    // Stereo rectification (repeated calculation for consistency)
    cv::Mat R1, R2, P1, P2, Q;
    cv::stereoRectify(K_, distCoeffs_, K_, distCoeffs_, 
                      imgL_color.size(), R, t, R1, R2, P1, P2, Q);
    
    // Generate rectification maps
    cv::Mat mapLx, mapLy, mapRx, mapRy;
    cv::initUndistortRectifyMap(K_, distCoeffs_, R1, P1, 
                                imgL_color.size(), CV_32FC1, mapLx, mapLy);
    cv::initUndistortRectifyMap(K_, distCoeffs_, R2, P2, 
                                imgR_color.size(), CV_32FC1, mapRx, mapRy);
    
    // Apply rectification to color images
    cv::remap(imgL_color, rectL_color, mapLx, mapLy, cv::INTER_LINEAR);
    cv::remap(imgR_color, rectR_color, mapRx, mapRy, cv::INTER_LINEAR);
    
    std::cout << "[INFO] Stereo rectification completed for color images" << std::endl;
    return true;
}

// **New: Methods to get color information**
cv::Mat DisparityProcessor::getLeftColorImage() const {
    return leftColorImage_.clone();
}

cv::Mat DisparityProcessor::getRightColorImage() const {
    return rightColorImage_.clone();
}

cv::Mat DisparityProcessor::getRectifiedLeftColor() const {
    return rectifiedLeftColor_.clone();
}

cv::Mat DisparityProcessor::getRectifiedRightColor() const {
    return rectifiedRightColor_.clone();
}

cv::Mat DisparityProcessor::getDisparity() const {
    return disparity_.clone();
}

cv::Mat DisparityProcessor::getQMatrix() const {
    return Q_.clone();
}