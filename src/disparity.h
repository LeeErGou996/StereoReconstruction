#ifndef DISPARITY_H
#define DISPARITY_H

#include <opencv2/opencv.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <string>
#include <vector>
#include "8point.h"
#include "denseMatching.h"


class DisparityProcessor {
public:
    // Constructor and destructor
    DisparityProcessor();
    ~DisparityProcessor();
    
    // Main processing function (preserves color information)
    bool computeDisparity(const std::string& leftImagePath, 
                         const std::string& rightImagePath,
                         const std::string& outputPath,
                         FeatureType featureType = FeatureType::ORB);
    
    // Parameter setting functions
    void setCameraMatrix(const cv::Mat& K, const cv::Mat& distCoeffs = cv::Mat::zeros(5, 1, CV_64F));
    void setStereoParams(int numDisparities = 64, int blockSize = 9);
    
    // Feature detection and matching (keep original interface)
    cv::Ptr<cv::Feature2D> createDetector(FeatureType type);
    bool detectAndMatch(const cv::Mat& imgL, const cv::Mat& imgR, 
                       cv::Ptr<cv::Feature2D> detector,
                       std::vector<cv::Point2f>& ptsL, 
                       std::vector<cv::Point2f>& ptsR);
    
    // **New: Methods to get processing results**
    cv::Mat getLeftColorImage() const;        // Get original left color image
    cv::Mat getRightColorImage() const;       // Get original right color image
    cv::Mat getRectifiedLeftColor() const;    // Get rectified left color image
    cv::Mat getRectifiedRightColor() const;   // Get rectified right color image
    cv::Mat getDisparity() const;             // Get disparity map
    cv::Mat getQMatrix() const;               // Get reprojection matrix
    cv::Mat getCameraMatrix() const { return K_.clone(); }
    cv::Mat getDistortionCoeffs() const { return distCoeffs_.clone(); }
    
    // Quality evaluation
    bool evaluateDisparityQuality(const cv::Mat& disparity) const;
    
private:
    // **Original member variables**
    cv::Mat K_;                    // Camera intrinsic matrix
    cv::Mat distCoeffs_;          // Distortion coefficients
    int numDisparities_;          // Disparity range
    int blockSize_;               // Block size
    
    // **New: Color information related member variables**
    cv::Mat leftColorImage_;       // Original left color image
    cv::Mat rightColorImage_;      // Original right color image
    cv::Mat rectifiedLeftColor_;   // Rectified left color image
    cv::Mat rectifiedRightColor_;  // Rectified right color image
    cv::Mat disparity_;            // Disparity map
    cv::Mat Q_;                    // Reprojection matrix
    
    // Private methods
    bool estimatePose(const std::vector<cv::Point2f>& ptsL, 
                     const std::vector<cv::Point2f>& ptsR,
                     cv::Mat& R, cv::Mat& t);
    
    bool rectifyImages(const cv::Mat& imgL, const cv::Mat& imgR,
                      const cv::Mat& R, const cv::Mat& t,
                      cv::Mat& rectL, cv::Mat& rectR, cv::Mat& Q);
    
    // **New: Color image rectification method**
    bool rectifyColorImages(const cv::Mat& imgL_color, const cv::Mat& imgR_color,
                           const cv::Mat& R, const cv::Mat& t,
                           cv::Mat& rectL_color, cv::Mat& rectR_color);
    
    bool computeDisparityMap(const cv::Mat& rectL, const cv::Mat& rectR,
                           cv::Mat& disparity);
    
    // Auxiliary methods
    bool validateInputs(const cv::Mat& imgL, const cv::Mat& imgR) const;
    bool validateCameraMatrix() const;
    void printProcessingInfo(const std::string& step, bool success) const;
};

// **New: Color processing utility functions**
namespace ColorUtils {
    /**
     * Create color disparity map
     * @param disparity Disparity map (CV_16S or CV_32F)
     * @param colormap OpenCV colormap type
     * @return Color disparity map
     */
    cv::Mat createColorDisparity(const cv::Mat& disparity, 
                                int colormap = cv::COLORMAP_JET);
    
    /**
     * Blend disparity map with color image
     * @param disparity Disparity map
     * @param colorImage Color image
     * @param alpha Blending ratio (0.0-1.0)
     * @return Blended image
     */
    cv::Mat blendDisparityWithColor(const cv::Mat& disparity,
                                   const cv::Mat& colorImage,
                                   double alpha = 0.5);
    
    /**
     * Apply depth coloring to color image based on disparity information
     * @param colorImage Color image
     * @param disparity Disparity map
     * @param maxDepth Maximum depth value
     * @return Depth colored image
     */
    cv::Mat applyDepthColoring(const cv::Mat& colorImage,
                              const cv::Mat& disparity,
                              float maxDepth = 1000.0f);
}

// Global utility functions (optional)
namespace DisparityUtils {
    // Disparity map post-processing
    bool postProcessDisparity(cv::Mat& disparity, int filterSize = 5);
    
    // Disparity map quality evaluation
    struct DisparityQuality {
        float validPixelRatio;
        float meanDisparity;
        float disparityRange;
        int totalPixels;
        int validPixels;
    };
    
    DisparityQuality analyzeDisparity(const cv::Mat& disparity);
    
    // **New: Save complete processing results (including color information)**
    bool saveCompleteResults(const cv::Mat& disparity, 
                           const cv::Mat& leftColor,
                           const cv::Mat& rightColor,
                           const cv::Mat& rectifiedLeftColor,
                           const cv::Mat& rectifiedRightColor,
                           const std::string& basePath);
}

#endif // DISPARITY_H