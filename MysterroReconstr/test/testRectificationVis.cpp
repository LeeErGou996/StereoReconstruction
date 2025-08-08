#include "../src/feature_match.h"
#include "../src/8point.h"
#include "../src/utils/rectification.h"
#include "../src/utils/imageutils.h"
#include <iostream>
#include <vector>
#include <fstream>
#include <sstream>
#include <cmath>
#include <algorithm>
#include <string>
#include <cassert>
#include <random>
#include <iomanip>
#include <numeric>
#include <filesystem>

int main() {
    std::string leftPath = "../data/left/test2.png";
    std::string rightPath = "../data/right/test2.png";
    std::string KPath = "../data/camera/test2.txt";
    
    // Extract image name from camera parameter file path (e.g., "test2" from "test2.txt")
    std::filesystem::path cameraPathObj(KPath);
    std::string imageName = cameraPathObj.stem().string(); // Gets "test2" from "test2.txt"
    
    // Create output directory with image name
    std::string outputDir = "../output/" + imageName;
    std::filesystem::create_directories(outputDir);
    
    std::string outMatchPath = outputDir + "/" + imageName + "_match.png";
    
    std::cout << "Output directory: " << outputDir << std::endl;

    // 1. 特征匹配
    MyImage imgL, imgR;
    if (!myImReadPNG(leftPath, imgL) || !myImReadPNG(rightPath, imgR)) {
        std::cerr << "Failed to read input images!" << std::endl;
        return 1;
    }
    
    // 灰度化处理
    if (imgL.channels > 1) {
        MyImage grayL;
        grayL.width = imgL.width;
        grayL.height = imgL.height;
        grayL.channels = 1;
        grayL.data.resize(imgL.width * imgL.height);
        int step = imgL.channels;
        for (int i = 0; i < imgL.width * imgL.height; ++i) {
            grayL.data[i] = 0.299f * imgL.data[i*step] + 0.587f * imgL.data[i*step+1] + 0.114f * imgL.data[i*step+2];
        }
        imgL = std::move(grayL);
    }
    
    if (imgR.channels > 1) {
        MyImage grayR;
        grayR.width = imgR.width;
        grayR.height = imgR.height;
        grayR.channels = 1;
        grayR.data.resize(imgR.width * imgR.height);
        int step = imgR.channels;
        for (int i = 0; i < imgR.width * imgR.height; ++i) {
            grayR.data[i] = 0.299f * imgR.data[i*step] + 0.587f * imgR.data[i*step+1] + 0.114f * imgR.data[i*step+2];
        }
        imgR = std::move(grayR);
    }

    // 使用修正后的detectAndMatch函数
    auto result = detectAndMatch(imgL, imgR);
    const auto& all_kp = result.first;
    const auto& matches = result.second;
    size_t kp_count = all_kp.size() / 2;
    std::vector<KeyPoint> kpL(all_kp.begin(), all_kp.begin() + kp_count);
    std::vector<KeyPoint> kpR(all_kp.begin() + kp_count, all_kp.end());
    std::cout << "Detected " << kpL.size() << " and " << kpR.size() << " keypoints." << std::endl;
    std::cout << "Matched " << matches.size() << " pairs." << std::endl;

    if (matches.empty()) {
        std::cerr << "No matches found!" << std::endl;
        return 1;
    }

    // 可视化匹配
    MyImage matchImg;
    drawMatches(imgL, kpL, imgR, kpR, matches, matchImg);
    myImWritePNG(outMatchPath, matchImg);
    std::cout << "Saved matching result to: " << outMatchPath << std::endl;

    // 2. 读取相机内参K
    EightPoint::Matrix K(3, 3);
    // 初始化为单位矩阵
    K.at(0, 0) = 1.0; K.at(1, 1) = 1.0; K.at(2, 2) = 1.0;
    
    std::ifstream fk(KPath);
    if (fk.is_open()) {
        std::string line;
        std::getline(fk, line);
        fk.close();
        // 提取中括号内内容
        auto l = line.find('[');
        auto r = line.find(']');
        if (l != std::string::npos && r != std::string::npos && r > l) {
            std::string nums = line.substr(l + 1, r - l - 1);
            std::replace(nums.begin(), nums.end(), ';', ' '); // 用空格替换分号
            std::istringstream iss(nums);
            double val;
            int idx = 0;
            while (iss >> val && idx < 9) {
                K.at(idx / 3, idx % 3) = val;
                idx++;
            }
        }
        std::cout << "Loaded camera intrinsics K:" << std::endl;
        K.print();
    } else {
        std::cerr << "Failed to read " << KPath << ", using identity matrix." << std::endl;
        std::cout << "Using identity camera matrix:" << std::endl;
        K.print();
    }

    // 3. 构造匹配点对
    std::vector<EightPoint::Point2f> ptsL, ptsR;
    for (const auto& m : matches) {
        ptsL.emplace_back(kpL[m.idx1].x, kpL[m.idx1].y);
        ptsR.emplace_back(kpR[m.idx2].x, kpR[m.idx2].y);
    }

    std::cout << "\n=== 1. Feature Matching Statistics ===" << std::endl;
    std::cout << "Successfully matched point pairs: " << ptsL.size() << std::endl;
    
    if (ptsL.size() < 8) {
        std::cerr << "Error: Insufficient matching points for 8-point algorithm (need at least 8, got " << ptsL.size() << ")" << std::endl;
        return 1;
    }

    // 4. 8点法估计位姿
    EightPoint::Matrix R(3, 3), t(3, 1);
    bool ok = EightPoint::estimatePose(ptsL, ptsR, K, R, t);
    
    if (!ok) {
        std::cerr << "8-point pose estimation failed!" << std::endl;
        return 1;
    }

    std::cout << "\n=== 3. Pose Estimation Results ===" << std::endl;
    std::cout << "Rotation matrix R:" << std::endl;
    R.print();
    
    std::cout << "Translation vector t:" << std::endl;
    std::cout << "[" << t.at(0, 0) << ", " << t.at(1, 0) << ", " << t.at(2, 0) << "]" << std::endl;
    
    // Calculate rotation angle (Rodrigues)
    double trace = R.at(0, 0) + R.at(1, 1) + R.at(2, 2);
    double angle = std::acos(std::max(-1.0, std::min(1.0, (trace - 1.0) / 2.0)));
    std::cout << "Rotation angle: " << angle * 180.0 / M_PI << " degrees" << std::endl;
    
    // Calculate translation distance
    double t_norm = std::sqrt(t.at(0, 0) * t.at(0, 0) + t.at(1, 0) * t.at(1, 0) + t.at(2, 0) * t.at(2, 0));
    std::cout << "Translation distance: " << t_norm << " (normalized)" << std::endl;

    // 5. Additional quality checks
    std::cout << "\n=== 4. Quality Assessment ===" << std::endl;
    
    // Check orthogonality of rotation matrix
    EightPoint::Matrix RTR = R.t() * R;
    EightPoint::Matrix I = EightPoint::Matrix::eye(3);
    double orthogonality_error = 0.0;
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            double diff = RTR.at(i, j) - I.at(i, j);
            orthogonality_error += diff * diff;
        }
    }
    orthogonality_error = std::sqrt(orthogonality_error);
    std::cout << "Rotation matrix orthogonality error: " << orthogonality_error << std::endl;
    
    if (orthogonality_error < 1e-6) {
        std::cout << "Rotation matrix quality: Excellent" << std::endl;
    } else if (orthogonality_error < 1e-4) {
        std::cout << "Rotation matrix quality: Good" << std::endl;
    } else {
        std::cout << "Rotation matrix quality: Needs improvement" << std::endl;
    }

    // Check reprojection error of matched points (simplified version)
    double total_error = 0.0;
    int valid_points = 0;
    for (size_t i = 0; i < std::min(ptsL.size(), size_t(20)); ++i) {
        // More detailed reprojection error calculation can be added here
        // Simplified processing: assume error is related to feature matching distance
        if (i < matches.size()) {
            total_error += matches[i].dist;
            valid_points++;
        }
    }
    
    if (valid_points > 0) {
        double avg_match_error = total_error / valid_points;
        std::cout << "Average feature matching error: " << avg_match_error << std::endl;
    }

    // === 5. Epipolar rectification (calling rectification.cpp implementation) ===
    MyImage imgL_rect, imgR_rect;
    myImReadPNG(leftPath, imgL_rect);
    myImReadPNG(rightPath, imgR_rect);
    using Mat3 = std::array<std::array<double, 3>, 3>;
    using Vec3 = std::array<double, 3>;
    Mat3 K1, K2, Rmat, H1, H2;
    Vec3 tvec;
    for (int r = 0; r < 3; ++r) for (int c = 0; c < 3; ++c) {
        K1[r][c] = K.at(r, c);
        K2[r][c] = K.at(r, c);
        Rmat[r][c] = R.at(r, c);
    }
    for (int r = 0; r < 3; ++r) tvec[r] = t.at(r, 0);
    computeRectification(K1, K2, Rmat, tvec, H1, H2);
    MyImage rectL, rectR;
    // Save original images before warping
    myImWritePNG(outputDir + "/debug_before_warp_left.png", imgL_rect);
    myImWritePNG(outputDir + "/debug_before_warp_right.png", imgR_rect);
    warpImage(imgL_rect, rectL, H1);
    warpImage(imgR_rect, rectR, H2);
    myImWritePNG(outputDir + "/rectified_left.png", rectL);
    myImWritePNG(outputDir + "/rectified_right.png", rectR);
    std::cout << "Rectification: saved rectified_left.png and rectified_right.png" << std::endl;

    // === 6. Before and after rectification comparison visualization ===
    std::cout << "\n=== 5. Generating Feature Point Comparison Images Before and After Rectification ===" << std::endl;
    
    // Ensure all images are converted to 3-channel RGB for visualization
    auto convertToRGB = [](const MyImage& src, MyImage& dst) {
        dst.width = src.width;
        dst.height = src.height;
        dst.channels = 3;
        dst.data.resize(src.width * src.height * 3);
        
        if (src.channels == 1) {
            // Grayscale to RGB
            for (int i = 0; i < src.width * src.height; ++i) {
                dst.data[i*3] = dst.data[i*3+1] = dst.data[i*3+2] = src.data[i];
            }
        } else if (src.channels == 3) {
            dst.data = src.data;
        } else if (src.channels == 4) {
            // RGBA to RGB, discard alpha channel
            for (int i = 0; i < src.width * src.height; ++i) {
                dst.data[i*3] = src.data[i*4];
                dst.data[i*3+1] = src.data[i*4+1];
                dst.data[i*3+2] = src.data[i*4+2];
            }
        }
    };
    
    // Convert to RGB format
    MyImage imgL_rgb, imgR_rgb, rectL_rgb, rectR_rgb;
    convertToRGB(imgL_rect, imgL_rgb);
    convertToRGB(imgR_rect, imgR_rgb);
    convertToRGB(rectL, rectL_rgb);
    convertToRGB(rectR, rectR_rgb);
    
    // Point transformation function: transform point coordinates through homography matrix
    auto transformPoint = [](double x, double y, const Mat3& H) -> std::pair<double, double> {
        double w = H[2][0] * x + H[2][1] * y + H[2][2];
        if (std::abs(w) < 1e-8) return {-1, -1}; // Invalid transformation
        double newX = (H[0][0] * x + H[0][1] * y + H[0][2]) / w;
        double newY = (H[1][0] * x + H[1][1] * y + H[1][2]) / w;
        return {newX, newY};
    };
    
    // Draw feature point function (supports different sizes and shapes)
    auto drawFeaturePoint = [](MyImage& img, int cx, int cy, int r, int g, int b, int size = 4, bool filled = true) {
        if (filled) {
            // Draw filled square marker
            for (int dy = -size; dy <= size; ++dy) {
                for (int dx = -size; dx <= size; ++dx) {
                    int x = cx + dx;
                    int y = cy + dy;
                    if (x >= 0 && x < img.width && y >= 0 && y < img.height) {
                        img.data[(y * img.width + x) * 3] = r;
                        img.data[(y * img.width + x) * 3 + 1] = g;
                        img.data[(y * img.width + x) * 3 + 2] = b;
                    }
                }
            }
        } else {
            // Draw hollow square marker (only border)
            for (int dx = -size; dx <= size; ++dx) {
                // Top and bottom borders
                int x = cx + dx;
                if (x >= 0 && x < img.width) {
                    if (cy - size >= 0) {
                        img.data[((cy - size) * img.width + x) * 3] = r;
                        img.data[((cy - size) * img.width + x) * 3 + 1] = g;
                        img.data[((cy - size) * img.width + x) * 3 + 2] = b;
                    }
                    if (cy + size < img.height) {
                        img.data[((cy + size) * img.width + x) * 3] = r;
                        img.data[((cy + size) * img.width + x) * 3 + 1] = g;
                        img.data[((cy + size) * img.width + x) * 3 + 2] = b;
                    }
                }
            }
            for (int dy = -size; dy <= size; ++dy) {
                // Left and right borders
                int y = cy + dy;
                if (y >= 0 && y < img.height) {
                    if (cx - size >= 0) {
                        img.data[(y * img.width + (cx - size)) * 3] = r;
                        img.data[(y * img.width + (cx - size)) * 3 + 1] = g;
                        img.data[(y * img.width + (cx - size)) * 3 + 2] = b;
                    }
                    if (cx + size < img.width) {
                        img.data[(y * img.width + (cx + size)) * 3] = r;
                        img.data[(y * img.width + (cx + size)) * 3 + 1] = g;
                        img.data[(y * img.width + (cx + size)) * 3 + 2] = b;
                    }
                }
            }
        }
    };
    
    // Create before and after rectification comparison images (Left image: Original | Rectified)
    int compW = imgL_rgb.width * 2 + 10; // Leave 10 pixel gap in the middle
    int compH = imgL_rgb.height;
    
    // Create left image comparison (Image 1 - Original vs Rectified)
    MyImage compLeft;
    compLeft.width = compW;
    compLeft.height = compH;
    compLeft.channels = 3;
    compLeft.data.resize(compW * compH * 3, 128); // Gray background
    
    // Copy original left image to left half
    for (int y = 0; y < imgL_rgb.height; ++y) {
        for (int x = 0; x < imgL_rgb.width; ++x) {
            for (int c = 0; c < 3; ++c) {
                compLeft.data[(y * compW + x) * 3 + c] = imgL_rgb.data[(y * imgL_rgb.width + x) * 3 + c];
            }
        }
    }
    
    // Copy rectified left image to right half
    int offsetX = imgL_rgb.width + 10;
    for (int y = 0; y < rectL_rgb.height && y < compH; ++y) {
        for (int x = 0; x < rectL_rgb.width && x + offsetX < compW; ++x) {
            for (int c = 0; c < 3; ++c) {
                compLeft.data[(y * compW + (x + offsetX)) * 3 + c] = rectL_rgb.data[(y * rectL_rgb.width + x) * 3 + c];
            }
        }
    }
    
    // Create right image comparison (Image 2 - Original vs Rectified)
    MyImage compRight;
    compRight.width = compW;
    compRight.height = compH;
    compRight.channels = 3;
    compRight.data.resize(compW * compH * 3, 128); // Gray background
    
    // Copy original right image to left half
    for (int y = 0; y < imgR_rgb.height; ++y) {
        for (int x = 0; x < imgR_rgb.width; ++x) {
            for (int c = 0; c < 3; ++c) {
                compRight.data[(y * compW + x) * 3 + c] = imgR_rgb.data[(y * imgR_rgb.width + x) * 3 + c];
            }
        }
    }
    
    // Copy rectified right image to right half
    for (int y = 0; y < rectR_rgb.height && y < compH; ++y) {
        for (int x = 0; x < rectR_rgb.width && x + offsetX < compW; ++x) {
            for (int c = 0; c < 3; ++c) {
                compRight.data[(y * compW + (x + offsetX)) * 3 + c] = rectR_rgb.data[(y * rectR_rgb.width + x) * 3 + c];
            }
        }
    }
    
    // Mark feature points on comparison images
    std::cout << "Marking feature point changes..." << std::endl;
    
    // Select top 20 best matching points for visualization
    int numPointsToShow = std::min(static_cast<int>(matches.size()), 20);
    
    for (int i = 0; i < numPointsToShow; ++i) {
        // Get original feature point coordinates
        int origX1 = static_cast<int>(ptsL[i].x);
        int origY1 = static_cast<int>(ptsL[i].y);
        int origX2 = static_cast<int>(ptsR[i].x);
        int origY2 = static_cast<int>(ptsR[i].y);
        
        // Calculate rectified feature point coordinates (through homography matrix transformation)
        auto rectPt1 = transformPoint(ptsL[i].x, ptsL[i].y, H1);
        auto rectPt2 = transformPoint(ptsR[i].x, ptsR[i].y, H2);
        
        int rectX1 = static_cast<int>(rectPt1.first);
        int rectY1 = static_cast<int>(rectPt1.second);
        int rectX2 = static_cast<int>(rectPt2.first);
        int rectY2 = static_cast<int>(rectPt2.second);
        
        // Mark points in left image comparison
        // Feature points in original image (left half) - green
        if (origX1 >= 0 && origX1 < imgL_rgb.width && origY1 >= 0 && origY1 < imgL_rgb.height) {
            drawFeaturePoint(compLeft, origX1, origY1, 0, 255, 0, 3);
        }
        
        // In rectified image:
        // 1. Green marker: original position found by feature matching algorithm (direct mapping)
        if (origX1 >= 0 && origX1 < rectL_rgb.width && origY1 >= 0 && origY1 < rectL_rgb.height) {
            drawFeaturePoint(compLeft, origX1 + offsetX, origY1, 0, 255, 0, 3);
        }
        // 2. Red marker: position after homography matrix transformation
        if (rectX1 >= 0 && rectX1 < rectL_rgb.width && rectY1 >= 0 && rectY1 < rectL_rgb.height) {
            drawFeaturePoint(compLeft, rectX1 + offsetX, rectY1, 255, 0, 0, 3);
        }
        
        // Mark points in right image comparison
        // Feature points in original image (left half) - green
        if (origX2 >= 0 && origX2 < imgR_rgb.width && origY2 >= 0 && origY2 < imgR_rgb.height) {
            drawFeaturePoint(compRight, origX2, origY2, 0, 255, 0, 3);
        }
        
        // In rectified image:
        // 1. Green marker: original position found by feature matching algorithm (direct mapping)
        if (origX2 >= 0 && origX2 < rectR_rgb.width && origY2 >= 0 && origY2 < rectR_rgb.height) {
            drawFeaturePoint(compRight, origX2 + offsetX, origY2, 0, 255, 0, 3);
        }
        // 2. Red marker: position after homography matrix transformation
        if (rectX2 >= 0 && rectX2 < rectR_rgb.width && rectY2 >= 0 && rectY2 < rectR_rgb.height) {
            drawFeaturePoint(compRight, rectX2 + offsetX, rectY2, 255, 0, 0, 3);
        }
    }
    
    // Draw separator line
    for (int y = 0; y < compH; ++y) {
        for (int x = imgL_rgb.width; x < imgL_rgb.width + 10; ++x) {
            if (x < compW) {
                compLeft.data[(y * compW + x) * 3] = 255;     // White separator line
                compLeft.data[(y * compW + x) * 3 + 1] = 255;
                compLeft.data[(y * compW + x) * 3 + 2] = 255;
                
                compRight.data[(y * compW + x) * 3] = 255;
                compRight.data[(y * compW + x) * 3 + 1] = 255;
                compRight.data[(y * compW + x) * 3 + 2] = 255;
            }
        }
    }
    
    // Save comparison images
    myImWritePNG(outputDir + "/rectification_feature_comparison_left.png", compLeft);
    myImWritePNG(outputDir + "/rectification_feature_comparison_right.png", compRight);
    
    std::cout << "Feature point comparison images before and after rectification saved:" << std::endl;
    std::cout << "  - " << outputDir << "/rectification_feature_comparison_left.png (Image 1: Original vs Rectified)" << std::endl;
    std::cout << "  - " << outputDir << "/rectification_feature_comparison_right.png (Image 2: Original vs Rectified)" << std::endl;
    std::cout << "  - Green squares in original image: positions found by feature matching algorithm" << std::endl;
    std::cout << "  - Green squares in rectified image: positions found by feature matching algorithm (direct mapping)" << std::endl;
    std::cout << "  - Red squares in rectified image: theoretical positions after homography matrix transformation" << std::endl;
    std::cout << "  - Distance between red and green markers reflects the accuracy of rectification transformation" << std::endl;
    std::cout << "  - Shows position changes of " << numPointsToShow << " best matching feature points" << std::endl;

    std::cout << "\n=== Processing Completed ===" << std::endl;
    std::cout << "Feature matching image saved to: " << outMatchPath << std::endl;
    std::cout << "8-point algorithm pose estimation completed successfully!" << std::endl;
    std::cout << "All results saved in directory: " << outputDir << std::endl;

    return 0;
}