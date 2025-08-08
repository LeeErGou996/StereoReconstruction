#include "depth.h"
#include <iostream>
#include <fstream>
#include <sstream>

namespace Depth {

// 自定义3D重投影函数实现
bool reprojectImageTo3D(const MyMat& disparity, std::vector<Point3f>& xyz, const Matrix& Q, bool handleMissingValues) {
    if (disparity.empty() || Q.empty()) {
        return false;
    }
    
    xyz.clear();
    xyz.reserve(disparity.rows * disparity.cols);
    
    for (int y = 0; y < disparity.rows; ++y) {
        for (int x = 0; x < disparity.cols; ++x) {
            float disp = disparity.at(y, x);
            
            if (handleMissingValues && (disp <= 0 || std::isnan(disp))) {
                xyz.push_back(Point3f(0, 0, 0)); // 无效点
                continue;
            }
            
            // 使用Q矩阵进行3D重投影
            // Q矩阵格式: [1 0 0 -cx; 0 1 0 -cy; 0 0 0 f; 0 0 -1/Tx 0]
            double cx = -Q.at(0, 3);
            double cy = -Q.at(1, 3);
            double f = Q.at(2, 3);
            double Tx = -1.0 / Q.at(3, 2);
            
            if (disp > 0) {
                double Z = f * Tx / disp;
                double X = (x - cx) * Z / f;
                double Y = (y - cy) * Z / f;
                xyz.push_back(Point3f(static_cast<float>(X), static_cast<float>(Y), static_cast<float>(Z)));
            } else {
                xyz.push_back(Point3f(0, 0, 0));
            }
        }
    }
    
    // 应用质心半径过滤
    std::cout << "[3D Reprojection] Applying centroid radius filtering..." << std::endl;
    xyz = filterPointCloudByCentroidRadius(xyz, 5.0f);
    
    return true;
}

// 自定义统计函数实现
bool minMaxLoc(const MyMat& mat, double& minVal, double& maxVal, 
               int* minLoc, int* maxLoc, const std::vector<bool>& mask) {
    if (mat.empty()) return false;
    
    minVal = std::numeric_limits<double>::max();
    maxVal = std::numeric_limits<double>::lowest();
    int minIdx = -1, maxIdx = -1;
    
    for (int i = 0; i < mat.rows * mat.cols; ++i) {
        if (!mask.empty() && i < mask.size() && !mask[i]) continue;
        
        float val = mat.data[i];
        if (std::isnan(val) || std::isinf(val)) continue;
        
        if (val < minVal) {
            minVal = val;
            minIdx = i;
        }
        if (val > maxVal) {
            maxVal = val;
            maxIdx = i;
        }
    }
    
    if (minLoc && minIdx >= 0) {
        *minLoc = minIdx;
    }
    if (maxLoc && maxIdx >= 0) {
        *maxLoc = maxIdx;
    }
    
    return minIdx >= 0 && maxIdx >= 0;
}

// 自定义计数函数实现
int countNonZero(const MyMat& mat) {
    int count = 0;
    for (float val : mat.data) {
        if (val != 0.0f && !std::isnan(val)) {
            count++;
        }
    }
    return count;
}

// 自定义均值计算实现
double mean(const MyMat& mat, const std::vector<bool>& mask) {
    if (mat.empty()) return 0.0;
    
    double sum = 0.0;
    int count = 0;
    
    for (size_t i = 0; i < mat.data.size(); ++i) {
        if (!mask.empty() && i < mask.size() && !mask[i]) continue;
        
        float val = mat.data[i];
        if (!std::isnan(val) && !std::isinf(val)) {
            sum += val;
            count++;
        }
    }
    
    return count > 0 ? sum / count : 0.0;
}

// 自定义位运算实现
std::vector<bool> bitwiseAnd(const std::vector<bool>& mask1, const std::vector<bool>& mask2) {
    size_t size = std::min(mask1.size(), mask2.size());
    std::vector<bool> result(size);
    for (size_t i = 0; i < size; ++i) {
        result[i] = mask1[i] && mask2[i];
    }
    return result;
}

// 计算点云质心
Point3f computePointCloudCentroid(const std::vector<Point3f>& points) {
    if (points.empty()) {
        return Point3f(0, 0, 0);
    }
    
    double sumX = 0.0, sumY = 0.0, sumZ = 0.0;
    int validCount = 0;
    
    for (const auto& point : points) {
        // 只考虑有效的3D点（非原点）
        if (point.x != 0.0f || point.y != 0.0f || point.z != 0.0f) {
            sumX += point.x;
            sumY += point.y;
            sumZ += point.z;
            validCount++;
        }
    }
    
    if (validCount == 0) {
        return Point3f(0, 0, 0);
    }
    
    return Point3f(static_cast<float>(sumX / validCount),
                   static_cast<float>(sumY / validCount),
                   static_cast<float>(sumZ / validCount));
}

// 计算点到质心的距离
float computeDistanceToCentroid(const Point3f& point, const Point3f& centroid) {
    float dx = point.x - centroid.x;
    float dy = point.y - centroid.y;
    float dz = point.z - centroid.z;
    return std::sqrt(dx * dx + dy * dy + dz * dz);
}

// 基于质心半径过滤点云
std::vector<Point3f> filterPointCloudByCentroidRadius(const std::vector<Point3f>& points, 
                                                      float radius) {
    if (points.empty()) {
        return std::vector<Point3f>();
    }
    
    // 1. 计算质心
    Point3f centroid = computePointCloudCentroid(points);
    std::cout << "[PointCloud] Computed centroid: (" << centroid.x 
              << ", " << centroid.y << ", " << centroid.z << ")" << std::endl;
    
    // 2. 过滤点云
    std::vector<Point3f> filteredPoints;
    int totalPoints = 0;
    int validPoints = 0;
    int filteredOutCount = 0;
    
    for (const auto& point : points) {
        totalPoints++;
        
        // 跳过无效点
        if (point.x == 0.0f && point.y == 0.0f && point.z == 0.0f) {
            continue;
        }
        
        validPoints++;
        
        // 计算到质心的距离
        float distance = computeDistanceToCentroid(point, centroid);
        
        // 保留半径范围内的点
        if (distance <= radius) {
            filteredPoints.push_back(point);
        } else {
            filteredOutCount++;
        }
    }
    
    std::cout << "[PointCloud] Radius filtering applied:" << std::endl;
    std::cout << "  - Centroid radius: " << radius << " meters" << std::endl;
    std::cout << "  - Total points: " << totalPoints << std::endl;
    std::cout << "  - Valid points: " << validPoints << std::endl;
    std::cout << "  - Points within radius: " << filteredPoints.size() << std::endl;
    std::cout << "  - Points filtered out: " << filteredOutCount << std::endl;
    if (validPoints > 0) {
        std::cout << "  - Retention rate: " << (filteredPoints.size() * 100.0 / validPoints) << "%" << std::endl;
    }
    
    return filteredPoints;
}

// 自定义图像保存函数实现（保存为简单的文本格式用于调试）
bool imwrite(const std::string& filename, const MyMat& img) {
    std::ofstream file(filename);
    if (!file.is_open()) return false;
    
    file << img.rows << " " << img.cols << std::endl;
    for (int y = 0; y < img.rows; ++y) {
        for (int x = 0; x < img.cols; ++x) {
            file << img.at(y, x) << " ";
        }
        file << std::endl;
    }
    
    return true;
}

bool computeDepthMap(const MyMat& disparity,
                     const Matrix& Q,
                     MyMat& depthMapOut) {

    // Debug: Print disparity and Q matrix info
    std::cout << "[DEBUG] Disparity map info: rows=" << disparity.rows 
              << ", cols=" << disparity.cols 
              << ", type=" << disparity.type << std::endl;
    std::cout << "[DEBUG] Q matrix info: rows=" << Q.rows() 
              << ", cols=" << Q.cols() << std::endl;

    if (disparity.empty() || Q.empty()) {
        std::cerr << "Error: Disparity map or Q matrix is empty, cannot compute depth map." << std::endl;
        return false;
    }

    // 1. 处理视差类型（如有必要，做缩放）
    MyMat processedDisparity;
    if (disparity.type == 2) { // CV_16S equivalent
        disparity.convertTo(processedDisparity, 1.0f/16.0f);
        std::cout << "[DEBUG] Converted CV_16S disparity to CV_32F, scale: 1/16" << std::endl;
    } else if (disparity.type == 1) { // CV_8U equivalent
        disparity.convertTo(processedDisparity);
        std::cout << "[DEBUG] Converted CV_8U disparity to CV_32F" << std::endl;
    } else if (disparity.type == 0) { // CV_32F equivalent
        processedDisparity = disparity.clone();
        std::cout << "[DEBUG] Disparity map is already CV_32F format" << std::endl;
    } else {
        std::cerr << "Error: Unsupported disparity map data type: " << disparity.type << std::endl;
        return false;
    }

    // 2. 创建有效像素掩码
    std::vector<bool> validMask(processedDisparity.data.size());
    for (size_t i = 0; i < processedDisparity.data.size(); ++i) {
        validMask[i] = processedDisparity.data[i] > 0;
    }

    if (countNonZero(processedDisparity) == 0) {
        std::cerr << "Warning: No valid disparity value!" << std::endl;
        depthMapOut = MyMat(processedDisparity.rows, processedDisparity.cols, 0);
        return true;
    }

    // 3. 提取 Q 矩阵参数
    double focalLength = Q.at(2, 3);
    double baseline = -1.0 / Q.at(3, 2);
    if (focalLength <= 0) {
        focalLength = 1000.0;
        std::cout << "[DEBUG] Using default focal length: " << focalLength << std::endl;
    } else {
        std::cout << "[DEBUG] Using focal length from Q matrix: " << focalLength << std::endl;
    }
    std::cout << "[DEBUG] Using baseline from Q matrix: " << baseline << std::endl;

    // 4. 计算深度并过滤大于20米的深度值
    depthMapOut = MyMat(processedDisparity.rows, processedDisparity.cols, 0);
    
    const double MAX_DEPTH_THRESHOLD = 20.0; // 20米阈值
    int filteredCount = 0;
    int totalValidPixels = 0;
    
    for (int y = 0; y < processedDisparity.rows; ++y) {
        for (int x = 0; x < processedDisparity.cols; ++x) {
            if (validMask[y * processedDisparity.cols + x]) {
                float disp = processedDisparity.at(y, x);
                if (disp > 0) {
                    float depth = baseline * focalLength / disp;
                    
                    // 过滤大于20米的深度值
                    if (depth > MAX_DEPTH_THRESHOLD) {
                        depthMapOut.at(y, x) = 0.0; // 设置为无效
                        filteredCount++;
                    } else {
                        depthMapOut.at(y, x) = depth;
                    }
                    totalValidPixels++;
                }
            }
        }
    }
    
    // 输出过滤统计信息
    std::cout << "[DEBUG] Depth filtering applied:" << std::endl;
    std::cout << "  - Max depth threshold: " << MAX_DEPTH_THRESHOLD << " meters" << std::endl;
    std::cout << "  - Total valid pixels before filtering: " << totalValidPixels << std::endl;
    std::cout << "  - Pixels filtered out (> " << MAX_DEPTH_THRESHOLD << "m): " << filteredCount << std::endl;
    std::cout << "  - Remaining valid pixels: " << (totalValidPixels - filteredCount) << std::endl;
    if (totalValidPixels > 0) {
        std::cout << "  - Filtering percentage: " << (filteredCount * 100.0 / totalValidPixels) << "%" << std::endl;
    }

    // 5. 统计和调试输出
    double minDepth, maxDepth;
    minMaxLoc(depthMapOut, minDepth, maxDepth, nullptr, nullptr, validMask);
    std::cout << "[DEBUG] Valid depth range: [" << minDepth << ", " << maxDepth << "] unit" << std::endl;
    std::cout << "[DEBUG] Number of valid depth pixels: " << countNonZero(depthMapOut) << std::endl;
    int centerX = depthMapOut.cols / 2;
    int centerY = depthMapOut.rows / 2;
    float centerDepth = depthMapOut.at(centerY, centerX);
    std::cout << "[DEBUG] Center depth: " << centerDepth << std::endl;

    return true;
}

// **New function**: 专门用于Middlebury评估的深度转换
bool computeDepthMapForMiddlebury(const MyMat& disparity,
                                  const Matrix& Q,
                                  MyMat& depthMapOut,
                                  double baseline) {
    
    if (disparity.empty()) {
        std::cerr << "Error: Disparity map is empty" << std::endl;
        return false;
    }

    // 1. 处理视差图数据类型
    MyMat processedDisparity;
    std::cout << "[Middlebury] Input disparity type: " << disparity.type << std::endl;
    
    // 检查视差图的值范围，判断是否需要缩放
    float minDisp = std::numeric_limits<float>::max();
    float maxDisp = std::numeric_limits<float>::lowest();
    for (size_t i = 0; i < disparity.data.size(); ++i) {
        if (disparity.data[i] > 0) {
            minDisp = std::min(minDisp, disparity.data[i]);
            maxDisp = std::max(maxDisp, disparity.data[i]);
        }
    }
    std::cout << "[Middlebury] Input disparity range: [" << minDisp << ", " << maxDisp << "]" << std::endl;
    
    if (disparity.type == 2) { // CV_16S equivalent
        // StereoBM/StereoSGBM输出，需要除以16
        disparity.convertTo(processedDisparity, 1.0f/16.0f);
        std::cout << "[Middlebury] Converted CV_16S disparity, scale: 1/16" << std::endl;
    } else if (disparity.type == 1) { // CV_8U equivalent
        disparity.convertTo(processedDisparity);
        std::cout << "[Middlebury] Converted CV_8U disparity" << std::endl;
    } else if (disparity.type == 0) { // CV_32F equivalent
        // 检查是否需要额外的缩放
        if (maxDisp <= 255.0f) {
            // 可能是归一化的视差图，需要缩放到实际范围
            processedDisparity = disparity.clone();
            for (size_t i = 0; i < processedDisparity.data.size(); ++i) {
                if (processedDisparity.data[i] > 0) {
                    processedDisparity.data[i] *= 256.0f; // 缩放到更合理的范围
                }
            }
            std::cout << "[Middlebury] Scaled normalized disparity by 256" << std::endl;
        } else {
            processedDisparity = disparity.clone();
            std::cout << "[Middlebury] Disparity already CV_32F, no scaling needed" << std::endl;
        }
    } else {
        std::cerr << "Error: Unsupported disparity type: " << disparity.type << std::endl;
        return false;
    }

    // 2. 创建有效像素掩码
    std::vector<bool> validMask(processedDisparity.data.size());
    for (size_t i = 0; i < processedDisparity.data.size(); ++i) {
        validMask[i] = processedDisparity.data[i] > 0;
    }
    
    if (countNonZero(processedDisparity) == 0) {
        std::cerr << "Error: No valid disparity pixels" << std::endl;
        return false;
    }

    // 3. 使用简化的深度公式：depth = baseline * focal_length / disparity
    // 从Q矩阵提取焦距 - 修正提取方式
    double focalLength = 1000.0; // 默认焦距
    if (Q.rows() >= 4 && Q.cols() >= 4) {
        // Q矩阵格式: [1 0 0 -cx; 0 1 0 -cy; 0 0 0 f; 0 0 -1/Tx 0]
        // 焦距在Q[2,3]位置
        double q23 = Q.at(2, 3);
        if (q23 > 0) {
            focalLength = q23;
            std::cout << "[Middlebury] Using focal length from Q matrix: " << focalLength << std::endl;
        } else {
            std::cout << "[Middlebury] Q[2,3] is not positive, using default focal length: " << focalLength << std::endl;
        }
    } else {
        std::cout << "[Middlebury] Q matrix size insufficient, using default focal length: " << focalLength << std::endl;
    }

    // 4. 计算深度图并过滤大于20米的深度值
    depthMapOut = MyMat(processedDisparity.rows, processedDisparity.cols, 0);
    
    const double MAX_DEPTH_THRESHOLD = 20.0; // 20米阈值
    int filteredCount = 0;
    int totalValidPixels = 0;
    
    for (int y = 0; y < processedDisparity.rows; ++y) {
        for (int x = 0; x < processedDisparity.cols; ++x) {
            if (validMask[y * processedDisparity.cols + x]) {
                float disp = processedDisparity.at(y, x);
                if (disp > 0) {
                    // 深度 = 基线 * 焦距 / 视差
                    float depth = baseline * focalLength / disp;
                    
                    // 过滤大于20米的深度值
                    if (depth > MAX_DEPTH_THRESHOLD) {
                        depthMapOut.at(y, x) = 0.0; // 设置为无效
                        filteredCount++;
                    } else {
                        depthMapOut.at(y, x) = depth;
                    }
                    totalValidPixels++;
                }
            }
        }
    }
    
    // 输出过滤统计信息
    std::cout << "[Middlebury] Depth filtering applied:" << std::endl;
    std::cout << "  - Max depth threshold: " << MAX_DEPTH_THRESHOLD << " meters" << std::endl;
    std::cout << "  - Total valid pixels before filtering: " << totalValidPixels << std::endl;
    std::cout << "  - Pixels filtered out (> " << MAX_DEPTH_THRESHOLD << "m): " << filteredCount << std::endl;
    std::cout << "  - Remaining valid pixels: " << (totalValidPixels - filteredCount) << std::endl;
    if (totalValidPixels > 0) {
        std::cout << "  - Filtering percentage: " << (filteredCount * 100.0 / totalValidPixels) << "%" << std::endl;
    }

    // 5. 统计信息
    double minDepth, maxDepth;
    minMaxLoc(depthMapOut, minDepth, maxDepth, nullptr, nullptr, validMask);
    std::cout << "[Middlebury] Depth range: [" << minDepth << ", " << maxDepth << "] meters" << std::endl;
    std::cout << "[Middlebury] Valid pixels: " << countNonZero(depthMapOut) << "/" 
              << (depthMapOut.rows * depthMapOut.cols) << std::endl;

    return true;
}

// **New function**: 将深度图转换为与真实深度图相同的尺度
bool normalizeDepthToGroundTruth(const MyMat& depthMap, 
                                 const MyMat& groundTruth,
                                 MyMat& normalizedDepth) {
    
    if (depthMap.empty() || groundTruth.empty()) {
        std::cerr << "Error: Input images are empty" << std::endl;
        return false;
    }

    // 1. 获取真实深度图的有效范围
    std::vector<bool> gtValid(groundTruth.data.size());
    for (size_t i = 0; i < groundTruth.data.size(); ++i) {
        gtValid[i] = groundTruth.data[i] > 0;
    }
    
    double gtMin, gtMax;
    minMaxLoc(groundTruth, gtMin, gtMax, nullptr, nullptr, gtValid);
    
    std::cout << "[Normalize] Ground truth range: [" << gtMin << ", " << gtMax << "]" << std::endl;

    // 2. 获取估计深度图的有效范围
    std::vector<bool> depthValid(depthMap.data.size());
    for (size_t i = 0; i < depthMap.data.size(); ++i) {
        depthValid[i] = depthMap.data[i] > 0;
    }
    
    double depthMin, depthMax;
    minMaxLoc(depthMap, depthMin, depthMax, nullptr, nullptr, depthValid);
    
    std::cout << "[Normalize] Estimated depth range: [" << depthMin << ", " << depthMax << "]" << std::endl;

    // 3. 线性映射到真实深度图的范围
    normalizedDepth = MyMat(depthMap.rows, depthMap.cols, 0);
    
    if (depthMax > depthMin && gtMax > gtMin) {
        double scale = (gtMax - gtMin) / (depthMax - depthMin);
        double offset = gtMin - depthMin * scale;
        
        std::cout << "[Normalize] Scale: " << scale << ", Offset: " << offset << std::endl;
        
        for (int y = 0; y < depthMap.rows; ++y) {
            for (int x = 0; x < depthMap.cols; ++x) {
                if (depthValid[y * depthMap.cols + x]) {
                    float depth = depthMap.at(y, x);
                    float normalized = depth * scale + offset;
                    normalizedDepth.at(y, x) = normalized;
                }
            }
        }
    } else {
        // 如果范围无效，直接复制
        normalizedDepth = depthMap.clone();
    }

    return true;
}

// **New feature**: Depth map quality evaluation
bool evaluateDepthQuality(const MyMat& depthMap, const MyMat& originalDisparity) {
    if (depthMap.empty() || originalDisparity.empty()) {
        std::cerr << "Error: Input image is empty" << std::endl;
        return false;
    }
    
    // Calculate valid pixel ratio
    std::vector<bool> validMask(depthMap.data.size());
    for (size_t i = 0; i < depthMap.data.size(); ++i) {
        validMask[i] = depthMap.data[i] > 0;
    }
    
    int validPixels = countNonZero(depthMap);
    int totalPixels = depthMap.rows * depthMap.cols;
    float validRatio = static_cast<float>(validPixels) / totalPixels;
    
    std::cout << "\n=== Depth map quality evaluation ===" << std::endl;
    std::cout << "Valid pixel ratio: " << (validRatio * 100) << "%" << std::endl;
    
    if (validRatio < 0.3) {
        std::cout << "Warning: Valid depth pixel ratio is low, you may need to adjust stereo matching parameters" << std::endl;
    }
    
    // Calculate depth distribution
    if (validPixels > 0) {
        double minVal, maxVal;
        minMaxLoc(depthMap, minVal, maxVal, nullptr, nullptr, validMask);
        
        double meanDepth = mean(depthMap, validMask);
        std::cout << "Depth stats - min: " << minVal 
                  << ", max: " << maxVal 
                  << ", mean: " << meanDepth << std::endl;
    }
    
    return true;
}

} // namespace Depth