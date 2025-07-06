#ifndef COLOR_DISPARITY_QUALITY_METRICS_H
#define COLOR_DISPARITY_QUALITY_METRICS_H

#include <opencv2/opencv.hpp>
#include <vector>
#include <string>

struct ColorDisparityQualityMetrics {
    // 基本质量指标
    double contrast;              // 对比度
    double saturation;            // 饱和度
    double brightness;            // 亮度
    double sharpness;             // 锐度
    
    // 连续性指标
    double smoothness;            // 平滑度
    double edgePreservation;      // 边缘保持
    double noiseLevel;            // 噪声水平
    
    // 结构指标
    double structuralSimilarity;  // 结构相似性
    double gradientMagnitude;     // 梯度幅值
    double textureRichness;       // 纹理丰富度
    
    // 视觉质量指标
    double visualQuality;         // 综合视觉质量分数
    double colorConsistency;      // 颜色一致性
    double depthPerception;       // 深度感知质量
    
    ColorDisparityQualityMetrics() : 
        contrast(0.0), saturation(0.0), brightness(0.0), sharpness(0.0),
        smoothness(0.0), edgePreservation(0.0), noiseLevel(0.0),
        structuralSimilarity(0.0), gradientMagnitude(0.0), textureRichness(0.0),
        visualQuality(0.0), colorConsistency(0.0), depthPerception(0.0) {}
};

class ColorDisparityQualityEvaluator {
public:
    // 计算色差图的综合质量指标
    static ColorDisparityQualityMetrics evaluateQuality(const cv::Mat& colorDisparity);
    
    // 计算对比度
    static double computeContrast(const cv::Mat& image);
    
    // 计算饱和度
    static double computeSaturation(const cv::Mat& image);
    
    // 计算亮度
    static double computeBrightness(const cv::Mat& image);
    
    // 计算锐度（基于拉普拉斯算子）
    static double computeSharpness(const cv::Mat& image);
    
    // 计算平滑度
    static double computeSmoothness(const cv::Mat& image);
    
    // 计算边缘保持能力
    static double computeEdgePreservation(const cv::Mat& image);
    
    // 计算噪声水平
    static double computeNoiseLevel(const cv::Mat& image);
    
    // 计算结构相似性
    static double computeStructuralSimilarity(const cv::Mat& image1, const cv::Mat& image2);
    
    // 计算梯度幅值
    static double computeGradientMagnitude(const cv::Mat& image);
    
    // 计算纹理丰富度
    static double computeTextureRichness(const cv::Mat& image);
    
    // 计算颜色一致性
    static double computeColorConsistency(const cv::Mat& image);
    
    // 计算深度感知质量
    static double computeDepthPerception(const cv::Mat& image);
    
    // 计算综合视觉质量分数
    static double computeVisualQualityScore(const ColorDisparityQualityMetrics& metrics);
    
    // 生成质量报告
    static std::string generateQualityReport(const ColorDisparityQualityMetrics& metrics, 
                                           const std::string& algorithmName);
    
    // 比较多个算法的质量
    static std::string compareAlgorithmsQuality(const std::vector<std::pair<std::string, ColorDisparityQualityMetrics>>& results);
    
    // 保存质量评估结果到文件
    static bool saveQualityReport(const std::string& filename, 
                                const std::vector<std::pair<std::string, ColorDisparityQualityMetrics>>& results);
};

#endif // COLOR_DISPARITY_QUALITY_METRICS_H 