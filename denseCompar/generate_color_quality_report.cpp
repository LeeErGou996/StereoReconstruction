#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <filesystem>
#include <opencv2/opencv.hpp>
#include "color_disparity_quality_metrics.h"

int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cout << "Usage: " << argv[0] << " <output_directory>" << std::endl;
        std::cout << "Example: " << argv[0] << " ../output/test1/" << std::endl;
        return -1;
    }
    
    std::string outputDir = argv[1];
    if (!std::filesystem::exists(outputDir)) {
        std::cout << "Error: Output directory does not exist: " << outputDir << std::endl;
        return -1;
    }
    
    std::cout << "=== Color Disparity Quality Report Generator ===" << std::endl;
    std::cout << "Analyzing directory: " << outputDir << std::endl;
    
    // 查找所有色差图文件
    std::vector<std::string> colorDisparityFiles;
    for (const auto& entry : std::filesystem::directory_iterator(outputDir)) {
        if (entry.is_regular_file()) {
            std::string filename = entry.path().filename().string();
            if (filename.find("disparity_") != std::string::npos && 
                filename.find("_color_jet.png") != std::string::npos) {
                colorDisparityFiles.push_back(entry.path().string());
            }
        }
    }
    
    if (colorDisparityFiles.empty()) {
        std::cout << "Error: No color disparity files found in " << outputDir << std::endl;
        return -1;
    }
    
    std::cout << "Found " << colorDisparityFiles.size() << " color disparity files" << std::endl;
    
    // 评估每个色差图的质量
    std::vector<std::pair<std::string, ColorDisparityQualityMetrics>> results;
    
    for (const auto& filepath : colorDisparityFiles) {
        std::string filename = std::filesystem::path(filepath).filename().string();
        std::string algorithmName = filename.substr(10, filename.find("_color_jet.png") - 10);
        
        std::cout << "\nProcessing: " << algorithmName << std::endl;
        
        // 读取色差图
        cv::Mat colorDisparity = cv::imread(filepath, cv::IMREAD_COLOR);
        if (colorDisparity.empty()) {
            std::cout << "Warning: Cannot read " << filepath << std::endl;
            continue;
        }
        
        // 评估质量
        ColorDisparityQualityMetrics metrics = ColorDisparityQualityEvaluator::evaluateQuality(colorDisparity);
        results.push_back({algorithmName, metrics});
        
        std::cout << "  Visual Quality Score: " << metrics.visualQuality << std::endl;
        std::cout << "  Contrast: " << metrics.contrast << std::endl;
        std::cout << "  Sharpness: " << metrics.sharpness << std::endl;
        std::cout << "  Smoothness: " << metrics.smoothness << std::endl;
        std::cout << "  Noise Level: " << metrics.noiseLevel << std::endl;
    }
    
    // 生成详细报告
    std::string reportFilename = outputDir + "/color_disparity_quality_report.txt";
    if (ColorDisparityQualityEvaluator::saveQualityReport(reportFilename, results)) {
        std::cout << "\nDetailed quality report saved to: " << reportFilename << std::endl;
    }
    
    // 生成比较报告
    std::string comparisonReport = ColorDisparityQualityEvaluator::compareAlgorithmsQuality(results);
    std::cout << "\n" << comparisonReport << std::endl;
    
    // 保存比较报告
    std::string comparisonFilename = outputDir + "/color_quality_comparison.txt";
    std::ofstream comparisonFile(comparisonFilename);
    if (comparisonFile.is_open()) {
        comparisonFile << comparisonReport;
        comparisonFile.close();
        std::cout << "Comparison report saved to: " << comparisonFilename << std::endl;
    }
    
    // 找出最佳算法
    if (!results.empty()) {
        auto bestAlgorithm = std::max_element(results.begin(), results.end(),
            [](const auto& a, const auto& b) {
                return a.second.visualQuality < b.second.visualQuality;
            });
        
        std::cout << "\n=== SUMMARY ===" << std::endl;
        std::cout << "Best overall algorithm: " << bestAlgorithm->first << std::endl;
        std::cout << "Best visual quality score: " << bestAlgorithm->second.visualQuality << std::endl;
        std::cout << "Best contrast: " << bestAlgorithm->second.contrast << std::endl;
        std::cout << "Best sharpness: " << bestAlgorithm->second.sharpness << std::endl;
    }
    
    return 0;
} 