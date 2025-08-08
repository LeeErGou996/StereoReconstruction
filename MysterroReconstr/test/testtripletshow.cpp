#include "../src/utils/imageutils.h"
#include <iostream>
#include <string>
#include <cassert>
#include <vector>
#include <filesystem>
#include <fstream>
#include <cmath>
#include <chrono>
#include <iomanip>
#include <sstream>

// Function to convert PNG disparity back to original scale
MyImage convertPNGToOriginalScale(const MyImage& png_disp, const MyImage& gt_disp) {
    MyImage original_scale_disp;
    original_scale_disp.width = png_disp.width;
    original_scale_disp.height = png_disp.height;
    original_scale_disp.channels = 1;
    original_scale_disp.data.resize(png_disp.width * png_disp.height);
    
    // Find min/max values in GT disparity for normalization reference
    float gt_min = 1e9f, gt_max = -1e9f;
    for (int i = 0; i < gt_disp.width * gt_disp.height; ++i) {
        float val = static_cast<float>(gt_disp.data[i]);
        if (val > 0 && val < 1000) {  // Valid disparity range
            gt_min = std::min(gt_min, val);
            gt_max = std::max(gt_max, val);
        }
    }
    
    // If GT has valid range, use it for normalization
    if (gt_max > gt_min) {
        float gt_range = gt_max - gt_min;
        for (int i = 0; i < png_disp.width * png_disp.height; ++i) {
            // Convert from 0-255 range back to original scale
            float normalized_val = static_cast<float>(png_disp.data[i]) / 255.0f;
            original_scale_disp.data[i] = static_cast<unsigned char>(normalized_val * gt_range + gt_min);
        }
    } else {
        // Fallback: assume PNG is already in reasonable scale
        original_scale_disp = png_disp;
    }
    
    return original_scale_disp;
}

// Function to calculate BAD2.0 metric with proper normalization
double calculateBAD2_0(const MyImage& gt_disp, const MyImage& pred_disp, const MyImage& mask = MyImage()) {
    if (gt_disp.width != pred_disp.width || gt_disp.height != pred_disp.height) {
        std::cerr << "Error: GT and predicted disparity maps have different dimensions" << std::endl;
        return -1.0;
    }
    
    int total_pixels = gt_disp.width * gt_disp.height;
    int error_pixels = 0;
    int valid_pixels = 0;
    
    // Check if mask is provided and valid
    bool use_mask = (mask.width == gt_disp.width && mask.height == gt_disp.height && mask.channels == 1);
    
    // Find valid range in GT disparity
    float gt_min = 1e9f, gt_max = -1e9f;
    for (int i = 0; i < gt_disp.width * gt_disp.height; ++i) {
        float val = static_cast<float>(gt_disp.data[i]);
        if (val > 0 && val < 1000) {  // Valid disparity range
            gt_min = std::min(gt_min, val);
            gt_max = std::max(gt_max, val);
        }
    }
    
    // Check if we need to normalize predicted disparity
    bool need_normalization = false;
    float pred_max = 0;
    for (int i = 0; i < pred_disp.width * pred_disp.height; ++i) {
        pred_max = std::max(pred_max, static_cast<float>(pred_disp.data[i]));
    }
    
    // If predicted disparity is in 0-255 range and GT is in larger range, normalize
    if (pred_max <= 255 && gt_max > 255) {
        need_normalization = true;
        std::cout << "  Normalizing predicted disparity from [0,255] to [" << gt_min << "," << gt_max << "]" << std::endl;
    }
    
    for (int y = 0; y < gt_disp.height; y++) {
        for (int x = 0; x < gt_disp.width; x++) {
            int idx = y * gt_disp.width + x;
            
            // Skip invalid pixels (if mask is provided)
            if (use_mask && mask.data[idx] == 0) {
                continue;
            }
            
            // Get disparity values
            float gt_val = static_cast<float>(gt_disp.data[idx]);
            float pred_val = static_cast<float>(pred_disp.data[idx]);
            
            // Normalize predicted disparity if needed
            if (need_normalization) {
                float normalized_val = pred_val / 255.0f;
                pred_val = normalized_val * (gt_max - gt_min) + gt_min;
            }
            
            // Skip invalid disparity values
            if (gt_val <= 0 || gt_val > 1000 || pred_val <= 0 || pred_val > 1000) {
                continue;
            }
            
            valid_pixels++;
            
            // Calculate absolute error
            float error = std::abs(gt_val - pred_val);
            
            // Count pixels with error > 2.0
            if (error > 2.0f) {
                error_pixels++;
            }
        }
    }
    
    if (valid_pixels == 0) {
        std::cerr << "Warning: No valid pixels found for BAD2.0 calculation" << std::endl;
        return -1.0;
    }
    
    double bad2_0 = (static_cast<double>(error_pixels) / static_cast<double>(valid_pixels)) * 100.0;
    return bad2_0;
}

// Function to save BAD2.0 results to text file
void saveBAD2_0Results(const std::string& outputDir, const std::string& datasetName, 
                       const std::vector<std::string>& results) {
    std::string resultsFile = outputDir + "/bad2_0_results.txt";
    std::ofstream outFile(resultsFile);
    
    if (!outFile.is_open()) {
        std::cerr << "❌ Failed to open results file: " << resultsFile << std::endl;
        return;
    }
    
    // Get current timestamp
    auto now = std::chrono::system_clock::now();
    auto time_t = std::chrono::system_clock::to_time_t(now);
    
    outFile << "=== BAD2.0 Metric Results for Dataset: " << datasetName << " ===" << std::endl;
    outFile << "Generated on: " << std::ctime(&time_t);
    outFile << "Format: filename | BAD2.0 (%)" << std::endl;
    outFile << "===============================================" << std::endl;
    
    // Calculate average BAD2.0
    double total_bad2_0 = 0.0;
    int valid_results = 0;
    
    for (const auto& result : results) {
        outFile << result << std::endl;
        
        // Extract BAD2.0 value from result string
        size_t pos = result.find(" | ");
        if (pos != std::string::npos) {
            size_t percent_pos = result.find("%", pos);
            if (percent_pos != std::string::npos) {
                std::string bad2_0_str = result.substr(pos + 3, percent_pos - pos - 3);
                try {
                    double bad2_0_val = std::stod(bad2_0_str);
                    total_bad2_0 += bad2_0_val;
                    valid_results++;
                } catch (...) {
                    // Ignore parsing errors
                }
            }
        }
    }
    
    outFile << "===============================================" << std::endl;
    if (valid_results > 0) {
        double avg_bad2_0 = total_bad2_0 / valid_results;
        outFile << "Average BAD2.0: " << std::fixed << std::setprecision(2) << avg_bad2_0 << "%" << std::endl;
        outFile << "Total files processed: " << valid_results << std::endl;
    }
    
    outFile.close();
    std::cout << "✅ BAD2.0 results saved to: " << resultsFile << std::endl;
}

void toGrayscale(const MyImage& src, MyImage& gray) {
    assert(src.channels == 3 || src.channels == 4);
    gray.width = src.width;
    gray.height = src.height;
    gray.channels = 1;
    gray.data.resize(gray.width * gray.height);

    for (int i = 0; i < gray.width * gray.height; ++i) {
        int r = src.data[i * src.channels + 0];
        int g = src.data[i * src.channels + 1];
        int b = src.data[i * src.channels + 2];
        gray.data[i] = static_cast<unsigned char>(0.299 * r + 0.587 * g + 0.114 * b);
    }
}

// Function to find all disparity_SGM PNG files in a directory
std::vector<std::string> findDisparitySGMFiles(const std::string& directory) {
    std::vector<std::string> files;
    
    try {
        for (const auto& entry : std::filesystem::directory_iterator(directory)) {
            if (entry.is_regular_file()) {
                std::string filename = entry.path().filename().string();
                if (filename.find("disparity_SGM") == 0 && filename.find(".png") != std::string::npos) {
                    files.push_back(entry.path().string());
                }
            }
        }
    } catch (const std::filesystem::filesystem_error& e) {
        std::cerr << "Error accessing directory " << directory << ": " << e.what() << std::endl;
    }
    
    return files;
}

// Function to find all disparity_ELAS PNG files in a directory
std::vector<std::string> findDisparityELASFiles(const std::string& directory) {
    std::vector<std::string> files;
    
    try {
        for (const auto& entry : std::filesystem::directory_iterator(directory)) {
            if (entry.is_regular_file()) {
                std::string filename = entry.path().filename().string();
                if (filename.find("disparity_ELAS") == 0 && filename.find(".png") != std::string::npos) {
                    files.push_back(entry.path().string());
                }
            }
        }
    } catch (const std::filesystem::filesystem_error& e) {
        std::cerr << "Error accessing directory " << directory << ": " << e.what() << std::endl;
    }
    
    return files;
}

// Function to find all disparity_ADCE PNG files in a directory
std::vector<std::string> findDisparityADCEFiles(const std::string& directory) {
    std::vector<std::string> files;
    
    try {
        for (const auto& entry : std::filesystem::directory_iterator(directory)) {
            if (entry.is_regular_file()) {
                std::string filename = entry.path().filename().string();
                if (filename.find("disparity_ADCE") == 0 && filename.find(".png") != std::string::npos) {
                    files.push_back(entry.path().string());
                }
            }
        }
    } catch (const std::filesystem::filesystem_error& e) {
        std::cerr << "Error accessing directory " << directory << ": " << e.what() << std::endl;
    }
    
    return files;
}

// Function to process a single disparity file and generate triplet
bool processDisparityFile(const std::string& disparityFile, const std::string& inputPath, 
                         const std::string& gtPath, const std::string& outputDir, 
                         std::string& bad2_0_result) {
    std::cout << "\n=== Processing: " << disparityFile << " ===" << std::endl;
    
    // Extract base name from file path
    std::filesystem::path filePath(disparityFile);
    std::string baseName = filePath.stem().string(); // Remove .png extension
    
    // Generate output filename
    std::string outPath = outputDir + "/triplet_" + baseName + ".png";
    
    MyImage input_img, gt_disp, pred_disp;

    // 读取输入图像
    if (!myImReadPNG(inputPath, input_img)) {
        std::cerr << "❌ Failed to read input image: " << inputPath << std::endl;
        return false;
    }

    // 读取GT视差图 - 支持PFM和PGM格式
    bool gt_loaded = false;
    std::filesystem::path gt_file_path(gtPath);
    std::string gt_extension = gt_file_path.extension().string();
    
    if (gt_extension == ".pfm") {
        gt_loaded = myImReadPFM(gtPath, gt_disp);
    } else if (gt_extension == ".pgm") {
        gt_loaded = myImReadPGM(gtPath, gt_disp);
    } else {
        // 尝试自动检测格式
        gt_loaded = myImReadPFM(gtPath, gt_disp);
        if (!gt_loaded) {
            gt_loaded = myImReadPGM(gtPath, gt_disp);
        }
    }
    
    if (!gt_loaded) {
        std::cerr << "❌ Failed to read GT disparity: " << gtPath << std::endl;
        std::cerr << "Supported formats: .pfm, .pgm" << std::endl;
        return false;
    }

    MyImage color_gt;
    colorize_disparity(gt_disp, color_gt);

    // 读取预测视差图
    if (!myImReadPNG(disparityFile, pred_disp)) {
        std::cerr << "❌ Failed to read predicted disparity: " << disparityFile << std::endl;
        return false;
    }

    // 将 RGBA 图转换为灰度图（只对 pred_disp 做，不影响其他逻辑）
    if (pred_disp.channels != 1) {
        MyImage gray_pred;
        toGrayscale(pred_disp, gray_pred);
        pred_disp = std::move(gray_pred);
    }

    // 彩色化处理
    MyImage color_pred;
    colorize_disparity(pred_disp, color_pred);
    
    // 强制补齐为 RGB 通道
    auto toRGB = [](MyImage& img) {
        if (img.channels == 1) {
            MyImage rgb;
            rgb.width = img.width;
            rgb.height = img.height;
            rgb.channels = 3;
            rgb.data.resize(rgb.width * rgb.height * 3);
            for (int i = 0; i < img.width * img.height; ++i) {
                unsigned char val = img.data[i];
                rgb.data[i * 3] = val;
                rgb.data[i * 3 + 1] = val;
                rgb.data[i * 3 + 2] = val;
            }
            img = std::move(rgb);
        }
        if (img.channels == 4) {
            MyImage rgb;
            rgb.width = img.width;
            rgb.height = img.height;
            rgb.channels = 3;
            rgb.data.resize(rgb.width * rgb.height * 3);
            for (int i = 0; i < img.width * img.height; ++i) {
                rgb.data[i * 3 + 0] = img.data[i * 4 + 0];
                rgb.data[i * 3 + 1] = img.data[i * 4 + 1];
                rgb.data[i * 3 + 2] = img.data[i * 4 + 2];
            }
            img = std::move(rgb);
        }
    };

    toRGB(input_img);
    toRGB(color_gt);
    toRGB(color_pred);
    
    std::cout << "Image channels - input: " << input_img.channels 
              << ", gt: " << color_gt.channels 
              << ", pred: " << color_pred.channels << std::endl;
    
    // 检查图像尺寸是否一致
    std::cout << "Image dimensions:" << std::endl;
    std::cout << "  Input: " << input_img.width << "x" << input_img.height << std::endl;
    std::cout << "  GT: " << color_gt.width << "x" << color_gt.height << std::endl;
    std::cout << "  Pred: " << color_pred.width << "x" << color_pred.height << std::endl;
    
    // 确保所有图像具有相同的尺寸
    int target_width = input_img.width;
    int target_height = input_img.height;
    
    // 简单的最近邻缩放函数
    auto resizeImage = [](const MyImage& src, MyImage& dst, int new_width, int new_height) {
        dst.width = new_width;
        dst.height = new_height;
        dst.channels = src.channels;
        dst.data.resize(new_width * new_height * src.channels);
        
        for (int y = 0; y < new_height; y++) {
            for (int x = 0; x < new_width; x++) {
                int src_x = (int)(x * src.width / (float)new_width);
                int src_y = (int)(y * src.height / (float)new_height);
                
                if (src_x >= src.width) src_x = src.width - 1;
                if (src_y >= src.height) src_y = src.height - 1;
                
                for (int c = 0; c < src.channels; c++) {
                    dst.data[(y * new_width + x) * src.channels + c] = 
                        src.data[(src_y * src.width + src_x) * src.channels + c];
                }
            }
        }
    };
    
    // 调整GT图像尺寸
    MyImage resized_gt;
    if (color_gt.width != target_width || color_gt.height != target_height) {
        std::cout << "Resizing GT image from " << color_gt.width << "x" << color_gt.height 
                  << " to " << target_width << "x" << target_height << std::endl;
        resizeImage(color_gt, resized_gt, target_width, target_height);
    } else {
        resized_gt = color_gt;
    }
    
    // 调整预测图像尺寸
    MyImage resized_pred;
    if (color_pred.width != target_width || color_pred.height != target_height) {
        std::cout << "Resizing predicted image from " << color_pred.width << "x" << color_pred.height 
                  << " to " << target_width << "x" << target_height << std::endl;
        resizeImage(color_pred, resized_pred, target_width, target_height);
    } else {
        resized_pred = color_pred;
    }
    
    std::cout << "Final image dimensions (all resized to match input):" << std::endl;
    std::cout << "  Input: " << input_img.width << "x" << input_img.height << std::endl;
    std::cout << "  GT: " << resized_gt.width << "x" << resized_gt.height << std::endl;
    std::cout << "  Pred: " << resized_pred.width << "x" << resized_pred.height << std::endl;

    // 拼接输出
    MyImage triplet;
    hstack3(input_img, resized_gt, resized_pred, triplet);
    if (!myImWritePNG(outPath, triplet)) {
        std::cerr << "❌ Failed to save triplet image: " << outPath << std::endl;
        return false;
    }

    std::cout << "✅ Triplet image saved to: " << outPath << std::endl;
    
    // 单独保存彩色视差图
    // 保存GT彩色视差图（只保存一次，避免重复）
    std::string gt_color_path = outputDir + "/gt_disparity_color.png";
    if (!std::filesystem::exists(gt_color_path)) {
        if (!myImWritePNG(gt_color_path, resized_gt)) {
            std::cerr << "❌ Failed to save GT color disparity: " << gt_color_path << std::endl;
        } else {
            std::cout << "✅ GT color disparity saved to: " << gt_color_path << std::endl;
        }
    } else {
        std::cout << "ℹ️  GT color disparity already exists: " << gt_color_path << std::endl;
    }
    
    // 保存预测彩色视差图（每个算法单独保存）
    std::string pred_color_path = outputDir + "/pred_disparity_color_" + baseName + ".png";
    if (!myImWritePNG(pred_color_path, resized_pred)) {
        std::cerr << "❌ Failed to save predicted color disparity: " << pred_color_path << std::endl;
    } else {
        std::cout << "✅ Predicted color disparity saved to: " << pred_color_path << std::endl;
    }

    // 显示视差值范围信息（使用原始视差图）
    float gt_min = 1e9f, gt_max = -1e9f;
    float pred_min = 1e9f, pred_max = -1e9f;
    for (int i = 0; i < gt_disp.width * gt_disp.height; ++i) {
        float gt_val = static_cast<float>(gt_disp.data[i]);
        float pred_val = static_cast<float>(pred_disp.data[i]);
        if (gt_val > 0 && gt_val < 1000) {
            gt_min = std::min(gt_min, gt_val);
            gt_max = std::max(gt_max, gt_val);
        }
        if (pred_val > 0 && pred_val < 1000) {
            pred_min = std::min(pred_min, pred_val);
            pred_max = std::max(pred_max, pred_val);
        }
    }
    std::cout << "  GT disparity range: [" << gt_min << ", " << gt_max << "]" << std::endl;
    std::cout << "  Pred disparity range: [" << pred_min << ", " << pred_max << "]" << std::endl;
    
    // 计算BAD2.0指标 - 需要确保GT和预测视差图尺寸一致
    // 创建调整后的GT和预测视差图用于计算
    MyImage gt_disp_resized, pred_disp_resized;
    
    // 调整GT视差图尺寸以匹配输入图像
    if (gt_disp.width != target_width || gt_disp.height != target_height) {
        std::cout << "Resizing GT disparity for BAD2.0 calculation from " << gt_disp.width << "x" << gt_disp.height 
                  << " to " << target_width << "x" << target_height << std::endl;
        resizeImage(gt_disp, gt_disp_resized, target_width, target_height);
    } else {
        gt_disp_resized = gt_disp;
    }
    
    // 调整预测视差图尺寸以匹配输入图像
    if (pred_disp.width != target_width || pred_disp.height != target_height) {
        std::cout << "Resizing predicted disparity for BAD2.0 calculation from " << pred_disp.width << "x" << pred_disp.height 
                  << " to " << target_width << "x" << target_height << std::endl;
        resizeImage(pred_disp, pred_disp_resized, target_width, target_height);
    } else {
        pred_disp_resized = pred_disp;
    }
    
    // 使用调整后的视差图计算BAD2.0
    double bad2_0 = calculateBAD2_0(gt_disp_resized, pred_disp_resized);
    if (bad2_0 >= 0.0) {
        std::cout << "✅ BAD2.0: " << std::fixed << std::setprecision(2) << bad2_0 << "%" << std::endl;
        
        // 提取文件名（不包含路径）
        std::filesystem::path filePath(disparityFile);
        std::string filename = filePath.filename().string();
        
        // 格式化结果字符串
        std::ostringstream resultStream;
        resultStream << filename << " | " << std::fixed << std::setprecision(2) << bad2_0 << "%";
        bad2_0_result = resultStream.str();
    } else {
        std::cout << "❌ BAD2.0 calculation failed" << std::endl;
        bad2_0_result = "";
    }

    return true;
}

int main(int argc, char* argv[]) {
    std::cout << "=== SGM, ELAS & ADCE Disparity Triplet Generation for Multiple Datasets ===" << std::endl;
    
    // Check command line arguments
    if (argc != 2) {
        std::cout << "Usage: " << argv[0] << " <dataset_name>" << std::endl;
        std::cout << "  dataset_name: any dataset folder name (e.g., test1, test2, test7, test8)" << std::endl;
        std::cout << "Example: " << argv[0] << " test8" << std::endl;
        return 1;
    }
    
    std::string datasetName = argv[1];
    
    // Validate dataset name (allow any non-empty string)
    if (datasetName.empty()) {
        std::cout << "Error: Dataset name cannot be empty" << std::endl;
        return 1;
    }
    
    // Define the dataset to process
    std::vector<std::string> datasets = {datasetName};
    
    int totalSuccessCount = 0;
    int totalFileCount = 0;
    
    // Process each dataset
    for (const auto& datasetName : datasets) {
        std::cout << "\n" << std::string(60, '=') << std::endl;
        std::cout << "Processing dataset: " << datasetName << std::endl;
        std::cout << std::string(60, '=') << std::endl;
        
        // 设置基础路径
        std::string input_path = "../data/left/" + datasetName + ".png";
        std::string outputDir = "../output/" + datasetName;
        
        // 自动检测GT视差图格式 (PFM或PGM)
        std::string gt_path_pfm = "../output/" + datasetName + "/disp0.pfm";
        std::string gt_path_pgm = "../output/" + datasetName + "/disp0.pgm";
        std::string gt_path;
        
        if (std::filesystem::exists(gt_path_pfm)) {
            gt_path = gt_path_pfm;
        } else if (std::filesystem::exists(gt_path_pgm)) {
            gt_path = gt_path_pgm;
        } else {
            gt_path = gt_path_pfm; // 默认使用PFM路径，错误会在后面处理
        }
        
        std::cout << "Input image: " << input_path << std::endl;
        std::cout << "GT disparity: " << gt_path << std::endl;
        std::cout << "Output directory: " << outputDir << std::endl;
        
        // 检查输入文件是否存在
        if (!std::filesystem::exists(input_path)) {
            std::cerr << "❌ Input image not found: " << input_path << std::endl;
            std::cerr << "Please ensure the dataset '" << datasetName << "' exists in ../data/left/" << std::endl;
            std::cout << "Skipping dataset " << datasetName << "..." << std::endl;
            continue; // Skip to next dataset
        }
        
        // 检查GT文件是否存在 (支持PFM和PGM格式)
        if (!std::filesystem::exists(gt_path_pfm) && !std::filesystem::exists(gt_path_pgm)) {
            std::cerr << "❌ GT disparity not found: " << std::endl;
            std::cerr << "  Checked: " << gt_path_pfm << std::endl;
            std::cerr << "  Checked: " << gt_path_pgm << std::endl;
            std::cerr << "Please ensure the ground truth file (disp0.pfm or disp0.pgm) exists in ../output/" << datasetName << "/" << std::endl;
            std::cout << "Skipping dataset " << datasetName << "..." << std::endl;
            continue; // Skip to next dataset
        }
        
        // 查找所有disparity_SGM文件
        std::vector<std::string> sgmFiles = findDisparitySGMFiles(outputDir);
        
        // 查找所有disparity_ELAS文件
        std::vector<std::string> elasFiles = findDisparityELASFiles(outputDir);
        
        // 查找所有disparity_ADCE文件
        std::vector<std::string> adceFiles = findDisparityADCEFiles(outputDir);
        
        if (sgmFiles.empty() && elasFiles.empty() && adceFiles.empty()) {
            std::cout << "No disparity_SGM*.png, disparity_ELAS*.png, or disparity_ADCE*.png files found in directory: " << outputDir << std::endl;
            std::cout << "Please run testdisparitySGM.cpp, testdisparityELAS.cpp, and testdisparityADCE.cpp first to generate disparity maps for " << datasetName << "." << std::endl;
            continue; // Skip to next dataset
        }
        
        int datasetSuccessCount = 0;
        int datasetFileCount = 0;
        std::vector<std::string> bad2_0_results;
        
        // Process SGM files
        if (!sgmFiles.empty()) {
            std::cout << "\n--- Processing SGM Disparity Files ---" << std::endl;
            std::cout << "Found " << sgmFiles.size() << " disparity_SGM files in " << datasetName << ":" << std::endl;
            for (const auto& file : sgmFiles) {
                std::cout << "  - " << std::filesystem::path(file).filename().string() << std::endl;
            }
            
            for (const auto& disparityFile : sgmFiles) {
                std::string bad2_0_result;
                if (processDisparityFile(disparityFile, input_path, gt_path, outputDir, bad2_0_result)) {
                    datasetSuccessCount++;
                    if (!bad2_0_result.empty()) {
                        bad2_0_results.push_back(bad2_0_result);
                    }
                }
                std::cout << std::string(50, '=') << std::endl;
            }
            datasetFileCount += sgmFiles.size();
        }
        
        // Process ELAS files
        if (!elasFiles.empty()) {
            std::cout << "\n--- Processing ELAS Disparity Files ---" << std::endl;
            std::cout << "Found " << elasFiles.size() << " disparity_ELAS files in " << datasetName << ":" << std::endl;
            for (const auto& file : elasFiles) {
                std::cout << "  - " << std::filesystem::path(file).filename().string() << std::endl;
            }
            
            for (const auto& disparityFile : elasFiles) {
                std::string bad2_0_result;
                if (processDisparityFile(disparityFile, input_path, gt_path, outputDir, bad2_0_result)) {
                    datasetSuccessCount++;
                    if (!bad2_0_result.empty()) {
                        bad2_0_results.push_back(bad2_0_result);
                    }
                }
                std::cout << std::string(50, '=') << std::endl;
            }
            datasetFileCount += elasFiles.size();
        }
        
        // Process ADCE files
        if (!adceFiles.empty()) {
            std::cout << "\n--- Processing ADCE Disparity Files ---" << std::endl;
            std::cout << "Found " << adceFiles.size() << " disparity_ADCE files in " << datasetName << ":" << std::endl;
            for (const auto& file : adceFiles) {
                std::cout << "  - " << std::filesystem::path(file).filename().string() << std::endl;
            }
            
            for (const auto& disparityFile : adceFiles) {
                std::string bad2_0_result;
                if (processDisparityFile(disparityFile, input_path, gt_path, outputDir, bad2_0_result)) {
                    datasetSuccessCount++;
                    if (!bad2_0_result.empty()) {
                        bad2_0_results.push_back(bad2_0_result);
                    }
                }
                std::cout << std::string(50, '=') << std::endl;
            }
            datasetFileCount += adceFiles.size();
        }
        
        // 总结当前数据集
        std::cout << "\n=== Processing Summary for " << datasetName << " ===" << std::endl;
        std::cout << "Total files processed: " << datasetFileCount << std::endl;
        std::cout << "  - SGM files: " << sgmFiles.size() << std::endl;
        std::cout << "  - ELAS files: " << elasFiles.size() << std::endl;
        std::cout << "  - ADCE files: " << adceFiles.size() << std::endl;
        std::cout << "Successful triplets: " << datasetSuccessCount << std::endl;
        std::cout << "Failed triplets: " << (datasetFileCount - datasetSuccessCount) << std::endl;
        
        if (datasetSuccessCount > 0) {
            std::cout << "\nGenerated images for " << datasetName << ":" << std::endl;
            std::cout << "Triplet images:" << std::endl;
            if (!sgmFiles.empty()) {
                std::cout << "  - triplet_disparity_SGM_[filename].png" << std::endl;
            }
            if (!elasFiles.empty()) {
                std::cout << "  - triplet_disparity_ELAS_[filename].png" << std::endl;
            }
            if (!adceFiles.empty()) {
                std::cout << "  - triplet_disparity_ADCE_[filename].png" << std::endl;
            }
            std::cout << "  Each triplet contains: Original | GT Disparity | Predicted Disparity" << std::endl;
            
            std::cout << "\nSeparate color disparity images:" << std::endl;
            std::cout << "  - gt_disparity_color.png (Ground Truth - saved once)" << std::endl;
            if (!sgmFiles.empty()) {
                std::cout << "  - pred_disparity_color_disparity_SGM_[filename].png" << std::endl;
            }
            if (!elasFiles.empty()) {
                std::cout << "  - pred_disparity_color_disparity_ELAS_[filename].png" << std::endl;
            }
            if (!adceFiles.empty()) {
                std::cout << "  - pred_disparity_color_disparity_ADCE_[filename].png" << std::endl;
            }
            
            // 保存BAD2.0结果到文件
            if (!bad2_0_results.empty()) {
                saveBAD2_0Results(outputDir, datasetName, bad2_0_results);
                std::cout << "\nBAD2.0 metrics calculated for " << bad2_0_results.size() << " files" << std::endl;
            }
        }
        
        // Update global counters
        totalSuccessCount += datasetSuccessCount;
        totalFileCount += datasetFileCount;
    }
    
    // Overall summary
    std::cout << "\n" << std::string(60, '=') << std::endl;
    std::cout << "=== OVERALL PROCESSING SUMMARY ===" << std::endl;
    std::cout << std::string(60, '=') << std::endl;
    std::cout << "Total datasets processed: " << datasets.size() << std::endl;
    std::cout << "Total files processed: " << totalFileCount << std::endl;
    std::cout << "Total successful triplets: " << totalSuccessCount << std::endl;
    std::cout << "Total failed triplets: " << (totalFileCount - totalSuccessCount) << std::endl;
    
    if (totalSuccessCount > 0) {
        std::cout << "\nGenerated images for all datasets:" << std::endl;
        std::cout << "Triplet images:" << std::endl;
        std::cout << "  - triplet_disparity_SGM_[filename].png" << std::endl;
        std::cout << "  - triplet_disparity_ELAS_[filename].png" << std::endl;
        std::cout << "  - triplet_disparity_ADCE_[filename].png" << std::endl;
        std::cout << "  Each triplet contains: Original | GT Disparity | Predicted Disparity" << std::endl;
        
        std::cout << "\nSeparate color disparity images:" << std::endl;
        std::cout << "  - gt_disparity_color.png (Ground Truth - one per dataset)" << std::endl;
        std::cout << "  - pred_disparity_color_disparity_SGM_[filename].png" << std::endl;
        std::cout << "  - pred_disparity_color_disparity_ELAS_[filename].png" << std::endl;
        std::cout << "  - pred_disparity_color_disparity_ADCE_[filename].png" << std::endl;
    }
    
    std::cout << "\n=== All Processing Completed! ===" << std::endl;
    std::cout << "Results saved in respective output directories:" << std::endl;
    for (const auto& datasetName : datasets) {
        std::cout << "  - ../output/" << datasetName << "/" << std::endl;
    }
    
    return 0;
}
