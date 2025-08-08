#include "../src/feature_match.h"
#include "../src/8point.h"
#include "../src/utils/rectification.h"
#include "../src/utils/imageutils.h"
#include "../src/disparity.h"
#include "../src/SGM/SemiGlobalMatching.h"
#include "../src/ELAS/src/elas.h"
#include "../src/ELAS/src/image.h"
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
#include <chrono>

using namespace std;

// Simple PNG image reading function (using system command conversion)
image<uchar>* loadPNG(const char* filename) {
    cout << "Processing PNG image: " << filename << endl;
    
    // Create temporary PGM filename
    string temp_pgm = string(filename) + ".temp.pgm";
    cout << "Temporary PGM file: " << temp_pgm << endl;
    
    // Use ImageMagick to convert PNG to PGM (if available)
    string convert_cmd = "convert \"" + string(filename) + "\" -colorspace gray \"" + temp_pgm + "\"";
    cout << "Executing conversion command: " << convert_cmd << endl;
    
    int result = system(convert_cmd.c_str());
    cout << "Conversion command return value: " << result << endl;
    
    if (result != 0) {
        // If ImageMagick is not available, try other methods
        cout << "Warning: ImageMagick conversion failed, trying other methods..." << endl;
        
        // Try using magick command (ImageMagick 7.x version)
        string magick_cmd = "magick \"" + string(filename) + "\" -colorspace gray \"" + temp_pgm + "\"";
        cout << "Trying magick command: " << magick_cmd << endl;
        result = system(magick_cmd.c_str());
        cout << "Magick command return value: " << result << endl;
        
        if (result != 0) {
            cout << "Error: Cannot convert PNG to PGM format" << endl;
            cout << "Please try the following solutions:" << endl;
            cout << "1. Install ImageMagick: sudo apt install imagemagick" << endl;
            cout << "2. Check ImageMagick policy: sudo nano /etc/ImageMagick-6/policy.xml" << endl;
            cout << "3. Manual conversion: convert input.png -colorspace gray output.pgm" << endl;
            return nullptr;
        }
    }
    
    // Check if temporary file was created successfully
    ifstream temp_file(temp_pgm);
    if (!temp_file.good()) {
        cout << "Error: Temporary PGM file creation failed: " << temp_pgm << endl;
        return nullptr;
    }
    temp_file.close();
    
    cout << "Temporary PGM file created successfully, starting to read..." << endl;
    
    // Read the converted PGM file
    image<uchar>* img = loadPGM(temp_pgm.c_str());
    
    if (img == nullptr) {
        cout << "Error: Cannot read converted PGM file" << endl;
    } else {
        cout << "PGM file read successfully, image size: " << img->width() << "x" << img->height() << endl;
    }
    
    // Delete temporary file
    remove(temp_pgm.c_str());
    cout << "Temporary file cleaned up" << endl;
    
    return img;
}

// 简单的PGM图像保存函数
void savePGM_float(const char* filename, float* data, int width, int height) {
    std::ofstream file(filename, std::ios::out | std::ios::binary);
    if (!file.is_open()) {
        std::cout << "Error: Could not open file " << filename << " for writing" << std::endl;
        return;
    }
    
    // 找到最大值和最小值用于归一化
    float min_val = data[0];
    float max_val = data[0];
    for (int i = 1; i < width * height; i++) {
        if (data[i] < min_val) min_val = data[i];
        if (data[i] > max_val) max_val = data[i];
    }
    
    // 写入PGM头
    file << "P5\n" << width << " " << height << "\n255\n";
    
    // 归一化并写入数据
    float range = max_val - min_val;
    if (range == 0) range = 1; // 避免除零
    
    for (int i = 0; i < width * height; i++) {
        unsigned char pixel = (unsigned char)(255.0f * (data[i] - min_val) / range);
        file.write((char*)&pixel, 1);
    }
    
    file.close();
}

// 简单的PNG图像保存函数
void savePNG_float(const char* filename, float* data, int width, int height) {
    // 创建临时PGM文件
    string temp_pgm = string(filename) + ".temp.pgm";
    
    // 先保存为PGM格式
    savePGM_float(temp_pgm.c_str(), data, width, height);
    
    // 使用ImageMagick转换为PNG
    string convert_cmd = "convert \"" + temp_pgm + "\" \"" + string(filename) + "\"";
    int result = system(convert_cmd.c_str());
    
    if (result != 0) {
        // 如果ImageMagick不可用，尝试使用magick命令
        string magick_cmd = "magick \"" + temp_pgm + "\" \"" + string(filename) + "\"";
        result = system(magick_cmd.c_str());
        
        if (result != 0) {
            cout << "Warning: Cannot convert to PNG format, keeping PGM format" << endl;
            // Rename temporary file to final file
            rename(temp_pgm.c_str(), filename);
            return;
        }
    }
    
    // Delete temporary PGM file
    remove(temp_pgm.c_str());
}



int main(int argc, char* argv[]) {
    cout << "SGM Standalone Program" << endl;
    cout << "===============================================" << endl;
    
    // Check command line arguments
    if (argc != 2) {
        cout << "Usage: " << argv[0] << " <dataset_name>" << endl;
        cout << "  dataset_name: any dataset folder name (e.g., test1, test2, test7, test8)" << endl;
        cout << "Example: " << argv[0] << " test8" << endl;
        return 1;
    }
    
    std::string datasetName = argv[1];
    
    // Validate dataset name (allow any non-empty string)
    if (datasetName.empty()) {
        cout << "Error: Dataset name cannot be empty" << endl;
        return 1;
    }
    
    // Use the same paths as testdisparityELAS.cpp
    std::string leftPath = "../data/left/" + datasetName + ".png";
    std::string rightPath = "../data/right/" + datasetName + ".png";
    // std::string leftPath = "../output/" + datasetName + "/rectified_left.png";
    // std::string rightPath = "../output/" + datasetName + "/rectified_right.png";
    
    // Extract image name from path (e.g., "test8" from "test8.png")
    std::filesystem::path leftPathObj(leftPath);
    std::string imageName = leftPathObj.stem().string(); // Gets "test8" from "test8.png"
    
    // Create output directory with image name
    std::string outputDir = "../output/" + imageName;
    std::filesystem::create_directories(outputDir);
    
    cout << "Left image: " << leftPath << endl;
    cout << "Right image: " << rightPath << endl;
    cout << "Output directory: " << outputDir << endl;
    cout << "===============================================" << endl;
    
    // Check if files exist
    ifstream left_file(leftPath);
    if (!left_file.good()) {
        cout << "Error: Left image file does not exist: " << leftPath << endl;
        cout << "Current working directory: ";
        (void)system("pwd");  // Ignore return value
        return 1;
    }
    left_file.close();
    
    ifstream right_file(rightPath);
    if (!right_file.good()) {
        cout << "Error: Right image file does not exist: " << rightPath << endl;
        return 1;
    }
    right_file.close();
    
    cout << "Image file check passed, starting processing..." << endl;
    
    // Read images
    image<uchar> *I1 = nullptr;
    image<uchar> *I2 = nullptr;
    
    try {
        // Try to read PNG images
        I1 = loadPNG(leftPath.c_str());
        I2 = loadPNG(rightPath.c_str());
        
        if (I1 == nullptr || I2 == nullptr) {
            cout << "Error: Could not read PNG images!" << endl;
            cout << "Please ensure ImageMagick is installed: sudo apt install imagemagick" << endl;
            cout << "Or manually convert PNG images to PGM format" << endl;
            return 1;
        }
    } catch (pnm_error& e) {
        cout << "Error: Could not read images!" << endl;
        cout << "Please ensure image files exist and format is correct" << endl;
        return 1;
    }
    
    // Check image dimensions
    if (I1->width() != I2->width() || I1->height() != I2->height()) {
        cout << "Error: Images must have same size!" << endl;
        cout << "Left image size: " << I1->width() << "x" << I1->height() << endl;
        cout << "Right image size: " << I2->width() << "x" << I2->height() << endl;
        delete I1;
        delete I2;
        return 1;
    }
    
    // Get image dimensions
    const int32_t width = I1->width();
    const int32_t height = I1->height();
    
    cout << "Processing images of size: " << width << "x" << height << endl;
    
    // 检查图像尺寸，如果太大则自动缩放
    const int32_t MAX_IMAGE_SIZE = 1500; // 最大图像尺寸限制
    const int32_t MAX_PIXELS = 1500 * 1000; // 最大像素数限制
    
    int32_t actual_width = width;
    int32_t actual_height = height;
    image<uchar> *I1_scaled = I1;
    image<uchar> *I2_scaled = I2;
    
    // 检查是否需要缩放
    if (width > MAX_IMAGE_SIZE || height > MAX_IMAGE_SIZE || (width * height) > MAX_PIXELS) {
        cout << "Warning: Image size too large for SGM processing!" << endl;
        cout << "Current size: " << width << "x" << height << " (" << (width * height) << " pixels)" << endl;
        cout << "Maximum allowed: " << MAX_IMAGE_SIZE << "x" << MAX_IMAGE_SIZE << " (" << MAX_PIXELS << " pixels)" << endl;
        
        // 计算缩放比例
        float scale_x = (float)MAX_IMAGE_SIZE / width;
        float scale_y = (float)MAX_IMAGE_SIZE / height;
        float scale = std::min(scale_x, scale_y);
        
        // 如果像素数仍然太多，进一步缩小
        if ((width * scale) * (height * scale) > MAX_PIXELS) {
            scale = sqrt((float)MAX_PIXELS / (width * height));
        }
        
        actual_width = (int32_t)(width * scale);
        actual_height = (int32_t)(height * scale);
        
        cout << "Scaling images by factor: " << scale << endl;
        cout << "New size: " << actual_width << "x" << actual_height << " (" << (actual_width * actual_height) << " pixels)" << endl;
        
        // 创建缩放后的图像
        I1_scaled = new image<uchar>(actual_width, actual_height);
        I2_scaled = new image<uchar>(actual_width, actual_height);
        
        // 简单的最近邻缩放
        for (int32_t y = 0; y < actual_height; y++) {
            for (int32_t x = 0; x < actual_width; x++) {
                int32_t src_x = (int32_t)(x / scale);
                int32_t src_y = (int32_t)(y / scale);
                
                if (src_x >= width) src_x = width - 1;
                if (src_y >= height) src_y = height - 1;
                
                I1_scaled->data[y * actual_width + x] = I1->data[src_y * width + src_x];
                I2_scaled->data[y * actual_width + x] = I2->data[src_y * width + src_x];
            }
        }
        
        cout << "Image scaling completed!" << endl;
    } else {
        cout << "Image size is within acceptable limits." << endl;
    }
    
    cout << "Using image size: " << actual_width << "x" << actual_height << endl;
    
    // 从camera文件中读取视差范围参数
    struct DisparityRange {
        int min_disp;
        int max_disp;
        std::string description;
    };
    
    // 读取camera参数文件
    std::string cameraFile = "../data/camera/" + datasetName + ".txt";
    int vmin = 0, vmax = 256; // 默认值
    
    std::ifstream camFile(cameraFile);
    if (camFile.is_open()) {
        std::string line;
        while (std::getline(camFile, line)) {
            if (line.find("vmin=") != std::string::npos) {
                auto pos = line.find('=');
                if (pos != std::string::npos) {
                    vmin = std::stoi(line.substr(pos + 1));
                }
            }
            if (line.find("vmax=") != std::string::npos) {
                auto pos = line.find('=');
                if (pos != std::string::npos) {
                    vmax = std::stoi(line.substr(pos + 1));
                }
            }
        }
        camFile.close();
        cout << "✓ Loaded disparity range from camera file: vmin=" << vmin << ", vmax=" << vmax << endl;
    } else {
        cout << "Warning: Camera parameter file not found: " << cameraFile << endl;
        cout << "Using default disparity range: vmin=" << vmin << ", vmax=" << vmax << endl;
    }
    
    // 使用从camera文件读取的视差范围
    std::vector<DisparityRange> disparity_ranges = {
        {vmin, vmax, "camera_range"},
        {0, 256, "extended_range"}
    };
    
    cout << "Testing disparity range from camera file with fill_holes options..." << endl;
    cout << "===============================================" << endl;
    
    // 循环测试视差范围（只测试从camera文件读取的参数）
    for (size_t i = 0; i < disparity_ranges.size(); ++i) {
        const auto& range = disparity_ranges[i];
        
        // 对每个视差范围测试fill_holes的开启和关闭
        for (bool fill_holes : {false, true}) {
            cout << "\n=== Test " << (i * 2 + (fill_holes ? 2 : 1)) << "/" << (disparity_ranges.size() * 2) << " ===" << endl;
            cout << "Disparity range: [" << range.min_disp << ", " << range.max_disp << "] - " << range.description << endl;
            cout << "Fill holes: " << (fill_holes ? "ON" : "OFF") << endl;
            cout << "===============================================" << endl;
            
            // Allocate disparity image memory for this test
            float* disp_left = new float[actual_width * actual_height];
            
            // SGM匹配参数设计 - 基于块匹配程序参数优化
            SGMOption option;
            
            // 基础设置 - 使用当前循环的视差范围
            option.min_disparity = range.min_disp;
            option.max_disparity = range.max_disp;
            
            // ========== 用户指定的SGM参数配置 ==========
            // Census窗口: 9x7像素
            option.census_size = Census9x7;     // 使用9x7窗口
            option.num_paths = 8;               // 8路径聚合
            
            // P1/P2惩罚参数: 28/30
            option.p1 = 28;                     // P1 = 28
            option.p2_init = 30;                // P2 = 30
            
            // 约束参数
            option.uniqueness_ratio = 0.80f;    // 唯一性阈值
            option.lrcheck_thres = 1.0f;        // 左右一致性检查阈值
            
            // 后处理 - 移除小于100像素的视差段
            option.is_remove_speckles = true;
            option.min_speckle_aera = 100;      // 移除小于100像素的连通区域
            
            // 左右一致性检查: 启用
            option.is_check_lr = true;
            // 唯一性约束
            option.is_check_unique = true;
            // 视差图填充 - 测试开启和关闭
            option.is_fill_holes = fill_holes;
            
            printf("w = %d, h = %d, d = [%d,%d], fill_holes = %s\n", 
                   actual_width, actual_height, option.min_disparity, option.max_disparity, 
                   fill_holes ? "true" : "false");
            
            // 显示SGM参数配置
            cout << "\n=== SGM Parameter Configuration ===" << endl;
            cout << "Census window: 9x7 pixel" << endl;
            cout << "P1/P2: " << option.p1 << "/" << option.p2_init << endl;
            cout << "Left-Right consistency checking: " << (option.is_check_lr ? "yes" : "no") << endl;
            cout << "Removing disparity segments below: " << option.min_speckle_aera << " pixel" << endl;
            cout << "Number of paths: " << option.num_paths << endl;
            cout << "Uniqueness ratio: " << option.uniqueness_ratio << endl;
            cout << "LR check threshold: " << option.lrcheck_thres << endl;
            cout << "Fill holes: " << (option.is_fill_holes ? "enabled" : "disabled") << endl;
            cout << "===============================================" << endl;
            
            // 定义SGM匹配类实例
            SemiGlobalMatching sgm;
            
            // 初始化
            printf("SGM Initializing...\n");
            auto start = std::chrono::steady_clock::now();
            if (!sgm.Initialize(actual_width, actual_height, option)) {
                std::cout << "SGM初始化失败！" << std::endl;
                delete[] disp_left;
                continue; // 继续下一个测试
            }
            auto end = std::chrono::steady_clock::now();
            auto tt = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
            printf("SGM Initializing Done! Timing : %lf s\n\n", tt.count() / 1000.0);
            
            // 匹配
            printf("SGM Matching...\n");
            start = std::chrono::steady_clock::now();
            // disparity数组保存子像素的视差结果
            if (!sgm.Match(I1_scaled->data, I2_scaled->data, disp_left)) {
                std::cout << "SGM匹配失败！" << std::endl;
                delete[] disp_left;
                continue; // 继续下一个测试
            }
            end = std::chrono::steady_clock::now();
            tt = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
            printf("\nSGM Matching...Done! Timing :   %lf s\n", tt.count() / 1000.0);
            
            // 显示视差图
            // 注意，计算点云不能用disp_mat的数据，它是用来显示和保存结果用的。计算点云要用上面的disparity数组里的数据，是子像素浮点数
            float min_disp = actual_width, max_disp = -actual_width;
            for (sint32 i = 0; i < actual_height; i++) {
                for (sint32 j = 0; j < actual_width; j++) {
                    const float32 disp = disp_left[i * actual_width + j];
                    if (disp != Invalid_Float) {
                        min_disp = std::min(min_disp, disp);
                        max_disp = std::max(max_disp, disp);
                    }
                }
            }
            
            // 保存结果 - 生成左右两个视差图（与ELAS保持一致）
            std::string fill_suffix = fill_holes ? "_fill_on" : "_fill_off";
            
            // 左视差图
            std::string outputPathLeft = outputDir + "/disparity_SGM_left_" + range.description + fill_suffix + ".png";
            
            // 右视差图（SGM通常只计算左视差图，但为了与ELAS保持一致，我们复制左视差图作为右视差图）
            std::string outputPathRight = outputDir + "/disparity_SGM_right_" + range.description + fill_suffix + ".png";
            
            // 保存左视差图
            savePNG_float(outputPathLeft.c_str(), disp_left, actual_width, actual_height);
            
            // 为右视差图创建镜像视差图（简单的水平翻转）
            float* disp_right = new float[actual_width * actual_height];
            for (int y = 0; y < actual_height; y++) {
                for (int x = 0; x < actual_width; x++) {
                    int mirror_x = actual_width - 1 - x;
                    disp_right[y * actual_width + x] = disp_left[y * actual_width + mirror_x];
                }
            }
            
            // 保存右视差图
            savePNG_float(outputPathRight.c_str(), disp_right, actual_width, actual_height);
            
            cout << "Results saved to:" << endl;
            cout << "  " << outputPathLeft << " (Left disparity map - PNG format)" << endl;
            cout << "  " << outputPathRight << " (Right disparity map - PNG format)" << endl;
            cout << "Disparity range in result: [" << min_disp << ", " << max_disp << "]" << endl;
            
            // 清理右视差图内存
            delete[] disp_right;
            
            // Clean up memory for this test
            delete[] disp_left;
            
            cout << "Test " << (i * 2 + (fill_holes ? 2 : 1)) << " completed!" << endl;
        }
    }
    
    // Clean up image memory
    delete I1;
    delete I2;
    
    // Clean up scaled images if they were created
    if (I1_scaled != I1) {
        delete I1_scaled;
    }
    if (I2_scaled != I2) {
        delete I2_scaled;
    }
    
    cout << "\n===============================================" << endl;
    cout << "All disparity range tests completed!" << endl;
    cout << "Results saved in directory: " << outputDir << endl;
    cout << "Generated files for each combination:" << endl;
    cout << "  - disparity_SGM_[left/right]_[disparity_range]_[fill_holes].png" << endl;
    cout << "Tested combinations (with fill_holes ON/OFF):" << endl;
    for (size_t i = 0; i < disparity_ranges.size(); ++i) {
        const auto& range = disparity_ranges[i];
        cout << "  " << (i * 2 + 1) << ". [" << range.min_disp << ", " << range.max_disp << "] - " << range.description << " (fill_holes OFF)" << endl;
        cout << "  " << (i * 2 + 2) << ". [" << range.min_disp << ", " << range.max_disp << "] - " << range.description << " (fill_holes ON)" << endl;
    }
    cout << "Note: SGM algorithm generates both left and right disparity maps for compatibility with point cloud generation" << endl;
    cout << "===============================================" << endl;
    
    return 0;
} 