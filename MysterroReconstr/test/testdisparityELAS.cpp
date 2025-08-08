#include "../src/feature_match.h"
#include "../src/8point.h"
#include "../src/utils/rectification.h"
#include "../src/utils/imageutils.h"
#include "../src/disparity.h"
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

using namespace std;

// 定义无效视差值（与SGM保持一致）
const float Invalid_Float = -1.0f;

// SGM风格的填充函数 - 与SGM原始实现保持一致
void FillHolesInDispMapSGM(float* disp_left, int32_t width, int32_t height, 
                          int32_t min_disparity, int32_t max_disparity) {
    std::vector<float> disp_collects;

    // 定义8个方向 - 与SGM保持一致
    const float pi = 3.1415926f;
    float angle1[8] = { pi, 3 * pi / 4, pi / 2, pi / 4, 0, 7 * pi / 4, 3 * pi / 2, 5 * pi / 4 };
    float angle2[8] = { pi, 5 * pi / 4, 3 * pi / 2, 7 * pi / 4, 0, pi / 4, pi / 2, 3 * pi / 4 };
    float *angle = angle1;
    
    // 最大搜索行程，与SGM保持一致
    const int32_t max_search_length = 1.0 * std::max(abs(max_disparity), abs(min_disparity));

    float* disp_ptr = disp_left;
    for (int32_t k = 0; k < 3; k++) {
        // 收集需要填充的像素
        std::vector<std::pair<int32_t, int32_t>> trg_pixels;
        
        if (k == 2) {
            // 第三次循环处理所有无效像素
            for (int32_t i = 0; i < height; i++) {
                for (int32_t j = 0; j < width; j++) {
                    if (disp_ptr[i * width + j] == Invalid_Float) {
                        trg_pixels.emplace_back(i, j);
                    }
                }
            }
        } else {
            // 前两次循环处理所有无效像素（简化版本，因为ELAS没有occlusions_和mismatches_）
            for (int32_t i = 0; i < height; i++) {
                for (int32_t j = 0; j < width; j++) {
                    if (disp_ptr[i * width + j] == Invalid_Float) {
                        trg_pixels.emplace_back(i, j);
                    }
                }
            }
        }
        
        if (trg_pixels.empty()) {
            continue;
        }
        
        std::vector<float> fill_disps(trg_pixels.size());

        // 遍历待处理像素
        for (auto n = 0u; n < trg_pixels.size(); n++) {
            auto& pix = trg_pixels[n];
            const int32_t y = pix.first;
            const int32_t x = pix.second;

            if (y == height / 2) {
                angle = angle2; 
            }

            // 收集8个方向上遇到的首个有效视差值
            disp_collects.clear();
            for (int32_t s = 0; s < 8; s++) {
                const float ang = angle[s];
                const float sina = float(sin(ang));
                const float cosa = float(cos(ang));
                for (int32_t m = 1; m < max_search_length; m++) {
                    const int32_t yy = lround(y + m * sina);
                    const int32_t xx = lround(x + m * cosa);
                    if (yy<0 || yy >= height || xx<0 || xx >= width) {
                        break;
                    }
                    const auto& disp = *(disp_ptr + yy*width + xx);
                    if (disp != Invalid_Float) {
                        disp_collects.push_back(disp);
                        break;
                    }
                }
            }
            if(disp_collects.empty()) {
                continue;
            }

            std::sort(disp_collects.begin(), disp_collects.end());

            // 如果是第一次循环，选择第二小的视差值（遮挡区策略）
            // 如果是第二次或第三次循环，选择中值（误匹配区策略）
            if (k == 0) {
                if (disp_collects.size() > 1) {
                    fill_disps[n] = disp_collects[1];
                }
                else {
                    fill_disps[n] = disp_collects[0];
                }
            }
            else{
                fill_disps[n] = disp_collects[disp_collects.size() / 2];
            }
        }
        
        // 应用填充结果
        for (auto n = 0u; n < trg_pixels.size(); n++) {
            auto& pix = trg_pixels[n];
            const int32_t y = pix.first;
            const int32_t x = pix.second;
            disp_ptr[y * width + x] = fill_disps[n];
        }
    }
}

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
    cout << "ELAS Standalone Program with SGM-style Fill Holes" << endl;
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
    
    // Use the same paths as testdisparityBM.cpp
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
    
    // Allocate disparity image memory dimensions
    int32_t dims[3];
    dims[0] = width;  // bytes per line = width
    dims[1] = height; // height
    dims[2] = width;  // bytes per line = width
    
    // 定义不同的视差范围组合进行测试
    struct DisparityRange {
        int32_t min_disp;
        int32_t max_disp;
        std::string description;
    };
    
    // 读取camera文件中的vmin和vmax作为视差范围
    std::string cameraFilePath = "../data/camera/" + datasetName + ".txt";
    int32_t vmin = 0, vmax = 256;  // 默认值
    
    std::ifstream cameraFile(cameraFilePath);
    if (cameraFile.is_open()) {
        std::string line;
        while (std::getline(cameraFile, line)) {
            if (line.find("vmin=") == 0) {
                vmin = std::stoi(line.substr(5));
            } else if (line.find("vmax=") == 0) {
                vmax = std::stoi(line.substr(5));
            }
        }
        cameraFile.close();
        cout << "Read disparity range from camera file: vmin=" << vmin << ", vmax=" << vmax << endl;
    } else {
        cout << "Warning: Could not open camera file: " << cameraFilePath << endl;
        cout << "Using default disparity range: vmin=" << vmin << ", vmax=" << vmax << endl;
    }
    
    // 使用camera文件中的vmin/vmax和原始设置作为视差范围组合
    std::vector<DisparityRange> disparity_ranges = {
        {0, 256, "original"},      // 原始设置作为对照
        {vmin, vmax, "camera_file"}, // 从camera文件读取的vmin/vmax
        // {40, 150, "large_range"},
        // {20, 150, "far_objects"},
        // {0, 64, "extended_range"}
    };
    
    // 定义不同的填充测试配置
    struct FillConfig {
        bool enable_fill;
        std::string description;
    };
    
    std::vector<FillConfig> fill_configs = {
        {false, "no_fill"},
        {true, "sgm_fill"}
    };
    
    cout << "Testing " << disparity_ranges.size() << " disparity ranges with " 
         << fill_configs.size() << " fill configurations..." << endl;
    cout << "Total tests: " << (disparity_ranges.size() * fill_configs.size()) << endl;
    cout << "Note: Including original [0, 256] range as baseline and camera file [vmin, vmax] range" << endl;
    cout << "===============================================" << endl;
    
    // 为每个视差范围和填充配置组合进行测试
    for (size_t disp_idx = 0; disp_idx < disparity_ranges.size(); ++disp_idx) {
        const auto& disp_range = disparity_ranges[disp_idx];
        
        for (size_t fill_idx = 0; fill_idx < fill_configs.size(); ++fill_idx) {
            const auto& fill_config = fill_configs[fill_idx];
            
            int test_num = disp_idx * fill_configs.size() + fill_idx + 1;
            int total_tests = disparity_ranges.size() * fill_configs.size();
            
            cout << "\n=== Test " << test_num << "/" << total_tests << " ===" << endl;
            cout << "Disparity range: [" << disp_range.min_disp << ", " << disp_range.max_disp 
                 << "] - " << disp_range.description << endl;
            cout << "Fill configuration: " << fill_config.description 
                 << " (fill: " << (fill_config.enable_fill ? "ON" : "OFF") << ")" << endl;
            cout << "===============================================" << endl;
            
            // 重新分配内存用于当前测试
            float *D1_test = (float*)malloc(width * height * sizeof(float));
            float *D2_test = (float*)malloc(width * height * sizeof(float));
            
            // Set ELAS parameters for current test
            Elas::parameters param;
            
            // ========== ELAS Algorithm Parameter Configuration ==========
            cout << "Configuring ELAS parameters for current test..." << endl;
            
            // ELAS disparity range parameters - 使用当前测试的视差范围
            param.disp_min = disp_range.min_disp;
            param.disp_max = disp_range.max_disp;
            
            // ========== ELAS Algorithm Configuration ==========
            // ELAS算法核心特性：
            // 1. 通过匹配可靠的支持点计算先验视差图
            // 2. 在支持点之间进行三角剖分
            // 3. 使用最大后验方法优化视差
            // 4. 检查左右图像的视差一致性
            // 5. 移除小于50像素的视差段
            // 6. 改进的彩色到灰度转换（2015年9月14日修复）
            // ========== Middlebury Benchmark Default Settings ==========
            // default settings for middlebury benchmark
            // (interpolate all missing disparities)
            param.support_threshold     = 0.95;
            param.support_texture       = 10;
            param.candidate_stepsize    = 5;
            param.incon_window_size     = 5;
            param.incon_threshold       = 5;
            param.incon_min_support     = 5;
            param.add_corners           = 1;
            param.grid_size             = 20;
            param.beta                  = 0.02;
            param.gamma                 = 5;
            param.sigma                 = 1;
            param.sradius               = 3;
            param.match_texture         = 0;
            param.lr_threshold          = 2;
            param.speckle_sim_threshold = 1;
            param.speckle_size          = 200;      // 移除小于50像素的视差段，符合ELAS描述要求
            param.ipol_gap_width        = 5000;
            param.filter_median         = 1;
            param.filter_adaptive_mean  = 0;
            param.postprocess_only_left = 0;
            param.subsampling           = 0;
            
            cout << "ELAS parameter configuration for current test (Middlebury Benchmark Settings):" << endl;
            cout << "  Disparity range: " << param.disp_min << " - " << param.disp_max << endl;
            cout << "  Support threshold: " << param.support_threshold << endl;
            cout << "  Support texture: " << param.support_texture << endl;
            cout << "  Candidate stepsize: " << param.candidate_stepsize << endl;
            cout << "  Inconsistent window size: " << param.incon_window_size << endl;
            cout << "  Inconsistent threshold: " << param.incon_threshold << endl;
            cout << "  Inconsistent min support: " << param.incon_min_support << endl;
            cout << "  Add corners: " << (param.add_corners ? "Yes" : "No") << endl;
            cout << "  Grid size: " << param.grid_size << endl;
            cout << "  Beta: " << param.beta << endl;
            cout << "  Gamma: " << param.gamma << endl;
            cout << "  Sigma: " << param.sigma << endl;
            cout << "  Sradius: " << param.sradius << endl;
            cout << "  Match texture: " << param.match_texture << endl;
            cout << "  Left-right threshold: " << param.lr_threshold << endl;
            cout << "  Speckle similarity threshold: " << param.speckle_sim_threshold << endl;
            cout << "  Speckle size: " << param.speckle_size << endl;
            cout << "  Interpolation gap width: " << param.ipol_gap_width << endl;
            cout << "  Filter median: " << (param.filter_median ? "Yes" : "No") << endl;
            cout << "  Filter adaptive mean: " << (param.filter_adaptive_mean ? "Yes" : "No") << endl;
            cout << "  Postprocess only left: " << (param.postprocess_only_left ? "Yes" : "No") << endl;
            cout << "  Subsampling: " << (param.subsampling ? "Yes" : "No") << endl;
            cout << endl;
            
            Elas elas(param);
            
            cout << "Processing ELAS with current parameters..." << endl;
            
            // 处理
            elas.process(I1->data, I2->data, D1_test, D2_test, dims);
            
            cout << "ELAS processing completed!" << endl;
            
            if (fill_config.enable_fill) {
                cout << "Applying SGM-style hole filling..." << endl;
                
                // 将ELAS的无效视差值转换为SGM的无效值格式
                for (int i = 0; i < width * height; i++) {
                    if (D1_test[i] < 0) {
                        D1_test[i] = Invalid_Float;
                    }
                    if (D2_test[i] < 0) {
                        D2_test[i] = Invalid_Float;
                    }
                }
                
                // 应用SGM风格的填充
                FillHolesInDispMapSGM(D1_test, width, height, param.disp_min, param.disp_max);
                FillHolesInDispMapSGM(D2_test, width, height, param.disp_min, param.disp_max);
                
                cout << "SGM-style hole filling completed!" << endl;
            } else {
                cout << "SGM-style hole filling disabled." << endl;
            }
            
            // 生成输出文件名
            std::string outputPathLeft = outputDir + "/disparity_ELAS_left_" + disp_range.description + "_" + fill_config.description + ".png";
            std::string outputPathRight = outputDir + "/disparity_ELAS_right_" + disp_range.description + "_" + fill_config.description + ".png";
            
            // Save left disparity map
            savePNG_float(outputPathLeft.c_str(), D1_test, width, height);
            
            // Save right disparity map
            savePNG_float(outputPathRight.c_str(), D2_test, width, height);
            
            cout << "Results saved to:" << endl;
            cout << "  " << outputPathLeft << " (Left disparity map - PNG format)" << endl;
            cout << "  " << outputPathRight << " (Right disparity map - PNG format)" << endl;
            
            // 统计有效像素数量
            int valid_pixels_left = 0, valid_pixels_right = 0;
            for (int i = 0; i < width * height; i++) {
                if (D1_test[i] > 0) valid_pixels_left++;
                if (D2_test[i] > 0) valid_pixels_right++;
            }
            
            cout << "Valid pixels - Left: " << valid_pixels_left << "/" << (width * height) 
                 << " (" << (100.0 * valid_pixels_left / (width * height)) << "%)" << endl;
            cout << "Valid pixels - Right: " << valid_pixels_right << "/" << (width * height)
                 << " (" << (100.0 * valid_pixels_right / (width * height)) << "%)" << endl;
            
            // 释放测试内存
            free(D1_test);
            free(D2_test);
            
            cout << "Test " << test_num << " completed!" << endl;
        }
    }
    
    cout << "\n=== Processing Summary ===" << endl;
    cout << "Total disparity ranges tested: " << disparity_ranges.size() << endl;
    cout << "Total fill configurations tested: " << fill_configs.size() << endl;
    cout << "Total tests performed: " << (disparity_ranges.size() * fill_configs.size()) << endl;
    
    cout << "\nTested disparity ranges:" << endl;
    for (const auto& range : disparity_ranges) {
        cout << "  - [" << range.min_disp << ", " << range.max_disp << "] - " << range.description << endl;
    }
    
    cout << "\nTested fill configurations:" << endl;
    for (const auto& config : fill_configs) {
        cout << "  - " << config.description << " (fill: " << (config.enable_fill ? "ON" : "OFF") << ")" << endl;
    }
    
    cout << "\nGenerated files for each combination:" << endl;
    cout << "  - disparity_ELAS_[left/right]_[disparity_range]_[fill_config].png" << endl;
    
    cout << "\nNote: ELAS algorithm generated two disparity maps (left and right disparity maps)" << endl;
    cout << "Each combination of disparity range and fill configuration has been tested" << endl;
    cout << "All results have been saved in directory: " << outputDir << endl;
    
    // Clean up memory
    delete I1;
    delete I2;
    
    cout << "Program execution completed!" << endl;
    cout << "All results saved in directory: " << outputDir << endl;
    return 0;
} 