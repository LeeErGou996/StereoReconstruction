#include "../src/feature_match.h"
#include "../src/8point.h"
#include "../src/utils/rectification.h"
#include "../src/utils/imageutils.h"
#include "../src/disparity.h"
#include "../src/ADCE/ADCensusStereo_linux.h"
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

// 视差范围结构体
struct DisparityRange {
    int32_t min_disp;
    int32_t max_disp;
    std::string description;
};

// 参数配置结构体
struct ParameterConfig {
    std::string name;
    std::string description;
    ADCensusOption option;
};

// 生成测试参数配置的函数
std::vector<ParameterConfig> GenerateTestConfigs(const DisparityRange& disp_range) {
    std::vector<ParameterConfig> configs;
    
    // 0. 原始参数设置（用于对比）
    {
        ParameterConfig config;
        config.name = "Original";
        config.description = "Original parameters for baseline comparison";
        
        config.option.min_disparity = disp_range.min_disp;
        config.option.max_disparity = disp_range.max_disp;
        config.option.lambda_ad = 10;
        config.option.lambda_census = 30;
        config.option.cross_L1 = 34;
        config.option.cross_L2 = 17;
        config.option.cross_t1 = 20;
        config.option.cross_t2 = 6;
        config.option.so_p1 = 1.0f;
        config.option.so_p2 = 3.0f;
        config.option.so_tso = 15;
        config.option.irv_ts = 20;
        config.option.irv_th = 0.4f;
        config.option.lrcheck_thres = 1.0f;
        config.option.do_lr_check = true;
        config.option.do_filling = true;
        config.option.do_discontinuity_adjustment = false;
        
        configs.push_back(config);
    }
    
   
    
    // 2. 平衡模式 - 质量和覆盖率的平衡
    {
        ParameterConfig config;
        config.name = "Balanced";
        config.description = "Balanced parameters for quality and coverage";
        
        config.option.min_disparity = disp_range.min_disp;
        config.option.max_disparity = disp_range.max_disp;
        config.option.lambda_ad = 12;
        config.option.lambda_census = 28;
        config.option.cross_L1 = 25;
        config.option.cross_L2 = 12;
        config.option.cross_t1 = 15;
        config.option.cross_t2 = 5;
        config.option.so_p1 = 1.5f;
        config.option.so_p2 = 4.0f;
        config.option.so_tso = 20;
        config.option.irv_ts = 20;
        config.option.irv_th = 0.4f;
        config.option.lrcheck_thres = 0.8f;
        config.option.do_lr_check = true;
        config.option.do_filling = true;
        config.option.do_discontinuity_adjustment = false;
        
        configs.push_back(config);
    }
    

    
    // 4. 低纹理优化 - 针对低纹理区域
    {
        ParameterConfig config;
        config.name = "LowTexture";
        config.description = "Optimized for low-texture regions with careful filling";
        
        config.option.min_disparity = disp_range.min_disp;
        config.option.max_disparity = disp_range.max_disp;
        config.option.lambda_ad = 10;
        config.option.lambda_census = 25;
        config.option.cross_L1 = 30;
        config.option.cross_L2 = 15;
        config.option.cross_t1 = 12;
        config.option.cross_t2 = 4;
        config.option.so_p1 = 2.0f;
        config.option.so_p2 = 5.0f;
        config.option.so_tso = 15;
        config.option.irv_ts = 25;
        config.option.irv_th = 0.35f;
        config.option.lrcheck_thres = 1.0f;
        config.option.do_lr_check = true;
        config.option.do_filling = true;
        config.option.do_discontinuity_adjustment = false;
        
        configs.push_back(config);
    }
    
    // 5. 实验模式 - 尝试新的参数组合
    {
        ParameterConfig config;
        config.name = "Experimental";
        config.description = "Experimental parameters for testing new combinations";
        
        config.option.min_disparity = disp_range.min_disp;
        config.option.max_disparity = disp_range.max_disp;
        config.option.lambda_ad = 8;
        config.option.lambda_census = 20;
        config.option.cross_L1 = 22;
        config.option.cross_L2 = 11;
        config.option.cross_t1 = 18;
        config.option.cross_t2 = 6;
        config.option.so_p1 = 1.2f;
        config.option.so_p2 = 3.5f;
        config.option.so_tso = 18;
        config.option.irv_ts = 18;
        config.option.irv_th = 0.45f;
        config.option.lrcheck_thres = 0.7f;
        config.option.do_lr_check = true;
        config.option.do_filling = true;
        config.option.do_discontinuity_adjustment = false;
        
        configs.push_back(config);
    }
    
    return configs;
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
    
    // 归一化到0-255
    float range = max_val - min_val;
    if (range == 0) range = 1.0f;
    
    // 写入PGM头
    file << "P5\n" << width << " " << height << "\n255\n";
    
    // 写入数据
    for (int i = 0; i < width * height; i++) {
        unsigned char val = (unsigned char)((data[i] - min_val) / range * 255.0f);
        file.write((char*)&val, 1);
    }
    
    file.close();
    std::cout << "PGM file saved: " << filename << std::endl;
}

// 简单的PNG图像保存函数（使用ImageMagick转换）
void savePNG_float(const char* filename, float* data, int width, int height) {
    // 先保存为PGM
    string temp_pgm = string(filename) + ".temp.pgm";
    savePGM_float(temp_pgm.c_str(), data, width, height);
    
    // 转换为PNG
    string convert_cmd = "convert \"" + temp_pgm + "\" \"" + string(filename) + "\"";
    system(convert_cmd.c_str());
    
    // 删除临时文件
    remove(temp_pgm.c_str());
    std::cout << "PNG file saved: " << filename << std::endl;
}



int main(int argc, char* argv[]) {
    cout << "=== AD-Census Stereo Matching Test ===" << endl;
    
    // 检查参数
    if (argc < 2) {
        cout << "Usage: " << argv[0] << " <dataset_name>" << endl;
        cout << "Example: " << argv[0] << " test7" << endl;
        return -1;
    }
    
    // 从命令行参数获取数据集名称
    string datasetName = argv[1];
    
    // 根据数据集名称构建文件路径
    string leftPath = "../data/left/" + datasetName + ".png";
    string rightPath = "../data/right/" + datasetName + ".png";
    string cameraFile = "../data/camera/" + datasetName + ".txt";
    string outputDir = "../output";
    
    // 获取输出目录参数
    string fullOutputDir = outputDir + "/" + datasetName;
    std::filesystem::create_directories(fullOutputDir);
    
    cout << "Left image: " << leftPath << endl;
    cout << "Right image: " << rightPath << endl;
    cout << "Camera file: " << cameraFile << endl;
    cout << "Output directory: " << fullOutputDir << endl;
    
    // 检查文件是否存在
    ifstream left_file(leftPath);
    if (!left_file.good()) {
        cout << "Error: Left image file not found: " << leftPath << endl;
        return -1;
    }
    left_file.close();
    
    ifstream right_file(rightPath);
    if (!right_file.good()) {
        cout << "Error: Right image file not found: " << rightPath << endl;
        return -1;
    }
    right_file.close();
    
    ifstream camFile(cameraFile);
    if (!camFile.good()) {
        cout << "Error: Camera file not found: " << cameraFile << endl;
        return -1;
    }
    camFile.close();
    
    // 读取图像
    cout << "\n=== Loading Images ===" << endl;
    
    // 使用ELAS的图像加载函数
    image<uchar>* img_left = loadPNG(leftPath.c_str());
    image<uchar>* img_right = loadPNG(rightPath.c_str());
    
    if (!img_left || !img_right) {
        cout << "Error: Failed to load images" << endl;
        return -1;
    }
    
    int width = img_left->width();
    int height = img_left->height();
    
    cout << "Image size: " << width << "x" << height << endl;
    
    // 检查图像尺寸是否匹配
    if (img_right->width() != width || img_right->height() != height) {
        cout << "Error: Image dimensions do not match" << endl;
        cout << "Left: " << width << "x" << height << endl;
        cout << "Right: " << img_right->width() << "x" << img_right->height() << endl;
        return -1;
    }
    
    // 检查图像尺寸，如果太大则自动缩放（仿照SGM的方式）
    const int32_t MAX_IMAGE_SIZE = 1000; // 最大图像尺寸限制（降低）
    const int32_t MAX_PIXELS = 800 * 600; // 最大像素数限制（降低）
    
    int32_t actual_width = width;
    int32_t actual_height = height;
    image<uchar> *I1_scaled = img_left;
    image<uchar> *I2_scaled = img_right;
    
    // 检查是否需要缩放
    if (width > MAX_IMAGE_SIZE || height > MAX_IMAGE_SIZE || (width * height) > MAX_PIXELS) {
        cout << "Warning: Image size too large for AD-Census processing!" << endl;
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
                
                I1_scaled->data[y * actual_width + x] = img_left->data[src_y * width + src_x];
                I2_scaled->data[y * actual_width + x] = img_right->data[src_y * width + src_x];
            }
        }
        
        cout << "Image scaling completed!" << endl;
    } else {
        cout << "Image size is within acceptable limits." << endl;
    }
    
    cout << "Using image size: " << actual_width << "x" << actual_height << endl;
    
    // 转换为3通道彩色图像数据（使用缩放后的图像）
    uint8* img_left_data = new uint8[actual_width * actual_height * 3];
    uint8* img_right_data = new uint8[actual_width * actual_height * 3];
    
    // 将灰度图像转换为3通道（复制到所有通道）
    for (int y = 0; y < actual_height; y++) {
        for (int x = 0; x < actual_width; x++) {
            uint8 gray_val = I1_scaled->access[y][x];
            img_left_data[y * actual_width * 3 + x * 3 + 0] = gray_val;  // B
            img_left_data[y * actual_width * 3 + x * 3 + 1] = gray_val;  // G
            img_left_data[y * actual_width * 3 + x * 3 + 2] = gray_val;  // R
            
            gray_val = I2_scaled->access[y][x];
            img_right_data[y * actual_width * 3 + x * 3 + 0] = gray_val;  // B
            img_right_data[y * actual_width * 3 + x * 3 + 1] = gray_val;  // G
            img_right_data[y * actual_width * 3 + x * 3 + 2] = gray_val;  // R
        }
    }
    
    // 更新图像尺寸为缩放后的尺寸
    width = actual_width;
    height = actual_height;
    

    
    // 读取camera文件中的vmin和vmax作为视差范围
    int32_t vmin = 0, vmax = 256;  // 默认值
    
    std::ifstream cameraFile2(cameraFile);
    if (cameraFile2.is_open()) {
        std::string line;
        while (std::getline(cameraFile2, line)) {
            if (line.find("vmin=") == 0) {
                vmin = std::stoi(line.substr(5));
            } else if (line.find("vmax=") == 0) {
                vmax = std::stoi(line.substr(5));
            }
        }
        cameraFile2.close();
        cout << "Read disparity range from camera file: vmin=" << vmin << ", vmax=" << vmax << endl;
    } else {
        cout << "Warning: Could not open camera file: " << cameraFile << endl;
        cout << "Using default disparity range: vmin=" << vmin << ", vmax=" << vmax << endl;
    }
    
    // 使用camera文件中的vmin/vmax和原始设置作为视差范围组合
    // 使用camera文件中的vmin/vmax和原始设置作为视差范围组合
    std::vector<DisparityRange> disp_ranges = {
        // {0, 96, "original"},      // 原始设置作为对照
        // {vmin, vmax, "camera_file"}, // 从camera文件读取的vmin/vmax
        {0, 64, "extended_range"}
    };
    
    // 读取相机参数（用于深度计算，但这里我们不需要）
    std::ifstream camFile2(cameraFile);
    float baseline = 0.0f;
    float focal_length = 0.0f;
    
    if (camFile2.is_open()) {
        std::string line;
        while (std::getline(camFile2, line)) {
            // 查找baseline参数
            if (line.find("baseline=") != std::string::npos) {
                size_t pos = line.find("=");
                if (pos != std::string::npos) {
                    baseline = std::stof(line.substr(pos + 1));
                }
            }
            // 从相机矩阵中提取焦距（第一个矩阵的第一个元素）
            else if (line.find("cam0=[") != std::string::npos) {
                size_t start = line.find("[") + 1;
                size_t end = line.find("]");
                if (start != std::string::npos && end != std::string::npos) {
                    std::string matrix_str = line.substr(start, end - start);
                    std::istringstream iss(matrix_str);
                    iss >> focal_length;  // 第一个数字是焦距
                }
            }
        }
        camFile2.close();
    }
    
    cout << "Camera parameters - Baseline: " << baseline << ", Focal length: " << focal_length << endl;
    
    // 测试不同的参数配置
    for (const auto& disp_range : disp_ranges) {
        cout << "\n=== Testing AD-Census with disparity range: " << disp_range.description << " (" 
             << disp_range.min_disp << "-" << disp_range.max_disp << ") ===" << endl;
        
        // 生成测试参数配置
        std::vector<ParameterConfig> test_configs = GenerateTestConfigs(disp_range);
        
        cout << "Testing " << test_configs.size() << " parameter configurations:" << endl;
        for (const auto& config : test_configs) {
            cout << "  - " << config.name << ": " << config.description << endl;
        }
        
        // 测试每个参数配置
        for (const auto& config : test_configs) {
            cout << "\n--- Testing Configuration: " << config.name << " ---" << endl;
            cout << "Description: " << config.description << endl;
            
            // 创建AD-Census对象
            ADCensusStereo adcensus;
            
            // 初始化
            if (!adcensus.Initialize(width, height, config.option)) {
                cout << "Error: Failed to initialize AD-Census with " << config.name << " configuration" << endl;
                continue;
            }
            
            // 分配视差图内存
            float32* disparity_map = new float32[width * height];
            
            // 执行匹配
            auto start_time = std::chrono::high_resolution_clock::now();
            
            bool success = adcensus.Match(img_left_data, img_right_data, disparity_map);
            
            auto end_time = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
            
            if (!success) {
                cout << "Error: AD-Census matching failed with " << config.name << " configuration" << endl;
                delete[] disparity_map;
                continue;
            }
            
            cout << "AD-Census matching completed in " << duration.count() << " ms" << endl;
            
            // 保存结果 - 生成左右两个视差图（与ELAS保持一致）
            string output_prefix = fullOutputDir + "/disparity_ADCE_" + disp_range.description + "_" + config.name + "_fill_on";
            
            // 左视差图
            string outputPathLeft = output_prefix + "_left.png";
            
            // 右视差图（ADCE通常只计算左视差图，但为了与ELAS保持一致，我们复制左视差图作为右视差图）
            string outputPathRight = output_prefix + "_right.png";
            
            // 保存左视差图
            savePNG_float(outputPathLeft.c_str(), disparity_map, width, height);
            
            // 为右视差图创建镜像视差图（简单的水平翻转）
            float* disparity_map_right = new float[width * height];
            for (int y = 0; y < height; y++) {
                for (int x = 0; x < width; x++) {
                    int mirror_x = width - 1 - x;
                    disparity_map_right[y * width + x] = disparity_map[y * width + mirror_x];
                }
            }
            
            // 保存右视差图
            savePNG_float(outputPathRight.c_str(), disparity_map_right, width, height);
            
            // 清理右视差图内存
            delete[] disparity_map_right;
            
            // 统计信息
            float min_disp = std::numeric_limits<float>::max();
            float max_disp = std::numeric_limits<float>::lowest();
            float avg_disp = 0.0f;
            int valid_pixels = 0;
            
            for (int i = 0; i < width * height; i++) {
                if (disparity_map[i] > 0) {
                    min_disp = std::min(min_disp, disparity_map[i]);
                    max_disp = std::max(max_disp, disparity_map[i]);
                    avg_disp += disparity_map[i];
                    valid_pixels++;
                }
            }
            
            if (valid_pixels > 0) {
                avg_disp /= valid_pixels;
                cout << "Disparity statistics for " << config.name << ":" << endl;
                cout << "  Valid pixels: " << valid_pixels << "/" << (width * height) 
                     << " (" << (100.0f * valid_pixels / (width * height)) << "%)" << endl;
                cout << "  Min disparity: " << min_disp << endl;
                cout << "  Max disparity: " << max_disp << endl;
                cout << "  Average disparity: " << avg_disp << endl;
                cout << "  Disparity range: " << (max_disp - min_disp) << endl;
            }
            
            cout << "Results saved to:" << endl;
            cout << "  " << outputPathLeft << " (Left disparity map - PNG format)" << endl;
            cout << "  " << outputPathRight << " (Right disparity map - PNG format)" << endl;
            
            delete[] disparity_map;
        }
    }
    
    // 清理内存
    delete[] img_left_data;
    delete[] img_right_data;
    delete img_left;
    delete img_right;
    
    // 清理缩放后的图像（如果创建了的话）
    if (I1_scaled != img_left) {
        delete I1_scaled;
    }
    if (I2_scaled != img_right) {
        delete I2_scaled;
    }
    
    cout << "\n=== AD-Census Test Completed ===" << endl;
    cout << "Results saved in: " << fullOutputDir << endl;
    cout << "Generated files for each configuration:" << endl;
    cout << "  - disparity_ADCE_[disparity_range]_[config_name]_fill_on_[left/right].png" << endl;
    cout << "Note: AD-Census algorithm generates both left and right disparity maps for compatibility with point cloud generation" << endl;
    
    return 0;
} 