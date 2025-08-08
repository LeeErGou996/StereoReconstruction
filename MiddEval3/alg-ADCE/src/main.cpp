#include "ADCensusStereo_linux.h"
#include "image.h"
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

// 保存PFM格式视差图
void savePFM(const char* filename, float* data, int width, int height) {
    FILE* file = fopen(filename, "wb");
    if (!file) {
        cerr << "Error: Cannot open file " << filename << " for writing" << endl;
        return;
    }
    
    // PFM header
    fprintf(file, "Pf\n");
    fprintf(file, "%d %d\n", width, height);
    fprintf(file, "-1.0\n");  // little endian
    
    // Write data in bottom-up order (PFM format requirement)
    // PFM格式要求从底部到顶部写入数据，避免180度旋转
    for (int y = height - 1; y >= 0; y--) {
        for (int x = 0; x < width; x++) {
            int index = y * width + x;
            fwrite(&data[index], sizeof(float), 1, file);
        }
    }
    
    fclose(file);
}



// 将ADCE视差图转换为PFM格式
void convertADCEToPFM(float* adce_disp, float* pfm_disp, int width, int height) {
    for (int i = 0; i < width * height; i++) {
        if (adce_disp[i] <= 0) {
            pfm_disp[i] = INFINITY;  // 无效值用INFINITY表示
        } else {
            pfm_disp[i] = adce_disp[i];
        }
    }
}

int main(int argc, char* argv[]) {
    // MiddEval3接口：<im0.pgm> <im1.pgm> <output.pfm> <maxdisp>
    if (argc != 5) {
        cerr << "Usage: " << argv[0] << " <im0.pgm> <im1.pgm> <output.pfm> <maxdisp>" << endl;
        cerr << "Received " << argc << " arguments:" << endl;
        for (int i = 0; i < argc; i++) {
            cerr << "  argv[" << i << "] = " << argv[i] << endl;
        }
        return 1;
    }
    
    string left_path = argv[1];
    string right_path = argv[2];
    string output_path = argv[3];
    int maxdisp = atoi(argv[4]);
    
    cout << "ADCE Algorithm for MiddEval3" << endl;
    cout << "============================" << endl;
    cout << "Left image: " << left_path << endl;
    cout << "Right image: " << right_path << endl;
    cout << "Output: " << output_path << endl;
    cout << "Max disparity: " << maxdisp << endl;
    cout << "============================" << endl;
    
    // 验证输入参数
    if (maxdisp <= 0) {
        cerr << "Error: Invalid max disparity value: " << maxdisp << endl;
        return 1;
    }
    
    // 检查输入文件是否存在
    ifstream left_file(left_path);
    if (!left_file.good()) {
        cerr << "Error: Cannot open left image file: " << left_path << endl;
        return 1;
    }
    left_file.close();
    
    ifstream right_file(right_path);
    if (!right_file.good()) {
        cerr << "Error: Cannot open right image file: " << right_path << endl;
        return 1;
    }
    right_file.close();
    
    // 读取图像
    image<uchar> *I1 = nullptr;
    image<uchar> *I2 = nullptr;
    
    try {
        // 直接读取PGM图像
        I1 = loadPGM(left_path.c_str());
        I2 = loadPGM(right_path.c_str());
        
        if (I1 == nullptr || I2 == nullptr) {
            cerr << "Error: Could not read PGM images!" << endl;
            if (I1) delete I1;
            if (I2) delete I2;
            return 1;
        }
    } catch (const pnm_error& e) {
        cerr << "Error: Could not read images! PNM error occurred." << endl;
        if (I1) delete I1;
        if (I2) delete I2;
        return 1;
    } catch (const std::exception& e) {
        cerr << "Error: Exception occurred while reading images: " << e.what() << endl;
        if (I1) delete I1;
        if (I2) delete I2;
        return 1;
    }
    
    // 检查图像尺寸
    if (I1->width() != I2->width() || I1->height() != I2->height()) {
        cerr << "Error: Images must have same size!" << endl;
        cerr << "Left image size: " << I1->width() << "x" << I1->height() << endl;
        cerr << "Right image size: " << I2->width() << "x" << I2->height() << endl;
        delete I1;
        delete I2;
        return 1;
    }
    
    // 获取图像尺寸
    const int32_t width = I1->width();
    const int32_t height = I1->height();
    
    cout << "Processing images of size: " << width << "x" << height << endl;
    
    // 检查图像尺寸，如果太大则自动缩放
    const int32_t MAX_IMAGE_SIZE = 800; // ADCE计算量大，限制更严格
    const int32_t MAX_PIXELS = 640 * 480; // 最大像素数限制
    
    int32_t actual_width = width;
    int32_t actual_height = height;
    image<uchar> *I1_scaled = I1;
    image<uchar> *I2_scaled = I2;
    bool need_scaling = false;
    
    // 检查是否需要缩放
    if (width > MAX_IMAGE_SIZE || height > MAX_IMAGE_SIZE || (width * height) > MAX_PIXELS) {
        cout << "Warning: Image size too large for ADCE processing!" << endl;
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
        need_scaling = true;
        
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
    
    // 转换为3通道彩色图像数据（ADCE需要3通道输入）
    uint8* img_left_data = new uint8[actual_width * actual_height * 3];
    uint8* img_right_data = new uint8[actual_width * actual_height * 3];
    
    // 将灰度图像转换为3通道（复制到所有通道）
    for (int y = 0; y < actual_height; y++) {
        for (int x = 0; x < actual_width; x++) {
            uint8 gray_val = I1_scaled->data[y * actual_width + x];
            img_left_data[y * actual_width * 3 + x * 3 + 0] = gray_val;  // B
            img_left_data[y * actual_width * 3 + x * 3 + 1] = gray_val;  // G
            img_left_data[y * actual_width * 3 + x * 3 + 2] = gray_val;  // R
            
            gray_val = I2_scaled->data[y * actual_width + x];
            img_right_data[y * actual_width * 3 + x * 3 + 0] = gray_val;  // B
            img_right_data[y * actual_width * 3 + x * 3 + 1] = gray_val;  // G
            img_right_data[y * actual_width * 3 + x * 3 + 2] = gray_val;  // R
        }
    }
    
    // 设置ADCE参数（优化用于MiddEval3）
    ADCensusOption option;
    option.min_disparity = 0;
    option.max_disparity = maxdisp;
    option.lambda_ad = 10;
    option.lambda_census = 25;
    option.cross_L1 = 30;
    option.cross_L2 = 15;
    option.cross_t1 = 12;
    option.cross_t2 = 4;
    option.so_p1 = 2.0f;
    option.so_p2 = 5.0f;
    option.so_tso = 15;
    option.irv_ts = 25;
    option.irv_th = 0.35f;
    option.lrcheck_thres = 1.0f;
    option.do_lr_check = true;
    option.do_filling = true;
    option.do_discontinuity_adjustment = false;
    
    cout << "ADCE Parameters (Optimized for MiddEval3):" << endl;
    cout << "  Disparity range: [" << option.min_disparity << ", " << option.max_disparity << "]" << endl;
    cout << "  Lambda AD: " << option.lambda_ad << endl;
    cout << "  Lambda Census: " << option.lambda_census << endl;
    cout << "  Cross aggregation L1/L2: " << option.cross_L1 << "/" << option.cross_L2 << endl;
    cout << "  Cross aggregation t1/t2: " << option.cross_t1 << "/" << option.cross_t2 << endl;
    cout << "  Scanline optimization P1/P2: " << option.so_p1 << "/" << option.so_p2 << endl;
    cout << "  SO tso: " << option.so_tso << endl;
    cout << "  IRV ts/th: " << option.irv_ts << "/" << option.irv_th << endl;
    cout << "  LR check threshold: " << option.lrcheck_thres << endl;
    cout << "  Do LR check: " << (option.do_lr_check ? "Yes" : "No") << endl;
    cout << "  Do filling: " << (option.do_filling ? "Yes" : "No") << endl;
    cout << "  Do discontinuity adjustment: " << (option.do_discontinuity_adjustment ? "Yes" : "No") << endl;
    cout << endl;
    
    // 分配视差图内存
    float32* disparity_map = nullptr;
    float* pfm_disp = nullptr;
    
    try {
        disparity_map = new float32[actual_width * actual_height];
        pfm_disp = new float[actual_width * actual_height];
        
        // 创建ADCE对象
        ADCensusStereo adcensus;
        
        // 初始化
        cout << "Initializing ADCE..." << endl;
        auto start = std::chrono::high_resolution_clock::now();
        if (!adcensus.Initialize(actual_width, actual_height, option)) {
            throw std::runtime_error("ADCE initialization failed!");
        }
        auto end = std::chrono::high_resolution_clock::now();
        auto tt = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
        cout << "ADCE initialization completed! Timing: " << (tt.count() / 1000.0) << "s" << endl;
        
        // 匹配
        cout << "ADCE matching..." << endl;
        start = std::chrono::high_resolution_clock::now();
        if (!adcensus.Match(img_left_data, img_right_data, disparity_map)) {
            throw std::runtime_error("ADCE matching failed!");
        }
        end = std::chrono::high_resolution_clock::now();
        tt = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
        cout << "ADCE matching completed! Timing: " << (tt.count() / 1000.0) << "s" << endl;
        
        // 如果进行了缩放，需要将视差图缩放回原始尺寸
        float* final_disp = disparity_map;
        if (need_scaling) {
            cout << "Scaling disparity map back to original size..." << endl;
            final_disp = new float[width * height];
            
            float scale = (float)actual_width / width;
            
            for (int y = 0; y < height; y++) {
                for (int x = 0; x < width; x++) {
                    int scaled_x = (int)(x * scale);
                    int scaled_y = (int)(y * scale);
                    
                    if (scaled_x >= actual_width) scaled_x = actual_width - 1;
                    if (scaled_y >= actual_height) scaled_y = actual_height - 1;
                    
                    float disp_val = disparity_map[scaled_y * actual_width + scaled_x];
                    
                    // 缩放视差值
                    if (disp_val > 0) {
                        final_disp[y * width + x] = disp_val / scale;
                    } else {
                        final_disp[y * width + x] = 0.0f;
                    }
                }
            }
            
            cout << "Disparity map scaling completed!" << endl;
            
            // 使用原始图像尺寸重新分配pfm_disp
            delete[] pfm_disp;
            pfm_disp = new float[width * height];
        }
        
        // 转换为PFM格式
        int final_width = need_scaling ? width : actual_width;
        int final_height = need_scaling ? height : actual_height;
        convertADCEToPFM(final_disp, pfm_disp, final_width, final_height);
        
        // 保存PFM文件
        savePFM(output_path.c_str(), pfm_disp, final_width, final_height);
        
        // 验证输出文件是否创建成功
        ifstream output_file(output_path);
        if (!output_file.good()) {
            throw std::runtime_error("Output file creation failed");
        }
        output_file.close();
        
        cout << "Disparity map saved to: " << output_path << endl;
        
        // 统计有效像素
        int valid_pixels = 0;
        for (int i = 0; i < final_width * final_height; i++) {
            if (final_disp[i] > 0) {
                valid_pixels++;
            }
        }
        
        cout << "Valid pixels: " << valid_pixels << "/" << (final_width * final_height) 
             << " (" << (100.0 * valid_pixels / (final_width * final_height)) << "%)" << endl;
        
        // 清理缩放后的视差图
        if (need_scaling && final_disp != disparity_map) {
            delete[] final_disp;
        }
        
    } catch (const std::exception& e) {
        cerr << "Error during processing: " << e.what() << endl;
        
        // 清理内存
        if (disparity_map) delete[] disparity_map;
        if (pfm_disp) delete[] pfm_disp;
        delete[] img_left_data;
        delete[] img_right_data;
        
        // 清理缩放后的图像
        if (need_scaling) {
            delete I1_scaled;
            delete I2_scaled;
        }
        delete I1;
        delete I2;
        return 1;
    }
    
    // 正常清理内存
    delete[] disparity_map;
    delete[] pfm_disp;
    delete[] img_left_data;
    delete[] img_right_data;
    
    // 清理缩放后的图像
    if (need_scaling) {
        delete I1_scaled;
        delete I2_scaled;
    }
    
    // 注意：不要手动删除I1->data和I2->data！
    // image类的析构函数会自动处理数据内存
    delete I1;
    delete I2;
    
    cout << "ADCE completed successfully!" << endl;
    return 0;
}