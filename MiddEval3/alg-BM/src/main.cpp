#include "imageutils.h"
#include <iostream>
#include <vector>
#include <thread>
#include <climits>
#include <cmath>
#include <algorithm>
#include <string>
#include <cassert>
#include <fstream>
#include <sstream>
#include <chrono>
#include <filesystem>

using namespace std;

// Simple x-derivative prefilter for MyImage (grayscale, 1 channel)
void xDerivativePrefilter(const MyImage& input, MyImage& output, int preFilterCap = 30) {
    const int offset = 256 * 4;
    unsigned char clampTable[offset * 2 + 256];
    for (int i = 0; i < offset * 2 + 256; i++) {
        int val = i - offset;
        if (val < -preFilterCap)
            clampTable[i] = 0;
        else if (val > preFilterCap)
            clampTable[i] = 2 * preFilterCap;
        else
            clampTable[i] = static_cast<unsigned char>(val + preFilterCap);
    }
    unsigned char defaultValue = clampTable[offset];
    output.width = input.width;
    output.height = input.height;
    output.channels = 1;
    output.data.resize(input.width * input.height);
    for (int y = 0; y < input.height - 1; y += 2) {
        const unsigned char* rowAbove = (y > 0) ? &input.data[(y - 1) * input.width] : &input.data[(y + 1) * input.width];
        const unsigned char* rowCurr = &input.data[y * input.width];
        const unsigned char* rowNext = &input.data[(y + 1) * input.width];
        const unsigned char* rowBelow = (y < input.height - 2) ? &input.data[(y + 2) * input.width] : rowCurr;
        unsigned char* dstRow0 = &output.data[y * input.width];
        unsigned char* dstRow1 = &output.data[(y + 1) * input.width];
        dstRow0[0] = dstRow0[input.width - 1] = defaultValue;
        dstRow1[0] = dstRow1[input.width - 1] = defaultValue;
        for (int x = 1; x < input.width - 1; x++) {
            int gradTop =    rowAbove[x + 1] - rowAbove[x - 1]
                           + 2 * (rowCurr[x + 1] - rowCurr[x - 1])
                           + rowNext[x + 1] - rowNext[x - 1];
            int gradBottom = rowCurr[x + 1] - rowCurr[x - 1]
                           + 2 * (rowNext[x + 1] - rowNext[x - 1])
                           + rowBelow[x + 1] - rowBelow[x - 1];
            dstRow0[x] = clampTable[gradTop + offset];
            dstRow1[x] = clampTable[gradBottom + offset];
        }
    }
    for (int y = (input.height / 2) * 2; y < input.height; y++) {
        std::fill(&output.data[y * input.width], &output.data[(y + 1) * input.width], defaultValue);
    }
}

void StereoBMPreFilterThreads(const MyImage& rectL, const MyImage& rectR, MyImage& disparity, int maxDisparity) {
    const int SADWindowSize = 32;
    const int preFilterCap = 30;
    const int minDisparity = 0;  // 从0开始，而不是固定的60
    const int uniquenessThreshold = 20;
    const int N_THREADS = std::thread::hardware_concurrency();
    
    MyImage leftPref, rightPref;
    xDerivativePrefilter(rectL, leftPref, preFilterCap);
    xDerivativePrefilter(rectR, rightPref, preFilterCap);
    
    disparity.width = rectL.width;
    disparity.height = rectL.height;
    disparity.channels = 1;
    disparity.data.resize(rectL.width * rectL.height, 0); // 0 means invalid
    
    const int rows = rectL.height, cols = rectL.width;
    const int win = SADWindowSize / 2;
    
    auto block_matching_function = [&](int threadID) {
        for (int y = win + threadID; y < rows - win; y += N_THREADS) {
            if (y % 10 == 0)
                std::cout << "Running: " << y << "/" << rows - win << std::endl;
            for (int x = win + maxDisparity; x < cols - win; x++) {
                int minSAD = INT_MAX, bestDisp = 0;
                int nextBestSAD[3] = {INT_MAX, INT_MAX, INT_MAX};
                for (int d = minDisparity; d < maxDisparity; d++) {
                    int sad = 0;
                    for (int wy = -win; wy <= win; wy++) {
                        const unsigned char* lRow = &leftPref.data[(y + wy) * cols];
                        const unsigned char* rRow = &rightPref.data[(y + wy) * cols];
                        for (int wx = -win; wx <= win; wx++) {
                            int lVal = lRow[x + wx];
                            int rVal = rRow[x + wx - d];
                            sad += std::abs(lVal - rVal);
                        }
                    }
                    if (sad < minSAD) {
                        nextBestSAD[2] = nextBestSAD[1];
                        nextBestSAD[1] = nextBestSAD[0];
                        nextBestSAD[0] = minSAD;
                        minSAD = sad;
                        bestDisp = d;
                    }
                }
                int sadThresh = minSAD + minSAD * uniquenessThreshold / 100;
                if (nextBestSAD[2] > sadThresh)
                    disparity.data[y * cols + x] = static_cast<unsigned char>(bestDisp);
            }
        }
    };
    
    std::vector<std::thread> pool(N_THREADS);
    for (int i = 0; i < N_THREADS; i++) {
        pool[i] = std::thread(block_matching_function, i);
    }
    for (int i = 0; i < N_THREADS; i++) {
        pool[i].join();
    }
}

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
    for (int y = height - 1; y >= 0; y--) {
        for (int x = 0; x < width; x++) {
            int index = y * width + x;
            fwrite(&data[index], sizeof(float), 1, file);
        }
    }
    
    fclose(file);
}

// 将BM视差图转换为PFM格式
void convertBMToPFM(unsigned char* bm_disp, float* pfm_disp, int width, int height) {
    for (int i = 0; i < width * height; i++) {
        if (bm_disp[i] == 0) {
            pfm_disp[i] = INFINITY;  // 无效值用INFINITY表示
        } else {
            pfm_disp[i] = static_cast<float>(bm_disp[i]);
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
    
    cout << "BM Algorithm for MiddEval3" << endl;
    cout << "==========================" << endl;
    cout << "Left image: " << left_path << endl;
    cout << "Right image: " << right_path << endl;
    cout << "Output: " << output_path << endl;
    cout << "Max disparity: " << maxdisp << endl;
    cout << "==========================" << endl;
    
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
    MyImage imgL, imgR;
    
    try {
        // 读取PGM图像
        if (!myImReadPGM(left_path.c_str(), imgL)) {
            cerr << "Error: Could not read left PGM image!" << endl;
            return 1;
        }
        
        if (!myImReadPGM(right_path.c_str(), imgR)) {
            cerr << "Error: Could not read right PGM image!" << endl;
            return 1;
        }
    } catch (const std::exception& e) {
        cerr << "Error: Exception occurred while reading images: " << e.what() << endl;
        return 1;
    }
    
    // 检查图像尺寸
    if (imgL.width != imgR.width || imgL.height != imgR.height) {
        cerr << "Error: Images must have same size!" << endl;
        cerr << "Left image size: " << imgL.width << "x" << imgL.height << endl;
        cerr << "Right image size: " << imgR.width << "x" << imgR.height << endl;
        return 1;
    }
    
    // 获取图像尺寸
    const int32_t width = imgL.width;
    const int32_t height = imgL.height;
    
    cout << "Processing images of size: " << width << "x" << height << endl;
    
    // 检查图像尺寸，如果太大则自动缩放
    const int32_t MAX_IMAGE_SIZE = 1024; // BM算法相对较快，可以处理更大的图像
    const int32_t MAX_PIXELS = 1024 * 768; // 最大像素数限制
    
    int32_t actual_width = width;
    int32_t actual_height = height;
    MyImage imgL_scaled = imgL;
    MyImage imgR_scaled = imgR;
    bool need_scaling = false;
    
    // 检查是否需要缩放
    if (width > MAX_IMAGE_SIZE || height > MAX_IMAGE_SIZE || (width * height) > MAX_PIXELS) {
        cout << "Warning: Image size too large for BM processing!" << endl;
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
        imgL_scaled.width = actual_width;
        imgL_scaled.height = actual_height;
        imgL_scaled.channels = 1;
        imgL_scaled.data.resize(actual_width * actual_height);
        
        imgR_scaled.width = actual_width;
        imgR_scaled.height = actual_height;
        imgR_scaled.channels = 1;
        imgR_scaled.data.resize(actual_width * actual_height);
        
        need_scaling = true;
        
        // 简单的最近邻缩放
        for (int32_t y = 0; y < actual_height; y++) {
            for (int32_t x = 0; x < actual_width; x++) {
                int32_t src_x = (int32_t)(x / scale);
                int32_t src_y = (int32_t)(y / scale);
                
                if (src_x >= width) src_x = width - 1;
                if (src_y >= height) src_y = height - 1;
                
                imgL_scaled.data[y * actual_width + x] = imgL.data[src_y * width + src_x];
                imgR_scaled.data[y * actual_width + x] = imgR.data[src_y * width + src_x];
            }
        }
        
        cout << "Image scaling completed!" << endl;
    } else {
        cout << "Image size is within acceptable limits." << endl;
    }
    
    cout << "Using image size: " << actual_width << "x" << actual_height << endl;
    
    // 确保图像是灰度图（1通道）
    if (imgL_scaled.channels != 1) {
        cerr << "Error: Left image must be grayscale (1 channel)!" << endl;
        return 1;
    }
    
    if (imgR_scaled.channels != 1) {
        cerr << "Error: Right image must be grayscale (1 channel)!" << endl;
        return 1;
    }
    
    // 分配视差图内存
    MyImage disparity;
    float* pfm_disp = nullptr;
    
    try {
        pfm_disp = new float[actual_width * actual_height];
        
        // 运行BM算法
        cout << "Running BM algorithm..." << endl;
        auto start = std::chrono::high_resolution_clock::now();
        
        StereoBMPreFilterThreads(imgL_scaled, imgR_scaled, disparity, maxdisp);
        
        auto end = std::chrono::high_resolution_clock::now();
        auto tt = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
        cout << "BM algorithm completed! Timing: " << (tt.count() / 1000.0) << "s" << endl;
        
        // 如果进行了缩放，需要将视差图缩放回原始尺寸
        MyImage final_disparity = disparity;
        if (need_scaling) {
            cout << "Scaling disparity map back to original size..." << endl;
            
            final_disparity.width = width;
            final_disparity.height = height;
            final_disparity.channels = 1;
            final_disparity.data.resize(width * height);
            
            float scale = (float)actual_width / width;
            
            for (int y = 0; y < height; y++) {
                for (int x = 0; x < width; x++) {
                    int scaled_x = (int)(x * scale);
                    int scaled_y = (int)(y * scale);
                    
                    if (scaled_x >= actual_width) scaled_x = actual_width - 1;
                    if (scaled_y >= actual_height) scaled_y = actual_height - 1;
                    
                    unsigned char disp_val = disparity.data[scaled_y * actual_width + scaled_x];
                    
                    // 缩放视差值
                    if (disp_val > 0) {
                        final_disparity.data[y * width + x] = (unsigned char)(disp_val / scale);
                    } else {
                        final_disparity.data[y * width + x] = 0;
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
        convertBMToPFM(final_disparity.data.data(), pfm_disp, final_width, final_height);
        
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
            if (final_disparity.data[i] > 0) {
                valid_pixels++;
            }
        }
        
        cout << "Valid pixels: " << valid_pixels << "/" << (final_width * final_height) 
             << " (" << (100.0 * valid_pixels / (final_width * final_height)) << "%)" << endl;
        
    } catch (const std::exception& e) {
        cerr << "Error during processing: " << e.what() << endl;
        
        // 清理内存
        if (pfm_disp) delete[] pfm_disp;
        return 1;
    }
    
    // 正常清理内存
    delete[] pfm_disp;
    
    cout << "BM completed successfully!" << endl;
    return 0;
} 