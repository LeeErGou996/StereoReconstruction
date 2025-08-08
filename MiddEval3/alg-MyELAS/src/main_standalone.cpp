/*
Copyright 2011. All rights reserved.
Institute of Measurement and Control Systems
Karlsruhe Institute of Technology, Germany

This file is part of libelas.
Authors: Andreas Geiger

libelas is free software; you can redistribute it and/or modify it under the
terms of the GNU General Public License as published by the Free Software
Foundation; either version 3 of the License, or any later version.

libelas is distributed in the hope that it will be useful, but WITHOUT ANY
WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
PARTICULAR PURPOSE. See the GNU General Public License for more details.

You should have received a copy of the GNU General Public License along with
libelas; if not, write to the Free Software Foundation, Inc., 51 Franklin
Street, Fifth Floor, Boston, MA 02110-1301, USA
*/

// Standalone demo program showing how libelas can be used without OpenCV
// 直接使用指定的图像文件路径

#include <iostream>
#include <string>
#include <fstream>
#include <cstring>

// 简单的图像读取库（stb_image的单头文件版本）
#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION

// 内联stb_image库
inline unsigned char* stbi_load(const char* filename, int* x, int* y, int* channels_in_desired, int desired_channels) {
    // 简单的PNG读取实现（这里我们使用一个简化的方法）
    // 在实际使用中，你可能需要安装stb_image库
    return nullptr;
}

#include "elas.h"
#include "image.h"

using namespace std;

// 简单的PNG图像读取函数（使用系统命令转换）
image<uchar>* loadPNG(const char* filename) {
    cout << "正在处理PNG图像: " << filename << endl;
    
    // 创建临时PGM文件名
    string temp_pgm = string(filename) + ".temp.pgm";
    cout << "临时PGM文件: " << temp_pgm << endl;
    
    // 使用ImageMagick转换PNG到PGM（如果可用）
    string convert_cmd = "convert \"" + string(filename) + "\" -colorspace gray \"" + temp_pgm + "\"";
    cout << "执行转换命令: " << convert_cmd << endl;
    
    int result = system(convert_cmd.c_str());
    cout << "转换命令返回值: " << result << endl;
    
    if (result != 0) {
        // 如果ImageMagick不可用，尝试使用其他方法
        cout << "Warning: ImageMagick转换失败，尝试其他方法..." << endl;
        
        // 尝试使用magick命令（ImageMagick 7.x版本）
        string magick_cmd = "magick \"" + string(filename) + "\" -colorspace gray \"" + temp_pgm + "\"";
        cout << "尝试magick命令: " << magick_cmd << endl;
        result = system(magick_cmd.c_str());
        cout << "magick命令返回值: " << result << endl;
        
        if (result != 0) {
            cout << "Error: 无法转换PNG到PGM格式" << endl;
            cout << "请尝试以下解决方案:" << endl;
            cout << "1. 安装ImageMagick: sudo apt install imagemagick" << endl;
            cout << "2. 检查ImageMagick策略: sudo nano /etc/ImageMagick-6/policy.xml" << endl;
            cout << "3. 手动转换: convert input.png -colorspace gray output.pgm" << endl;
            return nullptr;
        }
    }
    
    // 检查临时文件是否创建成功
    ifstream temp_file(temp_pgm);
    if (!temp_file.good()) {
        cout << "Error: 临时PGM文件创建失败: " << temp_pgm << endl;
        return nullptr;
    }
    temp_file.close();
    
    cout << "临时PGM文件创建成功，开始读取..." << endl;
    
    // 读取转换后的PGM文件
    image<uchar>* img = loadPGM(temp_pgm.c_str());
    
    if (img == nullptr) {
        cout << "Error: 无法读取转换后的PGM文件" << endl;
    } else {
        cout << "PGM文件读取成功，图像尺寸: " << img->width() << "x" << img->height() << endl;
    }
    
    // 删除临时文件
    remove(temp_pgm.c_str());
    cout << "临时文件已清理" << endl;
    
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
            cout << "Warning: 无法转换为PNG格式，保持PGM格式" << endl;
            // 重命名临时文件为最终文件
            rename(temp_pgm.c_str(), filename);
            return;
        }
    }
    
    // 删除临时PGM文件
    remove(temp_pgm.c_str());
}

// 保存原始浮点数据
void saveRawFloat(const char* filename, float* data, int width, int height) {
    std::ofstream file(filename, std::ios::out | std::ios::binary);
    if (!file.is_open()) {
        std::cout << "Error: Could not open file " << filename << " for writing" << std::endl;
        return;
    }
    
    file.write((char*)data, width * height * sizeof(float));
    file.close();
}

int main(int argc, char** argv) {
    
    // 直接指定图像文件路径（从build目录的相对路径）
    const char* left_image_path = "../../../data/left/test1.png";
    const char* right_image_path = "../../../data/right/test1.png";
    
    cout << "ELAS Standalone 程序" << endl;
    cout << "===============================================" << endl;
    cout << "左图像: " << left_image_path << endl;
    cout << "右图像: " << right_image_path << endl;
    cout << "===============================================" << endl;
    
    // 检查文件是否存在
    ifstream left_file(left_image_path);
    if (!left_file.good()) {
        cout << "Error: 左图像文件不存在: " << left_image_path << endl;
        cout << "当前工作目录: ";
        (void)system("pwd");  // 忽略返回值
        return 1;
    }
    left_file.close();
    
    ifstream right_file(right_image_path);
    if (!right_file.good()) {
        cout << "Error: 右图像文件不存在: " << right_image_path << endl;
        return 1;
    }
    right_file.close();
    
    cout << "图像文件检查通过，开始处理..." << endl;
    
    // 读取图像
    image<uchar> *I1 = nullptr;
    image<uchar> *I2 = nullptr;
    
    try {
        // 尝试读取PNG图像
        I1 = loadPNG(left_image_path);
        I2 = loadPNG(right_image_path);
        
        if (I1 == nullptr || I2 == nullptr) {
            cout << "Error: Could not read PNG images!" << endl;
            cout << "请确保已安装ImageMagick: sudo apt install imagemagick" << endl;
            cout << "或者手动将PNG图像转换为PGM格式" << endl;
            return 1;
        }
    } catch (pnm_error& e) {
        cout << "Error: Could not read images!" << endl;
        cout << "请确保图像文件存在且格式正确" << endl;
        return 1;
    }
    
    // 检查图像尺寸
    if (I1->width() != I2->width() || I1->height() != I2->height()) {
        cout << "Error: Images must have same size!" << endl;
        cout << "左图像尺寸: " << I1->width() << "x" << I1->height() << endl;
        cout << "右图像尺寸: " << I2->width() << "x" << I2->height() << endl;
        delete I1;
        delete I2;
        return 1;
    }
    
    // 获取图像尺寸
    const int32_t width = I1->width();
    const int32_t height = I1->height();
    
    cout << "Processing images of size: " << width << "x" << height << endl;
    
    // 分配视差图像内存
    int32_t dims[3];
    dims[0] = width;  // bytes per line = width
    dims[1] = height; // height
    dims[2] = width;  // bytes per line = width
    
    // 分配视差图像内存
    float *D1_data = (float*)malloc(width * height * sizeof(float));
    float *D2_data = (float*)malloc(width * height * sizeof(float));
    
    // 设置ELAS参数
    Elas::parameters param;
    
    // ========== ELAS算法参数配置 ==========
    cout << "配置ELAS参数..." << endl;
    
    // ELAS视差范围参数
    param.disp_min = 0;
    param.disp_max = 256;
    
    // ELAS支持点检测参数
    param.support_threshold = 0.5;  // 对应SGBM的uniquenessRatio=50 (50/100)
    param.support_texture = 31;     // 对应SGBM的preFilterCap=31
    param.candidate_stepsize = 5;
    param.add_corners = false;
    
    // ELAS一致性检查参数
    param.incon_window_size = 5;
    param.incon_threshold = 5;
    param.incon_min_support = 5;
    
    // ELAS网格和插值参数
    param.grid_size = 20;
    param.ipol_gap_width = 3;
    
    // ELAS概率模型参数
    param.beta = 0.02;
    param.gamma = 3.0;
    param.sigma = 1.0;
    param.sradius = 2.0;
    
    // ELAS匹配参数
    param.match_texture = 31;       // 对应SGBM的preFilterCap=31
    param.lr_threshold = 5;         // 对应SGBM的disp12MaxDiff=5
    
    // ELAS后处理参数
    param.speckle_sim_threshold = 1.0;
    param.speckle_size = 100;       // 对应SGBM的speckleWindowSize=100
    param.filter_median = false;
    param.filter_adaptive_mean = true;
    param.postprocess_only_left = true;
    
    // ELAS性能参数
    param.subsampling = false;
    
    cout << "ELAS参数配置完成:" << endl;
    cout << "  视差范围: " << param.disp_min << " - " << param.disp_max << endl;
    cout << "  支持阈值: " << param.support_threshold << endl;
    cout << "  纹理阈值: " << param.support_texture << endl;
    cout << "  左右阈值: " << param.lr_threshold << endl;
    cout << "  斑点大小: " << param.speckle_size << endl;
    cout << "  后处理: " << (param.postprocess_only_left ? "仅左图" : "双图") << endl;
    cout << endl;
    
    Elas elas(param);
    
    cout << "Processing ... " << endl;
    
    // 处理
    elas.process(I1->data, I2->data, D1_data, D2_data, dims);
    
    cout << "... done!" << endl;
    
    // 输出目录
    string output_dir = ".";
    
    // 保存结果
    string disparity_png = output_dir + "/disparity_test1.png";  // 只保存左视差图
    string disparity_raw = output_dir + "/disparity_test1.raw";  // 左视差原始数据
    
    // 只保存左视差图（主要的视差图）
    savePNG_float(disparity_png.c_str(), D1_data, width, height);
    saveRawFloat(disparity_raw.c_str(), D1_data, width, height);
    
    cout << "Results saved to:" << endl;
    cout << "  " << disparity_png << " (视差图 - PNG格式)" << endl;
    cout << "  " << disparity_raw << " (视差原始数据)" << endl;
    cout << endl;
    cout << "注意: ELAS算法生成了两个视差图（左视差图和右视差图）" << endl;
    cout << "这里只保存了左视差图，它是主要的视差图，质量更好。" << endl;
    cout << "右视差图主要用于左右一致性检查和算法优化。" << endl;
    
    // 清理内存
    delete I1;
    delete I2;
    free(D1_data);
    free(D2_data);
    
    cout << "程序执行完成！" << endl;
    return 0;
} 