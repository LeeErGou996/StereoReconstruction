#include "disparity.h"
#include "ELAS/src/elas.h"
#include "utils/imageutils.h"
#include <vector>
#include <algorithm>
#include <cstdint>
#include <iostream>
#include <cmath>
#include <string>

void compute_disparity_elas(const std::string& left_path, const std::string& right_path, const std::string& out_path, const Elas::parameters& param, const std::string& disparityType) {
    MyImage imgL, imgR;
    if (!myImReadPNG(left_path, imgL) || !myImReadPNG(right_path, imgR)) {
        std::cerr << "Failed to read input images for disparity!" << std::endl;
        return;
    }
    // 转灰度
    auto to_gray = [](const MyImage& src) -> std::vector<uint8_t> {
        std::vector<uint8_t> gray(src.width * src.height);
        int step = src.channels;
        for (int i = 0; i < src.width * src.height; ++i) {
            if (step == 1) {
                gray[i] = src.data[i];
            } else {
                gray[i] = static_cast<uint8_t>(0.299f * src.data[i*step] + 0.587f * src.data[i*step+1] + 0.114f * src.data[i*step+2]);
            }
        }
        return gray;
    };
    std::vector<uint8_t> grayL = to_gray(imgL);
    std::vector<uint8_t> grayR = to_gray(imgR);
    int32_t dims[3] = {imgL.width, imgL.height, imgL.width};
    std::vector<float> dispL(imgL.width * imgL.height, 0.0f);
    std::vector<float> dispR(imgL.width * imgL.height, 0.0f);

    Elas elas(param);
    elas.process(grayL.data(), grayR.data(), dispL.data(), dispR.data(), dims);

    // 选定视差方向
    const float* disp = (disparityType == "right") ? dispR.data() : dispL.data();

    // 加入调试打印
    std::cout << "=== Disparity Sample Dump ===" << std::endl;
    int w = imgL.width;
    int h = imgL.height;
    for (int dy = 0; dy < 5; ++dy) {
        for (int dx = 0; dx < 5; ++dx) {
            int x = w / 3 + dx;
            int y = h / 3 + dy;
            float d = disp[y * w + x];
            std::cout << "disp[" << y << "," << x << "] = " << d << "\t";
        }
        std::cout << std::endl;
    }



    // 调试代码
    int valid_count = 0;
    float min_val = 1e9, max_val = -1e9;
    for (int i = 0; i < imgL.width * imgL.height; ++i) {
        float d = dispL[i];
        if (d > 0.0f && d < 1e4f) {  // 排除无效或极端大值
            valid_count++;
            if (d < min_val) min_val = d;
            if (d > max_val) max_val = d;
        }
    }
    std::cout << "[ELAS] Valid disparities: " << valid_count
            << ", Min: " << min_val << ", Max: " << max_val << std::endl;
            
    // 归一化并保存为PNG
    float min_disp = 1e9, max_disp = -1e9;
    for (int i = 0; i < imgL.width * imgL.height; ++i) {
        if (disp[i] > 0) {
            min_disp = std::min(min_disp, disp[i]);
            max_disp = std::max(max_disp, disp[i]);
        }
    }

    MyImage dispImg;
    dispImg.width = imgL.width;
    dispImg.height = imgL.height;
    dispImg.channels = 1;
    dispImg.data.resize(imgL.width * imgL.height);

    for (int i = 0; i < imgL.width * imgL.height; ++i) {
        if (disp[i] > 0 && max_disp > min_disp) {
            dispImg.data[i] = static_cast<unsigned char>(255.0f * (disp[i] - min_disp) / (max_disp - min_disp));
        } else {
            dispImg.data[i] = 0;
        }
    }
    myImWritePNG(out_path, dispImg);
    std::cout << "Saved disparity map to: " << out_path << std::endl;
}

// 兼容旧接口
void compute_disparity_elas(const std::string& left_path, const std::string& right_path, const std::string& out_path, const std::string& disparityType) {
    Elas::parameters param(Elas::MIDDLEBURY);
    compute_disparity_elas(left_path, right_path, out_path, param, disparityType);
} 