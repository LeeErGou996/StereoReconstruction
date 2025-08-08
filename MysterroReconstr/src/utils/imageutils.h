#pragma once
#include <string>
#include <vector>
#include <array> // Added for std::array

struct MyImage {
    int width = 0;
    int height = 0;
    int channels = 0; // 1=gray, 3=rgb, 4=rgba
    std::vector<unsigned char> data; // 行优先，RGBRGB... 或 YYY...
};

// 读取PNG文件到MyImage
bool myImReadPNG(const std::string& filename, MyImage& img);
// 保存MyImage为PNG文件
bool myImWritePNG(const std::string& filename, const MyImage& img);
// 对MyImage应用3x3单应变换（最近邻插值）
void warpImage(const MyImage& src, MyImage& dst, const std::array<std::array<double, 3>, 3>& H); 

// 新增功能：绘图推荐图A工具
void colorize_disparity(const MyImage& gray, MyImage& color);
void hstack3(const MyImage& a, const MyImage& b, const MyImage& c, MyImage& out);

bool myImReadPFM(const std::string& filename, MyImage& img);

// 读取PGM文件到MyImage
bool myImReadPGM(const std::string& filename, MyImage& img);
