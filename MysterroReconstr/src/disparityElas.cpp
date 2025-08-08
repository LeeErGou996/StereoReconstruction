#include "../src/feature_match.h"
#include "../src/8point.h"
#include "../src/utils/rectification.h"
#include "../src/utils/imageutils.h"
#include "../src/disparity.h"
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

int main() {
    std::string leftPath = "../data/left/test1.png";
    std::string rightPath = "../data/right/test1.png";
    std::string KPath = "../data/camera/test1.txt";
    std::string outMatchPath = "test1_match.png";
    std::string elasConfigPath = "config.txt";

    // 1. 特征匹配
    MyImage imgL, imgR;
    if (!myImReadPNG(leftPath, imgL) || !myImReadPNG(rightPath, imgR)) {
        std::cerr << "Failed to read input images!" << std::endl;
        return 1;
    }
    if (imgL.channels > 1) {
        MyImage grayL;
        grayL.width = imgL.width;
        grayL.height = imgL.height;
        grayL.channels = 1;
        grayL.data.resize(imgL.width * imgL.height);
        int step = imgL.channels;
        for (int i = 0; i < imgL.width * imgL.height; ++i) {
            grayL.data[i] = 0.299f * imgL.data[i*step] + 0.587f * imgL.data[i*step+1] + 0.114f * imgL.data[i*step+2];
        }
        imgL = std::move(grayL);
    }
    if (imgR.channels > 1) {
        MyImage grayR;
        grayR.width = imgR.width;
        grayR.height = imgR.height;
        grayR.channels = 1;
        grayR.data.resize(imgR.width * imgR.height);
        int step = imgR.channels;
        for (int i = 0; i < imgR.width * imgR.height; ++i) {
            grayR.data[i] = 0.299f * imgR.data[i*step] + 0.587f * imgR.data[i*step+1] + 0.114f * imgR.data[i*step+2];
        }
        imgR = std::move(grayR);
    }
    auto result = detectAndMatch(imgL, imgR);
    const auto& all_kp = result.first;
    const auto& matches = result.second;
    size_t kp_count = all_kp.size() / 2;
    std::vector<KeyPoint> kpL(all_kp.begin(), all_kp.begin() + kp_count);
    std::vector<KeyPoint> kpR(all_kp.begin() + kp_count, all_kp.end());
    std::cout << "Detected " << kpL.size() << " and " << kpR.size() << " keypoints." << std::endl;
    std::cout << "Matched " << matches.size() << " pairs." << std::endl;
    if (matches.empty()) {
        std::cerr << "No matches found!" << std::endl;
        return 1;
    }
    MyImage matchImg;
    drawMatches(imgL, kpL, imgR, kpR, matches, matchImg);
    myImWritePNG(outMatchPath, matchImg);
    std::cout << "Saved matching result to: " << outMatchPath << std::endl;
    // ...（省略中间流程，直接调用disparity）...
    // 直接设置ELAS参数
    Elas::parameters elasParam(Elas::MIDDLEBURY);
    elasParam.disp_min = 0;
    elasParam.disp_max = 256;           // 缩小最大视差，聚焦常见物体区域
    elasParam.support_threshold = 0.15f; // 大幅降低支持阈值，提升有效区域
    elasParam.support_texture = 3;      // 降低纹理要求
    elasParam.candidate_stepsize = 3;   // 提高匹配密度
    elasParam.add_corners = true;       // 增加角点支持
    elasParam.incon_window_size = 3;    // 降低不一致窗口
    elasParam.incon_threshold = 3;      // 降低不一致阈值
    elasParam.incon_min_support = 2;    // 降低最小支持点数
    elasParam.grid_size = 1;           // 提高稠密度
    elasParam.ipol_gap_width = 20;      // 增大插值宽度，填补空洞
    elasParam.beta = 0.01f;             // 降低平滑项
    elasParam.gamma = 2.0f;             // 降低先验
    elasParam.sigma = 1.0f;
    elasParam.sradius = 2.0f;
    elasParam.match_texture = 3;        // 降低匹配纹理要求
    elasParam.lr_threshold = 8;         // 放宽左右一致性
    elasParam.speckle_sim_threshold = 2.0f; // 放宽斑点相似性
    elasParam.speckle_size = 30;        // 降低斑点过滤
    elasParam.filter_median = false;
    elasParam.filter_adaptive_mean = true;
    elasParam.postprocess_only_left = true;
    elasParam.subsampling = false;
    std::string rectL = "rectified_left.png";
    std::string rectR = "rectified_right.png";
    std::string outDisp = "disparity_ELAS.png";
    compute_disparity_elas(rectL, rectR, outDisp, elasParam, "left");
    std::cout << "Disparity map generated: " << outDisp << std::endl;
    return 0;
} 