#include "../utils/imageutils.h"
#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <random>
#include <bitset>

// --- 特征点结构 ---
struct KeyPoint {
    int x, y;
    float response;
    float angle;
};

// --- 高效的ORB-like特征检测器 ---
class ORBDetector {
private:
    int max_features;
    int threshold;
    
    // 预计算的圆形模式
    static constexpr int circle_offsets[16][2] = {
        {0, -3}, {1, -3}, {2, -2}, {3, -1}, {3, 0}, {3, 1}, {2, 2}, {1, 3},
        {0, 3}, {-1, 3}, {-2, 2}, {-3, 1}, {-3, 0}, {-3, -1}, {-2, -2}, {-1, -3}
    };
    
    // 高效的角点响应计算
    inline bool isFastKeypoint(const MyImage& img, int x, int y) const {
        const int w = img.width;
        const float center = img.data[y * w + x];
        
        // 快速检查四个主要方向
        const float vals[4] = {
            static_cast<float>(img.data[(y-3)*w + x]),      // 上
            static_cast<float>(img.data[y*w + (x+3)]),      // 右
            static_cast<float>(img.data[(y+3)*w + x]),      // 下
            static_cast<float>(img.data[y*w + (x-3)])       // 左
        };
        
        int bright = 0, dark = 0;
        for (int i = 0; i < 4; ++i) {
            if (vals[i] > center + threshold) bright++;
            else if (vals[i] < center - threshold) dark++;
        }
        
        if (bright < 3 && dark < 3) return false;
        
        // 完整的16点检查
        int continuous = 0, max_continuous = 0;
        bool was_bright = false, was_dark = false;
        
        for (int i = 0; i < 16; ++i) {
            const float val = img.data[(y + circle_offsets[i][1]) * w + (x + circle_offsets[i][0])];
            const bool is_bright = val > center + threshold;
            const bool is_dark = val < center - threshold;
            
            if ((is_bright && was_bright) || (is_dark && was_dark)) {
                continuous++;
            } else if (is_bright || is_dark) {
                continuous = 1;
                was_bright = is_bright;
                was_dark = is_dark;
            } else {
                continuous = 0;
                was_bright = was_dark = false;
            }
            
            max_continuous = std::max(max_continuous, continuous);
            if (max_continuous >= 9) return true;
        }
        
        return false;
    }
    
    // 计算Harris响应作为角点强度
    inline float computeResponse(const MyImage& img, int x, int y) const {
        const int w = img.width;
        float Ixx = 0, Iyy = 0, Ixy = 0;
        
        // 3x3窗口内的梯度计算
        for (int dy = -1; dy <= 1; ++dy) {
            for (int dx = -1; dx <= 1; ++dx) {
                const int px = x + dx, py = y + dy;
                if (px > 0 && px < w-1 && py > 0 && py < img.height-1) {
                    const float gx = img.data[py*w + px+1] - img.data[py*w + px-1];
                    const float gy = img.data[(py+1)*w + px] - img.data[(py-1)*w + px];
                    Ixx += gx * gx;
                    Iyy += gy * gy;
                    Ixy += gx * gy;
                }
            }
        }
        
        const float det = Ixx * Iyy - Ixy * Ixy;
        const float trace = Ixx + Iyy;
        return det - 0.04f * trace * trace;
    }
    
public:
    ORBDetector(int max_feat = 1000, int thresh = 20) : max_features(max_feat), threshold(thresh) {}
    
    std::vector<KeyPoint> detect(const MyImage& img) {
        if (img.channels != 1) return {};
        
        std::vector<KeyPoint> candidates;
        candidates.reserve(max_features * 2);
        
        // 网格化检测，提高分布均匀性
        const int grid_size = 32;
        const int grid_w = img.width / grid_size;
        const int grid_h = img.height / grid_size;
        
        for (int gy = 0; gy < grid_h; ++gy) {
            for (int gx = 0; gx < grid_w; ++gx) {
                const int start_x = gx * grid_size + 4;
                const int end_x = std::min((gx + 1) * grid_size, img.width - 4);
                const int start_y = gy * grid_size + 4;
                const int end_y = std::min((gy + 1) * grid_size, img.height - 4);
                
                std::vector<KeyPoint> grid_points;
                
                for (int y = start_y; y < end_y; y += 2) {  // 步长为2，减少计算量
                    for (int x = start_x; x < end_x; x += 2) {
                        if (isFastKeypoint(img, x, y)) {
                            const float response = computeResponse(img, x, y);
                            if (response > 1000) {  // 响应阈值
                                grid_points.push_back({x, y, response, 0});
                            }
                        }
                    }
                }
                
                // 每个网格保留最强的几个点
                if (!grid_points.empty()) {
                    std::partial_sort(grid_points.begin(), 
                                    grid_points.begin() + std::min(4, (int)grid_points.size()),
                                    grid_points.end(),
                                    [](const KeyPoint& a, const KeyPoint& b) { return a.response > b.response; });
                    
                    for (int i = 0; i < std::min(4, (int)grid_points.size()); ++i) {
                        candidates.push_back(grid_points[i]);
                    }
                }
            }
        }
        
        // 全局非极大值抑制
        if (candidates.size() > max_features) {
            std::partial_sort(candidates.begin(), candidates.begin() + max_features, candidates.end(),
                            [](const KeyPoint& a, const KeyPoint& b) { return a.response > b.response; });
            candidates.resize(max_features);
        }
        
        return candidates;
    }
};

// --- 高效的ORB描述子 ---
class ORBDescriptor {
private:
    static constexpr int DESCRIPTOR_SIZE = 256;
    
    // 预定义的采样模式 (ORB论文中的优化模式)
    struct SamplePair {
        int x1, y1, x2, y2;
    };
    
    static constexpr SamplePair pattern[DESCRIPTOR_SIZE] = {
        {-8,-3,9,5},{4,2,7,-12},{-11,9,-8,2},{7,-12,12,-13},{2,-13,2,12},{1,-7,1,6},{-2,-10,-2,-4},{-13,-13,-11,-8},
        {-13,-3,-12,-9},{10,4,11,9},{-13,-8,-8,-9},{-11,7,-9,12},{7,7,12,6},{-4,-5,-3,0},{-13,2,-8,-8},{-7,-6,-6,-11},
        {-10,5,-10,6},{2,-4,3,-10},{-13,0,-13,5},{-13,-7,-12,12},{-13,3,-11,8},{-7,12,-4,7},{6,-10,12,8},{-9,-1,-7,-6},
        {-2,-5,0,12},{-12,5,-7,5},{3,-10,5,-8},{-7,-7,-4,5},{-3,-2,-1,-7},{2,9,5,-11},{-11,-13,-5,-13},{-1,6,0,-1},
        {5,-3,5,2},{-4,-13,-4,12},{-9,-6,-9,6},{-12,-10,-8,-4},{10,2,12,-3},{7,12,12,12},{-7,-13,-6,5},{-4,9,-3,4},
        {7,-1,12,2},{-7,6,-5,1},{-13,11,-12,5},{-3,7,-2,-6},{7,-8,12,-7},{-13,-7,-10,-15},{1,-6,4,-2},{-13,11,-11,-5},
        {11,11,12,-8},{3,-6,7,-15},{7,-1,12,9},{-1,-9,3,12},{9,1,12,6},{-1,5,1,3},{11,-13,11,-5},{5,-13,6,10},
        {2,-12,2,3},{3,8,4,-6},{2,6,12,-13},{9,-12,10,3},{-8,4,-7,9},{-11,12,-4,-6},{1,12,2,-8},{6,-9,7,-4},
        {2,3,3,-2},{6,-13,14,-8},{-13,-4,-6,-8},{-4,-2,1,4},{-5,-7,-3,4},{-8,-12,-6,7},{5,-6,13,-1},{-8,-1,-4,-3},
        {6,-1,6,8},{-13,-12,-8,-13},{-2,-2,0,5},{4,11,9,12},{0,-5,6,0},{-9,-13,-10,-8},{5,-1,10,12},{-1,5,3,13},
        {-7,-6,-5,-3},{-1,-6,2,6},{-1,-2,-1,8},{-13,-8,-8,-2},{-8,-8,-6,-14},{6,-12,8,5},{-13,0,-8,-4},{3,3,7,1},
        {8,9,12,-6},{-4,-10,-1,-1},{4,1,8,-4},{-2,-2,2,-13},{2,-12,12,12},{-2,-13,0,-6},{4,1,9,3},{-6,-10,-3,-5},
        {-3,-13,-1,1},{7,5,12,-11},{4,-2,5,-7},{-13,9,-9,-5},{7,1,8,6},{7,-8,7,6},{-7,-4,-7,1},{-8,11,-7,-8},
        {-13,6,-12,-8},{2,4,3,9},{10,-5,12,3},{-6,-5,-6,7},{8,-3,9,-8},{2,-12,2,8},{-11,-2,-10,3},{-12,-13,-7,-9},
        {-11,0,-10,-5},{5,-3,11,8},{-2,-13,-1,12},{-1,-8,0,9},{-13,-11,-12,-5},{-10,-2,-10,11},{-3,9,-2,-13},{2,-3,3,2},
        {-9,-13,-4,0},{-4,6,-3,-10},{-4,12,-2,-7},{-6,-11,-4,9},{6,-3,6,11},{-13,11,-5,5},{11,11,12,6},{7,-5,12,-2},
        {-1,12,0,7},{-4,-8,-3,-2},{-7,1,-6,7},{-13,-12,-8,-5},{-7,-2,-6,-8},{-8,5,-6,-9},{-5,-1,-4,5},{-13,7,-8,10},
        {1,5,5,-13},{1,0,10,-13},{9,12,10,-1},{5,-8,10,-9},{-1,11,1,-13},{-9,-3,-6,2},{-1,-10,1,12},{-13,1,-8,-10},
        {8,-11,10,-6},{2,-13,3,-6},{7,-13,12,-9},{-10,-10,-5,-7},{-10,-8,-8,-13},{4,-6,8,5},{3,12,8,-13},{-4,2,-3,-3},
        {5,-13,10,-12},{4,-13,5,-1},{-9,9,-4,3},{0,3,3,-9},{-12,1,-6,1},{3,2,4,-8},{-10,-10,-10,9},{8,-13,12,12},
        {-8,-12,-6,-5},{2,2,3,7},{10,6,11,-8},{6,8,8,-12},{-7,10,-6,5},{-3,-9,-3,9},{-1,-13,-1,5},{-3,-7,-3,4},
        {-8,-2,-8,3},{4,2,12,12},{2,-5,3,11},{6,-9,11,-13},{3,-1,7,12},{11,-1,12,4},{-3,0,-3,6},{4,-11,4,12},
        {2,-4,2,1},{-10,-6,-8,1},{-13,7,-11,1},{-13,12,-11,2},{6,0,11,-13},{0,-1,1,4},{-13,3,-9,-2},{-9,8,-6,-3},
        {-13,-6,-8,-2},{5,-9,8,10},{2,7,3,-9},{-1,-6,-1,-1},{9,5,11,-2},{11,-3,12,-8},{3,0,3,5},{-1,4,0,10},
        {3,-6,4,5},{-13,0,-10,5},{5,8,12,11},{8,9,9,-6},{7,-4,8,-12},{-10,4,-10,9},{7,3,12,4},{9,-7,10,-2},
        {7,0,12,-2},{-1,-6,0,-11}
    };
    
public:
    std::bitset<DESCRIPTOR_SIZE> compute(const MyImage& img, const KeyPoint& kp) {
        std::bitset<DESCRIPTOR_SIZE> descriptor;
        const int w = img.width, h = img.height;
        
        for (int i = 0; i < DESCRIPTOR_SIZE; ++i) {
            const auto& p = pattern[i];
            
            const int x1 = std::clamp(kp.x + p.x1, 0, w - 1);
            const int y1 = std::clamp(kp.y + p.y1, 0, h - 1);
            const int x2 = std::clamp(kp.x + p.x2, 0, w - 1);
            const int y2 = std::clamp(kp.y + p.y2, 0, h - 1);
            
            descriptor[i] = img.data[y1 * w + x1] < img.data[y2 * w + x2];
        }
        
        return descriptor;
    }
};

// --- 高效匹配器 ---
struct Match { int idx1, idx2; float dist; };

class BFMatcher {
public:
    static std::vector<Match> match(const std::vector<std::bitset<256>>& desc1,
                                   const std::vector<std::bitset<256>>& desc2,
                                   float ratio_threshold = 0.75f) {
        std::vector<Match> matches;
        matches.reserve(desc1.size());
        
        for (size_t i = 0; i < desc1.size(); ++i) {
            int best_dist = 257, second_best_dist = 257;
            int best_idx = -1;
            
            // 向量化汉明距离计算
            for (size_t j = 0; j < desc2.size(); ++j) {
                const int dist = static_cast<int>((desc1[i] ^ desc2[j]).count());
                
                if (dist < best_dist) {
                    second_best_dist = best_dist;
                    best_dist = dist;
                    best_idx = static_cast<int>(j);
                } else if (dist < second_best_dist) {
                    second_best_dist = dist;
                }
            }
            
            // Lowe's ratio test
            if (best_idx >= 0 && best_dist < 80 && 
                static_cast<float>(best_dist) / second_best_dist < ratio_threshold) {
                matches.push_back({static_cast<int>(i), best_idx, static_cast<float>(best_dist)});
            }
        }
        
        return matches;
    }
};

// --- 主要的detectAndMatch接口 ---
std::pair<std::vector<KeyPoint>, std::vector<Match>> 
detectAndMatch(const MyImage& img1, const MyImage& img2) {
    ORBDetector detector(1000, 20);
    ORBDescriptor descriptor;
    
    // 检测
    auto kp1 = detector.detect(img1);
    auto kp2 = detector.detect(img2);
    
    if (kp1.empty() || kp2.empty()) {
        return {{}, {}};
    }
    
    // 计算描述子
    std::vector<std::bitset<256>> desc1, desc2;
    desc1.reserve(kp1.size());
    desc2.reserve(kp2.size());
    
    for (const auto& kp : kp1) {
        desc1.push_back(descriptor.compute(img1, kp));
    }
    
    for (const auto& kp : kp2) {
        desc2.push_back(descriptor.compute(img2, kp));
    }
    
    // 匹配
    auto matches = BFMatcher::match(desc1, desc2);
    
    // 合并关键点
    std::vector<KeyPoint> all_kp = kp1;
    all_kp.insert(all_kp.end(), kp2.begin(), kp2.end());
    
    return {all_kp, matches};
}

// --- 可视化函数 ---
void drawMatches(const MyImage& img1, const std::vector<KeyPoint>& kp1,
                 const MyImage& img2, const std::vector<KeyPoint>& kp2,
                 const std::vector<Match>& matches, MyImage& outImg) {
    const int w = img1.width + img2.width;
    const int h = std::max(img1.height, img2.height);
    
    outImg.width = w;
    outImg.height = h;
    outImg.channels = 3;
    outImg.data.assign(w * h * 3, 255);
    
    // 复制图像
    for (int y = 0; y < img1.height; ++y) {
        for (int x = 0; x < img1.width; ++x) {
            const auto val = static_cast<unsigned char>(std::max(0, std::min(255, static_cast<int>(img1.data[y * img1.width + x]))));
            for (int c = 0; c < 3; ++c) {
                outImg.data[(y * w + x) * 3 + c] = val;
            }
        }
    }
    
    for (int y = 0; y < img2.height; ++y) {
        for (int x = 0; x < img2.width; ++x) {
            const auto val = static_cast<unsigned char>(std::max(0, std::min(255, static_cast<int>(img2.data[y * img2.width + x]))));
            for (int c = 0; c < 3; ++c) {
                outImg.data[(y * w + (x + img1.width)) * 3 + c] = val;
            }
        }
    }
    
    // 绘制匹配
    std::mt19937 rng(42);
    for (const auto& m : matches) {
        const int x1 = kp1[m.idx1].x, y1 = kp1[m.idx1].y;
        const int x2 = kp2[m.idx2].x + img1.width, y2 = kp2[m.idx2].y;
        
        const unsigned char r = rng() % 256, g = rng() % 256, b = rng() % 256;
        
        // 绘制关键点
        for (int dy = -2; dy <= 2; ++dy) {
            for (int dx = -2; dx <= 2; ++dx) {
                const int xx1 = x1 + dx, yy1 = y1 + dy;
                const int xx2 = x2 + dx, yy2 = y2 + dy;
                
                if (xx1 >= 0 && xx1 < w && yy1 >= 0 && yy1 < h) {
                    outImg.data[(yy1 * w + xx1) * 3] = r;
                    outImg.data[(yy1 * w + xx1) * 3 + 1] = g;
                    outImg.data[(yy1 * w + xx1) * 3 + 2] = b;
                }
                
                if (xx2 >= 0 && xx2 < w && yy2 >= 0 && yy2 < h) {
                    outImg.data[(yy2 * w + xx2) * 3] = r;
                    outImg.data[(yy2 * w + xx2) * 3 + 1] = g;
                    outImg.data[(yy2 * w + xx2) * 3 + 2] = b;
                }
            }
        }
        
        // 绘制连线
        const int dx = std::abs(x2 - x1), dy = std::abs(y2 - y1);
        const int sx = x1 < x2 ? 1 : -1, sy = y1 < y2 ? 1 : -1;
        int err = dx - dy, cx = x1, cy = y1;
        
        while (true) {
            if (cx >= 0 && cx < w && cy >= 0 && cy < h) {
                outImg.data[(cy * w + cx) * 3] = r;
                outImg.data[(cy * w + cx) * 3 + 1] = g;
                outImg.data[(cy * w + cx) * 3 + 2] = b;
            }
            
            if (cx == x2 && cy == y2) break;
            
            const int e2 = 2 * err;
            if (e2 > -dy) { err -= dy; cx += sx; }
            if (e2 < dx) { err += dx; cy += sy; }
        }
    }
}

// --- 保持原接口兼容性 ---
void run_feature_matching(const std::string& leftPath, const std::string& rightPath, const std::string& outPath) {
    MyImage imgL, imgR;
    
    if (!myImReadPNG(leftPath, imgL) || !myImReadPNG(rightPath, imgR)) {
        std::cout << "Failed to read input images: " << leftPath << ", " << rightPath << std::endl;
        return;
    }
    // 保存原始读取的图像
    myImWritePNG("step1_readL.png", imgL);
    myImWritePNG("step1_readR.png", imgR);
    std::cout << "[step1] imgL: width=" << imgL.width << ", height=" << imgL.height << ", channels=" << imgL.channels << ", data.size()=" << imgL.data.size() << std::endl;
    std::cout << "[step1] imgR: width=" << imgR.width << ", height=" << imgR.height << ", channels=" << imgR.channels << ", data.size()=" << imgR.data.size() << std::endl;
    // 转换为灰度图
    if (imgL.channels == 3) {
        MyImage grayL;
        grayL.width = imgL.width;
        grayL.height = imgL.height;
        grayL.channels = 1;
        grayL.data.resize(imgL.width * imgL.height);
        for (int i = 0; i < imgL.width * imgL.height; ++i) {
            grayL.data[i] = 0.299f * imgL.data[i*3] + 0.587f * imgL.data[i*3+1] + 0.114f * imgL.data[i*3+2];
        }
        imgL = std::move(grayL);
    } else if (imgL.channels == 4) {
        MyImage grayL;
        grayL.width = imgL.width;
        grayL.height = imgL.height;
        grayL.channels = 1;
        grayL.data.resize(imgL.width * imgL.height);
        for (int i = 0; i < imgL.width * imgL.height; ++i) {
            grayL.data[i] = 0.299f * imgL.data[i*4] + 0.587f * imgL.data[i*4+1] + 0.114f * imgL.data[i*4+2];
            // 忽略alpha通道
        }
        imgL = std::move(grayL);
    } else if (imgL.channels == 1) {
        // 已经是灰度图，什么都不做
    } else {
        std::cout << "Unsupported channel count in imgL: " << imgL.channels << std::endl;
        return;
    }
    myImWritePNG("step2_grayL.png", imgL);
    std::cout << "[step2] imgL: width=" << imgL.width << ", height=" << imgL.height << ", channels=" << imgL.channels << ", data.size()=" << imgL.data.size() << std::endl;
    if (imgR.channels == 3) {
        MyImage grayR;
        grayR.width = imgR.width;
        grayR.height = imgR.height;
        grayR.channels = 1;
        grayR.data.resize(imgR.width * imgR.height);
        for (int i = 0; i < imgR.width * imgR.height; ++i) {
            grayR.data[i] = 0.299f * imgR.data[i*3] + 0.587f * imgR.data[i*3+1] + 0.114f * imgR.data[i*3+2];
        }
        imgR = std::move(grayR);
    } else if (imgR.channels == 4) {
        MyImage grayR;
        grayR.width = imgR.width;
        grayR.height = imgR.height;
        grayR.channels = 1;
        grayR.data.resize(imgR.width * imgR.height);
        for (int i = 0; i < imgR.width * imgR.height; ++i) {
            grayR.data[i] = 0.299f * imgR.data[i*4] + 0.587f * imgR.data[i*4+1] + 0.114f * imgR.data[i*4+2];
            // 忽略alpha通道
        }
        imgR = std::move(grayR);
    } else if (imgR.channels == 1) {
        // 已经是灰度图，什么都不做
    } else {
        std::cout << "Unsupported channel count in imgR: " << imgR.channels << std::endl;
        return;
    }
    myImWritePNG("step2_grayR.png", imgR);
    std::cout << "[step2] imgR: width=" << imgR.width << ", height=" << imgR.height << ", channels=" << imgR.channels << ", data.size()=" << imgR.data.size() << std::endl;
    
    // 使用高效的detectAndMatch
    auto [all_kp, matches] = detectAndMatch(imgL, imgR);
    
    // 分离关键点
    std::vector<KeyPoint> kpL(all_kp.begin(), all_kp.begin() + (all_kp.size() >= matches.size() ? all_kp.size()/2 : 0));
    std::vector<KeyPoint> kpR(all_kp.begin() + kpL.size(), all_kp.end());
    
    std::cout << "Detected " << kpL.size() << " and " << kpR.size() << " keypoints." << std::endl;
    std::cout << "Matched " << matches.size() << " pairs." << std::endl;
    
    // 可视化和保存
    MyImage matchImg;
    drawMatches(imgL, kpL, imgR, kpR, matches, matchImg);
    myImWritePNG(outPath, matchImg);
    std::cout << "Saved " << outPath << std::endl;
}

int get_feature_count(const std::string& imgPath) {
    MyImage img;
    if (!myImReadPNG(imgPath, img)) return 0;
    
    if (img.channels > 1) {
        MyImage gray;
        gray.width = img.width;
        gray.height = img.height;
        gray.channels = 1;
        gray.data.resize(img.width * img.height);
        
        for (int i = 0; i < img.width * img.height; ++i) {
            gray.data[i] = 0.299f * img.data[i*3] + 0.587f * img.data[i*3+1] + 0.114f * img.data[i*3+2];
        }
        img = std::move(gray);
    }
    
    ORBDetector detector;
    auto keypoints = detector.detect(img);
    return static_cast<int>(keypoints.size());
}

int get_match_count(const std::string& leftPath, const std::string& rightPath) {
    MyImage imgL, imgR;
    if (!myImReadPNG(leftPath, imgL) || !myImReadPNG(rightPath, imgR)) return 0;
    
    if (imgL.channels > 1) {
        MyImage grayL;
        grayL.width = imgL.width;
        grayL.height = imgL.height; 
        grayL.channels = 1;
        grayL.data.resize(imgL.width * imgL.height);
        
        for (int i = 0; i < imgL.width * imgL.height; ++i) {
            grayL.data[i] = 0.299f * imgL.data[i*3] + 0.587f * imgL.data[i*3+1] + 0.114f * imgL.data[i*3+2];
        }
        imgL = std::move(grayL);
    }
    
    if (imgR.channels > 1) {
        MyImage grayR;
        grayR.width = imgR.width;
        grayR.height = imgR.height;
        grayR.channels = 1; 
        grayR.data.resize(imgR.width * imgR.height);
        
        for (int i = 0; i < imgR.width * imgR.height; ++i) {
            grayR.data[i] = 0.299f * imgR.data[i*3] + 0.587f * imgR.data[i*3+1] + 0.114f * imgR.data[i*3+2];
        }
        imgR = std::move(grayR);
    }
    
    auto [all_kp, matches] = detectAndMatch(imgL, imgR);
    return static_cast<int>(matches.size());
}