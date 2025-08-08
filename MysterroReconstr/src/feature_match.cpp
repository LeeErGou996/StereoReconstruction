#include "utils/imageutils.h"
#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <random>
#include <array>
#include <set>
#include <limits>

// --- 常量定义 ---
constexpr float SIFT_SIGMA = 1.6f;
constexpr int SIFT_OCTAVE_LAYERS = 3;
constexpr float SIFT_CONTRAST_THRESHOLD = 0.04f;
constexpr float SIFT_EDGE_THRESHOLD = 10.0f;
constexpr int SIFT_DESCRIPTOR_SIZE = 128;
constexpr int SIFT_DESCRIPTOR_WIDTH = 4;
constexpr int SIFT_HIST_BINS = 36;

// --- 特征点结构 ---
struct SIFTKeyPoint {
    float x, y;           // 位置
    float scale;          // 尺度
    float response;       // 响应强度
    float angle;          // 主方向
    int octave;           // 金字塔层
    int layer;            // 金字塔内层数
    std::vector<float> descriptor;  // 128维描述子
    
    SIFTKeyPoint() : x(0), y(0), scale(0), response(0), angle(0), octave(0), layer(0) {
        descriptor.resize(SIFT_DESCRIPTOR_SIZE, 0.0f);
    }
};

// --- SIFT匹配结构 ---
struct SIFTMatch {
    int idx1, idx2;
    float distance;
};

// --- 兼容接口结构体 ---
struct KeyPoint {
    int x, y;
    float response;
    float angle;
    float scale;
    
    KeyPoint(const SIFTKeyPoint& sift_kp) 
        : x(static_cast<int>(sift_kp.x)), y(static_cast<int>(sift_kp.y)), 
          response(sift_kp.response), angle(sift_kp.angle), scale(sift_kp.scale) {}
    
    KeyPoint() : x(0), y(0), response(0), angle(0), scale(0) {}
};

struct Match {
    int idx1, idx2;
    float dist;
    
    Match(const SIFTMatch& sift_match) 
        : idx1(sift_match.idx1), idx2(sift_match.idx2), dist(sift_match.distance) {}
    
    Match() : idx1(0), idx2(0), dist(0) {}
};

// --- 高斯模糊 ---
class GaussianBlur {
public:
    static MyImage blur(const MyImage& src, float sigma) {
        int kernel_size = static_cast<int>(std::ceil(sigma * 6.0f));
        if (kernel_size % 2 == 0) kernel_size++;
        
        std::vector<float> kernel = createGaussianKernel(kernel_size, sigma);
        
        // 先水平模糊
        MyImage temp = blurHorizontal(src, kernel);
        // 再垂直模糊
        return blurVertical(temp, kernel);
    }
    
private:
    static std::vector<float> createGaussianKernel(int size, float sigma) {
        std::vector<float> kernel(size);
        int center = size / 2;
        float sum = 0.0f;
        
        for (int i = 0; i < size; ++i) {
            int x = i - center;
            kernel[i] = std::exp(-x * x / (2.0f * sigma * sigma));
            sum += kernel[i];
        }
        
        // 归一化
        for (float& k : kernel) {
            k /= sum;
        }
        
        return kernel;
    }
    
    static MyImage blurHorizontal(const MyImage& src, const std::vector<float>& kernel) {
        MyImage result = src;
        int radius = kernel.size() / 2;
        
        for (int y = 0; y < src.height; ++y) {
            for (int x = 0; x < src.width; ++x) {
                float sum = 0.0f;
                
                for (int k = 0; k < static_cast<int>(kernel.size()); ++k) {
                    int px = x + k - radius;
                    px = std::clamp(px, 0, src.width - 1);
                    sum += src.data[y * src.width + px] * kernel[k];
                }
                
                result.data[y * src.width + x] = sum;
            }
        }
        
        return result;
    }
    
    static MyImage blurVertical(const MyImage& src, const std::vector<float>& kernel) {
        MyImage result = src;
        int radius = kernel.size() / 2;
        
        for (int y = 0; y < src.height; ++y) {
            for (int x = 0; x < src.width; ++x) {
                float sum = 0.0f;
                
                for (int k = 0; k < static_cast<int>(kernel.size()); ++k) {
                    int py = y + k - radius;
                    py = std::clamp(py, 0, src.height - 1);
                    sum += src.data[py * src.width + x] * kernel[k];
                }
                
                result.data[y * src.width + x] = sum;
            }
        }
        
        return result;
    }
};

// --- 图像缩放 ---
class ImagePyramid {
public:
    static MyImage downSample(const MyImage& src) {
        int new_width = src.width / 2;
        int new_height = src.height / 2;
        
        MyImage result;
        result.width = new_width;
        result.height = new_height;
        result.channels = src.channels;
        result.data.resize(new_width * new_height);
        
        for (int y = 0; y < new_height; ++y) {
            for (int x = 0; x < new_width; ++x) {
                result.data[y * new_width + x] = src.data[(y * 2) * src.width + (x * 2)];
            }
        }
        
        return result;
    }
};

// --- SIFT检测器 ---
class SIFTDetector {
private:
    int n_octaves;
    int n_layers;
    float sigma0;
    float contrast_threshold;
    float edge_threshold;
    
    // 高斯金字塔
    std::vector<std::vector<MyImage>> gaussian_pyramid;
    // DoG金字塔
    std::vector<std::vector<MyImage>> dog_pyramid;
    
public:
    SIFTDetector(int octaves = 4, int layers = SIFT_OCTAVE_LAYERS, 
                 float sigma = SIFT_SIGMA, float contrast_thresh = SIFT_CONTRAST_THRESHOLD,
                 float edge_thresh = SIFT_EDGE_THRESHOLD)
        : n_octaves(octaves), n_layers(layers), sigma0(sigma), 
          contrast_threshold(contrast_thresh), edge_threshold(edge_thresh) {}
    
    std::vector<SIFTKeyPoint> detect(const MyImage& image) {
        // 构建高斯金字塔
        buildGaussianPyramid(image);
        
        // 构建DoG金字塔
        buildDoGPyramid();
        
        // 检测关键点
        auto keypoints = detectKeyPoints();
        
        // 计算方向
        assignOrientations(keypoints);
        
        // 计算描述子
        computeDescriptors(keypoints);
        
        return keypoints;
    }
    
private:
    void buildGaussianPyramid(const MyImage& base_image) {
        gaussian_pyramid.clear();
        gaussian_pyramid.resize(n_octaves);
        
        // 预处理基础图像
        MyImage base = GaussianBlur::blur(base_image, sigma0);
        
        for (int octave = 0; octave < n_octaves; ++octave) {
            gaussian_pyramid[octave].resize(n_layers + 3);
            
            if (octave == 0) {
                gaussian_pyramid[octave][0] = base;
            } else {
                // 从上一个八度的倒数第三张图像下采样
                gaussian_pyramid[octave][0] = ImagePyramid::downSample(
                    gaussian_pyramid[octave-1][n_layers]);
            }
            
            // 构建当前八度的高斯图像
            float k = std::pow(2.0f, 1.0f / n_layers);
            for (int layer = 1; layer < n_layers + 3; ++layer) {
                float sigma = sigma0 * std::pow(k, layer);
                gaussian_pyramid[octave][layer] = GaussianBlur::blur(
                    gaussian_pyramid[octave][layer-1], sigma * std::sqrt(k*k - 1));
            }
        }
    }
    
    void buildDoGPyramid() {
        dog_pyramid.clear();
        dog_pyramid.resize(n_octaves);
        
        for (int octave = 0; octave < n_octaves; ++octave) {
            dog_pyramid[octave].resize(n_layers + 2);
            
            for (int layer = 0; layer < n_layers + 2; ++layer) {
                const auto& img1 = gaussian_pyramid[octave][layer];
                const auto& img2 = gaussian_pyramid[octave][layer + 1];
                
                MyImage dog;
                dog.width = img1.width;
                dog.height = img1.height;
                dog.channels = 1;
                dog.data.resize(img1.width * img1.height);
                
                for (size_t i = 0; i < dog.data.size(); ++i) {
                    dog.data[i] = img2.data[i] - img1.data[i];
                }
                
                dog_pyramid[octave][layer] = std::move(dog);
            }
        }
    }
    
    std::vector<SIFTKeyPoint> detectKeyPoints() {
        std::vector<SIFTKeyPoint> keypoints;
        
        for (int octave = 0; octave < n_octaves; ++octave) {
            for (int layer = 1; layer < n_layers + 1; ++layer) {
                detectKeypointsInLayer(octave, layer, keypoints);
            }
        }
        
        return keypoints;
    }
    
    void detectKeypointsInLayer(int octave, int layer, std::vector<SIFTKeyPoint>& keypoints) {
        const auto& dog_img = dog_pyramid[octave][layer];
        const int width = dog_img.width;
        const int height = dog_img.height;
        
        for (int y = 1; y < height - 1; ++y) {
            for (int x = 1; x < width - 1; ++x) {
                float val = dog_img.data[y * width + x];
                
                if (std::abs(val) < contrast_threshold) continue;
                
                if (isLocalExtremum(octave, layer, x, y)) {
                    SIFTKeyPoint kp = refineKeypoint(octave, layer, x, y);
                    if (kp.response > 0 && !isEdgeResponse(octave, layer, static_cast<int>(kp.x), static_cast<int>(kp.y))) {
                        keypoints.push_back(kp);
                    }
                }
            }
        }
    }
    
    bool isLocalExtremum(int octave, int layer, int x, int y) {
        const auto& current = dog_pyramid[octave][layer];
        const auto& above = dog_pyramid[octave][layer + 1];
        const auto& below = dog_pyramid[octave][layer - 1];
        
        float val = current.data[y * current.width + x];
        bool is_max = true, is_min = true;
        
        // 检查3x3x3邻域
        for (int dz = -1; dz <= 1; ++dz) {
            const auto& img = (dz == -1) ? below : (dz == 0) ? current : above;
            
            for (int dy = -1; dy <= 1; ++dy) {
                for (int dx = -1; dx <= 1; ++dx) {
                    if (dz == 0 && dy == 0 && dx == 0) continue;
                    
                    float neighbor = img.data[(y + dy) * img.width + (x + dx)];
                    if (val <= neighbor) is_max = false;
                    if (val >= neighbor) is_min = false;
                    
                    if (!is_max && !is_min) return false;
                }
            }
        }
        
        return is_max || is_min;
    }
    
    SIFTKeyPoint refineKeypoint(int octave, int layer, int x, int y) {
        SIFTKeyPoint kp;
        
        // 简化的亚像素精度定位
        const auto& dog_img = dog_pyramid[octave][layer];
        
        kp.x = x;
        kp.y = y;
        kp.octave = octave;
        kp.layer = layer;
        kp.scale = sigma0 * std::pow(2.0f, octave + static_cast<float>(layer) / n_layers);
        kp.response = std::abs(dog_img.data[y * dog_img.width + x]);
        
        return kp;
    }
    
    bool isEdgeResponse(int octave, int layer, int x, int y) {
        const auto& dog_img = dog_pyramid[octave][layer];
        const int w = dog_img.width;
        
        // 计算Hessian矩阵
        float dxx = dog_img.data[y * w + x + 1] + dog_img.data[y * w + x - 1] - 2 * dog_img.data[y * w + x];
        float dyy = dog_img.data[(y + 1) * w + x] + dog_img.data[(y - 1) * w + x] - 2 * dog_img.data[y * w + x];
        float dxy = (dog_img.data[(y + 1) * w + x + 1] - dog_img.data[(y + 1) * w + x - 1] -
                     dog_img.data[(y - 1) * w + x + 1] + dog_img.data[(y - 1) * w + x - 1]) / 4.0f;
        
        float trace = dxx + dyy;
        float det = dxx * dyy - dxy * dxy;
        
        if (det <= 0) return true;
        
        float ratio = trace * trace / det;
        float threshold = (edge_threshold + 1) * (edge_threshold + 1) / edge_threshold;
        
        return ratio > threshold;
    }
    
    void assignOrientations(std::vector<SIFTKeyPoint>& keypoints) {
        for (auto& kp : keypoints) {
            const auto& img = gaussian_pyramid[kp.octave][kp.layer];
            
            // 计算梯度直方图
            std::vector<float> hist(SIFT_HIST_BINS, 0.0f);
            
            float scale = 1.5f * kp.scale;
            int radius = static_cast<int>(std::round(3 * scale));
            
            for (int dy = -radius; dy <= radius; ++dy) {
                for (int dx = -radius; dx <= radius; ++dx) {
                    int px = static_cast<int>(kp.x) + dx;
                    int py = static_cast<int>(kp.y) + dy;
                    
                    if (px <= 0 || px >= img.width - 1 || py <= 0 || py >= img.height - 1) continue;
                    
                    if (dx * dx + dy * dy > radius * radius) continue;
                    
                    float gx = img.data[py * img.width + px + 1] - img.data[py * img.width + px - 1];
                    float gy = img.data[(py + 1) * img.width + px] - img.data[(py - 1) * img.width + px];
                    
                    float mag = std::sqrt(gx * gx + gy * gy);
                    float angle = std::atan2(gy, gx);
                    
                    // 高斯权重
                    float weight = std::exp(-(dx * dx + dy * dy) / (2 * scale * scale));
                    
                    // 添加到直方图
                    int bin = static_cast<int>(std::round((angle + M_PI) * SIFT_HIST_BINS / (2 * M_PI))) % SIFT_HIST_BINS;
                    hist[bin] += mag * weight;
                }
            }
            
            // 找到主方向
            int max_bin = std::max_element(hist.begin(), hist.end()) - hist.begin();
            kp.angle = (max_bin * 2 * M_PI / SIFT_HIST_BINS) - M_PI;
        }
    }
    
    void computeDescriptors(std::vector<SIFTKeyPoint>& keypoints) {
        for (auto& kp : keypoints) {
            computeDescriptor(kp);
        }
    }
    
    void computeDescriptor(SIFTKeyPoint& kp) {
        const auto& img = gaussian_pyramid[kp.octave][kp.layer];
        
        float cos_angle = std::cos(kp.angle);
        float sin_angle = std::sin(kp.angle);
        
        std::vector<float> descriptor(SIFT_DESCRIPTOR_SIZE, 0.0f);
        
        float scale = kp.scale * 3.0f;  // 描述子窗口尺度
        int radius = static_cast<int>(std::round(scale * SIFT_DESCRIPTOR_WIDTH * 0.5f * std::sqrt(2.0f)));
        
        for (int dy = -radius; dy <= radius; ++dy) {
            for (int dx = -radius; dx <= radius; ++dx) {
                // 旋转坐标
                float rx = dx * cos_angle - dy * sin_angle;
                float ry = dx * sin_angle + dy * cos_angle;
                
                int px = static_cast<int>(kp.x) + dx;
                int py = static_cast<int>(kp.y) + dy;
                
                if (px <= 0 || px >= img.width - 1 || py <= 0 || py >= img.height - 1) continue;
                
                // 计算在描述子网格中的位置
                float grid_x = rx / scale + SIFT_DESCRIPTOR_WIDTH * 0.5f - 0.5f;
                float grid_y = ry / scale + SIFT_DESCRIPTOR_WIDTH * 0.5f - 0.5f;
                
                if (grid_x < 0 || grid_x >= SIFT_DESCRIPTOR_WIDTH || 
                    grid_y < 0 || grid_y >= SIFT_DESCRIPTOR_WIDTH) continue;
                
                // 计算梯度
                float gx = img.data[py * img.width + px + 1] - img.data[py * img.width + px - 1];
                float gy = img.data[(py + 1) * img.width + px] - img.data[(py - 1) * img.width + px];
                
                float mag = std::sqrt(gx * gx + gy * gy);
                float angle = std::atan2(gy, gx) - kp.angle;
                
                // 归一化角度
                while (angle < 0) angle += 2 * M_PI;
                while (angle >= 2 * M_PI) angle -= 2 * M_PI;
                
                // 高斯权重
                float weight = std::exp(-(rx * rx + ry * ry) / (2 * scale * scale));
                
                // 三线性插值到描述子中
                addToDescriptor(descriptor, grid_x, grid_y, angle, mag * weight);
            }
        }
        
        // 归一化描述子
        float norm = 0.0f;
        for (float val : descriptor) {
            norm += val * val;
        }
        norm = std::sqrt(norm);
        
        if (norm > 0) {
            for (float& val : descriptor) {
                val /= norm;
                val = std::min(val, 0.2f);  // 截断大值
            }
            
            // 重新归一化
            norm = 0.0f;
            for (float val : descriptor) {
                norm += val * val;
            }
            norm = std::sqrt(norm);
            
            if (norm > 0) {
                for (float& val : descriptor) {
                    val /= norm;
                }
            }
        }
        
        kp.descriptor = std::move(descriptor);
    }
    
    void addToDescriptor(std::vector<float>& descriptor, float x, float y, float angle, float weight) {
        int x0 = static_cast<int>(std::floor(x));
        int y0 = static_cast<int>(std::floor(y));
        
        float dx = x - x0;
        float dy = y - y0;
        
        // 角度bin
        float angle_bin = angle * 8.0f / (2 * M_PI);
        int angle_bin0 = static_cast<int>(std::floor(angle_bin)) % 8;
        int angle_bin1 = (angle_bin0 + 1) % 8;
        float d_angle = angle_bin - std::floor(angle_bin);
        
        // 三线性插值
        for (int dy_idx = 0; dy_idx <= 1; ++dy_idx) {
            int y_idx = y0 + dy_idx;
            if (y_idx < 0 || y_idx >= SIFT_DESCRIPTOR_WIDTH) continue;
            
            float wy = (dy_idx == 0) ? (1 - dy) : dy;
            
            for (int dx_idx = 0; dx_idx <= 1; ++dx_idx) {
                int x_idx = x0 + dx_idx;
                if (x_idx < 0 || x_idx >= SIFT_DESCRIPTOR_WIDTH) continue;
                
                float wx = (dx_idx == 0) ? (1 - dx) : dx;
                
                int base_idx = (y_idx * SIFT_DESCRIPTOR_WIDTH + x_idx) * 8;
                
                descriptor[base_idx + angle_bin0] += weight * wx * wy * (1 - d_angle);
                descriptor[base_idx + angle_bin1] += weight * wx * wy * d_angle;
            }
        }
    }
};

// --- 优化的SIFT匹配器 ---
class OptimizedSIFTMatcher {
public:
    static std::vector<SIFTMatch> match(const std::vector<SIFTKeyPoint>& kp1, 
                                       const std::vector<SIFTKeyPoint>& kp2,
                                       float ratio_threshold = 0.7f) {  // 放宽ratio test
        
        std::cout << "Starting SIFT matching with " << kp1.size() << " and " << kp2.size() << " keypoints..." << std::endl;
        
        // 第一步：基础匹配
        auto candidates = computeBasicMatches(kp1, kp2, ratio_threshold);
        std::cout << "Step 1 - Basic matching: " << candidates.size() << " matches" << std::endl;
        
        // 第二步：交叉验证
        auto cross_checked = crossValidation(candidates, kp1, kp2, ratio_threshold);
        std::cout << "Step 2 - Cross validation: " << cross_checked.size() << " matches" << std::endl;
        
        // 第三步：几何约束过滤
        auto geo_filtered = geometricFilter(cross_checked, kp1, kp2);
        std::cout << "Step 3 - Geometric filtering: " << geo_filtered.size() << " matches" << std::endl;
        
        // 如果匹配数量足够，进行RANSAC；否则跳过RANSAC
        if (geo_filtered.size() >= 25) {  // 提高RANSAC的最小要求
            auto ransac_filtered = ransacFilter(geo_filtered, kp1, kp2);
            std::cout << "Step 4 - RANSAC filtering: " << ransac_filtered.size() << " matches" << std::endl;
            return ransac_filtered;
        } else {
            std::cout << "Step 4 - RANSAC skipped (using geometric filtering results: " << geo_filtered.size() << " matches)" << std::endl;
            return geo_filtered;
        }
    }
    
private:
    static std::vector<SIFTMatch> computeBasicMatches(const std::vector<SIFTKeyPoint>& kp1,
                                                     const std::vector<SIFTKeyPoint>& kp2,
                                                     float ratio_threshold) {
        std::vector<SIFTMatch> matches;
        
        for (size_t i = 0; i < kp1.size(); ++i) {
            float best_dist = std::numeric_limits<float>::max();
            float second_best_dist = std::numeric_limits<float>::max();
            int best_idx = -1;
            
            for (size_t j = 0; j < kp2.size(); ++j) {
                float dist = computeDistance(kp1[i].descriptor, kp2[j].descriptor);
                
                if (dist < best_dist) {
                    second_best_dist = best_dist;
                    best_dist = dist;
                    best_idx = static_cast<int>(j);
                } else if (dist < second_best_dist) {
                    second_best_dist = dist;
                }
            }
            
            // 放宽筛选条件
            if (best_idx >= 0 && 
                best_dist < 0.4f &&  // 放宽绝对距离阈值
                second_best_dist > 0 && 
                best_dist / second_best_dist < ratio_threshold) {
                matches.push_back({static_cast<int>(i), best_idx, best_dist});
            }
        }
        
        return matches;
    }
    
    static std::vector<SIFTMatch> crossValidation(const std::vector<SIFTMatch>& forward_matches,
                                                 const std::vector<SIFTKeyPoint>& kp1,
                                                 const std::vector<SIFTKeyPoint>& kp2,
                                                 float ratio_threshold) {
        // 反向匹配
        auto backward_matches = computeBasicMatches(kp2, kp1, ratio_threshold);
        
        // 交叉验证
        std::vector<SIFTMatch> cross_checked;
        for (const auto& fm : forward_matches) {
            for (const auto& bm : backward_matches) {
                if (fm.idx1 == bm.idx2 && fm.idx2 == bm.idx1) {
                    cross_checked.push_back(fm);
                    break;
                }
            }
        }
        
        return cross_checked;
    }
    
    static std::vector<SIFTMatch> geometricFilter(const std::vector<SIFTMatch>& matches,
                                                 const std::vector<SIFTKeyPoint>& kp1,
                                                 const std::vector<SIFTKeyPoint>& kp2) {
        if (matches.size() < 5) return matches;  // 降低最小匹配要求
        
        std::vector<SIFTMatch> filtered;
        
        // 计算主要的视差方向和分布
        std::vector<float> disparities, y_diffs;
        for (const auto& match : matches) {
            const auto& p1 = kp1[match.idx1];
            const auto& p2 = kp2[match.idx2];
            
            float disparity = p1.x - p2.x;
            float y_diff = std::abs(p1.y - p2.y);
            
            disparities.push_back(disparity);
            y_diffs.push_back(y_diff);
        }
        
        // 计算统计量
        std::sort(disparities.begin(), disparities.end());
        std::sort(y_diffs.begin(), y_diffs.end());
        
        float median_disparity = disparities[disparities.size() / 2];
        float median_y_diff = y_diffs[y_diffs.size() / 2];
        
        // 计算MAD (Median Absolute Deviation)
        std::vector<float> disp_deviations, y_deviations;
        for (float d : disparities) {
            disp_deviations.push_back(std::abs(d - median_disparity));
        }
        for (float y : y_diffs) {
            y_deviations.push_back(std::abs(y - median_y_diff));
        }
        
        std::sort(disp_deviations.begin(), disp_deviations.end());
        std::sort(y_deviations.begin(), y_deviations.end());
        
        float mad_disparity = disp_deviations[disp_deviations.size() / 2];
        float mad_y = y_deviations[y_deviations.size() / 2];
        
        // 放宽过滤条件
        for (const auto& match : matches) {
            const auto& p1 = kp1[match.idx1];
            const auto& p2 = kp2[match.idx2];
            
            float disparity = p1.x - p2.x;
            float y_diff = std::abs(p1.y - p2.y);
            
            // 放宽立体几何约束
            bool valid_disparity = disparity > -10 &&  // 允许少量负视差
                                  std::abs(disparity - median_disparity) < 5 * mad_disparity;  // 放宽到5倍MAD
            
            bool valid_y = y_diff < std::max(5.0f, 5 * mad_y);  // 放宽y坐标约束
            
            // 放宽尺度一致性检查
            bool valid_scale = std::abs(std::log(p1.scale / p2.scale)) < 1.0f;  // 放宽尺度约束
            
            if (valid_disparity && valid_y && valid_scale) {
                filtered.push_back(match);
            }
        }
        
        return filtered;
    }
    
    static std::vector<SIFTMatch> ransacFilter(const std::vector<SIFTMatch>& matches,
                                              const std::vector<SIFTKeyPoint>& kp1,
                                              const std::vector<SIFTKeyPoint>& kp2) {
        if (matches.size() < 8) return matches;
        
        std::mt19937 rng(std::random_device{}());
        std::uniform_int_distribution<int> dist(0, matches.size() - 1);
        
        std::vector<SIFTMatch> best_inliers;
        int best_inlier_count = 0;
        const int max_iterations = 500;  // 减少迭代次数
        const float threshold = 5.0f;    // 放宽阈值
        
        for (int iter = 0; iter < max_iterations; ++iter) {
            // 随机选择8个点估计基本矩阵
            std::vector<int> sample_indices;
            std::set<int> used_indices;
            
            while (sample_indices.size() < 8) {
                int idx = dist(rng);
                if (used_indices.find(idx) == used_indices.end()) {
                    sample_indices.push_back(idx);
                    used_indices.insert(idx);
                }
            }
            
            // 简化的一致性检查（基于局部邻域一致性）
            std::vector<SIFTMatch> current_inliers;
            
            for (const auto& match : matches) {
                const auto& p1 = kp1[match.idx1];
                const auto& p2 = kp2[match.idx2];
                
                int consistent_neighbors = 0;
                
                // 检查与采样点的一致性
                for (int sample_idx : sample_indices) {
                    const auto& sample_match = matches[sample_idx];
                    const auto& sp1 = kp1[sample_match.idx1];
                    const auto& sp2 = kp2[sample_match.idx2];
                    
                    // 计算相对位移
                    float dx1 = p1.x - sp1.x;
                    float dy1 = p1.y - sp1.y;
                    float dx2 = p2.x - sp2.x;
                    float dy2 = p2.y - sp2.y;
                    
                    // 检查相对位移的一致性
                    float diff_x = std::abs(dx1 - dx2);
                    float diff_y = std::abs(dy1 - dy2);
                    
                    if (diff_x < threshold * 10 && diff_y < threshold * 5) {  // 放宽阈值
                        consistent_neighbors++;
                    }
                }
                
                // 降低一致性要求
                if (consistent_neighbors >= 4) {  // 从6降到4
                    current_inliers.push_back(match);
                }
            }
            
            if (current_inliers.size() > best_inlier_count) {
                best_inlier_count = current_inliers.size();
                best_inliers = current_inliers;
            }
        }
        
        // 如果RANSAC结果太少，返回原始匹配
        if (best_inliers.size() < matches.size() * 0.3f) {  // 至少保留30%
            std::cout << "RANSAC filtering too aggressive, returning geometric filter results" << std::endl;
            return matches;
        }
        
        return best_inliers;
    }
    
    static float computeDistance(const std::vector<float>& desc1, const std::vector<float>& desc2) {
        float dist = 0.0f;
        for (size_t i = 0; i < desc1.size(); ++i) {
            float diff = desc1[i] - desc2[i];
            dist += diff * diff;
        }
        return std::sqrt(dist);
    }
};

// --- 优化的SIFT检测器 ---
class OptimizedSIFTDetector {
private:
    SIFTDetector base_detector;
    
public:
    OptimizedSIFTDetector() : base_detector(4, 3, 1.6f, 0.06f, 12.0f) {}  // 稍微放宽参数
    
    std::vector<SIFTKeyPoint> detect(const MyImage& image) {
        auto keypoints = base_detector.detect(image);
        
        // 额外的质量过滤
        std::vector<SIFTKeyPoint> filtered_keypoints;
        
        // 按响应强度排序
        std::sort(keypoints.begin(), keypoints.end(),
                 [](const SIFTKeyPoint& a, const SIFTKeyPoint& b) {
                     return a.response > b.response;
                 });
        
        // 放宽特征点数量和距离限制
        const int max_keypoints = 500;  // 增加特征点数量
        const float min_distance = 6.0f;  // 减少最小距离
        
        for (const auto& kp : keypoints) {
            if (filtered_keypoints.size() >= max_keypoints) break;
            
            bool too_close = false;
            for (const auto& existing : filtered_keypoints) {
                float dx = kp.x - existing.x;
                float dy = kp.y - existing.y;
                if (dx*dx + dy*dy < min_distance*min_distance) {
                    too_close = true;
                    break;
                }
            }
            
            if (!too_close) {
                filtered_keypoints.push_back(kp);
            }
        }
        
        return filtered_keypoints;
    }
};

// --- 最优8点选择器 ---
class Best8PointSelector {
public:
    static std::vector<SIFTMatch> selectBest8Points(const std::vector<SIFTMatch>& matches,
                                                    const std::vector<SIFTKeyPoint>& kp1,
                                                    const std::vector<SIFTKeyPoint>& kp2) {
        if (matches.size() <= 8) {
            std::cout << "Using all " << matches.size() << " matches for 8-point algorithm" << std::endl;
            return matches;
        }
        
        std::cout << "Selecting best 8 points from " << matches.size() << " matches..." << std::endl;
        
        // 方法1：简单质量选择 - 按匹配距离排序
        auto quality_selected = selectByQuality(matches, kp1, kp2);
        
        // 方法2：分布式选择 - 确保点的分布均匀
        auto distributed_selected = selectByDistribution(matches, kp1, kp2);
        
        // 选择更好的结果（根据覆盖面积判断）
        float area1 = computeCoverageArea(quality_selected, kp1);
        float area2 = computeCoverageArea(distributed_selected, kp1);
        
        if (area2 > area1 * 0.8f) {  // 如果分布式选择的覆盖面积不差太多，优先选择
            std::cout << "Selected 8 points using distribution-based method (coverage: " << area2 << ")" << std::endl;
            return distributed_selected;
        } else {
            std::cout << "Selected 8 points using quality-based method (coverage: " << area1 << ")" << std::endl;
            return quality_selected;
        }
    }
    
private:
    // 按质量选择：匹配距离 + 特征强度
    static std::vector<SIFTMatch> selectByQuality(const std::vector<SIFTMatch>& matches,
                                                  const std::vector<SIFTKeyPoint>& kp1,
                                                  const std::vector<SIFTKeyPoint>& kp2) {
        // 计算综合质量分数
        std::vector<std::pair<float, int>> quality_scores;
        
        for (size_t i = 0; i < matches.size(); ++i) {
            const auto& match = matches[i];
            const auto& p1 = kp1[match.idx1];
            const auto& p2 = kp2[match.idx2];
            
            // 综合质量分数：匹配距离（越小越好）+ 特征响应强度（越大越好）
            float quality = -match.distance + 0.1f * (p1.response + p2.response);
            quality_scores.push_back({quality, static_cast<int>(i)});
        }
        
        // 按质量排序
        std::sort(quality_scores.begin(), quality_scores.end(), 
                 [](const auto& a, const auto& b) { return a.first > b.first; });
        
        // 选择前8个
        std::vector<SIFTMatch> selected;
        for (int i = 0; i < 8; ++i) {
            selected.push_back(matches[quality_scores[i].second]);
        }
        
        return selected;
    }
    
    // 按分布选择：确保点分布均匀
    static std::vector<SIFTMatch> selectByDistribution(const std::vector<SIFTMatch>& matches,
                                                       const std::vector<SIFTKeyPoint>& kp1,
                                                       const std::vector<SIFTKeyPoint>& kp2) {
        std::vector<SIFTMatch> selected;
        std::vector<bool> used(matches.size(), false);
        
        // 首先选择质量最好的点
        std::vector<std::pair<float, int>> quality_scores;
        for (size_t i = 0; i < matches.size(); ++i) {
            const auto& match = matches[i];
            const auto& p1 = kp1[match.idx1];
            const auto& p2 = kp2[match.idx2];
            float quality = -match.distance + 0.1f * (p1.response + p2.response);
            quality_scores.push_back({quality, static_cast<int>(i)});
        }
        
        std::sort(quality_scores.begin(), quality_scores.end(), 
                 [](const auto& a, const auto& b) { return a.first > b.first; });
        
        // 选择第一个点（质量最好的）
        int first_idx = quality_scores[0].second;
        selected.push_back(matches[first_idx]);
        used[first_idx] = true;
        
        // 迭代选择剩余7个点，每次选择与已选点距离最远的高质量点
        for (int round = 1; round < 8; ++round) {
            int best_idx = -1;
            float best_score = -1;
            
            for (const auto& [quality, idx] : quality_scores) {
                if (used[idx]) continue;
                
                const auto& p1 = kp1[matches[idx].idx1];
                
                // 计算与已选点的最小距离
                float min_dist = std::numeric_limits<float>::max();
                for (const auto& selected_match : selected) {
                    const auto& sp1 = kp1[selected_match.idx1];
                    float dist = std::sqrt((p1.x - sp1.x) * (p1.x - sp1.x) + 
                                         (p1.y - sp1.y) * (p1.y - sp1.y));
                    min_dist = std::min(min_dist, dist);
                }
                
                // 综合分数：距离权重 * 质量权重
                float score = min_dist * 0.7f + quality * 0.3f;
                if (score > best_score) {
                    best_score = score;
                    best_idx = idx;
                }
            }
            
            if (best_idx >= 0) {
                selected.push_back(matches[best_idx]);
                used[best_idx] = true;
            }
        }
        
        return selected;
    }
    
    // 计算点集的覆盖面积（用于评估分布质量）
    static float computeCoverageArea(const std::vector<SIFTMatch>& matches,
                                    const std::vector<SIFTKeyPoint>& kp1) {
        if (matches.size() < 3) return 0.0f;
        
        // 计算包围盒面积
        float min_x = std::numeric_limits<float>::max();
        float max_x = std::numeric_limits<float>::lowest();
        float min_y = std::numeric_limits<float>::max();
        float max_y = std::numeric_limits<float>::lowest();
        
        for (const auto& match : matches) {
            const auto& p = kp1[match.idx1];
            min_x = std::min(min_x, p.x);
            max_x = std::max(max_x, p.x);
            min_y = std::min(min_y, p.y);
            max_y = std::max(max_y, p.y);
        }
        
        return (max_x - min_x) * (max_y - min_y);
    }
};

// --- 转换函数 ---
std::vector<KeyPoint> convertSIFTKeyPoints(const std::vector<SIFTKeyPoint>& sift_kps) {
    std::vector<KeyPoint> kps;
    kps.reserve(sift_kps.size());
    for (const auto& sift_kp : sift_kps) {
        kps.emplace_back(sift_kp);
    }
    return kps;
}

std::vector<Match> convertSIFTMatches(const std::vector<SIFTMatch>& sift_matches) {
    std::vector<Match> matches;
    matches.reserve(sift_matches.size());
    for (const auto& sift_match : sift_matches) {
        matches.emplace_back(sift_match);
    }
    return matches;
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
    
    // 对于8点算法选择的点，使用特殊标记
    bool is_best_8 = (matches.size() == 8);
    
    for (size_t i = 0; i < matches.size(); ++i) {
        const auto& m = matches[i];
        const int x1 = kp1[m.idx1].x;
        const int y1 = kp1[m.idx1].y;
        const int x2 = kp2[m.idx2].x + img1.width;
        const int y2 = kp2[m.idx2].y;
        
        unsigned char r, g, b;
        
        if (is_best_8) {
            // 为8个最佳点使用醒目的颜色
            switch (i % 4) {
                case 0: r = 255; g = 0; b = 0; break;      // 红色
                case 1: r = 0; g = 255; b = 0; break;      // 绿色
                case 2: r = 0; g = 0; b = 255; break;      // 蓝色
                case 3: r = 255; g = 255; b = 0; break;    // 黄色
            }
        } else {
            // 根据匹配质量着色
            float min_dist = 0.1f;  // 假设的最小距离
            float max_dist = 0.5f;  // 假设的最大距离
            float quality = 1.0f - (m.dist - min_dist) / (max_dist - min_dist + 1e-6f);
            quality = std::clamp(quality, 0.0f, 1.0f);
            
            r = static_cast<unsigned char>(255 * (1 - quality));
            g = static_cast<unsigned char>(255 * quality);
            b = 0;
        }
        
        // 绘制关键点（最佳8点使用更大的标记）
        int radius = is_best_8 ? 8 : std::max(2, std::min(6, static_cast<int>(kp1[m.idx1].scale)));
        
        for (int dy = -radius; dy <= radius; ++dy) {
            for (int dx = -radius; dx <= radius; ++dx) {
                if (dx*dx + dy*dy <= radius*radius) {
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
        }
        
        // 绘制连线（最佳8点使用更粗的线）
        int line_thickness = is_best_8 ? 2 : 1;
        
        const int dx = std::abs(x2 - x1), dy = std::abs(y2 - y1);
        const int sx = x1 < x2 ? 1 : -1, sy = y1 < y2 ? 1 : -1;
        int err = dx - dy, cx = x1, cy = y1;
        
        while (true) {
            // 绘制粗线
            for (int ldy = -line_thickness; ldy <= line_thickness; ++ldy) {
                for (int ldx = -line_thickness; ldx <= line_thickness; ++ldx) {
                    int lx = cx + ldx, ly = cy + ldy;
                    if (lx >= 0 && lx < w && ly >= 0 && ly < h) {
                        outImg.data[(ly * w + lx) * 3] = r;
                        outImg.data[(ly * w + lx) * 3 + 1] = g;
                        outImg.data[(ly * w + lx) * 3 + 2] = b;
                    }
                }
            }
            
            if (cx == x2 && cy == y2) break;
            
            const int e2 = 2 * err;
            if (e2 > -dy) { err -= dy; cx += sx; }
            if (e2 < dx) { err += dx; cy += sy; }
        }
    }
}

// --- 主要接口 ---
std::pair<std::vector<KeyPoint>, std::vector<Match>> 
detectAndMatch(const MyImage& img1, const MyImage& img2) {
    OptimizedSIFTDetector detector;  // 使用优化版本
    
    // 检测SIFT特征
    auto sift_kp1 = detector.detect(img1);
    auto sift_kp2 = detector.detect(img2);
    
    if (sift_kp1.empty() || sift_kp2.empty()) {
        return {{}, {}};
    }
    
    // 使用优化的匹配器
    auto sift_matches = OptimizedSIFTMatcher::match(sift_kp1, sift_kp2, 0.7f);
    
    // 如果找到足够多的匹配，选择最好的8个用于8点算法
    if (sift_matches.size() > 8) {
        auto best_8_matches = Best8PointSelector::selectBest8Points(sift_matches, sift_kp1, sift_kp2);
        sift_matches = best_8_matches;
    }
    
    // 转换为兼容类型
    auto kp1 = convertSIFTKeyPoints(sift_kp1);
    auto kp2 = convertSIFTKeyPoints(sift_kp2);
    auto matches = convertSIFTMatches(sift_matches);
    
    // 合并关键点
    std::vector<KeyPoint> all_kp = kp1;
    all_kp.insert(all_kp.end(), kp2.begin(), kp2.end());
    
    return {all_kp, matches};
}

void run_feature_matching(const std::string& leftPath, const std::string& rightPath, const std::string& outPath) {
    MyImage imgL, imgR;
    
    if (!myImReadPNG(leftPath, imgL) || !myImReadPNG(rightPath, imgR)) {
        std::cout << "Failed to read input images: " << leftPath << ", " << rightPath << std::endl;
        return;
    }
    
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
        }
        imgL = std::move(grayL);
    }
    
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
        }
        imgR = std::move(grayR);
    }
    
    // 使用SIFT检测和匹配
    auto [all_kp, matches] = detectAndMatch(imgL, imgR);
    
    // 分离关键点
    size_t split_point = all_kp.size() / 2;
    std::vector<KeyPoint> kpL(all_kp.begin(), all_kp.begin() + split_point);
    std::vector<KeyPoint> kpR(all_kp.begin() + split_point, all_kp.end());
    
    std::cout << "Optimized SIFT: Detected " << kpL.size() << " and " << kpR.size() << " keypoints." << std::endl;
    std::cout << "Optimized SIFT: Final matched " << matches.size() << " pairs for 8-point algorithm." << std::endl;
    
    if (matches.size() >= 8) {
        std::cout << "✓ Sufficient matches for 8-point algorithm!" << std::endl;
    } else if (matches.size() > 0) {
        std::cout << "⚠ Only " << matches.size() << " matches found (need at least 8 for robust estimation)" << std::endl;
    }
    
    if (matches.size() > 0) {
        // 计算匹配质量统计
        float avg_distance = 0.0f;
        float min_distance = std::numeric_limits<float>::max();
        float max_distance = 0.0f;
        
        for (const auto& match : matches) {
            avg_distance += match.dist;
            min_distance = std::min(min_distance, match.dist);
            max_distance = std::max(max_distance, match.dist);
        }
        avg_distance /= matches.size();
        
        std::cout << "Match quality - Avg: " << avg_distance 
                  << ", Min: " << min_distance 
                  << ", Max: " << max_distance << std::endl;
    }
    
    // 可视化和保存
    MyImage matchImg;
    drawMatches(imgL, kpL, imgR, kpR, matches, matchImg);
    myImWritePNG(outPath, matchImg);
    std::cout << "Saved " << outPath << std::endl;
}

// 辅助函数
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
    
    OptimizedSIFTDetector detector;  // 使用优化版本
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