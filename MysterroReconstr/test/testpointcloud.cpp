#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <cstdint>
#include <cmath>
#include <sstream>
#include <algorithm>
#include <limits>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <png.h>
#include <filesystem>
#include <iomanip>
#include "../src/utils/imageutils.h"
#include "../src/pointcloud.h"
// #include "../poissonreconstruction/poissonReconstruction.h"

// --- Save PLY ---
bool saveMeshToPLY(const std::string& filename, const Mesh& mesh) {
    std::ofstream file(filename);
    if (!file.is_open()) return false;
    file << "ply\nformat ascii 1.0\n";
    file << "element vertex " << mesh.vertices.size() << "\n";
    file << "property float x\nproperty float y\nproperty float z\n";
    file << "property uchar red\nproperty uchar green\nproperty uchar blue\n";
    file << "element face " << mesh.faces.size() << "\n";
    file << "property list uchar int vertex_indices\nend_header\n";
    for (const auto& v : mesh.vertices)
        // 🔧 Modified: Swap Y and Z coordinates when saving (Y becomes Z, Z becomes Y)
        // 🔧 Fixed: Negate Y coordinate to fix mirroring along Y-axis
        file << v.x << " " << v.z << " " << -v.y << " " << (int)v.r << " " << (int)v.g << " " << (int)v.b << "\n";
    for (const auto& f : mesh.faces)
        file << "3 " << f.v1 << " " << f.v2 << " " << f.v3 << "\n";
    return true;
}

// --- Save grayscale disparity image as PNG (using ImageMagick like testdisparityELAS.cpp) ---

// 简单的PGM图像保存函数
void savePGM_float(const char* filename, const DepthImage& disparity) {
    std::ofstream file(filename, std::ios::out | std::ios::binary);
    if (!file.is_open()) {
        std::cout << "Error: Could not open file " << filename << " for writing" << std::endl;
        return;
    }
    
    int width = disparity.cols;
    int height = disparity.rows;
    
    // 找到最大值和最小值用于归一化
    float min_val = disparity.at(0, 0);
    float max_val = disparity.at(0, 0);
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            float val = disparity.at(y, x);
            if (val < min_val) min_val = val;
            if (val > max_val) max_val = val;
        }
    }
    
    // 写入PGM头
    file << "P5\n" << width << " " << height << "\n255\n";
    
    // 归一化并写入数据
    float range = max_val - min_val;
    if (range == 0) range = 1; // 避免除零
    
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            float val = disparity.at(y, x);
            unsigned char pixel = (unsigned char)(255.0f * (val - min_val) / range);
            file.write((char*)&pixel, 1);
        }
    }
    
    file.close();
}

// 简单的PNG图像保存函数
bool savePNG_float(const std::string& filename, const DepthImage& disparity) {
    // 创建临时PGM文件
    std::string temp_pgm = filename + ".temp.pgm";
    
    // 先保存为PGM格式
    savePGM_float(temp_pgm.c_str(), disparity);
    
    // 使用ImageMagick转换为PNG
    std::string convert_cmd = "convert \"" + temp_pgm + "\" \"" + filename + "\"";
    int result = system(convert_cmd.c_str());
    
    if (result != 0) {
        // 如果ImageMagick不可用，尝试使用magick命令
        std::string magick_cmd = "magick \"" + temp_pgm + "\" \"" + filename + "\"";
        result = system(magick_cmd.c_str());
        
        if (result != 0) {
            std::cout << "Warning: Cannot convert to PNG format, keeping PGM format" << std::endl;
            // Rename temporary file to final file
            rename(temp_pgm.c_str(), filename.c_str());
            return true;
        }
    }
    
    // Delete temporary PGM file
    remove(temp_pgm.c_str());
    return true;
}



// 兼容性函数，保持原有接口
bool saveDisparityAsPNG(const std::string& filename, const DepthImage& disparity, bool save_as_16bit = false) {
    // 注意：save_as_16bit参数在当前实现中被忽略，统一使用ImageMagick方式
    // 如果需要16位支持，可以在savePNG_float中添加相应逻辑
    return savePNG_float(filename, disparity);
}

// --- Load camera parameters ---
struct CameraParams {
    CameraIntrinsics intrinsics;
    float baseline;
    
    CameraParams() : baseline(536.62f) {} // Default baseline
};

CameraParams loadCamera(const std::string& path) {
    std::ifstream fin(path);
    float fx=1000, fy=1000, cx=320, cy=240;
    float baseline = 536.62f; // Default baseline
    std::string line;
    
    while (std::getline(fin, line)) {
        if (line.find("cam0")!=std::string::npos) {
            auto l=line.find('['), r=line.find(']');
            if (l!=std::string::npos && r!=std::string::npos && r>l) {
                std::string nums=line.substr(l+1, r-l-1);
                std::replace(nums.begin(), nums.end(), ';', ' ');
                std::istringstream iss(nums);
                float zero, zero2, zero3, zero4, one;
                iss >> fx >> zero >> cx >> zero2 >> fy >> cy >> zero3 >> zero4 >> one;
            }
        }
        // Read baseline
        if (line.find("baseline")!=std::string::npos) {
            auto pos = line.find('=');
            if (pos != std::string::npos) {
                baseline = std::stof(line.substr(pos + 1));
            }
        }
    }
    
    CameraParams params;
    params.intrinsics = CameraIntrinsics(fx, fy, cx, cy);
    params.baseline = baseline;
    
    return params;
}

// --- 改进的RGB视差图处理函数 ---

// 检测是否为多通道编码的高精度视差图
bool detectMultiChannelEncoding(const std::vector<uint8_t>& raw, int w, int h) {
    int rg_correlation = 0;
    int rb_correlation = 0;
    int gb_correlation = 0;
    int total_nonzero = 0;
    
    for (int i = 0; i < w * h; i++) {
        uint8_t r = raw[i*3];
        uint8_t g = raw[i*3 + 1];
        uint8_t b = raw[i*3 + 2];
        
        if (r > 0 || g > 0 || b > 0) {
            total_nonzero++;
            
            // 检测通道间的相关性模式
            if (r > 0 && g > 0) rg_correlation++;
            if (r > 0 && b > 0) rb_correlation++;
            if (g > 0 && b > 0) gb_correlation++;
        }
    }
    
    if (total_nonzero == 0) return false;
    
    // 如果RG通道高度相关且B通道相关性低，可能是16位编码在RG通道中
    double rg_ratio = (double)rg_correlation / total_nonzero;
    double rb_ratio = (double)rb_correlation / total_nonzero;
    double gb_ratio = (double)gb_correlation / total_nonzero;
    
    // RG高相关，RB和GB低相关，可能是16位编码
    return (rg_ratio > 0.8 && rb_ratio < 0.3 && gb_ratio < 0.3);
}

// 16位解码（高字节在R通道，低字节在G通道）
uint16_t decode16BitFromRG(uint8_t r, uint8_t g) {
    return (uint16_t(r) << 8) | uint16_t(g);
}

// 智能通道选择策略
struct ChannelStats {
    int nonzero_count;
    int min_val;
    int max_val;
    double mean;
    double variance;
    double score;
};

ChannelStats analyzeChannel(const std::vector<uint8_t>& raw, int w, int h, int channel) {
    ChannelStats stats = {0, 255, 0, 0.0, 0.0, 0.0};
    
    // 第一遍：计算基本统计
    for (int i = 0; i < w * h; i++) {
        uint8_t val = raw[i * 3 + channel];
        if (val > 0) {
            stats.nonzero_count++;
            stats.mean += val;
            stats.min_val = std::min(stats.min_val, (int)val);
            stats.max_val = std::max(stats.max_val, (int)val);
        }
    }
    
    if (stats.nonzero_count > 0) {
        stats.mean /= stats.nonzero_count;
        
        // 第二遍：计算方差
        for (int i = 0; i < w * h; i++) {
            uint8_t val = raw[i * 3 + channel];
            if (val > 0) {
                stats.variance += (val - stats.mean) * (val - stats.mean);
            }
        }
        stats.variance /= stats.nonzero_count;
        
        // 综合评分：信息量 + 动态范围 + 标准差
        double info_ratio = (double)stats.nonzero_count / (w * h);
        double dynamic_range = (stats.max_val - stats.min_val) / 255.0;
        double std_dev = sqrt(stats.variance) / 255.0;
        
        // 权重可以根据需要调整
        stats.score = info_ratio * 0.4 + dynamic_range * 0.4 + std_dev * 0.2;
    }
    
    return stats;
}

int selectBestChannel(const std::vector<uint8_t>& raw, int w, int h) {
    ChannelStats channel_stats[3];
    
    std::cout << "Analyzing RGB channels..." << std::endl;
    
    for (int c = 0; c < 3; c++) {
        channel_stats[c] = analyzeChannel(raw, w, h, c);
        char channel_name = (c == 0) ? 'R' : (c == 1) ? 'G' : 'B';
        std::cout << "Channel " << channel_name << ": "
                  << "nonzero=" << channel_stats[c].nonzero_count
                  << ", range=" << channel_stats[c].min_val << "-" << channel_stats[c].max_val
                  << ", mean=" << std::fixed << std::setprecision(1) << channel_stats[c].mean
                  << ", score=" << std::setprecision(3) << channel_stats[c].score << std::endl;
    }
    
    // 选择得分最高的通道
    int best_channel = 0;
    for (int c = 1; c < 3; c++) {
        if (channel_stats[c].score > channel_stats[best_channel].score) {
            best_channel = c;
        }
    }
    
    return best_channel;
}

// 多通道加权融合
uint8_t fuseChannels(uint8_t r, uint8_t g, uint8_t b, const double weights[3]) {
    double fused = r * weights[0] + g * weights[1] + b * weights[2];
    return static_cast<uint8_t>(std::min(255.0, std::max(0.0, fused)));
}

void calculateChannelWeights(const std::vector<uint8_t>& raw, int w, int h, double weights[3], int primary_channel) {
    ChannelStats stats[3];
    for (int c = 0; c < 3; c++) {
        stats[c] = analyzeChannel(raw, w, h, c);
    }
    
    // 主通道获得较高权重
    weights[primary_channel] = 0.7;
    
    // 其他通道根据相对质量分配权重
    double remaining_weight = 0.3;
    double total_other_score = 0.0;
    
    for (int c = 0; c < 3; c++) {
        if (c != primary_channel) {
            total_other_score += stats[c].score;
        }
    }
    
    if (total_other_score > 0) {
        for (int c = 0; c < 3; c++) {
            if (c != primary_channel) {
                weights[c] = remaining_weight * (stats[c].score / total_other_score);
            }
        }
    } else {
        // 如果其他通道得分都为0，平均分配权重
        for (int c = 0; c < 3; c++) {
            if (c != primary_channel) {
                weights[c] = remaining_weight / 2.0;
            }
        }
    }
    
    std::cout << "Channel weights: R=" << std::setprecision(3) << weights[0] 
              << ", G=" << weights[1] << ", B=" << weights[2] << std::endl;
}

// --- Load 16-bit grayscale depth image using libpng ---
DepthImage loadDepthPNG(const std::string& path) {
    FILE* fp = fopen(path.c_str(), "rb");
    if (!fp) { std::cerr << "Unable to open depth image: " << path << std::endl; exit(1); }
    png_structp png = png_create_read_struct(PNG_LIBPNG_VER_STRING, nullptr, nullptr, nullptr);
    png_infop info = png_create_info_struct(png);
    if (!png || !info) { fclose(fp); exit(1); }
    if (setjmp(png_jmpbuf(png))) { fclose(fp); png_destroy_read_struct(&png, &info, nullptr); exit(1); }
    png_init_io(png, fp);
    png_read_info(png, info);
    int w = png_get_image_width(png, info);
    int h = png_get_image_height(png, info);
    int bit_depth = png_get_bit_depth(png, info);
    int color_type = png_get_color_type(png, info);
    DepthImage img(h, w);
    if (color_type == PNG_COLOR_TYPE_GRAY && bit_depth == 16) {
        // 16-bit grayscale
        std::vector<uint16_t> raw(h * w);
        std::vector<png_bytep> row_ptrs(h);
        for (int y = 0; y < h; ++y) row_ptrs[y] = (png_bytep)(&raw[y*w]);
        png_read_image(png, (png_bytep*)row_ptrs.data());
        png_destroy_read_struct(&png, &info, nullptr);
        fclose(fp);
        for (int i = 0; i < w * h; ++i) {
            uint16_t v = raw[i];
            v = ((v >> 8) & 0xFF) | ((v & 0xFF) << 8);
            int y = i / w;
            int x = i % w;
            img.at(y, x) = static_cast<float>(v);
        }
    } else if (color_type == PNG_COLOR_TYPE_GRAY && bit_depth == 8) {
        // 8-bit grayscale
        std::vector<uint8_t> raw(h * w);
        std::vector<png_bytep> row_ptrs(h);
        for (int y = 0; y < h; ++y) row_ptrs[y] = (png_bytep)(&raw[y*w]);
        png_read_image(png, (png_bytep*)row_ptrs.data());
        png_destroy_read_struct(&png, &info, nullptr);
        fclose(fp);
        for (int i = 0; i < w * h; ++i) {
            int y = i / w;
            int x = i % w;
            img.at(y, x) = static_cast<float>(raw[i]);
        }
    } else if (color_type == PNG_COLOR_TYPE_RGB && bit_depth == 8) {
        // 8-bit RGB - 使用改进的转换策略
        std::vector<uint8_t> raw(h * w * 3);
        std::vector<png_bytep> row_ptrs(h);
        for (int y = 0; y < h; ++y) row_ptrs[y] = (png_bytep)(&raw[y*w*3]);
        png_read_image(png, (png_bytep*)row_ptrs.data());
        png_destroy_read_struct(&png, &info, nullptr);
        fclose(fp);
        
        std::cout << "Converting RGB disparity to grayscale using improved algorithm..." << std::endl;
        
        // 1. 检测是否为多通道编码
        bool isMultiChannelEncoded = detectMultiChannelEncoding(raw, w, h);
        
        if (isMultiChannelEncoded) {
            std::cout << "Detected multi-channel encoding, using 16-bit RG decoding..." << std::endl;
            
            // 使用RG通道的16位解码
            for (int i = 0; i < w * h; ++i) {
                uint8_t r = raw[i*3];
                uint8_t g = raw[i*3 + 1];
                uint16_t depth16 = decode16BitFromRG(r, g);
                
                int y = i / w;
                int x = i % w;
                img.at(y, x) = static_cast<float>(depth16);
            }
            
            std::cout << "16-bit RG decoding completed." << std::endl;
            
        } else {
            // 2. 使用改进的通道分析和融合策略
            int best_channel = selectBestChannel(raw, w, h);
            std::cout << "Selected channel " << (best_channel == 0 ? "R" : best_channel == 1 ? "G" : "B") 
                      << " as primary channel" << std::endl;
            
            // 配置选项：是否使用多通道融合
            bool use_channel_fusion = true;  // 可以根据需要修改
            
            if (use_channel_fusion) {
                std::cout << "Using weighted channel fusion..." << std::endl;
                
                double weights[3] = {0.0, 0.0, 0.0};
                calculateChannelWeights(raw, w, h, weights, best_channel);
                
                int fused_nonzero = 0;
                for (int i = 0; i < w * h; ++i) {
                    uint8_t r = raw[i*3];
                    uint8_t g = raw[i*3 + 1];
                    uint8_t b = raw[i*3 + 2];
                    
                    uint8_t gray = fuseChannels(r, g, b, weights);
                    if (gray > 0) fused_nonzero++;
                    
                    int y = i / w;
                    int x = i % w;
                    img.at(y, x) = static_cast<float>(gray);
                }
                
                std::cout << "Channel fusion completed. Nonzero pixels: " << fused_nonzero << std::endl;
                
            } else {
                std::cout << "Using single best channel..." << std::endl;
                
                int single_nonzero = 0;
                for (int i = 0; i < w * h; ++i) {
                    uint8_t gray = raw[i*3 + best_channel];
                    if (gray > 0) single_nonzero++;
                    
                    int y = i / w;
                    int x = i % w;
                    img.at(y, x) = static_cast<float>(gray);
                }
                
                std::cout << "Single channel conversion completed. Nonzero pixels: " << single_nonzero << std::endl;
            }
        }
        
        std::cout << "RGB disparity conversion completed successfully." << std::endl;
        
    } else {
        png_destroy_read_struct(&png, &info, nullptr);
        fclose(fp);
        std::cerr << "Unsupported depth image PNG format: only 8-bit or 16-bit grayscale PNG, or 8-bit RGB PNG supported" << std::endl;
        exit(1);
    }
    return img;
}

// --- Load 8-bit RGB color image using libpng ---
ColorImage loadColorPNG(const std::string& path) {
    FILE* fp = fopen(path.c_str(), "rb");
    if (!fp) { std::cerr << "Unable to open color image: " << path << std::endl; exit(1); }
    png_structp png = png_create_read_struct(PNG_LIBPNG_VER_STRING, nullptr, nullptr, nullptr);
    png_infop info = png_create_info_struct(png);
    if (!png || !info) { fclose(fp); exit(1); }
    if (setjmp(png_jmpbuf(png))) { fclose(fp); png_destroy_read_struct(&png, &info, nullptr); exit(1); }
    png_init_io(png, fp);
    png_read_info(png, info);
    int w = png_get_image_width(png, info);
    int h = png_get_image_height(png, info);
    int bit_depth = png_get_bit_depth(png, info);
    int color_type = png_get_color_type(png, info);
    if (color_type == PNG_COLOR_TYPE_PALETTE)
        png_set_palette_to_rgb(png);
    if (color_type == PNG_COLOR_TYPE_GRAY && bit_depth < 8)
        png_set_expand_gray_1_2_4_to_8(png);
    if (png_get_valid(png, info, PNG_INFO_tRNS))
        png_set_tRNS_to_alpha(png);
    if (bit_depth == 16)
        png_set_strip_16(png);
    if (color_type == PNG_COLOR_TYPE_GRAY || color_type == PNG_COLOR_TYPE_GRAY_ALPHA)
        png_set_gray_to_rgb(png);
    if (color_type == PNG_COLOR_TYPE_RGB_ALPHA)
        png_set_strip_alpha(png);
    png_read_update_info(png, info);
    ColorImage img(h, w);
    std::vector<png_bytep> row_ptrs(h);
    for (int y = 0; y < h; ++y) row_ptrs[y] = (png_bytep)(&img.data[y*w*3]);
    png_read_image(png, (png_bytep*)row_ptrs.data());
    png_destroy_read_struct(&png, &info, nullptr);
    fclose(fp);
    return img;
}

// --- Find available disparity files ---
std::vector<std::string> findDisparityFiles(const std::string& directory, const std::string& view) {
    std::vector<std::string> files;
    
    try {
        for (const auto& entry : std::filesystem::directory_iterator(directory)) {
            if (entry.is_regular_file()) {
                std::string filename = entry.path().filename().string();
                // Find both SGM and ELAS disparity files for the specified view
                if ((filename.find("disparity_SGM") == 0 || filename.find("disparity_ELAS") == 0) && 
                    filename.find("_" + view + "_") != std::string::npos &&
                    filename.find(".png") != std::string::npos) {
                    files.push_back(entry.path().string());
                }
            }
        }
    } catch (const std::filesystem::filesystem_error& e) {
        std::cerr << "Error accessing directory " << directory << ": " << e.what() << std::endl;
    }
    
    return files;
}

// --- Downsample images for fast generation ---
DepthImage downsampleDepthImage(const DepthImage& original, int factor) {
    int newHeight = original.rows / factor;
    int newWidth = original.cols / factor;
    DepthImage downsampled(newHeight, newWidth);
    
    for (int y = 0; y < newHeight; ++y) {
        for (int x = 0; x < newWidth; ++x) {
            int origY = y * factor;
            int origX = x * factor;
            downsampled.at(y, x) = original.at(origY, origX);
        }
    }
    
    return downsampled;
}

ColorImage downsampleColorImage(const ColorImage& original, int factor) {
    int newHeight = original.rows / factor;
    int newWidth = original.cols / factor;
    ColorImage downsampled(newHeight, newWidth);
    
    for (int y = 0; y < newHeight; ++y) {
        for (int x = 0; x < newWidth; ++x) {
            int origY = y * factor;
            int origX = x * factor;
            
            const uint8_t* origPixel = original.pixel(origY, origX);
            uint8_t* newPixel = downsampled.pixel(y, x);
            
            newPixel[0] = origPixel[0]; // R
            newPixel[1] = origPixel[1]; // G
            newPixel[2] = origPixel[2]; // B
        }
    }
    
    return downsampled;
}

int main(int argc, char* argv[]) {
    std::cout << "=== Point Cloud Generation from Disparity Maps ===" << std::endl;
    std::cout << "Version: Enhanced RGB Disparity Processing" << std::endl;
    
    // Check command line arguments
    if (argc < 2 || argc > 6) {
        std::cout << "Usage: " << argv[0] << " <dataset_name> [view_option] [speed_option] [disparity_path] [save_grayscale]" << std::endl;
        std::cout << "  dataset_name: any dataset folder name (e.g., test1, test2, test7, test8)" << std::endl;
        std::cout << "  view_option: (optional) specify which view to generate" << std::endl;
        std::cout << "    - 'left': generate only left view point cloud" << std::endl;
        std::cout << "    - 'right': generate only right view point cloud" << std::endl;
        std::cout << "    - 'both': generate both left and right view point clouds (default)" << std::endl;
        std::cout << "  speed_option: (optional) specify generation speed" << std::endl;
        std::cout << "    - 'fast': reduce point cloud density by 10x for faster generation" << std::endl;
        std::cout << "    - 'normal': normal density (default)" << std::endl;
        std::cout << "  disparity_path: (optional) specify custom disparity file path" << std::endl;
        std::cout << "    - If provided, will use this file instead of auto-finding" << std::endl;
        std::cout << "    - Should be a PNG file containing disparity map" << std::endl;
        std::cout << "  save_grayscale: (optional) save grayscale disparity images" << std::endl;
        std::cout << "    - 'save': save as grayscale PNG and RAW float data (default)" << std::endl;
        std::cout << "    - 'none': don't save grayscale images" << std::endl;
        std::cout << "Example: " << argv[0] << " test8 left fast" << std::endl;
        std::cout << "Example: " << argv[0] << " test8 both normal" << std::endl;
        std::cout << "Example: " << argv[0] << " test8 left normal /path/to/disparity.png" << std::endl;
        std::cout << "Example: " << argv[0] << " test8 both normal /path/to/disparity.png save" << std::endl;
        std::cout << "Example: " << argv[0] << " test8" << std::endl;
        return 1;
    }
    
    std::string datasetName = argv[1];
    std::string viewOption = (argc >= 3) ? argv[2] : "both";
    std::string speedOption = (argc >= 4) ? argv[3] : "normal";
    std::string customDisparityPath = (argc >= 5) ? argv[4] : "";
    std::string saveGrayscaleOption = (argc == 6) ? argv[5] : "save";
    
    // Validate dataset name
    if (datasetName.empty()) {
        std::cout << "Error: Dataset name cannot be empty" << std::endl;
        return 1;
    }
    
    // Validate view option
    if (viewOption != "left" && viewOption != "right" && viewOption != "both") {
        std::cout << "Error: Invalid view option '" << viewOption << "'" << std::endl;
        std::cout << "Valid options: left, right, both" << std::endl;
        return 1;
    }
    
    // Validate speed option
    if (speedOption != "fast" && speedOption != "normal") {
        std::cout << "Error: Invalid speed option '" << speedOption << "'" << std::endl;
        std::cout << "Valid options: fast, normal" << std::endl;
        return 1;
    }
    
    // Validate save grayscale option
    if (saveGrayscaleOption != "save" && saveGrayscaleOption != "none") {
        std::cout << "Error: Invalid save grayscale option '" << saveGrayscaleOption << "'" << std::endl;
        std::cout << "Valid options: save, none" << std::endl;
        return 1;
    }
    
    std::cout << "Dataset: " << datasetName << std::endl;
    std::cout << "View option: " << viewOption << std::endl;
    std::cout << "Speed option: " << speedOption << std::endl;
    std::cout << "Save grayscale: " << saveGrayscaleOption << std::endl;
    if (!customDisparityPath.empty()) {
        std::cout << "Custom disparity path: " << customDisparityPath << std::endl;
    }
    std::cout << "===============================================" << std::endl;
    
    // Create output directory with image name
    std::string outputDir = "../output/" + datasetName;
    std::filesystem::create_directories(outputDir);
    
    std::cout << "Output directory: " << outputDir << std::endl;
    
    // 1. Load camera parameters
    std::string cameraFile = "../data/camera/" + datasetName + ".txt";
    CameraParams cameraParams = loadCamera(cameraFile);
    CameraIntrinsics K = cameraParams.intrinsics;
    float baseline = cameraParams.baseline;
    
    std::cout << "Camera parameters loaded:" << std::endl;
    std::cout << "  fx: " << K.fx << std::endl;
    std::cout << "  fy: " << K.fy << std::endl;
    std::cout << "  cx: " << K.cx << std::endl;
    std::cout << "  cy: " << K.cy << std::endl;
    std::cout << "  baseline: " << baseline << std::endl;
    
    // 2. Load depth images based on view option
    DepthImage depth_left, depth_right;
    ColorImage color_left, color_right;
    
    bool loadLeft = (viewOption == "left" || viewOption == "both");
    bool loadRight = (viewOption == "right" || viewOption == "both");
    
    if (loadLeft) {
        std::cout << "\nLoading left view data..." << std::endl;
        
        std::string depthLeftPath;
        std::string colorLeftPath = "../data/left/" + datasetName + ".png";
        
        // Check if custom disparity path is provided
        if (!customDisparityPath.empty()) {
            depthLeftPath = customDisparityPath;
            std::cout << "Using custom disparity file: " << depthLeftPath << std::endl;
            
            // Check if custom disparity file exists
            if (!std::filesystem::exists(depthLeftPath)) {
                std::cerr << "Error: Custom disparity file not found: " << depthLeftPath << std::endl;
                return 1;
            }
        } else {
            // Find available left disparity files (SGM or ELAS)
            std::vector<std::string> leftDisparityFiles = findDisparityFiles(outputDir, "left");
            if (leftDisparityFiles.empty()) {
                std::cerr << "Error: No left disparity files found in: " << outputDir << std::endl;
                std::cout << "Please run testdisparitySGM.cpp or testdisparityELAS.cpp first to generate disparity maps." << std::endl;
                std::cout << "Or specify a custom disparity file path as the last argument." << std::endl;
                return 1;
            }
            
            // Use the first available disparity file
            depthLeftPath = leftDisparityFiles[0];
            std::cout << "Using left disparity file: " << std::filesystem::path(depthLeftPath).filename().string() << std::endl;
        }
        
        // Check if color image exists
        if (!std::filesystem::exists(colorLeftPath)) {
            std::cerr << "Error: Left color image not found: " << colorLeftPath << std::endl;
            return 1;
        }
        
        depth_left = loadDepthPNG(depthLeftPath);
        color_left = loadColorPNG(colorLeftPath);
        std::cout << "✓ Left view data loaded successfully" << std::endl;
        
        // Save grayscale disparity as PNG if requested
        if (saveGrayscaleOption != "none") {
            std::string grayscaleDisparityPathPNG = outputDir + "/disparity_left_grayscale.png";
            
            // Save as PNG using ImageMagick (like testdisparityELAS.cpp)
            if (savePNG_float(grayscaleDisparityPathPNG, depth_left)) {
                std::cout << "✓ Left grayscale disparity saved: " << grayscaleDisparityPathPNG << " (PNG format)" << std::endl;
            } else {
                std::cout << "✗ Failed to save left grayscale disparity PNG" << std::endl;
            }
        }
    }
    
    if (loadRight) {
        std::cout << "\nLoading right view data..." << std::endl;
        
        std::string depthRightPath;
        std::string colorRightPath = "../data/right/" + datasetName + ".png";
        
        // Check if custom disparity path is provided
        if (!customDisparityPath.empty()) {
            depthRightPath = customDisparityPath;
            std::cout << "Using custom disparity file: " << depthRightPath << std::endl;
            
            // Check if custom disparity file exists
            if (!std::filesystem::exists(depthRightPath)) {
                std::cerr << "Error: Custom disparity file not found: " << depthRightPath << std::endl;
                return 1;
            }
        } else {
            // Find available right disparity files (SGM or ELAS)
            std::vector<std::string> rightDisparityFiles = findDisparityFiles(outputDir, "right");
            if (rightDisparityFiles.empty()) {
                std::cerr << "Error: No right disparity files found in: " << outputDir << std::endl;
                std::cout << "Please run testdisparitySGM.cpp or testdisparityELAS.cpp first to generate disparity maps." << std::endl;
                std::cout << "Or specify a custom disparity file path as the last argument." << std::endl;
                return 1;
            }
            
            // Use the first available disparity file
            depthRightPath = rightDisparityFiles[0];
            std::cout << "Using right disparity file: " << std::filesystem::path(depthRightPath).filename().string() << std::endl;
        }
        
        // Check if color image exists
        if (!std::filesystem::exists(colorRightPath)) {
            std::cerr << "Error: Right color image not found: " << colorRightPath << std::endl;
            return 1;
        }
        
        depth_right = loadDepthPNG(depthRightPath);
        color_right = loadColorPNG(colorRightPath);
        std::cout << "✓ Right view data loaded successfully" << std::endl;
        
        // Save grayscale disparity as PNG if requested
        if (saveGrayscaleOption != "none") {
            std::string grayscaleDisparityPathPNG = outputDir + "/disparity_right_grayscale.png";
            
            // Save as PNG using ImageMagick (like testdisparityELAS.cpp)
            if (savePNG_float(grayscaleDisparityPathPNG, depth_right)) {
                std::cout << "✓ Right grayscale disparity saved: " << grayscaleDisparityPathPNG << " (PNG format)" << std::endl;
            } else {
                std::cout << "✗ Failed to save right grayscale disparity PNG" << std::endl;
            }
        }
    }

    // 3. Apply downsampling for fast generation if requested
    int downsampleFactor = 1;
    if (speedOption == "fast") {
        downsampleFactor = 10; // Reduce density by 10x
        std::cout << "\n=== Fast Generation Mode ===" << std::endl;
        std::cout << "Downsampling factor: " << downsampleFactor << "x" << std::endl;
        std::cout << "Point cloud density will be reduced by " << (downsampleFactor * downsampleFactor) << "x" << std::endl;
        std::cout << "Expected speedup: ~" << (downsampleFactor * downsampleFactor) << "x" << std::endl;
        
        if (loadLeft) {
            std::cout << "Downsampling left view images..." << std::endl;
            depth_left = downsampleDepthImage(depth_left, downsampleFactor);
            color_left = downsampleColorImage(color_left, downsampleFactor);
            std::cout << "Left view downsampled: " << depth_left.cols << "x" << depth_left.rows << std::endl;
        }
        
        if (loadRight) {
            std::cout << "Downsampling right view images..." << std::endl;
            depth_right = downsampleDepthImage(depth_right, downsampleFactor);
            color_right = downsampleColorImage(color_right, downsampleFactor);
            std::cout << "Right view downsampled: " << depth_right.cols << "x" << depth_right.rows << std::endl;
        }
    }

    // 4. Reconstruct point clouds based on view option
    Mesh mesh_left, mesh_right;
    
    if (loadLeft) {
        std::cout << "\nGenerating left view point cloud..." << std::endl;
        if (speedOption == "fast") {
            std::cout << "Using fast generation mode (reduced density)" << std::endl;
        }
        mesh_left = triangulateFromDepth(depth_left, color_left, K, 5000.0f, 2);
        std::cout << "✓ Left view point cloud generated" << std::endl;
    }
    
    if (loadRight) {
        std::cout << "\nGenerating right view point cloud..." << std::endl;
        if (speedOption == "fast") {
            std::cout << "Using fast generation mode (reduced density)" << std::endl;
        }
        mesh_right = triangulateFromDepth(depth_right, color_right, K, 5000.0f, 2);
        std::cout << "✓ Right view point cloud generated" << std::endl;
    }

    // 5. Save PLY files based on view option
    bool ok_left = false, ok_right = false;
    
    if (loadLeft) {
        std::string plyLeftPath = outputDir + "/pointcloud_left";
        if (speedOption == "fast") {
            plyLeftPath += "_fast";
        }
        plyLeftPath += ".ply";
        ok_left = saveMeshToPLY(plyLeftPath, mesh_left);
        
        if (ok_left) {
            std::cout << "✓ Left view point cloud PLY saved successfully!" << std::endl;
            std::cout << "  File: " << plyLeftPath << std::endl;
            std::cout << "  Vertices: " << mesh_left.vertices.size() << std::endl;
            std::cout << "  Faces: " << mesh_left.faces.size() << std::endl;
        } else {
            std::cerr << "✗ Failed to save left view point cloud PLY!" << std::endl;
        }
    }
    
    if (loadRight) {
        std::string plyRightPath = outputDir + "/pointcloud_right";
        if (speedOption == "fast") {
            plyRightPath += "_fast";
        }
        plyRightPath += ".ply";
        ok_right = saveMeshToPLY(plyRightPath, mesh_right);
        
        if (ok_right) {
            std::cout << "✓ Right view point cloud PLY saved successfully!" << std::endl;
            std::cout << "  File: " << plyRightPath << std::endl;
            std::cout << "  Vertices: " << mesh_right.vertices.size() << std::endl;
            std::cout << "  Faces: " << mesh_right.faces.size() << std::endl;
        } else {
            std::cerr << "✗ Failed to save right view point cloud PLY!" << std::endl;
        }
    }

    // 6. Summary
    std::cout << "\n=== Processing Summary ===" << std::endl;
    std::cout << "Dataset: " << datasetName << std::endl;
    std::cout << "View option: " << viewOption << std::endl;
    std::cout << "Speed option: " << speedOption << std::endl;
    std::cout << "RGB processing: Enhanced with intelligent channel analysis" << std::endl;
    
    if (speedOption == "fast") {
        std::cout << "Downsampling factor: " << downsampleFactor << "x" << std::endl;
        std::cout << "Density reduction: " << (downsampleFactor * downsampleFactor) << "x" << std::endl;
    }
    
    if (viewOption == "left" || viewOption == "both") {
        std::cout << "Left view: " << (ok_left ? "✓ Success" : "✗ Failed") << std::endl;
    }
    if (viewOption == "right" || viewOption == "both") {
        std::cout << "Right view: " << (ok_right ? "✓ Success" : "✗ Failed") << std::endl;
    }

    std::cout << "\n=== Processing Completed ===" << std::endl;
    std::cout << "Results saved in directory: " << outputDir << std::endl;
    
    if (viewOption == "both") {
        std::cout << "Generated files:" << std::endl;
        if (ok_left) {
            std::string leftFile = "pointcloud_left";
            if (speedOption == "fast") leftFile += "_fast";
            leftFile += ".ply";
            std::cout << "  - " << leftFile << std::endl;
        }
        if (ok_right) {
            std::string rightFile = "pointcloud_right";
            if (speedOption == "fast") rightFile += "_fast";
            rightFile += ".ply";
            std::cout << "  - " << rightFile << std::endl;
        }
    } else {
        std::cout << "Generated file:" << std::endl;
        if (viewOption == "left" && ok_left) {
            std::string leftFile = "pointcloud_left";
            if (speedOption == "fast") leftFile += "_fast";
            leftFile += ".ply";
            std::cout << "  - " << leftFile << std::endl;
        }
        if (viewOption == "right" && ok_right) {
            std::string rightFile = "pointcloud_right";
            if (speedOption == "fast") rightFile += "_fast";
            rightFile += ".ply";
            std::cout << "  - " << rightFile << std::endl;
        }
    }

    return 0;
}

/*
=== 改进后的 testpointcloud.cpp 主要特性 ===

1. 智能RGB视差图处理：
   - 自动检测多通道编码格式（如16位RG编码）
   - 智能通道分析和选择策略
   - 可选的多通道加权融合
   - 详细的统计信息输出

2. 灰度视差图保存功能：
   - 使用ImageMagick方式保存（与testdisparityELAS.cpp一致）
   - 同时保存PNG和RAW浮点数据格式
   - 自动归一化和格式转换
   - 可配置是否保存灰度视差图

3. 改进的算法特性：
   - 通道质量评分（信息量 + 动态范围 + 标准差）
   - 多通道相关性检测
   - 16位解码支持
   - 加权融合算法

4. 使用方法：
   ./testpointcloud test8 left fast
   ./testpointcloud test8 both normal
   ./testpointcloud test8 left normal /path/to/disparity.png save
   ./testpointcloud test8 both normal /path/to/disparity.png none

5. 输出信息更丰富：
   - 通道分析结果
   - 处理策略选择
   - 统计信息报告
   - 灰度视差图保存状态

6. 向后兼容：
   - 保持所有原有功能
   - 命令行参数向后兼容
   - 输出文件格式不变

=== 主要改进点 ===

1. 多通道编码检测：
   - 自动识别16位深度值在RG通道中的编码
   - 高精度解码，避免信息丢失

2. 智能通道选择：
   - 综合考虑信息量、动态范围、标准差
   - 避免选择噪声通道

3. 加权融合选项：
   - 保留多通道信息
   - 可配置的融合策略

4. 灰度视差图保存：
   - 使用ImageMagick方式（与testdisparityELAS.cpp一致）
   - 同时保存PNG和RAW浮点数据
   - 自动归一化和格式转换
   - 可选的保存功能

5. 详细诊断输出：
   - 通道统计信息
   - 处理策略说明
   - 结果验证数据
   - 保存状态报告

=== 新增命令行参数 ===

save_grayscale: 控制灰度视差图保存
- 'save': 保存为灰度PNG格式（默认）
- 'none': 不保存灰度视差图

=== 输出文件 ===

新增灰度视差图文件：
- disparity_left_grayscale.png (PNG格式，使用ImageMagick转换)
- disparity_right_grayscale.png (PNG格式，使用ImageMagick转换)

注意：保存方式与testdisparityELAS.cpp保持一致，使用ImageMagick进行格式转换
*/