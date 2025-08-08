#include "elas.h"
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

using namespace std;

// 定义无效视差值（与SGM保持一致）
const float Invalid_Float = -1.0f;

// SGM风格的填充函数 - 保持原始算法不变
void FillHolesInDispMapSGM(float* disp_left, int32_t width, int32_t height, 
                          int32_t min_disparity, int32_t max_disparity) {
    std::vector<float> disp_collects;

    // 定义8个方向 - 与SGM保持一致
    const float pi = 3.1415926f;
    float angle1[8] = { pi, 3 * pi / 4, pi / 2, pi / 4, 0, 7 * pi / 4, 3 * pi / 2, 5 * pi / 4 };
    float angle2[8] = { pi, 5 * pi / 4, 3 * pi / 2, 7 * pi / 4, 0, pi / 4, pi / 2, 3 * pi / 4 };
    float *angle = angle1;
    
    // 最大搜索行程，与SGM保持一致
    const int32_t max_search_length = 1.0 * std::max(abs(max_disparity), abs(min_disparity));

    float* disp_ptr = disp_left;
    for (int32_t k = 0; k < 3; k++) {
        // 收集需要填充的像素
        std::vector<std::pair<int32_t, int32_t>> trg_pixels;
        
        if (k == 2) {
            // 第三次循环处理所有无效像素
            for (int32_t i = 0; i < height; i++) {
                for (int32_t j = 0; j < width; j++) {
                    if (disp_ptr[i * width + j] == Invalid_Float) {
                        trg_pixels.emplace_back(i, j);
                    }
                }
            }
        } else {
            // 前两次循环处理所有无效像素（简化版本，因为ELAS没有occlusions_和mismatches_）
            for (int32_t i = 0; i < height; i++) {
                for (int32_t j = 0; j < width; j++) {
                    if (disp_ptr[i * width + j] == Invalid_Float) {
                        trg_pixels.emplace_back(i, j);
                    }
                }
            }
        }
        
        if (trg_pixels.empty()) {
            continue;
        }
        
        std::vector<float> fill_disps(trg_pixels.size());

        // 遍历待处理像素
        for (auto n = 0u; n < trg_pixels.size(); n++) {
            auto& pix = trg_pixels[n];
            const int32_t y = pix.first;
            const int32_t x = pix.second;

            if (y == height / 2) {
                angle = angle2; 
            }

            // 收集8个方向上遇到的首个有效视差值
            disp_collects.clear();
            for (int32_t s = 0; s < 8; s++) {
                const float ang = angle[s];
                const float sina = float(sin(ang));
                const float cosa = float(cos(ang));
                for (int32_t m = 1; m < max_search_length; m++) {
                    const int32_t yy = lround(y + m * sina);
                    const int32_t xx = lround(x + m * cosa);
                    if (yy<0 || yy >= height || xx<0 || xx >= width) {
                        break;
                    }
                    const auto& disp = *(disp_ptr + yy*width + xx);
                    if (disp != Invalid_Float) {
                        disp_collects.push_back(disp);
                        break;
                    }
                }
            }
            if(disp_collects.empty()) {
                continue;
            }

            std::sort(disp_collects.begin(), disp_collects.end());

            // 如果是第一次循环，选择第二小的视差值（遮挡区策略）
            // 如果是第二次或第三次循环，选择中值（误匹配区策略）
            if (k == 0) {
                if (disp_collects.size() > 1) {
                    fill_disps[n] = disp_collects[1];
                }
                else {
                    fill_disps[n] = disp_collects[0];
                }
            }
            else{
                fill_disps[n] = disp_collects[disp_collects.size() / 2];
            }
        }
        
        // 应用填充结果
        for (auto n = 0u; n < trg_pixels.size(); n++) {
            auto& pix = trg_pixels[n];
            const int32_t y = pix.first;
            const int32_t x = pix.second;
            disp_ptr[y * width + x] = fill_disps[n];
        }
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
    // PFM格式要求从底部到顶部写入数据，避免180度旋转
    for (int y = height - 1; y >= 0; y--) {
        for (int x = 0; x < width; x++) {
            int index = y * width + x;
            fwrite(&data[index], sizeof(float), 1, file);
        }
    }
    
    fclose(file);
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

// 将ELAS视差图转换为PFM格式
void convertElasToPFM(float* elas_disp, float* pfm_disp, int width, int height) {
    for (int i = 0; i < width * height; i++) {
        if (elas_disp[i] < 0 || elas_disp[i] == Invalid_Float) {
            pfm_disp[i] = INFINITY;  // 无效值用INFINITY表示
        } else {
            pfm_disp[i] = elas_disp[i];
        }
    }
}

int main(int argc, char* argv[]) {
    // MiddEval3接口：<im0.png> <im1.png> <output.pfm> <maxdisp>
    if (argc != 5) {
        cerr << "Usage: " << argv[0] << " <im0.png> <im1.png> <output.pfm> <maxdisp>" << endl;
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
    
    cout << "MyELAS Algorithm for MiddEval3" << endl;
    cout << "==============================" << endl;
    cout << "Left image: " << left_path << endl;
    cout << "Right image: " << right_path << endl;
    cout << "Output: " << output_path << endl;
    cout << "Max disparity: " << maxdisp << endl;
    cout << "==============================" << endl;
    
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
        // 尝试读取PNG图像
        I1 = loadPNG(left_path.c_str());
        I2 = loadPNG(right_path.c_str());
        
        if (I1 == nullptr || I2 == nullptr) {
            cerr << "Error: Could not read PNG images!" << endl;
            cerr << "Please ensure ImageMagick is installed: sudo apt install imagemagick" << endl;
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
    
    // 分配视差图内存
    int32_t dims[3];
    dims[0] = width;  // bytes per line = width
    dims[1] = height; // height
    dims[2] = width;  // bytes per line = width
    
    // 设置ELAS参数 - 使用Middlebury基准测试的默认设置
    Elas::parameters param;
    param.disp_min = 0;
    param.disp_max = maxdisp;
    param.support_threshold = 0.95;
    param.support_texture = 10;
    param.candidate_stepsize = 5;
    param.incon_window_size = 5;
    param.incon_threshold = 5;
    param.incon_min_support = 5;
    param.add_corners = 1;
    param.grid_size = 20;
    param.beta = 0.02;
    param.gamma = 5;
    param.sigma = 1;
    param.sradius = 3;
    param.match_texture = 0;
    param.lr_threshold = 2;
    param.speckle_sim_threshold = 1;
    param.speckle_size = 200;
    param.ipol_gap_width = 5000;
    param.filter_median = 1;
    param.filter_adaptive_mean = 0;
    param.postprocess_only_left = 0;
    param.subsampling = 0;
    
    cout << "ELAS Parameters (Middlebury Benchmark Settings):" << endl;
    cout << "  Disparity range: [" << param.disp_min << ", " << param.disp_max << "]" << endl;
    cout << "  Support threshold: " << param.support_threshold << endl;
    cout << "  Support texture: " << param.support_texture << endl;
    cout << "  Grid size: " << param.grid_size << endl;
    cout << "  Speckle size: " << param.speckle_size << endl;
    cout << "  Filter median: " << (param.filter_median ? "Yes" : "No") << endl;
    cout << "  Postprocess only left: " << (param.postprocess_only_left ? "Yes" : "No") << endl;
    cout << endl;
    
    // 创建ELAS对象
    Elas elas(param);
    
    // 分配视差图内存
    float* D1 = nullptr;
    float* D2 = nullptr;
    float* pfm_disp = nullptr;
    
    try {
        D1 = new float[width * height];
        D2 = new float[width * height];
        pfm_disp = new float[width * height];
        
        cout << "Processing ELAS..." << endl;
        
        // 处理立体匹配
        elas.process(I1->data, I2->data, D1, D2, dims);
        
        cout << "ELAS processing completed!" << endl;
        
        // 应用SGM风格的填充
        cout << "Applying SGM-style hole filling..." << endl;
        
        // 将ELAS的无效视差值转换为SGM的无效值格式
        for (int i = 0; i < width * height; i++) {
            if (D1[i] < 0) {
                D1[i] = Invalid_Float;
            }
            if (D2[i] < 0) {
                D2[i] = Invalid_Float;
            }
        }
        
        // 应用SGM风格的填充
        FillHolesInDispMapSGM(D1, width, height, param.disp_min, param.disp_max);
        
        cout << "SGM-style hole filling completed!" << endl;
        
        // 转换为PFM格式
        convertElasToPFM(D1, pfm_disp, width, height);
        
        // 保存PFM文件
        savePFM(output_path.c_str(), pfm_disp, width, height);
        
        // 验证输出文件是否创建成功
        ifstream output_file(output_path);
        if (!output_file.good()) {
            cerr << "Error: Failed to create output file: " << output_path << endl;
            throw std::runtime_error("Output file creation failed");
        }
        output_file.close();
        
        cout << "Disparity map saved to: " << output_path << endl;
        
        // 统计有效像素
        int valid_pixels_left = 0;
        for (int i = 0; i < width * height; i++) {
            if (D1[i] != Invalid_Float && D1[i] > 0) {
                valid_pixels_left++;
            }
        }
        
        cout << "Valid pixels: " << valid_pixels_left << "/" << (width * height) 
             << " (" << (100.0 * valid_pixels_left / (width * height)) << "%)" << endl;
        
    } catch (const std::exception& e) {
        cerr << "Error during processing: " << e.what() << endl;
        
        // 清理内存
        if (D1) delete[] D1;
        if (D2) delete[] D2;
        if (pfm_disp) delete[] pfm_disp;
        delete I1;
        delete I2;
        return 1;
    }
    
    // 正常清理内存
    delete[] D1;
    delete[] D2;
    delete[] pfm_disp;
    
    // 注意：不要手动删除I1->data和I2->data！
    // image类的析构函数会自动处理数据内存
    delete I1;
    delete I2;
    
    cout << "MyELAS completed successfully!" << endl;
    return 0;
}