#include "SemiGlobalMatching.h"
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

// 将SGM视差图转换为PFM格式
void convertSGMToPFM(float* sgm_disp, float* pfm_disp, int width, int height) {
    for (int i = 0; i < width * height; i++) {
        if (sgm_disp[i] < 0 || sgm_disp[i] == Invalid_Float) {
            pfm_disp[i] = INFINITY;  // 无效值用INFINITY表示
        } else {
            pfm_disp[i] = sgm_disp[i];
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
    
    cout << "SGM Algorithm for MiddEval3" << endl;
    cout << "===========================" << endl;
    cout << "Left image: " << left_path << endl;
    cout << "Right image: " << right_path << endl;
    cout << "Output: " << output_path << endl;
    cout << "Max disparity: " << maxdisp << endl;
    cout << "===========================" << endl;
    
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
    
    // 设置SGM参数
    SGMOption option;
    option.min_disparity = 0;
    option.max_disparity = maxdisp;
    option.num_paths = 8;  // 8路径聚合
    option.census_size = Census5x5;  // 5x5 census窗口
    option.is_check_unique = true;
    option.uniqueness_ratio = 0.95f;
    option.is_check_lr = true;
    option.lrcheck_thres = 1.0f;
    option.is_remove_speckles = true;
    option.min_speckle_aera = 20;
    option.is_fill_holes = true;
    option.p1 = 10;
    option.p2_init = 150;
    
    cout << "SGM Parameters:" << endl;
    cout << "  Disparity range: [" << option.min_disparity << ", " << option.max_disparity << "]" << endl;
    cout << "  Number of paths: " << (int)option.num_paths << endl;
    cout << "  Census window: " << (option.census_size == Census5x5 ? "5x5" : "9x7") << endl;
    cout << "  P1 penalty: " << option.p1 << endl;
    cout << "  P2 penalty: " << option.p2_init << endl;
    cout << "  Uniqueness ratio: " << option.uniqueness_ratio << endl;
    cout << "  LR check threshold: " << option.lrcheck_thres << endl;
    cout << "  Min speckle area: " << option.min_speckle_aera << endl;
    cout << endl;
    
    // 创建SGM对象
    SemiGlobalMatching sgm;
    
    // 初始化SGM
    if (!sgm.Initialize(width, height, option)) {
        cerr << "Error: Failed to initialize SGM!" << endl;
        delete I1;
        delete I2;
        return 1;
    }
    
    // 分配视差图内存
    float* disp_left = nullptr;
    float* pfm_disp = nullptr;
    
    try {
        disp_left = new float[width * height];
        pfm_disp = new float[width * height];
        
        cout << "Processing SGM..." << endl;
        
        // 执行SGM匹配
        if (!sgm.Match(I1->data, I2->data, disp_left)) {
            cerr << "Error: SGM matching failed!" << endl;
            throw std::runtime_error("SGM matching failed");
        }
        
        cout << "SGM processing completed!" << endl;
        
        // 转换为PFM格式
        convertSGMToPFM(disp_left, pfm_disp, width, height);
        
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
        int valid_pixels = 0;
        for (int i = 0; i < width * height; i++) {
            if (disp_left[i] != Invalid_Float && disp_left[i] > 0) {
                valid_pixels++;
            }
        }
        
        cout << "Valid pixels: " << valid_pixels << "/" << (width * height) 
             << " (" << (100.0 * valid_pixels / (width * height)) << "%)" << endl;
        
    } catch (const std::exception& e) {
        cerr << "Error during processing: " << e.what() << endl;
        
        // 清理内存
        if (disp_left) delete[] disp_left;
        if (pfm_disp) delete[] pfm_disp;
        delete I1;
        delete I2;
        return 1;
    }
    
    // 正常清理内存
    delete[] disp_left;
    delete[] pfm_disp;
    
    // 注意：不要手动删除I1->data和I2->data！
    // image类的析构函数会自动处理数据内存
    delete I1;
    delete I2;
    
    cout << "SGM completed successfully!" << endl;
    return 0;
}