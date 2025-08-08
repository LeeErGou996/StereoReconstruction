#include "../src/depth.h"
#include "../src/utils/imageutils.h"
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <cmath>
#include <algorithm>
#include <random>
#include <filesystem>

// 从相机参数文件读取Q矩阵
bool loadQMatrix(const std::string& path, Matrix& Q) {
    std::ifstream file(path);
    if (!file.is_open()) {
        std::cerr << "无法打开相机参数文件: " << path << std::endl;
        return false;
    }
    
    double fx = 1000.0, fy = 1000.0, cx = 320.0, cy = 240.0, baseline = 0.1;
    
    std::string line;
    while (std::getline(file, line)) {
        // 读取 cam0
        if (line.find("cam0") != std::string::npos) {
            auto l = line.find('[');
            auto r = line.find(']');
            if (l != std::string::npos && r != std::string::npos && r > l) {
                std::string nums = line.substr(l + 1, r - l - 1);
                std::replace(nums.begin(), nums.end(), ';', ' '); // 用空格替换分号
                std::istringstream iss(nums);
                double zero, zero2, zero3, zero4, one;
                iss >> fx >> zero >> cx >> zero2 >> fy >> cy >> zero3 >> zero4 >> one;
                std::cout << "读取到相机内参: fx=" << fx << ", fy=" << fy << ", cx=" << cx << ", cy=" << cy << std::endl;
            }
        }
        // 读取 baseline
        if (line.find("baseline") != std::string::npos) {
            auto pos = line.find('=');
            if (pos != std::string::npos) {
                baseline = std::stod(line.substr(pos + 1));
                std::cout << "读取到基线: " << baseline << std::endl;
            }
        }
    }
    file.close();
    
    // 构建Q矩阵 (4x4)
    Q = Matrix(4, 4);
    Q.printInfo();
    std::cout << "Q矩阵构造函数调用后，尺寸: " << Q.rows() << "x" << Q.cols() << std::endl;
    
    // Q矩阵格式: [1 0 0 -cx; 0 1 0 -cy; 0 0 0 f; 0 0 -1/Tx 0]
    Q.at(0, 0) = 1.0; Q.at(0, 3) = -cx;     // cx
    Q.at(1, 1) = 1.0; Q.at(1, 3) = -cy;     // cy
    Q.at(2, 2) = 0.0; Q.at(2, 3) = fx;      // f
    Q.at(3, 2) = -1.0/baseline; Q.at(3, 3) = 0.0; // Tx
    
    std::cout << "Q矩阵已创建，尺寸: " << Q.rows() << "x" << Q.cols() << std::endl;
    Q.print();
    
    std::cout << "Loaded camera parameters:" << std::endl;
    std::cout << "  fx: " << fx << std::endl;
    std::cout << "  fy: " << fy << std::endl;
    std::cout << "  cx: " << cx << std::endl;
    std::cout << "  cy: " << cy << std::endl;
    std::cout << "  baseline: " << baseline << std::endl;
    
    return true;
}

// 从PNG文件读取视差图
MyMat loadDisparityFromPNG(const std::string& filename) {
    MyImage img;
    if (!myImReadPNG(filename, img)) {
        std::cerr << "无法读取PNG文件: " << filename << std::endl;
        return MyMat(); // 返回空矩阵
    }
    
    std::cout << "读取PNG文件: " << filename << std::endl;
    std::cout << "  尺寸: " << img.width << "x" << img.height << std::endl;
    std::cout << "  通道数: " << img.channels << std::endl;
    
    // 创建MyMat对象
    MyMat disparity(img.height, img.width, 0); // float type
    
    // 转换数据
    if (img.channels == 1) {
        // 灰度图像，直接转换
        for (int y = 0; y < img.height; ++y) {
            for (int x = 0; x < img.width; ++x) {
                float value = static_cast<float>(img.data[y * img.width + x]);
                disparity.at(y, x) = value;
            }
        }
    } else if (img.channels == 3) {
        // RGB图像，转换为灰度
        for (int y = 0; y < img.height; ++y) {
            for (int x = 0; x < img.width; ++x) {
                int idx = (y * img.width + x) * 3;
                float r = static_cast<float>(img.data[idx]);
                float g = static_cast<float>(img.data[idx + 1]);
                float b = static_cast<float>(img.data[idx + 2]);
                // 转换为灰度: 0.299*R + 0.587*G + 0.114*B
                float gray = 0.299f * r + 0.587f * g + 0.114f * b;
                disparity.at(y, x) = gray;
            }
        }
    } else if (img.channels == 4) {
        // RGBA图像，转换为灰度（忽略alpha通道）
        for (int y = 0; y < img.height; ++y) {
            for (int x = 0; x < img.width; ++x) {
                int idx = (y * img.width + x) * 4;
                float r = static_cast<float>(img.data[idx]);
                float g = static_cast<float>(img.data[idx + 1]);
                float b = static_cast<float>(img.data[idx + 2]);
                // 转换为灰度: 0.299*R + 0.587*G + 0.114*B
                float gray = 0.299f * r + 0.587f * g + 0.114f * b;
                disparity.at(y, x) = gray;
            }
        }
    } else {
        std::cerr << "不支持的图像通道数: " << img.channels << std::endl;
        return MyMat();
    }
    
    std::cout << "视差图转换完成，尺寸: " << disparity.rows << "x" << disparity.cols << std::endl;
    return disparity;
}

// 保存深度图为文本文件（用于调试）
bool saveDepthMapText(const std::string& filename, const MyMat& depthMap) {
    std::ofstream file(filename);
    if (!file.is_open()) {
        std::cerr << "无法打开文件保存深度图: " << filename << std::endl;
        return false;
    }
    
    file << depthMap.rows << " " << depthMap.cols << std::endl;
    for (int y = 0; y < depthMap.rows; ++y) {
        for (int x = 0; x < depthMap.cols; ++x) {
            file << depthMap.at(y, x) << " ";
        }
        file << std::endl;
    }
    
    std::cout << "深度图已保存到: " << filename << std::endl;
    return true;
}

// 保存深度图为PNG文件
bool saveDepthMapPNG(const std::string& filename, const MyMat& depthMap, bool useColorMap = true) {
    if (depthMap.empty()) {
        std::cerr << "深度图为空，无法保存PNG" << std::endl;
        return false;
    }
    
    // 找到深度范围
    double minVal = std::numeric_limits<double>::max();
    double maxVal = std::numeric_limits<double>::lowest();
    
    for (int y = 0; y < depthMap.rows; ++y) {
        for (int x = 0; x < depthMap.cols; ++x) {
            double val = depthMap.at(y, x);
            if (std::isfinite(val) && val > 0) {  // 只考虑有效的深度值
                minVal = std::min(minVal, val);
                maxVal = std::max(maxVal, val);
            }
        }
    }
    
    if (minVal >= maxVal) {
        std::cerr << "深度范围无效: min=" << minVal << ", max=" << maxVal << std::endl;
        return false;
    }
    
    std::cout << "深度范围: [" << minVal << ", " << maxVal << "]" << std::endl;
    
    // 创建图像数据
    MyImage img;
    img.width = depthMap.cols;
    img.height = depthMap.rows;
    
    if (useColorMap) {
        // 彩色深度图（使用jet颜色映射）
        img.channels = 3;
        img.data.resize(img.width * img.height * 3);
        
        for (int y = 0; y < img.height; ++y) {
            for (int x = 0; x < img.width; ++x) {
                double depth = depthMap.at(y, x);
                int idx = (y * img.width + x) * 3;
                
                if (std::isfinite(depth) && depth > 0) {
                    // 归一化到[0,1]
                    double normalized = (depth - minVal) / (maxVal - minVal);
                    normalized = std::max(0.0, std::min(1.0, normalized));
                    
                    // Jet颜色映射: 蓝->青->绿->黄->红
                    double r, g, b;
                    if (normalized < 0.25) {
                        // 蓝到青
                        double t = normalized / 0.25;
                        r = 0.0;
                        g = t;
                        b = 1.0;
                    } else if (normalized < 0.5) {
                        // 青到绿
                        double t = (normalized - 0.25) / 0.25;
                        r = 0.0;
                        g = 1.0;
                        b = 1.0 - t;
                    } else if (normalized < 0.75) {
                        // 绿到黄
                        double t = (normalized - 0.5) / 0.25;
                        r = t;
                        g = 1.0;
                        b = 0.0;
                    } else {
                        // 黄到红
                        double t = (normalized - 0.75) / 0.25;
                        r = 1.0;
                        g = 1.0 - t;
                        b = 0.0;
                    }
                    
                    img.data[idx] = static_cast<unsigned char>(r * 255);
                    img.data[idx + 1] = static_cast<unsigned char>(g * 255);
                    img.data[idx + 2] = static_cast<unsigned char>(b * 255);
                } else {
                    // 无效深度值设为黑色
                    img.data[idx] = 0;
                    img.data[idx + 1] = 0;
                    img.data[idx + 2] = 0;
                }
            }
        }
    } else {
        // 灰度深度图
        img.channels = 1;
        img.data.resize(img.width * img.height);
        
        for (int y = 0; y < img.height; ++y) {
            for (int x = 0; x < img.width; ++x) {
                double depth = depthMap.at(y, x);
                int idx = y * img.width + x;
                
                if (std::isfinite(depth) && depth > 0) {
                    // 归一化到[0,255]
                    double normalized = (depth - minVal) / (maxVal - minVal);
                    normalized = std::max(0.0, std::min(1.0, normalized));
                    img.data[idx] = static_cast<unsigned char>(normalized * 255);
                } else {
                    // 无效深度值设为0
                    img.data[idx] = 0;
                }
            }
        }
    }
    
    // 保存PNG文件
    if (myImWritePNG(filename, img)) {
        std::cout << "深度图PNG已保存到: " << filename << std::endl;
        return true;
    } else {
        std::cerr << "保存PNG文件失败: " << filename << std::endl;
        return false;
    }
}

// 创建简单的Q矩阵（如果文件读取失败）
Matrix createDefaultQMatrix() {
    Matrix Q(4, 4);
    
    // 默认相机参数
    double fx = 1000.0, fy = 1000.0, cx = 320.0, cy = 240.0, baseline = 0.12;
    
    // Q矩阵格式: [1 0 0 -cx; 0 1 0 -cy; 0 0 0 f; 0 0 -1/Tx 0]
    Q.at(0, 0) = 1.0; Q.at(0, 3) = -cx;     // cx
    Q.at(1, 1) = 1.0; Q.at(1, 3) = -cy;     // cy
    Q.at(2, 2) = 0.0; Q.at(2, 3) = fx;      // f
    Q.at(3, 2) = -1.0/baseline; Q.at(3, 3) = 0.0; // Tx
    
    std::cout << "使用默认相机参数:" << std::endl;
    std::cout << "  fx: " << fx << std::endl;
    std::cout << "  fy: " << fy << std::endl;
    std::cout << "  cx: " << cx << std::endl;
    std::cout << "  cy: " << cy << std::endl;
    std::cout << "  baseline: " << baseline << std::endl;
    
    return Q;
}

// 查找文件夹中所有视差图文件
std::vector<std::string> findDisparityFiles(const std::string& directory) {
    std::vector<std::string> files;
    
    try {
        for (const auto& entry : std::filesystem::directory_iterator(directory)) {
            if (entry.is_regular_file()) {
                std::string filename = entry.path().filename().string();
                // 只查找以disparity_ELAS或disparity_SGM开头的PNG文件
                if ((filename.find("disparity_ELAS") == 0 || filename.find("disparity_SGM") == 0) && 
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

// 处理单个视差图文件生成深度图
bool processDisparityFile(const std::string& disparityFile, const std::string& outputDir, 
                         const Matrix& Q, double baseline = 0.12) {
    std::cout << "\n=== Processing: " << disparityFile << " ===" << std::endl;
    
    // 提取文件名（不包含路径）
    std::filesystem::path filePath(disparityFile);
    std::string filename = filePath.filename().string();
    std::string baseName = filePath.stem().string(); // 移除.png扩展名
    
    // 加载视差图
    MyMat disparity = loadDisparityFromPNG(disparityFile);
    if (disparity.empty()) {
        std::cerr << "❌ Failed to load disparity file: " << disparityFile << std::endl;
        return false;
    }
    
    std::cout << "✓ Loaded disparity map: " << filename << std::endl;
    std::cout << "  Size: " << disparity.rows << "x" << disparity.cols << std::endl;
    
    // 计算深度图
    MyMat depthMap;
    bool success = false;
    
    // 尝试使用Middlebury方法
    success = Depth::computeDepthMapForMiddlebury(disparity, Q, depthMap, baseline);
    
    if (!success) {
        std::cout << "Middlebury method failed, trying standard method..." << std::endl;
        success = Depth::computeDepthMap(disparity, Q, depthMap);
    }
    
    if (!success) {
        std::cerr << "❌ Failed to compute depth map for: " << filename << std::endl;
        return false;
    }
    
    std::cout << "✓ Depth computation successful!" << std::endl;
    std::cout << "  Depth map size: " << depthMap.rows << "x" << depthMap.cols << std::endl;
    
    // 计算深度统计信息
    double minVal, maxVal;
    if (Depth::minMaxLoc(depthMap, minVal, maxVal)) {
        std::cout << "  Depth range: [" << minVal << ", " << maxVal << "]" << std::endl;
    }
    
    int validPixels = Depth::countNonZero(depthMap);
    std::cout << "  Valid depth pixels: " << validPixels << "/" << (depthMap.rows * depthMap.cols) << std::endl;
    
    // 生成输出文件名
    std::string depthTextFile = outputDir + "/depth_" + baseName + ".txt";
    std::string depthColorFile = outputDir + "/depth_" + baseName + "_color.png";
    std::string depthGrayFile = outputDir + "/depth_" + baseName + "_gray.png";
    
    // 保存深度图
    bool saveSuccess = true;
    
    // 保存文本格式（已注释）
    /*
    if (!saveDepthMapText(depthTextFile, depthMap)) {
        std::cerr << "❌ Failed to save depth text file: " << depthTextFile << std::endl;
        saveSuccess = false;
    } else {
        std::cout << "✓ Saved depth text file: " << depthTextFile << std::endl;
    }
    */
    
    // 保存彩色PNG
    if (!saveDepthMapPNG(depthColorFile, depthMap, true)) {
        std::cerr << "❌ Failed to save depth color PNG: " << depthColorFile << std::endl;
        saveSuccess = false;
    } else {
        std::cout << "✓ Saved depth color PNG: " << depthColorFile << std::endl;
    }
    
    // 保存灰度PNG
    if (!saveDepthMapPNG(depthGrayFile, depthMap, false)) {
        std::cerr << "❌ Failed to save depth gray PNG: " << depthGrayFile << std::endl;
        saveSuccess = false;
    } else {
        std::cout << "✓ Saved depth gray PNG: " << depthGrayFile << std::endl;
    }
    
    // 质量评估
    std::cout << "\n=== Quality Assessment ===" << std::endl;
    Depth::evaluateDepthQuality(depthMap, disparity);
    
    // 归一化处理
    std::cout << "\n=== Normalization ===" << std::endl;
    MyMat groundTruth = depthMap.clone(); // 使用深度图作为"真实"深度图
    MyMat normalizedDepth;
    bool normSuccess = Depth::normalizeDepthToGroundTruth(depthMap, groundTruth, normalizedDepth);
    if (normSuccess) {
        // 注释掉保存归一化txt文件
        /*
        std::string normalizedFile = outputDir + "/depth_" + baseName + "_normalized.txt";
        if (saveDepthMapText(normalizedFile, normalizedDepth)) {
            std::cout << "✓ Saved normalized depth: " << normalizedFile << std::endl;
        }
        */
        std::cout << "✓ Normalization completed (txt file saving disabled)" << std::endl;
    } else {
        std::cout << "✗ Normalization failed" << std::endl;
    }
    
    return saveSuccess;
}

int main(int argc, char* argv[]) {
    std::cout << "=== Custom Depth Computation for Multiple Disparity Files ===" << std::endl;
    
    // 检查命令行参数
    if (argc != 2) {
        std::cout << "Usage: " << argv[0] << " <dataset_name>" << std::endl;
        std::cout << "  dataset_name: any dataset folder name (e.g., test1, test2, test7, test8)" << std::endl;
        std::cout << "Example: " << argv[0] << " test8" << std::endl;
        return 1;
    }
    
    std::string datasetName = argv[1];
    
    // 验证数据集名称
    if (datasetName.empty()) {
        std::cout << "Error: Dataset name cannot be empty" << std::endl;
        return 1;
    }
    
    std::cout << "\n" << std::string(60, '=') << std::endl;
    std::cout << "Processing dataset: " << datasetName << std::endl;
    std::cout << std::string(60, '=') << std::endl;
    
    // 设置路径
    std::string outputDir = "../output/" + datasetName;
    std::string cameraFile = "../data/camera/" + datasetName + ".txt";
    
    // 检查输出目录是否存在
    if (!std::filesystem::exists(outputDir)) {
        std::cerr << "❌ Output directory not found: " << outputDir << std::endl;
        std::cerr << "Please ensure the dataset '" << datasetName << "' exists in ../output/" << std::endl;
        return 1;
    }
    
    // 尝试加载Q矩阵
    Matrix Q;
    bool qLoaded = loadQMatrix(cameraFile, Q);
    if (!qLoaded) {
        std::cout << "Using default Q matrix..." << std::endl;
        Q = createDefaultQMatrix();
    }
    
    // 查找所有视差图文件
    std::vector<std::string> disparityFiles = findDisparityFiles(outputDir);
    
    if (disparityFiles.empty()) {
        std::cout << "No disparity_ELAS*.png or disparity_SGM*.png files found in directory: " << outputDir << std::endl;
        std::cout << "Please run disparity computation first to generate disparity maps for " << datasetName << "." << std::endl;
        return 1;
    }
    
    std::cout << "\nFound " << disparityFiles.size() << " disparity_ELAS/SGM files in " << datasetName << ":" << std::endl;
    for (const auto& file : disparityFiles) {
        std::cout << "  - " << std::filesystem::path(file).filename().string() << std::endl;
    }
    
    // 处理每个视差图文件
    int successCount = 0;
    int totalCount = disparityFiles.size();
    
    for (const auto& disparityFile : disparityFiles) {
        if (processDisparityFile(disparityFile, outputDir, Q)) {
            successCount++;
        }
        std::cout << std::string(50, '=') << std::endl;
    }
    
    // 总结处理结果
    std::cout << "\n" << std::string(60, '=') << std::endl;
    std::cout << "=== PROCESSING SUMMARY ===" << std::endl;
    std::cout << std::string(60, '=') << std::endl;
    std::cout << "Dataset: " << datasetName << std::endl;
    std::cout << "Total disparity files: " << totalCount << std::endl;
    std::cout << "Successful depth computations: " << successCount << std::endl;
    std::cout << "Failed depth computations: " << (totalCount - successCount) << std::endl;
    
    if (successCount > 0) {
        std::cout << "\nGenerated depth maps for " << datasetName << ":" << std::endl;
        std::cout << "  - depth_[filename].txt (text format)" << std::endl;
        std::cout << "  - depth_[filename]_color.png (color visualization)" << std::endl;
        std::cout << "  - depth_[filename]_gray.png (grayscale visualization)" << std::endl;
        std::cout << "  - depth_[filename]_normalized.txt (normalized depth)" << std::endl;
        std::cout << "  All files saved in: " << outputDir << std::endl;
    }
    
    std::cout << "\n=== All Processing Completed! ===" << std::endl;
    std::cout << "Results saved in: " << outputDir << std::endl;
    
    return 0;
} 