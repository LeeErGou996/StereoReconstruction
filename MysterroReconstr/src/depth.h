#ifndef DEPTH_H
#define DEPTH_H

#include <vector>
#include <iostream>
#include <limits>
#include <algorithm>
#include <cmath>

// 自定义数据结构替代OpenCV
struct MyMat {
    int rows = 0;
    int cols = 0;
    int type = 0; // 0=float, 1=uchar, 2=short
    std::vector<float> data; // 统一使用float存储

    MyMat() = default;
    MyMat(int r, int c, int t = 0) : rows(r), cols(c), type(t), data(r * c, 0.0f) {}
    
    bool empty() const { return data.empty() || rows == 0 || cols == 0; }
    int getType() const { return type; }
    
    // 访问元素
    float& at(int r, int c) {
        return data[r * cols + c];
    }
    const float& at(int r, int c) const {
        return data[r * cols + c];
    }
    
    // 类型转换
    void convertTo(MyMat& dst, float scale = 1.0f, float shift = 0.0f) const {
        dst = MyMat(rows, cols, 0); // 转换为float类型
        for (size_t i = 0; i < data.size(); ++i) {
            dst.data[i] = data[i] * scale + shift;
        }
    }
    
    // 克隆
    MyMat clone() const {
        MyMat result(rows, cols, type);
        result.data = data;
        return result;
    }
    
    // 设置值
    void setTo(float value, const std::vector<bool>& mask = {}) {
        if (mask.empty()) {
            std::fill(data.begin(), data.end(), value);
        } else {
            for (size_t i = 0; i < data.size() && i < mask.size(); ++i) {
                if (mask[i]) data[i] = value;
            }
        }
    }
    
    // 获取大小
    std::pair<int, int> size() const { return {rows, cols}; }
};

// 自定义3D点结构
struct Point3f {
    float x, y, z;
    Point3f() : x(0), y(0), z(0) {}
    Point3f(float x_, float y_, float z_) : x(x_), y(y_), z(z_) {}
};

// 自定义矩阵类（用于Q矩阵等）- 参考rectification.cpp的风格
using Matrix4x4 = double[4][4];

// 简单的4x4矩阵类
class Matrix {
private:
    Matrix4x4 data_;

public:
    // 默认构造函数
    Matrix() {
        for (int i = 0; i < 4; ++i) {
            for (int j = 0; j < 4; ++j) {
                data_[i][j] = 0.0;
            }
        }
    }
    
    // 构造函数
    Matrix(int rows, int cols) {
        if (rows == 4 && cols == 4) {
            for (int i = 0; i < 4; ++i) {
                for (int j = 0; j < 4; ++j) {
                    data_[i][j] = 0.0;
                }
            }
            std::cout << "Matrix构造函数: 4x4矩阵" << std::endl;
        } else {
            std::cout << "Matrix构造函数: 只支持4x4矩阵，输入: " << rows << "x" << cols << std::endl;
        }
    }
    
    // 访问元素
    double& at(int row, int col) { 
        if (row >= 0 && row < 4 && col >= 0 && col < 4) {
            return data_[row][col]; 
        } else {
            std::cerr << "Matrix访问越界: row=" << row << ", col=" << col << std::endl;
            static double dummy = 0.0;
            return dummy;
        }
    }
    
    const double& at(int row, int col) const { 
        if (row >= 0 && row < 4 && col >= 0 && col < 4) {
            return data_[row][col]; 
        } else {
            std::cerr << "Matrix访问越界: row=" << row << ", col=" << col << std::endl;
            static double dummy = 0.0;
            return dummy;
        }
    }
    
    // 获取尺寸
    int rows() const { return 4; }
    int cols() const { return 4; }
    bool empty() const { return false; }
    
    // 调试方法
    void printInfo() const {
        std::cout << "Matrix信息: 4x4矩阵" << std::endl;
    }
    
    // 打印矩阵内容
    void print() const {
        std::cout << "Matrix内容:" << std::endl;
        for (int i = 0; i < 4; ++i) {
            printf("[%8.3f %8.3f %8.3f %8.3f]\n", 
                   data_[i][0], data_[i][1], data_[i][2], data_[i][3]);
        }
    }
};

namespace Depth {

// 自定义3D重投影函数
bool reprojectImageTo3D(const MyMat& disparity, std::vector<Point3f>& xyz, const Matrix& Q, bool handleMissingValues = true);

// 自定义统计函数
bool minMaxLoc(const MyMat& mat, double& minVal, double& maxVal, 
               int* minLoc = nullptr, int* maxLoc = nullptr, 
               const std::vector<bool>& mask = {});

// 自定义计数函数
int countNonZero(const MyMat& mat);

// 自定义均值计算
double mean(const MyMat& mat, const std::vector<bool>& mask = {});

// 自定义位运算
std::vector<bool> bitwiseAnd(const std::vector<bool>& mask1, const std::vector<bool>& mask2);

// 自定义图像保存函数
bool imwrite(const std::string& filename, const MyMat& img);

// 主要接口函数
bool computeDepthMap(const MyMat& disparity,
                     const Matrix& Q,
                     MyMat& depthMap);

// 专门用于Middlebury评估的深度转换
bool computeDepthMapForMiddlebury(const MyMat& disparity,
                                  const Matrix& Q,
                                  MyMat& depthMapOut,
                                  double baseline);

// 将深度图转换为与真实深度图相同的尺度
bool normalizeDepthToGroundTruth(const MyMat& depthMap, 
                                 const MyMat& groundTruth,
                                 MyMat& normalizedDepth);

// 深度图质量评估
bool evaluateDepthQuality(const MyMat& depthMap, const MyMat& originalDisparity);

// 点云质心计算和半径过滤函数
Point3f computePointCloudCentroid(const std::vector<Point3f>& points);
float computeDistanceToCentroid(const Point3f& point, const Point3f& centroid);
std::vector<Point3f> filterPointCloudByCentroidRadius(const std::vector<Point3f>& points, float radius = 5.0f);

} // namespace Depth

#endif // DEPTH_H 