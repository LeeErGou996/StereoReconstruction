#ifndef EIGHT_POINT_H
#define EIGHT_POINT_H

#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <limits>
#include <string>
#include <cassert>
#include <random>
#include <iomanip>
#include <numeric>

namespace EightPoint {

// --- 基础数据结构 ---

/**
 * @brief 2D点结构
 */
struct Point2f {
    float x, y;
    
    Point2f();
    Point2f(float x_, float y_);
    
    // 运算符重载
    Point2f operator+(const Point2f& other) const;
    Point2f operator-(const Point2f& other) const;
    Point2f operator*(float s) const;
    
    // 计算向量长度
    float norm() const;
};

/**
 * @brief 矩阵类
 */
class Matrix {
private:
    std::vector<double> data;
    int rows_, cols_;

public:
    // 构造函数
    Matrix();
    Matrix(int rows, int cols);
    Matrix(int rows, int cols, const std::vector<double>& values);
    
    // 访问元素
    double& at(int row, int col);
    const double& at(int row, int col) const;
    
    // 基本属性
    int rows() const;
    int cols() const;
    bool empty() const;
    
    // 矩阵运算
    Matrix operator*(const Matrix& other) const;
    Matrix t() const;  // 转置
    Matrix inv() const;  // 求逆
    double det() const;  // 行列式
    
    // 获取行
    Matrix row(int r) const;
    
    // 重塑矩阵
    Matrix reshape(int new_rows, int new_cols) const;
    
    // 静态方法
    static Matrix diag(const std::vector<double>& diagonal);
    static Matrix eye(int n);
    
    // 打印矩阵
    void print() const;
};

/**
 * @brief SVD分解结果
 */
struct SVDResult {
    Matrix U;   // 左奇异向量
    Matrix S;   // 奇异值
    Matrix Vt;  // 右奇异向量的转置
};

// --- 核心算法函数 ---

/**
 * @brief 计算矩阵的SVD分解
 * @param A 输入矩阵
 * @return SVD分解结果
 */
SVDResult computeSVD(const Matrix& A);

/**
 * @brief 使用8点算法计算本质矩阵
 * @param ptsL 左图像点
 * @param ptsR 右图像点
 * @param K 相机内参矩阵
 * @return 本质矩阵
 */
Matrix computeEssentialMatrix8Point(const std::vector<Point2f>& ptsL,
                                   const std::vector<Point2f>& ptsR,
                                   const Matrix& K);

/**
 * @brief 从本质矩阵恢复相机位姿
 * @param E 本质矩阵
 * @param ptsL 左图像点
 * @param ptsR 右图像点
 * @param K 相机内参矩阵
 * @param R 输出旋转矩阵
 * @param t 输出平移向量
 * @return 是否成功恢复位姿
 */
bool recoverPose(const Matrix& E, 
                const std::vector<Point2f>& ptsL, 
                const std::vector<Point2f>& ptsR, 
                const Matrix& K, 
                Matrix& R, 
                Matrix& t);

/**
 * @brief 估计相机位姿（主要接口函数）
 * @param ptsL 左图像点
 * @param ptsR 右图像点
 * @param K 相机内参矩阵
 * @param R 输出旋转矩阵
 * @param t 输出平移向量
 * @return 是否成功估计位姿
 */
bool estimatePose(const std::vector<Point2f>& ptsL,
                 const std::vector<Point2f>& ptsR,
                 const Matrix& K,
                 Matrix& R,
                 Matrix& t);

// --- 距离计算函数 ---

/**
 * @brief 计算Sampson距离
 * @param pt1 第一个图像中的点
 * @param pt2 第二个图像中的点
 * @param E 本质矩阵
 * @return Sampson距离
 */
double computeSampsonDistance(const Point2f& pt1, const Point2f& pt2, const Matrix& E);

/**
 * @brief 计算对称对极线距离
 * @param pt1 第一个图像中的点
 * @param pt2 第二个图像中的点
 * @param E 本质矩阵
 * @return 对称对极线距离
 */
double computeSymmetricEpipolarDistance(const Point2f& pt1, const Point2f& pt2, const Matrix& E);

// --- 评估函数 ---

/**
 * @brief 评估对极线距离误差
 * @param ptsL 左图像点
 * @param ptsR 右图像点
 * @param E 本质矩阵
 * @param methodName 方法名称（用于显示）
 */
void evaluateEpipolarError(const std::vector<Point2f>& ptsL, 
                          const std::vector<Point2f>& ptsR, 
                          const Matrix& E,
                          const std::string& methodName);

// --- 辅助函数 ---

/**
 * @brief 创建相机内参矩阵
 * @param fx 水平焦距
 * @param fy 垂直焦距
 * @param cx 主点x坐标
 * @param cy 主点y坐标
 * @return 内参矩阵
 */
inline Matrix createCameraMatrix(double fx, double fy, double cx, double cy) {
    Matrix K(3, 3);
    K.at(0, 0) = fx; K.at(0, 2) = cx;
    K.at(1, 1) = fy; K.at(1, 2) = cy;
    K.at(2, 2) = 1.0;
    return K;
}

/**
 * @brief 从欧拉角创建旋转矩阵
 * @param roll 滚转角(弧度)
 * @param pitch 俯仰角(弧度)  
 * @param yaw 偏航角(弧度)
 * @return 旋转矩阵
 */
inline Matrix eulerToRotationMatrix(double roll, double pitch, double yaw) {
    double cr = std::cos(roll), sr = std::sin(roll);
    double cp = std::cos(pitch), sp = std::sin(pitch);
    double cy = std::cos(yaw), sy = std::sin(yaw);
    
    Matrix R(3, 3);
    R.at(0, 0) = cy * cp;
    R.at(0, 1) = cy * sp * sr - sy * cr;
    R.at(0, 2) = cy * sp * cr + sy * sr;
    R.at(1, 0) = sy * cp;
    R.at(1, 1) = sy * sp * sr + cy * cr;
    R.at(1, 2) = sy * sp * cr - cy * sr;
    R.at(2, 0) = -sp;
    R.at(2, 1) = cp * sr;
    R.at(2, 2) = cp * cr;
    
    return R;
}

/**
 * @brief 从旋转矩阵提取欧拉角
 * @param R 旋转矩阵
 * @param roll 输出滚转角(弧度)
 * @param pitch 输出俯仰角(弧度)
 * @param yaw 输出偏航角(弧度)
 */
inline void rotationMatrixToEuler(const Matrix& R, double& roll, double& pitch, double& yaw) {
    pitch = std::asin(-R.at(2, 0));
    
    if (std::cos(pitch) > 1e-6) {
        yaw = std::atan2(R.at(1, 0), R.at(0, 0));
        roll = std::atan2(R.at(2, 1), R.at(2, 2));
    } else {
        yaw = std::atan2(-R.at(0, 1), R.at(1, 1));
        roll = 0.0;
    }
}

/**
 * @brief 计算重投影误差
 * @param ptsL 左图像点
 * @param ptsR 右图像点
 * @param R 旋转矩阵
 * @param t 平移向量
 * @param K 相机内参矩阵
 * @return 平均重投影误差
 */
double computeReprojectionError(const std::vector<Point2f>& ptsL,
                               const std::vector<Point2f>& ptsR,
                               const Matrix& R,
                               const Matrix& t,
                               const Matrix& K);

} // namespace EightPoint

#endif // EIGHT_POINT_H