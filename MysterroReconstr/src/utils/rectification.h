#pragma once
#include <vector>
#include <array>

// 3x3矩阵类型
using Mat3 = std::array<std::array<double, 3>, 3>;
// 3x1向量类型
using Vec3 = std::array<double, 3>;

// 计算极线矫正变换（Hartley方法）
// K1, K2: 左右相机内参，R, t: 外参，E: 本征矩阵
// 输出：H1, H2为左右图像的单应矩阵
void computeRectification(
    const Mat3& K1, const Mat3& K2,
    const Mat3& R, const Vec3& t,
    Mat3& H1, Mat3& H2
); 