#pragma once
#include <string>
#include "ELAS/src/elas.h"
// 生成视差图，输入左右校正图像路径，输出视差图路径
// disparityType: "left" 或 "right"
void compute_disparity_elas(const std::string& left_path, const std::string& right_path, const std::string& out_path, const Elas::parameters& param, const std::string& disparityType = "left");
// 兼容旧接口
void compute_disparity_elas(const std::string& left_path, const std::string& right_path, const std::string& out_path, const std::string& disparityType = "left"); 