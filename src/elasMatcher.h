#pragma once
#include <opencv2/opencv.hpp>

class ElasMatcher {
public:
    // ELAS参数结构
    struct ElasParameters {
        int32_t disp_min = 0;               // 最小视差
        int32_t disp_max = 255;             // 最大视差
        float support_threshold = 0.85f;    // 唯一性比率阈值
        int32_t support_texture = 10;       // 支持点最小纹理强度
        int32_t candidate_stepsize = 5;     // 支持点网格步长
        int32_t incon_window_size = 5;      // 不一致性检查窗口大小
        int32_t incon_threshold = 5;        // 视差相似性阈值
        int32_t incon_min_support = 5;      // 最小一致性支持点数
        bool add_corners = false;           // 是否在角落添加支持点
        int32_t grid_size = 20;             // 支持点外推邻域大小
        float beta = 0.02f;                 // 图像似然参数
        float gamma = 3.0f;                 // 先验常数
        float sigma = 1.0f;                 // 先验标准差
        float sradius = 2.0f;               // 先验标准差半径
        int32_t match_texture = 1;          // 稠密匹配最小纹理
        int32_t lr_threshold = 2;           // 左右一致性检查阈值
        float speckle_sim_threshold = 1.0f; // 斑点分割相似性阈值
        int32_t speckle_size = 200;         // 最大斑点大小
        int32_t ipol_gap_width = 3;         // 间隙插值宽度
        bool filter_median = false;         // 是否使用中值滤波
        bool filter_adaptive_mean = true;   // 是否使用自适应均值滤波
        bool postprocess_only_left = true;  // 是否只后处理左图
        bool subsampling = false;           // 是否使用子采样
    };

    // 预设配置
    enum Preset {
        ROBOTICS,    // 机器人环境配置
        MIDDLEBURY   // Middlebury基准配置
    };

    ElasMatcher();
    ElasMatcher(const ElasParameters& params);
    ElasMatcher(Preset preset);
    
    // 设置参数
    void setParameters(const ElasParameters& params);
    void setPreset(Preset preset);
    
    // 从配置文件加载参数
    void loadParametersFromConfig();
    
    // 获取当前参数
    const ElasParameters& getParameters() const { return params_; }
    
    // 输入为灰度图像，输出为CV_32F视差图
    bool computeDisparity(const cv::Mat& left, const cv::Mat& right, cv::Mat& disparity);

private:
    ElasParameters params_;
    
    // 创建预设配置
    ElasParameters createPreset(Preset preset);
    
    // 打印当前参数
    void printCurrentParameters();
}; 