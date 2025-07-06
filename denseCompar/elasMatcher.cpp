#include "elasMatcher.h"
#include "elas/elas.h"
#include "config.h"
#include <vector>
#include <cstdint>

ElasMatcher::ElasMatcher() {
    // 从config读取参数
    loadParametersFromConfig();
}

ElasMatcher::ElasMatcher(const ElasParameters& params) : params_(params) {}

ElasMatcher::ElasMatcher(Preset preset) {
    params_ = createPreset(preset);
}

void ElasMatcher::setParameters(const ElasParameters& params) {
    params_ = params;
}

void ElasMatcher::setPreset(Preset preset) {
    params_ = createPreset(preset);
}

void ElasMatcher::loadParametersFromConfig() {
    // 从config读取参数
    std::string preset = Config::instance().elasPreset;
    
    std::cout << "[ELAS] Loading parameters from config..." << std::endl;
    std::cout << "[ELAS] Preset: " << preset << std::endl;
    
    if (preset == "ROBOTICS") {
        params_ = createPreset(ROBOTICS);
        std::cout << "[ELAS] Using ROBOTICS preset configuration" << std::endl;
    } else if (preset == "MIDDLEBURY") {
        params_ = createPreset(MIDDLEBURY);
        std::cout << "[ELAS] Using MIDDLEBURY preset configuration" << std::endl;
    } else {
        // 自定义参数
        std::cout << "[ELAS] Using CUSTOM configuration from config.txt" << std::endl;
        params_.disp_min = Config::instance().elasDispMin;
        params_.disp_max = Config::instance().elasDispMax;
        params_.support_threshold = Config::instance().elasSupportThreshold;
        params_.support_texture = Config::instance().elasSupportTexture;
        params_.candidate_stepsize = Config::instance().elasCandidateStepsize;
        params_.incon_window_size = Config::instance().elasInconWindowSize;
        params_.incon_threshold = Config::instance().elasInconThreshold;
        params_.incon_min_support = Config::instance().elasInconMinSupport;
        params_.add_corners = Config::instance().elasAddCorners;
        params_.grid_size = Config::instance().elasGridSize;
        params_.beta = Config::instance().elasBeta;
        params_.gamma = Config::instance().elasGamma;
        params_.sigma = Config::instance().elasSigma;
        params_.sradius = Config::instance().elasSradius;
        params_.match_texture = Config::instance().elasMatchTexture;
        params_.lr_threshold = Config::instance().elasLrThreshold;
        params_.speckle_sim_threshold = Config::instance().elasSpeckleSimThreshold;
        params_.speckle_size = Config::instance().elasSpeckleSize;
        params_.ipol_gap_width = Config::instance().elasIpolGapWidth;
        params_.filter_median = Config::instance().elasFilterMedian;
        params_.filter_adaptive_mean = Config::instance().elasFilterAdaptiveMean;
        params_.postprocess_only_left = Config::instance().elasPostprocessOnlyLeft;
        params_.subsampling = Config::instance().elasSubsampling;
    }
    
    // 打印当前参数
    printCurrentParameters();
}

ElasMatcher::ElasParameters ElasMatcher::createPreset(Preset preset) {
    ElasParameters params;
    
    if (preset == ROBOTICS) {
        // 机器人环境配置
        params.disp_min = 0;
        params.disp_max = 255;
        params.support_threshold = 0.85f;
        params.support_texture = 10;
        params.candidate_stepsize = 5;
        params.incon_window_size = 5;
        params.incon_threshold = 5;
        params.incon_min_support = 5;
        params.add_corners = false;
        params.grid_size = 20;
        params.beta = 0.02f;
        params.gamma = 3.0f;
        params.sigma = 1.0f;
        params.sradius = 2.0f;
        params.match_texture = 1;
        params.lr_threshold = 2;
        params.speckle_sim_threshold = 1.0f;
        params.speckle_size = 200;
        params.ipol_gap_width = 3;
        params.filter_median = false;
        params.filter_adaptive_mean = true;
        params.postprocess_only_left = true;
        params.subsampling = false;
    } else if (preset == MIDDLEBURY) {
        // Middlebury基准配置
        params.disp_min = 0;
        params.disp_max = 255;
        params.support_threshold = 0.95f;
        params.support_texture = 10;
        params.candidate_stepsize = 5;
        params.incon_window_size = 5;
        params.incon_threshold = 5;
        params.incon_min_support = 5;
        params.add_corners = true;
        params.grid_size = 20;
        params.beta = 0.02f;
        params.gamma = 5.0f;
        params.sigma = 1.0f;
        params.sradius = 3.0f;
        params.match_texture = 0;
        params.lr_threshold = 2;
        params.speckle_sim_threshold = 1.0f;
        params.speckle_size = 200;
        params.ipol_gap_width = 5000;
        params.filter_median = true;
        params.filter_adaptive_mean = false;
        params.postprocess_only_left = false;
        params.subsampling = false;
    }
    
    return params;
}

void ElasMatcher::printCurrentParameters() {
    std::cout << "[ELAS] Current parameters:" << std::endl;
    std::cout << "  Disparity range: [" << params_.disp_min << ", " << params_.disp_max << "]" << std::endl;
    std::cout << "  Support threshold: " << params_.support_threshold << std::endl;
    std::cout << "  Support texture: " << params_.support_texture << std::endl;
    std::cout << "  Candidate stepsize: " << params_.candidate_stepsize << std::endl;
    std::cout << "  Incon window size: " << params_.incon_window_size << std::endl;
    std::cout << "  Incon threshold: " << params_.incon_threshold << std::endl;
    std::cout << "  Incon min support: " << params_.incon_min_support << std::endl;
    std::cout << "  Add corners: " << (params_.add_corners ? "true" : "false") << std::endl;
    std::cout << "  Grid size: " << params_.grid_size << std::endl;
    std::cout << "  Beta: " << params_.beta << std::endl;
    std::cout << "  Gamma: " << params_.gamma << std::endl;
    std::cout << "  Sigma: " << params_.sigma << std::endl;
    std::cout << "  Sradius: " << params_.sradius << std::endl;
    std::cout << "  Match texture: " << params_.match_texture << std::endl;
    std::cout << "  LR threshold: " << params_.lr_threshold << std::endl;
    std::cout << "  Speckle sim threshold: " << params_.speckle_sim_threshold << std::endl;
    std::cout << "  Speckle size: " << params_.speckle_size << std::endl;
    std::cout << "  Ipol gap width: " << params_.ipol_gap_width << std::endl;
    std::cout << "  Filter median: " << (params_.filter_median ? "true" : "false") << std::endl;
    std::cout << "  Filter adaptive mean: " << (params_.filter_adaptive_mean ? "true" : "false") << std::endl;
    std::cout << "  Postprocess only left: " << (params_.postprocess_only_left ? "true" : "false") << std::endl;
    std::cout << "  Subsampling: " << (params_.subsampling ? "true" : "false") << std::endl;
    std::cout << "[ELAS] Parameters loaded successfully" << std::endl;
}

bool ElasMatcher::computeDisparity(const cv::Mat& left, const cv::Mat& right, cv::Mat& disparity) {
    if (left.empty() || right.empty() || left.size() != right.size()) return false;
    
    int width = left.cols, height = left.rows;
    std::cout << "[ELAS] Processing images: " << width << "x" << height << std::endl;
    
    std::vector<float> D_left(width * height, 0);
    std::vector<float> D_right(width * height, 0);

    // 创建ELAS参数并转换为ELAS库的格式
    Elas::parameters elas_params;
    elas_params.disp_min = params_.disp_min;
    elas_params.disp_max = params_.disp_max;
    elas_params.support_threshold = params_.support_threshold;
    elas_params.support_texture = params_.support_texture;
    elas_params.candidate_stepsize = params_.candidate_stepsize;
    elas_params.incon_window_size = params_.incon_window_size;
    elas_params.incon_threshold = params_.incon_threshold;
    elas_params.incon_min_support = params_.incon_min_support;
    elas_params.add_corners = params_.add_corners;
    elas_params.grid_size = params_.grid_size;
    elas_params.beta = params_.beta;
    elas_params.gamma = params_.gamma;
    elas_params.sigma = params_.sigma;
    elas_params.sradius = params_.sradius;
    elas_params.match_texture = params_.match_texture;
    elas_params.lr_threshold = params_.lr_threshold;
    elas_params.speckle_sim_threshold = params_.speckle_sim_threshold;
    elas_params.speckle_size = params_.speckle_size;
    elas_params.ipol_gap_width = params_.ipol_gap_width;
    elas_params.filter_median = params_.filter_median;
    elas_params.filter_adaptive_mean = params_.filter_adaptive_mean;
    elas_params.postprocess_only_left = params_.postprocess_only_left;
    elas_params.subsampling = params_.subsampling;

    std::cout << "[ELAS] Parameters converted to ELAS library format" << std::endl;
    std::cout << "[ELAS] Starting ELAS processing..." << std::endl;

    Elas elas(elas_params);
    int32_t dims[3] = {width, height, width};
    elas.process(left.data, right.data, D_left.data(), D_right.data(), dims);
    
    std::cout << "[ELAS] Processing completed successfully" << std::endl;

    disparity = cv::Mat(height, width, CV_32F, D_left.data()).clone();
    return true;
} 