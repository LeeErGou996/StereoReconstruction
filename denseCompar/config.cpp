#include "config.h"
#include <fstream>
#include <sstream>
#include <iostream>

ConfigData Config::data_;

ConfigData& Config::instance() {
    return data_;
}

void Config::load(const std::string& filename) {
    std::ifstream infile(filename);
    if (!infile.is_open()) {
        std::cerr << "[Config] Failed to open config file: " << filename << std::endl;
        return;
    }
    std::string line;
    while (std::getline(infile, line)) {
        if (line.empty() || line[0] == '#') continue;
        std::istringstream iss(line);
        std::string key, eq, value;
        if (!(iss >> key)) continue;
        size_t eqpos = key.find('=');
        if (eqpos != std::string::npos) {
            value = key.substr(eqpos+1);
            key = key.substr(0, eqpos);
        } else if (!(iss >> eq >> value) || eq != "=") {
            continue;
        }
        // 去除value前后空格
        value.erase(0, value.find_first_not_of(" \t"));
        value.erase(value.find_last_not_of(" \t")+1);
        // 解析并赋值
        if (key == "numDisparities") Config::data_.numDisparities = std::stoi(value);
        else if (key == "blockSize") Config::data_.blockSize = std::stoi(value);
        else if (key == "leftDir") Config::data_.leftDir = value;
        else if (key == "rightDir") Config::data_.rightDir = value;
        else if (key == "cameraDir") Config::data_.cameraDir = value;
        else if (key == "outputDir") Config::data_.outputDir = value;
        else if (key == "distCoeffs") {
            // 支持5个逗号分隔的浮点数
            std::istringstream vs(value);
            double v[5] = {0};
            char c;
            for (int i=0; i<5 && vs >> v[i]; ++i) vs >> c;
            Config::data_.distCoeffs = cv::Mat(5, 1, CV_64F, v).clone();
        }
        // denseMatching参数
        else if (key == "sgbmP1") Config::data_.sgbmP1 = std::stoi(value);
        else if (key == "sgbmP2") Config::data_.sgbmP2 = std::stoi(value);
        else if (key == "preFilterCap") Config::data_.preFilterCap = std::stoi(value);
        else if (key == "uniquenessRatio") Config::data_.uniquenessRatio = std::stoi(value);
        else if (key == "speckleWindowSize") Config::data_.speckleWindowSize = std::stoi(value);
        else if (key == "speckleRange") Config::data_.speckleRange = std::stoi(value);
        else if (key == "disp12MaxDiff") Config::data_.disp12MaxDiff = std::stoi(value);
        // 特征匹配和位姿估计参数
        else if (key == "use8pointAlgorithm") Config::data_.use8pointAlgorithm = (value == "true");
        else if (key == "featureType") Config::data_.featureType = value;
        
        // ELAS算法参数（可选，如果注释掉则使用SGBM参数）
        else if (key == "elasPreset") Config::data_.elasPreset = value;
        else if (key == "elasDispMin") Config::data_.elasDispMin = std::stoi(value);
        else if (key == "elasDispMax") Config::data_.elasDispMax = std::stoi(value);
        else if (key == "elasSupportThreshold") Config::data_.elasSupportThreshold = std::stof(value);
        else if (key == "elasSupportTexture") Config::data_.elasSupportTexture = std::stoi(value);
        else if (key == "elasCandidateStepsize") Config::data_.elasCandidateStepsize = std::stoi(value);
        else if (key == "elasInconWindowSize") Config::data_.elasInconWindowSize = std::stoi(value);
        else if (key == "elasInconThreshold") Config::data_.elasInconThreshold = std::stoi(value);
        else if (key == "elasInconMinSupport") Config::data_.elasInconMinSupport = std::stoi(value);
        else if (key == "elasAddCorners") Config::data_.elasAddCorners = (value == "true");
        else if (key == "elasGridSize") Config::data_.elasGridSize = std::stoi(value);
        else if (key == "elasBeta") Config::data_.elasBeta = std::stof(value);
        else if (key == "elasGamma") Config::data_.elasGamma = std::stof(value);
        else if (key == "elasSigma") Config::data_.elasSigma = std::stof(value);
        else if (key == "elasSradius") Config::data_.elasSradius = std::stof(value);
        else if (key == "elasMatchTexture") Config::data_.elasMatchTexture = std::stoi(value);
        else if (key == "elasLrThreshold") Config::data_.elasLrThreshold = std::stoi(value);
        else if (key == "elasSpeckleSimThreshold") Config::data_.elasSpeckleSimThreshold = std::stof(value);
        else if (key == "elasSpeckleSize") Config::data_.elasSpeckleSize = std::stoi(value);
        else if (key == "elasIpolGapWidth") Config::data_.elasIpolGapWidth = std::stoi(value);
        else if (key == "elasFilterMedian") Config::data_.elasFilterMedian = (value == "true");
        else if (key == "elasFilterAdaptiveMean") Config::data_.elasFilterAdaptiveMean = (value == "true");
        else if (key == "elasPostprocessOnlyLeft") Config::data_.elasPostprocessOnlyLeft = (value == "true");
        else if (key == "elasSubsampling") Config::data_.elasSubsampling = (value == "true");
    }
} 