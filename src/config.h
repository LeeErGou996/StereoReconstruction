#pragma once
#include "8point.h" // 必须在最前面，确保FeatureType可见
#include <string>
#include <fstream>
#include <sstream>
#include <opencv2/core.hpp>
#include "meshReconstruction.h" // for ReconstructionParams
#include "poissonReconstruction.h"

struct ConfigData {
    int reconstructionMode;
    int numDisparities;
    int blockSize;
    FeatureType algorithm;
    std::string leftDir, rightDir, cameraDir, outputDir;
    cv::Mat distCoeffs;
    MeshReconstruction::ReconstructionParams meshParams;
    // denseMatching参数
    int sgbmP1 = 1;
    int sgbmP2 = 4;
    int preFilterCap = 31;
    int uniquenessRatio = 30;
    int speckleWindowSize = 100;
    int speckleRange = 32;
    int disp12MaxDiff = 1;
};

class Config {
public:
    static ConfigData& instance();
    static void load(const std::string& filename);
    static MeshReconstruction::Mesh generatePoissonMeshes(const cv::Mat& depthMap, const cv::Mat& colorImage, const cv::Mat& K);

    // 参数声明
    inline int getReconstructionMode() const { return data_.reconstructionMode; }
    inline int getNumDisparities() const { return data_.numDisparities; }
    inline int getBlockSize() const { return data_.blockSize; }
    inline FeatureType getAlgorithm() const { return data_.algorithm; }
    inline std::string getLeftDir() const { return data_.leftDir; }
    inline std::string getRightDir() const { return data_.rightDir; }
    inline std::string getCameraDir() const { return data_.cameraDir; }
    inline std::string getOutputDir() const { return data_.outputDir; }
    inline const cv::Mat& getDistCoeffs() const { return data_.distCoeffs; }
    inline const MeshReconstruction::ReconstructionParams& getMeshParams() const { return data_.meshParams; }

private:
    static ConfigData data_;

    // 从txt文件读取参数，支持key=value格式
    void loadConfigFromFile(const std::string& filename);
}; 