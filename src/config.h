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
    std::string eightPointMethod = "opencv";
    // 新增：稠密匹配方法选择（BM或SGBM）
    std::string denseMethod = "SGBM";
    // 新增：深度计算方法选择（OpenCV或Custom）
    std::string depthMethod = "OpenCV";
    // ELAS算法参数
    std::string elasPreset = "CUSTOM";
    int32_t elasDispMin = 0;
    int32_t elasDispMax = 256;
    float elasSupportThreshold = 0.5f;
    int32_t elasSupportTexture = 31;
    int32_t elasCandidateStepsize = 5;
    int32_t elasInconWindowSize = 5;
    int32_t elasInconThreshold = 5;
    int32_t elasInconMinSupport = 5;
    bool elasAddCorners = false;
    int32_t elasGridSize = 20;
    float elasBeta = 0.02f;
    float elasGamma = 3.0f;
    float elasSigma = 1.0f;
    float elasSradius = 2.0f;
    int32_t elasMatchTexture = 31;
    int32_t elasLrThreshold = 5;
    float elasSpeckleSimThreshold = 1.0f;
    int32_t elasSpeckleSize = 100;
    int32_t elasIpolGapWidth = 3;
    bool elasFilterMedian = false;
    bool elasFilterAdaptiveMean = true;
    bool elasPostprocessOnlyLeft = true;
    bool elasSubsampling = false;
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
    inline std::string getEightPointMethod() const { return data_.eightPointMethod; }
    inline std::string getDenseMethod() const { return data_.denseMethod; }
    inline std::string getDepthMethod() const { return data_.depthMethod; }
    inline std::string getElasPreset() const { return data_.elasPreset; }
    inline int32_t getElasDispMin() const { return data_.elasDispMin; }
    inline int32_t getElasDispMax() const { return data_.elasDispMax; }
    inline float getElasSupportThreshold() const { return data_.elasSupportThreshold; }
    inline int32_t getElasSupportTexture() const { return data_.elasSupportTexture; }
    inline int32_t getElasCandidateStepsize() const { return data_.elasCandidateStepsize; }
    inline int32_t getElasInconWindowSize() const { return data_.elasInconWindowSize; }
    inline int32_t getElasInconThreshold() const { return data_.elasInconThreshold; }
    inline int32_t getElasInconMinSupport() const { return data_.elasInconMinSupport; }
    inline bool getElasAddCorners() const { return data_.elasAddCorners; }
    inline int32_t getElasGridSize() const { return data_.elasGridSize; }
    inline float getElasBeta() const { return data_.elasBeta; }
    inline float getElasGamma() const { return data_.elasGamma; }
    inline float getElasSigma() const { return data_.elasSigma; }
    inline float getElasSradius() const { return data_.elasSradius; }
    inline int32_t getElasMatchTexture() const { return data_.elasMatchTexture; }
    inline int32_t getElasLrThreshold() const { return data_.elasLrThreshold; }
    inline float getElasSpeckleSimThreshold() const { return data_.elasSpeckleSimThreshold; }
    inline int32_t getElasSpeckleSize() const { return data_.elasSpeckleSize; }
    inline int32_t getElasIpolGapWidth() const { return data_.elasIpolGapWidth; }
    inline bool getElasFilterMedian() const { return data_.elasFilterMedian; }
    inline bool getElasFilterAdaptiveMean() const { return data_.elasFilterAdaptiveMean; }
    inline bool getElasPostprocessOnlyLeft() const { return data_.elasPostprocessOnlyLeft; }
    inline bool getElasSubsampling() const { return data_.elasSubsampling; }

private:
    static ConfigData data_;

    // 从txt文件读取参数，支持key=value格式
    void loadConfigFromFile(const std::string& filename);
}; 