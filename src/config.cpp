#include "config.h"
#include <fstream>
#include <sstream>
#include <iostream>
#include <numeric>

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
        if (key == "reconstructionMode") Config::data_.reconstructionMode = std::stoi(value);
        else if (key == "numDisparities") Config::data_.numDisparities = std::stoi(value);
        else if (key == "blockSize") Config::data_.blockSize = std::stoi(value);
        else if (key == "algorithm") {
            if (value == "ORB") Config::data_.algorithm = FeatureType::ORB;
            else if (value == "SIFT") Config::data_.algorithm = FeatureType::SIFT;
            else if (value == "SURF") Config::data_.algorithm = FeatureType::SURF;
        }
        else if (key == "featureType") {
            // 兼容 denseCompar 配置
            if (value == "ORB") Config::data_.algorithm = FeatureType::ORB;
            else if (value == "SIFT") Config::data_.algorithm = FeatureType::SIFT;
            else if (value == "SURF") Config::data_.algorithm = FeatureType::SURF;
        }
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
        // meshParams相关
        else if (key == "meshFormat") Config::data_.meshParams.meshFormat = value;
        else if (key == "triangulationStep") Config::data_.meshParams.triangulationStep = std::stoi(value);
        else if (key == "maxDepthDifference") Config::data_.meshParams.maxDepthDifference = std::stof(value);
        else if (key == "useColor") Config::data_.meshParams.useColor = (value == "1" || value == "true");
        else if (key == "depthThreshold") Config::data_.meshParams.depthThreshold = std::stof(value);
        else if (key == "depthDiffThreshold") Config::data_.meshParams.depthDiffThreshold = std::stof(value);
        else if (key == "normalNeighbors") Config::data_.meshParams.normalNeighbors = std::stoi(value);
        // Poisson mesh参数
        else if (key == "poissonDepth") Config::data_.meshParams.poissonDepth = std::stof(value);
        else if (key == "poissonSolverDivide") Config::data_.meshParams.poissonSolverDivide = std::stof(value);
        else if (key == "poissonSamplesPerNode") Config::data_.meshParams.poissonSamplesPerNode = std::stof(value);
        else if (key == "poissonFullDepth") Config::data_.meshParams.poissonFullDepth = std::stof(value);
        else if (key == "poissonTrim") Config::data_.meshParams.poissonTrim = std::stof(value);
        else if (key == "poissonUseConfidence") Config::data_.meshParams.poissonUseConfidence = (value == "1" || value == "true");
        else if (key == "poissonManifold") Config::data_.meshParams.poissonManifold = (value == "1" || value == "true");
        else if (key == "poissonOutputPolygons") Config::data_.meshParams.poissonOutputPolygons = (value == "1" || value == "true");
        // denseMatching参数
        else if (key == "sgbmP1") Config::data_.sgbmP1 = std::stoi(value);
        else if (key == "sgbmP2") Config::data_.sgbmP2 = std::stoi(value);
        else if (key == "preFilterCap") Config::data_.preFilterCap = std::stoi(value);
        else if (key == "uniquenessRatio") Config::data_.uniquenessRatio = std::stoi(value);
        else if (key == "speckleWindowSize") Config::data_.speckleWindowSize = std::stoi(value);
        else if (key == "speckleRange") Config::data_.speckleRange = std::stoi(value);
        else if (key == "disp12MaxDiff") Config::data_.disp12MaxDiff = std::stoi(value);
        else if (key == "eightPointMethod") Config::data_.eightPointMethod = value;
        else if (key == "use8pointAlgorithm") {
            // 兼容 denseCompar 配置
            if (value == "true" || value == "1") Config::data_.eightPointMethod = "opencv";
            // false 时不覆盖，保持原有配置
        }
        else if (key == "denseMethod") Config::data_.denseMethod = value;
        else if (key == "depthMethod") Config::data_.depthMethod = value;
        else if (key == "elasPreset") Config::data_.elasPreset = value;
        else if (key == "elasDispMin") Config::data_.elasDispMin = std::stoi(value);
        else if (key == "elasDispMax") Config::data_.elasDispMax = std::stoi(value);
        else if (key == "elasSupportThreshold") Config::data_.elasSupportThreshold = std::stof(value);
        else if (key == "elasSupportTexture") Config::data_.elasSupportTexture = std::stoi(value);
        else if (key == "elasCandidateStepsize") Config::data_.elasCandidateStepsize = std::stoi(value);
        else if (key == "elasInconWindowSize") Config::data_.elasInconWindowSize = std::stoi(value);
        else if (key == "elasInconThreshold") Config::data_.elasInconThreshold = std::stoi(value);
        else if (key == "elasInconMinSupport") Config::data_.elasInconMinSupport = std::stoi(value);
        else if (key == "elasAddCorners") Config::data_.elasAddCorners = (value == "1" || value == "true");
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
        else if (key == "elasFilterMedian") Config::data_.elasFilterMedian = (value == "1" || value == "true");
        else if (key == "elasFilterAdaptiveMean") Config::data_.elasFilterAdaptiveMean = (value == "1" || value == "true");
        else if (key == "elasPostprocessOnlyLeft") Config::data_.elasPostprocessOnlyLeft = (value == "1" || value == "true");
        else if (key == "elasSubsampling") Config::data_.elasSubsampling = (value == "1" || value == "true");
    }
}

MeshReconstruction::Mesh Config::generatePoissonMeshes(const cv::Mat& depthMap, const cv::Mat& colorImage, const cv::Mat& K) {
    // 1. 用 config.txt 里的参数设置 Poisson 重建参数
    const auto& params = Config::instance().meshParams;
    // MeshReconstruction::setPoissonReconstructionParams(params.voxelSize, params.triangulationStep, params.depthThreshold);
    // MeshReconstruction::setPCLPoissonParams(
    //     static_cast<int>(params.poissonDepth),
    //     static_cast<int>(params.poissonSolverDivide),
    //     params.poissonSamplesPerNode,
    //     params.poissonUseConfidence,
    //     params.poissonManifold,
    //     params.poissonOutputPolygons
    // );
    // 2. 直接调用主重建函数
    MeshReconstruction::Mesh mesh = MeshReconstruction::generatePoissonMesh(depthMap, colorImage, K);
    // 自动裁剪（trim）
    if (params.poissonTrim > 0.0f && !mesh.vertices.empty()) {
        // 计算mesh质心
        float cx = 0, cy = 0, cz = 0;
        for (const auto& v : mesh.vertices) {
            cx += v.x; cy += v.y; cz += v.z;
        }
        cx /= mesh.vertices.size();
        cy /= mesh.vertices.size();
        cz /= mesh.vertices.size();
        // 计算所有点到质心的距离均值
        std::vector<float> dists;
        dists.reserve(mesh.vertices.size());
        for (const auto& v : mesh.vertices) {
            float d = std::sqrt((v.x-cx)*(v.x-cx)+(v.y-cy)*(v.y-cy)+(v.z-cz)*(v.z-cz));
            dists.push_back(d);
        }
        float meanDist = std::accumulate(dists.begin(), dists.end(), 0.0f) / dists.size();
        float trimThresh = params.poissonTrim * meanDist;
        // 标记保留的顶点
        std::vector<bool> keep(mesh.vertices.size(), false);
        for (size_t i=0; i<mesh.vertices.size(); ++i) {
            if (dists[i] < trimThresh) keep[i] = true;
        }
        // 新顶点和索引映射
        std::vector<MeshReconstruction::Point3D> newVerts;
        std::vector<int> old2new(mesh.vertices.size(), -1);
        for (size_t i=0; i<mesh.vertices.size(); ++i) {
            if (keep[i]) {
                old2new[i] = (int)newVerts.size();
                newVerts.push_back(mesh.vertices[i]);
            }
        }
        // 新三角形
        std::vector<MeshReconstruction::Triangle> newFaces;
        for (const auto& tri : mesh.faces) {
            if (keep[tri.v1] && keep[tri.v2] && keep[tri.v3]) {
                newFaces.emplace_back(old2new[tri.v1], old2new[tri.v2], old2new[tri.v3]);
            }
        }
        mesh.vertices = std::move(newVerts);
        mesh.faces = std::move(newFaces);
    }
    // 3. 删除/注释掉硬编码参数的调用
    // MeshReconstruction::setPoissonReconstructionParams(0.003f, 1, 2.5f);
    // MeshReconstruction::Mesh highQualityMesh = MeshReconstruction::generatePoissonMesh(depthMap, colorImage, K);
    // MeshReconstruction::setPoissonReconstructionParams(0.01f, 4, 2.0f);
    // MeshReconstruction::Mesh fastMesh = MeshReconstruction::generatePoissonMesh(depthMap, colorImage, K);
    return mesh;
} 