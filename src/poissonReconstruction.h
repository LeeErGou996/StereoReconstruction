#pragma once
#include "meshReconstruction.h"
#include <vector>
#include <opencv2/opencv.hpp>

// PCL前向声明
#ifdef HAVE_PCL
namespace pcl {
    struct PolygonMesh;
}
#endif

namespace MeshReconstruction {

// ==================== Poisson重建相关接口 ====================

// 主重建函数
Mesh generatePoissonMesh(const cv::Mat& depthMap, const cv::Mat& colorImage, const cv::Mat& K);

// 法线估计函数
std::vector<PointWithNormal> estimateNormalsFromDepth(const cv::Mat& depthMap, const cv::Mat& K);
std::vector<PointWithNormal> estimateNormals(const std::vector<Point3D>& points, int kNeighbors = 10);

// ==================== 参数配置接口 ====================

// 重置为默认参数
void resetPoissonReconstructionParams();

// 调试模式 - 使用宽松参数便于诊断问题
void setDebugPoissonParams();

// ==================== PCL相关接口 ====================

// 检查PCL是否可用
bool isPCLAvailable();

#ifdef HAVE_PCL
// PCL网格转换接口（带颜色）
Mesh convertFromPCLMesh(const pcl::PolygonMesh& pclMesh, const cv::Mat& colorImage, const cv::Mat& K);
#endif

} // namespace MeshReconstruction