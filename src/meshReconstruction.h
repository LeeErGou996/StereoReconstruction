#ifndef MESH_RECONSTRUCTION_H
#define MESH_RECONSTRUCTION_H

#include <opencv2/opencv.hpp>
#include <vector>
#include <string>

namespace MeshReconstruction {

/**
 * Point cloud data structure
 */
struct Point3D {
    float x, y, z;
    uint8_t r, g, b;
    int img_x, img_y; // 新增：像素坐标
    
    Point3D() : x(0), y(0), z(0), r(0), g(0), b(0), img_x(-1), img_y(-1) {}
    Point3D(float x_, float y_, float z_, uint8_t r_ = 0, uint8_t g_ = 0, uint8_t b_ = 0, int ix = -1, int iy = -1)
        : x(x_), y(y_), z(z_), r(r_), g(g_), b(b_), img_x(ix), img_y(iy) {}
};

/**
 * Triangle structure for mesh
 */
struct Triangle {
    int v1, v2, v3;  // Vertex indices
    Triangle(int a, int b, int c) : v1(a), v2(b), v3(c) {}
};

/**
 * Mesh structure
 */
struct Mesh {
    std::vector<Point3D> vertices;
    std::vector<Triangle> faces;
};

/**
 * Mesh reconstruction parameters
 */
struct ReconstructionParams {
    float depthThreshold = 100.0f;  // Depth threshold
    float voxelSize = 0.005f;           // Voxel size
    bool useColor = true;             // Use color or not
    bool smoothMesh = true;           // Smooth mesh or not
    int decimationTarget = 100000;    // Target triangle count for mesh decimation
    
    // Reconstruction mode: 0: Point cloud only, 1: Triangulated mesh, 2: Poisson surface reconstruction
    int reconstructionMode = 0;
    
    // Triangulation specific parameters
    int triangulationStep = 1;        // Sampling step for triangulation (pixels)
    float maxDepthDifference = 500.0f; // Maximum depth difference for triangle validation (same unit as depth)
    
    // Poisson reconstruction parameters
    float poissonDepth;        // Depth of the octree used for reconstruction
    float poissonSolverDivide; // Depth at which a block Gauss-Seidel solver is used
    float poissonSamplesPerNode; // Minimum number of sample points that fall within an octree node
    float poissonFullDepth;    // Depth at which the mesh is fully reconstructed
    float poissonTrim;         // Trimming parameter for mesh cleaning
    bool poissonUseConfidence; // Use confidence weights in reconstruction
    bool poissonManifold;      // 保持流形
    bool poissonOutputPolygons; // 输出多边形
    
    // Poisson参数补充
    float depthDiffThreshold; // 深度差异阈值系数（如0.1）
    int normalNeighbors;      // 法线估计邻居数（如15）
    
    // Mesh format options
    bool saveAsMesh = true;           // Save as mesh format (with faces) instead of point cloud
    std::string meshFormat = "ply";   // Output format: "ply", "obj", "stl"
};

/**
 * Normal vector structure for Poisson reconstruction
 */
struct Normal3D {
    float nx, ny, nz;
    
    Normal3D() : nx(0), ny(0), nz(0) {}
    Normal3D(float nx_, float ny_, float nz_) : nx(nx_), ny(ny_), nz(nz_) {}
    
    // Normalize the normal vector
    void normalize() {
        float length = sqrt(nx*nx + ny*ny + nz*nz);
        if (length > 1e-6f) {
            nx /= length;
            ny /= length;
            nz /= length;
        }
    }
};

/**
 * Point with normal for Poisson reconstruction
 */
struct PointWithNormal {
    Point3D point;
    Normal3D normal;
    
    PointWithNormal() {}
    PointWithNormal(const Point3D& p, const Normal3D& n) : point(p), normal(n) {}
};

/**
 * Set reconstruction parameters
 */
void setReconstructionParams(const ReconstructionParams& params);

/**
 * Reconstruct and save mesh from depth map and color image
 * @param depthMap Depth map (CV_32F or CV_16U)
 * @param colorImage Color image (optional, for texture)
 * @param outputPath Output file path (.ply, .obj, or .stl)
 * @return Success or not
 */
bool reconstructAndSaveMesh(const cv::Mat& depthMap, 
                           const cv::Mat& colorImage, 
                           const std::string& outputPath);

/**
 * Reconstruct and save mesh from disparity map
 * @param disparityMap Disparity map
 * @param Q Reprojection matrix
 * @param colorImage Color image
 * @param outputPath Output file path
 * @return Success or not
 */
bool reconstructFromDisparity(const cv::Mat& disparityMap,
                             const cv::Mat& Q,
                             const cv::Mat& colorImage,
                             const std::string& outputPath);

/**
 * Generate point cloud from depth map
 */
std::vector<Point3D> generatePointCloud(const cv::Mat& depthMap, 
                                       const cv::Mat& colorImage,
                                       const cv::Mat& K);

/**
 * Generate triangulated mesh using Delaunay triangulation
 * @param depthMap Input depth map
 * @param colorImage Input color image (optional)
 * @param K Camera intrinsic matrix (3x3)
 * @return Triangulated mesh
 */
Mesh generateTriangulatedMesh(const cv::Mat& depthMap, 
                             const cv::Mat& colorImage,
                             const cv::Mat& K);

/**
 * Save point cloud as PLY format
 */
bool savePointCloudPLY(const std::vector<Point3D>& points, 
                      const std::string& filename);

/**
 * Save triangulated mesh to different formats (PLY, OBJ, STL)
 * @param mesh Input mesh with vertices and faces
 * @param filename Output filename
 * @param format Output format ("ply", "obj", "stl")
 * @return true if successful, false otherwise
 */
bool saveMeshFile(const Mesh& mesh, const std::string& filename, const std::string& format = "ply");

std::vector<Point3D> filterPointCloud(const std::vector<Point3D>& input, float maxNeighborDist);

} // namespace MeshReconstruction

#endif // MESH_RECONSTRUCTION_H