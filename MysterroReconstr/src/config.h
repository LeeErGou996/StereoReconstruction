#pragma once

#include <memory>
#include <chrono>

// ==================== 配置类定义 ====================

class Config {
public:
    struct MeshParams {
        float voxelSize = 0.005f;                    // 体素大小
        int triangulationStep = 2;                   // 三角化步长
        float depthThreshold = 5.0f;                 // 深度阈值
        float depthDiffThreshold = 0.1f;             // 深度差异阈值
        int normalNeighbors = 10;                    // 法向量计算邻居数
        int sdfNeighbors = 8;                        // SDF计算邻居数
        
        // PCL Poisson参数
        int poissonDepth = 8;                        // Poisson深度
        int poissonSolverDivide = 8;                 // 求解器分割
        float poissonSamplesPerNode = 1.5f;          // 每节点样本数
        bool poissonUseConfidence = false;           // 是否使用置信度
        bool poissonManifold = false;                // 是否强制流形
        bool poissonOutputPolygons = false;          // 是否输出多边形
    };
    
    MeshParams meshParams;
    
    static Config& instance() {
        static Config config;
        return config;
    }
    
private:
    Config() = default;
};

// ==================== 数据结构定义 ====================

// 3D点结构
struct Point3D {
    float x, y, z;
    uint8_t r, g, b;  // 颜色
    int u, v;         // 像素坐标
    
    Point3D() : x(0), y(0), z(0), r(128), g(128), b(128), u(0), v(0) {}
    Point3D(float x_, float y_, float z_, uint8_t r_ = 128, uint8_t g_ = 128, uint8_t b_ = 128, int u_ = 0, int v_ = 0)
        : x(x_), y(y_), z(z_), r(r_), g(g_), b(b_), u(u_), v(v_) {}
};

// 3D法向量结构
struct Normal3D {
    float nx, ny, nz;
    
    Normal3D() : nx(0), ny(0), nz(1) {}
    Normal3D(float nx_, float ny_, float nz_) : nx(nx_), ny(ny_), nz(nz_) {}
    
    void normalize() {
        float length = sqrt(nx * nx + ny * ny + nz * nz);
        if (length > 1e-6f) {
            nx /= length;
            ny /= length;
            nz /= length;
        }
    }
};

// 带法向量的点
struct PointWithNormal {
    Point3D point;
    Normal3D normal;
    
    PointWithNormal() {}
    PointWithNormal(const Point3D& p, const Normal3D& n) : point(p), normal(n) {}
};

// 三角形面
struct Triangle {
    int v1, v2, v3;  // 顶点索引
    
    Triangle() : v1(0), v2(0), v3(0) {}
    Triangle(int a, int b, int c) : v1(a), v2(b), v3(c) {}
};

// 网格结构
struct Mesh {
    std::vector<Point3D> vertices;
    std::vector<Triangle> faces;
    
    bool empty() const {
        return vertices.empty() && faces.empty();
    }
    
    void clear() {
        vertices.clear();
        faces.clear();
    }
};

// ==================== 性能监测工具 ====================

// 性能计时器
struct PerformanceTimer {
    std::chrono::high_resolution_clock::time_point start_time;
    std::string name;
    
    PerformanceTimer(const std::string& timer_name) : name(timer_name) {
        start_time = std::chrono::high_resolution_clock::now();
        std::cout << "[TIMER] " << name << " started..." << std::endl;
    }
    
    ~PerformanceTimer() {
        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
        std::cout << "[TIMER] " << name << " completed in " << duration.count() << "ms" << std::endl;
    }
    
    void reset() {
        start_time = std::chrono::high_resolution_clock::now();
    }
    
    double getElapsedMs() const {
        auto current_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(current_time - start_time);
        return static_cast<double>(duration.count());
    }
};

// 内存使用统计
struct MemoryUsage {
    size_t points_memory;
    size_t normals_memory;
    size_t mesh_memory;
    size_t total_memory;
    
    MemoryUsage() : points_memory(0), normals_memory(0), mesh_memory(0), total_memory(0) {}
    
    static MemoryUsage calculate(const std::vector<PointWithNormal>& points, const Mesh& mesh) {
        MemoryUsage usage;
        usage.points_memory = points.size() * sizeof(PointWithNormal);
        usage.mesh_memory = mesh.vertices.size() * sizeof(Point3D) + mesh.faces.size() * sizeof(Triangle);
        usage.total_memory = usage.points_memory + usage.normals_memory + usage.mesh_memory;
        return usage;
    }
    
    void print() const {
        std::cout << "\n=== 内存使用统计 ===" << std::endl;
        std::cout << "  点云内存: " << (points_memory / 1024.0 / 1024.0) << " MB" << std::endl;
        std::cout << "  法向量内存: " << (normals_memory / 1024.0 / 1024.0) << " MB" << std::endl;
        std::cout << "  网格内存: " << (mesh_memory / 1024.0 / 1024.0) << " MB" << std::endl;
        std::cout << "  总内存: " << (total_memory / 1024.0 / 1024.0) << " MB" << std::endl;
    }
};

// ==================== 工具函数 ====================

// 验证深度图
void validateDepthMap(const class ImageData& depthMap, const std::string& context = "");

// 验证法向量
void validateNormals(const std::vector<PointWithNormal>& normals, const std::string& context = "");

// 打印图像数据信息
void printImageDataInfo(const class ImageData& img, const std::string& name = "ImageData");

// ==================== 配置函数声明 ====================

// 重置为默认参数
void resetPoissonReconstructionParams();

// 调试模式 - 使用宽松参数便于诊断问题
void setDebugPoissonParams();

// 快速重建模式 - 使用较低质量但速度更快的参数
void setFastPoissonParams();

// 高质量重建模式 - 使用高质量但速度较慢的参数
void setHighQualityPoissonParams();

// 设置基本Poisson重建参数
void setPoissonReconstructionParams(float voxelSize, int triangulationStep, float depthThreshold);

// 设置高级Poisson参数
void setAdvancedPoissonParams(int maxGridSize, int normalNeighbors, int sdfNeighbors);

#ifdef HAVE_PCL
// 设置PCL Poisson参数
void setPCLPoissonParams(int depth, int solverDivide, float samplesPerNode,
                        bool confidence, bool manifold, bool outputPolygons);
#endif