#pragma once
#include <vector>
#include <cstdint>
#include <cmath>
#include <iostream>

// 简单3D点结构
struct Point3D {
    float x, y, z;
    Point3D(float x_=0, float y_=0, float z_=0) : x(x_), y(y_), z(z_) {}
};

// 法向量结构
struct Normal3D {
    float nx, ny, nz;
    Normal3D(float nx_=0, float ny_=0, float nz_=1) : nx(nx_), ny(ny_), nz(nz_) {}
    void normalize() {
        float n = std::sqrt(nx*nx + ny*ny + nz*nz);
        if (n > 1e-6f) { nx/=n; ny/=n; nz/=n; }
    }
};

// 点+法线
struct PointWithNormal {
    Point3D point;
    Normal3D normal;
    PointWithNormal(const Point3D& p, const Normal3D& n) : point(p), normal(n) {}
};

// Mesh顶点
struct Vertex {
    float x, y, z;
    uint8_t r, g, b;
    Vertex(float x_=0, float y_=0, float z_=0, uint8_t r_=128, uint8_t g_=128, uint8_t b_=128)
        : x(x_), y(y_), z(z_), r(r_), g(g_), b(b_) {}
};

// Mesh三角面
struct Face {
    int v1, v2, v3;
    Face(int a, int b, int c) : v1(a), v2(b), v3(c) {}
};

// Mesh结构
struct Mesh {
    std::vector<Vertex> vertices;
    std::vector<Face> faces;
};

// 灰度深度图（float）
struct DepthImage {
    int rows, cols;
    std::vector<float> data;
    DepthImage(int r=0, int c=0) : rows(r), cols(c), data(r*c, 0.0f) {}
    float& at(int y, int x) { return data[y*cols + x]; }
    const float& at(int y, int x) const { return data[y*cols + x]; }
};

// 彩色图像（uint8 3通道）
struct ColorImage {
    int rows, cols;
    std::vector<uint8_t> data; // RGBRGB...
    ColorImage(int r=0, int c=0) : rows(r), cols(c), data(r*c*3, 128) {}
    uint8_t* pixel(int y, int x) { return &data[(y*cols + x)*3]; }
    const uint8_t* pixel(int y, int x) const { return &data[(y*cols + x)*3]; }
};

// 相机内参
struct CameraIntrinsics {
    float fx, fy, cx, cy;
    CameraIntrinsics(float fx_=1, float fy_=1, float cx_=0, float cy_=0)
        : fx(fx_), fy(fy_), cx(cx_), cy(cy_) {}
};

// 主接口：Poisson重建
Mesh poissonReconstruction(const DepthImage& depth, const ColorImage& color, const CameraIntrinsics& K, float depthThreshold=5000.0f, int step=2, int normalNeighbors=10); 

Mesh triangulateFromDepth(const DepthImage& depth,
                          const ColorImage& color,
                          const CameraIntrinsics& K,
                          float depthThreshold,
                          int stepSize);
