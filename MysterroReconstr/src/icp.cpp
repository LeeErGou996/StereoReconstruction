// src/icp.cpp
#include "icp.h"
#include <cmath>
#include <cfloat>
#include <iostream>
#include <algorithm>
#include "utils/kdtree.h"
#include <Eigen/Dense>

// 计算点云质心
static Point3D computeCentroid(const std::vector<Point3D>& pts) {
    Point3D c = {0, 0, 0};
    for (const auto& p : pts) {
        c.x += p.x;
        c.y += p.y;
        c.z += p.z;
    }
    float N = static_cast<float>(pts.size());
    c.x /= N; c.y /= N; c.z /= N;
    return c;
}

// 构建协方差矩阵 H = A^T * B
static void computeCovariance(const std::vector<Point3D>& A, const std::vector<Point3D>& B,
                               float H[3][3]) {
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j)
            H[i][j] = 0;

    for (size_t k = 0; k < A.size(); ++k) {
        H[0][0] += A[k].x * B[k].x;
        H[0][1] += A[k].x * B[k].y;
        H[0][2] += A[k].x * B[k].z;
        H[1][0] += A[k].y * B[k].x;
        H[1][1] += A[k].y * B[k].y;
        H[1][2] += A[k].y * B[k].z;
        H[2][0] += A[k].z * B[k].x;
        H[2][1] += A[k].z * B[k].y;
        H[2][2] += A[k].z * B[k].z;
    }
}

static void computeSVDApprox(const float H[3][3], float R_out[3][3]) {
    Eigen::Matrix3f H_eigen;
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j)
            H_eigen(i,j) = H[i][j];

    Eigen::JacobiSVD<Eigen::Matrix3f> svd(H_eigen, Eigen::ComputeFullU | Eigen::ComputeFullV);
    Eigen::Matrix3f R = svd.matrixV() * svd.matrixU().transpose();

    if (R.determinant() < 0) {
        Eigen::Matrix3f V = svd.matrixV();
        V.col(2) *= -1;
        R = V * svd.matrixU().transpose();
    }

    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j)
            R_out[i][j] = R(i,j);
}

static Point3D rotate(const float R[3][3], const Point3D& p) {
    Point3D r;
    r.x = R[0][0]*p.x + R[0][1]*p.y + R[0][2]*p.z;
    r.y = R[1][0]*p.x + R[1][1]*p.y + R[1][2]*p.z;
    r.z = R[2][0]*p.x + R[2][1]*p.y + R[2][2]*p.z;
    return r;
}

// 矩阵乘法: result = A * B
static void matrixMultiply(const float A[3][3], const float B[3][3], float result[3][3]) {
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            result[i][j] = 0;
            for (int k = 0; k < 3; ++k) {
                result[i][j] += A[i][k] * B[k][j];
            }
        }
    }
}

// 计算两点间距离
static float computeDistance(const Point3D& p1, const Point3D& p2) {
    float dx = p1.x - p2.x;
    float dy = p1.y - p2.y;
    float dz = p1.z - p2.z;
    return std::sqrt(dx * dx + dy * dy + dz * dz);
}

// 计算匹配点对的中位数距离，用于更鲁棒的阈值设定
static float computeMedianDistance(const std::vector<Point3D>& src_matched, 
                                   const std::vector<Point3D>& tgt_matched) {
    std::vector<float> distances;
    distances.reserve(src_matched.size());
    
    for (size_t i = 0; i < src_matched.size(); ++i) {
        float dist = computeDistance(src_matched[i], tgt_matched[i]);
        distances.push_back(dist);
    }
    
    if (distances.empty()) return 0.0f;
    
    std::sort(distances.begin(), distances.end());
    size_t mid = distances.size() / 2;
    return distances[mid];
}

// 改进的异常值剔除，使用统计学方法
static std::pair<std::vector<Point3D>, std::vector<Point3D>> 
removeOutliersAdvanced(const std::vector<Point3D>& src_matched, 
                      const std::vector<Point3D>& tgt_matched) {
    
    if (src_matched.size() < 10) {
        return std::make_pair(src_matched, tgt_matched);
    }
    
    std::vector<float> distances;
    distances.reserve(src_matched.size());
    
    for (size_t i = 0; i < src_matched.size(); ++i) {
        float dist = computeDistance(src_matched[i], tgt_matched[i]);
        distances.push_back(dist);
    }
    
    // 计算统计量
    std::vector<float> sorted_distances = distances;
    std::sort(sorted_distances.begin(), sorted_distances.end());
    
    size_t n = sorted_distances.size();
    float q1 = sorted_distances[n/4];
    float q3 = sorted_distances[3*n/4];
    float iqr = q3 - q1;
    float threshold = q3 + 1.5f * iqr; // 标准异常值检测
    
    // 但不要太激进，至少保留70%的点
    size_t min_keep = static_cast<size_t>(n * 0.7f);
    if (min_keep > 0) {
        size_t threshold_idx = std::min(n-1, static_cast<size_t>(n * 0.85f));
        threshold = std::min(threshold, sorted_distances[threshold_idx]);
    }
    
    std::vector<Point3D> filtered_src, filtered_tgt;
    filtered_src.reserve(n);
    filtered_tgt.reserve(n);
    
    for (size_t i = 0; i < src_matched.size(); ++i) {
        if (distances[i] <= threshold) {
            filtered_src.push_back(src_matched[i]);
            filtered_tgt.push_back(tgt_matched[i]);
        }
    }
    
    return std::make_pair(filtered_src, filtered_tgt);
}

// 移除异常值的函数
static std::pair<std::vector<Point3D>, std::vector<Point3D>> 
removeOutliers(const std::vector<Point3D>& src_matched, 
               const std::vector<Point3D>& tgt_matched, 
               float outlier_ratio = 0.8f) {
    
    std::vector<float> distances;
    distances.reserve(src_matched.size());
    
    for (size_t i = 0; i < src_matched.size(); ++i) {
        float dist = computeDistance(src_matched[i], tgt_matched[i]);
        distances.push_back(dist);
    }
    
    // 排序距离
    std::vector<float> sorted_distances = distances;
    std::sort(sorted_distances.begin(), sorted_distances.end());
    
    // 计算阈值（保留最好的outlier_ratio比例的匹配）
    size_t keep_count = static_cast<size_t>(sorted_distances.size() * outlier_ratio);
    if (keep_count == 0) keep_count = 1;
    if (keep_count >= sorted_distances.size()) keep_count = sorted_distances.size() - 1;
    
    float threshold = sorted_distances[keep_count - 1];
    
    std::vector<Point3D> filtered_src, filtered_tgt;
    filtered_src.reserve(keep_count);
    filtered_tgt.reserve(keep_count);
    
    for (size_t i = 0; i < src_matched.size(); ++i) {
        if (distances[i] <= threshold) {
            filtered_src.push_back(src_matched[i]);
            filtered_tgt.push_back(tgt_matched[i]);
        }
    }
    
    return std::make_pair(filtered_src, filtered_tgt);
}

Transform RigidICP::align(const std::vector<PointWithNormal>& sourceInput,
                          const std::vector<PointWithNormal>& target,
                          int maxIterations,
                          float tolerance) {
    
    // 异常情况检查
    if (sourceInput.empty() || target.empty()) {
        std::cerr << "[ICP] Error: Empty point clouds!" << std::endl;
        Transform emptyT;
        for (int i = 0; i < 3; ++i) {
            emptyT.t[i] = 0.0f;
            for (int j = 0; j < 3; ++j)
                emptyT.R[i][j] = (i == j) ? 1.0f : 0.0f;
        }
        return emptyT;
    }

    // 初始化累积变换
    Transform totalT;
    for (int i = 0; i < 3; ++i) {
        totalT.t[i] = 0.0f;
        for (int j = 0; j < 3; ++j)
            totalT.R[i][j] = (i == j) ? 1.0f : 0.0f;
    }

    // 复制源点云
    std::vector<Point3D> src;
    src.reserve(sourceInput.size());
    for (const auto& p : sourceInput) {
        src.push_back(p.point);
    }

    // 构建KD树
    KDTreeWrapper kdtree(target);
    
    float prevRMSE = FLT_MAX;
    int stagnant_iterations = 0;
    const int max_stagnant = 3; // 允许的停滞迭代次数
    
    std::cout << "[ICP] Starting alignment with " << src.size() << " source points and " 
              << target.size() << " target points" << std::endl;

    for (int iter = 0; iter < maxIterations; ++iter) {
        // 预分配匹配点容器
        std::vector<Point3D> src_matched, tgt_matched;
        src_matched.reserve(src.size());
        tgt_matched.reserve(src.size());

        // 改进的多阶段匹配策略
        float base_threshold = 0.02f; // 减小基础阈值
        float max_match_distance;
        
        if (iter < 2) {
            // 早期阶段：使用较大阈值，但不要太大
            max_match_distance = base_threshold * 3.0f;
        } else if (iter < 5) {
            // 中期阶段：逐渐收紧
            max_match_distance = base_threshold * (2.5f - 0.3f * (iter - 2));
        } else {
            // 后期阶段：更严格的阈值
            max_match_distance = std::max(base_threshold * 0.8f, prevRMSE * 1.5f);
        }
        
        // 找到匹配点对
        for (const auto& p : src) {
            int j = kdtree.findClosest(p);
            if (j >= 0) {
                const Point3D& q = target[j].point;
                float dist = computeDistance(p, q);

                if (dist < max_match_distance) {
                    src_matched.push_back(p);
                    tgt_matched.push_back(q);
                }
            }
        }

        // 检查匹配点数量
        if (src_matched.size() < 3) {
            std::cerr << "[ICP] Warning: Too few matched points (" << src_matched.size() 
                      << ") at iteration " << iter << std::endl;
            break;
        }
        
        // 改进的异常值剔除策略
        std::vector<Point3D> filtered_src = src_matched;
        std::vector<Point3D> filtered_tgt = tgt_matched;
        
        if (iter >= 1 && src_matched.size() > 8) {
            if (iter < 3) {
                // 早期：温和的异常值剔除
                auto filtered = removeOutliers(src_matched, tgt_matched, 0.9f);
                filtered_src = filtered.first;
                filtered_tgt = filtered.second;
            } else {
                // 后期：使用统计学方法
                auto filtered = removeOutliersAdvanced(src_matched, tgt_matched);
                filtered_src = filtered.first;
                filtered_tgt = filtered.second;
            }
            
            if (filtered_src.size() < 3) {
                std::cerr << "[ICP] Warning: Too few points after outlier removal" << std::endl;
                filtered_src = src_matched;
                filtered_tgt = tgt_matched;
            }
        }
        
        src_matched = filtered_src;
        tgt_matched = filtered_tgt;

        // 计算质心
        Point3D cs = computeCentroid(src_matched);
        Point3D ct = computeCentroid(tgt_matched);

        // 去质心化
        std::vector<Point3D> src_c(src_matched), tgt_c(tgt_matched);
        for (auto& p : src_c) { 
            p.x -= cs.x; p.y -= cs.y; p.z -= cs.z; 
        }
        for (auto& p : tgt_c) { 
            p.x -= ct.x; p.y -= ct.y; p.z -= ct.z; 
        }

        // 计算协方差矩阵和旋转矩阵
        float H[3][3];
        computeCovariance(src_c, tgt_c, H);

        float R[3][3];
        computeSVDApprox(H, R);

        // 计算平移向量
        Point3D Rc = rotate(R, cs);
        float t[3] = { ct.x - Rc.x, ct.y - Rc.y, ct.z - Rc.z };

        // 应用变换到源点云
        for (auto& p : src) {
            Point3D rp = rotate(R, p);
            p.x = rp.x + t[0];
            p.y = rp.y + t[1];
            p.z = rp.z + t[2];
        }

        // 累积变换 - 正确的矩阵乘法
        float newR[3][3];
        matrixMultiply(R, totalT.R, newR);
        for (int i = 0; i < 3; ++i) {
            for (int j = 0; j < 3; ++j) {
                totalT.R[i][j] = newR[i][j];
            }
        }

        // 累积平移 - 考虑旋转的影响
        Point3D rotated_t = rotate(R, {totalT.t[0], totalT.t[1], totalT.t[2]});
        totalT.t[0] = rotated_t.x + t[0];
        totalT.t[1] = rotated_t.y + t[1];
        totalT.t[2] = rotated_t.z + t[2];

        // 重新计算RMSE（基于变换后的点）
        float rmse = 0.0f;
        int valid_matches = 0;
        
        for (const auto& p : src) {
            int j = kdtree.findClosest(p);
            if (j >= 0) {
                const Point3D& q = target[j].point;
                float dist = computeDistance(p, q);
                if (dist < max_match_distance) {
                    rmse += dist * dist;
                    valid_matches++;
                }
            }
        }
        
        if (valid_matches > 0) {
            rmse = std::sqrt(rmse / valid_matches);
        } else {
            rmse = FLT_MAX;
        }

        std::cout << "[ICP Iteration " << iter << "] Matched: " << src_matched.size() 
                  << "/" << src.size() << ", RMSE: " << rmse 
                  << ", Threshold: " << max_match_distance << std::endl;

        // 增强的收敛判断
        if (rmse < tolerance) {
            std::cout << "[ICP] Converged at iteration " << iter 
                      << " with RMSE: " << rmse << std::endl;
            break;
        }

        // 检查RMSE改善情况和停滞检测
        float improvement = prevRMSE - rmse;
        if (iter > 0) {
            if (improvement < tolerance * 0.01f) {
                stagnant_iterations++;
                std::cout << "[ICP] Small improvement: " << improvement 
                          << ", stagnant count: " << stagnant_iterations << std::endl;
            } else {
                stagnant_iterations = 0; // 重置停滞计数
            }
            
            // 如果停滞太久，尝试扰动
            if (stagnant_iterations >= max_stagnant && iter < maxIterations - 5) {
                std::cout << "[ICP] Applying perturbation to escape local minimum" << std::endl;
                
                // 小幅随机扰动源点云以跳出局部最优
                const float perturbation = rmse * 0.1f;
                for (auto& p : src) {
                    p.x += (static_cast<float>(rand()) / RAND_MAX - 0.5f) * perturbation;
                    p.y += (static_cast<float>(rand()) / RAND_MAX - 0.5f) * perturbation;
                    p.z += (static_cast<float>(rand()) / RAND_MAX - 0.5f) * perturbation;
                }
                stagnant_iterations = 0;
                prevRMSE = FLT_MAX; // 重置以重新评估
                continue;
            }
            
            // 如果持续停滞，提前退出
            if (stagnant_iterations >= max_stagnant * 2) {
                std::cout << "[ICP] Stopping due to persistent stagnation at iteration " 
                          << iter << std::endl;
                break;
            }
        }

        prevRMSE = rmse;
    }

    std::cout << "[ICP] Final alignment completed" << std::endl;
    return totalT;
}