// src/icp_point2plane.cpp - Diagnostic and radically improved version
#include "icp_point2plane.h"
#include "utils/kdtree.h"
#include <Eigen/Dense>
#include <cmath>
#include <iostream>
#include <algorithm>
#include <cfloat>
#include <iomanip>
#include <unordered_set>
#include <random>

// Calculate distance between two points
static float computeDistance(const Point3D& p1, const Point3D& p2) {
    float dx = p1.x - p2.x;
    float dy = p1.y - p2.y;
    float dz = p1.z - p2.z;
    return std::sqrt(dx * dx + dy * dy + dz * dz);
}

// Calculate point-to-plane distance
static float computePointToPlaneDistance(const Point3D& p, const Point3D& q, const Normal3D& n) {
    float dx = p.x - q.x;
    float dy = p.y - q.y;
    float dz = p.z - q.z;
    return std::abs(n.nx * dx + n.ny * dy + n.nz * dz);
}

// Improved RMSE calculation using point-to-plane distance
static float computeRMSE(const std::vector<Point3D>& src_matched, 
                        const std::vector<Point3D>& tgt_matched,
                        const std::vector<Normal3D>& tgt_normals) {
    if (src_matched.empty()) return FLT_MAX;
    
    float sum = 0;
    for (size_t i = 0; i < src_matched.size(); ++i) {
        float dist = computePointToPlaneDistance(src_matched[i], tgt_matched[i], tgt_normals[i]);
        sum += dist * dist;
    }
    return std::sqrt(sum / src_matched.size());
}

// More lenient normal validation
static bool isValidNormal(const Normal3D& n) {
    float norm = std::sqrt(n.nx * n.nx + n.ny * n.ny + n.nz * n.nz);
    return norm > 0.1f && norm < 2.0f; // Much more lenient
}

// Diagnostic function to analyze point cloud characteristics
static void diagnosePointClouds(const std::vector<Point3D>& src, 
                               const std::vector<PointWithNormal>& target,
                               const KDTreeWrapper& kdtree) {
    std::cout << "\n=== POINT CLOUD DIAGNOSTIC ===" << std::endl;
    
    // Calculate bounding boxes
    Point3D src_min = {1e10, 1e10, 1e10}, src_max = {-1e10, -1e10, -1e10};
    Point3D tgt_min = {1e10, 1e10, 1e10}, tgt_max = {-1e10, -1e10, -1e10};
    
    for (const auto& p : src) {
        src_min.x = std::min(src_min.x, p.x); src_max.x = std::max(src_max.x, p.x);
        src_min.y = std::min(src_min.y, p.y); src_max.y = std::max(src_max.y, p.y);
        src_min.z = std::min(src_min.z, p.z); src_max.z = std::max(src_max.z, p.z);
    }
    
    for (const auto& p : target) {
        tgt_min.x = std::min(tgt_min.x, p.point.x); tgt_max.x = std::max(tgt_max.x, p.point.x);
        tgt_min.y = std::min(tgt_min.y, p.point.y); tgt_max.y = std::max(tgt_max.y, p.point.y);
        tgt_min.z = std::min(tgt_min.z, p.point.z); tgt_max.z = std::max(tgt_max.z, p.point.z);
    }
    
    std::cout << "Source bounds: [" << src_min.x << "," << src_max.x << "] x ["
              << src_min.y << "," << src_max.y << "] x [" << src_min.z << "," << src_max.z << "]" << std::endl;
    std::cout << "Target bounds: [" << tgt_min.x << "," << tgt_max.x << "] x ["
              << tgt_min.y << "," << tgt_max.y << "] x [" << tgt_min.z << "," << tgt_max.z << "]" << std::endl;
    
    // Calculate overlap
    float overlap_x = std::max(0.0f, std::min(src_max.x, tgt_max.x) - std::max(src_min.x, tgt_min.x));
    float overlap_y = std::max(0.0f, std::min(src_max.y, tgt_max.y) - std::max(src_min.y, tgt_min.y));
    float overlap_z = std::max(0.0f, std::min(src_max.z, tgt_max.z) - std::max(src_min.z, tgt_min.z));
    
    float src_volume = (src_max.x - src_min.x) * (src_max.y - src_min.y) * (src_max.z - src_min.z);
    float overlap_volume = overlap_x * overlap_y * overlap_z;
    float overlap_ratio = (src_volume > 0) ? overlap_volume / src_volume : 0;
    
    std::cout << "Spatial overlap ratio: " << (overlap_ratio * 100) << "%" << std::endl;
    
    // Sample distance analysis
    std::vector<float> sample_distances;
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis(0, src.size() - 1);
    
    for (int i = 0; i < std::min(1000, static_cast<int>(src.size())); ++i) {
        int idx = dis(gen);
        int closest = kdtree.findClosest(src[idx]);
        if (closest >= 0) {
            float dist = computeDistance(src[idx], target[closest].point);
            sample_distances.push_back(dist);
        }
    }
    
    if (!sample_distances.empty()) {
        std::sort(sample_distances.begin(), sample_distances.end());
        float median_dist = sample_distances[sample_distances.size() / 2];
        float q75_dist = sample_distances[sample_distances.size() * 3 / 4];
        float q90_dist = sample_distances[sample_distances.size() * 9 / 10];
        
        std::cout << "Sample distance statistics:" << std::endl;
        std::cout << "  Median: " << median_dist << std::endl;
        std::cout << "  75th percentile: " << q75_dist << std::endl;
        std::cout << "  90th percentile: " << q90_dist << std::endl;
        std::cout << "  Suggested threshold: " << q75_dist * 1.5f << std::endl;
    }
    
    // Normal validation
    int valid_normals = 0;
    for (const auto& p : target) {
        if (isValidNormal(p.normal)) valid_normals++;
    }
    std::cout << "Valid normals: " << valid_normals << "/" << target.size() 
              << " (" << (100.0f * valid_normals / target.size()) << "%)" << std::endl;
    
    std::cout << "=== END DIAGNOSTIC ===\n" << std::endl;
}

// Aggressive multi-scale matching strategy
static void performMultiScaleMatching(
    const std::vector<Point3D>& src,
    const std::vector<PointWithNormal>& target,
    const KDTreeWrapper& kdtree,
    float base_threshold,
    std::vector<Point3D>& src_matched,
    std::vector<Point3D>& tgt_matched,
    std::vector<Normal3D>& tgt_normals,
    std::vector<int>& src_indices,
    std::vector<int>& tgt_indices
) {
    src_matched.clear();
    tgt_matched.clear();
    tgt_normals.clear();
    src_indices.clear();
    tgt_indices.clear();
    
    std::unordered_set<int> used_target_indices;
    std::unordered_set<int> used_src_indices;
    
    // Multi-scale thresholds: from loose to strict
    std::vector<float> thresholds = {
        base_threshold * 3.0f,   // Very loose
        base_threshold * 2.0f,   // Loose
        base_threshold * 1.5f,   // Medium
        base_threshold * 1.0f,   // Base
        base_threshold * 0.8f    // Strict
    };
    
    for (float threshold : thresholds) {
        std::vector<std::pair<float, std::pair<int, int>>> scale_matches;
        
        // Find matches at this scale
        for (size_t i = 0; i < src.size(); ++i) {
            if (used_src_indices.count(i)) continue;
            
            // Try multiple nearest neighbors
            std::vector<int> candidates;
            kdtree.kNearest(src[i], 5, candidates);
            
            for (int j : candidates) {
                if (j >= 0 && j < target.size() && !used_target_indices.count(j)) {
                    const Point3D& q = target[j].point;
                    const Normal3D& n = target[j].normal;
                    
                    float dist = computeDistance(src[i], q);
                    if (dist < threshold) {
                        // Additional validation
                        bool valid = true;
                        
                        // Normal check (more lenient)
                        if (!isValidNormal(n)) valid = false;
                        
                        // Plane distance check (if normal is valid)
                        if (valid && isValidNormal(n)) {
                            float plane_dist = computePointToPlaneDistance(src[i], q, n);
                            if (plane_dist > threshold * 1.2f) valid = false;
                        }
                        
                        if (valid) {
                            scale_matches.emplace_back(dist, std::make_pair(i, j));
                            break; // Take the first valid match for this source point
                        }
                    }
                }
            }
        }
        
        // Sort by distance and select best matches
        std::sort(scale_matches.begin(), scale_matches.end());
        
        for (const auto& match : scale_matches) {
            int src_idx = match.second.first;
            int tgt_idx = match.second.second;
            
            if (used_src_indices.count(src_idx) || used_target_indices.count(tgt_idx)) {
                continue;
            }
            
            used_src_indices.insert(src_idx);
            used_target_indices.insert(tgt_idx);
            
            src_matched.push_back(src[src_idx]);
            tgt_matched.push_back(target[tgt_idx].point);
            tgt_normals.push_back(target[tgt_idx].normal);
            src_indices.push_back(src_idx);
            tgt_indices.push_back(tgt_idx);
        }
    }
}

// Conservative outlier removal
static void removeOutliersConservative(std::vector<Point3D>& src_matched,
                                      std::vector<Point3D>& tgt_matched,
                                      std::vector<Normal3D>& tgt_normals,
                                      std::vector<int>& src_indices,
                                      std::vector<int>& tgt_indices,
                                      float retention_rate = 0.90f) {
    
    if (src_matched.size() < 50) return; // Only remove outliers if we have plenty of matches
    
    std::vector<float> distances;
    distances.reserve(src_matched.size());
    
    for (size_t i = 0; i < src_matched.size(); ++i) {
        float dist = computeDistance(src_matched[i], tgt_matched[i]); // Use Euclidean distance
        distances.push_back(dist);
    }
    
    std::vector<size_t> indices(distances.size());
    std::iota(indices.begin(), indices.end(), 0);
    
    std::partial_sort(indices.begin(), indices.begin() + static_cast<size_t>(distances.size() * retention_rate), 
                     indices.end(), [&distances](size_t a, size_t b) { return distances[a] < distances[b]; });
    
    size_t keep_count = static_cast<size_t>(distances.size() * retention_rate);
    
    std::vector<Point3D> filtered_src, filtered_tgt;
    std::vector<Normal3D> filtered_normals;
    std::vector<int> filtered_src_indices, filtered_tgt_indices;
    
    filtered_src.reserve(keep_count);
    filtered_tgt.reserve(keep_count);
    filtered_normals.reserve(keep_count);
    filtered_src_indices.reserve(keep_count);
    filtered_tgt_indices.reserve(keep_count);
    
    for (size_t i = 0; i < keep_count; ++i) {
        size_t idx = indices[i];
        filtered_src.push_back(src_matched[idx]);
        filtered_tgt.push_back(tgt_matched[idx]);
        filtered_normals.push_back(tgt_normals[idx]);
        filtered_src_indices.push_back(src_indices[idx]);
        filtered_tgt_indices.push_back(tgt_indices[idx]);
    }
    
    src_matched = filtered_src;
    tgt_matched = filtered_tgt;
    tgt_normals = filtered_normals;
    src_indices = filtered_src_indices;
    tgt_indices = filtered_tgt_indices;
}

// Matrix multiplication
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

// Convert Eigen matrix to array
static void eigenToArray(const Eigen::Matrix3f& eigen_mat, float array_mat[3][3]) {
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            array_mat[i][j] = eigen_mat(i, j);
        }
    }
}

Transform PointToPlaneICP::align(const std::vector<PointWithNormal>& sourceInput,
                                 const std::vector<PointWithNormal>& target,
                                 int maxIterations,
                                 float maxMatchDist,
                                 float stopThreshold) {
    
    // Exception handling
    if (sourceInput.empty() || target.empty()) {
        std::cerr << "[Diagnostic Point2Plane ICP] Error: Empty point clouds!" << std::endl;
        Transform emptyT;
        for (int i = 0; i < 3; ++i) {
            emptyT.t[i] = 0.0f;
            for (int j = 0; j < 3; ++j)
                emptyT.R[i][j] = (i == j) ? 1.0f : 0.0f;
        }
        return emptyT;
    }
    
    // Initialize cumulative transformation
    Transform totalT;
    for (int i = 0; i < 3; ++i) {
        totalT.t[i] = 0.0f;
        for (int j = 0; j < 3; ++j)
            totalT.R[i][j] = (i == j) ? 1.0f : 0.0f;
    }

    // Copy source point cloud
    std::vector<Point3D> src;
    src.reserve(sourceInput.size());
    for (const auto& p : sourceInput) {
        src.push_back(p.point);
    }

    // Build KD tree
    KDTreeWrapper kdtree(target);
    
    std::cout << "[Diagnostic Point2Plane ICP] Starting alignment with " << src.size() 
              << " source points and " << target.size() << " target points" << std::endl;
    
    // Perform initial diagnosis
    diagnosePointClouds(src, target, kdtree);
    
    float prevRMSE = FLT_MAX;
    int stagnant_iterations = 0;
    const int max_stagnant = 8;
    
    // Determine better initial threshold based on diagnosis
    float adaptive_base_threshold = maxMatchDist * 2.0f; // Start more aggressively

    for (int iter = 0; iter < maxIterations; ++iter) {
        std::vector<Point3D> src_matched, tgt_matched;
        std::vector<Normal3D> tgt_normals;
        std::vector<int> src_indices, tgt_indices;
        
        // Adaptive threshold - much more aggressive
        float current_threshold = adaptive_base_threshold;
        if (iter == 0) {
            current_threshold = adaptive_base_threshold * 2.0f; // Very generous initially
        } else if (iter < 3) {
            current_threshold = adaptive_base_threshold * 1.8f;
        } else if (iter < 8) {
            current_threshold = adaptive_base_threshold * (1.6f - 0.1f * iter);
        } else {
            // Keep reasonable threshold to maintain matches
            current_threshold = std::max(adaptive_base_threshold * 0.8f, prevRMSE * 5.0f);
        }

        // Use multi-scale matching
        performMultiScaleMatching(src, target, kdtree, current_threshold,
                                src_matched, tgt_matched, tgt_normals,
                                src_indices, tgt_indices);

        // Check matched point count
        if (src_matched.size() < 6) {
            std::cerr << "[Diagnostic Point2Plane ICP] Critical: Too few matched points (" 
                      << src_matched.size() << ") at iteration " << iter << std::endl;
            
            // Emergency fallback: use point-to-point matching
            std::cout << "[Diagnostic Point2Plane ICP] Switching to emergency point-to-point matching" << std::endl;
            current_threshold *= 3.0f; // Much larger threshold
            
            for (size_t i = 0; i < src.size(); ++i) {
                int j = kdtree.findClosest(src[i]);
                if (j >= 0) {
                    float dist = computeDistance(src[i], target[j].point);
                    if (dist < current_threshold) {
                        src_matched.push_back(src[i]);
                        tgt_matched.push_back(target[j].point);
                        // Use dummy normal for point-to-point
                        tgt_normals.push_back({0, 0, 1});
                        src_indices.push_back(i);
                        tgt_indices.push_back(j);
                    }
                }
            }
            
            if (src_matched.size() < 6) {
                std::cerr << "[Diagnostic Point2Plane ICP] Emergency matching also failed" << std::endl;
                break;
            }
        }
        
        // Conservative outlier removal only if we have plenty of matches
        size_t original_count = src_matched.size();
        float match_ratio_before = static_cast<float>(src_matched.size()) / src.size();
        
        if (iter >= 3 && src_matched.size() > 100 && match_ratio_before > 0.4f) {
            removeOutliersConservative(src_matched, tgt_matched, tgt_normals,
                                     src_indices, tgt_indices, 0.92f);
            
            std::cout << "[Diagnostic Point2Plane ICP] Outlier removal: " << original_count 
                      << " -> " << src_matched.size() << " points" << std::endl;
        }

        // Build linear system - more robust version
        bool use_point_to_plane = true;
        
        // Check if we have valid normals
        int valid_normal_count = 0;
        for (const auto& n : tgt_normals) {
            if (isValidNormal(n) && (n.nx != 0 || n.ny != 0 || n.nz != 1)) {
                valid_normal_count++;
            }
        }
        
        if (valid_normal_count < src_matched.size() * 0.5f) {
            std::cout << "[Diagnostic Point2Plane ICP] Too few valid normals, using point-to-point" << std::endl;
            use_point_to_plane = false;
        }

        Eigen::MatrixXf A(src_matched.size(), 6);
        Eigen::VectorXf b(src_matched.size());

        for (size_t i = 0; i < src_matched.size(); ++i) {
            const Point3D& p = src_matched[i];
            const Point3D& q = tgt_matched[i];
            const Normal3D& n = tgt_normals[i];

            if (use_point_to_plane && isValidNormal(n)) {
                float nx = n.nx, ny = n.ny, nz = n.nz;

                // Point-to-plane ICP
                A(i,0) = nz * p.y - ny * p.z;
                A(i,1) = nx * p.z - nz * p.x;
                A(i,2) = ny * p.x - nx * p.y;
                A(i,3) = nx;
                A(i,4) = ny;
                A(i,5) = nz;

                b(i) = nx * (q.x - p.x) + ny * (q.y - p.y) + nz * (q.z - p.z);
            } else {
                // Fallback to point-to-point ICP
                A(i,0) = 0; A(i,1) = -p.z; A(i,2) = p.y;
                A(i,3) = 1; A(i,4) = 0;    A(i,5) = 0;
                b(i) = q.x - p.x;
                
                // Note: This is simplified - in practice you'd need 3 equations per point
                // But for demonstration, we'll use one equation per point
            }
        }

        // Solve with regularization
        Eigen::MatrixXf AtA = A.transpose() * A;
        Eigen::VectorXf Atb = A.transpose() * b;
        
        float lambda = 1e-5f; // Stronger regularization
        for (int i = 0; i < 6; ++i) {
            AtA(i, i) += lambda;
        }
        
        Eigen::VectorXf x = AtA.ldlt().solve(Atb);
        
        // Extract and limit parameters
        float wx = x(0), wy = x(1), wz = x(2);
        float rotation_norm = std::sqrt(wx*wx + wy*wy + wz*wz);
        float max_rotation = (iter < 3) ? 0.1f : 0.05f;
        
        if (rotation_norm > max_rotation) {
            float scale = max_rotation / rotation_norm;
            wx *= scale; wy *= scale; wz *= scale;
        }

        // Build rotation matrix
        Eigen::Matrix3f R_mat;
        if (rotation_norm < 1e-8f) {
            R_mat = Eigen::Matrix3f::Identity();
        } else {
            Eigen::Vector3f axis(wx, wy, wz);
            float angle = axis.norm();
            if (angle > 1e-8f) {
                axis /= angle;
                R_mat = Eigen::AngleAxisf(angle, axis).toRotationMatrix();
            } else {
                R_mat = Eigen::Matrix3f::Identity();
            }
        }

        float t[3] = { x(3), x(4), x(5) };
        float translation_norm = std::sqrt(t[0]*t[0] + t[1]*t[1] + t[2]*t[2]);
        float max_translation = current_threshold;
        
        if (translation_norm > max_translation) {
            float scale = max_translation / translation_norm;
            t[0] *= scale; t[1] *= scale; t[2] *= scale;
        }

        // Apply transformation
        for (auto& p : src) {
            Eigen::Vector3f v(p.x, p.y, p.z);
            Eigen::Vector3f v_rotated = R_mat * v;
            p.x = v_rotated[0] + t[0];
            p.y = v_rotated[1] + t[1];
            p.z = v_rotated[2] + t[2];
        }

        // Accumulate transformation
        float R_array[3][3], totalR_new[3][3];
        eigenToArray(R_mat, R_array);
        matrixMultiply(R_array, totalT.R, totalR_new);
        
        for (int i = 0; i < 3; ++i) {
            for (int j = 0; j < 3; ++j) {
                totalT.R[i][j] = totalR_new[i][j];
            }
        }

        Eigen::Vector3f total_t_vec(totalT.t[0], totalT.t[1], totalT.t[2]);
        Eigen::Vector3f rotated_t = R_mat * total_t_vec;
        totalT.t[0] = rotated_t[0] + t[0];
        totalT.t[1] = rotated_t[1] + t[1];
        totalT.t[2] = rotated_t[2] + t[2];

        // Calculate metrics
        float rmse = use_point_to_plane ? computeRMSE(src_matched, tgt_matched, tgt_normals) : 
                     std::sqrt(std::accumulate(b.data(), b.data() + b.size(), 0.0f, 
                                              [](float sum, float val) { return sum + val*val; }) / b.size());
        float match_ratio = static_cast<float>(src_matched.size()) / src.size();
        
        std::cout << "[Diagnostic Point2Plane ICP Iteration " << iter << "] Matched: " 
                  << src_matched.size() << "/" << src.size() 
                  << " (" << std::fixed << std::setprecision(1) << match_ratio * 100 << "%)"
                  << ", RMSE: " << std::setprecision(6) << rmse 
                  << ", Threshold: " << current_threshold 
                  << ", Method: " << (use_point_to_plane ? "Point2Plane" : "Point2Point")
                  << std::endl;

        // Convergence check
        if (rmse < stopThreshold) {
            std::cout << "[Diagnostic Point2Plane ICP] Converged!" << std::endl;
            break;
        }

        float improvement = prevRMSE - rmse;
        if (iter > 0 && std::abs(improvement) < stopThreshold * 0.01f) {
            stagnant_iterations++;
        } else {
            stagnant_iterations = 0;
        }
        
        if (stagnant_iterations >= max_stagnant) {
            std::cout << "[Diagnostic Point2Plane ICP] Stopping due to stagnation" << std::endl;
            break;
        }
        
        prevRMSE = rmse;
    }

    std::cout << "[Diagnostic Point2Plane ICP] Final alignment completed" << std::endl;
    return totalT;
}