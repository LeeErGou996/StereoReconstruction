// test/testicp.cpp - Modified version with selective point saving
#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <sstream>
#include <cstdlib>
#include <cmath>
#include <set>
#include <filesystem>

#include <Eigen/Dense>
#include "../src/utils/kdtree.h"

#include "../src/icp.h"
#include "../src/pointcloud.h"
#include "../src/icp_point2plane.h"

Point3D rotate(const float R[3][3], const Point3D& p) {
    Point3D r;
    r.x = R[0][0]*p.x + R[0][1]*p.y + R[0][2]*p.z;
    r.y = R[1][0]*p.x + R[1][1]*p.y + R[1][2]*p.z;
    r.z = R[2][0]*p.x + R[2][1]*p.y + R[2][2]*p.z;
    return r;
}

// Load PointWithNormal point cloud from PLY file (assuming format: x y z r g b)
bool loadPLY(const std::string& filename, std::vector<PointWithNormal>& points) {
    std::ifstream fin(filename);
    if (!fin.is_open()) {
        std::cerr << "Unable to open PLY file: " << filename << std::endl;
        return false;
    }

    std::string line;
    bool inHeader = true;
    while (std::getline(fin, line)) {
        if (inHeader) {
            if (line == "end_header")
                inHeader = false;
            continue;
        }

        std::istringstream iss(line);
        float x, y, z;
        int r, g, b;
        if (!(iss >> x >> y >> z >> r >> g >> b)) continue;

        Point3D pt(x, y, z);
        Normal3D n(0, 0, 0);  // Default normal is 0 (not used in ICP but maintains structure consistency)
        PointWithNormal p(pt, n);
        points.push_back(p);
    }

    return true;
}

// Save aligned PLY file
bool savePLY(const std::string& filename, const std::vector<PointWithNormal>& points) {
    std::ofstream fout(filename);
    if (!fout.is_open()) return false;

    fout << "ply\nformat ascii 1.0\n";
    fout << "element vertex " << points.size() << "\n";
    fout << "property float x\nproperty float y\nproperty float z\n";
    fout << "property uchar red\nproperty uchar green\nproperty uchar blue\n";
    fout << "end_header\n";
    for (const auto& p : points) {
        fout << p.point.x << " " << p.point.y << " " << p.point.z
             << " 200 200 200\n";
    }
    return true;
}

// Save aligned colored PLY, A in green (left), B in red (right)
bool saveColoredPLY(const std::string& filename,
                    const std::vector<PointWithNormal>& A,
                    const std::vector<PointWithNormal>& B) {
    std::ofstream fout(filename);
    if (!fout.is_open()) return false;

    fout << "ply\nformat ascii 1.0\n";
    fout << "element vertex " << (A.size() + B.size()) << "\n";
    fout << "property float x\nproperty float y\nproperty float z\n";
    fout << "property uchar red\nproperty uchar green\nproperty uchar blue\n";
    fout << "end_header\n";

    for (const auto& p : A) {
        fout << p.point.x << " " << p.point.y << " " << p.point.z << " 0 255 0\n";  // Green
    }
    for (const auto& p : B) {
        fout << p.point.x << " " << p.point.y << " " << p.point.z << " 255 0 0\n";  // Red
    }

    return true;
}

// New function: Save selective point cloud (A complete + B matched only)
bool saveSelectivePLY(const std::string& filename,
                      const std::vector<PointWithNormal>& A_complete,
                      const std::vector<PointWithNormal>& B_matched_only) {
    std::ofstream fout(filename);
    if (!fout.is_open()) return false;

    fout << "ply\nformat ascii 1.0\n";
    fout << "element vertex " << (A_complete.size() + B_matched_only.size()) << "\n";
    fout << "property float x\nproperty float y\nproperty float z\n";
    fout << "property uchar red\nproperty uchar green\nproperty uchar blue\n";
    fout << "end_header\n";

    // Save all points from A (green - reference)
    for (const auto& p : A_complete) {
        fout << p.point.x << " " << p.point.y << " " << p.point.z << " 0 255 0\n";  // Green
    }
    
    // Save only matched points from B (blue - matched)
    for (const auto& p : B_matched_only) {
        fout << p.point.x << " " << p.point.y << " " << p.point.z << " 0 0 255\n";  // Blue
    }

    return true;
}

// Modified ICP class to extract matched point indices
class SelectivePointToPlaneICP {
public:
    static Transform alignWithMatchedPoints(
        const std::vector<PointWithNormal>& sourceInput,
        const std::vector<PointWithNormal>& target,
        int maxIterations,
        float maxMatchDist,
        float stopThreshold,
        std::vector<int>& matched_target_indices  // Output: indices of matched target points
    ) {
        // Clear the output vector
        matched_target_indices.clear();
        
        // Use the standard Point-to-Plane ICP
        Transform result = PointToPlaneICP::align(sourceInput, target, maxIterations, maxMatchDist, stopThreshold);
        
        // After ICP, find the final matched points
        // Apply the transformation to source points
        std::vector<PointWithNormal> transformed_source;
        for (const auto& p : sourceInput) {
            Point3D q = rotate(result.R, p.point);
            q.x += result.t[0];
            q.y += result.t[1];
            q.z += result.t[2];
            transformed_source.emplace_back(q, p.normal);
        }
        
        // Build KD-tree for target
        KDTreeWrapper kdtree(target);
        
        // Find matches using the same criteria as the final ICP iteration
        float current_threshold = maxMatchDist;
        if (maxIterations > 10) {
            current_threshold = std::max(maxMatchDist * 0.8f, stopThreshold * 2.0f);
        }
        
        std::set<int> matched_indices_set;
        
        for (const auto& p : transformed_source) {
            int j = kdtree.findClosest(p.point);
            if (j >= 0) {
                const Point3D& q = target[j].point;
                float dx = p.point.x - q.x;
                float dy = p.point.y - q.y;
                float dz = p.point.z - q.z;
                float dist = std::sqrt(dx * dx + dy * dy + dz * dz);
                
                if (dist < current_threshold) {
                    matched_indices_set.insert(j);
                }
            }
        }
        
        // Convert set to vector
        matched_target_indices.assign(matched_indices_set.begin(), matched_indices_set.end());
        
        std::cout << "Final matching: " << matched_target_indices.size() << "/" << target.size() 
                  << " target points matched (" << (100.0 * matched_target_indices.size() / target.size()) << "%)" << std::endl;
        
        return result;
    }
};

// Use KD-Tree to find neighborhood and compute covariance matrix principal direction as normal
static Normal3D computeNormal(const std::vector<PointWithNormal>& pts,
                              const KDTreeWrapper& kdtree,
                              const Point3D& query,
                              int k = 20) {
    std::vector<int> indices;
    kdtree.kNearest(query, k, indices);

    float cx = 0, cy = 0, cz = 0;
    for (int i : indices) {
        cx += pts[i].point.x;
        cy += pts[i].point.y;
        cz += pts[i].point.z;
    }
    cx /= indices.size();
    cy /= indices.size();
    cz /= indices.size();

    float cov[3][3] = {};
    for (int i : indices) {
        float dx = pts[i].point.x - cx;
        float dy = pts[i].point.y - cy;
        float dz = pts[i].point.z - cz;
        cov[0][0] += dx * dx; cov[0][1] += dx * dy; cov[0][2] += dx * dz;
        cov[1][0] += dy * dx; cov[1][1] += dy * dy; cov[1][2] += dy * dz;
        cov[2][0] += dz * dx; cov[2][1] += dz * dy; cov[2][2] += dz * dz;
    }

    Eigen::Matrix3f eigMat;
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j)
            eigMat(i, j) = cov[i][j];

    Eigen::SelfAdjointEigenSolver<Eigen::Matrix3f> eig(eigMat);
    Eigen::Vector3f n = eig.eigenvectors().col(0);  // Direction corresponding to smallest eigenvalue = normal

    return Normal3D(n[0], n[1], n[2]);
}

static std::vector<PointWithNormal> estimateNormals(const std::vector<PointWithNormal>& pts, int k = 20) {
    std::vector<PointWithNormal> result;
    result.reserve(pts.size());

    KDTreeWrapper kdtree(pts);

    for (const auto& p : pts) {
        Normal3D n = computeNormal(pts, kdtree, p.point, k);
        result.emplace_back(p.point, n);
    }

    return result;
}

void updateNormals(std::vector<PointWithNormal>& pts, int k = 20) {
    KDTreeWrapper kdtree(pts);
    for (auto& p : pts) {
        p.normal = computeNormal(pts, kdtree, p.point, k);
    }
}


int main() {
    // Extract image name from camera parameter file path (e.g., "test12" from "test12.txt")
    std::filesystem::path cameraPathObj("../data/camera/test12.txt");
    std::string imageName = cameraPathObj.stem().string(); // Gets "test12" from "test12.txt"
    
    // Create output directory with image name
    std::string outputDir = "../output/" + imageName;
    std::filesystem::create_directories(outputDir);
    
    std::cout << "Output directory: " << outputDir << std::endl;
    
    std::vector<PointWithNormal> A, B;

    if (!loadPLY(outputDir + "/pointcloud_left.ply", A)) {
        std::cerr << "❌ Failed to load source point cloud" << std::endl;
        return -1;
    }

    if (!loadPLY(outputDir + "/pointcloud_right.ply", B)) {
        std::cerr << "❌ Failed to load target point cloud" << std::endl;
        return -1;
    }

    std::cout << "Loaded point clouds: A(source)=" << A.size() << " points, B(target)=" << B.size() << " points" << std::endl;

    // Coarse registration: manually translate right view point cloud  baseline ≈ +0.1m to the right
    // for (auto& p : B) {
    //     p.point.x += 0.1f;  // ❗️You can adjust according to actual baseline, unit: m
    // }
    
    // Automatically estimate normals
    updateNormals(A);
    updateNormals(B);

    auto normalizePointCloud = [](std::vector<PointWithNormal>& pts) {
        if (pts.empty()) return;

        Point3D c = {0, 0, 0};
        for (const auto& p : pts) {
            c.x += p.point.x;
            c.y += p.point.y;
            c.z += p.point.z;
        }
        c.x /= pts.size(); c.y /= pts.size(); c.z /= pts.size();

        float scale = 0;
        for (const auto& p : pts) {
            float dx = p.point.x - c.x;
            float dy = p.point.y - c.y;
            float dz = p.point.z - c.z;
            scale += std::sqrt(dx*dx + dy*dy + dz*dz);
        }
        scale /= pts.size();

        for (auto& p : pts) {
            p.point.x = (p.point.x - c.x) / scale;
            p.point.y = (p.point.y - c.y) / scale;
            p.point.z = (p.point.z - c.z) / scale;
        }
    };

    // Insert position: after loadPLY
    normalizePointCloud(A);
    normalizePointCloud(B);

    // Use modified ICP to get matched point indices
    std::vector<int> matched_B_indices;
    Transform T = SelectivePointToPlaneICP::alignWithMatchedPoints(A, B, 50, 0.05f, 1e-4f, matched_B_indices);

    // Apply transformation to all B points first
    std::vector<PointWithNormal> B_aligned;
    for (const auto& p : B) {
        Point3D q = rotate(T.R, p.point);
        q.x += T.t[0];
        q.y += T.t[1];
        q.z += T.t[2];
        B_aligned.emplace_back(q, p.normal);  // Preserve normal structure
    }

    std::cout << "Transformation matrix R:" << std::endl;
    for (int i = 0; i < 3; ++i)
        std::cout << T.R[i][0] << " " << T.R[i][1] << " " << T.R[i][2] << std::endl;

    std::cout << "Translation vector t: "
              << T.t[0] << " " << T.t[1] << " " << T.t[2] << std::endl;

    // Create matched-only B point cloud
    std::vector<PointWithNormal> B_matched_only;
    B_matched_only.reserve(matched_B_indices.size());
    
    for (int idx : matched_B_indices) {
        if (idx >= 0 && idx < B_aligned.size()) {
            B_matched_only.push_back(B_aligned[idx]);
        }
    }
    
    std::cout << "Created matched B point cloud: " << B_matched_only.size() << " points" << std::endl;

    // Save transformation matrix to text file
    std::ofstream transformFile(outputDir + "/icp_transformation.txt");
    if (transformFile.is_open()) {
        transformFile << "ICP Transformation Results" << std::endl;
        transformFile << "==========================" << std::endl;
        transformFile << std::endl;
        transformFile << "Rotation Matrix R:" << std::endl;
        for (int i = 0; i < 3; ++i) {
            transformFile << T.R[i][0] << " " << T.R[i][1] << " " << T.R[i][2] << std::endl;
        }
        transformFile << std::endl;
        transformFile << "Translation Vector t:" << std::endl;
        transformFile << T.t[0] << " " << T.t[1] << " " << T.t[2] << std::endl;
        transformFile << std::endl;
        transformFile << "Point Cloud Statistics:" << std::endl;
        transformFile << "Source points (left): " << A.size() << std::endl;
        transformFile << "Target points (right): " << B.size() << std::endl;
        transformFile << "Aligned points (all): " << B_aligned.size() << std::endl;
        transformFile << "Matched points only: " << B_matched_only.size() << std::endl;
        transformFile << "Match percentage: " << (100.0 * matched_B_indices.size() / B.size()) << "%" << std::endl;
        transformFile << "Unmatched points removed: " << (B.size() - B_matched_only.size()) << std::endl;
        transformFile.close();
        std::cout << "Transformation matrix saved to: " << outputDir << "/icp_transformation.txt" << std::endl;
    }

    // Save different versions of the result
    
    // 1. Original: A (green) + B_aligned (red) - complete alignment result
    if (!saveColoredPLY(outputDir + "/aligned_colored_complete.ply", A, B_aligned)){
        std::cerr << "❌ Failed to save complete alignment result" << std::endl;
        return -1;
    }
    std::cout << "✅ Complete alignment result saved to " << outputDir << "/aligned_colored_complete.ply" << std::endl;
    
    // 2. Selective: A (green) + B matched only (blue) - your requested version
    if (!saveSelectivePLY(outputDir + "/aligned_selective.ply", A, B_matched_only)){
        std::cerr << "❌ Failed to save selective result" << std::endl;
        return -1;
    }
    std::cout << "✅ Selective alignment result saved to " << outputDir << "/aligned_selective.ply" << std::endl;
    std::cout << "   - A (complete): " << A.size() << " points (green)" << std::endl;
    std::cout << "   - B (matched only): " << B_matched_only.size() << " points (blue)" << std::endl;
    std::cout << "   - Total: " << (A.size() + B_matched_only.size()) << " points" << std::endl;
    std::cout << "   - Reduction: " << (B.size() - B_matched_only.size()) << " unmatched points removed from B" << std::endl;

    // 3. Also save the traditional version for comparison
    if (!saveColoredPLY(outputDir + "/aligned_colored.ply", A, B_aligned)){
        std::cerr << "❌ Failed to save traditional result" << std::endl;
        return -1;
    }
    std::cout << "✅ Traditional alignment result saved to " << outputDir << "/aligned_colored.ply" << std::endl;

    // 4. Save individual aligned point clouds
    if (!savePLY(outputDir + "/aligned_left.ply", A)) {
        std::cerr << "❌ Failed to save aligned left point cloud" << std::endl;
    } else {
        std::cout << "Aligned left point cloud saved to: " << outputDir << "/aligned_left.ply" << std::endl;
    }

    if (!savePLY(outputDir + "/aligned_right.ply", B_aligned)) {
        std::cerr << "❌ Failed to save aligned right point cloud" << std::endl;
    } else {
        std::cout << "Aligned right point cloud saved to: " << outputDir << "/aligned_right.ply" << std::endl;
    }

    // 5. Save matched-only right point cloud
    if (!savePLY(outputDir + "/aligned_right_matched_only.ply", B_matched_only)) {
        std::cerr << "❌ Failed to save matched-only right point cloud" << std::endl;
    } else {
        std::cout << "Matched-only right point cloud saved to: " << outputDir << "/aligned_right_matched_only.ply" << std::endl;
    }

    std::cout << "\n=== Processing Completed ===" << std::endl;
    std::cout << "All results saved in directory: " << outputDir << std::endl;

    return 0;
}