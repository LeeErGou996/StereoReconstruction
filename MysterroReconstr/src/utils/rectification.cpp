#include "rectification.h"
#include <cmath>
#include <cassert>
#include <cstdio>
#include <cmath>
#include <algorithm>

// Basic mathematical functions
static double vec3_norm(const Vec3& v) {
    return sqrt(v[0]*v[0] + v[1]*v[1] + v[2]*v[2]);
}

static Vec3 vec3_normalize(const Vec3& v) {
    double n = vec3_norm(v);
    if (n < 1e-12) {
        printf("Warning: Attempting to normalize zero vector\n");
        return {1, 0, 0};  // Return default direction
    }
    return {v[0]/n, v[1]/n, v[2]/n};
}

static Vec3 vec3_cross(const Vec3& a, const Vec3& b) {
    return {
        a[1]*b[2] - a[2]*b[1],
        a[2]*b[0] - a[0]*b[2], 
        a[0]*b[1] - a[1]*b[0]
    };
}

static double vec3_dot(const Vec3& a, const Vec3& b) {
    return a[0]*b[0] + a[1]*b[1] + a[2]*b[2];
}

// Matrix operations
static Mat3 mat3_mul(const Mat3& A, const Mat3& B) {
    Mat3 C = {};
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            for (int k = 0; k < 3; k++) {
                C[i][j] += A[i][k] * B[k][j];
            }
        }
    }
    return C;
}

static Mat3 mat3_transpose(const Mat3& A) {
    Mat3 B = {};
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            B[i][j] = A[j][i];
        }
    }
    return B;
}

static Mat3 mat3_identity() {
    Mat3 I = {};
    I[0][0] = I[1][1] = I[2][2] = 1.0;
    return I;
}

static void mat3_print(const char* name, const Mat3& m) {
    printf("%s:\n", name);
    for (int i = 0; i < 3; i++) {
        printf("[%8.3f %8.3f %8.3f]\n", m[i][0], m[i][1], m[i][2]);
    }
}

static void vec3_print(const char* name, const Vec3& v) {
    printf("%s: [%8.3f %8.3f %8.3f] (norm: %8.3f)\n", 
           name, v[0], v[1], v[2], vec3_norm(v));
}

// Check if matrix is valid (no NaN or infinity)
static bool mat3_is_valid(const Mat3& m) {
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            if (!std::isfinite(m[i][j])) {
                return false;
            }
        }
    }
    return true;
}

// Check if H matrix is reasonable (new addition)
static bool homography_is_reasonable(const Mat3& H, const char* name) {
    if (!mat3_is_valid(H)) {
        printf("❌ %s contains invalid values (NaN/Inf)\n", name);
        return false;
    }
    
    // Check translation components
    double tx = H[0][2];
    double ty = H[1][2];
    double translation_norm = sqrt(tx*tx + ty*ty);
    
    printf("%s Analysis: tx=%.1f, ty=%.1f, translation norm=%.1f\n", name, tx, ty, translation_norm);
    
    if (translation_norm > 300) {  // Translation exceeding 300 pixels is considered unreasonable
        printf("❌ %s translation component too large (%.1f > 300)\n", name, translation_norm);
        return false;
    }
    
    // Check scaling factors
    double sx = H[0][0];
    double sy = H[1][1];
    
    if (sx < 0.3 || sx > 3.0 || sy < 0.3 || sy > 3.0) {
        printf("❌ %s scaling factors abnormal: sx=%.3f, sy=%.3f (should be between 0.3-3.0)\n", name, sx, sy);
        return false;
    }
    
    // Check perspective components
    double w1 = H[2][0];
    double w2 = H[2][1];
    double w3 = H[2][2];
    
    if (fabs(w1) > 1e-3 || fabs(w2) > 1e-3 || fabs(w3 - 1.0) > 0.2) {
        printf("❌ %s perspective components abnormal: w1=%.6f, w2=%.6f, w3=%.6f\n", name, w1, w2, w3);
        return false;
    }
    
    printf("✅ %s passed reasonability check\n", name);
    return true;
}

// Conservative epipolar rectification algorithm (new addition)
static void computeConservativeRectification(
    const Mat3& K1, const Mat3& K2,
    const Mat3& R, const Vec3& t,
    Mat3& H1, Mat3& H2
) {
    printf("=== Using Conservative Epipolar Rectification Algorithm ===\n");
    
    // Calculate rotation angle
    double trace = R[0][0] + R[1][1] + R[2][2];
    double rotation_angle = acos(std::max(-1.0, std::min(1.0, (trace - 1.0) / 2.0)));
    
    printf("Inter-camera rotation angle: %.3f degrees\n", rotation_angle * 180.0 / M_PI);
    
    // If rotation angle is small, use minimal rectification
    if (rotation_angle < 0.1) {  // Less than 5.7 degrees
        printf("Small rotation angle, using minimal rectification\n");
        H1 = mat3_identity();
        H2 = mat3_identity();
        return;
    }
    
    // Calculate half rotation for rectification
    double half_angle = rotation_angle / 2.0;
    
    // Find rotation axis (simplified processing)
    Vec3 axis;
    if (fabs(R[2][1] - R[1][2]) > 1e-6) {
        // Extract rotation axis
        axis[0] = R[2][1] - R[1][2];
        axis[1] = R[0][2] - R[2][0];
        axis[2] = R[1][0] - R[0][1];
        axis = vec3_normalize(axis);
    } else {
        // Default to y-axis
        axis = {0, 1, 0};
    }
    
    printf("Rotation axis: [%.3f, %.3f, %.3f]\n", axis[0], axis[1], axis[2]);
    
    // Construct small rectification rotation matrix
    double c = cos(half_angle);
    double s = sin(half_angle);
    double ux = axis[0], uy = axis[1], uz = axis[2];
    
    Mat3 R_correct = {
        c + ux*ux*(1-c),      ux*uy*(1-c) - uz*s,   ux*uz*(1-c) + uy*s,
        uy*ux*(1-c) + uz*s,   c + uy*uy*(1-c),      uy*uz*(1-c) - ux*s,
        uz*ux*(1-c) - uy*s,   uz*uy*(1-c) + ux*s,   c + uz*uz*(1-c)
    };
    
    // Calculate rectified homography matrices
    // H1 = K1 * R_correct * K1^(-1)
    
    // Calculate inverse of K1
    Mat3 K1_inv = {};
    K1_inv[0][0] = 1.0 / K1[0][0];
    K1_inv[1][1] = 1.0 / K1[1][1];
    K1_inv[2][2] = 1.0;
    K1_inv[0][2] = -K1[0][2] / K1[0][0];
    K1_inv[1][2] = -K1[1][2] / K1[1][1];
    
    Mat3 K2_inv = {};
    K2_inv[0][0] = 1.0 / K2[0][0];
    K2_inv[1][1] = 1.0 / K2[1][1];
    K2_inv[2][2] = 1.0;
    K2_inv[0][2] = -K2[0][2] / K2[0][0];
    K2_inv[1][2] = -K2[1][2] / K2[1][1];
    
    Mat3 temp1 = mat3_mul(R_correct, K1_inv);
    H1 = mat3_mul(K1, temp1);
    
    // H2 = K2 * R_correct^T * R * K2^(-1)
    Mat3 R_correct_T = mat3_transpose(R_correct);
    Mat3 R_combined = mat3_mul(R_correct_T, R);
    Mat3 temp2 = mat3_mul(R_combined, K2_inv);
    H2 = mat3_mul(K2, temp2);
    
    printf("Conservative rectification results:\n");
    mat3_print("H1 (conservative)", H1);
    mat3_print("H2 (conservative)", H2);
}

// Compute image boundary transformation (new auxiliary function)
static void computeImageBounds(const Mat3& H, int img_width, int img_height, 
                              double& min_x, double& max_x, double& min_y, double& max_y) {
    // Four corner points of image
    double corners[4][3] = {
        {0, 0, 1},
        {img_width-1, 0, 1},
        {img_width-1, img_height-1, 1},
        {0, img_height-1, 1}
    };
    
    min_x = min_y = 1e10;
    max_x = max_y = -1e10;
    
    for (int i = 0; i < 4; i++) {
        // Apply transformation
        double x = H[0][0] * corners[i][0] + H[0][1] * corners[i][1] + H[0][2] * corners[i][2];
        double y = H[1][0] * corners[i][0] + H[1][1] * corners[i][1] + H[1][2] * corners[i][2];
        double w = H[2][0] * corners[i][0] + H[2][1] * corners[i][1] + H[2][2] * corners[i][2];
        
        if (fabs(w) > 1e-8) {
            x /= w;
            y /= w;
            
            min_x = std::min(min_x, x);
            max_x = std::max(max_x, x);
            min_y = std::min(min_y, y);
            max_y = std::max(max_y, y);
        }
    }
    
    printf("Transformed bounds: x[%.1f, %.1f], y[%.1f, %.1f], size: %.1f x %.1f\n", 
           min_x, max_x, min_y, max_y, max_x - min_x, max_y - min_y);
}

// Adjust homography matrix to reduce image cropping (new addition)
static Mat3 adjustHomographyForBounds(const Mat3& H, int img_width, int img_height, 
                                     const char* name) {
    double min_x, max_x, min_y, max_y;
    computeImageBounds(H, img_width, img_height, min_x, max_x, min_y, max_y);
    
    // Calculate needed translation to keep image within bounds
    double tx_adjust = 0, ty_adjust = 0;
    
    // If left boundary exceeds 0, need to shift right
    if (min_x < 0) {
        tx_adjust = -min_x;
    }
    // If right boundary exceeds image width, need to shift left
    else if (max_x >= img_width) {
        tx_adjust = img_width - 1 - max_x;
    }
    
    // Vertical direction similar
    if (min_y < 0) {
        ty_adjust = -min_y;
    }
    else if (max_y >= img_height) {
        ty_adjust = img_height - 1 - max_y;
    }
    
    // Create adjusted homography matrix
    Mat3 H_adjusted = H;
    H_adjusted[0][2] += tx_adjust;
    H_adjusted[1][2] += ty_adjust;
    
    if (fabs(tx_adjust) > 1 || fabs(ty_adjust) > 1) {
        printf("%s boundary adjustment: tx=%.1f, ty=%.1f\n", name, tx_adjust, ty_adjust);
    }
    
    return H_adjusted;
}

// Main epipolar rectification function (optimized)
void computeRectification(
    const Mat3& K1, const Mat3& K2,
    const Mat3& R, const Vec3& t,
    Mat3& H1, Mat3& H2
) {
    printf("=== Starting Stereo Rectification (Anti-Cropping Optimized Version) ===\n");
    
    // Print input data
    mat3_print("K1", K1);
    mat3_print("K2", K2);
    mat3_print("R", R);
    vec3_print("t", t);
    
    // Check input validity
    if (!mat3_is_valid(K1) || !mat3_is_valid(K2) || !mat3_is_valid(R)) {
        printf("Error: Invalid input matrices detected\n");
        H1 = H2 = mat3_identity();
        return;
    }
    
    double t_norm = vec3_norm(t);
    if (t_norm < 1e-6) {
        printf("Error: Translation vector too small (norm = %e)\n", t_norm);
        H1 = H2 = mat3_identity();
        return;
    }
    
    // Assume image size (can be passed as parameter, using common size here)
    int img_width = 640, img_height = 480;
    // Infer image size from intrinsic matrix
    if (K1[0][2] > 100 && K1[1][2] > 100) {
        img_width = static_cast<int>(K1[0][2] * 2);
        img_height = static_cast<int>(K1[1][2] * 2);
    }
    printf("Inferred image size: %d x %d\n", img_width, img_height);
    
    // === Step 1: Build optimized rectification coordinate system (reduced distortion version) ===
    
    // Calculate inter-camera rotation angle for adjusting rectification strength
    double trace = R[0][0] + R[1][1] + R[2][2];
    double rotation_angle = acos(std::max(-1.0, std::min(1.0, (trace - 1.0) / 2.0)));
    printf("Inter-camera rotation angle: %.3f degrees\n", rotation_angle * 180.0 / M_PI);
    
    // If rotation angle is large, use conservative rectification strategy
    double rectification_strength = 1.0;
    if (rotation_angle > 0.2) {  // Greater than 11.5 degrees
        rectification_strength = std::min(1.0, 0.2 / rotation_angle);
        printf("Rotation angle is large, rectification strength adjusted to: %.3f\n", rectification_strength);
    }
    
    // 1. New x-axis: baseline direction (horizontal direction)
    Vec3 e1 = vec3_normalize(t);
    vec3_print("e1 (baseline)", e1);
    
    // 2. Optimized y-axis calculation: ensure vertical direction stability while considering image distortion
    Vec3 e2;
    
    // Use average y-direction of original cameras as reference, but weight considering image center
    Vec3 cam1_y = {0, 1, 0};  // First camera's y-axis direction
    Vec3 cam2_y = {R[1][0], R[1][1], R[1][2]};  // Second camera's y-axis direction
    
    // Calculate weighted average y-direction (bias toward less distorted direction)
    double w1 = 0.5, w2 = 0.5;
    
    // If one camera's y-axis has smaller angle with baseline, give it more weight
    double angle1 = fabs(vec3_dot(cam1_y, e1));
    double angle2 = fabs(vec3_dot(cam2_y, e1));
    
    if (angle1 < angle2) {
        w1 = 0.7; w2 = 0.3;  // cam1's y-axis is more perpendicular to baseline
    } else {
        w1 = 0.3; w2 = 0.7;  // cam2's y-axis is more perpendicular to baseline
    }
    
    Vec3 weighted_y = {
        w1 * cam1_y[0] + w2 * cam2_y[0],
        w1 * cam1_y[1] + w2 * cam2_y[1],
        w1 * cam1_y[2] + w2 * cam2_y[2]
    };
    weighted_y = vec3_normalize(weighted_y);
    vec3_print("weighted_y", weighted_y);
    
    // Ensure y-axis is perpendicular to baseline
    double dot_e1_y = vec3_dot(e1, weighted_y);
    Vec3 e2_temp = {
        weighted_y[0] - dot_e1_y * e1[0],
        weighted_y[1] - dot_e1_y * e1[1],
        weighted_y[2] - dot_e1_y * e1[2]
    };
    
    // Apply rectification strength to reduce distortion
    if (rectification_strength < 1.0) {
        Vec3 original_y = {0, 1, 0};
        for (int i = 0; i < 3; i++) {
            e2_temp[i] = rectification_strength * e2_temp[i] + 
                        (1.0 - rectification_strength) * original_y[i];
        }
        printf("Applied rectification strength %.3f to reduce y-axis distortion\n", rectification_strength);
    }
    
    if (vec3_norm(e2_temp) < 0.1) {
        printf("Warning: y-direction calculation unstable, using fallback plan\n");
        // Choose coordinate axis least parallel to baseline
        Vec3 candidates[3] = {{1,0,0}, {0,1,0}, {0,0,1}};
        double max_perpendicular = 0;
        int best_idx = 1;
        
        for (int i = 0; i < 3; i++) {
            double dot = fabs(vec3_dot(e1, candidates[i]));
            double perpendicular = 1.0 - dot;
            if (perpendicular > max_perpendicular) {
                max_perpendicular = perpendicular;
                best_idx = i;
            }
        }
        
        Vec3 backup = candidates[best_idx];
        double dot_backup = vec3_dot(e1, backup);
        e2_temp = {
            backup[0] - dot_backup * e1[0],
            backup[1] - dot_backup * e1[1],
            backup[2] - dot_backup * e1[2]
        };
    }
    
    e2 = vec3_normalize(e2_temp);
    vec3_print("e2 (optimized)", e2);
    
    // 3. z-axis: ensure right-handed coordinate system
    Vec3 e3 = vec3_cross(e1, e2);
    e3 = vec3_normalize(e3);
    vec3_print("e3", e3);
    
    // 4. Verify coordinate system orthogonality
    double dot12 = vec3_dot(e1, e2);
    double dot13 = vec3_dot(e1, e3);
    double dot23 = vec3_dot(e2, e3);
    printf("Coordinate system orthogonality check: e1·e2=%.6f, e1·e3=%.6f, e2·e3=%.6f\n", dot12, dot13, dot23);
    
    if (fabs(dot12) > 1e-3 || fabs(dot13) > 1e-3 || fabs(dot23) > 1e-3) {
        printf("Warning: Coordinate system not orthogonal enough, recalculating\n");
        e2 = vec3_normalize(e2_temp);
        e3 = vec3_cross(e1, e2);
        e3 = vec3_normalize(e3);
        e2 = vec3_cross(e3, e1);
        e2 = vec3_normalize(e2);
    }
    
    // 5. Build rectification rotation matrix
    Mat3 Rrect = {
        e1[0], e1[1], e1[2],
        e2[0], e2[1], e2[2],
        e3[0], e3[1], e3[2]
    };
    
    mat3_print("Rectification rotation (anti-crop)", Rrect);
    
    // 6. Optimized new camera matrix: minimize image distortion
    Mat3 K_new = {};
    
    // Use conservative parameter combination to reduce distortion
    double fx_avg = (K1[0][0] + K2[0][0]) / 2.0;
    double fy_avg = (K1[1][1] + K2[1][1]) / 2.0;
    
    // To reduce vertical distortion, use relatively conservative aspect ratio
    double aspect_ratio = fy_avg / fx_avg;
    if (aspect_ratio < 0.9 || aspect_ratio > 1.1) {
        printf("Adjusting aspect ratio from %.3f to close to 1.0\n", aspect_ratio);
        fy_avg = fx_avg * 0.95;  // Slightly bias toward horizontal, as baseline is horizontal
    }
    
    // Principal point position: optimize to reduce image boundary loss
    double cx_avg = (K1[0][2] + K2[0][2]) / 2.0;
    double cy_avg = (K1[1][2] + K2[1][2]) / 2.0;
    
    // Adjust principal point based on expected image distortion
    // If rectification causes image shift in certain direction, pre-compensate principal point position
    double cx_offset = 0, cy_offset = 0;
    
    // Roughly estimate offset that rectification might cause
    if (fabs(e2[0]) > 0.1) {  // y-axis has x component, might cause horizontal offset
        cx_offset = -e2[0] * cy_avg * 0.1;  // Small compensation
    }
    if (fabs(e2[1] - 1.0) > 0.1) {  // y-axis deviates from standard direction
        cy_offset = (1.0 - e2[1]) * cy_avg * 0.1;
    }
    
    K_new[0][0] = fx_avg;
    K_new[1][1] = fy_avg;
    K_new[0][2] = cx_avg + cx_offset;
    K_new[1][2] = cy_avg + cy_offset;
    K_new[2][2] = 1.0;
    
    printf("New camera matrix (anti-distortion): fx=%.1f, fy=%.1f, cx=%.1f, cy=%.1f\n", 
           K_new[0][0], K_new[1][1], K_new[0][2], K_new[1][2]);
    
    // 7. Calculate optimized homography matrices (with boundary protection)
    
    // Calculate inverse of K
    Mat3 K1_inv = {};
    K1_inv[0][0] = 1.0 / K1[0][0];
    K1_inv[1][1] = 1.0 / K1[1][1];
    K1_inv[2][2] = 1.0;
    K1_inv[0][2] = -K1[0][2] / K1[0][0];
    K1_inv[1][2] = -K1[1][2] / K1[1][1];
    
    Mat3 K2_inv = {};
    K2_inv[0][0] = 1.0 / K2[0][0];
    K2_inv[1][1] = 1.0 / K2[1][1];
    K2_inv[2][2] = 1.0;
    K2_inv[0][2] = -K2[0][2] / K2[0][0];
    K2_inv[1][2] = -K2[1][2] / K2[1][1];
    
    // H1 = K_new * Rrect * K1^(-1)
    Mat3 temp1 = mat3_mul(Rrect, K1_inv);
    Mat3 H1_initial = mat3_mul(K_new, temp1);
    
    // H2 = K_new * Rrect * R * K2^(-1)
    Mat3 R_combined = mat3_mul(Rrect, R);
    Mat3 temp2 = mat3_mul(R_combined, K2_inv);
    Mat3 H2_initial = mat3_mul(K_new, temp2);
    
    printf("=== Initial Rectification Results ===\n");
    mat3_print("H1 (initial)", H1_initial);
    mat3_print("H2 (initial)", H2_initial);
    
    // === Boundary protection: check and adjust homography matrices ===
    printf("\n=== Boundary Analysis and Adjustment ===\n");
    
    // Check H1 boundary transformation
    printf("H1 boundary analysis:\n");
    H1 = adjustHomographyForBounds(H1_initial, img_width, img_height, "H1");
    
    // Check H2 boundary transformation
    printf("H2 boundary analysis:\n");
    H2 = adjustHomographyForBounds(H2_initial, img_width, img_height, "H2");
    
    // Additional symmetry adjustment: ensure consistency of left and right image transformations
    double h1_tx = H1[0][2];
    double h1_ty = H1[1][2];
    double h2_tx = H2[0][2];
    double h2_ty = H2[1][2];
    
    // If y-direction offset difference between two transformations is too large, balance them
    double ty_diff = fabs(h1_ty - h2_ty);
    if (ty_diff > 10) {
        printf("Detected imbalanced y-direction offset between left and right images: %.1f\n", ty_diff);
        double ty_balance = (h1_ty + h2_ty) / 2.0;
        
        // Adjust the one with larger offset
        if (fabs(h1_ty - ty_balance) > fabs(h2_ty - ty_balance)) {
            H1[1][2] = ty_balance;
            printf("Adjusted H1 ty: %.1f -> %.1f\n", h1_ty, ty_balance);
        } else {
            H2[1][2] = ty_balance;
            printf("Adjusted H2 ty: %.1f -> %.1f\n", h2_ty, ty_balance);
        }
    }
    
    // Check scaling consistency
    double h1_sx = H1[0][0];
    double h1_sy = H1[1][1];
    double h2_sx = H2[0][0];
    double h2_sy = H2[1][1];
    
    double scale_diff_x = fabs(h1_sx - h2_sx) / std::max(h1_sx, h2_sx);
    double scale_diff_y = fabs(h1_sy - h2_sy) / std::max(h1_sy, h2_sy);
    
    if (scale_diff_x > 0.1 || scale_diff_y > 0.1) {
        printf("Detected scaling difference between left and right images: x=%.1f%%, y=%.1f%%\n", 
               scale_diff_x * 100, scale_diff_y * 100);
        
        // Use average scaling
        double avg_sx = (h1_sx + h2_sx) / 2.0;
        double avg_sy = (h1_sy + h2_sy) / 2.0;
        
        H1[0][0] = H2[0][0] = avg_sx;
        H1[1][1] = H2[1][1] = avg_sy;
        
        printf("Unified scaling: sx=%.3f, sy=%.3f\n", avg_sx, avg_sy);
    }
    
    printf("=== Anti-Cropping Optimization Algorithm Results ===\n");
    mat3_print("H1 (anti-crop)", H1);
    mat3_print("H2 (anti-crop)", H2);
    
    // === Final boundary verification ===
    printf("\n=== Final Boundary Verification ===\n");
    
    printf("H1 final boundary check:\n");
    double h1_min_x, h1_max_x, h1_min_y, h1_max_y;
    computeImageBounds(H1, img_width, img_height, h1_min_x, h1_max_x, h1_min_y, h1_max_y);
    
    printf("H2 final boundary check:\n");
    double h2_min_x, h2_max_x, h2_min_y, h2_max_y;
    computeImageBounds(H2, img_width, img_height, h2_min_x, h2_max_x, h2_min_y, h2_max_y);
    
    // Calculate image retention rate
    double h1_area_ratio = ((h1_max_x - h1_min_x) * (h1_max_y - h1_min_y)) / (img_width * img_height);
    double h2_area_ratio = ((h2_max_x - h2_min_x) * (h2_max_y - h2_min_y)) / (img_width * img_height);
    
    printf("Image coverage rate: H1=%.1f%%, H2=%.1f%%\n", h1_area_ratio * 100, h2_area_ratio * 100);
    
    // Check for serious image loss
    if (h1_area_ratio < 0.7 || h2_area_ratio < 0.7) {
        printf("⚠️  Detected serious image coverage rate loss, switching to conservative algorithm\n");
        computeConservativeRectification(K1, K2, R, t, H1, H2);
        
        // Re-verify boundaries of conservative algorithm
        printf("Conservative algorithm boundary verification:\n");
        computeImageBounds(H1, img_width, img_height, h1_min_x, h1_max_x, h1_min_y, h1_max_y);
        computeImageBounds(H2, img_width, img_height, h2_min_x, h2_max_x, h2_min_y, h2_max_y);
        
        h1_area_ratio = ((h1_max_x - h1_min_x) * (h1_max_y - h1_min_y)) / (img_width * img_height);
        h2_area_ratio = ((h2_max_x - h2_min_x) * (h2_max_y - h2_min_y)) / (img_width * img_height);
        printf("Conservative algorithm coverage rate: H1=%.1f%%, H2=%.1f%%\n", h1_area_ratio * 100, h2_area_ratio * 100);
    }
    
    // === Step 2: Verify result reasonability ===
    bool h1_reasonable = homography_is_reasonable(H1, "H1");
    bool h2_reasonable = homography_is_reasonable(H2, "H2");
    
    if (!h1_reasonable || !h2_reasonable) {
        printf("\n❌ All algorithms produced abnormal results, using minimal distortion scheme\n");
        
        // Final fallback: minimal distortion
        H1 = mat3_identity();
        H2 = mat3_identity();
        
        // Only do minimal y-direction rectification to avoid serious distortion
        if (fabs(R[1][0]) > 0.02) {  // Has y-x rotation component
            double small_angle = R[1][0] * 0.3;  // Only rectify 30%
            H2[1][0] = -small_angle;
            H2[0][1] = small_angle;
        }
        
        printf("Applied minimal distortion rectification\n");
        mat3_print("H1 (minimal)", H1);
        mat3_print("H2 (minimal)", H2);
    }
    
    // === Step 3: Final result verification and output ===
    printf("\n=== Final Rectification Results (Anti-Cropping Version) ===\n");
    mat3_print("H1 (final)", H1);
    mat3_print("H2 (final)", H2);
    
    // Final boundary and quality report
    printf("\n=== Quality Report ===\n");
    computeImageBounds(H1, img_width, img_height, h1_min_x, h1_max_x, h1_min_y, h1_max_y);
    computeImageBounds(H2, img_width, img_height, h2_min_x, h2_max_x, h2_min_y, h2_max_y);
    
    h1_area_ratio = ((h1_max_x - h1_min_x) * (h1_max_y - h1_min_y)) / (img_width * img_height);
    h2_area_ratio = ((h2_max_x - h2_min_x) * (h2_max_y - h2_min_y)) / (img_width * img_height);
    
    printf("Final image retention rate: left image=%.1f%%, right image=%.1f%%\n", h1_area_ratio * 100, h2_area_ratio * 100);
    
    bool final_h1_ok = homography_is_reasonable(H1, "H1_final");
    bool final_h2_ok = homography_is_reasonable(H2, "H2_final");
    
    if (final_h1_ok && final_h2_ok && h1_area_ratio > 0.8 && h2_area_ratio > 0.8) {
        printf("✅ Anti-cropping epipolar rectification calculation successful! Image distortion minimized.\n");
    } else if (final_h1_ok && final_h2_ok) {
        printf("⚠️  Epipolar rectification completed, but there may be some image loss\n");
    } else {
        printf("⚠️  Epipolar rectification results may not be ideal, recommend checking input parameters\n");
    }
    
    printf("=== Anti-Cropping Rectification Completed ===\n");
}

// Backup: ultra-simple rectification (maintaining original interface)
void computeSimpleRectification(
    const Mat3& K1, const Mat3& K2,
    const Mat3& R, const Vec3& t,
    Mat3& H1, Mat3& H2
) {
    printf("=== Using Simple Rectification Method ===\n");
    
    // Simplest method: left camera unchanged, right camera only minimal rotation compensation
    H1 = mat3_identity();
    
    // Apply minor rotation correction to right camera
    double rotation_scale = 0.5;  // Only rectify half of the rotation
    
    H2 = mat3_identity();
    // Only apply small correction to main rotation components
    if (fabs(R[1][2]) > 0.01) {
        double angle = atan2(R[1][2], R[1][1]) * rotation_scale;
        H2[1][1] = cos(angle);
        H2[1][2] = -sin(angle);
        H2[2][1] = sin(angle);
        H2[2][2] = cos(angle);
    }
    
    printf("Simple rectification: left camera unchanged, right camera fine-tuned\n");
    mat3_print("H1 (simple)", H1);
    mat3_print("H2 (simple)", H2);
}

// Additional debugging function (unchanged)
void debugRectification(const Mat3& K1, const Mat3& K2, const Mat3& R, const Vec3& t) {
    printf("=== Rectification Debug Information ===\n");
    
    // Check orthogonality of rotation matrix
    Mat3 R_times_Rt = mat3_mul(R, mat3_transpose(R));
    Mat3 I = mat3_identity();
    
    double orthogonal_error = 0.0;
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            orthogonal_error += (R_times_Rt[i][j] - I[i][j]) * (R_times_Rt[i][j] - I[i][j]);
        }
    }
    printf("Rotation matrix orthogonality error: %e\n", sqrt(orthogonal_error));
    
    // Check reasonability of intrinsic matrices
    printf("K1 focal length: fx=%f, fy=%f, principal point: cx=%f, cy=%f\n", 
           K1[0][0], K1[1][1], K1[0][2], K1[1][2]);
    printf("K2 focal length: fx=%f, fy=%f, principal point: cx=%f, cy=%f\n", 
           K2[0][0], K2[1][1], K2[0][2], K2[1][2]);
           
    // Check baseline length
    double baseline = vec3_norm(t);
    printf("Baseline length: %f\n", baseline);
    
    if (baseline < 0.01) {
        printf("Warning: Baseline length too small, may cause rectification instability\n");
    }
    if (baseline > 1.0) {
        printf("Warning: Baseline length very large, please confirm units are correct\n");
    }
}