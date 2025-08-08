#include "../src/feature_match.cpp"
#include "../src/8point.cpp"
#include <iostream>
#include <vector>
#include <fstream>
#include <sstream>
#include <filesystem>
#include <string>

int main() {
    std::string leftPath = "../data/left/test1.png";
    std::string rightPath = "../data/right/test1.png";
    std::string KPath = "../data/camera/test1.txt";
    
    // Extract image name from path (e.g., "test1" from "test1.png")
    std::filesystem::path leftPathObj(leftPath);
    std::string imageName = leftPathObj.stem().string(); // Gets "test1" from "test1.png"
    
    // Create output directory with image name
    std::string outputDir = "../output/" + imageName;
    std::filesystem::create_directories(outputDir);
    
    std::string outMatchPath = outputDir + "/" + imageName + "_match.png";

    // 1. 特征匹配
    MyImage imgL, imgR;
    if (!myImReadPNG(leftPath, imgL) || !myImReadPNG(rightPath, imgR)) {
        std::cerr << "Failed to read input images!" << std::endl;
        return 1;
    }
    
    // 灰度化处理
    if (imgL.channels > 1) {
        MyImage grayL;
        grayL.width = imgL.width;
        grayL.height = imgL.height;
        grayL.channels = 1;
        grayL.data.resize(imgL.width * imgL.height);
        int step = imgL.channels;
        for (int i = 0; i < imgL.width * imgL.height; ++i) {
            grayL.data[i] = 0.299f * imgL.data[i*step] + 0.587f * imgL.data[i*step+1] + 0.114f * imgL.data[i*step+2];
        }
        imgL = std::move(grayL);
    }
    
    if (imgR.channels > 1) {
        MyImage grayR;
        grayR.width = imgR.width;
        grayR.height = imgR.height;
        grayR.channels = 1;
        grayR.data.resize(imgR.width * imgR.height);
        int step = imgR.channels;
        for (int i = 0; i < imgR.width * imgR.height; ++i) {
            grayR.data[i] = 0.299f * imgR.data[i*step] + 0.587f * imgR.data[i*step+1] + 0.114f * imgR.data[i*step+2];
        }
        imgR = std::move(grayR);
    }

    // 使用修正后的detectAndMatch函数
    auto result = detectAndMatch(imgL, imgR);
    const auto& all_kp = result.first;
    const auto& matches = result.second;
    size_t kp_count = all_kp.size() / 2;
    std::vector<KeyPoint> kpL(all_kp.begin(), all_kp.begin() + kp_count);
    std::vector<KeyPoint> kpR(all_kp.begin() + kp_count, all_kp.end());
    std::cout << "Detected " << kpL.size() << " and " << kpR.size() << " keypoints." << std::endl;
    std::cout << "Matched " << matches.size() << " pairs." << std::endl;

    if (matches.empty()) {
        std::cerr << "No matches found!" << std::endl;
        return 1;
    }

    // 可视化匹配
    MyImage matchImg;
    drawMatches(imgL, kpL, imgR, kpR, matches, matchImg);
    myImWritePNG(outMatchPath, matchImg);
    std::cout << "Saved matching result to: " << outMatchPath << std::endl;
    std::cout << "Output directory created: " << outputDir << std::endl;

    // 2. 读取相机内参K
    EightPoint::Matrix K(3, 3);
    // 初始化为单位矩阵
    K.at(0, 0) = 1.0; K.at(1, 1) = 1.0; K.at(2, 2) = 1.0;
    
    std::ifstream fk(KPath);
    if (fk.is_open()) {
        std::string line;
        std::getline(fk, line);
        fk.close();
        // 提取中括号内内容
        auto l = line.find('[');
        auto r = line.find(']');
        if (l != std::string::npos && r != std::string::npos && r > l) {
            std::string nums = line.substr(l + 1, r - l - 1);
            std::replace(nums.begin(), nums.end(), ';', ' '); // 用空格替换分号
            std::istringstream iss(nums);
            double val;
            int idx = 0;
            while (iss >> val && idx < 9) {
                K.at(idx / 3, idx % 3) = val;
                idx++;
            }
        }
        std::cout << "Loaded camera intrinsics K:" << std::endl;
        K.print();
    } else {
        std::cerr << "Failed to read " << KPath << ", using identity matrix." << std::endl;
        std::cout << "Using identity camera matrix:" << std::endl;
        K.print();
    }

    // 3. 构造匹配点对
    std::vector<EightPoint::Point2f> ptsL, ptsR;
    for (const auto& m : matches) {
        ptsL.emplace_back(kpL[m.idx1].x, kpL[m.idx1].y);
        ptsR.emplace_back(kpR[m.idx2].x, kpR[m.idx2].y);
    }

    std::cout << "\n=== 1. Feature Matching Statistics ===" << std::endl;
    std::cout << "Number of successfully matched pairs: " << ptsL.size() << std::endl;
    
    if (ptsL.size() < 8) {
        std::cerr << "Error: Insufficient matching points for 8-point algorithm (need at least 8, got " << ptsL.size() << ")" << std::endl;
        return 1;
    }

    // 4. 8-point pose estimation
    EightPoint::Matrix R(3, 3), t(3, 1);
    bool ok = EightPoint::estimatePose(ptsL, ptsR, K, R, t);
    
    if (!ok) {
        std::cerr << "8-point pose estimation failed!" << std::endl;
        return 1;
    }

    std::cout << "\n=== 3. Pose Estimation Results ===" << std::endl;
    std::cout << "Rotation matrix R:" << std::endl;
    R.print();
    
    std::cout << "Translation vector t:" << std::endl;
    std::cout << "[" << t.at(0, 0) << ", " << t.at(1, 0) << ", " << t.at(2, 0) << "]" << std::endl;
    
    // Calculate rotation angle (Rodrigues)
    double trace = R.at(0, 0) + R.at(1, 1) + R.at(2, 2);
    double angle = std::acos(std::max(-1.0, std::min(1.0, (trace - 1.0) / 2.0)));
    std::cout << "Rotation angle: " << angle * 180.0 / M_PI << " degrees" << std::endl;
    
    // Calculate translation distance
    double t_norm = std::sqrt(t.at(0, 0) * t.at(0, 0) + t.at(1, 0) * t.at(1, 0) + t.at(2, 0) * t.at(2, 0));
    std::cout << "Translation distance: " << t_norm << " (normalized)" << std::endl;

    // 5. Additional quality check
    std::cout << "\n=== 4. Quality Check ===" << std::endl;
    
    // Check orthogonality of rotation matrix
    EightPoint::Matrix RTR = R.t() * R;
    EightPoint::Matrix I = EightPoint::Matrix::eye(3);
    double orthogonality_error = 0.0;
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            double diff = RTR.at(i, j) - I.at(i, j);
            orthogonality_error += diff * diff;
        }
    }
    orthogonality_error = std::sqrt(orthogonality_error);
    std::cout << "Orthogonality error of rotation matrix: " << orthogonality_error << std::endl;
    
    if (orthogonality_error < 1e-6) {
        std::cout << "Rotation matrix quality: Excellent" << std::endl;
    } else if (orthogonality_error < 1e-4) {
        std::cout << "Rotation matrix quality: Good" << std::endl;
    } else {
        std::cout << "Rotation matrix quality: Needs improvement" << std::endl;
    }

    // Check reprojection error of matching points (simplified)
    double total_error = 0.0;
    int valid_points = 0;
    for (size_t i = 0; i < std::min(ptsL.size(), size_t(20)); ++i) {
        // Here you can add more detailed reprojection error calculation
        // Simplified: assume error is related to feature matching distance
        if (i < matches.size()) {
            total_error += matches[i].dist;
            valid_points++;
        }
    }
    
    if (valid_points > 0) {
        double avg_match_error = total_error / valid_points;
        std::cout << "Average feature matching error: " << avg_match_error << std::endl;
    }

    std::cout << "\n=== Processing Completed ===" << std::endl;
    std::cout << "Feature matching image saved to: " << outMatchPath << std::endl;
    std::cout << "All results saved in directory: " << outputDir << std::endl;
    std::cout << "8-point algorithm pose estimation completed successfully!" << std::endl;

    return 0;
}