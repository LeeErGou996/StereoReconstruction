#pragma once
#include <string>
#include <vector>

struct MyImage;

struct KeyPoint {
    int x, y;
    float response;
    float angle;
    float scale;
    KeyPoint();
    KeyPoint(const struct SIFTKeyPoint& sift_kp);
};

struct Match {
    int idx1, idx2;
    float dist;
    Match();
    Match(const struct SIFTMatch& sift_match);
};

std::pair<std::vector<KeyPoint>, std::vector<Match>> detectAndMatch(const MyImage& img1, const MyImage& img2);

void run_feature_matching(const std::string& leftPath, const std::string& rightPath, const std::string& outPath);
int get_feature_count(const std::string& imgPath);
int get_match_count(const std::string& leftPath, const std::string& rightPath);

void drawMatches(const MyImage& img1, const std::vector<KeyPoint>& kp1,
                 const MyImage& img2, const std::vector<KeyPoint>& kp2,
                 const std::vector<Match>& matches, MyImage& outImg); 