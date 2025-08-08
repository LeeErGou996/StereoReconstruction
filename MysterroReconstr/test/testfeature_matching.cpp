#include "../src/feature_match.h"
#include <iostream>

int main() {
    std::string leftImg = "../data/left/test1.png";
    std::string rightImg = "../data/right/test1.png";
    std::string outName = "../output/feature_matches_test1.png";
    std::cout << "Matching: " << leftImg << " <-> " << rightImg << std::endl;
    run_feature_matching(leftImg, rightImg, outName);
    int featL = get_feature_count(leftImg);
    int featR = get_feature_count(rightImg);
    int matches = get_match_count(leftImg, rightImg);
    std::cout << "Features: " << featL << " (L), " << featR << " (R), Matches: " << matches << std::endl;
    std::cout << "Test finished." << std::endl;
    return 0;
} 