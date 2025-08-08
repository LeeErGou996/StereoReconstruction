#include "../src/utils/imageutils.h"
#include <iostream>

int main() {
    std::string file = "../data/left/test1.png";
    std::string outName = "../output/test1_out.png";
    MyImage img;
    std::cout << "Reading: " << file << std::endl;
    if (!myImReadPNG(file, img)) {
        std::cout << "Failed to read: " << file << std::endl;
        return 1;
    }
    std::cout << "Saving to: " << outName << std::endl;
    if (!myImWritePNG(outName, img)) {
        std::cout << "Failed to write: " << outName << std::endl;
        return 1;
    }
    std::cout << "Test finished." << std::endl;
    return 0;
} 