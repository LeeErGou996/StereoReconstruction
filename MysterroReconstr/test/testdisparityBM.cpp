#include "../src/utils/imageutils.h"
#include <iostream>
#include <vector>
#include <thread>
#include <climits>
#include <cmath>
#include <algorithm>

// Simple x-derivative prefilter for MyImage (grayscale, 1 channel)
void xDerivativePrefilter(const MyImage& input, MyImage& output, int preFilterCap = 30) {
    const int offset = 256 * 4;
    unsigned char clampTable[offset * 2 + 256];
    for (int i = 0; i < offset * 2 + 256; i++) {
        int val = i - offset;
        if (val < -preFilterCap)
            clampTable[i] = 0;
        else if (val > preFilterCap)
            clampTable[i] = 2 * preFilterCap;
        else
            clampTable[i] = static_cast<unsigned char>(val + preFilterCap);
    }
    unsigned char defaultValue = clampTable[offset];
    output.width = input.width;
    output.height = input.height;
    output.channels = 1;
    output.data.resize(input.width * input.height);
    for (int y = 0; y < input.height - 1; y += 2) {
        const unsigned char* rowAbove = (y > 0) ? &input.data[(y - 1) * input.width] : &input.data[(y + 1) * input.width];
        const unsigned char* rowCurr = &input.data[y * input.width];
        const unsigned char* rowNext = &input.data[(y + 1) * input.width];
        const unsigned char* rowBelow = (y < input.height - 2) ? &input.data[(y + 2) * input.width] : rowCurr;
        unsigned char* dstRow0 = &output.data[y * input.width];
        unsigned char* dstRow1 = &output.data[(y + 1) * input.width];
        dstRow0[0] = dstRow0[input.width - 1] = defaultValue;
        dstRow1[0] = dstRow1[input.width - 1] = defaultValue;
        for (int x = 1; x < input.width - 1; x++) {
            int gradTop =    rowAbove[x + 1] - rowAbove[x - 1]
                           + 2 * (rowCurr[x + 1] - rowCurr[x - 1])
                           + rowNext[x + 1] - rowNext[x - 1];
            int gradBottom = rowCurr[x + 1] - rowCurr[x - 1]
                           + 2 * (rowNext[x + 1] - rowNext[x - 1])
                           + rowBelow[x + 1] - rowBelow[x - 1];
            dstRow0[x] = clampTable[gradTop + offset];
            dstRow1[x] = clampTable[gradBottom + offset];
        }
    }
    for (int y = (input.height / 2) * 2; y < input.height; y++) {
        std::fill(&output.data[y * input.width], &output.data[(y + 1) * input.width], defaultValue);
    }
}

void StereoBMPreFilterThreads(const MyImage& rectL, const MyImage& rectR, MyImage& disparity) {
    const int numDisparities = 128;
    const int SADWindowSize = 32;
    const int preFilterCap = 30;
    const int minDisparity = 60;
    const int uniquenessThreshold = 20;
    const int N_THREADS = std::thread::hardware_concurrency();
    MyImage leftPref, rightPref;
    xDerivativePrefilter(rectL, leftPref, preFilterCap);
    xDerivativePrefilter(rectR, rightPref, preFilterCap);
    disparity.width = rectL.width;
    disparity.height = rectL.height;
    disparity.channels = 1;
    disparity.data.resize(rectL.width * rectL.height, 0); // 0 means invalid
    const int rows = rectL.height, cols = rectL.width;
    const int win = SADWindowSize / 2;
    auto block_matching_function = [&](int threadID) {
        for (int y = win + threadID; y < rows - win; y += N_THREADS) {
            if (y % 10 == 0)
                std::cout << "Running: " << y << "/" << rows - win << std::endl;
            for (int x = win + numDisparities; x < cols - win; x++) {
                int minSAD = INT_MAX, bestDisp = 0;
                int nextBestSAD[3] = {INT_MAX, INT_MAX, INT_MAX};
                for (int d = minDisparity; d < numDisparities; d++) {
                    int sad = 0;
                    for (int wy = -win; wy <= win; wy++) {
                        const unsigned char* lRow = &leftPref.data[(y + wy) * cols];
                        const unsigned char* rRow = &rightPref.data[(y + wy) * cols];
                        for (int wx = -win; wx <= win; wx++) {
                            int lVal = lRow[x + wx];
                            int rVal = rRow[x + wx - d];
                            sad += std::abs(lVal - rVal);
                        }
                    }
                    if (sad < minSAD) {
                        nextBestSAD[2] = nextBestSAD[1];
                        nextBestSAD[1] = nextBestSAD[0];
                        nextBestSAD[0] = minSAD;
                        minSAD = sad;
                        bestDisp = d;
                    }
                }
                int sadThresh = minSAD + minSAD * uniquenessThreshold / 100;
                if (nextBestSAD[2] > sadThresh)
                    disparity.data[y * cols + x] = static_cast<unsigned char>(bestDisp);
            }
        }
    };
    std::vector<std::thread> pool(N_THREADS);
    for (int i = 0; i < N_THREADS; i++) {
        pool[i] = std::thread(block_matching_function, i);
    }
    for (int i = 0; i < N_THREADS; i++) {
        pool[i].join();
    }
}

void medianFilter3x3(const MyImage& input, MyImage& output) {
    output.width = input.width;
    output.height = input.height;
    output.channels = input.channels;
    output.data.resize(input.width * input.height);

    for (int y = 1; y < input.height - 1; ++y) {
        for (int x = 1; x < input.width - 1; ++x) {
            std::vector<unsigned char> window;
            for (int dy = -1; dy <= 1; ++dy) {
                for (int dx = -1; dx <= 1; ++dx) {
                    unsigned char val = input.data[(y + dy) * input.width + (x + dx)];
                    if (val > 0) window.push_back(val);
                }
            }
            if (!window.empty()) {
                std::sort(window.begin(), window.end());
                output.data[y * input.width + x] = window[window.size() / 2];
            } else {
                output.data[y * input.width + x] = 0;
            }
        }
    }

    for (int x = 0; x < input.width; ++x) {
        output.data[x] = 0;
        output.data[(input.height - 1) * input.width + x] = 0;
    }
    for (int y = 0; y < input.height; ++y) {
        output.data[y * input.width] = 0;
        output.data[y * input.width + input.width - 1] = 0;
    }
}

int main() {
    std::string leftPath = "../data/left/test1.png";
    std::string rightPath = "../data/right/test1.png";
    MyImage imgL, imgR;
    if (!myImReadPNG(leftPath, imgL) || !myImReadPNG(rightPath, imgR)) {
        std::cerr << "Failed to read input images!" << std::endl;
        return 1;
    }
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
    MyImage disparity;
    StereoBMPreFilterThreads(imgL, imgR, disparity);

    MyImage filtered;
    medianFilter3x3(disparity, filtered);
    disparity = std::move(filtered);


    // Normalize and save disparity as PNG
    unsigned char minDisp = 255, maxDisp = 0;
    for (unsigned char v : disparity.data) {
        if (v > 0) {
            minDisp = std::min(minDisp, v);
            maxDisp = std::max(maxDisp, v);
        }
    }
    MyImage dispVis = disparity;
    for (auto& v : dispVis.data) {
        if (v == 0) v = 0;
        else v = static_cast<unsigned char>(255.0f * (v - minDisp) / (maxDisp - minDisp + 1e-5f));
    }
    myImWritePNG("../output/disparity_BM_custom.png", dispVis);
    std::cout << "Disparity map saved to disparity_BM_custom.png" << std::endl;
    return 0;
} 