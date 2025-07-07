cmake_minimum_required(VERSION 3.10)
project(stereo_vision)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(OPENCV_ENABLE_NONFREE ON)

# Set OpenCV directory to the compiled installation cmake directory
set(OpenCV_DIR "${CMAKE_SOURCE_DIR}/lib/opencv/install/lib/cmake/opencv4")

find_package(OpenCV REQUIRED ximgproc)
include_directories(${OpenCV_INCLUDE_DIRS})

add_executable(main src/main.cpp src/disparity.cpp src/8point.cpp src/denseMatching.cpp 
src/depth.cpp src/meshReconstruction.cpp)
target_link_libraries(main ${OpenCV_LIBS} opencv_ximgproc opencv_xfeatures2d) 