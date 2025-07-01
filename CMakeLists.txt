cmake_minimum_required(VERSION 3.10)
project(stereo_vision)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 指定 OpenCV 安装路径
set(OpenCV_DIR "/config/workspace/StereoReconstruction/lib/opencv_build")

# 引用非自由模块（如 SIFT/SURF）
set(OPENCV_ENABLE_NONFREE ON)

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

# 查找PCL库（可选）
find_package(PCL QUIET)
if(PCL_FOUND)
    add_definitions(-DHAVE_PCL)
    include_directories(${PCL_INCLUDE_DIRS})
    link_directories(${PCL_LIBRARY_DIRS})
    add_definitions(${PCL_DEFINITIONS})
    message(STATUS "PCL found: ${PCL_VERSION}")
else()
    message(WARNING "PCL not found. PCL Poisson reconstruction will not be available.")
endif()

add_executable(main 
    src/main.cpp 
    src/disparity.cpp 
    src/8point.cpp 
    src/denseMatching.cpp 
    src/depth.cpp 
    src/meshReconstruction.cpp
    src/sparseMatching.cpp
    src/pointCloudFilter.cpp
    src/poissonReconstruction.cpp
)

# 添加测试程序
add_executable(test_poisson
    src/test_poisson.cpp
    src/poissonReconstruction.cpp
    src/meshReconstruction.cpp
)

target_link_libraries(main ${OpenCV_LIBS})

# 如果PCL可用，链接PCL库
if(PCL_FOUND)
    target_link_libraries(main ${PCL_LIBRARIES})
    target_link_libraries(test_poisson ${PCL_LIBRARIES})
endif()

target_link_libraries(test_poisson ${OpenCV_LIBS})
