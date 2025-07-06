cmake_minimum_required(VERSION 3.10)
project(groundtruth_poisson)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 编译优化选项
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O2")
set(CMAKE_BUILD_TYPE Release)

# 启用并行编译
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp")

# 指定 OpenCV 安装路径
set(OpenCV_DIR "../lib/opencv_build")

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

# Groundtruth Poisson重建程序
add_executable(groundtruth_poisson
    ../src/groundtruth_poisson.cpp
    ../src/poissonReconstruction.cpp
    ../src/meshReconstruction.cpp
    ../src/config.cpp
)

target_link_libraries(groundtruth_poisson ${OpenCV_LIBS})

# 如果PCL可用，链接PCL库
if(PCL_FOUND)
    target_link_libraries(groundtruth_poisson ${PCL_LIBRARIES})
endif() 