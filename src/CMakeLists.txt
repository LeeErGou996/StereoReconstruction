cmake_minimum_required(VERSION 3.10)
project(RootBA)

# 设置C++标准
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 指定 OpenCV 安装路径（参照根目录配置）
set(OpenCV_DIR "/config/workspace/StereoReconstruction/lib/opencv_build")

# 引用非自由模块（如 SIFT/SURF）
set(OPENCV_ENABLE_NONFREE ON)
add_definitions(-DOPENCV_ENABLE_NONFREE)

# 禁用CImg显示功能以加快编译
add_definitions(-Dcimg_display=0)

# 查找OpenCV包
find_package(OpenCV REQUIRED)

# 尝试查找xfeatures2d模块（用于SURF）
find_package(OpenCV QUIET COMPONENTS xfeatures2d)
if(OpenCV_xfeatures2d_FOUND)
    message(STATUS "OpenCV xfeatures2d module found")
    add_definitions(-DHAVE_OPENCV_XFEATURES2D)
else()
    message(WARNING "OpenCV xfeatures2d module not found, SURF will not be available")
endif()

# 检查OpenCV版本
if(OpenCV_VERSION VERSION_LESS "4.0")
    message(WARNING "OpenCV ${OpenCV_VERSION} found. Version 4.0+ recommended.")
endif()

# 包含OpenCV头文件目录
include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/elas)
include_directories(${CMAKE_CURRENT_BINARY_DIR}/CImg-v.3.2.4)
include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${CMAKE_CURRENT_SOURCE_DIR}/elas
    ${OpenCV_INCLUDE_DIRS}
)

# 列出所有源文件
set(SOURCES
    main.cpp
    denseMatching.cpp
    depth.cpp
    disparity.cpp
    8point.cpp
    meshReconstruction.cpp
    pointCloudFilter.cpp
    poissonReconstruction.cpp
    sparseMatching.cpp
    config.cpp
    elas/elas.cpp
    elas/triangle.cpp
    elas/descriptor.cpp
    elas/matrix.cpp
    elas/filter.cpp

    elasMatcher.cpp
)

# 检查所有文件是否存在
foreach(source_file ${SOURCES})
    if(NOT EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/${source_file}")
        message(FATAL_ERROR "Required source file not found: ${source_file}")
    endif()
endforeach()

# 查找所有头文件
file(GLOB_RECURSE HEADERS 
    "src/*.h"
    "*.h"
)

# 创建可执行文件
add_executable(main ${SOURCES})

# 链接OpenCV库
target_link_libraries(main ${OpenCV_LIBS})

# 链接X11库（CImg需要）
find_package(PkgConfig REQUIRED)
pkg_check_modules(X11 REQUIRED x11)
target_link_libraries(main ${X11_LIBRARIES})
target_include_directories(main PRIVATE ${X11_INCLUDE_DIRS})

# 设置编译选项
if(MSVC)
    target_compile_options(main PRIVATE /W4)
else()
    target_compile_options(main PRIVATE -Wall -Wextra -O2 -ffast-math -march=native)
endif()

# 输出目录设置 - 直接输出到build目录
set_target_properties(main PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}
)

# 输出调试信息
message(STATUS "Found sources: ${SOURCES}")
message(STATUS "Found headers: ${HEADERS}")
message(STATUS "OpenCV version: ${OpenCV_VERSION}")
message(STATUS "OpenCV libraries: ${OpenCV_LIBS}")
message(STATUS "OpenCV include path: ${OpenCV_INCLUDE_DIRS}")