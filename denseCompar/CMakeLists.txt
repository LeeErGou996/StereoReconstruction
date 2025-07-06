cmake_minimum_required(VERSION 3.10)
project(DenseCompar)

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

# 添加可执行文件
add_executable(denseCompar 
    main.cpp
    denseMatching.cpp
    sgbmMatcher.cpp
    bmMatcher.cpp
    depth.cpp
    config.cpp
    sparseMatching.cpp
    8point.cpp
    elasMatcher.cpp
    elas/elas.cpp
    elas/triangle.cpp
    elas/descriptor.cpp
    elas/matrix.cpp
    elas/filter.cpp
    color_disparity_quality_metrics.cpp

)

# 添加色差图质量评估报告生成器
add_executable(color_quality_report
    generate_color_quality_report.cpp
    color_disparity_quality_metrics.cpp
)

# 链接OpenCV库
target_link_libraries(denseCompar ${OpenCV_LIBS})
target_link_libraries(color_quality_report ${OpenCV_LIBS})



# 链接X11库（CImg需要）
find_package(PkgConfig REQUIRED)
pkg_check_modules(X11 REQUIRED x11)
target_link_libraries(denseCompar ${X11_LIBRARIES})
target_include_directories(denseCompar PRIVATE ${X11_INCLUDE_DIRS})
target_link_libraries(color_quality_report ${X11_LIBRARIES})
target_include_directories(color_quality_report PRIVATE ${X11_INCLUDE_DIRS})

# 设置编译选项
if(MSVC)
    target_compile_options(denseCompar PRIVATE /W4)
    target_compile_options(color_quality_report PRIVATE /W4)
else()
    target_compile_options(denseCompar PRIVATE -Wall -Wextra -O2 -ffast-math -march=native)
    target_compile_options(color_quality_report PRIVATE -Wall -Wextra -O2 -ffast-math -march=native)
endif()

# 输出目录设置 - 直接输出到build目录
set_target_properties(denseCompar PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}
)
set_target_properties(color_quality_report PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}
)

# 打印OpenCV版本信息
message(STATUS "OpenCV version: ${OpenCV_VERSION}")
message(STATUS "OpenCV libraries: ${OpenCV_LIBS}")
message(STATUS "OpenCV include path: ${OpenCV_INCLUDE_DIRS}") 