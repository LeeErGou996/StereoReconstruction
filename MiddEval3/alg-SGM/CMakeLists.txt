cmake_minimum_required(VERSION 3.10)
project(SGM)

# 设置C++标准
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 设置编译选项
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -Wall")

# 包含目录
include_directories(src)
include_directories(../../../MysterroReconstr/src/utils)
include_directories(../../../MysterroReconstr/src/SGM)

# 查找libpng
find_package(PkgConfig REQUIRED)
pkg_check_modules(PNG REQUIRED libpng)

# 源文件
set(SOURCES
    src/main.cpp
    ../../../MysterroReconstr/src/SGM/SemiGlobalMatching_linux.cpp
    ../../../MysterroReconstr/src/SGM/sgm_util_linux.cpp
    ../../../MysterroReconstr/src/utils/imageutils.cpp
)

# 创建可执行文件
add_executable(sgm ${SOURCES})

# 链接libpng
target_link_libraries(sgm ${PNG_LIBRARIES})
target_include_directories(sgm PRIVATE ${PNG_INCLUDE_DIRS})

# 设置输出目录
set_target_properties(sgm PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}
) 