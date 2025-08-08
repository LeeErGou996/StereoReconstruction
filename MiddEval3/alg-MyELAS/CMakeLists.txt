cmake_minimum_required(VERSION 3.10)
project(MyELAS)

# 设置C++标准
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 设置编译选项
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -Wall")

# 包含目录
include_directories(src)

# 源文件
set(SOURCES
    src/main.cpp
    src/elas.cpp
    src/descriptor.cpp
    src/filter.cpp
    src/matrix.cpp
    src/triangle.cpp
)

# 创建可执行文件
add_executable(my_elas ${SOURCES})

# 设置输出目录
set_target_properties(my_elas PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}
) 