cmake_minimum_required(VERSION 3.10)
project(testimage)

set(CMAKE_CXX_STANDARD 17)

# 设置CMake策略以处理PCL相关警告
cmake_policy(SET CMP0144 NEW)

# 查找OpenMP支持
find_package(OpenMP)
if(OpenMP_CXX_FOUND)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    message(STATUS "OpenMP found and enabled")
else()
    message(STATUS "OpenMP not found, continuing without OpenMP support")
endif()

# 设置CMake策略以处理导入目标
cmake_policy(SET CMP0111 NEW)

# 查找 libpng
find_package(PNG REQUIRED)

# 查找 PCL
find_package(PCL 1.8 REQUIRED)
if(PCL_FOUND)
    add_definitions(-DHAVE_PCL)
    message(STATUS "PCL found: ${PCL_VERSION}")
else()
    message(STATUS "PCL not found, Poisson reconstruction will use simplified method")
endif()

find_package(Eigen3 REQUIRED)

include_directories(
    ${PNG_INCLUDE_DIR}
    ${PCL_INCLUDE_DIRS}
    ../src/utils
    ../src
    ../src/ELAS/src
    ../src/SGM
)

add_executable(testtripletshow 
    test/testtripletshow.cpp 
    ../src/utils/imageutils.cpp
)
target_link_libraries(testtripletshow PNG::PNG)

add_executable(testimage test/testimage.cpp ../src/utils/imageutils.cpp)
target_link_libraries(testimage PNG::PNG)

add_executable(testfeature_matching test/testfeature_matching.cpp ../src/feature_match.cpp ../src/utils/imageutils.cpp)
target_link_libraries(testfeature_matching PNG::PNG)

add_executable(test8point test/test8point.cpp ../src/utils/imageutils.cpp)
target_link_libraries(test8point PNG::PNG)

add_executable(testRectification test/testRectification.cpp ../src/feature_match.cpp ../src/8point.cpp ../src/utils/imageutils.cpp ../src/utils/rectification.cpp)
target_link_libraries(testRectification PNG::PNG)

add_executable(testdisparity test/testdisparity.cpp ../src/feature_match.cpp ../src/8point.cpp ../src/utils/imageutils.cpp ../src/utils/rectification.cpp ../src/disparity.cpp ../src/ELAS/src/elas.cpp ../src/ELAS/src/triangle.cpp ../src/ELAS/src/matrix.cpp ../src/ELAS/src/filter.cpp ../src/ELAS/src/descriptor.cpp)
target_link_libraries(testdisparity PNG::PNG)

add_executable(testdisparityELAS test/testdisparityELAS.cpp ../src/ELAS/src/elas.cpp ../src/ELAS/src/triangle.cpp ../src/ELAS/src/matrix.cpp ../src/ELAS/src/filter.cpp ../src/ELAS/src/descriptor.cpp)
target_link_libraries(testdisparityELAS PNG::PNG)

# SGM disparity test program
add_executable(testdisparitySGM test/testdisparitySGM.cpp 
    ../src/SGM/SemiGlobalMatching_linux.cpp 
    ../src/SGM/sgm_util_linux.cpp 
    ../src/ELAS/src/elas.cpp 
    ../src/ELAS/src/triangle.cpp 
    ../src/ELAS/src/matrix.cpp 
    ../src/ELAS/src/filter.cpp 
    ../src/ELAS/src/descriptor.cpp)
target_link_libraries(testdisparitySGM PNG::PNG)

# AD-Census disparity test program
add_executable(testdisparityADCE test/testdisparityADCE.cpp 
    ../src/ADCE/ADCensusStereo_linux.cpp 
    ../src/ADCE/cost_computor.cpp 
    ../src/ADCE/cross_aggregator.cpp 
    ../src/ADCE/scanline_optimizer.cpp 
    ../src/ADCE/multistep_refiner.cpp 
    ../src/ADCE/adcensus_util.cpp 
    ../src/ELAS/src/elas.cpp 
    ../src/ELAS/src/triangle.cpp 
    ../src/ELAS/src/matrix.cpp 
    ../src/ELAS/src/filter.cpp 
    ../src/ELAS/src/descriptor.cpp)
target_link_libraries(testdisparityADCE PNG::PNG)

add_executable(testdepth test/testdepth.cpp ../src/feature_match.cpp ../src/8point.cpp ../src/utils/imageutils.cpp ../src/utils/rectification.cpp ../src/disparity.cpp ../src/depth.cpp ../src/ELAS/src/elas.cpp ../src/ELAS/src/triangle.cpp ../src/ELAS/src/matrix.cpp ../src/ELAS/src/filter.cpp ../src/ELAS/src/descriptor.cpp)
target_link_libraries(testdepth PNG::PNG)
# 在某些系统上需要显式链接filesystem库
if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU" AND CMAKE_CXX_COMPILER_VERSION VERSION_LESS "9.0")
    target_link_libraries(testdepth stdc++fs)
endif()

add_executable(testRectificationVis test/testRectificationVis.cpp ../src/feature_match.cpp ../src/8point.cpp ../src/utils/imageutils.cpp ../src/utils/rectification.cpp)
target_link_libraries(testRectificationVis PNG::PNG)

# 添加Poisson重建测试程序
add_executable(testpointcloud test/testpointcloud.cpp 
    src/pointcloud.cpp 
    src/depth.cpp 
    src/utils/imageutils.cpp)
target_include_directories(testpointcloud PRIVATE src)
target_link_libraries(testpointcloud PNG::PNG)
if(OpenMP_CXX_FOUND)
    target_link_libraries(testpointcloud OpenMP::OpenMP_CXX)
endif()
if(PCL_FOUND)
    target_link_libraries(testpointcloud ${PCL_LIBRARIES})
endif()

# 如需其它程序也用到poissonReconstruction.cpp，可仿照上面写法添加

add_executable(testdisparityBM test/testdisparityBM.cpp ../src/utils/imageutils.cpp)
target_link_libraries(testdisparityBM ${PNG_LIBRARY})

# add_executable(testicp 
#     test/testicp.cpp 
#     ../src/icp.cpp 
#     ../src/utils/imageutils.cpp
# )
# target_include_directories(testicp PRIVATE ../src ../src/utils)
# target_link_libraries(testicp PNG::PNG Eigen3::Eigen)

add_executable(testicp 
    test/testicp.cpp 
    ../src/icp.cpp 
    ../src/icp_point2plane.cpp
    ../src/pointcloud.cpp
    ../src/utils/imageutils.cpp
)
target_include_directories(testicp PRIVATE ../src ../src/utils)
target_link_libraries(testicp PNG::PNG Eigen3::Eigen)

add_executable(testicp_plane 
    test/testicp.cpp 
    ../src/icp_point2plane.cpp
    ../src/pointcloud.cpp 
    ../src/utils/imageutils.cpp
)
target_include_directories(testicp_plane PRIVATE ../src ../src/utils)
target_link_libraries(testicp_plane PNG::PNG Eigen3::Eigen)
