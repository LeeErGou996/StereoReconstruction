@echo off
echo === 编译Poisson重建测试程序 ===

REM 设置编译器标志
set CXX_FLAGS=-std=c++14 -O2 -Wall -Wextra
set INCLUDE_DIRS=-I../src -I../src/utils

REM 检查OpenMP支持
echo int main() { return 0; } > omp_test.cpp
g++ -fopenmp omp_test.cpp -o omp_test.exe >nul 2>&1
if %ERRORLEVEL% EQU 0 (
    set CXX_FLAGS=%CXX_FLAGS% -fopenmp
    echo ✓ OpenMP支持已启用
) else (
    echo ⚠ OpenMP不可用，继续编译
)
del omp_test.cpp omp_test.exe >nul 2>&1

REM 编译Poisson重建测试程序
echo 编译testpoisson...
g++ %CXX_FLAGS% %INCLUDE_DIRS% ^
    testpoisson.cpp ^
    ../src/poissonReconstruction.cpp ^
    ../src/depth.cpp ^
    ../src/utils/imageutils.cpp ^
    -lpng ^
    -o testpoisson.exe

if %ERRORLEVEL% EQU 0 (
    echo ✓ testpoisson编译成功!
    echo 运行测试: testpoisson.exe
) else (
    echo ✗ testpoisson编译失败!
    pause
    exit /b 1
)

REM 编译深度图测试程序（如果需要）
echo 编译testdepth...
g++ %CXX_FLAGS% %INCLUDE_DIRS% ^
    testdepth.cpp ^
    ../src/depth.cpp ^
    ../src/utils/imageutils.cpp ^
    -lpng ^
    -o testdepth.exe

if %ERRORLEVEL% EQU 0 (
    echo ✓ testdepth编译成功!
    echo 运行测试: testdepth.exe
) else (
    echo ✗ testdepth编译失败!
)

echo === 编译完成 ===
pause 