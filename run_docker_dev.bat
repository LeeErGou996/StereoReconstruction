@echo off
REM Launch Docker container for Stereo Reconstruction project
REM Mount current directory to /workspace and enter bash shell

REM Switch to project directory

REM Launch Docker container with volume mount

docker run -it --rm ^
    -v %cd%:/workspace ^
    --entrypoint /bin/bash ^
    leeergou/3dproj