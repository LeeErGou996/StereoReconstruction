@echo off
chcp 65001 >nul

echo Generating file list...
set OUTPUT=filelist.txt

REM 清空输出文件
echo > %OUTPUT%

REM 输出一级目录和文件
echo [Level 1] >> %OUTPUT%
for %%F in (*) do echo %%F >> %OUTPUT%
for /d %%D in (*) do echo %%D\ >> %OUTPUT%

REM 输出二级目录和文件
echo. >> %OUTPUT%
echo [Level 2] >> %OUTPUT%
for /d %%D in (*) do (
    for %%F in (%%D\*) do echo %%D\%%~nxF >> %OUTPUT%
    for /d %%S in (%%D\*) do echo %%D\%%S\ >> %OUTPUT%
)

echo Done. File list saved to %OUTPUT%
pause
