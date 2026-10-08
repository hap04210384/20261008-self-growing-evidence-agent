@echo off
call "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat" >nul 2>&1
set NVCC="C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.5\bin\nvcc.exe"
cd /d "%~dp0"
%NVCC% -O3 -w -Xcompiler /openmp -arch=sm_86 -c data.cpp -o data.obj
if errorlevel 1 exit /b 1
%NVCC% -O3 -w -Xcompiler /openmp -arch=sm_86 -c main.cu -o main.obj
if errorlevel 1 exit /b 1
%NVCC% -arch=sm_86 -o GMiner.exe data.obj main.obj
if errorlevel 1 exit /b 1
echo BUILD OK
