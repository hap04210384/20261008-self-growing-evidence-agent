@echo off
call "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat" >nul 2>&1
cd /d "%~dp0"
set FPSRC=..\FPmax-LIB-main\src
cl /O2 /EHsc /std:c++17 /openmp /DMFI /I %FPSRC% dmm.cpp %FPSRC%\buffer.cpp %FPSRC%\data.cpp %FPSRC%\fitemset.cpp %FPSRC%\fp_node.cpp %FPSRC%\fp_tree.cpp %FPSRC%\fpmax.cpp %FPSRC%\fsout.cpp /Fe:dmm.exe
if errorlevel 1 exit /b 1
echo BUILD OK
