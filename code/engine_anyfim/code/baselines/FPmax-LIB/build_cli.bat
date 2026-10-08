@echo off
call "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat" >nul 2>&1
cd /d "%~dp0"
cl /O2 /EHsc /std:c++17 /DMFI /I src run_fpmax.cpp src\buffer.cpp src\data.cpp src\fitemset.cpp src\fp_node.cpp src\fp_tree.cpp src\fpmax.cpp src\fsout.cpp /Fe:run_fpmax.exe
