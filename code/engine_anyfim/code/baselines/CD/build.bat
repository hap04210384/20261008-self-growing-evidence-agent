@echo off
call "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat" >nul 2>&1
cd /d "%~dp0"
cl /O2 /EHsc /openmp /std:c++17 cd.cpp /Fe:cd.exe
if errorlevel 1 exit /b 1
echo BUILD OK
