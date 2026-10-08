@echo off
call "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat"
cd /d "E:\20261003-paper-application of AnyFIM and TensorFIM-03\kimi_word\code\engine_anyfim\code\AnyFIM-Anytime\AnytimeMining"
"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.5\bin\nvcc.exe" -O2 -arch=sm_86 kernel.cu -o AnytimeMining.exe -Xcompiler /openmp 2>&1
echo BUILD_EXIT=%ERRORLEVEL%
