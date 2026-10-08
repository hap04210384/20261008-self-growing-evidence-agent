# GMiner Windows / CUDA 12.5 Porting Notes

Source: opensourcesavvy/GMiner (2018, CUDA 8 era), ported to CUDA 12.5 + MSVC + sm_86 (RTX 3060 Ti). Five changes in total:

1. `Global.h`: added `typedef unsigned int uint;`; replaced `gettimeofday` timing with `std::chrono::steady_clock`.
2. `main.cu`: removed `sys/time.h`, `unistd.h`, `sys/sysinfo.h` includes.
3. `Framework.cuh`: removed boost/pthread/unistd includes (boost was included but never used); on Windows, `sysconf(_SC_NPROCESSORS_ONLN)` replaced by `omp_get_num_procs()`.
4. `Materialization.cuh`: removed boost includes.
5. Build: `build_win.bat` with nvcc `-arch=sm_86 -Xcompiler /openmp`.

Usage: `GMiner.exe -i <data> -o <output> -s <relative_threshold_0-1> -w 1` (`-w 1` writes all frequent itemsets).

Note: GMiner mines **all** frequent itemsets (FI), not maximal itemsets (MFI); its wall-clock includes a fixed GPU-initialization overhead (framework build ~3.4–4.3 s) before the actual mining step. Both phases are timed separately in the run logs.
