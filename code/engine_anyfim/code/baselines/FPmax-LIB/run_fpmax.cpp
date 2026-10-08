// CLI runner for FPmax* (file-based I/O)
// usage: run_fpmax.exe <input> <output> <minsup_abs>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include "fpmax.h"

int main(int argc, char* argv[])
{
    if (argc < 4) {
        fprintf(stderr, "usage: run_fpmax <input> <output> <minsup_abs>\n");
        return 1;
    }
    setvbuf(stderr, NULL, _IONBF, 0);
    setvbuf(stdout, NULL, _IONBF, 0);
    unsigned int minsup = (unsigned int)strtoul(argv[3], NULL, 10);
    auto t0 = std::chrono::steady_clock::now();
    fpmax(argv[1], argv[2], minsup);
    auto t1 = std::chrono::steady_clock::now();
    double sec = std::chrono::duration<double>(t1 - t0).count();
    printf("ELAPSED %.6f\n", sec);
    return 0;
}
