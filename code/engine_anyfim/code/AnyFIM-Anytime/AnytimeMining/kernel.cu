#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <atlstr.h>
#include <string>
#include <iostream>
#include <fstream>
#include <omp.h>
#include <ppl.h>
#include <windows.h>  
#include <chrono>
#include <sstream>
#include <climits>
#include <atlconv.h> 
#include <vector>
#include <set>
#include <algorithm>
#include <numeric>
//=================================================================================================================




CString transSetFile = _T("..\\TransactionSets\\TCGA_BRCA_transactions.txt"); int UptoStage = 10;//dimension: 60660

// ==== local patch (2026-10-07): allow CLI override of dataset & stage count ====
//   AnytimeMining.exe <transaction_file> [upto_stage]
//   e.g. AnytimeMining.exe E:\data\skab_valve1.txt 24
static void applyArgv(int argc, char** argv) {
    if (argc > 1) { USES_CONVERSION; transSetFile = CA2T(argv[1]); }
    if (argc > 2) { UptoStage = atoi(argv[2]); }
}
// ==== local patch end ====

//CString transSetFile = _T("..\\TransactionSets\\chess.txt"); int UptoStage = 10;//dimension: 75
//CString transSetFile = _T("..\\TransactionSets\\connect.txt"); int UptoStage = 20;//dimension: 129
//CString transSetFile = _T("..\\TransactionSets\\mushroom.txt"); int UptoStage = 50;//dimension: 119
//CString transSetFile = _T("..\\TransactionSets\\T10I4D100K.txt"); int UptoStage = 500;//dimension: 870
//CString transSetFile = _T("..\\TransactionSets\\retail.txt"); int UptoStage = 800;//dimension: 16470

//CString transSetFile = _T("..\\TransactionSets\\accidents.txt"); int UptoStage = 20;//dimension: 468
//CString transSetFile = _T("..\\TransactionSets\\pumsb_star.txt"); int UptoStage = 52;//dimension: 2088
//CString transSetFile = _T("..\\TransactionSets\\pumsb.txt"); int UptoStage = 20;//dimension: 2113
//CString transSetFile = _T("..\\TransactionSets\\T40I10D100K.txt"); int UptoStage = 545;//dimension: 942
//CString transSetFile = _T("..\\TransactionSets\\kosarak.txt"); int UptoStage = 100;//dimension: 41270

//=================================================================================================================
struct FI {
    std::vector<int> itemsSet;
    int frequency;
    int root;
    int progress;
};

std::vector<std::vector<int>> h_transSet;
int* d_transSet;

int dimension;
int transNum; 
int itemIDmax, itemIDmin;
size_t transLengthMax;
size_t transLengthMin; 
double transLengthMean;

int transNumReduced;

size_t transLengthReducedMin;
size_t transLengthReducedMax;
double transLengthReducedMean;

std::vector<int> freqPerItem;
std::vector<int> freqPerItem_inSort;
std::vector<int> items_inFreqSort;

std::vector<FI> FIsStack;
std::mutex mtxFIsStackk;

std::vector<std::vector<FI>> MFIsPool;
std::mutex mtxMFIsPool;

double preprocessing_time;

std::vector<double> runtimePerStage;
std::vector<double> frequencyThrePerStage;
std::vector<double> supportThrePerStage;
std::vector<int> dimensionPerStage;
std::vector<int> MFIsNumPerStage;
std::vector<int> itemsNumPerStage;

std::vector<int> MFIsNumPerStage_withRemoveDuplicate;
std::vector<int> itemsNumPerStage_withRemoveDuplicate;
//=================================================================================================================
int CPUphysicalCores;
int CPUlogicalCores;
double GPUtaskPercentage;

__device__ cudaError_t cudaStatus;
__device__ int* d_TransSetReducedRAM;
//=================================================================================================================
// 根据计算能力获取每个SM的CUDA核心数
static int getCoresPerSM(int major, int minor) {
    switch (major) {
    case 2: return (minor == 0) ? 32 : 48; // Fermi
    case 3: return 192; // Kepler
    case 5: return 128; // Maxwell
    case 6: return 128; // Pascal
    case 7: return 64;  // Volta
    case 8: return 128;  // Ampere
    case 9: return 128; // Ada Lovelace
    case 10: return 192; // Blackwell (示例，需以NVIDIA官方数据为准)
    default: return -1; // 未知架构
    }
}

CString getVSversion() {
    CString vsInfo;
    if (_MSC_VER >= 1940) vsInfo = _T("2022 (v143)");  // VS2022 17.10+ 仍归属v143
    else if (_MSC_VER >= 1930) vsInfo = _T("2022 (v143)");  // VS2022 17.0~17.9
    else if (_MSC_VER >= 1920) vsInfo = _T("2019 (v142)");  // VS2019 16.x
    else if (_MSC_VER >= 1910) vsInfo = _T("2017 (v141)");  // VS2017 15.x
    else if (_MSC_VER == 1900) vsInfo = _T("2015 (v140)");  // VS2015 14.0
    else if (_MSC_VER == 1800) vsInfo = _T("2013 (v120)");  // VS2013 12.0
    else if (_MSC_VER == 1700) vsInfo = _T("2012 (v110)");  // VS2012 11.0
    else if (_MSC_VER == 1600) vsInfo = _T("2010 (v100)");  // VS2010 10.0
    else {
        CString strMSCVer;
        strMSCVer.Format(_T("%d"), _MSC_VER);
        vsInfo = _T("Unknown Visual Studio Version (MSC_VER = ") + strMSCVer + _T(")");
    }
    return vsInfo;
}
//=================================================================================================================
void getCPUInfo(CString& cpuName, int& physicalCores, int& logicalCores) {
    // 1. 初始化参数
    cpuName = _T("Unknown CPU");  // 宽字符兼容，适配Unicode
    physicalCores = 0;
    logicalCores = 0;

    // 2. 获取逻辑核心数
    SYSTEM_INFO sysInfo{};
    GetSystemInfo(&sysInfo);
    logicalCores = static_cast<int>(sysInfo.dwNumberOfProcessors);

    // 3. 获取物理核心数（简化容错）
    DWORD bufferSize = 0;
    GetLogicalProcessorInformation(NULL, &bufferSize);
    if (bufferSize > 0) {
        char* buffer = new (std::nothrow) char[bufferSize];
        if (buffer && GetLogicalProcessorInformation((SYSTEM_LOGICAL_PROCESSOR_INFORMATION*)buffer, &bufferSize)) {
            SYSTEM_LOGICAL_PROCESSOR_INFORMATION* info = (SYSTEM_LOGICAL_PROCESSOR_INFORMATION*)buffer;
            DWORD infoCount = bufferSize / sizeof(SYSTEM_LOGICAL_PROCESSOR_INFORMATION);
            for (DWORD i = 0; i < infoCount; i++) {
                if (info[i].Relationship == RelationProcessorCore) {
                    physicalCores++;
                }
            }
        }
        delete[] buffer;
    }
    // 容错：获取失败则物理核心数=逻辑核心数
    physicalCores = (physicalCores == 0) ? logicalCores : physicalCores;

    // 4. 获取CPU名称（转CString）
    int cpuInfo[4] = { 0 };
    char cpuBrand[0x40] = { 0 };  // 临时存储ASCII格式CPU名称
    __cpuid(cpuInfo, 0x80000000);
    if (cpuInfo[0] >= 0x80000004) {
        // 分3次读取CPU名称
        __cpuid(cpuInfo, 0x80000002); memcpy(cpuBrand + 0, cpuInfo, 16);
        __cpuid(cpuInfo, 0x80000003); memcpy(cpuBrand + 16, cpuInfo, 16);
        __cpuid(cpuInfo, 0x80000004); memcpy(cpuBrand + 32, cpuInfo, 16);

        // 转换为CString（自动处理ASCII→Unicode）
        cpuName = CString(cpuBrand);
        // 修剪前后多余空格（美化）
        cpuName.Trim();
    }
}

static void displayWorkingEnvironment() {
    int deviceCount;
    cudaGetDeviceCount(&deviceCount); // Get and display GPU information
    std::cout << "======================================================================" << std::endl;
    for (int i = 0; i < deviceCount; ++i)
    {
        cudaDeviceProp deviceProp;
        cudaGetDeviceProperties(&deviceProp, i);
        std::cout << "GPU " << i << ": " << deviceProp.name << std::endl;
        std::cout << "  Multi-Processor Count: " << deviceProp.multiProcessorCount << std::endl;
        std::cout << "  Max Threads Per Multi-Processor: " << deviceProp.maxThreadsPerMultiProcessor << std::endl;
        std::cout << "  Warp Size: " << deviceProp.warpSize << std::endl;
        std::cout << "  Max Threads Per Block: " << deviceProp.maxThreadsPerBlock << std::endl;
        std::cout << "  Compute Capability: " << deviceProp.major << "." << deviceProp.minor << std::endl;
        std::cout << "  Number of Multi-Processors: " << deviceProp.multiProcessorCount << std::endl;
        int coresPerSM = getCoresPerSM(deviceProp.major, deviceProp.minor);
        if (coresPerSM == -1) {
            std::cout << "Unknown architecture, unable to calculate the number of CUDA cores !" << std::endl;
        }
        else {
            int totalCores = coresPerSM * deviceProp.multiProcessorCount;
            std::cout << "  Number of CUDA Cores Per SM: " << coresPerSM << std::endl;
            std::cout << "  Total Number of CUDA Cores: " << totalCores << std::endl;
        }
        std::cout << std::endl;
    }

    CString cpuName;
    getCPUInfo(cpuName, CPUphysicalCores, CPUlogicalCores);
    std::cout << "CPU: " << cpuName << std::endl; 
    {
        std::cout << "  Number of Physical Cores: " << CPUphysicalCores << std::endl;
        std::cout << "  Number of Logical Cores: " << CPUlogicalCores << std::endl << std::endl;
    }

    std::cout << "Visual Studio: " << getVSversion() << ", MSC_VER: " << _MSC_VER<< std::endl;
    int rt_ver = 0;
    cudaError_t err = cudaRuntimeGetVersion(&rt_ver);
    if (err == cudaSuccess) std::cout << "CUDA Version: " << rt_ver / 1000 << "." << (rt_ver % 1000) / 10 << std::endl;
    else std::cout << "CUDA Version not detected: " << cudaGetErrorString(err) << std::endl;
}

//=================================================================================================================
static bool readTransSetFile(CString transSetFile) {
    std::ifstream fin(transSetFile); if (!fin) { std::cout << transSetFile << " 文件打开失败 ！" << std::endl; return false; }
    std::string line;
    while (getline(fin, line)) {
        std::istringstream iss(line);
        std::vector<int> row;
        int num;
        while (iss >> num) if (num >= 0) row.push_back(num);
        h_transSet.push_back(row);
    }
    fin.close();

    itemIDmin = INT_MAX;
    itemIDmax = 0;
    transNum = 0;
    transLengthMin = INT_MAX;
    transLengthMax = 0;
    transLengthMean = 0;

    long long len = 0;
    for (auto& row : h_transSet) {
        for (int x : row) {
            if (x > itemIDmax) itemIDmax = x;
            if (x < itemIDmin) itemIDmin = x;
        }
        len = row.size();
        if (len < transLengthMin) transLengthMin = len;
        if (len > transLengthMax) transLengthMax = len;
        transLengthMean += len;
        ++transNum;
    }
    if (transNum > 0) transLengthMean /= transNum;

    dimension = 0;
    std::vector<int> itemList(itemIDmax + 1, 0);//to get dimension
    for (auto& row : h_transSet) {
        for (int x : row) {
            if (itemList[x] == 0) {
                dimension++;
                itemList[x]++;
            }
        }
    }
    return true;
}

static void getItemsFrequenc() {
    freqPerItem.insert(freqPerItem.end(), itemIDmax + 1, 0);
    for (auto& row : h_transSet) {
        for (int x : row) {
            freqPerItem[x]++;
        }
    }
    for (int i = 0; i < freqPerItem.size(); i++)
    {
        freqPerItem_inSort.push_back(freqPerItem[i]);
        items_inFreqSort.push_back(i);
    }
    // 按频度降序排序（stable_sort 与原先冒泡同为稳定排序，等频项保持原次序，语义完全等价，
    // 复杂度从 O(d²) 降到 O(d log d)——高维数据集如 TCGA 上预处理提速显著）
    std::stable_sort(items_inFreqSort.begin(), items_inFreqSort.end(),
        [&](int a, int b) { return freqPerItem[a] > freqPerItem[b]; });
    for (size_t k = 0; k < items_inFreqSort.size(); k++) freqPerItem_inSort[k] = freqPerItem[items_inFreqSort[k]];

    /**/
    std::cout << "======================================================================" << std::endl;
    std::cout << "freqPerItem: ";
    for (int val : freqPerItem) std::cout << val << " ";
    std::cout << std::endl;

    std::cout << std::endl << "freqPerItem_inSort: ";
    for (int val : freqPerItem_inSort) std::cout << val << " ";
    std::cout << std::endl;

    std::cout << std::endl << "items_inFreqSort: ";
    for (int val : items_inFreqSort) std::cout << val << " ";
    std::cout << std::endl;
}

static bool FIsStackkPop(FI* result) { // 安全出栈操作
    std::lock_guard<std::mutex> lock(mtxFIsStackk);
    if (!FIsStack.empty())
    {
        *result = FIsStack.back();
        FIsStack.pop_back();
        return true;
    }
    else return false;
}
static void FIsStackkPush(FI* pushOne) { // 安全入栈操作
    std::lock_guard<std::mutex> lock(mtxFIsStackk);
    FIsStack.push_back(*pushOne);
}

static void MFIsPoolPush(int generation, FI* pushOne) { // 安全入栈操作
    std::lock_guard<std::mutex> lock(mtxMFIsPool);
    MFIsPool[generation].push_back(*pushOne);
}

static bool isOutside(int item, FI* one)
{
    int i;
    for (i = 0; i < one->itemsSet.size(); i++) {
        if (item == one->itemsSet[i]) break;
    }
    if (i < one->itemsSet.size()) return false;
    else return true;
}

//===================== 位图（binary vector）支持度统计：全局结构 =====================
// 每个项一个位向量：第 t 位置 1 表示第 t 条（精减后）事务包含该项。
// 候选 X 的条件库 = X 中各项位向量的 AND；项 e 在其中的频度 = popcount(AND 结果 & bv[e])。
// 与原先逐事务扫描完全等价，但把 O(N×L) 的比较变成 O(N/64) 的位运算。
// 注意：精减按最终阶段（UptoStage）的最低阈值一次性完成，各阶段共享同一份位图，
// 阶段阈值只影响"达到阈值的项数"统计与扩展判断，不影响位图本身。
static std::vector<unsigned long long> h_bitmap;   // (itemIDmax+1) × bitmapWords 的扁平数组
static int bitmapWords = 0;
static std::vector<int> freqItems;                 // 精减后幸存项（freqPerItem >= 最终阶段阈值）的 ID 列表
static int freqItemsNum = 0;
static unsigned long long* d_bitmap = nullptr;     // GPU 端位图（只读，所有工作线程共享）
static int* d_freqItems = nullptr;                 // GPU 端幸存项 ID 列表

static void buildBitmap() {
    bitmapWords = (transNumReduced + 63) / 64;
    h_bitmap.assign((size_t)(itemIDmax + 1) * bitmapWords, 0ULL);
    for (int t = 0; t < transNumReduced; t++) {
        for (int x : h_transSet[t]) {
            if (x >= 0 && x <= itemIDmax)
                h_bitmap[(size_t)x * bitmapWords + (t >> 6)] |= (1ULL << (t & 63));
        }
    }
    // 幸存项判定与 reduceTransSet 的保留条件严格一致：freqPerItem[i] >= 精减阈值
    int finalThreshold = freqPerItem_inSort[UptoStage];
    freqItems.clear();
    for (int i = 1; i <= itemIDmax; i++) {
        if (freqPerItem[i] >= finalThreshold) freqItems.push_back(i);
    }
    freqItemsNum = (int)freqItems.size();
}

static void uploadBitmapToGPU() {
    size_t bitmapBytes = (size_t)(itemIDmax + 1) * bitmapWords * sizeof(unsigned long long);
    if (cudaMalloc(&d_bitmap, bitmapBytes) != cudaSuccess) { std::cerr << "d_bitmap分配失败" << std::endl; return; }
    if (cudaMemcpy(d_bitmap, h_bitmap.data(), bitmapBytes, cudaMemcpyHostToDevice) != cudaSuccess) { std::cerr << "d_bitmap拷贝失败" << std::endl; return; }
    if (cudaMalloc(&d_freqItems, freqItemsNum * sizeof(int)) != cudaSuccess) { std::cerr << "d_freqItems分配失败" << std::endl; return; }
    if (cudaMemcpy(d_freqItems, freqItems.data(), freqItemsNum * sizeof(int), cudaMemcpyHostToDevice) != cudaSuccess) { std::cerr << "d_freqItems拷贝失败" << std::endl; return; }
}

static void getResFrequency(FI newOne, std::vector<int>* resFrequency, int frequencyThreshold) {
    std::fill(resFrequency->begin(), resFrequency->end(), 0);
    const int W = bitmapWords;
    static thread_local std::vector<unsigned long long> mask;  // 每线程复用，避免反复分配
    mask.assign(W, ~0ULL);
    for (int x : newOne.itemsSet) {
        const unsigned long long* bv = h_bitmap.data() + (size_t)x * W;
        for (int w = 0; w < W; w++) mask[w] &= bv[w];
    }
    for (int e : freqItems) {
        const unsigned long long* bv = h_bitmap.data() + (size_t)e * W;
        unsigned long long s = 0;
        for (int w = 0; w < W; w++) s += __popcnt64(mask[w] & bv[w]);
        resFrequency->at(e) = (int)s;
    }
    int cnt = 0;
    for (int e : freqItems) if (resFrequency->at(e) >= frequencyThreshold) cnt++;
    resFrequency->at(itemIDmax + 1) = cnt;
}

// 位图版核函数：每个 block 负责一个幸存项 e，块内线程按字跨步计算
// popcount(bv[e] & bv[X0] & bv[X1] & ...)，warp shuffle + shared memory 两级归约。
__global__ void getResFrequencyCudaKernel(int* d_resFrequencyCompact, int* d_itemsSet, int itemsNum,
    unsigned long long* d_bitmap, int bitmapWords, int* d_freqItems, int freqItemsNum)
{
    int eIdx = blockIdx.x;
    if (eIdx >= freqItemsNum) return;
    int e = d_freqItems[eIdx];
    unsigned long long* bvE = d_bitmap + (size_t)e * bitmapWords;

    unsigned long long local = 0;
    for (int w = threadIdx.x; w < bitmapWords; w += blockDim.x) {
        unsigned long long m = bvE[w];
        for (int i = 0; i < itemsNum; i++)
            m &= d_bitmap[(size_t)d_itemsSet[i] * bitmapWords + w];
        local += __popcll(m);
    }
    // warp 内归约
    for (int offset = 16; offset > 0; offset >>= 1)
        local += __shfl_down_sync(0xffffffffULL, local, offset);
    // warp 间归约
    __shared__ unsigned long long warpSums[32];
    int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    if (lane == 0) warpSums[warp] = local;
    __syncthreads();
    if (warp == 0) {
        int numWarps = (blockDim.x + 31) >> 5;
        unsigned long long v = (lane < numWarps) ? warpSums[lane] : 0ULL;
        for (int offset = 16; offset > 0; offset >>= 1)
            v += __shfl_down_sync(0xffffffffULL, v, offset);
        if (lane == 0) d_resFrequencyCompact[eIdx] = (int)v;
    }
}

// 每个工作线程各自的常驻设备缓冲区与非阻塞流：懒分配一次、复用全程，
// 消除了原先每次调用 cudaMalloc×2/cudaFree×2 + cudaDeviceSynchronize 全局同步的巨额开销。
static thread_local int* tl_d_resFrequency = nullptr;
static thread_local int* tl_d_itemsSet = nullptr;
static thread_local cudaStream_t tl_stream = nullptr;

cudaError_t getResFrequencyCuda(FI newOne, std::vector<int>* resFrequency, int frequencyThreshold) {
    // 前置校验：避免空指针/非法参数
    if (resFrequency == nullptr || resFrequency->empty() || newOne.itemsSet.empty()) {
        std::cerr << "错误：输入参数为空或无效" << std::endl;
        return cudaErrorInvalidValue;
    }
    if (itemIDmax <= 0 || bitmapWords <= 0 || freqItemsNum <= 0 || d_bitmap == nullptr || d_freqItems == nullptr) {
        std::cerr << "错误：位图未初始化或全局变量非法" << std::endl;
        return cudaErrorInvalidValue;
    }

    // 1. 初始化输出vector
    std::fill(resFrequency->begin(), resFrequency->end(), 0);

    // 2. 懒分配本线程常驻缓冲区与非阻塞流
    //    位图版结果采用紧凑布局：只有 freqItemsNum 个频度值，回传量从 (itemIDmax+1) 个 int 降到幸存项数个
    if (tl_d_resFrequency == nullptr) {
        cudaStatus = cudaMalloc(&tl_d_resFrequency, freqItemsNum * sizeof(int));
        if (cudaStatus != cudaSuccess) {
            std::cerr << "分配d_resFrequency失败：" << cudaGetErrorString(cudaStatus) << std::endl;
            tl_d_resFrequency = nullptr;
            return cudaStatus;
        }
    }
    if (tl_d_itemsSet == nullptr) {
        cudaStatus = cudaMalloc(&tl_d_itemsSet, (itemIDmax + 1) * sizeof(int));
        if (cudaStatus != cudaSuccess) {
            std::cerr << "分配d_itemsSet失败：" << cudaGetErrorString(cudaStatus) << std::endl;
            tl_d_itemsSet = nullptr;
            return cudaStatus;
        }
    }
    if (tl_stream == nullptr) {
        cudaStatus = cudaStreamCreateWithFlags(&tl_stream, cudaStreamNonBlocking);
        if (cudaStatus != cudaSuccess) {
            std::cerr << "创建CUDA流失败：" << cudaGetErrorString(cudaStatus) << std::endl;
            tl_stream = nullptr;
            return cudaStatus;
        }
    }

    // 3. 仅拷贝候选项集本身（几十字节）
    int itemsNum = newOne.itemsSet.size();
    cudaStatus = cudaMemcpyAsync(tl_d_itemsSet, newOne.itemsSet.data(),
        itemsNum * sizeof(int), cudaMemcpyHostToDevice, tl_stream);
    if (cudaStatus != cudaSuccess) {
        std::cerr << "拷贝d_itemsSet失败：" << cudaGetErrorString(cudaStatus) << std::endl;
        return cudaStatus;
    }

    // 4. 每个 block 负责一个幸存项，256 线程/块，在本线程私有流上启动
    getResFrequencyCudaKernel << <freqItemsNum, 256, 0, tl_stream >> > (
        tl_d_resFrequency, tl_d_itemsSet, itemsNum, d_bitmap, bitmapWords, d_freqItems, freqItemsNum
        );
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        std::cerr << "核函数启动失败：" << cudaGetErrorString(cudaStatus) << std::endl;
        return cudaStatus;
    }

    // 5. 紧凑结果在同一私有流上拷回，随后只等待本流完成（不再全局 deviceSynchronize）
    static thread_local std::vector<int> h_compact;
    h_compact.resize(freqItemsNum);
    cudaStatus = cudaMemcpyAsync(h_compact.data(), tl_d_resFrequency,
        freqItemsNum * sizeof(int), cudaMemcpyDeviceToHost, tl_stream);
    if (cudaStatus != cudaSuccess) {
        std::cerr << "拷贝结果回主机端失败：" << cudaGetErrorString(cudaStatus) << std::endl;
        return cudaStatus;
    }
    cudaStatus = cudaStreamSynchronize(tl_stream);
    if (cudaStatus != cudaSuccess) {
        std::cerr << "核函数执行失败：" << cudaGetErrorString(cudaStatus) << std::endl;
        return cudaStatus;
    }

    // 6. 散射回 resFrequency（按幸存项 ID），并按本阶段阈值统计达到阈值的项数
    int cnt = 0;
    for (int idx = 0; idx < freqItemsNum; idx++) {
        int v = h_compact[idx];
        resFrequency->at(freqItems[idx]) = v;
        if (v >= frequencyThreshold) cnt++;
    }
    resFrequency->at(itemIDmax + 1) = cnt;

    return cudaSuccess;
}

static void recursiveExpansion(FI newOne, int frequencyThreshold, int UptoDimensionStep) {
    std::vector<int> resFrequency(itemIDmax + 1 + 1, 0);
    //getResFrequency(newOne, &resFrequency, frequencyThreshold);
    getResFrequencyCuda(newOne, &resFrequency, frequencyThreshold);

    if (resFrequency[itemIDmax + 1] == newOne.itemsSet.size()) {
        MFIsPoolPush(UptoDimensionStep, &newOne);
    }
    else
    {
        //for (int i = newOne.progress + 1; i <= itemIDmax; i++)//Big-endian mode 
        for (int i = 0; i <= newOne.progress - 1; i++)//little-endian mode
        {
            if (resFrequency[i] >= frequencyThreshold)
            {
                if (isOutside(i, &newOne)) {
                    FI superOne = newOne;
                    superOne.itemsSet.push_back(i);
                    superOne.frequency = resFrequency[i];
                    superOne.progress = i;
                    recursiveExpansion(superOne, frequencyThreshold, UptoDimensionStep);
                }
            }
        }
    }
}

static void displayTransSetInformation() {
    std::cout << "======================================================================" << std::endl;
    std::cout << "transSetFile: " << transSetFile << std::endl;
    std::cout << "dimension: " << dimension << std::endl;
    std::cout << "transNum: " << transNum << std::endl;
    std::cout << "itemIDmin: " << itemIDmin << std::endl;
    std::cout << "itemIDmax: " << itemIDmax << std::endl;
    std::cout << "transLengthMin: " << transLengthMin << std::endl;
    std::cout << "transLengthMax: " << transLengthMax << std::endl;
    std::cout << "transLengthMean: " << transLengthMean << std::endl;

    std::cout << std::endl << "UptoStage: " << UptoStage << std::endl;
    std::cout << "preprocessing_time(s): " << preprocessing_time << std::endl;
}

static void displayMFIsPool(int UptoDimensionStep) {
    std::cout << "====================== " << UptoDimensionStep << "th stage MFIs Number: " << MFIsPool[UptoDimensionStep].size() << " ======================" << std::endl;
    for (int i = 0; i < MFIsPool[UptoDimensionStep].size(); i++) {
        std::cout << i + 1 << "th: " << MFIsPool[UptoDimensionStep][i].itemsSet.size() << " { ";
        for (int j = 0; j < MFIsPool[UptoDimensionStep][i].itemsSet.size(); j++) {
            std::cout << MFIsPool[UptoDimensionStep][i].itemsSet[j] << " ";
        }
        std::cout << "}   support:" << double(MFIsPool[UptoDimensionStep][i].frequency) / transNum << "  frequency: " << MFIsPool[UptoDimensionStep][i].frequency << std::endl;
    }
    std::cout << std::endl << "frequencyThreshold: " << freqPerItem_inSort[UptoDimensionStep - 1] << std::endl;
}

static void outputWorkingEnvironment2file(std::ofstream& outfile) {
    if (outfile.is_open()) {
        int deviceCount;
        cudaGetDeviceCount(&deviceCount); // Get and display GPU information
        for (int i = 0; i < deviceCount; ++i)
        {
            cudaDeviceProp deviceProp;
            cudaGetDeviceProperties(&deviceProp, i);
            outfile << "GPU " << i << ": " << deviceProp.name << std::endl;
            outfile << "     Multi-Processor Count: " << deviceProp.multiProcessorCount << std::endl;
            outfile << "     Max Threads Per Multi-Processor: " << deviceProp.maxThreadsPerMultiProcessor << std::endl;
            outfile << "     Warp Size: " << deviceProp.warpSize << std::endl;
            outfile << "     Max Threads Per Block: " << deviceProp.maxThreadsPerBlock << std::endl;
            outfile << "     Compute Capability: " << deviceProp.major << "." << deviceProp.minor << std::endl;
            outfile << "     Number of Multi-Processors: " << deviceProp.multiProcessorCount << std::endl;
            int coresPerSM = getCoresPerSM(deviceProp.major, deviceProp.minor);
            if (coresPerSM == -1) {
                outfile << "Unknown architecture, unable to calculate the number of CUDA cores !" << std::endl;
            }
            else {
                int totalCores = coresPerSM * deviceProp.multiProcessorCount;
                outfile << "     Number of CUDA Cores Per SM: " << coresPerSM << std::endl;
                outfile << "     Total Number of CUDA Cores: " << totalCores << std::endl;
            }
            outfile << std::endl;
        }

        CString cpuName;
        getCPUInfo(cpuName, CPUphysicalCores, CPUlogicalCores);
        outfile << "CPU: " << cpuName << std::endl;
        {
            outfile << "     Number of Physical Cores: " << CPUphysicalCores << std::endl;
            outfile << "     Number of Logical Cores: " << CPUlogicalCores << std::endl << std::endl;
        }

        outfile << "Visual Studio: " << getVSversion() << ", MSC_VER: " << _MSC_VER << std::endl;
        int rt_ver = 0;
        cudaError_t err = cudaRuntimeGetVersion(&rt_ver);
        if (err == cudaSuccess) outfile << "CUDA Version: " << rt_ver / 1000 << "." << (rt_ver % 1000) / 10 << std::endl;
        else outfile << "CUDA Version not detected: " << cudaGetErrorString(err) << std::endl;
    }
}

static void output2file(CString resultFile, int dimensionReduced) {
    std::ofstream outfile(resultFile);
    outfile << "=================================================================" << std::endl;
    outputWorkingEnvironment2file(outfile);
    outfile << "=================================================================" << std::endl;
    outfile << "transSetFile: " << transSetFile << std::endl;
    outfile << "dimension: " << dimension << std::endl;
    outfile << "transNum: " << transNum << std::endl;
    outfile << "itemIDmin: " << itemIDmin << std::endl;
    outfile << "itemIDmax: " << itemIDmax << std::endl;
    outfile << "transLengthMin: " << transLengthMin << std::endl;
    outfile << "transLengthMax: " << transLengthMax << std::endl;
    outfile << "transLengthMean: " << transLengthMean << std::endl;

    outfile << std::endl << "UptoStage: " << UptoStage << std::endl;
    outfile << "preprocessing_time(s): " << preprocessing_time << std::endl;
    outfile << "GPUtaskPercentage: " << GPUtaskPercentage << "%" << std::endl;
    outfile << "=================================================================" << std::endl;
    outfile << "runtimePerStage(s):"; for (double val : runtimePerStage) outfile << " " << val << ","; outfile << std::endl;

    outfile << std::endl << "frequencyThrePerStage:"; for (double val : frequencyThrePerStage) outfile << " " << val << ",";
    outfile << std::endl << "supportThrePerStage:"; for (double val : supportThrePerStage) outfile <<" " << val << ","; outfile << std::endl;

    outfile << std::endl << "dimensionPerStage:"; for (int val : dimensionPerStage) outfile << " " << val << ","; outfile << std::endl;

    outfile << std::endl << "MFIsNumPerStage:"; for (int val : MFIsNumPerStage) outfile << " " << val << ",";
    outfile << std::endl << "itemsNumPerStage:"; for (int val : itemsNumPerStage) outfile << " " << val << ","; outfile << std::endl;

    outfile << std::endl << "MFIsNumPerStage_withRemoveDuplicate:"; for (int val : MFIsNumPerStage_withRemoveDuplicate) outfile << " " << val << ",";
    outfile << std::endl << "itemsNumPerStage_withRemoveDuplicate:"; for (int val : itemsNumPerStage_withRemoveDuplicate) outfile << " " << val << ","; outfile << std::endl;
   
    for (int Step = 1; Step <= UptoStage; Step++) {
        outfile << "====================== " << Step << "th stage MFIs Number: " << MFIsPool[Step].size() << " ======================" << std::endl;
        for (int i = 0; i < MFIsPool[Step].size(); i++) {
            outfile << i + 1 << "th: " << MFIsPool[Step][i].itemsSet.size() << " { ";
            for (int j = 0; j < MFIsPool[Step][i].itemsSet.size(); j++) {
                outfile << MFIsPool[Step][i].itemsSet[j] << " ";
            }
            outfile << "}   support: " << double(MFIsPool[Step][i].frequency) / transNum << "  frequency: " << MFIsPool[Step][i].frequency << std::endl;
        }
        outfile << std::endl << "frequencyThreshold: " << freqPerItem_inSort[Step - 1] << std::endl;
    }
    outfile << "=================================================================" << std::endl;
    outfile.close();
}

static void reduceTransSet(int frequencyThreshold) {//精减事务集
    for (auto row = h_transSet.begin(); row != h_transSet.end(); ) {
        row->erase(std::remove_if(row->begin(), row->end(),
            [&](int x) { return (x >= 0 && x < freqPerItem.size()) && (freqPerItem[x] < frequencyThreshold); }),
            row->end());

        if (row->empty())  row = h_transSet.erase(row);// 如果该行变空，则删除整行
        else ++row;
    }
    //for (const auto& r : h_transSet) { for (int val : r) std::cout << val << ' '; std::cout << std::endl; }// 输出结果
}

static void create_d_transSet(int max_transLengthReduced, int transNumReduced)
{
    // 1. 参数合法性校验
    if (max_transLengthReduced <= 0 || transNumReduced <= 0) {
        std::cout << "参数错误：max_transLengthReduced 或 transNumReduced 不能为非正数" << std::endl;
        return;
    }
    cudaError_t cudaStatus = cudaSuccess;
    // 2. 释放旧内存（避免内存泄漏）
    if (d_transSet != nullptr) {
        cudaFree(d_transSet);
        d_transSet = nullptr;
    }
    // 3. 分配设备端连续内存
    size_t totalSize = transNumReduced * (max_transLengthReduced + 1) * sizeof(int);
    cudaStatus = cudaMalloc(&d_transSet, totalSize);
    if (cudaStatus != cudaSuccess) {
        std::cout << "设备端d_transSet连续内存分配失败: " << cudaGetErrorString(cudaStatus) << std::endl;
        return;
    }
    // 4. 分配主机端临时行缓存（统一管理内存）
    int* rowData = new (std::nothrow) int[max_transLengthReduced + 1];
    if (rowData == nullptr) {
        std::cout << "主机端rowData内存分配失败" << std::endl;
        cudaFree(d_transSet);
        d_transSet = nullptr;
        return;
    }
    // 5. 逐行拷贝主机端数据到设备端
    for (int i = 0; i < transNumReduced; ++i) {
        // 校验当前事务长度是否超过最大值（避免越界）
        if (h_transSet[i].size() > (size_t)max_transLengthReduced) {
            std::cout << "第" << i << "行事务长度超过最大值：" << h_transSet[i].size() << " > " << max_transLengthReduced << std::endl;
            cudaFree(d_transSet);
            d_transSet = nullptr;
            delete[] rowData;
            rowData = nullptr;
            return;
        }
        // 填充行数据：第0位存长度，后续存事务内容
        rowData[0] = static_cast<int>(h_transSet[i].size());
        for (size_t j = 0; j < h_transSet[i].size(); j++) {
            rowData[j + 1] = h_transSet[i][j];
        }
        // 计算设备端当前行起始地址
        int* d_row_start = d_transSet + i * (max_transLengthReduced + 1);
        size_t rowSize = (h_transSet[i].size() + 1) * sizeof(int);
        // 拷贝当前行数据
        cudaStatus = cudaMemcpy(d_row_start, rowData, rowSize, cudaMemcpyHostToDevice);
        if (cudaStatus != cudaSuccess) {
            std::cout << "第" << i << "行设备事务集数据拷贝失败: " << cudaGetErrorString(cudaStatus) << std::endl;
            // 异常时释放所有已分配资源
            cudaFree(d_transSet);
            d_transSet = nullptr;
            delete[] rowData;
            rowData = nullptr;
            return;
        }
    }
    // 6. 释放主机端临时内存
    delete[] rowData;
    rowData = nullptr;
    //std::cout << std::endl << "The transaction set on the device side has been created successfully !" << std::endl;
}

static double getGPUtaskPercentage(int parallelNum)
{
    std::vector<double> CPU_test_duration(parallelNum + 1, 0.0);
    std::vector<double> GPU_test_duration(parallelNum + 1, 0.0);
    int frequencyThreshold = freqPerItem_inSort[int(UptoStage / 2)];

    Concurrency::parallel_for
    (1, parallelNum + 1, [&](int parallelID)
        {
            FI newOne;
            newOne.itemsSet.push_back(items_inFreqSort[0]);
            newOne.frequency = freqPerItem_inSort[0];
            newOne.root = items_inFreqSort[0];
            newOne.progress = items_inFreqSort[0];

            std::vector<int> resFrequency(itemIDmax + 1 + 1, 0);
            int times = 10;

            auto t3 = std::chrono::high_resolution_clock::now();
            for (int i = 0; i < times; i++)  getResFrequencyCuda(newOne, &resFrequency, frequencyThreshold);
            auto t4 = std::chrono::high_resolution_clock::now();
            float GPU_time = std::chrono::duration<float, std::milli>(t4 - t3).count();

            auto t1 = std::chrono::high_resolution_clock::now();
            for (int i = 0; i < times; i++)  getResFrequency(newOne, &resFrequency, frequencyThreshold);
            auto t2 = std::chrono::high_resolution_clock::now();
            float CPU_time = std::chrono::duration<float, std::milli>(t2 - t1).count();

            CPU_test_duration[parallelID] = CPU_time;
            GPU_test_duration[parallelID] = GPU_time;
        }
    );

    for (int i = 1; i < parallelNum + 1; i++)
    {
        CPU_test_duration[0] += CPU_test_duration[i];
        GPU_test_duration[0] += GPU_test_duration[i];
    }

    double GPUpercentage = 100 * CPU_test_duration[0] / (CPU_test_duration[0] + GPU_test_duration[0]);
    std::cout << std::endl << "CPU_test_duration: " << CPU_test_duration[0] << "; GPU_test_duration: " << GPU_test_duration[0] << "; GPUpercentage: " << GPUpercentage << std::endl << std::endl;
    return GPUpercentage;
}

void removeDuplicateMFIsGlobal(std::vector<std::vector<FI>>& MFIsPool) {
    std::set<std::vector<int>> seenSets;

    for (auto& fiVec : MFIsPool) {
        size_t writeIdx = 0;
        for (const auto& fi : fiVec) {
            if (seenSets.count(fi.itemsSet) == 0) {
                seenSets.insert(fi.itemsSet);
                fiVec[writeIdx++] = fi;
            }
        }
        fiVec.resize(writeIdx);
    }
}

//=====================================================================================================================================================================================================================================================
int main(int argc, char** argv)
{
    applyArgv(argc, argv);
    displayWorkingEnvironment();
    //=================================================================================================================
    if (!readTransSetFile(transSetFile)) return 1;
    if (UptoStage > dimension) {
        std::cout << "Up to the stage number being greater than the dimension !" << std::endl;
        return 0;
    }
    //=================================================================================================================
    //=================================================================================================================
    getItemsFrequenc();
    //=================================================================================================================
    int CPU_parallel_num = CPUlogicalCores;

    clock_t tic, toc;
    tic = clock();
    reduceTransSet(freqPerItem_inSort[UptoStage]);
    toc = clock();
    preprocessing_time = (double)(toc - tic) / CLOCKS_PER_SEC;
    transNumReduced = h_transSet.size();
   
    transLengthReducedMin = INT_MAX;
    transLengthReducedMax = 0;
    transLengthReducedMean = 0;
    for (const auto& row : h_transSet) {//获取最小最大长度
        if (row.size() < transLengthReducedMin) transLengthReducedMin = row.size();
        if (row.size() > transLengthReducedMax) transLengthReducedMax = row.size();
        transLengthReducedMean += row.size();
    }
    transLengthReducedMean /= transNumReduced;
    //=================================================================================================================
    // 构建位图（binary vector mapping）：每个项一个位向量，计时并入 preprocessing_time
    tic = clock();
    buildBitmap();
    toc = clock();
    preprocessing_time += (double)(toc - tic) / CLOCKS_PER_SEC;
    //=================================================================================================================
    displayTransSetInformation();
    //=================================================================================================================
    uploadBitmapToGPU();// 位图版：GPU 侧只需位图与幸存项列表，不再逐行传输事务集
    //=================================================================================================================
    MFIsPool.resize(UptoStage + 1);
    GPUtaskPercentage = getGPUtaskPercentage(CPU_parallel_num);
    //=================================================================================================================
    for (int Step = 1; Step <= UptoStage; Step++) {//正文
        auto stage_t0 = std::chrono::high_resolution_clock::now();// 毫秒级以下耗时需高精度计时
        int frequencyThreshold = freqPerItem_inSort[Step - 1]; 
        frequencyThrePerStage.push_back(double(frequencyThreshold));
        supportThrePerStage.push_back(double(frequencyThreshold) / transNum);

        for (int i = 0; i < Step; i++) {
            FI newOne;
            newOne.itemsSet.push_back(items_inFreqSort[i]);
            newOne.frequency = freqPerItem_inSort[i];
            newOne.root = items_inFreqSort[i];
            newOne.progress = items_inFreqSort[i];

            //recursiveExpansion(newOne, frequencyThreshold, UptoDimensionStep);
            FIsStackkPush(&newOne);
        }

        /**/
        Concurrency::parallel_for
        (1, CPU_parallel_num + 1, [&](int parallelID)
            {
                FI popOne;
                std::vector<int> resFrequency(itemIDmax + 1 + 1, 0);

                while (true)
                {
                    if (!FIsStackkPop(&popOne)) break;
                    else
                    {
                        if ((rand() % 10000 + 1) <= GPUtaskPercentage * 100) getResFrequencyCuda(popOne, &resFrequency, frequencyThreshold); else getResFrequency(popOne, &resFrequency, frequencyThreshold);
                        //getResFrequencyCuda(popOne, &resFrequency, frequencyThreshold);

                        if (resFrequency[itemIDmax + 1] == popOne.itemsSet.size()) {
                            MFIsPoolPush(Step, &popOne);
                        }
                        else
                        {
                            //for (int i = popOne.progress + 1; i <= itemIDmax; i++)//Big-endian mode 
                            for (int i = itemIDmin; i <= popOne.progress - 1; i++)//little-endian mode
                            {              
                                if (resFrequency[i] >= frequencyThreshold)                                
                                {
                                    if (isOutside(i, &popOne)) {
                                        FI superOne = popOne;
                                        superOne.itemsSet.push_back(i);
                                        superOne.frequency = resFrequency[i];
                                        superOne.progress = i;
                                        FIsStackkPush(&superOne);
                                    }
                                }
                            }
                        }
                    }
                }
            }
        );
        
        auto stage_t1 = std::chrono::high_resolution_clock::now();
        runtimePerStage.push_back(std::chrono::duration<double>(stage_t1 - stage_t0).count());
        dimensionPerStage.push_back(Step);

        //displayMFIsPool(Step);
        std::cout << Step << "th stage" << " MFIsNum: " << MFIsPool[Step].size() << ", runtime: " << runtimePerStage.back() << std::endl;
    }

    for (int i = 1; i <= UptoStage; i++) {//sort by frequency
        auto& vec = MFIsPool[i];
        std::sort(vec.begin(), vec.end(),
            [](const auto& a, const auto& b) {
                return a.frequency > b.frequency; // 降序：frequency大的在前
            });
    }

    for (int i = 1; i <= UptoStage; i++) {//get MFIsNumPerStage and itemsNumPerStage
        MFIsNumPerStage.push_back(MFIsPool[i].size());

        int itemsNum = 0;
        for (int j = 0; j < MFIsPool[i].size(); j++) itemsNum += MFIsPool[i][j].itemsSet.size();
        itemsNumPerStage.push_back(itemsNum);
    }
    //removeDuplicateMFIsGlobal(MFIsPool);//MFIsPool全局去重
    for (int i = 1; i <= UptoStage; i++) {//get MFIsNumPerStage and itemsNumPerStage
        MFIsNumPerStage_withRemoveDuplicate.push_back(MFIsPool[i].size());

        int itemsNum = 0;
        for (int j = 0; j < MFIsPool[i].size(); j++) itemsNum += MFIsPool[i][j].itemsSet.size();
        itemsNumPerStage_withRemoveDuplicate.push_back(itemsNum);
    }
    std::cout << "======================================================================" << std::endl;
    std::cout << "runtimePerStage(s):"; for (double val : runtimePerStage) std::cout << " " << val << ","; std::cout << std::endl;

    std::cout << std::endl << "frequencyThrePerStage:"; for (double val : frequencyThrePerStage) std::cout << " " << val << ",";
    std::cout << std::endl << "supportThrePerStage:"; for (double val : supportThrePerStage) std::cout << " " << val << ","; std::cout << std::endl;

    std::cout << std::endl << "dimensionPerStage:"; for (int val : dimensionPerStage) std::cout << " " << val << ","; std::cout << std::endl;

    std::cout << std::endl << "MFIsNumPerStage:"; for (int val : MFIsNumPerStage) std::cout << " " << val << ",";
    std::cout << std::endl << "itemsNumPerStage:"; for (int val : itemsNumPerStage) std::cout << " " << val << ","; std::cout << std::endl;

    std::cout << std::endl << "MFIsNumPerStage_withRemoveDuplicate:"; for (int val : MFIsNumPerStage_withRemoveDuplicate) std::cout << " " << val << ",";
    std::cout << std::endl << "itemsNumPerStage_withRemoveDuplicate:"; for (int val : itemsNumPerStage_withRemoveDuplicate) std::cout << " " << val << ","; std::cout << std::endl;

    CString resultFile;//生成结果文件 
    resultFile.Format(_T("%s-%d-stages=Results.txt"), transSetFile, UptoStage);
    output2file(resultFile, UptoStage);
    std::cout << "======================================================================" << std::endl;
    return 1;
}