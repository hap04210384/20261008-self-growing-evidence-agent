// MFI_mining.cpp : 此文件包含 "main" 函数。程序执行将在此处开始并结束。
//
// 运行程序: Ctrl + F5 或调试 >“开始执行(不调试)”菜单
// 调试程序: F5 或调试 >“开始调试”菜单

// 入门使用技巧: 
//   1. 使用解决方案资源管理器窗口添加/管理文件
//   2. 使用团队资源管理器窗口连接到源代码管理
//   3. 使用输出窗口查看生成输出和其他消息
//   4. 使用错误列表窗口查看错误
//   5. 转到“项目”>“添加新项”以创建新的代码文件，或转到“项目”>“添加现有项”以将现有代码文件添加到项目
//   6. 将来，若要再次打开此项目，请转到“文件”>“打开”>“项目”并选择 .sln 文件

//#include <cuda_runtime.h>
#include <atlstr.h>
#include <string>
#include <iostream>
#include <fstream>
#include <omp.h>
using namespace std;
const int MaxPossibleTransLength = 100000;

struct MFI
{
    int* chromosome;
    int number;
    double support;
    int progress;
    double times;
    int* ItemSets;
    string* ItemSetsStr;
    int root;
};


MFI** MaximalPopulationMat;
int MaximalPopulationSize = 0;
MFI* MaximalPopulation;


int MaxItemID, MinItemID, TransLengthMax, TransLengthMin; double TransLengthMean;
int** TransSetReducedRAM; double SupportThreshold; int TransSetDimensionReduced; int TransNum; int TransNumReduced;
//=====================================================================================================================================================
void GetTransSetSize(CString TransSet)
{
    MaxItemID = 0;
    MinItemID = INT_MAX;
    TransNum = 0;
    TransLengthMax = 0;
    TransLengthMin = INT_MAX;
    TransLengthMean = 0;

    ifstream infile(TransSet);
    string line_string;
    int* transaction = new int[MaxPossibleTransLength];
    while (std::getline(infile, line_string))
    {
        transaction[0] = 0;
        bool is_new = true;
        for (int i = 0; line_string[i] != '\0'; i++)
        {
            if (line_string[i] != ' ')
            {
                if (is_new)
                {
                    transaction[0]++;
                    transaction[transaction[0]] = int(line_string[i]) - 48;
                    is_new = false;
                }
                else
                {
                    transaction[transaction[0]] = transaction[transaction[0]] * 10 + static_cast<int>(line_string[i] - 48);
                }
            }
            else
            {
                is_new = true;
            }
        }

        //cout << TransactionsNum + 1 << "th: " << transaction[0] << ": ";
        for (int i = 1; i <= transaction[0]; i++)
        {
            if (transaction[i] < MinItemID) MinItemID = transaction[i];
            if (transaction[i] > MaxItemID) MaxItemID = transaction[i];

            //cout << transaction[i] << " ";
        }
        //cout << endl;

        if (transaction[0] < TransLengthMin) TransLengthMin = transaction[0];
        if (transaction[0] > TransLengthMax) TransLengthMax = transaction[0];
        TransLengthMean = TransLengthMean + transaction[0];
        TransNum++;
    }
    infile.close();

    if (MinItemID == 0) MaxItemID = MaxItemID + 1;
    if (TransNum > 0) TransLengthMean = TransLengthMean / TransNum;
    cout << "TransNum:" << TransNum << endl;
    delete[] transaction;
}
//=====================================================================================================================================================
void ExpandingOperation(MFI NewOne)
{
    double* SupportItem = new double[TransSetDimensionReduced + 1];
    double* SupportItem_temp = new double[TransSetDimensionReduced + 1];

    for (int i = 1; i <= TransSetDimensionReduced; i++) SupportItem[i] = 0;

    for (int TransID = 0; TransID < TransNumReduced; TransID++)
    {
        int find = 0;

        for (int i = 1; i <= TransSetDimensionReduced; i++) SupportItem_temp[i] = 0;

        for (int i = 1; i <= TransSetReducedRAM[TransID][0]; i++)
        {
            if (NewOne.chromosome[TransSetReducedRAM[TransID][i]] > 0) find++;
            SupportItem_temp[TransSetReducedRAM[TransID][i]] = 1;
        }
        if (find == NewOne.number)
        {
            for (int i = 1; i <= TransSetDimensionReduced; i++) if (SupportItem_temp[i] > 0) SupportItem[i]++;
        }
    }

    int FrequentNum = 0;
    for (int i = 1; i <= TransSetDimensionReduced; i++)
    {
        SupportItem[i] = SupportItem[i] / TransNum;
        if (SupportItem[i] >= SupportThreshold) FrequentNum++;
    }

    if (FrequentNum == NewOne.number)
    {
        MaximalPopulationMat[NewOne.root][0].number++;
        MaximalPopulationMat[NewOne.root][MaximalPopulationMat[NewOne.root][0].number] = NewOne;

        cout << NewOne.root << " -> [ ";
        for (int i = 1; i <= TransSetDimensionReduced; i++) cout << NewOne.chromosome[i] << " ";
        cout << "]" << endl;
    }
    else
    {
        for (int i = NewOne.progress + 1; i <= TransSetDimensionReduced; i++)
        {
            if ((SupportItem[i] >= SupportThreshold) && (NewOne.chromosome[i] == 0))
            {
                MFI SuperOne = NewOne;
                SuperOne.chromosome = new int[TransSetDimensionReduced + 1];
                for (int j = 0; j <= TransSetDimensionReduced; j++) SuperOne.chromosome[j] = NewOne.chromosome[j];
                SuperOne.chromosome[i] = 1;
                SuperOne.number = SuperOne.number + 1;
                SuperOne.support = SupportItem[i];
                SuperOne.progress = i;

                ExpandingOperation(SuperOne);
            }
        }
        delete[] NewOne.chromosome;
    }

    delete[] SupportItem_temp;
    delete[] SupportItem;
}
//=====================================================================================================================================================
int main()
{
    clock_t tic, toc;
    tic = clock();
    double CPUTimePre;
    double CPUTime;

    CString TransSet = _T("..\\chess.txt"); SupportThreshold = 0.95;
    //CString TransSet = _T("..\\connect.txt"); SupportThreshold = 0.965;
    //CString TransSet = _T("..\\mushroom.txt"); SupportThreshold = 0.52;
    //CString TransSet = _T("..\\T10I4D100K.txt"); SupportThreshold = 0.0225;
    //CString TransSet = _T("..\\retail.txt"); SupportThreshold = 0.00125;

    //CString TransSet = _T("..\\accidents.txt"); SupportThreshold = 0.5;
    //CString TransSet = _T("..\\pumsb_star.txt"); SupportThreshold = 0.275;
    //CString TransSet = _T("..\\pumsb.txt"); SupportThreshold = 0.8;
    //CString TransSet = _T("..\\T40I10D100K.txt"); SupportThreshold = 0.025;
    //CString TransSet = _T("..\\kosarak.txt"); SupportThreshold = 0.0024;

    CString ItemStrIndex = _T("..\\ItemIndex.txt");
    //=====================================================================================================================================================
    CString OutputResult;//生成结果文件
    OutputResult.Format(_T("%s-%f=Result.txt"), TransSet, SupportThreshold);
    //=====================================================================================================================================================
    GetTransSetSize(TransSet);//以扫描的方式获取事务集轮廓
    //=====================================================================================================================================================
    double* Support1D = new double[MaxItemID + 1];//扫描事务集生成1维支持度
    for (int i = 0; i <= MaxItemID; i++) Support1D[i] = 0;

    ifstream infile(TransSet);
    string line_string;
    int* transaction = new int[TransLengthMax + 1];
    while (getline(infile, line_string))
    {
        transaction[0] = 0;
        bool is_new = true;
        for (int i = 0; line_string[i] != '\0'; i++)
        {
            if (line_string[i] != ' ')
            {
                if (is_new)
                {
                    transaction[0]++;
                    transaction[transaction[0]] = int(line_string[i]) - 48;
                    is_new = false;
                }
                else
                {
                    transaction[transaction[0]] = transaction[transaction[0]] * 10 + static_cast<int>(line_string[i] - 48);
                }
            }
            else
            {
                is_new = true;
            }
        }

        for (int i = 1; i <= transaction[0]; i++)
        {
            if (MinItemID == 0) transaction[i]++;
            bool IsRepeat = false;
            for (int j = 1; j <= i - 1; j++) if (transaction[i] == transaction[j]) IsRepeat = true;
            if (!IsRepeat) Support1D[transaction[i]]++;
        }
    }
    infile.close();

    cout << "Support1D: ";
    for (int i = 0; i <= MaxItemID; i++) cout << Support1D[i] << " ";
    cout << endl;
    //=====================================================================================================================================================
    TransSetDimensionReduced = 0;//获取Support1DReduced长度，支持度归一化，
    for (int i = 0; i <= MaxItemID; i++)
    {
        Support1D[i] = Support1D[i] / TransNum;
        if (Support1D[i] >= SupportThreshold) TransSetDimensionReduced++;
    }
    if (TransSetDimensionReduced == 0)
    {
        cout << endl << "TransSetDimensionReduced == 0, no result!" << endl;
        return 0;
    }
    cout << "TransSetDimensionReduced: " << TransSetDimensionReduced;
    //=====================================================================================================================================================
    double* Support1DReduced = new double[TransSetDimensionReduced + 1];//生成精简1项集 
    int* Support1DReducedItemIndex = new int[TransSetDimensionReduced + 1];
    for (int i = 0; i <= TransSetDimensionReduced; i++)
    {
        Support1DReduced[i] = 0;
        Support1DReducedItemIndex[i] = 0;
    }
    double* Support1DReducedItemIndexReverse = new double[MaxItemID + 1];
    for (int i = 0; i <= MaxItemID; i++) Support1DReducedItemIndexReverse[i] = 0;
    int j = 0;
    double ItemSupportMin = 1;
    double ItemSupportMax = 0;
    int TransSetDimension = 0;
    for (int i = 1; i <= MaxItemID; i++)
    {
        if (Support1D[i] >= SupportThreshold)
        {
            j++;
            Support1DReduced[j] = Support1D[i];
            Support1DReducedItemIndex[j] = i;
            Support1DReducedItemIndexReverse[i] = j;
        }
        if (Support1D[i] < ItemSupportMin && Support1D[i] > 0) ItemSupportMin = Support1D[i];
        if (Support1D[i] > ItemSupportMax) ItemSupportMax = Support1D[i];
        if (Support1D[i] > 0) TransSetDimension++;
    }
    cout << endl << "Support1DReduced: ";
    for (int i = 0; i <= TransSetDimensionReduced; i++) cout << Support1DReduced[i] << " ";
    cout << endl << "Support1DReducedItemIndex: ";
    for (int i = 0; i <= TransSetDimensionReduced; i++) cout << Support1DReducedItemIndex[i] << " ";
    cout << endl << "Support1DReducedItemIndexReverse: ";
    for (int i = 0; i <= MaxItemID; i++) cout << Support1DReducedItemIndexReverse[i] << " ";
    //=====================================================================================================================================================
    CString TransSetReduced = _T("..\\TransSetReduced.txt");//扫描事务集生成精简事务集文件，使用新ItemID
    ofstream outfile(TransSetReduced);
    ifstream infile2(TransSet);
    TransNumReduced = 0;
    int TransLengthMaxReduced = 0;
    int TransLengthMinReduced = INT_MAX;
    double TransLengthMeanReduced = 0;

    while (getline(infile2, line_string))
    {
        transaction[0] = 0;
        bool is_new = true;
        for (int i = 0; line_string[i] != '\0'; i++)
        {
            if (line_string[i] != ' ')
            {
                if (is_new)
                {
                    transaction[0]++;
                    transaction[transaction[0]] = int(line_string[i]) - 48;
                    is_new = false;
                }
                else
                {
                    transaction[transaction[0]] = transaction[transaction[0]] * 10 + static_cast<int>(line_string[i] - 48);
                }
            }
            else
            {
                is_new = true;
            }
        }
        int WriteTimes = 0;
        for (int i = 1; i <= transaction[0]; i++)
        {
            if (MinItemID == 0) transaction[i]++;
            int IsRepeat = 0;
            for (int j = 1; j <= i - 1; j++)
            {
                if (transaction[i] == transaction[j]) IsRepeat = 1;
            }
            if ((Support1D[transaction[i]] >= SupportThreshold) && (IsRepeat == 0))
            {
                outfile << Support1DReducedItemIndexReverse[transaction[i]];
                if (i < transaction[0]) outfile << " ";
                WriteTimes = WriteTimes + 1;
                TransLengthMeanReduced = TransLengthMeanReduced + 1;
            }
        }
        if (WriteTimes > 0)
        {
            outfile << endl;
            TransNumReduced = TransNumReduced + 1;
        }
        if ((WriteTimes < TransLengthMinReduced) && (WriteTimes > 0)) TransLengthMinReduced = WriteTimes;
        if (WriteTimes > TransLengthMaxReduced) TransLengthMaxReduced = WriteTimes;
    }
    TransLengthMeanReduced = TransLengthMeanReduced / TransNumReduced;
    double TransSetSizeReductionRate = TransNumReduced * TransLengthMeanReduced / TransNum / TransLengthMean;
    double DimensionReductionRate = double(TransSetDimensionReduced) / TransSetDimension;

    infile2.close();
    outfile.close();
    //=====================================================================================================================================================
    TransSetReducedRAM = new int* [TransNumReduced + 1];//扫描精简事务集生成内存矩阵，
    ifstream infile3(TransSetReduced);
    cout << endl << endl;
    int TransSetTh = 0;
    while (getline(infile3, line_string))
    {
        transaction[0] = 0;
        bool is_new = true;
        for (int i = 0; line_string[i] != '\0'; i++)
        {
            if (line_string[i] != ' ')
            {
                if (is_new)
                {
                    transaction[0]++;
                    transaction[transaction[0]] = int(line_string[i]) - 48;
                    is_new = false;
                }
                else
                {
                    transaction[transaction[0]] = transaction[transaction[0]] * 10 + static_cast<int>(line_string[i] - 48);
                }
            }
            else
            {
                is_new = true;
            }
        }

        TransSetReducedRAM[TransSetTh] = new int[TransSetDimensionReduced + 1];
        //cout << TransSetTh + 1 << "th: " << transaction[0] << ": ";
        for (int i = 0; i <= transaction[0]; i++)
        {
            TransSetReducedRAM[TransSetTh][i] = transaction[i];
            //cout << TransSetReducedRAM[TransSetTh][i]<<" ";
        }
        //cout << endl;
        TransSetTh++;
    }
    infile3.close();

    //cout << endl << endl;
    for (int i = 0; i < TransNumReduced; i++)
    {
        //cout << i + 1 << "th: " << TransSetReducedRAM[i][0] << ": ";
        for (int j = 1; j <= TransSetReducedRAM[i][0]; j++)
        {
            //cout << TransSetReducedRAM[i][j] << " ";
        }
        //cout << endl;
    }
    cout << endl;
    cout << "TransNumReduced: " << TransNumReduced << endl;
    cout << "TransSetDimensionReduced: " << TransSetDimensionReduced << endl;
    cout << "TransLengthMaxReduced: " << TransLengthMaxReduced << endl;
    cout << "TransLengthMinReduced: " << TransLengthMinReduced << endl;
    cout << "TransLengthMeanReduced: " << TransLengthMeanReduced << endl << endl;

    delete[] transaction;
    //=====================================================================================================================================================
    toc = clock();
    CPUTimePre = (double)(toc - tic) / CLOCKS_PER_SEC;
    MaximalPopulation = new MFI[TransNumReduced + 1];
    MaximalPopulationMat = new MFI * [TransSetDimensionReduced + 1];

    cout << "MFIs:" << endl;
    tic = clock();
    //#  pragma omp parallel for
    for (int i = 1; i <= TransSetDimensionReduced; i++)//正文
    {
        //cout << i <<"->" << endl;
        MFI NewOne;
        NewOne.chromosome = new int[TransSetDimensionReduced + 1];
        for (int j = 1; j <= TransSetDimensionReduced; j++) NewOne.chromosome[j] = 0;
        NewOne.chromosome[i] = 1;
        NewOne.number = 1;
        NewOne.support = Support1DReduced[i];
        NewOne.progress = i;
        NewOne.root = i;

        MaximalPopulationMat[i] = new MFI[TransNumReduced + 1];
        MaximalPopulationMat[i][0].number = 0;

        ExpandingOperation(NewOne);
    }
    toc = clock();
    CPUTime = (double)(toc - tic) / CLOCKS_PER_SEC;


    for (int i = 1; i <= TransSetDimensionReduced; i++)//二维MFI整理到一维
    {
        if (MaximalPopulationMat[i][0].number == 0) continue;
        for (int j = 1; j <= MaximalPopulationMat[i][0].number; j++)
        {
            MaximalPopulation[MaximalPopulationSize] = MaximalPopulationMat[i][j];
            MaximalPopulationSize++;
        }
    }
    //=====================================================================================================================================================
    int* ItemID = new int[MaxItemID + 1]; //解码
    string* ItemCode = new string[MaxItemID + 1];
    string* ItemStr = new string[MaxItemID + 1];
    ifstream infile4(ItemStrIndex);
    for (int i = 1; i <= MaxItemID; i++)
    {
        infile4 >> ItemID[i];
        infile4 >> ItemCode[i];
        infile4 >> ItemStr[i];
    }
    infile4.close();
    for (int i = 0; i < MaximalPopulationSize; i++)
    {
        int TransLength = 0;
        MaximalPopulation[i].ItemSets = new int[TransSetDimensionReduced + 1];
        MaximalPopulation[i].ItemSetsStr = new string[TransSetDimensionReduced + 1];
        for (int j = 1; j <= TransSetDimensionReduced; j++)
        {
            if (MaximalPopulation[i].chromosome[j] != 0)
            {
                TransLength++;

                MaximalPopulation[i].ItemSets[TransLength] = Support1DReducedItemIndex[j];
                MaximalPopulation[i].ItemSetsStr[TransLength] = ItemStr[Support1DReducedItemIndex[j]];
            }
        }
        MaximalPopulation[i].times = int(MaximalPopulation[i].support * TransNum + 0.5);
    }
    for (int i = 0; i < MaximalPopulationSize - 1; i++) //sorting by support
    {
        for (int j = 0; j < MaximalPopulationSize - i; j++)
        {
            if (MaximalPopulation[j].support < MaximalPopulation[j + 1].support)
            {
                MFI ExOne = MaximalPopulation[j];
                MaximalPopulation[j] = MaximalPopulation[j + 1];
                MaximalPopulation[j + 1] = ExOne;
            }
        }
    }

    //=====================================================================================================================================================
    ofstream outfile2(OutputResult); //生成结果文件

    outfile2 << "TransSet: " << CStringA(TransSet) << endl;
    outfile2 << "ItemStrIndex: " << CStringA(ItemStrIndex) << endl << endl;

    outfile2 << "MinItemID: " << MinItemID << endl;
    outfile2 << "MaxItemID: " << MaxItemID << endl;
    outfile2 << "TransNum: " << TransNum << endl;
    outfile2 << "TransSetDimension: " << TransSetDimension << endl;
    outfile2 << "TransLengthMin: " << TransLengthMin << endl;
    outfile2 << "TransLengthMax: " << TransLengthMax << endl;
    outfile2 << "TransLengthMean: " << TransLengthMean << endl;
    outfile2 << "MinItemSupport: " << ItemSupportMin << endl;
    outfile2 << "MaxItemSupport: " << ItemSupportMax << endl;
    outfile2 << endl;

    outfile2 << "SupportThreshold: " << SupportThreshold << endl;

    outfile2 << endl;
    outfile2 << "TransNumReduced: " << TransNumReduced << endl;
    outfile2 << "TransSetDimensionReduced: " << TransSetDimensionReduced << endl;
    outfile2 << "TransLengthMinReduced: " << TransLengthMinReduced << endl;
    outfile2 << "TransLengthMaxReduced: " << TransLengthMaxReduced << endl;
    outfile2 << "TransLengthMeanReduced: " << TransLengthMeanReduced << endl;
    outfile2 << "TransSetSizeReductionRate: " << TransSetSizeReductionRate << endl;
    outfile2 << "DimensionReductionRate: " << DimensionReductionRate << endl;
    outfile2 << "TransSetReduced: " << CStringA(TransSetReduced) << endl;

    outfile2 << endl;
    outfile2 << "MaximalPopulationSize: " << MaximalPopulationSize << endl;
    outfile2 << "OutputResult: " << CStringA(OutputResult) << endl;
    outfile2 << "CPUTimePre: " << CPUTimePre << "s" << endl;
    outfile2 << "CPUTime: " << CPUTime << "s" << endl;
    outfile2 << endl;
    outfile2 << endl;


    for (int i = 0; i < MaximalPopulationSize; i++)
    {
        outfile2 << i + 1 << ": " << MaximalPopulation[i].number << " { ";
        for (int j = 1; j <= MaximalPopulation[i].number; j++)
        {
            if (MinItemID == 0) MaximalPopulation[i].ItemSets[j] = MaximalPopulation[i].ItemSets[j] - 1;
            if (j < MaximalPopulation[i].number) outfile2 << MaximalPopulation[i].ItemSets[j] << " ";
            else outfile2 << MaximalPopulation[i].ItemSets[j];
        }
        outfile2 << " }     support: " << MaximalPopulation[i].support << "     times: " << MaximalPopulation[i].times << "     { ";
        for (int j = 1; j <= MaximalPopulation[i].number; j++)
        {
            if (j < MaximalPopulation[i].number) outfile2 << MaximalPopulation[i].ItemSetsStr[j] << ",   ";
            else outfile2 << MaximalPopulation[i].ItemSetsStr[j];

        }
        outfile2 << " }" << endl;
    }
    outfile2 << endl;

    outfile2.close();
    //=====================================================================================================================================================
    cout << endl << "//======以结果文件格式打印输出======" << endl << endl;
    cout << "TransSet: " << CStringA(TransSet) << endl;
    cout << "ItemStrIndex: " << CStringA(ItemStrIndex) << endl << endl;

    cout << "MinItemID: " << MinItemID << endl;
    cout << "MaxItemID: " << MaxItemID << endl;
    cout << "TransNum: " << TransNum << endl;
    cout << "TransSetDimension: " << TransSetDimension << endl;
    cout << "TransLengthMin: " << TransLengthMin << endl;
    cout << "TransLengthMax: " << TransLengthMax << endl;
    cout << "TransLengthMean: " << TransLengthMean << endl;
    cout << "MinItemSupport: " << ItemSupportMin << endl;
    cout << "MaxItemSupport: " << ItemSupportMax << endl;
    cout << endl;

    cout << "SupportThreshold: " << SupportThreshold << endl;

    cout << endl;
    cout << "TransNumReduced: " << TransNumReduced << endl;
    cout << "TransSetDimensionReduced: " << TransSetDimensionReduced << endl;
    cout << "TransLengthMinReduced: " << TransLengthMinReduced << endl;
    cout << "TransLengthMaxReduced: " << TransLengthMaxReduced << endl;
    cout << "TransLengthMeanReduced: " << TransLengthMeanReduced << endl;
    cout << "TransSetSizeReductionRate: " << TransSetSizeReductionRate << endl;
    cout << "DimensionReductionRate: " << DimensionReductionRate << endl;
    cout << "TransSetReduced: " << CStringA(TransSetReduced) << endl;

    cout << endl;
    cout << "MaximalPopulationSize: " << MaximalPopulationSize << endl;
    cout << "OutputResult: " << CStringA(OutputResult) << endl;
    cout << "CPUTimePre: " << CPUTimePre << "s" << endl;
    cout << "CPUTime: " << CPUTime << "s" << endl;
    cout << endl;
    cout << endl;

    for (int i = 0; i < MaximalPopulationSize; i++)
    {
        cout << i + 1 << ": " << MaximalPopulation[i].number << " { ";
        for (int j = 1; j <= MaximalPopulation[i].number; j++)
        {
            if (MinItemID == 0) MaximalPopulation[i].ItemSets[j] = MaximalPopulation[i].ItemSets[j] - 1;
            if (j < MaximalPopulation[i].number) cout << MaximalPopulation[i].ItemSets[j] << " ";
            else cout << MaximalPopulation[i].ItemSets[j];
        }
        cout << " }     support: " << MaximalPopulation[i].support << "     times: " << MaximalPopulation[i].times << "     { ";
        for (int j = 1; j <= MaximalPopulation[i].number; j++)
        {
            if (j < MaximalPopulation[i].number) cout << MaximalPopulation[i].ItemSetsStr[j] << ",   ";
            else cout << MaximalPopulation[i].ItemSetsStr[j];

        }
        cout << " }" << endl;
    }
    cout << endl;
    //system("pause");
    //=====================================================================================================================================================
    return 0;
}


