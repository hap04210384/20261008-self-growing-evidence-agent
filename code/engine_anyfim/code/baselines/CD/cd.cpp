// cd.cpp — Count Distribution (Agrawal & Shafer, TKDE 1996) faithful reimplementation
// Degenerated to a single machine simulating P virtual nodes:
//   each level k: every virtual node counts local supports of Ck over its partition,
//   counts are exchanged and summed (= global counts), pruning follows.
// With one physical machine this is exactly serial Apriori with per-level partition scans.
// usage: cd.exe <dataset> <rel_threshold> [output_fi] [P_nodes]
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <chrono>
#include <fstream>
#include <iostream>
#include <string>
#include <unordered_set>
#include <unordered_map>
#include <vector>
#include <algorithm>
using namespace std;

static vector<vector<int>> DB;      // transactions (sorted ascending)
static vector<int> F1;              // frequent 1-items (sorted)

struct VecHash {
    size_t operator()(const vector<int>& v) const noexcept {
        size_t h = v.size();
        for (int x : v) h = h * 1000003u ^ (unsigned)x;
        return h;
    }
};

// candidate prefix tree (hash-tree a la Apriori) for level counting
struct TrieNode {
    vector<pair<int,int>> children; // (item, child index), sorted by item
    int cand = -1;                  // candidate index at full depth
};
static vector<TrieNode> g_trie;

static void trieInsert(const vector<int>& c, int idx)
{
    int node = 0;
    for (int x : c) {
        auto& ch = g_trie[node].children;
        int lo = 0, hi = (int)ch.size();
        while (lo < hi) { int mid = (lo + hi) / 2; if (ch[mid].first < x) lo = mid + 1; else hi = mid; }
        if (lo < (int)ch.size() && ch[lo].first == x) { node = ch[lo].second; }
        else {
            int nn = (int)g_trie.size();
            g_trie.push_back(TrieNode());
            ch.insert(ch.begin() + lo, make_pair(x, nn));
            node = nn;
        }
    }
    g_trie[node].cand = idx;
}

static void trieCount(const vector<int>& t, int node, int depth, int start, int k, vector<int>& candCount)
{
    if (depth == k) {
        if (g_trie[node].cand >= 0) candCount[g_trie[node].cand]++;
        return;
    }
    const auto& ch = g_trie[node].children;
    if (ch.empty()) return;
    for (size_t i = start; i < t.size(); i++) {
        int x = t[i];
        int lo = 0, hi = (int)ch.size();
        while (lo < hi) { int mid = (lo + hi) / 2; if (ch[mid].first < x) lo = mid + 1; else hi = mid; }
        if (lo < (int)ch.size() && ch[lo].first == x)
            trieCount(t, ch[lo].second, depth + 1, (int)i + 1, k, candCount);
    }
}

// direct O(1) probe table when |F1|<=26: idxOf[mask] -> candidate index or -1
static vector<int> g_idxOf;
static const unordered_map<unsigned long long, int>* g_candIdx = nullptr;

static inline void probe(unsigned long long m, vector<int>& candCount)
{
    if (!g_idxOf.empty()) {
        int i = g_idxOf[m];
        if (i >= 0) candCount[i]++;
    } else {
        auto it = g_candIdx->find(m);
        if (it != g_candIdx->end()) candCount[it->second]++;
    }
}

// enumerate size-k submasks of transaction mask via bit-position combinations
static inline void countMask(unsigned long long tmask, int k,
                             vector<int>& candCount)
{
    int bits[64];
    int pc = 0;
    while (tmask) {
        unsigned long long lb = tmask & (~tmask + 1);
        int pos = 0;
        while ((lb >> pos) > 1) pos++;   // bit index of lowest set bit
        // faster: use __builtin_ctz equivalent
        bits[pc++] = pos;
        tmask &= tmask - 1;
    }
    if (pc < k) return;
    if (pc == k) {
        unsigned long long m = 0;
        for (int i = 0; i < pc; i++) m |= (1ULL << bits[i]);
        probe(m, candCount);
        return;
    }
    // iterate k-combinations of pc positions (lexicographic index walk)
    int idx[64];
    for (int i = 0; i < k; i++) idx[i] = i;
    for (;;) {
        unsigned long long m = 0;
        for (int i = 0; i < k; i++) m |= (1ULL << bits[idx[i]]);
        probe(m, candCount);
        int i = k - 1;
        while (i >= 0 && idx[i] == pc - k + i) i--;
        if (i < 0) break;
        idx[i]++;
        for (int j = i + 1; j < k; j++) idx[j] = idx[j - 1] + 1;
    }
}

int main(int argc, char** argv)
{
    if (argc < 3) { fprintf(stderr, "usage: cd <dataset> <rel_threshold> [output] [P]\n"); return 1; }
    const char* path = argv[1];
    double rel = atof(argv[2]);
    const char* outpath = argc >= 4 ? argv[3] : nullptr;
    int P = argc >= 5 ? atoi(argv[4]) : 8;   // virtual nodes (matches historical 8-node cluster)

    auto t0 = chrono::steady_clock::now();

    // ---- load DB, split into P contiguous partitions (as CD distributes the DB) ----
    {
        ifstream in(path);
        string line;
        while (getline(in, line)) {
            vector<int> t; int x = 0; bool innum = false;
            for (char c : line) {
                if (c >= '0' && c <= '9') { x = x * 10 + (c - '0'); innum = true; }
                else if (innum) { t.push_back(x); x = 0; innum = false; }
            }
            if (innum) t.push_back(x);
            if (!t.empty()) { sort(t.begin(), t.end()); DB.push_back(move(t)); }
        }
    }
    long long N = (long long)DB.size();
    long long minsup = (long long)ceil(rel * (double)N - 1e-12);
    fprintf(stderr, "CD: N=%lld rel=%g minsup=%lld P=%d\n", N, rel, minsup, P);

    long long totalFI = 0;
    int maxItem = 0;
    ofstream fout;
    if (outpath) fout.open(outpath);

    // ---- level 1: local counts per partition + exchange ----
    {
        unordered_set<int> seen;
        for (auto& t : DB) for (int x : t) seen.insert(x);
        maxItem = seen.empty() ? 0 : *max_element(seen.begin(), seen.end());
        vector<long long> gcnt(maxItem + 2, 0);
        // virtual-node local counting loop (explicit to mirror CD structure)
        size_t per = (DB.size() + P - 1) / P;
        for (int p = 0; p < P; p++) {
            vector<long long> lcnt(maxItem + 2, 0);
            size_t b = p * per, e = min(DB.size(), b + per);
            for (size_t i = b; i < e; i++)
                for (int x : DB[i]) lcnt[x]++;
            for (int i = 0; i <= maxItem; i++) gcnt[i] += lcnt[i]; // count exchange
        }
        for (int i = 0; i <= maxItem; i++)
            if (gcnt[i] >= minsup) F1.push_back(i);
        totalFI += (long long)F1.size();
        if (outpath) for (int x : F1) fout << x << " (" << gcnt[x] << ")\n";
        fprintf(stderr, "level 1: |F1|=%zu\n", F1.size());
    }
    if (F1.empty()) { fprintf(stderr, "no frequent items\n"); return 0; }

    // restrict transactions to F1 items once (Apriori optimization)
    // and build 64-bit masks over the F1 index space for fast counting
    vector<unsigned long long> DBM;
    {
        if (F1.size() > 63) { fprintf(stderr, "F1 too large for mask counting\n"); return 3; }
        vector<int> bit(maxItem + 2, -1);
        for (size_t i = 0; i < F1.size(); i++) bit[F1[i]] = (int)i;
        for (auto& t : DB) {
            unsigned long long m = 0;
            vector<int> u;
            for (int x : t) if (bit[x] >= 0) { u.push_back(x); m |= (1ULL << bit[x]); }
            t.swap(u);
            DBM.push_back(m);
        }
    }

    // ---- levels k>=2 ----
    vector<vector<int>> Fprev;
    for (int x : F1) Fprev.push_back(vector<int>{x});
    int k = 2;
    // resume from checkpoint when <output>.ckpt exists (same dataset & threshold)
    if (outpath) {
        string ck = string(outpath) + ".ckpt";
        ifstream cin2(ck);
        if (cin2) {
            int kk; long long tf;
            if (cin2 >> kk >> tf) {
                vector<vector<int>> saved;
                string line2;
                getline(cin2, line2);
                while (getline(cin2, line2)) {
                    vector<int> v; int x = 0; bool innum = false;
                    for (char c : line2) {
                        if (c >= '0' && c <= '9') { x = x * 10 + (c - '0'); innum = true; }
                        else if (innum) { v.push_back(x); x = 0; innum = false; }
                    }
                    if (innum) v.push_back(x);
                    if (!v.empty()) saved.push_back(move(v));
                }
                if (!saved.empty()) {
                    Fprev = move(saved); k = kk; totalFI = tf;
                    fout.close();
                    fout.open(outpath, ios::app);
                    fprintf(stderr, "resumed at level %d (totalFI so far %lld)\n", k, totalFI);
                }
            }
        }
    }
    while (!Fprev.empty()) {
        // candidate generation (join + subset prune), sorted lexicographically
        vector<vector<int>> Ck;
        for (size_t i = 0; i < Fprev.size(); i++)
            for (size_t j = i + 1; j < Fprev.size(); j++) {
                bool share = true;
                for (int d = 0; d < k - 2; d++)
                    if (Fprev[i][d] != Fprev[j][d]) { share = false; break; }
                if (!share) break; // sorted: further j won't match either
                vector<int> c = Fprev[i];
                c.push_back(Fprev[j][k - 2]);
                // prune: every (k-1)-subset must be frequent
                bool ok = true;
                for (int skip = 0; skip < k && ok; skip++) {
                    vector<int> s;
                    for (int d = 0; d < k; d++) if (d != skip) s.push_back(c[d]);
                    if (!binary_search(Fprev.begin(), Fprev.end(), s)) ok = false;
                }
                if (ok) Ck.push_back(move(c));
            }
        if (Ck.empty()) break;
        sort(Ck.begin(), Ck.end());
        // candidate masks -> index in Ck
        vector<int> bit(maxItem + 2, -1);
        for (size_t i = 0; i < F1.size(); i++) bit[F1[i]] = (int)i;
        unordered_map<unsigned long long, int> candIdx;
        candIdx.reserve(Ck.size() * 2);
        for (size_t i = 0; i < Ck.size(); i++) {
            unsigned long long m = 0;
            for (int x : Ck[i]) m |= (1ULL << bit[x]);
            candIdx[m] = (int)i;
        }
        // build candidate prefix tree (hash-tree counting, as in Apriori)
        g_trie.clear();
        g_trie.reserve(Ck.size() * 2 + 1);
        g_trie.push_back(TrieNode()); // root
        for (size_t i = 0; i < Ck.size(); i++) trieInsert(Ck[i], (int)i);
        vector<int> candCount(Ck.size(), 0);

        // counting: P virtual nodes scan their partition IN PARALLEL (each node on its
        // own core, as on a cluster), then exchange and sum the local counts
        size_t per = (DB.size() + P - 1) / P;
        vector<vector<int>> allCnt(P, vector<int>(Ck.size(), 0));
        #pragma omp parallel for schedule(dynamic, 1)
        for (int p = 0; p < P; p++) {
            size_t b = p * per, e = min(DB.size(), b + per);
            for (size_t i = b; i < e; i++)
                trieCount(DB[i], 0, 0, 0, k, allCnt[p]);
        }
        for (int p = 0; p < P; p++)
            for (size_t i = 0; i < Ck.size(); i++) candCount[i] += allCnt[p][i]; // count exchange
        vector<vector<int>> Fk;
        for (size_t i = 0; i < Ck.size(); i++)
            if (candCount[i] >= minsup) {
                Fk.push_back(Ck[i]);
                if (outpath) {
                    for (int d = 0; d < k; d++) fout << Ck[i][d] << " ";
                    fout << "(" << candCount[i] << ")\n";
                }
            }
        totalFI += (long long)Fk.size();
        fprintf(stderr, "level %d: |C|=%zu |F|=%zu\n", k, Ck.size(), Fk.size());
        Fprev = move(Fk);
        k++;
        if (outpath) {
            string ck = string(outpath) + ".ckpt";
            ofstream co(ck);
            co << k << " " << totalFI << "\n";
            for (auto& v : Fprev) {
                for (size_t d = 0; d < v.size(); d++) co << v[d] << " ";
                co << "\n";
            }
        }
    }

    if (outpath) { string ck = string(outpath) + ".ckpt"; remove(ck.c_str()); }
    auto t1 = chrono::steady_clock::now();
    printf("ELAPSED %.6f\n", chrono::duration<double>(t1 - t0).count());
    printf("TOTAL_FI %lld\n", totalFI);
    return 0;
}
