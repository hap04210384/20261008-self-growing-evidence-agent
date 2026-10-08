// dmm.cpp — Distributed Max-Miner (Chung & Luo, KAIS 2008) faithful reimplementation
// Two phases per the paper:
//   Local mining phase : each (virtual) node mines its local partition for LOCAL maximal
//                        frequent itemsets, using the same relative support threshold
//                        (local minsup = ceil(theta * localN)). Any local miner works;
//                        we use FPmax* (FPmax-LIB in-memory API), as the paper allows.
//   Global mining phase: the union of local MFIs forms the maximal candidate set.
//                        A top-down search counts candidates globally with a prefix tree;
//                        infrequent candidates are expanded to their (s-1)-subsets for the
//                        next counting round; frequent candidates not covered by an already
//                        found global MFI are reported as global MFIs.
// The 8 virtual nodes are simulated on one machine (matches the historical 8-node cluster).
// usage: dmm.exe <dataset> <rel_threshold> [output_mfi] [P_nodes]
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <chrono>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>
#include <set>
#include <map>
#include <functional>
#include <algorithm>
#include <omp.h>
#include "fpmax.h"
using namespace std;

// ------------------------- prefix tree for global candidate counting -------------------------
struct TrieNode {
    vector<pair<int,int>> children; // (item, child), sorted by item
    int cand = -1;                  // candidate index ending at this node
};
static vector<TrieNode> g_trie;

static void trieInsert(const vector<int>& c, int idx)
{
    int node = 0;
    for (int x : c) {
        // NOTE: never hold a reference to children across push_back (reallocation)
        int lo = 0, hi = (int)g_trie[node].children.size();
        while (lo < hi) { int mid = (lo + hi) / 2; if (g_trie[node].children[mid].first < x) lo = mid + 1; else hi = mid; }
        if (lo < (int)g_trie[node].children.size() && g_trie[node].children[lo].first == x) {
            node = g_trie[node].children[lo].second;
        } else {
            int nn = (int)g_trie.size();
            g_trie.push_back(TrieNode());
            g_trie[node].children.insert(g_trie[node].children.begin() + lo, make_pair(x, nn));
            node = nn;
        }
    }
    g_trie[node].cand = idx;
}

// walk the transaction through the tree; EVERY candidate whose itemset is a subset
// of the transaction gets counted (candidates may have mixed sizes)
static void trieCount(const vector<int>& t, int node, int start, vector<long long>& cnt)
{
    if (g_trie[node].cand >= 0) cnt[g_trie[node].cand]++;
    const auto& ch = g_trie[node].children;
    for (size_t i = start; i < t.size(); i++) {
        int x = t[i];
        int lo = 0, hi = (int)ch.size();
        while (lo < hi) { int mid = (lo + hi) / 2; if (ch[mid].first < x) lo = mid + 1; else hi = mid; }
        if (lo < (int)ch.size() && ch[lo].first == x)
            trieCount(t, ch[lo].second, (int)i + 1, cnt);
    }
}

int main(int argc, char** argv)
{
    if (argc < 3) { fprintf(stderr, "usage: dmm <dataset> <rel_threshold> [output] [P]\n"); return 1; }
    const char* path = argv[1];
    double theta = atof(argv[2]);
    const char* outpath = argc >= 4 ? argv[3] : nullptr;
    int P = argc >= 5 ? atoi(argv[4]) : 8;

    auto t0 = chrono::steady_clock::now();

    // ---- load DB ----
    vector<vector<int>> DB;
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
    long long gminsup = (long long)ceil(theta * (double)N - 1e-12);
    fprintf(stderr, "DMM: N=%lld theta=%g global_minsup=%lld P=%d\n", N, theta, gminsup, P);

    // ---- local mining phase: each virtual node mines its partition for local MFIs ----
    size_t per = (DB.size() + P - 1) / P;
    vector<vector<int>> candPool;        // union of local MFIs (sorted vectors)
    {
        set<vector<int>> dedup;
        for (int p = 0; p < P; p++) {
            size_t b = p * per, e = min(DB.size(), b + per);
            if (b >= e) continue;
            long long localN = (long long)(e - b);
            long long lminsup = (long long)ceil(theta * (double)localN - 1e-12);
            Dataset ds;
            for (size_t i = b; i < e; i++) ds.push_back(set<int>(DB[i].begin(), DB[i].end()));
            FISet* local = fpmax(&ds, (unsigned int)lminsup, 0);
            size_t nlocal = local->size();
            for (auto& fi : *local) {
                vector<int> v(fi.begin(), fi.end());
                sort(v.begin(), v.end());
                dedup.insert(move(v));
            }
            delete local;
            fprintf(stderr, "node %d: localN=%lld lminsup=%lld localMFIs=%zu (pool=%zu)\n",
                    p, localN, lminsup, nlocal, dedup.size());
        }
        candPool.assign(dedup.begin(), dedup.end());
        // longest first (top-down search)
        sort(candPool.begin(), candPool.end(),
             [](const vector<int>& a, const vector<int>& b) {
                 if (a.size() != b.size()) return a.size() > b.size();
                 return a < b;
             });
    }
    fprintf(stderr, "local phase done: %zu maximal candidates\n", candPool.size());

    // ---- global mining phase: level-wise top-down counting with the prefix tree ----
    // Levels are processed in STRICTLY decreasing candidate size: a size-s level is
    // counted only after every size-(s+1) candidate has been processed, so that when
    // a frequent set X is reported, ALL of its potential frequent supersets have
    // already been seen (they either entered the initial pool as local MFIs or were
    // generated as subsets of an infrequent superset one level up).
    vector<vector<int>> globalMFI;
    {
        // poolBySize[s] = deduped candidates of size s awaiting counting
        map<int, vector<vector<int>>, greater<int>> poolBySize;
        {
            set<vector<int>> seen;
            for (auto& c : candPool)
                if (seen.insert(c).second) poolBySize[(int)c.size()].push_back(c);
        }
        while (!poolBySize.empty()) {
            auto lvl = poolBySize.begin();           // current largest size
            int sz = lvl->first;
            vector<vector<int>> pool = move(lvl->second);
            poolBySize.erase(lvl);

            g_trie.clear();
            g_trie.reserve(pool.size() * 4 + 1);
            g_trie.push_back(TrieNode());
            for (size_t i = 0; i < pool.size(); i++) trieInsert(pool[i], (int)i);
            // support counting: parallel over transactions for large DBs,
            // serial when the DB is small and the candidate pool is huge
            vector<long long> cnt(pool.size(), 0);
            if (DB.size() >= 100000) {
                int nt = omp_get_max_threads();
                vector<vector<long long>> part(nt, vector<long long>(pool.size(), 0));
                #pragma omp parallel for schedule(dynamic, 8192)
                for (long long ti = 0; ti < (long long)DB.size(); ti++)
                    trieCount(DB[ti], 0, 0, part[omp_get_thread_num()]);
                for (int t = 0; t < nt; t++)
                    for (size_t i = 0; i < pool.size(); i++) cnt[i] += part[t][i];
            } else {
                for (auto& t : DB) trieCount(t, 0, 0, cnt);
            }

            // pass 1: this level's frequent candidates
            vector<size_t> freqIdx;
            for (size_t i = 0; i < pool.size(); i++)
                if (cnt[i] >= gminsup) freqIdx.push_back(i);
            // pass 2: maximal w.r.t. globalMFI and this level's frequent sets
            // (same size => mutual coverage impossible, so globalMFI suffices;
            //  the check against freqIdx is kept for safety but only larger sets
            //  can cover, and none exist at this level)
            for (size_t a = 0; a < freqIdx.size(); a++) {
                const vector<int>& x = pool[freqIdx[a]];
                bool covered = false;
                for (auto& m : globalMFI)
                    if (m.size() > x.size() &&
                        includes(m.begin(), m.end(), x.begin(), x.end())) { covered = true; break; }
                if (!covered) globalMFI.push_back(x);
            }
            // pass 3: expand infrequent candidates to (s-1)-subsets for the next level
            if (sz > 1) {
                auto& nxt = poolBySize[sz - 1];
                set<vector<int>> dedup(nxt.begin(), nxt.end());
                for (size_t i = 0; i < pool.size(); i++) {
                    if (cnt[i] >= gminsup) continue;
                    for (int skip = 0; skip < sz; skip++) {
                        vector<int> sub;
                        for (int d = 0; d < sz; d++)
                            if (d != skip) sub.push_back(pool[i][d]);
                        bool covered = false;
                        for (auto& m : globalMFI)
                            if (m.size() > sub.size() &&
                                includes(m.begin(), m.end(), sub.begin(), sub.end())) { covered = true; break; }
                        if (!covered && dedup.insert(sub).second) nxt.push_back(move(sub));
                    }
                }
            }
            fprintf(stderr, "global level %d: pool=%zu found=%zu\n",
                    sz, pool.size(), globalMFI.size());
        }
    }

    auto t1 = chrono::steady_clock::now();
    if (outpath) {
        ofstream out(outpath);
        for (auto& m : globalMFI) {
            for (size_t d = 0; d < m.size(); d++) out << m[d] << " ";
            out << "\n";
        }
    }
    printf("ELAPSED %.6f\n", chrono::duration<double>(t1 - t0).count());
    printf("TOTAL_MFI %zu\n", globalMFI.size());
    return 0;
}
