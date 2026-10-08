# AnyFIM

**AnyFIM: an anytime frequent itemset mining framework driven by progressive item activation, with CPU–GPU heterogeneous collaboration.**

This repository accompanies the paper

> *Anytime Frequent Itemset Mining: A Threshold-Free Progressive Item-Activation Framework with CPU–GPU Heterogeneous Collaboration*

and provides everything needed to reproduce the reported results on a single workstation: the proposed algorithm (two versions), all re-measured baselines, the benchmark datasets, and the expected outputs.

## Key properties

- **Threshold-free / anytime mining.** Instead of a prespecified minimum support, AnyFIM progressively activates items one at a time in descending order of support and mines maximal frequent itemsets (MFIs) over the currently active item universe, emitting a support-ordered MFI stream that is interruptible and resumable at any round boundary.
- **Single-side extension operator.** The right-hand-side expansion of our previous work is generalized to a single-side extension principle with two provably equivalent orientations (left/right), with completeness and non-redundancy guarantees.
- **CPU–GPU heterogeneous engine.** Word-level bitmap counting, node-level dynamic task-stack scheduling, and a measurement-driven task splitter; on the benchmark suite the full collaborative mode is one to two orders of magnitude faster than re-measured CPU, GPU, and distributed baselines on identical hardware.

## Repository layout

```
code/
  AnyFIM-GivenThreshold/   given-threshold version (CoParaCG, CUDA/C++)
  AnyFIM-Anytime/          anytime (threshold-free) version (AnytimeMining, CUDA/C++)
  baselines/                 RHSE, FPmax-LIB (FPMAX*), CD, DMM (faithful re-implementations), GMiner
datasets/                    chess, mushroom, pumsb, accidents (FIMI benchmark corpus)
tools/replicate.py           generate x2/x4/x8 replicated datasets for scalability tests
results/expected/            expected MFI outputs and ablation summary for verification
```

## Environment

- Windows 10/11 x64, Visual Studio 2022 (v143), CUDA Toolkit 12.5
- Tested on: Intel Core i9-12900 (16 physical / 24 logical cores), NVIDIA RTX 3060 Ti (sm_86), 128 GB RAM
- All reported numbers are measured in **Release x64** builds.

## Build and run

### AnyFIM (given-threshold version)

Open `code/AnyFIM-GivenThreshold/CoParaCG.sln`, build **Release | x64**, then:

```
CoParaCG.exe <RUN_MODE> <CPU_CORES> <dataset_path> <relative_threshold>
```

- `RUN_MODE`: 0 = serial (recursive, CPU counting) · 1 = GPU-only counting · 2 = full collaborative (task stack + splitter) · 3 = task stack + all-GPU counting (ablation modes A/B/D/C of Table II)
- Example: `CoParaCG.exe 2 16 datasets\pumsb.txt 0.80`
- Output: a `<dataset>-<threshold>=Results.txt` file next to the dataset, plus console timing (`computation_time(s)`); compare against `results/expected/`.

### AnyFIM (anytime version)

Open `code/AnyFIM-Anytime/AnytimeMining/AnytimeMining.vcxproj`, build **Release | x64**, and run; it iteratively activates items and writes per-round results without any threshold.

### Baselines

- **FPMAX*** (`baselines/FPmax-LIB`): `build_cli.bat` (cl /O2 /DMFI), then `run_fpmax.exe <input> <output> <abs_minsup>`.
- **RHSE** (`baselines/RHSE`): open the project, set dataset/threshold in `main()` (marked lines), build Release x64, run with the dataset one directory above the working directory.
- **CD / DMM** (`baselines/CD`, `baselines/DMM`): `build.bat`, then `cd.exe <dataset> <rel_threshold> [output] [P_nodes]` / `dmm.exe <dataset> <rel_threshold> [output] [P_nodes]`.
- **GMiner** (`baselines/GMiner`): `build_win.bat` (nvcc -arch=sm_86), then `GMiner.exe -i <data> -o <output> -s <rel_threshold> -w 1`. Note: GMiner enumerates **all** frequent itemsets (a superset of MFIs) and its wall-clock includes a fixed GPU-initialization overhead; the Windows/CUDA-12.5 port notes are in `baselines/GMiner/PORTING_NOTES.md`.

### Scalability datasets

```
python tools/replicate.py datasets/accidents.txt 4 accidents_x4.txt
python tools/replicate.py datasets/pumsb.txt 8 pumsb_x8.txt
```

## Verification

On all six benchmark configurations (Chess 0.95, Mushroom 0.52, Accidents 0.40, Pumsb 0.80, and the 4×/8× extended variants), the MFI family emitted by AnyFIM is **set-identical** to that of FPMAX* executed on the same files (see `results/expected/`); FPMAX* itself was validated set-identical against the SPMF open-source reference. The four ablation modes also produce identical MFI sets and identical search-tree node counts.

## Third-party code

`code/baselines/` contains third-party algorithms (FPmax-LIB, GMiner) retained under their original authors' terms for review and reproduction purposes, and our own faithful re-implementations (CD, DMM) of published cluster algorithms for which no public code is available.

## License

MIT (see `LICENSE`) for the AnyFIM code; third-party baselines remain under their original licenses/terms.
