# Self-Growing Sensor Evidence for Interpretable IIoT Fault Diagnosis

Reproducibility package for the manuscript submitted to *Sensors* (MDPI).

Industrial IoT sensor networks generate massive multivariate telemetry, yet
diagnosis pipelines that exploit cross-sensor co-occurrence patterns remain
impractical: mining algorithms demand preset support thresholds that field
engineers rarely know in advance, and the evidence consumed by LLM-based
diagnosis is manually curated, costly, and quickly stale. This project mines
the equipment's **own sensor stream** — without any threshold and
interruptible at any moment — into a library of maximal frequent patterns
(MFIs) with counted supports, and lets an LLM attribute faults by **citing**
those patterns, so every diagnostic claim is auditable against mined evidence.

Two mining engines are included as source code (each described by pseudocode
in the manuscript):

| Engine | Role | Location |
|---|---|---|
| **AnyFIM** | anytime, threshold-free MFI mining (CUDA) | `code/engine_anyfim/` |
| **TensorFIM** | tensor-core (BMMA) accelerated support counting (CUDA) | `code/engine_tensorfim/` |

## Repository layout

```
code/
  engine_anyfim/     AnyFIM engine source + baselines (CD, DMM, FPmax, GMiner)
  engine_tensorfim/  TensorFIM engine source + prebuilt Windows x64 binaries
  pipeline/          full analysis pipeline: discretization, transactionization,
                     engine adapter, and experiment scripts E1/E4/E5
experiments/         ablation scripts: E6/E7 (front end) and LLM diagnosis
tools/               figure plotting scripts (all manuscript figures)
results/             complete experiment outputs reported in the manuscript
data/                exported FIMI transactions + download instructions
                     for the raw benchmark datasets (not committed)
```

## Setup

```bash
git clone https://github.com/YOUR-USERNAME/self-growing-evidence.git
cd self-growing-evidence
pip install -r requirements.txt
```

Python ≥ 3.10, numpy/pandas/matplotlib only. The mining engines are Windows
x64 CUDA programs; TensorFIM ships prebuilt binaries in
`code/engine_tensorfim/portable/prebuilt/`, and both engines can be rebuilt
from source (Visual Studio + CUDA toolkit, see each engine's `README.md`).

Download the raw benchmark datasets per [`data/README.md`](data/README.md)
(SKAB, C-MAPSS, CWRU, AI4I 2020, HAI — all public) and verify them against the
committed SHA-256 manifests.

## Experiment → paper mapping

All commands are run from the repository root. `results/` already contains the
exact outputs reported in the manuscript; every command regenerates them.

| Paper output | Experiment | Command |
|---|---|---|
| §3.1 pattern–fault cross-check | E1 | `python code/pipeline/exp_e1_crosscheck.py` |
| Fig. 2 anytime coverage curve | E4 | `python code/pipeline/exp_e4_anytime.py` then `python tools/plot_e4_curve.py` |
| Fig. 3 threshold sensitivity | E5 data | `python code/pipeline/exp_e5_efficiency.py` then `python tools/plot_threshold_sensitivity.py` |
| Fig. 4 + Table 2 BMMA speedup | E5 | `python tools/plot_bmma_speedup.py` |
| §3.3 cross-dataset patterns | spike | `python code/pipeline/multi_dataset_spike.py --data data --out results/spike_v2` |
| Fig. 5 LLM diagnosis ablation | E-LLM | see below |
| Fig. 6 discretizer & window ablation | E6/E7 | `python experiments/exp_e6e7_ablation.py` then `python tools/plot_e6e7.py` |
| Fig. 1 framework schematic | — | `python tools/plot_framework_fig.py` |

Auxiliary: `python code/pipeline/exp_export_datasets.py` re-exports the raw
datasets into the integer-transaction files under `data/fimi/` that the
engines consume; `python code/pipeline/validate_mine_mfi.py` cross-checks the
reference Python miner against both engines on gold datasets.

### LLM diagnosis (Fig. 5)

```bash
cp experiments/.env.example experiments/.env   # fill in your own keys
python experiments/exp_llm_diagnosis.py        # 360 calls, ~minutes with keys
python experiments/summarize_llm_diag.py       # aggregates to results/llm_diag/summary.json
```

Three providers are used exactly as in the paper (DeepSeek-V3, Qwen-Plus,
GLM-4-Air); all calls go through standard OpenAI-compatible chat endpoints.
The frozen prompt, sampling (60 balanced SKAB windows, seed 42), and parsing
logic are embedded in the script for reviewer inspection.

## License

MIT, see [LICENSE](LICENSE). Benchmark datasets retain their own licenses and
are not redistributed here.

## Citation

See [CITATION.cff](CITATION.cff). If the manuscript is accepted, the entry
will be updated with volume and DOI.
