# Raw datasets

Raw benchmark data are **not** part of this repository (licenses and sizes
differ per source). Download them into the layout below; every experiment
script reads from `data/` relative to the repository root.

## Directory layout

```
data/
├── SKAB_raw/                 # git clone of the SKAB repository
│   └── data/
│       ├── valve1/*.csv
│       ├── valve2/*.csv
│       └── other/*.csv
├── cmapss/CMAPSSData/train_FD001.txt
├── cwru/*.mat                # 12 files: normal + 4 fault types × 3 loads
├── hai/test1.csv.gz
└── ai4i/ai4i2020.csv
```

`data/fimi/` holds the exported integer-transaction files used by the mining
engines directly. These transaction files are **not** committed: they are
derived from the public benchmark datasets above, which we do not
redistribute — please download them from their original sources. Regenerate
the transactions at any time with
`python code/pipeline/exp_export_datasets.py` after the raw data are in
place. What is committed under `data/fimi/` are only the item/label mappings
and the engine output (`*=Results.txt`) files needed to audit the reported
results.

## Sources

| Dataset | Source | Notes |
|---|---|---|
| SKAB | https://github.com/waico/SKAB | clone; uses the `data/` folder inside the repo |
| C-MAPSS (FD001) | https://www.nasa.gov/intelligent-systems-division/discovery-and-systems-health/pcoe/pcoe-data-set-repository/ | NASA turbofan degradation |
| CWRU | https://engineering.case.edu/bearingdatacenter | 12 .mat files: normal, IR/OR/BF × loads 0/1/2 |
| AI4I 2020 | https://archive.ics.uci.edu/ml/datasets/AI4I+2020+Predictive+Maintenance+Dataset | `ai4i2020.csv` |
| HAI | https://github.com/icsdataset/hai | test1.csv.gz of the HAI 1.0 test set |
| FIMI benchmarks | https://fimi.uantwerpen.be/ | only needed to rebuild engine gold results; see `code/engine_tensorfim/data/DOWNLOAD.md` and `code/engine_anyfim/README.md` |

## Integrity verification

SHA-256 manifests for the copies used in the paper are committed:

```bash
# from the repository root (Git Bash / Linux); manifest paths are relative to data/
cd data
sha256sum -c SKAB_manifest.sha256
sha256sum -c manifest_all.sha256
```
