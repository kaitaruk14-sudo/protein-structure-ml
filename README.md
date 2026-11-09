# protein-structure-ml

## Requirements

- macOS or Linux (developed on an Apple Silicon MacBook; PyTorch uses the `mps` GPU there, otherwise CUDA, otherwise CPU)
- [Miniforge / conda](https://github.com/conda-forge/miniforge)
- Python 3.10 or newer
- About 1 GB of free disk space for the prepared data (about 5 GB if you rebuild the dataset yourself)

## Installation

```bash
git clone https://github.com/kaitaruk14-sudo/protein-structure-ml.git
cd protein-structure-ml

conda create -n protein_ml python=3.12 -y
conda activate protein_ml
pip install torch numpy matplotlib scikit-learn joblib requests
```

All commands below assume the `protein_ml` environment is active (the prompt starts with `(protein_ml)`, not `(base)`).

## Data

### Option A: download the prepared data (recommended)

1. Go to the release [`data-v1`](https://github.com/kaitaruk14-sudo/protein-structure-ml/releases/tag/data-v1).
2. Under **Assets**, download `protein_data_v1.zip` (about 405 MB). Not the "Source code" files.
3. Move it into the repository folder and unzip it there:

```bash
unzip protein_data_v1.zip
ls data/protein_structures/backbone | wc -l    # should print 46607
rm protein_data_v1.zip                         # so it is never committed
```

If `unzip` asks to overwrite `backbone_index.tsv` or `dataset_split.tsv`, answer yes (the files are identical).

### Option B: rebuild the data from the RCSB PDB

The PDB is updated every week, so a rebuild gives a slightly different dataset and train/val/test split than Option A.
Needs MMseqs2 and takes a few hours (mostly the structure download).

```bash
conda install -c conda-forge -c bioconda mmseqs2 -y
cd data/protein_structures
python get_human_entities.py    # -> human_protein_entities_v2.txt
python prepare_dataset.py       # -> dataset_entities.txt
python cluster_split.py         # -> dataset_split.tsv
python download_backbones.py    # -> backbone/*.npz, backbone_index.tsv (resumable: re-run after Ctrl+C)
cd ../..
```

## Pipeline

Run everything from the repository root unless stated otherwise. Trained models are included in `model/checkpoints/`, so the evaluation steps work without retraining.

| Step | Command | Output |
|---|---|---|
| Dataset analysis | `cd analysis && python analyze_dataset.py && cd ..` | figures and `summary.md` in `analysis/` |
| Baselines | `python model/baselines.py` | `results/progress_report/` |
| Neural model: quick check | `python model/train.py --quick` | `results/progress_report/quick_test/` |
| Neural model: training | `python model/train.py` | `model/checkpoints/basic_best.pt`, `train_log.csv` |
| Neural model: evaluation | `python model/evaluate.py` | table and figures in `results/progress_report/` |
| Tree models: training | `python model/tree_models.py train` | `model/checkpoints/*_best.joblib` |
| Tree models: evaluation | `python model/tree_models.py evaluate` | adds columns to `per_chain_val.csv`, updates the table |
| Paired comparison | `python model/compare_methods.py` | `results/progress_report/neural_vs_logreg.md` |
| Sanity checks | `python model/sanity_checks.py \| tee results/progress_report/sanity_checks.txt` | printed, and saved to `sanity_checks.txt` |
| Error analysis | `python model/error_analysis.py` | `results/progress_report/error_analysis/` |

### Training options

```bash
python model/train.py --run_name long --epochs 16 --max_minutes 240
```

- `--run_name` saves to `model/checkpoints/<run_name>_best.pt` and `train_log_<run_name>.csv` (default `basic`).
- `--epochs`, `--max_minutes`: training stops at whichever comes first.
- Training can be stopped with Ctrl+C and resumed by running the same command again.
- On macOS, prefix long runs with `caffeinate -is` and keep the laptop plugged in with the lid open so it doesn't sleep.

## Troubleshooting

- **`ModuleNotFoundError: No module named 'torch'` (or `matplotlib`)**: the wrong environment is active. Run `conda activate protein_ml`.
- **`OMP: Error #15` or `segmentation fault` (exit code 139) on macOS**: two copies of the OpenMP library are loaded. The scripts in `model/` set the fix automatically through `model/common.py`; for any other script, run it as
  `KMP_DUPLICATE_LIB_OK=TRUE OMP_NUM_THREADS=1 python <script>.py`
- **`FileNotFoundError` for a `.npz` file**: the data isn't unzipped in the repository root (see [Data](#data)).
