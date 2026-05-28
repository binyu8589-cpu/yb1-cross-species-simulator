# Small-RNA-aware cross-species models predict engineered bacterial cell states

Code, baselines, and reproduction scripts for the manuscript
*"Small-RNA-aware cross-species models predict engineered bacterial cell states"* (under review).

A graph-constrained, small-RNA-aware transformer that predicts the transcriptional state of an
engineered, non-model bacterium across species, trained by an *E. coli* → *Salmonella* → in-house
curriculum with the engineered strain **YB1 held out from all training**. The repository reproduces
every quantitative result and display item (Tables 1–2, Figures 1–4) from processed data and
released checkpoints.

## Repository layout
```
model/        cross-species curriculum + sRNA-gate model
  train_stage1_ecoli.py            Stage 1 (E. coli, Tjaden 2023 compendium)
  train_stage2_v2.py / _kroger.py  Stage 2 (Salmonella; NCBI-scaled / Kröger-curated)
  train_stage3_v2_1_conditioned.py Stage 3 (in-house fine-tune, condition tokens)
  train_stage3_inhouse.py          in-house data utilities
  ecoli_lim_dataloader.py          E. coli compendium loader (Tjaden 2023)
  kroger_dataloader.py             Salmonella Kröger loader
baselines/    benchmarks + architecture ablations
  run_genie3.py                    GENIE3 (ExtraTrees) baseline
  run_log2fc_cmp.py                v5 vs GENIE3 vs mean-floor, shared-mask protocol (Table 2)
  run_log2fc_model_only.py         Table-2 protocol for any checkpoint
  run_nograph_seed.sh              no-graph curriculum ablation (3 seeds)
eval/         held-out YB1 evaluation
  eval_pathway_no_context_and_log2fc.py   pathway context-Δ + log2FC
  eval_yb1_mrna_n7.py / eval_yb1_holdout_n7.py
figures/      figure-generation scripts
data/processed/   processed count matrices (large matrices on Zenodo; see below)
checkpoints/      trained model checkpoints (on Zenodo; see below)
reproduce/        end-to-end scripts: count matrices + checkpoints -> Tables 1-2, Figures 1-4
```

## Installation
Training/evaluation environment: Python 3.12.3, PyTorch 2.12.0 + CUDA 13.0 (NVIDIA RTX 5090).

```bash
# Conda (recommended)
conda env create -f environment.yml && conda activate yb1sim

# or pip into a venv (GPU, CUDA 13 wheels)
python3.12 -m venv yb1sim && source yb1sim/bin/activate
pip install --extra-index-url https://download.pytorch.org/whl/cu130 -r requirements.txt
```
`requirements-lock.txt` gives a byte-exact lock (all transitive NVIDIA wheels) for GPU reproduction. For CPU-only, drop the index URL and the `+cu130` suffixes (`torch==2.12.0`, `torchvision==0.27.0`).

## Data and checkpoints
- **In-repo** (version-controlled): reference tables under `data/reference/` (gene vocabulary,
  *E. coli*↔*Salmonella* ortholog map, Kröger sRNA list, sRNA→target edges, *E. coli* sample
  metadata) and processed count matrices under `data/processed/*.tsv`; the nine Supplementary
  Data tables under `data/supplementary/`.
- **From Zenodo** (DOI **[10.5281/zenodo.20411440](https://doi.org/10.5281/zenodo.20411440)**):
  trained checkpoints (per curriculum stage × seed, incl. no-graph ablation) and the processed
  expression matrices — `master_expression_matrix.parquet` (cross-species training corpus) and
  `master_expression_matrix_v2.parquet` (the curated corpus the GENIE3/floor baselines use).
  Extract so that `checkpoints/` and `data/processed/*.parquet` sit at the repo root.
- **Raw nanopore RNA-seq** (YB1, SL7207, Δ*asd*, PW): NCBI GEO **[GSE333492](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE333492)** (27 runs; raw reads brokered to SRA; private during peer review, released on publication).
- **Public datasets**: *E. coli* compendium — Tjaden, *RNA Biology* 2023 (Harvard Dataverse
  [doi:10.7910/DVN/QBMC9D](https://doi.org/10.7910/DVN/QBMC9D)); *Salmonella* Kröger et al. 2013;
  RegulonDB; iML1515 (BiGG); KEIO; reference genome SL1344 (RefSeq GCF_000210855.2).

### Path configuration
Scripts resolve inputs relative to the repository by default; no editing is needed once the
Zenodo archive is extracted at the repo root. Override with environment variables if your data
live elsewhere:

| Variable | Default | Used for |
|---|---|---|
| `YB1_REPO` | repo root | base for all relative paths |
| `YB1_DATA` | `data/processed` | count matrices, master matrix |
| `YB1_REF`  | `data/reference` | vocab, ortholog map, sRNA tables |
| `YB1_CKPT` | `checkpoints` | trained checkpoints |
| `LIM_PARQUET` | `data/processed/lim2023_annotated.parquet` | Stage-1 *E. coli* compendium (download from Dataverse QBMC9D) |
| `WETLAB_NANOPORE`, `WETLAB_FACTORIAL`, `SL1344_GFF`, `WETLAB_RNA_DIR` | dev paths | `reproduce/count_*.py` (need the GEO raw BAMs) |

Stage-1 training and `reproduce/count_*.py` use upstream inputs (the public *E. coli* compendium
and the raw nanopore BAMs deposited at GEO) that are not bundled here; the evaluation scripts that
regenerate Tables 1–2 and Figures 3–4 run from the in-repo + Zenodo artifacts alone.

## Script → display-item map
| Item | Script |
|---|---|
| Absolute-expression numbers (in main text) | `eval/eval_yb1_mrna_n7.py`, `baselines/run_genie3.py` |
| Table 1 (strain-specific differential) | `baselines/run_log2fc_cmp.py`, `baselines/run_log2fc_model_only.py` |
| Table 2 (per-pathway differential, v5 vs GENIE3 vs floor) | `baselines/per_pathway_baselines.py` |
| Fig 1 (architectural obedience) | edge-holdout + causal-mask-removal eval |
| Fig 2 (model-vs-experiment double-blind) | v4 Δ*asd*/YB1 prediction + measured tables |
| Fig 3 (mRNA–sRNA complementarity) | `eval/eval_pathway_no_context_and_log2fc.py` + sRNA-category eval |
| Fig 4 (three-strain DAP-matched + phenotype growth curves) | `figures/` (count → log2FC → panel + growth curves) |
| no-graph ablation (Table 1, SI S5) | `baselines/run_nograph_seed.sh` + `baselines/run_log2fc_model_only.py` (3 seeds) |

## Reviewer reproduction (quick start)
End-to-end regeneration of the main quantitative results (Tables 1–2, Fig 3–4) from the
deposited checkpoints + count matrices. Verified from a clean checkout on a single
NVIDIA RTX 5090 (CUDA 13); the GPU steps take ~5 min after the environment is built.

```bash
# 1. Code
git clone https://github.com/binyu8589-cpu/yb1-cross-species-simulator.git
cd yb1-cross-species-simulator

# 2. Environment (Python 3.12, PyTorch 2.12 + CUDA 13)
python3.12 -m venv yb1sim && source yb1sim/bin/activate
pip install --extra-index-url https://download.pytorch.org/whl/cu130 -r requirements.txt

# 3. Data + checkpoints from Zenodo (DOI 10.5281/zenodo.20411440):
#    extract the archive so that  checkpoints/  and  data/processed/*.parquet
#    sit at the repo root (the small reference/count tables are already version-controlled).
#    The archive contains both master_expression_matrix.parquet and
#    master_expression_matrix_v2.parquet (the latter is required by the Table-1/2 baselines).

# 4. Run
bash reproduce/reproduce_all.sh
```

The script prints results to stdout, writes per-eval JSON next to each checkpoint, and renders
`figures/figure9_axis_control_framework.png` (Fig 4) and `figures/figure5d_dap_matched.png` (Fig 5d).

### Expected key numbers (held-out YB1)
| Quantity | Value |
|---|---|
| Absolute mRNA cosine, n=7 BAMs: **v5** / GENIE3 / mean-floor | **+0.948** / +0.917 / +0.934 |
| **Table 1** — strain-specific log2FC cosine (aer / ana): **v5** | **+0.744 / +0.669** |
| Table 1 — GENIE3 (aer / ana) | +0.653 / +0.600 |
| Table 1 — mean-floor (aer / ana) | +0.594 / +0.543 |
| Table 1 — v5 per seed S42 / S0 / S1 (aer) | +0.744 / +0.750 / +0.663 |
| **Table 2** — per-pathway differential: **v5 SPI-1 aer** vs GENIE3 vs floor | **+0.879** vs +0.635 vs +0.652 |
| Table 2 — Flagellar (aer / ana): v5 vs GENIE3 vs floor | +0.793/+0.818 vs +0.828/+0.827 vs +0.856/+0.539 |
| Table 2 — Chemotaxis (aer / ana): v5 vs GENIE3 vs floor | +0.941/+0.732 vs +0.922/+0.908 vs +0.903/+0.854 |
| Fig 3 — context-Δ (full − no-context): Flagellar / SPI-2 / SPI-1 / Chemotaxis | +0.13 / +0.08 / +0.04 / +0.04 |

Figures 1 (edge-holdout / causal-mask-removal) and 2 (model-vs-experiment double-blind) are
summarized in Supplementary Tables S3 / S6 (see the script map above). Stage-1/2/3 *training*
from scratch additionally needs the public *E. coli* compendium and the GEO raw BAMs (see
**Data and checkpoints**); the reproduction above runs from the in-repo + Zenodo artifacts alone.

## Citation
Yubin et al. *Small-RNA-aware cross-species models predict engineered bacterial cell states.* (under review). [DOI on publication]

## License
MIT — see [LICENSE](LICENSE).
