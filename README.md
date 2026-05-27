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
  trained checkpoints (per curriculum stage × seed, incl. no-graph ablation) and the large
  processed matrix. Extract so that `checkpoints/` and `data/processed/master_expression_matrix.parquet`
  sit at the repo root.
- **Raw nanopore RNA-seq** (YB1, SL7207, Δ*asd*, PW): NCBI GEO **[GSE-XXXXXX]** / SRA **[PRJNA-XXXXXX]** (to be assigned).
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
| Table 1 (absolute-expression benchmark) | `eval/eval_yb1_mrna_n7.py`, `baselines/run_genie3.py` |
| Table 2 (strain-specific differential) | `baselines/run_log2fc_cmp.py`, `baselines/run_log2fc_model_only.py` |
| Fig 1 (architectural obedience) | edge-holdout + causal-mask-removal eval |
| Fig 2 (model-vs-experiment double-blind) | v4 Δ*asd*/YB1 prediction + measured tables |
| Fig 3 (mRNA–sRNA complementarity) | `eval/eval_pathway_no_context_and_log2fc.py` + sRNA-category eval |
| Fig 4 (three-strain DAP-matched) | `figures/` (count → log2FC → panel) |
| no-graph ablation (Table 2, SI S5) | `baselines/run_nograph_seed.sh` + `run_log2fc_model_only.py` (3 seeds) |

## Reproduce
```bash
bash reproduce/reproduce_all.sh   # from deposited count matrices + checkpoints -> all tables & figures
```

## Citation
Yubin et al. *Small-RNA-aware cross-species models predict engineered bacterial cell states.* (under review). [DOI on publication]

## License
MIT — see [LICENSE](LICENSE).
