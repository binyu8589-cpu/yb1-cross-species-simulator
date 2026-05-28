#!/usr/bin/env bash
# Regenerate the main quantitative results (Tables 1-2, Fig 3-4) from the
# deposited checkpoints + count matrices. Run from the repository root after:
#   1. installing the environment (see ../README.md: environment.yml / requirements.txt)
#   2. extracting the Zenodo archive so that checkpoints/ and
#      data/processed/*.parquet sit at the repo root (the small reference and
#      count tables under data/ are already version-controlled).
#
# Paths default to repo-relative locations; override with YB1_REPO / YB1_DATA /
# YB1_REF / YB1_CKPT if your data live elsewhere (see README "Path configuration").
set -euo pipefail
cd "$(dirname "$0")/.."

CKPT="${YB1_CKPT:-checkpoints}/checkpoints_stage3v2_1_cond_S42/best.pt"
PY="${PYTHON:-python}"

echo "== Absolute-expression numbers (held-out YB1; reported in text) =="
$PY eval/eval_yb1_mrna_n7.py --ckpt "$CKPT"
$PY baselines/run_genie3.py                      # GENIE3 + mean-expression floor

echo "== Table 1: strain-specific differential (v5 vs GENIE3 vs floor) =="
$PY baselines/run_log2fc_cmp.py
for S in 42 0 1; do
  $PY baselines/run_log2fc_model_only.py \
      --ckpt "checkpoints/checkpoints_stage3v2_1_cond_S${S}/best.pt" --tag "S${S}"
done

echo "== Table 2: per-pathway differential (v5 vs GENIE3 vs floor by regulon) =="
$PY baselines/per_pathway_baselines.py

echo "== Fig 3: cross-layer context ablation + sRNA categories =="
$PY eval/eval_pathway_no_context_and_log2fc.py --ckpt "$CKPT"

echo "== Fig 4 / 5d: three-strain DAP-matched panels =="
$PY figures/make_figure9_axis_control_framework.py
$PY figures/make_figure5d_dap_matched.py || true   # needs DAP-matched count tables

echo "DONE. Figure 1 (edge-holdout / mask-removal) and Fig 2 (double-blind) are"
echo "summarized in Supplementary Tables S3 / S6; see README script map."
