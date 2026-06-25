# Forward-drive dosage shifts a coordinated regulatory state in logic-gated Salmonella

  Code and reproducibility materials for the manuscript *"Forward-drive dosage shifts a 
  coordinated regulatory state in logic-gated Salmonella"* (Yu, Ge, Cui)

  ## What this is
  A graph-constrained, small-RNA-aware transformer for bacterial transcriptional state, and
  — the point of the paper — a **map of where such a model can and cannot be trusted**. We
  localize reliability to four separately-measurable boundaries: **structural** (the
  curated prior), **supervision** (what the training has seen), **representation** (what
  the substrate can express), and **observational** (what is measured of the target state).

  The cross-species model reconstructs engineered-versus-parent transcriptional
  **differentials when the target strain is partly observed** (≈70% context); it is a
  **context-conditioned reconstructor, not a design-to-state predictor**, and it collapses
  toward zero without context. Reported comparisons use a hierarchical bootstrap; the model
  exceeds an abundance floor but shows **no consistent superiority over a GENIE3-style 
  ExtraTrees baseline** (the paired v5 − baseline 95% CI excludes 0 only for YB1 aerobic).

  ## Reproducing the figures
  ```
  pip install -r requirements.txt
  python h_fig1.py   # structural boundary: edge-holdout precision collapse +
  master-regulator recovery
  python h_fig2.py   # representation boundary: edit encoded but not propagated (predicted
  vs measured engineered-state residual)
  python h_fig3.py   # absolute (saturated) vs engineered differential cosine
  python h_fig4.py   # differential reconstruction at 70% context; no-context collapse
  (observational boundary)
  python h_fig5.py   # DAP-matched panels + sRNA signature
  python h_fig6.py   # the four-boundary epistemic map
  ```
  Hierarchical-bootstrap CIs: `bootstrap_ci.json`. Shared-gene-space recompute for Fig. 2:
  `fig2_shared_recompute.py`. Per-figure source values: `Source_Data/`.

  ## Scope note (important)
  This repository contains only the **condition-static** synthetic-lethal benchmark used
  for the supervision-boundary result (1/1 unconditional pair recovered, 7/7
  condition-dependent pairs missed). Triggering-condition / G×G×E **recovery** experiments
  are **not** part of this work and are deliberately excluded.

  ## Data and models
  - RNA-seq: NCBI GEO **GSE333492** (SRA BioProject **PRJNA1471189**).
  - Trained checkpoints + processed expression matrix: Zenodo **10.5281/zenodo.20411440**
  (CC-BY-4.0).
  - License: MIT.
