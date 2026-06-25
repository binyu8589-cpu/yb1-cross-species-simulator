Code and analyses for **"Forward-drive dosage stratifies a coordinated regulatory state in logic-gated *Salmonella*"** (Yu, Ge & Cui, submitted to *Cell Systems*, 2026), together with the graph-constrained transcriptional model used in that study as a measuring instrument.

> This repository was previously associated with an earlier manuscript on model epistemic boundaries; it now hosts the current *Cell Systems* biology-first study and its analysis code.

## What's here

- **`cell_systems_2026/`** — analysis and figure scripts for the paper: the forward-drive operating-point test, the network-integration analysis (module coordination with Benjamini–Hochberg correction, and the genotype-by-condition interaction), the technical-batch and equal-depth audits, the *asd* engineered-locus coverage, and all main and supplementary figure scripts.
- **model/** — the graph-constrained, small-RNA-aware transformer (architecture, cross-species training curriculum, baselines). In the paper the model is used only as a conservative comparator (Fig. 4), not as positive evidence for any biological claim; its bounded role is documented in Supplementary Note 1.

## The study, briefly

A logic-gated *Salmonella* supplies the essential gene *asd* under an anaerobically induced *pepT* promoter together with a *sodA*-driven antisense module. Two strains share this design and differ only in *pepT*-*asd* cassette dosage: YB1 (1×) and PW (2×). Under matched anaerobic no-DAP conditions and a shared antisense background, PW and YB1 separate completely in a 70-gene invasion/motility score (group means −0.39 versus +1.61, about 2.0 log2-fold; two-sided permutation p = 0.057). The accompanying state is coordinated in selected modules (BH q ≤ 4×10⁻⁴), condition-dependent (interaction p = 0.0029), and genome-wide, indicating it is constructed by the host regulatory network rather than read from the construct.

## Reproduce

Requirements: Python 3.9+, `numpy` (<2), `pandas`, `matplotlib`, `pysam`. Each script reads the per-library count matrices and BAMs; edit the `TSV` / data paths at the top of each script to point to the count matrices provided as the paper's Source Data / Supplementary Data.

```bash
cd cell_systems_2026
python network_integration_analysis.py   # Fig 3 stats: coordination (9999 + plus-one + BH) and interaction
python fig2_setpoint.py                   # Fig 2: operating point (a) + asd engineered-locus coverage (b)
python fig_intelligence.py                # Fig 3: coordination / forward-drive shift / condition-dependence
python fig3_genomewide.py                 # Fig 4: genome-wide engineered-state breadth
python fig4_conditional.py                # Fig 5: DAP-matched controls
python fig5_srna.py                       # Fig 6: regulatory small-RNA layer
python figS1_S2.py                        # Supplementary Figures S1, S2
# audits
python technical_batch_audit.py           # sequencing-date / depth audit
python equal_depth_downsampling.py        # equal-depth downsampling control
```

## Data and code availability

- **RNA-seq:** NCBI GEO **GSE333492** (released on publication; reviewer token available from the editor).
- **Code:** this repository (MIT licence).
- **Trained model checkpoints:** Zenodo **10.5281/zenodo.20411440**, released upon publication; a reviewer access link is available from the editor.
- No human or clinical data.

## Citation

Yu B., Ge H., Cui D. *Forward-drive dosage stratifies a coordinated regulatory state in logic-gated Salmonella.* Submitted, 2026.
