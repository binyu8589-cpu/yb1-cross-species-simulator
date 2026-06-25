"""Figure 3 — The accompanying state extends broadly beyond the edited locus.
Dimension-normalized RMS residual (comparable across gene-set sizes) + PROPORTION of
measured genes above threshold. Model = comparator only, NOT biological evidence."""
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, numpy as np
OUT = "/Users/dryu/claude_paper1_rebuild"
conds = ["aerobic\n(DAP-matched)", "anaerobic\n(sensitivity)"]; x = np.arange(2); w = 0.36
# L2 norms (Fig2b, locked) over their own gene spaces -> dimension-normalized RMS (norm/sqrt(N))
norm_pred = np.array([0.038, 0.324]); N_pred = 4354
norm_meas = np.array([74.456, 67.895]); N_meas = 2658
rms_pred = norm_pred / np.sqrt(N_pred); rms_meas = norm_meas / np.sqrt(N_meas)
frac_pred = [0.0, 0.0]; frac_meas = [1689 / N_meas * 100, 1864 / N_meas * 100]
fig, ax = plt.subplots(1, 2, figsize=(8.6, 4.0))
fig.suptitle("Figure 4 | The accompanying state is genome-wide (DAP-matched aerobic comparison)", fontsize=10.6, fontweight="bold")
a = ax[0]
a.bar(x - w/2, rms_pred, w, color="#c2c2c2", label="predicted (model)")
a.bar(x + w/2, rms_meas, w, color="#1f6feb", label="measured")
a.set_yscale("log"); a.set_xticks(x); a.set_xticklabels(conds)
a.set_ylabel("dimension-normalized RMS residual\n(YB1 − Δ$asd$ per gene, $asd$ excluded)", fontsize=8.3)
a.set_title("a  Per-gene residual (RMS)", loc="left", fontsize=9.4, fontweight="bold"); a.legend(fontsize=7.5)
for i in range(2):
    a.text(i - w/2, rms_pred[i], f"{rms_pred[i]:.4f}", ha="center", va="bottom", fontsize=6.6)
    a.text(i + w/2, rms_meas[i], f"{rms_meas[i]:.2f}", ha="center", va="bottom", fontsize=6.8)
b = ax[1]
b.bar(x - w/2, frac_pred, w, color="#c2c2c2", label="predicted (model)")
b.bar(x + w/2, frac_meas, w, color="#1f6feb", label="measured")
b.set_xticks(x); b.set_xticklabels(conds); b.set_ylim(0, 100)
b.set_ylabel("% exceeding |Δlog2FC|>0.5 effect-size\nthreshold (of 2,658; not per-gene sig.)", fontsize=8.0)
b.set_title("b  Fraction of trans-response genes", loc="left", fontsize=9.4, fontweight="bold"); b.legend(fontsize=7.5)
for i in range(2):
    b.text(i - w/2, frac_pred[i] + 2, "0%", ha="center", va="bottom", fontsize=7)
    b.text(i + w/2, frac_meas[i] + 2, f"{frac_meas[i]:.1f}%", ha="center", va="bottom", fontsize=7.2)
fig.text(0.5, -0.02, "Primary (DAP-matched aerobic): 63.5% of profiled genes exceeded the prespecified |Δlog2FC|>0.5 effect-size threshold beyond the edited $asd$ locus (not a per-gene significance test) vs ≈0 for the conservative propagator. "
         "The anaerobic 70.1% is NOT DAP-matched (YB1 −DAP vs Δasd +DAP) and is shown only as a sensitivity analysis. Model = comparator, not biological evidence.", ha="center", va="top", fontsize=6.3, wrap=True)
fig.tight_layout(rect=[0, 0.05, 1, 0.94])
fig.savefig(f"{OUT}/fig3_genomewide.png", dpi=200, bbox_inches="tight"); fig.savefig(f"{OUT}/fig3_genomewide.pdf", bbox_inches="tight")
print("saved fig3 (RMS + proportions)")
