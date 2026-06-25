"""Figure 5 — Post-transcriptional composition (per-library points + SD).
6 regulatory sRNAs, Δasd (n=2) vs YB1 (n=3) aerobic, vs pooled SL7207 aerobic.
MicF/RprA/GcvB/MicL reverse direction. Model attribution -> Supplement (not on main fig)."""
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, numpy as np, pandas as pd
from matplotlib.patches import Patch
OUT = "/Users/dryu/claude_paper1_rebuild"; PS = 1.0
SRNA = "/Users/dryu/v2_data/wetlab/nanopore/v4_full_factorial/srna_counts_v4_with_0518_0520.tsv"
df = pd.read_csv(SRNA, sep="\t"); df["_n"] = df[df.columns[0]].astype(str).str.lower()
def cpm(lib, s):
    r = df[df["_n"] == s.lower()]; col = f"{lib}_cpm"
    return float(r[col].values[0]) if (len(r) and col in df.columns) else np.nan
srnas = ["MicC", "MicF", "RprA", "GcvB", "MicL", "RybB"]
WT = ["SL7207_aer_0507", "SL7207_aer_0509"]; DASD = ["dAsd_aer_0511", "dAsd_aer_0514"]
YB1 = ["YB1_aer_MinION", "YB1_aer_0508", "YB1_aer_clone_0516"]
def lfc(libs, s):
    wp = np.mean([cpm(w, s) for w in WT]); return [np.log2((cpm(l, s) + PS) / (wp + PS)) for l in libs]
fig, ax = plt.subplots(figsize=(7.4, 4.5)); x = np.arange(len(srnas)); w = 0.38
for i, s in enumerate(srnas):
    dv, yv = lfc(DASD, s), lfc(YB1, s)
    for vals, off, col in [(dv, -w/2, "#8a8a8a"), (yv, +w/2, "#1f6feb")]:
        ax.bar(i + off, np.mean(vals), w, color=col, alpha=0.5, zorder=1)
        ax.errorbar(i + off, np.mean(vals), yerr=np.std(vals, ddof=1), color=col, lw=1.2, capsize=3, zorder=3)
        jit = (np.random.default_rng(i).random(len(vals)) - 0.5) * 0.16
        ax.scatter(i + off + jit, vals, s=22, color=col, edgecolor="white", lw=0.5, zorder=4)
    if np.mean(dv) * np.mean(yv) < 0 and abs(np.mean(yv)) > 0.5:
        ax.text(i, max(np.mean(dv), np.mean(yv)) + 0.45, "↕", ha="center", fontsize=12, color="#b03a2e")
ax.axhline(0, color="#444", lw=0.8); ax.set_xticks(x); ax.set_xticklabels(srnas, fontsize=9.5)
ax.set_ylabel("log2FC vs SL7207 (aerobic, per library)", fontsize=9)
ax.set_title("Figure 6 | A regulatory small-RNA layer of the shifted state", fontsize=10.3, fontweight="bold")
ax.legend(handles=[Patch(color="#8a8a8a", label="Δ$asd$ (n=2)"), Patch(color="#1f6feb", label="YB1 (n=3)")], fontsize=8, loc="lower right")
fig.text(0.5, -0.02, "MicF, RprA and GcvB (↕) change in opposite directions between Δ$asd$ and YB1; MicL/RybB show strain-specific, non-reversing changes — a distinct regulated state, not a "
         "DAP-deficient copy. Reversals of MicF/RprA/GcvB hold with the early MinION library excluded (n=2); GcvB is a group-mean reversal (1 of 3 YB1 replicates does not reverse). Per-library points; means ± SD.", ha="center", va="top", fontsize=6.0, wrap=True)
fig.tight_layout(rect=[0, 0.05, 1, 1])
fig.savefig(f"{OUT}/fig5_srna.png", dpi=200, bbox_inches="tight"); fig.savefig(f"{OUT}/fig5_srna.pdf", bbox_inches="tight")
print("saved fig5 (per-library points)")
