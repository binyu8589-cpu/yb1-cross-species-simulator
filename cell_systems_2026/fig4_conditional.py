"""Figure 4 — DAP-matched controls resolve the condition-dependent engineered state.
a DAP alone shifts virulence programs (pooled WT contrast). b per-LIBRARY SPI-1 invasion
score for YB1 (n=4) & PW (n=3) vs matched (-DAP) and wrong (+DAP) controls — points + mean±SD."""
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, numpy as np, pandas as pd
OUT = "/Users/dryu/claude_paper1_rebuild"; PS = 1.0
SPI1 = ["invA","invB","invC","invE","invF","invG","invH","invI","invJ","sipA","sipB","sipC","sipD",
        "sopB","sopD","sopE","sopE2","sopA","prgH","prgI","prgJ","prgK","hilA","hilC","hilD",
        "iagB","sicA","sicP","sptP","spaO","spaP","spaQ","spaR","spaS"]
SPI1 = [x.lower() for x in SPI1]
# --- panel a: DAP-effect (from Source_Data, pooled) ---
df5 = pd.read_csv(f"{OUT}/Source_Data/Fig5a_dap_matched_panels.csv")
def val(c, ctrl, p): r = df5[(df5.contrast == c) & (df5.control == ctrl) & (df5.panel == p)]; return float(r.mean_log2FC.values[0]) if len(r) else np.nan
panels = ["SPI-1 (invasion T3SS)", "SPI-2 (intracellular T3SS)", "Flagellar / chemotaxis", "DAP cascade"]; plab = ["SPI-1", "SPI-2", "Flagellar", "DAP\ncascade"]
# --- panel b: per-library SPI-1 from mRNA counts ---
TSV = "/Users/dryu/v2_data/wetlab/nanopore/v4_full_factorial/mrna_counts_v4_with_0525_0526.tsv"
dm = pd.read_csv(TSV, sep="\t"); gn = dm[dm.columns[0]].astype(str).str.lower().values
cpm = {c[:-4]: dict(zip(gn, dm[c].values)) for c in dm.columns if c.endswith("_cpm")}
def pool(libs): return {g: np.mean([cpm[l][g] for l in libs]) for g in gn}
def spi1(lib, ctrl): c = cpm[lib]; return float(np.mean([np.log2((c[x]+PS)/(ctrl[x]+PS)) for x in SPI1 if x in c]))
NOD = pool(["SL7207_ana_0509","SL7207_ana_0522"]); DAP = pool(["SL7207_dap_ana_0520","SL7207_dap_ana_0522"])
YB1 = ["YB1_ana_v2","YB1_ana_0507","YB1_ana_0508","YB1_ana_clone_0516"]; PW = ["PW_ana_0515","PW_ana_0518","PW_ana_0521"]
yb1_m = [spi1(l, NOD) for l in YB1]; yb1_w = [spi1(l, DAP) for l in YB1]
pw_m = [spi1(l, NOD) for l in PW]; pw_w = [spi1(l, DAP) for l in PW]

fig, ax = plt.subplots(1, 2, figsize=(9.6, 4.2))
fig.suptitle("Figure 5 | DAP-matched controls resolve the condition-dependent engineered state", fontsize=10.8, fontweight="bold")
# a
a = ax[0]; x = np.arange(4); w = 0.38
aer = [val("DAPeffect_aerobic","WT_aer_noDAP",p) for p in panels]; ana = [val("DAPeffect_anaerobic","WT_ana_noDAP",p) for p in panels]
a.bar(x - w/2, aer, w, color="#7fb3d5", label="aerobic"); a.bar(x + w/2, ana, w, color="#2e5f8a", label="anaerobic")
a.axhline(0, color="#444", lw=0.8); a.set_xticks(x); a.set_xticklabels(plab, fontsize=7.8)
a.set_ylabel("mean log2FC, +DAP vs −DAP (WT)", fontsize=8.5); a.set_title("a  DAP alone shifts virulence programs", loc="left", fontsize=9.0, fontweight="bold"); a.legend(fontsize=7.5)
a.text(0.02, 0.04, "pooled WT contrast; matched controls required", transform=a.transAxes, fontsize=6.4, va="bottom", bbox=dict(boxstyle="round", fc="white", ec="#ccc", alpha=0.9))
# b — per-library points + mean ± SD
b = ax[1]; rng = np.random.default_rng(0)
groups = [("YB1 (1×)", 0, yb1_m, yb1_w), ("PW (2×)", 1, pw_m, pw_w)]
for lab, xi, m, wv in groups:
    for vals, off, col, name in [(m, -w/2, "#1a7a4a", "matched −DAP"), (wv, +w/2, "#c9a9a9", "wrong +DAP")]:
        b.bar(xi + off, np.mean(vals), w, color=col, alpha=0.55, zorder=1)
        b.errorbar(xi + off, np.mean(vals), yerr=np.std(vals, ddof=1), color=col, lw=1.3, capsize=3, zorder=3)
        jit = (rng.random(len(vals)) - 0.5) * 0.18
        b.scatter(xi + off + jit, vals, s=24, color=col, edgecolor="white", lw=0.6, zorder=4)
b.axhline(0, color="#444", lw=0.8); b.set_xticks([0, 1]); b.set_xticklabels(["YB1 (1×)\n−DAP", "PW (2×)\n−DAP"], fontsize=8.2)
b.set_ylabel("SPI-1 invasion mean log2FC (per library)", fontsize=8.5)
b.set_title("b  Per-library readout: matched vs wrong control", loc="left", fontsize=9.0, fontweight="bold")
from matplotlib.patches import Patch
b.legend(handles=[Patch(color="#1a7a4a", label="matched −DAP control"), Patch(color="#c9a9a9", label="wrong +DAP control")], fontsize=7.0, loc="lower right")
b.text(0.02, 0.97, "matched −DAP: YB1 (1×) < PW (2×)", transform=b.transAxes, ha="left", fontsize=6.5, va="top", color="#555")
fig.text(0.5, -0.02, "DAP-matching is required (a); for the anaerobic −DAP engineered strains the matched control is −DAP, with per-library points shown (b). "
         "Group means ± SD; YB1 n=4, PW n=3 independent cultures.", ha="center", va="top", fontsize=6.4, wrap=True)
fig.tight_layout(rect=[0, 0.05, 1, 0.94])
fig.savefig(f"{OUT}/fig4_conditional.png", dpi=200, bbox_inches="tight"); fig.savefig(f"{OUT}/fig4_conditional.pdf", bbox_inches="tight")
print("saved fig4 (per-library points)")
