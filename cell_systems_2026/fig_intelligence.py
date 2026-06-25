"""Figure 3 — the shifted state is coordinated and condition-dependent.
a coordination: directional coherence vs per-module null 95% bands; stars = BH-corrected (6 tests).
b forward-drive-dependent module shifts (YB1 1x -> PW 2x; Δasd gray off-state side anchor).
c condition-dependent (genotype x condition interaction p=0.0029; oxygen/DAP confounded)."""
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch, Rectangle
OUT = "/Users/dryu/claude_paper1_rebuild"; PS = 1.0
SPI1 = ["invA","invB","invC","invE","invF","invG","invH","invI","invJ","sipA","sipB","sipC","sipD","sopB","sopD","sopE","sopE2","sopA","prgH","prgI","prgJ","prgK","hilA","hilC","hilD","iagB","sicA","sicP","sptP","spaO","spaP","spaQ","spaR","spaS"]
SPI2 = ["ssaB","ssaC","ssaD","ssaE","ssaG","ssaH","ssaJ","ssaK","ssaL","ssaM","ssaN","ssaO","ssaP","ssaQ","ssaR","ssaS","ssaT","ssaU","ssaV","sseA","sseB","sseC","sseD","sseE","sseF","sseG","sseI","sseJ","ssrA","ssrB","sopD2","pipB","pipB2","sifA","sifB"]
FLAG = ["flhD","flhC","fliA","fliC","fljB","fliD","fliS","fliT","flgB","flgC","flgD","flgE","flgF","flgG","flgH","flgI","flgK","flgL","flgM","flgN","fliF","fliG","fliM","fliN","flhA","flhB","motA","motB","cheA","cheW","cheY","cheZ","cheR","cheB","tar","tsr"]
MOD = {"SPI-1": SPI1, "SPI-2": SPI2, "Flagellar": FLAG}
df = pd.read_csv("/Users/dryu/v2_data/wetlab/nanopore/v4_full_factorial/mrna_counts_v4_with_0525_0526.tsv", sep="\t")
gn = df[df.columns[0]].astype(str).str.lower().values; cpm = {c[:-4]: dict(zip(gn, df[c].values)) for c in df.columns if c.endswith("_cpm")}
pool = lambda L: {g: np.mean([cpm[l][g] for l in L]) for g in gn}
def lfc(libs, ctrl):
    p = pool(libs); return {g: np.log2((p[g] + PS) / (ctrl[g] + PS)) for g in gn}
NOD = pool(["SL7207_ana_0509","SL7207_ana_0522"]); DAP = pool(["SL7207_dap_ana_0520","SL7207_dap_ana_0522"]); NODa = pool(["SL7207_dap_aer_0520","SL7207_dap_aer_0525"])
dA = lfc(["dAsd_ana_0512","dAsd_ana_0514"], DAP); Y = lfc(["YB1_ana_v2","YB1_ana_0507","YB1_ana_0508","YB1_ana_clone_0516"], NOD); P = lfc(["PW_ana_0515","PW_ana_0518","PW_ana_0521"], NOD)
Ya = lfc(["YB1_aer_MinION","YB1_aer_0508","YB1_aer_clone_0516"], NODa); Pa = lfc(["PW_aer_0515","PW_aer_0518","PW_aer_0521"], NODa)
det = [g for g in gn if NOD[g] > 0]; rng = np.random.default_rng(0); N = 9999
def coh(genes, L):
    v = np.array([L[g] for g in genes if g in L and NOD[g] > 0]); return max((v > 0).mean(), (v < 0).mean()), len(v)
def nulldist(n, L): return np.array([coh(list(rng.choice(det, n, replace=False)), L)[0] for _ in range(N)])
mm = lambda genes, L: float(np.mean([L[g] for g in genes if g in L]))
mods = list(MOD); cols = {"SPI-1": "#c0392b", "SPI-2": "#1a7a4a", "Flagellar": "#1f6feb"}; BL, RD = "#1f6feb", "#c0392b"
# coherence + raw plus-one p per (strain,module); BH over 6 tests; null band per module (PW differential)
obs = {}; keys = []
for s, L in [("YB1", Y), ("PW", P)]:
    for m in mods:
        c, n = coh([g.lower() for g in MOD[m]], L); nd = nulldist(n, L); p = (1 + (nd >= c).sum()) / (N + 1)
        obs[(s, m)] = (c, p); keys.append((s, m))
pv = np.array([obs[k][1] for k in keys]); oo = np.argsort(pv)[::-1]; q = np.empty(len(pv)); prev = 1.0
for r, idx in enumerate(oo):
    rank = len(pv) - r; prev = min(prev, pv[idx] * len(pv) / rank); q[idx] = prev
qd = {k: q[i] for i, k in enumerate(keys)}
band = {m: nulldist(coh([g.lower() for g in MOD[m]], P)[1], P) for m in mods}
star = lambda qq: "***" if qq < 1e-3 else ("**" if qq < 1e-2 else ("*" if qq < 0.05 else "n.s."))
plt.rcParams.update({"font.size": 9})
fig, ax = plt.subplots(1, 3, figsize=(13.4, 4.8)); x = np.arange(3); w = 0.38
fig.suptitle("Figure 3 | The shifted state is coordinated and condition-dependent", fontsize=11.5, fontweight="bold", y=1.0)
# a — coordination with per-module null 95% bands + BH stars
a = ax[0]
for i, m in enumerate(mods):
    lo, hi = np.percentile(band[m], 2.5), np.percentile(band[m], 97.5)
    a.add_patch(Rectangle((i - 0.46, lo), 0.92, hi - lo, fc="#e6e6e6", ec="none", zorder=0))
    a.hlines(np.percentile(band[m], 50), i - 0.46, i + 0.46, color="#bbb", lw=0.8, zorder=1)
for off, s, col in [(-w/2, "YB1", BL), (w/2, "PW", RD)]:
    for i, m in enumerate(mods):
        c, _ = obs[(s, m)]; a.bar(i + off, c, w, color=col, zorder=2)
        a.text(i + off, c + 0.015, star(qd[(s, m)]), ha="center", va="bottom", fontsize=8, color=col if qd[(s, m)] < 0.05 else "#999", fontweight="bold")
a.set_xticks(x); a.set_xticklabels(mods, fontsize=9); a.set_ylim(0, 1.14); a.set_ylabel("directional coherence", fontsize=9.5)
a.set_title("a  Coordination (selective)", loc="left", fontsize=10, fontweight="bold")
a.legend(handles=[Patch(color=BL, label="YB1 (1×)"), Patch(color=RD, label="PW (2×)"), Patch(fc="#e6e6e6", label="null 95% (random)")], fontsize=7.2, loc="upper center", ncol=3, frameon=False)
a.text(0.98, 0.02, "stars = BH-corrected (6 tests)\n*** q<0.001  n.s. not sig.", transform=a.transAxes, ha="right", va="bottom", fontsize=6.2, color="#555")
# b — forward-drive-dependent module shifts
b = ax[1]
for m in mods:
    gl = [g.lower() for g in MOD[m]]
    b.plot([1, 2], [mm(gl, Y), mm(gl, P)], "-o", color=cols[m], lw=2.2, ms=7, zorder=3, label=m)
    b.plot([0], [mm(gl, dA)], "o", mfc="white", mec="#8a8a8a", mew=1.4, ms=7, zorder=2)
b.axvspan(-0.4, 0.4, color="#f2f2f2", zorder=0); b.axhline(0, color="#bbb", ls=":", lw=0.9)
b.set_xticks([0, 1, 2]); b.set_xticklabels(["Δ$asd$\n(off-state\nanchor)", "YB1\n1×", "PW\n2×"], fontsize=8.3); b.set_xlim(-0.5, 2.4)
b.set_ylabel("module-mean log2FC", fontsize=9.5); b.set_title("b  Forward-drive-dependent module shifts", loc="left", fontsize=10, fontweight="bold"); b.legend(fontsize=8, loc="lower right", frameon=False)
b.text(0.5, 0.02, "dosage comparison = YB1→PW; Δasd is +DAP, not a 0× point", transform=b.transAxes, ha="center", va="bottom", fontsize=6.3, color="#555")
# c — condition-dependent
c = ax[2]
ana = [mm([g.lower() for g in MOD[m]], P) - mm([g.lower() for g in MOD[m]], Y) for m in mods]
aer = [mm([g.lower() for g in MOD[m]], Pa) - mm([g.lower() for g in MOD[m]], Ya) for m in mods]
c.bar(x - w/2, ana, w, color=[cols[m] for m in mods], zorder=2); c.bar(x + w/2, aer, w, color=[cols[m] for m in mods], alpha=0.4, hatch="//", edgecolor="white", zorder=2)
c.axhline(0, color="#444", lw=0.9); c.set_xticks(x); c.set_xticklabels(mods, fontsize=9)
c.set_ylabel("dosage effect  PW − YB1  (log2FC)", fontsize=9.5); c.set_title("c  Condition-dependent (genotype×condition)", loc="left", fontsize=10, fontweight="bold")
c.legend(handles=[Patch(fc="#777", label="anaerobic (−DAP)"), Patch(fc="#777", alpha=0.4, hatch="//", label="aerobic (+DAP)")], fontsize=7.5, loc="upper right", frameon=False)
c.text(0.02, 0.02, "interaction p=0.0029; oxygen & DAP\nco-vary — not oxygen alone", transform=c.transAxes, ha="left", va="bottom", fontsize=6.3, color="#555")
fig.text(0.5, -0.015, "Coherence is selective — only PW SPI-1/flagellar and YB1 SPI-2 exceed the null after BH correction (a); the dosage comparison YB1(1×)→PW(2×) raises the "
         "modules with Δasd as an off-state anchor (b); the dosage effect is condition-dependent (c), but oxygen and DAP status co-vary.", ha="center", va="top", fontsize=6.8, wrap=True)
fig.tight_layout(rect=[0, 0.03, 1, 0.95])
fig.savefig(f"{OUT}/fig_intelligence.png", dpi=200, bbox_inches="tight"); fig.savefig(f"{OUT}/fig_intelligence.pdf", bbox_inches="tight")
print("saved fig3 (BH + null bands); BH q:", {k: round(qd[k],4) for k in keys})
