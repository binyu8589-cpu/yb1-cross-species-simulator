"""Supplementary Figures S1 + S2.
S1: operating-point ordering is consistent for SPI-1-only and flagellar-only sub-panels (anaerobic, DAP-matched).
S2: aerobic (all +DAP) comparison — the dosage shift is condition-specific (YB1 ≈ PW aerobically)."""
import numpy as np, pandas as pd, itertools
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
OUT = "/Users/dryu/claude_paper1_rebuild"; PS = 1.0
SPI1 = [x.lower() for x in ["invA","invB","invC","invE","invF","invG","invH","invI","invJ","sipA","sipB","sipC","sipD","sopB","sopD","sopE","sopE2","sopA","prgH","prgI","prgJ","prgK","hilA","hilC","hilD","iagB","sicA","sicP","sptP","spaO","spaP","spaQ","spaR","spaS"]]
FLAG = [x.lower() for x in ["flhD","flhC","fliA","fliC","fljB","fliD","fliS","fliT","flgB","flgC","flgD","flgE","flgF","flgG","flgH","flgI","flgK","flgL","flgM","flgN","fliF","fliG","fliM","fliN","flhA","flhB","motA","motB","cheA","cheW","cheY","cheZ","cheR","cheB","tar","tsr"]]
df = pd.read_csv("/Users/dryu/v2_data/wetlab/nanopore/v4_full_factorial/mrna_counts_v4_with_0525_0526.tsv", sep="\t")
gn = df[df.columns[0]].astype(str).str.lower().values; cpm = {c[:-4]: dict(zip(gn, df[c].values)) for c in df.columns if c.endswith("_cpm")}
pool = lambda L: {g: np.mean([cpm[l][g] for l in L]) for g in gn}
score = lambda lib, ctrl, panel: float(np.mean([np.log2((cpm[lib][x] + PS) / (ctrl[x] + PS)) for x in panel if x in cpm[lib]]))
def jt2(y, p):
    U = lambda a, b: sum((x < z) + 0.5*(x == z) for x in a for z in b); obs = U(y, p); pl = np.concatenate([y, p]); ge = t = 0
    for c in itertools.combinations(range(len(pl)), len(y)):
        xs = pl[list(c)]; zs = pl[[i for i in range(len(pl)) if i not in c]]; t += 1; ge += (U(xs, zs) >= obs)
    return ge / t
COL = {"Δ$asd$": "#8a8a8a", "YB1": "#1f6feb", "PW": "#b03a2e"}; rng = np.random.default_rng(0)

# ---------- S1 ----------
NOD = pool(["SL7207_ana_0509","SL7207_ana_0522"]); DAP = pool(["SL7207_dap_ana_0520","SL7207_dap_ana_0522"])
G = {"Δ$asd$": (["dAsd_ana_0512","dAsd_ana_0514"], DAP), "YB1": (["YB1_ana_v2","YB1_ana_0507","YB1_ana_0508","YB1_ana_clone_0516"], NOD), "PW": (["PW_ana_0515","PW_ana_0518","PW_ana_0521"], NOD)}
fig, ax = plt.subplots(1, 2, figsize=(8.4, 4.0)); fig.suptitle("Figure S1 | Operating-point ordering is consistent across sub-panels (anaerobic, DAP-matched)", fontsize=9.8, fontweight="bold")
for k, (panel, pname) in enumerate([(SPI1, "SPI-1 invasion (34 genes)"), (FLAG, "Flagellar/chemotaxis (36 genes)")]):
    a = ax[k]; order = list(G); sc = {g: np.array([score(l, ctrl, panel) for l in libs]) for g, (libs, ctrl) in G.items()}
    for i, g in enumerate(order):
        v = sc[g]; a.scatter(i + (rng.random(len(v))-0.5)*0.2, v, s=34, color=COL[g], edgecolor="white", lw=0.7, zorder=3)
        a.hlines(v.mean(), i-0.27, i+0.27, color=COL[g], lw=2.3); a.errorbar(i, v.mean(), yerr=v.std(ddof=1), color=COL[g], lw=1.2, capsize=3)
    a.plot(range(3), [sc[g].mean() for g in order], "--", color="#999", lw=0.9)
    a.axhline(0, color="#aaa", lw=0.7, ls=":"); a.set_xticks(range(3)); a.set_xticklabels(order, fontsize=9)
    a.set_title(f"{'ab'[k]}  {pname}", loc="left", fontsize=8.8, fontweight="bold"); a.set_ylabel("mean log2FC vs matched SL7207", fontsize=8)
    p = jt2(sc["YB1"], sc["PW"]); a.text(0.5, 0.96, f"YB1<PW one-sided p={p:.3f}", transform=a.transAxes, ha="center", va="top", fontsize=7)
fig.tight_layout(rect=[0, 0, 1, 0.93]); fig.savefig(f"{OUT}/figS1_subpanels.png", dpi=200, bbox_inches="tight"); fig.savefig(f"{OUT}/figS1_subpanels.pdf", bbox_inches="tight"); print("saved figS1")

# ---------- S2 (aerobic, all +DAP) ----------
PANEL = SPI1 + FLAG; DAPa = pool(["SL7207_dap_aer_0520","SL7207_dap_aer_0525"])
Ga = {"Δ$asd$": ["dAsd_aer_0511","dAsd_aer_0514","dAsd_aer_0526"], "YB1": ["YB1_aer_MinION","YB1_aer_0508","YB1_aer_clone_0516"], "PW": ["PW_aer_0515","PW_aer_0518","PW_aer_0521"]}
fig, ax = plt.subplots(figsize=(5.0, 4.2)); order = list(Ga); sc = {g: np.array([score(l, DAPa, PANEL) for l in libs]) for g, libs in Ga.items()}
for i, g in enumerate(order):
    v = sc[g]; ax.scatter(i + (rng.random(len(v))-0.5)*0.2, v, s=40, color=COL[g], edgecolor="white", lw=0.7, zorder=3)
    ax.hlines(v.mean(), i-0.27, i+0.27, color=COL[g], lw=2.4); ax.errorbar(i, v.mean(), yerr=v.std(ddof=1), color=COL[g], lw=1.3, capsize=4)
ax.axhline(0, color="#aaa", lw=0.7, ls=":"); ax.set_xticks(range(3)); ax.set_xticklabels([f"{g}\n+DAP" for g in order], fontsize=9)
ax.set_ylabel("invasion/motility log2FC vs SL7207+DAP (aerobic)", fontsize=8.5)
ax.set_title("Figure S2 | Aerobic (all +DAP): the dosage shift is\ncondition-specific (YB1 ≈ PW)", fontsize=9.6, fontweight="bold")
fig.text(0.5, -0.02, f"Aerobically (all +DAP, fully matched), YB1 ({sc['YB1'].mean():+.2f}) ≈ PW ({sc['PW'].mean():+.2f}) — the YB1<PW dosage shift is "
         "specific to anaerobic conditions, where pepT is induced. n=3 each.", ha="center", va="top", fontsize=6.4, wrap=True)
fig.tight_layout(rect=[0, 0.05, 1, 1]); fig.savefig(f"{OUT}/figS2_aerobic.png", dpi=200, bbox_inches="tight"); fig.savefig(f"{OUT}/figS2_aerobic.pdf", bbox_inches="tight"); print("saved figS2")
