"""Fig 2 — forward-drive dosage and the invasion-motility operating point.
a operating point: YB1 (1x) vs PW (2x), both anaerobic -DAP, shared antisense; two-sided p primary.
b forward-drive (asd) verification from RNA-seq at the canonical engineered locus (SL1344_3506):
clean deletion in dAsd; PW (2x) ~1.7x YB1 (1x); native SL7207 reference. Underpowered (PW n=2),
no orthogonal (qPCR) confirmation. Source: Source_Data_asd_canonical_locus_coverage.csv."""
import numpy as np, pandas as pd, itertools
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
TSV = "/Users/dryu/v2_data/wetlab/nanopore/v4_full_factorial/mrna_counts_v4_with_0525_0526.tsv"
OUT = "/Users/dryu/claude_paper1_rebuild"; PS = 1.0
SPI1 = ["invA","invB","invC","invE","invF","invG","invH","invI","invJ","sipA","sipB","sipC","sipD","sopB","sopD","sopE","sopE2","sopA","prgH","prgI","prgJ","prgK","hilA","hilC","hilD","iagB","sicA","sicP","sptP","spaO","spaP","spaQ","spaR","spaS"]
FLAG = ["flhD","flhC","fliA","fliC","fljB","fliD","fliS","fliT","flgB","flgC","flgD","flgE","flgF","flgG","flgH","flgI","flgK","flgL","flgM","flgN","fliF","fliG","fliM","fliN","flhA","flhB","motA","motB","cheA","cheW","cheY","cheZ","cheR","cheB","tar","tsr"]
PANEL = [x.lower() for x in SPI1 + FLAG]
df = pd.read_csv(TSV, sep="\t"); gn = df[df.columns[0]].astype(str).str.lower().values
cpm = {c[:-4]: dict(zip(gn, df[c].values)) for c in df.columns if c.endswith("_cpm")}
def pool(libs): return {g: np.mean([cpm[l][g] for l in libs]) for g in gn}
def score(lib, ctrl): c = cpm[lib]; return float(np.mean([np.log2((c[x]+PS)/(ctrl[x]+PS)) for x in PANEL if x in c]))
DAP = pool(["SL7207_dap_ana_0520","SL7207_dap_ana_0522"]); NOD = pool(["SL7207_ana_0509","SL7207_ana_0522"])
GROUPS = {"Δ$asd$": (["dAsd_ana_0512","dAsd_ana_0514"], DAP, "+DAP", "0×"),
          "YB1":    (["YB1_ana_v2","YB1_ana_0507","YB1_ana_0508","YB1_ana_clone_0516"], NOD, "−DAP", "1×"),
          "PW":     (["PW_ana_0515","PW_ana_0518","PW_ana_0521"], NOD, "−DAP", "2×")}
order = list(GROUPS); COL = {"Δ$asd$":"#8a8a8a","YB1":"#1f6feb","PW":"#b03a2e"}
sc = {k: np.array([score(l, ctrl) for l in libs]) for k,(libs,ctrl,_,_) in GROUPS.items()}
def jtU(gv): return sum((x<y)+0.5*(x==y) for a in range(len(gv)) for b in range(a+1,len(gv)) for x in gv[a] for y in gv[b])
def jtp(gv):
    s=[len(x) for x in gv]; av=np.concatenate(gv); n=len(av); obs=jtU(gv); ge=tot=0
    for ai in itertools.combinations(range(n),s[0]):
        rem=[i for i in range(n) if i not in ai]
        for br in itertools.combinations(range(len(rem)),s[1]):
            bi=[rem[i] for i in br]; ci=[i for i in rem if i not in bi]; tot+=1; ge+=(jtU([av[list(ai)],av[bi],av[ci]])>=obs)
    return obs, sum(len(gv[a])*len(gv[b]) for a in range(3) for b in range(a+1,3)), ge/tot
U,Um,P = jtp([sc[k] for k in order])
y,pw = sc["YB1"], sc["PW"]; obsu=sum((a<b)+0.5*(a==b) for a in y for b in pw); pooled=np.concatenate([y,pw]); ge=tot=0
for comb in itertools.combinations(range(len(pooled)),len(y)):
    xs=pooled[list(comb)]; ys=pooled[[i for i in range(len(pooled)) if i not in comb]]; tot+=1; ge+=(sum((a<b)+0.5*(a==b) for a in xs for b in ys)>=obsu)
p1=ge/tot; p2=min(1,2*p1)
asd = pd.read_csv(f"{OUT}/Source_Data_asd_canonical_locus_coverage.csv")
fig, ax = plt.subplots(1, 2, figsize=(10.2, 4.8)); rng = np.random.default_rng(0)
fig.suptitle("Figure 2 | Forward-drive dosage and the invasion–motility operating point", fontsize=11.2, fontweight="bold")
# --- a: operating point ---
a = ax[0]; xs = {k:i for i,k in enumerate(order)}
for k in order:
    x0=xs[k]; vals=sc[k]; jit=(rng.random(len(vals))-0.5)*0.22
    a.scatter(x0+jit, vals, s=46, color=COL[k], edgecolor="white", lw=0.8, zorder=3)
    m=vals.mean(); sd=vals.std(ddof=1); a.hlines(m, x0-0.28, x0+0.28, color=COL[k], lw=2.6, zorder=4); a.errorbar(x0, m, yerr=sd, color=COL[k], lw=1.4, capsize=4, zorder=2)
a.plot(range(3), [sc[k].mean() for k in order], color="#999", lw=1.0, ls="--", zorder=1); a.axhline(0, color="#aaa", lw=0.8, ls=":")
yb=max(sc["YB1"].max(),sc["PW"].max())+0.45; a.plot([1,1,2,2],[yb,yb+0.16,yb+0.16,yb], color="#333", lw=1.1)
a.text(1.5, yb+0.22, f"−DAP, shared antisense\ntwo-sided $p$={p2:.3f}; complete separation", ha="center", fontsize=6.6, color="#222")
a.set_xticks(range(3)); a.set_xticklabels([f"{k}\n{GROUPS[k][3]} $pepT$·{GROUPS[k][2]}" for k in order], fontsize=8.2)
a.set_ylabel("invasion/motility log2FC vs DAP-matched SL7207 (per library)", fontsize=8.2)
a.set_xlim(-0.7, 2.6); a.set_ylim(-4.3, 3.5); a.set_title("a  Operating-point shift (all PW > all YB1)", loc="left", fontsize=9.4, fontweight="bold")
a.annotate("Δ$asd$: +DAP off-state anchor\n(no $asd$; not a 0× point)", xy=(0, sc["Δ$asd$"].mean()), xytext=(-0.4,-2.1), fontsize=6.3, color="#888", arrowprops=dict(arrowstyle="->", color="#bbb", lw=0.8))
a.text(0.97,0.97,f"three-state ordering\nJT U={U:.0f}/{Um}, $p$={P:.4f}", transform=a.transAxes, ha="right", va="top", fontsize=6.6, bbox=dict(boxstyle="round", fc="#f4f4f4", ec="#bbb"))
# --- b: forward-drive asd ---
b = ax[1]; ob = ["dAsd","YB1","PW","SL7207"]; LB={"dAsd":"Δ$asd$\n0× (del.)","YB1":"YB1\n1×","PW":"PW\n2×","SL7207":"SL7207\nnative"}; CB={"dAsd":"#8a8a8a","YB1":"#1f6feb","PW":"#b03a2e","SL7207":"#1a7a4a"}
mns={}
for i,s in enumerate(ob):
    v=asd[asd.strain==s].canonical_asd_cov_per_Mread.values; mns[s]=v.mean(); jit=(rng.random(len(v))-0.5)*0.2
    b.scatter(i+jit, v, s=46, color=CB[s], edgecolor="white", lw=0.8, zorder=3)
    b.hlines(v.mean(), i-0.28, i+0.28, color=CB[s], lw=2.6, zorder=4)
    if len(v)>1: b.errorbar(i, v.mean(), yerr=v.std(ddof=1), color=CB[s], lw=1.4, capsize=4, zorder=2)
b.set_xticks(range(4)); b.set_xticklabels([LB[s] for s in ob], fontsize=8.2)
b.set_ylabel("cassette $asd$ coverage at engineered locus\n(per Mread; SL1344_3506)", fontsize=8.2)
b.set_ylim(-0.2, 2.8); b.set_title("b  RNA-seq support: engineered-locus expression", loc="left", fontsize=9.2, fontweight="bold")
b.annotate(f"PW ≈ {mns['PW']/mns['YB1']:.1f}× YB1\n(one-sided $p$=0.133, PW n=2)", xy=(2, mns['PW']), xytext=(1.5, 2.55), fontsize=6.6, color="#222", ha="center")
b.text(0.5,0.02,"clean deletion (Δ$asd$ 0% covered); native SL7207 reference; underpowered, no qPCR", transform=b.transAxes, ha="center", va="bottom", fontsize=6.0, color="#666")
fig.text(0.5, -0.03, f"Primary matched dosage contrast (a): YB1 (1× $pepT$) vs PW (2×), both anaerobic −DAP, shared $sodA$-antisense — every PW culture > every YB1 (two-sided $p$={p2:.3f}); "
         f"genotype nested with sequencing date (Limitations). (b) the 2× cassette shows ~1.7× $asd$ at the engineered locus, consistent with stronger forward drive (RNA-seq; underpowered). "
         f"n: Δ$asd$ 2, YB1 4, PW 3 (a) / PW 2 (b; 0518 BAM not generated).", ha="center", va="top", fontsize=6.0, wrap=True)
fig.tight_layout(rect=[0,0.05,1,0.95])
fig.savefig(f"{OUT}/fig2_setpoint.png", dpi=200, bbox_inches="tight"); fig.savefig(f"{OUT}/fig2_setpoint.pdf", bbox_inches="tight")
print(f"saved fig2 (2-panel); asd PW/YB1={mns['PW']/mns['YB1']:.2f}x; two-sided p={p2:.3f}")
