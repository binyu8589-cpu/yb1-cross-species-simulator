"""network_integration_analysis.py — reproduces the Fig. 3 statistics exactly.
(1) Coordination: directional coherence vs 9999 random equal-size gene sets, plus-one
    permutation p, Benjamini-Hochberg corrected across the six strain x module tests.
(2) Condition-dependence: genotype x condition interaction on the invasion/motility score,
    exact permutation. Writes Source_Data_coordination_BHcorrected.csv."""
import numpy as np, pandas as pd, itertools, csv
TSV = "/Users/dryu/v2_data/wetlab/nanopore/v4_full_factorial/mrna_counts_v4_with_0525_0526.tsv"; PS = 1.0
OUT = "/Users/dryu/claude_paper1_rebuild"
SPI1 = ["invA","invB","invC","invE","invF","invG","invH","invI","invJ","sipA","sipB","sipC","sipD","sopB","sopD","sopE","sopE2","sopA","prgH","prgI","prgJ","prgK","hilA","hilC","hilD","iagB","sicA","sicP","sptP","spaO","spaP","spaQ","spaR","spaS"]
SPI2 = ["ssaB","ssaC","ssaD","ssaE","ssaG","ssaH","ssaJ","ssaK","ssaL","ssaM","ssaN","ssaO","ssaP","ssaQ","ssaR","ssaS","ssaT","ssaU","ssaV","sseA","sseB","sseC","sseD","sseE","sseF","sseG","sseI","sseJ","ssrA","ssrB","sopD2","pipB","pipB2","sifA","sifB"]
FLAG = ["flhD","flhC","fliA","fliC","fljB","fliD","fliS","fliT","flgB","flgC","flgD","flgE","flgF","flgG","flgH","flgI","flgK","flgL","flgM","flgN","fliF","fliG","fliM","fliN","flhA","flhB","motA","motB","cheA","cheW","cheY","cheZ","cheR","cheB","tar","tsr"]
MOD = {"SPI-1": SPI1, "SPI-2": SPI2, "Flagellar": FLAG}; PANEL = [x.lower() for x in SPI1 + FLAG]
df = pd.read_csv(TSV, sep="\t"); gn = df[df.columns[0]].astype(str).str.lower().values
cpm = {c[:-4]: dict(zip(gn, df[c].values)) for c in df.columns if c.endswith("_cpm")}
pool = lambda L: {g: np.mean([cpm[l][g] for l in L]) for g in gn}
def lfc(libs, ctrl): p = pool(libs); return {g: np.log2((p[g] + PS) / (ctrl[g] + PS)) for g in gn}
NOD = pool(["SL7207_ana_0509","SL7207_ana_0522"]); DAPa = pool(["SL7207_dap_aer_0520","SL7207_dap_aer_0525"])
YB1 = ["YB1_ana_v2","YB1_ana_0507","YB1_ana_0508","YB1_ana_clone_0516"]; PW = ["PW_ana_0515","PW_ana_0518","PW_ana_0521"]
Y, P = lfc(YB1, NOD), lfc(PW, NOD); det = [g for g in gn if NOD[g] > 0]; rng = np.random.default_rng(0); N = 9999
def coh(genes, L): v = np.array([L[g] for g in genes if g in L and NOD[g] > 0]); return max((v > 0).mean(), (v < 0).mean()), len(v)

print("=== (1) Coordination: 9999 plus-one permutation + Benjamini-Hochberg (6 tests) ===")
res = []
for s, L in [("YB1", Y), ("PW", P)]:
    for m, G in MOD.items():
        c, n = coh([x.lower() for x in G], L); ge = sum(coh(list(rng.choice(det, n, replace=False)), L)[0] >= c for _ in range(N))
        res.append([s, m, round(c, 3), n, (1 + ge) / (N + 1)])
pv = np.array([r[4] for r in res]); oo = np.argsort(pv)[::-1]; q = np.empty(len(pv)); prev = 1.0
for r, idx in enumerate(oo):
    rank = len(pv) - r; prev = min(prev, pv[idx] * len(pv) / rank); q[idx] = prev
rows = [("strain","module","coherence","n_genes","plus_one_p","BH_q","sig_BH_0.05")]
for r, qv in zip(res, q):
    rows.append((r[0], r[1], r[2], r[3], round(r[4], 4), round(qv, 4), qv < 0.05))
    print(f"  {r[0]:3s} {r[1]:10s} coh={r[2]:.2f}  p={r[4]:.4f}  BH_q={qv:.4f}  {'SIG' if qv < 0.05 else 'n.s.'}")
csv.writer(open(f"{OUT}/Source_Data_coordination_BHcorrected.csv", "w", newline="")).writerows(rows)

print("\n=== (2) genotype x condition interaction (invasion/motility score), exact permutation ===")
score = lambda lib, ctrl: float(np.mean([np.log2((cpm[lib][x] + PS) / (ctrl[x] + PS)) for x in PANEL if x in cpm[lib]]))
ya = [score(l, NOD) for l in YB1]; pa = [score(l, NOD) for l in PW]
yr = [score(l, DAPa) for l in ["YB1_aer_MinION","YB1_aer_0508","YB1_aer_clone_0516"]]; pr = [score(l, DAPa) for l in ["PW_aer_0515","PW_aer_0518","PW_aer_0521"]]
obs = (np.mean(pa) - np.mean(ya)) - (np.mean(pr) - np.mean(yr)); ana, aer = ya + pa, yr + pr; cnt = tot = 0
for ca in itertools.combinations(range(7), 4):
    ia = [ana[i] for i in ca]; pa2 = [ana[i] for i in range(7) if i not in ca]
    for cr in itertools.combinations(range(6), 3):
        ir = [aer[i] for i in cr]; pr2 = [aer[i] for i in range(6) if i not in cr]
        cnt += abs((np.mean(pa2) - np.mean(ia)) - (np.mean(pr2) - np.mean(ir))) >= abs(obs); tot += 1
print(f"  PW-YB1 anaerobic={np.mean(pa)-np.mean(ya):+.2f}, aerobic={np.mean(pr)-np.mean(yr):+.2f}; interaction={obs:+.2f}, two-sided p={cnt/tot:.4f} (n={tot})")
print("  (aerobic=+DAP / anaerobic=-DAP: oxygen and DAP co-vary; condition-dependent, not oxygen alone)")
