"""tech_batch_audit.py — sequencing-date technical-batch sensitivity audit.
Genotype is nested in sequencing date (YB1 & PW sequenced on different days, <=2/day),
so we cannot statistically separate genotype from date; these checks instead test whether
the YB1<PW operating-point shift tracks any obvious technical axis (depth, detected genes,
WT temporal drift, global structure)."""
import numpy as np, pandas as pd
TSV = "/Users/dryu/v2_data/wetlab/nanopore/v4_full_factorial/mrna_counts_v4_with_0525_0526.tsv"; PS = 1.0
SPI1 = ["invA","invB","invC","invE","invF","invG","invH","invI","invJ","sipA","sipB","sipC","sipD","sopB","sopD","sopE","sopE2","sopA","prgH","prgI","prgJ","prgK","hilA","hilC","hilD","iagB","sicA","sicP","sptP","spaO","spaP","spaQ","spaR","spaS"]
FLAG = ["flhD","flhC","fliA","fliC","fljB","fliD","fliS","fliT","flgB","flgC","flgD","flgE","flgF","flgG","flgH","flgI","flgK","flgL","flgM","flgN","fliF","fliG","fliM","fliN","flhA","flhB","motA","motB","cheA","cheW","cheY","cheZ","cheR","cheB","tar","tsr"]
PANEL = [x.lower() for x in SPI1 + FLAG]
df = pd.read_csv(TSV, sep="\t"); gn = df[df.columns[0]].astype(str).str.lower().values
cpm = {c[:-4]: dict(zip(gn, df[c].values)) for c in df.columns if c.endswith("_cpm")}
ALL = {c[:-4]: np.array([cpm[c[:-4]][g] for g in gn]) for c in df.columns if c.endswith("_cpm")}
def pool(libs): return {g: np.mean([cpm[l][g] for l in libs]) for g in gn}
def score(lib, ctrl): c = cpm[lib]; return float(np.mean([np.log2((c[x]+PS)/(ctrl[x]+PS)) for x in PANEL if x in c]))
NOD = pool(["SL7207_ana_0509","SL7207_ana_0522"])
LIBS = {"YB1": ["YB1_ana_v2","YB1_ana_0507","YB1_ana_0508","YB1_ana_clone_0516"],
        "PW": ["PW_ana_0515","PW_ana_0518","PW_ana_0521"],
        "SL7207": ["SL7207_ana_0509","SL7207_ana_0522"], "dAsd": ["dAsd_ana_0512","dAsd_ana_0514"]}
date = lambda l: "".join([c for c in l.split("_")[-1] if c.isdigit()])[:4]

print("=== per-library QC (detected genes = CPM>0) + 70-gene score (vs −DAP WT) ===")
recs = []
for geno, libs in LIBS.items():
    for l in libs:
        det = int((ALL[l] > 0).sum()); s = score(l, NOD)
        recs.append((geno, l, date(l), det, round(s, 3)))
        print(f"  {geno:7s} {l:22s} seq_date={date(l)}  detected_genes={det:5d}  score={s:+.3f}")
R = pd.DataFrame(recs, columns=["geno","lib","date","detected","score"])

print("\n=== does the YB1<PW shift track depth/quality? (YB1+PW libraries) ===")
yp = R[R.geno.isin(["YB1","PW"])]
r = np.corrcoef(yp.detected, yp.score)[0, 1]
print(f"  corr(detected_genes, score) over YB1+PW = {r:+.3f}  (n={len(yp)})")
print(f"  detected genes: YB1 mean={R[R.geno=='YB1'].detected.mean():.0f}, PW mean={R[R.geno=='PW'].detected.mean():.0f} "
      f"(if PW≈YB1, the shift is not a depth artifact)")

print("\n=== WT temporal drift: SL7207 −DAP sequenced on different dates (0509 vs 0522) ===")
a, b = ALL["SL7207_ana_0509"], ALL["SL7207_ana_0522"]
la, lb = np.log2(a + PS), np.log2(b + PS); m = (a > 0) & (b > 0)
print(f"  SL7207_ana 0509 vs 0522: Pearson r(log2CPM)={np.corrcoef(la[m], lb[m])[0,1]:.3f} over {int(m.sum())} genes "
      f"(high r = little date drift in the WT control)")

print("\n=== PCA of anaerobic libraries (log2 CPM+1): genotype vs date structure ===")
samps = [l for libs in LIBS.values() for l in libs]
X = np.vstack([np.log2(ALL[l] + PS) for l in samps]); X = X - X.mean(0)
U, S, Vt = np.linalg.svd(X, full_matrices=False); pcs = U[:, :2] * S[:2]
ev = (S**2 / (S**2).sum())[:2] * 100
g_of = {l: g for g, ls in LIBS.items() for l in ls}
print(f"  PC1 {ev[0]:.0f}% var, PC2 {ev[1]:.0f}% var")
for i, l in enumerate(samps):
    print(f"    {g_of[l]:7s} {l:22s} date={date(l)}  PC1={pcs[i,0]:+6.2f}  PC2={pcs[i,1]:+6.2f}")
