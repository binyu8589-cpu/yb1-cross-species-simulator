"""downsample_audit.py — equal-depth downsampling control for the YB1<PW shift.
Downsample every anaerobic library to a common read count (multinomial), recompute the
70-gene invasion/motility score from downsampled CPM, test YB1<PW. Repeated over seeds.
Removes the library-depth covariate by construction."""
import numpy as np, pandas as pd
TSV = "/Users/dryu/v2_data/wetlab/nanopore/v4_full_factorial/mrna_counts_v4_with_0525_0526.tsv"
SPI1 = ["invA","invB","invC","invE","invF","invG","invH","invI","invJ","sipA","sipB","sipC","sipD","sopB","sopD","sopE","sopE2","sopA","prgH","prgI","prgJ","prgK","hilA","hilC","hilD","iagB","sicA","sicP","sptP","spaO","spaP","spaQ","spaR","spaS"]
FLAG = ["flhD","flhC","fliA","fliC","fljB","fliD","fliS","fliT","flgB","flgC","flgD","flgE","flgF","flgG","flgH","flgI","flgK","flgL","flgM","flgN","fliF","fliG","fliM","fliN","flhA","flhB","motA","motB","cheA","cheW","cheY","cheZ","cheR","cheB","tar","tsr"]
PANEL = set(x.lower() for x in SPI1 + FLAG)
df = pd.read_csv(TSV, sep="\t"); gn = df[df.columns[0]].astype(str).str.lower().values
cnt = {c[:-6]: df[c].values.astype(float) for c in df.columns if c.endswith("_count")}
pidx = np.array([i for i, g in enumerate(gn) if g in PANEL])
YB1 = ["YB1_ana_v2","YB1_ana_0507","YB1_ana_0508","YB1_ana_clone_0516"]
PW = ["PW_ana_0515","PW_ana_0518","PW_ana_0521"]; CTRL = ["SL7207_ana_0509","SL7207_ana_0522"]
print("library raw totals:", {l: int(cnt[l].sum()) for l in YB1 + PW + CTRL})

def ds(c, T, rng):
    n = int(c.sum())
    return rng.multinomial(T, c / n) if n > T else c
def run(Y, label, nseed=300):
    libs = Y + PW + CTRL; T = int(min(cnt[l].sum() for l in libs))
    ymeans, pmeans, sep = [], [], 0
    for s in range(nseed):
        rng = np.random.default_rng(s)
        d = {l: ds(cnt[l], T, rng) for l in libs}
        cpm = {l: d[l] / d[l].sum() * 1e6 for l in libs}
        ctrl = np.mean([cpm[c] for c in CTRL], 0)
        ys = [float(np.mean(np.log2((cpm[l][pidx] + 1) / (ctrl[pidx] + 1)))) for l in Y]
        ps = [float(np.mean(np.log2((cpm[l][pidx] + 1) / (ctrl[pidx] + 1)))) for l in PW]
        ymeans.append(np.mean(ys)); pmeans.append(np.mean(ps)); sep += (min(ps) > max(ys))
    print(f"\n[{label}]  common depth T={T:,} reads, {nseed} downsampling seeds")
    print(f"  YB1 mean = {np.mean(ymeans):+.2f} ± {np.std(ymeans):.2f}   PW mean = {np.mean(pmeans):+.2f} ± {np.std(pmeans):.2f}")
    print(f"  complete separation (every PW > every YB1): {sep}/{nseed} seeds = {100*sep/nseed:.0f}%")
run(YB1, "all YB1 (incl. low-depth v2)")
run(["YB1_ana_0507","YB1_ana_0508","YB1_ana_clone_0516"], "drop v2 outlier")
