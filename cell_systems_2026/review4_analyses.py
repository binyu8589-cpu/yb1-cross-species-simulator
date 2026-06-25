"""review4_analyses.py — foundational analyses for the DRAFT-4 review:
(1) asd transcript per library: does PW (2x) > YB1 (1x) anaerobically? (forward-drive verification)
(2) PW vs YB1 genome-wide differential (anaerobic, both -DAP) — the DIRECT dosage contrast
(3) coordination: 9999 plus-one permutation + BH multiple-comparison correction (6 tests)
Writes formal result files to the rebuild dir."""
import numpy as np, pandas as pd, itertools, csv, json
TSV = "/Users/dryu/v2_data/wetlab/nanopore/v4_full_factorial/mrna_counts_v4_with_0525_0526.tsv"; PS = 1.0; OUT = "/Users/dryu/claude_paper1_rebuild"
SPI1 = ["invA","invB","invC","invE","invF","invG","invH","invI","invJ","sipA","sipB","sipC","sipD","sopB","sopD","sopE","sopE2","sopA","prgH","prgI","prgJ","prgK","hilA","hilC","hilD","iagB","sicA","sicP","sptP","spaO","spaP","spaQ","spaR","spaS"]
SPI2 = ["ssaB","ssaC","ssaD","ssaE","ssaG","ssaH","ssaJ","ssaK","ssaL","ssaM","ssaN","ssaO","ssaP","ssaQ","ssaR","ssaS","ssaT","ssaU","ssaV","sseA","sseB","sseC","sseD","sseE","sseF","sseG","sseI","sseJ","ssrA","ssrB","sopD2","pipB","pipB2","sifA","sifB"]
FLAG = ["flhD","flhC","fliA","fliC","fljB","fliD","fliS","fliT","flgB","flgC","flgD","flgE","flgF","flgG","flgH","flgI","flgK","flgL","flgM","flgN","fliF","fliG","fliM","fliN","flhA","flhB","motA","motB","cheA","cheW","cheY","cheZ","cheR","cheB","tar","tsr"]
MOD = {"SPI-1": SPI1, "SPI-2": SPI2, "Flagellar": FLAG}
df = pd.read_csv(TSV, sep="\t"); gn = df[df.columns[0]].astype(str).str.lower().values; cpm = {c[:-4]: dict(zip(gn, df[c].values)) for c in df.columns if c.endswith("_cpm")}
YB1L = ["YB1_ana_v2","YB1_ana_0507","YB1_ana_0508","YB1_ana_clone_0516"]; PWL = ["PW_ana_0515","PW_ana_0518","PW_ana_0521"]
dAL = ["dAsd_ana_0512","dAsd_ana_0514"]; WTL = ["SL7207_ana_0509","SL7207_ana_0522"]
pool = lambda L: {g: np.mean([cpm[l][g] for l in L]) for g in gn}

print("=== (1) asd transcript per library (forward-drive verification) ===")
rows = [("library","strain","asd_CPM")]
for L, s in [(WTL,"SL7207"),(dAL,"Δasd"),(YB1L,"YB1(1×)"),(PWL,"PW(2×)")]:
    for lib in L: rows.append((lib, s, round(cpm[lib]["asd"],1))); print(f"  {s:9s} {lib:22s} asd CPM = {cpm[lib]['asd']:.1f}")
yb1_asd = [cpm[l]["asd"] for l in YB1L]; pw_asd = [cpm[l]["asd"] for l in PWL]
lfc_asd = np.log2((np.mean(pw_asd)+PS)/(np.mean(yb1_asd)+PS))
# exact rank permutation YB1 vs PW (one/two-sided)
allv = yb1_asd+pw_asd; obs = np.mean(pw_asd)-np.mean(yb1_asd); ge=t=0
for c in itertools.combinations(range(7),4):
    g1=[allv[i] for i in c]; g2=[allv[i] for i in range(7) if i not in c]; d=np.mean(g2)-np.mean(g1); t+=1; ge+=(abs(d)>=abs(obs))
print(f"  YB1 mean={np.mean(yb1_asd):.1f}  PW mean={np.mean(pw_asd):.1f}  log2FC(PW/YB1)={lfc_asd:+.2f}  two-sided perm p={ge/t:.3f}")
csv.writer(open(f"{OUT}/Source_Data_asd_transcript_per_library.csv","w",newline="")).writerows(rows)

print("\n=== (2) PW vs YB1 genome-wide differential (anaerobic, both −DAP, DAP-matched dosage contrast) ===")
Yp, Pp = pool(YB1L), pool(PWL); det = [g for g in gn if (Yp[g]>0 or Pp[g]>0) and g!="asd"]
d = {g: np.log2((Pp[g]+PS)/(Yp[g]+PS)) for g in det}
n_big = sum(abs(v)>0.5 for v in d.values()); frac = 100*n_big/len(det)
print(f"  genes |Δlog2FC(PW−YB1)|>0.5 (asd excluded): {n_big}/{len(det)} = {frac:.1f}%")
# compare to YB1 vs Δasd (the broader engineered-state, for context) — aerobic DAP-matched already 63.5%
print(f"  (context: YB1 vs Δasd aerobic DAP-matched = 63.5%; that is the broader engineered state, a DIFFERENT contrast)")

print("\n=== (3) coordination: 9999 plus-one + BH correction (6 tests) ===")
NOD = pool(WTL); rng = np.random.default_rng(0); detC = [g for g in gn if NOD[g]>0]
def lfc(libs,ctrl): p=pool(libs); return {g:np.log2((p[g]+PS)/(ctrl[g]+PS)) for g in gn}
Y, P = lfc(YB1L,NOD), lfc(PWL,NOD)
def coh(genes,L): v=np.array([L[g] for g in genes if g in L and NOD[g]>0]); return max((v>0).mean(),(v<0).mean()), len(v)
res=[]; N=9999
for s,L in [("YB1",Y),("PW",P)]:
    for m,G in MOD.items():
        c,n=coh([x.lower() for x in G],L); ge=sum(coh(list(rng.choice(detC,n,replace=False)),L)[0]>=c for _ in range(N)); p=(1+ge)/(N+1)
        res.append([s,m,round(c,3),n,p])
ps=[r[4] for r in res]; order=np.argsort(ps); m=len(ps); q=[0]*m  # Benjamini-Hochberg
for rank,idx in enumerate(order[::-1]): i=m-1-rank; q[order[i]] = min((ps[order[i]]*m/(i+1)), (q[order[i+1]] if i+1<m else 1))
# simpler BH:
qs=np.array(ps)*m/(np.argsort(np.argsort(ps))+1); qs=np.minimum.accumulate(qs[np.argsort(ps)][::-1])[::-1]; qord=np.empty(m); qord[np.argsort(ps)]=qs
rows3=[("strain","module","coherence","n_genes","raw_p","BH_q","sig_BH_0.05")]
for r,qv in zip(res,qord): rows3.append((r[0],r[1],r[2],r[3],round(r[4],4),round(min(qv,1),4),qv<0.05)); print(f"  {r[0]:3s} {r[1]:10s} coh={r[2]:.2f}  p={r[4]:.4f}  BH_q={min(qv,1):.4f}  {'SIG' if qv<0.05 else 'ns'}")
csv.writer(open(f"{OUT}/Source_Data_coordination_BHcorrected.csv","w",newline="")).writerows(rows3)
print("\n  -> formal result files written: Source_Data_asd_transcript_per_library.csv, Source_Data_coordination_BHcorrected.csv")
