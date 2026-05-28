"""per_pathway_baselines.py — extend run_log2fc_cmp.py:
   compute v5 / GENIE3 / mean-floor log2FC cosine PER PATHWAY (SPI-1, Flagellar,
   Chemotaxis, SPI-2). Same shared-mask protocol, same informative-gene filter
   (|measured log2FC| > 0.5), additional pathway-membership filter."""
import os as _os
_REPO = _os.environ.get("YB1_REPO", _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
_DATA = _os.environ.get("YB1_DATA", _os.path.join(_REPO, "data", "processed"))
_REF  = _os.environ.get("YB1_REF",  _os.path.join(_REPO, "data", "reference"))
_CKPT = _os.environ.get("YB1_CKPT", _os.path.join(_REPO, "checkpoints"))
import sys, json, numpy as np, pandas as pd, torch
import torch.nn.functional as F
from joblib import Parallel, delayed
from sklearn.ensemble import ExtraTreesRegressor
sys.path.insert(0, _os.path.join(_REPO, "model"))
from train_stage1_ecoli import Stage1Model
from train_stage3_inhouse import map_inhouse_to_stage_vocab, build_sample_vectors

CKPT=_os.path.join(_CKPT, "checkpoints_stage3v2_1_cond_S42", "best.pt")
COMB=_os.path.join(_DATA, "combined_counts_v1.tsv"); PARQ=_os.path.join(_DATA, "master_expression_matrix_v2.parquet"); VOCAB=_os.path.join(_REF, "sl1344_vocab.tsv")
YB1=["YB1_aer_MinION","YB1_aer_0508","YB1_ana_v2","YB1_ana_0507","YB1_ana_0508","YB1_aer_clone_0516","YB1_ana_clone_0516"]
WT=["SL7207_aer_0507","SL7207_aer_0509","SL7207_ana_0509"]
YB1_AER=[0,1,5]; YB1_ANA=[2,3,4,6]; WT_AER=[0,1]; WT_ANA=[2]
dev="cpu"; N_EST=10; N_SUB=700

# pathway gene-name sets (mirror eval_pathway_no_context_and_log2fc.py)
SPI1 = {"invG","invE","invF","invH","invA","invB","invC","invI","invJ","sipA","sipB","sipC","sipD","sptP","sopB","sopE","sopE2","sopD","hilA","hilC","hilD","prgH","prgJ","prgK","spaN","spaO","spaP"}
FLAG = {"fliA","fliC","fliD","fliE","fliF","fliG","fliH","fliI","fliJ","fliK","fliL","fliM","fliN","fliP","fliQ","fliR","fliS","flgA","flgB","flgC","flgD","flgE","flgF","flgG","flgH","flgI","flgJ","flgK","flgL","flgM","flhA","flhB","flhC","flhD","flhE","motA","motB"}
CHEMO = {"cheA","cheB","cheR","cheW","cheY","cheZ","tar","tsr","trg","tap"}
SPI2  = {"siiA","siiB","siiC","siiD","siiE","siiF","ssaA","ssaB","ssaC","ssaD","ssaE","ssaG","ssaH","ssaI","ssaJ","ssaK","ssaL","ssaM","ssaN","ssaO","ssaP","ssaQ","ssaR","ssaS","ssaT","ssaU","ssaV"}
PATHWAYS = {"SPI-1":SPI1, "Flagellar":FLAG, "Chemotaxis":CHEMO, "SPI-2":SPI2}

ck=torch.load(CKPT,map_location=dev,weights_only=False)
nm,ns=ck["n_mrna"],ck["n_srna"]; nt=nm+ns
df=pd.read_csv(COMB,sep="\t")
mp=map_inhouse_to_stage_vocab(df["name"].tolist(),ck["mrna_names"],ck["srna_names"])
eY,hY=build_sample_vectors(df,mp,YB1,nt,dev); eW,hW=build_sample_vectors(df,mp,WT,nt,dev)
print(f"YB1 {tuple(eY.shape)} WT {tuple(eW.shape)}",flush=True)

# pathway index masks (over the n_total vocab; mRNA positions only)
def names_to_mask(name_set):
    m = torch.zeros(nt, dtype=torch.bool)
    name_to_idx = {n: i for i, n in enumerate(ck["mrna_names"])}
    hit = miss = 0
    for n in name_set:
        if n in name_to_idx:
            m[name_to_idx[n]] = True; hit += 1
        else:
            miss += 1
    return m, hit, miss

pmask = {}
for p, names in PATHWAYS.items():
    m, hit, miss = names_to_mask(names)
    pmask[p] = m
    print(f"  pathway {p:12s}: {hit}/{hit+miss} genes in vocab", flush=True)

# shared masks (seed 42)
torch.manual_seed(42)
mY=[((torch.rand_like(eY[i])<0.3)&hY[i]) for i in range(len(YB1))]
mW=[((torch.rand_like(eW[i])<0.3)&hW[i]) for i in range(len(WT))]

# ---- corpus for GENIE3/mean ----
vocab=pd.read_csv(VOCAB,sep="\t").sort_values("idx").reset_index(drop=True)
mat=pd.read_parquet(PARQ); lib=mat.sum(0).clip(lower=1)
lc=np.log2(mat.div(lib,axis=1)*1e6+1.0).astype(np.float32)
l2r={l:i for i,l in enumerate(lc.index)}; arr=lc.values; G=len(vocab)
X=np.zeros((lc.shape[1],G),np.float32); present=np.zeros(G,bool)
for vi,lt in zip(vocab["idx"].values,vocab["locus_tag"].values):
    r=l2r.get(lt)
    if r is not None: X[:,vi]=arr[r]; present[vi]=True
rng=np.random.default_rng(0); X=X[rng.choice(X.shape[0],N_SUB,replace=False)]
cmean=X.mean(0)
mrna_idx=np.where(vocab["type"].values=="mrna")[0]

print("fitting GENIE3...",flush=True)
def fit(g):
    y=X[:,g]
    if y.std()<1e-6: return g,None
    m=ExtraTreesRegressor(n_estimators=N_EST,max_features="sqrt",n_jobs=1,random_state=0)
    m.fit(X[:,np.arange(G)!=g],y); return g,m
models=dict(Parallel(n_jobs=-1)(delayed(fit)(g) for g in mrna_idx if present[g]))
print(f"  fitted {sum(m is not None for m in models.values())}",flush=True)

@torch.no_grad()
def v5_pred(model,e,m):
    out=e.clone(); p=model(e.unsqueeze(0),m.unsqueeze(0))[0]; out[m]=p[m]; return out
def genie_pred(e,m):
    out=e.clone(); xin=e.numpy().copy(); midx=np.where(m.numpy())[0]
    xin[m.numpy()]=cmean[m.numpy()]
    for g in midx:
        mm=models.get(int(g))
        out[g]=float(mm.predict(np.delete(xin,g).reshape(1,-1))[0]) if mm is not None else float(cmean[g])
    return out
def mean_pred(e,m):
    out=e.clone(); out[m]=torch.tensor(cmean)[m]; return out

# build v5 model
model=Stage1Model(n_mrna=nm,n_srna=ns,srna_edges=ck["edges"],srna_effects=ck["effects"],
                  embed_dim=ck.get("config",{}).get("embed_dim",128),n_heads=4,n_layers=4).to(dev)
model.load_state_dict(ck["state_dict"],strict=False); model.eval()

def cos(p,t):
    if p.numel() < 3: return float('nan')
    return F.cosine_similarity(p.unsqueeze(0),t.unsqueeze(0)).item()

def per_pathway_cos(predfn):
    """Returns {pathway: {aer:(cos,n), ana:(cos,n)}}."""
    pY=torch.stack([predfn(eY[i],mY[i]) for i in range(len(YB1))])
    pW=torch.stack([predfn(eW[i],mW[i]) for i in range(len(WT))])
    out = {p: {} for p in PATHWAYS}
    for tag,yi,wi in [("aer",YB1_AER,WT_AER),("ana",YB1_ANA,WT_ANA)]:
        pf=pY[yi].mean(0)-pW[wi].mean(0)
        mf=eY[yi].mean(0)-eW[wi].mean(0)
        hh_base = hY[yi].all(0)&hW[wi].all(0)&(mf.abs()>0.5)
        for p, pmask_p in pmask.items():
            hh = hh_base & pmask_p
            n = int(hh.sum())
            c = cos(pf[hh], mf[hh]) if n >= 3 else float('nan')
            out[p][tag] = (c, n)
    return out

print("\n=== Per-pathway log2FC cosine (informative |measured|>0.5, within pathway) ===", flush=True)
results = {}
for name, fn in [("v5", lambda e,m:v5_pred(model,e,m)),
                 ("GENIE3", genie_pred),
                 ("mean-floor", mean_pred)]:
    print(f"\n  ---- {name} ----", flush=True)
    r = per_pathway_cos(fn)
    results[name] = r
    for p in PATHWAYS:
        a = r[p]["aer"]; n = r[p]["ana"]
        print(f"    {p:12s}  aer cos={a[0]:+.4f} (n={a[1]:3d})   ana cos={n[0]:+.4f} (n={n[1]:3d})", flush=True)

# clean summary table
print("\n\n=== Summary table (cosine, n_genes in pathway∩informative) ===", flush=True)
methods = ["v5", "GENIE3", "mean-floor"]
print(f"\n  Aerobic:")
print(f"    {'Pathway':12s} | " + " | ".join(f"{m:>16s}" for m in methods))
print(f"    {'-'*12} | " + " | ".join("-"*16 for _ in methods))
for p in PATHWAYS:
    row = []
    for m in methods:
        c, n = results[m][p]["aer"]
        row.append(f"{c:+.3f} (n={n:>3d})")
    print(f"    {p:12s} | " + " | ".join(f"{x:>16s}" for x in row))
print(f"\n  Anaerobic:")
print(f"    {'Pathway':12s} | " + " | ".join(f"{m:>16s}" for m in methods))
print(f"    {'-'*12} | " + " | ".join("-"*16 for _ in methods))
for p in PATHWAYS:
    row = []
    for m in methods:
        c, n = results[m][p]["ana"]
        row.append(f"{c:+.3f} (n={n:>3d})")
    print(f"    {p:12s} | " + " | ".join(f"{x:>16s}" for x in row))

# save
out_path = _os.path.join(_REPO, "eval_results", "per_pathway_baselines.json")
_os.makedirs(_os.path.dirname(out_path), exist_ok=True)
with open(out_path, "w") as f:
    serial = {m: {p: {t: list(v) for t, v in d.items()} for p, d in r.items()} for m, r in results.items()}
    json.dump(serial, f, indent=2)
print(f"\nSaved: {out_path}", flush=True)
print("DONE_PER_PATHWAY", flush=True)
