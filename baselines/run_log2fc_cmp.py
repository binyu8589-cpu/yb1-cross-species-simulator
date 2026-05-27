"""log2FC discriminative comparison: v5 vs GENIE3 vs mean-floor, identical protocol.

Replicates eval_pathway_no_context_and_log2fc.py Experiment B exactly:
  predicted_full_bam = measured, with 30% (seed-42) positions replaced by the
  method's prediction. log2FC = mean(pred YB1) - mean(pred WT) per gene.
  cosine over informative genes (observed in all + |measured log2FC|>0.5), aer & ana.
All three methods share the SAME per-BAM masks for fairness.
"""
# --- repository-relative paths (override via env vars; see README) ---
import os as _os
_REPO = _os.environ.get("YB1_REPO", _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
_DATA = _os.environ.get("YB1_DATA", _os.path.join(_REPO, "data", "processed"))
_REF  = _os.environ.get("YB1_REF",  _os.path.join(_REPO, "data", "reference"))
_CKPT = _os.environ.get("YB1_CKPT", _os.path.join(_REPO, "checkpoints"))
# --- end repo-relative paths ---
import sys, json, numpy as np, pandas as pd, torch
import torch.nn.functional as F
from joblib import Parallel, delayed
from sklearn.ensemble import ExtraTreesRegressor
sys.path.insert(0, _os.path.join(_REPO, "model"))
from train_stage1_ecoli import Stage1Model
from train_stage3_inhouse import map_inhouse_to_stage_vocab, build_sample_vectors

V5=_REPO; CKPT=_os.path.join(_CKPT, "checkpoints_stage3v2_1_cond_S42", "best.pt")
COMB=_os.path.join(_DATA, "combined_counts_v1.tsv"); PARQ=_os.path.join(_DATA, "master_expression_matrix_v2.parquet"); VOCAB=_os.path.join(_REF, "sl1344_vocab.tsv")
YB1=["YB1_aer_MinION","YB1_aer_0508","YB1_ana_v2","YB1_ana_0507","YB1_ana_0508","YB1_aer_clone_0516","YB1_ana_clone_0516"]
WT=["SL7207_aer_0507","SL7207_aer_0509","SL7207_ana_0509"]
YB1_AER=[0,1,5]; YB1_ANA=[2,3,4,6]; WT_AER=[0,1]; WT_ANA=[2]
dev="cpu"; N_EST=10; N_SUB=700

ck=torch.load(CKPT,map_location=dev,weights_only=False)
nm,ns=ck["n_mrna"],ck["n_srna"]; nt=nm+ns
df=pd.read_csv(COMB,sep="\t")
mp=map_inhouse_to_stage_vocab(df["name"].tolist(),ck["mrna_names"],ck["srna_names"])
eY,hY=build_sample_vectors(df,mp,YB1,nt,dev); eW,hW=build_sample_vectors(df,mp,WT,nt,dev)
print(f"YB1 {tuple(eY.shape)} WT {tuple(eW.shape)}",flush=True)

# shared masks (seed 42, sequential like eval_pathway: YB1 BAMs then WT BAMs)
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

# build model
model=Stage1Model(n_mrna=nm,n_srna=ns,srna_edges=ck["edges"],srna_effects=ck["effects"],
                  embed_dim=ck.get("config",{}).get("embed_dim",128),n_heads=4,n_layers=4).to(dev)
model.load_state_dict(ck["state_dict"],strict=False); model.eval()

def cos(p,t): return F.cosine_similarity(p.unsqueeze(0),t.unsqueeze(0)).item()
def log2fc_cos(predfn):
    pY=torch.stack([predfn(eY[i],mY[i]) for i in range(len(YB1))])
    pW=torch.stack([predfn(eW[i],mW[i]) for i in range(len(WT))])
    res={}
    for tag,yi,wi in [("aer",YB1_AER,WT_AER),("ana",YB1_ANA,WT_ANA)]:
        pf=pY[yi].mean(0)-pW[wi].mean(0); mf=eY[yi].mean(0)-eW[wi].mean(0)
        hh=hY[yi].all(0)&hW[wi].all(0)&(mf.abs()>0.5)
        res[tag]=(cos(pf[hh],mf[hh]),int(hh.sum()))
    return res

print("\n=== log2FC cosine (informative |measured|>0.5), identical protocol ===",flush=True)
for name,fn in [("v5",lambda e,m:v5_pred(model,e,m)),("GENIE3",genie_pred),("mean-floor",mean_pred)]:
    r=log2fc_cos(fn)
    print(f"  {name:<11} aer={r['aer'][0]:+.4f}(n={r['aer'][1]})  ana={r['ana'][0]:+.4f}(n={r['ana'][1]})",flush=True)
print("DONE_LOG2FC")
