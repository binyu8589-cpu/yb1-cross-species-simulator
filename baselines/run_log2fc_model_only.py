"""Table-2-protocol log2FC cosine for ONE ckpt (model row only).
Reuses run_log2fc_cmp.py mask construction byte-for-byte (seed 42, shared masks,
sequential YB1 then WT) so numbers are directly comparable to the Table 2 v5/GENIE3/floor rows.
Usage: python run_log2fc_model_only.py --ckpt <path> [--tag NAME]
"""
import sys, argparse, torch
import torch.nn.functional as F
sys.path.insert(0, "/home/razer/v5_pathD")
from train_stage1_ecoli import Stage1Model
from train_stage3_inhouse import map_inhouse_to_stage_vocab, build_sample_vectors
import pandas as pd

V5="/home/razer/v5_pathD"
COMB=f"{V5}/wetlab_data/combined_counts_v1.tsv"
YB1=["YB1_aer_MinION","YB1_aer_0508","YB1_ana_v2","YB1_ana_0507","YB1_ana_0508","YB1_aer_clone_0516","YB1_ana_clone_0516"]
WT=["SL7207_aer_0507","SL7207_aer_0509","SL7207_ana_0509"]
YB1_AER=[0,1,5]; YB1_ANA=[2,3,4,6]; WT_AER=[0,1]; WT_ANA=[2]
dev="cpu"

ap=argparse.ArgumentParser(); ap.add_argument("--ckpt",required=True); ap.add_argument("--tag",default="")
args=ap.parse_args()
ck=torch.load(args.ckpt,map_location=dev,weights_only=False)
nm,ns=ck["n_mrna"],ck["n_srna"]; nt=nm+ns
df=pd.read_csv(COMB,sep="\t")
mp=map_inhouse_to_stage_vocab(df["name"].tolist(),ck["mrna_names"],ck["srna_names"])
eY,hY=build_sample_vectors(df,mp,YB1,nt,dev); eW,hW=build_sample_vectors(df,mp,WT,nt,dev)
torch.manual_seed(42)
mY=[((torch.rand_like(eY[i])<0.3)&hY[i]) for i in range(len(YB1))]
mW=[((torch.rand_like(eW[i])<0.3)&hW[i]) for i in range(len(WT))]
model=Stage1Model(n_mrna=nm,n_srna=ns,srna_edges=ck["edges"],srna_effects=ck["effects"],
                  embed_dim=ck.get("stage1_config",{}).get("embed_dim") or ck.get("config",{}).get("embed_dim") or 128,n_heads=4,n_layers=4).to(dev)
model.load_state_dict(ck["state_dict"],strict=False); model.eval()
def cos(p,t): return F.cosine_similarity(p.unsqueeze(0),t.unsqueeze(0)).item()
@torch.no_grad()
def v5_pred(e,m):
    out=e.clone(); p=model(e.unsqueeze(0),m.unsqueeze(0))[0]; out[m]=p[m]; return out
pY=torch.stack([v5_pred(eY[i],mY[i]) for i in range(len(YB1))])
pW=torch.stack([v5_pred(eW[i],mW[i]) for i in range(len(WT))])
res={}
for tag,yi,wi in [("aer",YB1_AER,WT_AER),("ana",YB1_ANA,WT_ANA)]:
    pf=pY[yi].mean(0)-pW[wi].mean(0); mf=eY[yi].mean(0)-eW[wi].mean(0)
    hh=hY[yi].all(0)&hW[wi].all(0)&(mf.abs()>0.5)
    res[tag]=(cos(pf[hh],mf[hh]),int(hh.sum()))
print(f"  {args.tag or args.ckpt:<34} edges={len(ck['edges'])}  aer={res['aer'][0]:+.4f}(n={res['aer'][1]})  ana={res['ana'][0]:+.4f}(n={res['ana'][1]})",flush=True)
