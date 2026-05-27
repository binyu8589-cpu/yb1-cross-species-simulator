"""GENIE3-style baseline for Table 1, apples-to-apples with the v5 YB1 eval.

Protocol (matches eval_yb1_mrna_n7.py exactly where it matters):
  - YB1 per-BAM vectors + observed mask via build_sample_vectors (log2(CPM+1),
    vocab idx order, n=7 BAMs) — identical to what the model is scored on.
  - Same masking: torch.manual_seed(42); mask = (rand < 0.3) & has, per BAM.
  - Corpus = master_expression_matrix_v2.parquet, normalized log2(CPM+1) in the
    SAME vocab idx order (exactly SalmonellaDataset's normalization).
  - GENIE3: per mRNA gene, ExtraTreesRegressor(n_estimators=40, max_features='sqrt')
    predicting that gene from ALL other genes (the canonical GENIE3 regressor).
  - Predict masked mRNA positions from the observed YB1 vector (masked inputs
    imputed with corpus mean in log space). cosine(pred_masked, true_masked) per BAM.
  - Also: mean-expression floor (predict masked = corpus gene mean).
"""
import sys, json, time
import numpy as np, pandas as pd, torch
import torch.nn.functional as F
from joblib import Parallel, delayed
from sklearn.ensemble import ExtraTreesRegressor

sys.path.insert(0, "/home/razer/v5_pathD")
from train_stage1_ecoli import Stage1Model  # noqa
from train_stage3_inhouse import map_inhouse_to_stage_vocab, build_sample_vectors

V5 = "/home/razer/v5_pathD"
CKPT = f"{V5}/checkpoints_razer_stage3v2_1_S42/best.pt"
PARQUET = f"{V5}/master_expression_matrix_v2.parquet"
VOCAB = f"{V5}/sl1344_vocab.tsv"
INHOUSE = f"{V5}/wetlab_data/mrna_counts_v1.tsv"
ALL_7 = ["YB1_aer_MinION","YB1_aer_0508","YB1_ana_v2","YB1_ana_0507","YB1_ana_0508",
         "YB1_aer_clone_0516","YB1_ana_clone_0516"]
N_EST = 10
N_SUB = 700  # subsample corpus samples to keep runtime under the ssh-drop window
device = "cpu"

print("[1] loading ckpt vocab + YB1 vectors (same as model eval) ...", flush=True)
ck = torch.load(CKPT, map_location="cpu", weights_only=False)
n_mrna, n_srna = ck["n_mrna"], ck["n_srna"]; n_genes = n_mrna + n_srna
df = pd.read_csv(INHOUSE, sep="\t")
mapping = map_inhouse_to_stage_vocab(df["name"].tolist(), ck["mrna_names"], ck["srna_names"])
expr_yb1, has_yb1 = build_sample_vectors(df, mapping, ALL_7, n_genes, device)  # [7, G]
expr_yb1 = expr_yb1.numpy(); has_yb1 = has_yb1.numpy()
print(f"    YB1 vectors {expr_yb1.shape}, mRNA n_genes={n_mrna} srna={n_srna}", flush=True)

print("[2] building corpus log2(CPM+1) in vocab idx order ...", flush=True)
vocab = pd.read_csv(VOCAB, sep="\t").sort_values("idx").reset_index(drop=True)
mat = pd.read_parquet(PARQUET)  # [genes(locus_tag) x samples]
lib = mat.sum(axis=0).clip(lower=1)
logcpm = np.log2(mat.div(lib, axis=1) * 1e6 + 1.0).astype(np.float32)  # [genes x samples]
# project to vocab idx order by locus_tag
loc2row = {l: i for i, l in enumerate(logcpm.index)}
G = len(vocab); Ns = logcpm.shape[1]
X = np.zeros((Ns, G), dtype=np.float32)  # [samples x vocab_genes]
present = np.zeros(G, dtype=bool)
arr = logcpm.values
for vi, lt in zip(vocab["idx"].values, vocab["locus_tag"].values):
    r = loc2row.get(lt)
    if r is not None:
        X[:, vi] = arr[r]; present[vi] = True
if X.shape[0] > N_SUB:
    rng = np.random.default_rng(0)
    sel = rng.choice(X.shape[0], N_SUB, replace=False)
    X = X[sel]
    print(f"    subsampled corpus to {X.shape[0]} samples (speed)", flush=True)
corpus_mean = X.mean(axis=0)  # [G]
print(f"    corpus X {X.shape}; vocab genes present in parquet: {present.sum()}/{G}", flush=True)

# mRNA target indices that are observed in YB1 (so they can be masked/scored)
mrna_idx = np.where(vocab["type"].values == "mrna")[0]
print(f"    mRNA vocab positions: {len(mrna_idx)}", flush=True)

print(f"[3] fitting GENIE3 ExtraTrees per mRNA gene (n_est={N_EST}) ...", flush=True)
t0 = time.time()
def fit_one(g):
    cols = np.arange(G) != g
    y = X[:, g]
    if y.std() < 1e-6:
        return g, None  # constant gene → skip (predict mean)
    m = ExtraTreesRegressor(n_estimators=N_EST, max_features="sqrt",
                            n_jobs=1, random_state=0)
    m.fit(X[:, cols], y)
    return g, m
# fit only for mRNA genes that are present (others predicted by mean)
targets = [g for g in mrna_idx if present[g]]
res = Parallel(n_jobs=-1, verbose=5)(delayed(fit_one)(g) for g in targets)
models = {g: m for g, m in res}
print(f"    fitted {sum(1 for m in models.values() if m is not None)} regressors "
      f"in {time.time()-t0:.0f}s", flush=True)

print("[4] predicting masked YB1 positions, per BAM ...", flush=True)
def cosine(a, b):
    a = torch.tensor(a); b = torch.tensor(b)
    return F.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0)).item()

genie_cos, mean_cos = [], []
for i, bam in enumerate(ALL_7):
    torch.manual_seed(42)
    has = torch.tensor(has_yb1[i]); rnd = torch.rand_like(torch.tensor(expr_yb1[i]))
    mask = ((rnd < 0.3) & has).numpy()
    midx = np.where(mask)[0]
    midx = np.array([g for g in midx if vocab["type"].values[g] == "mrna"])  # mRNA only
    if len(midx) < 3:
        continue
    # build input vector: observed YB1 values; masked positions -> corpus mean
    xin = expr_yb1[i].copy()
    xin[mask] = corpus_mean[mask]
    true = expr_yb1[i][midx]
    # GENIE3 prediction
    pred_g = np.empty(len(midx), dtype=np.float32)
    for k, g in enumerate(midx):
        m = models.get(g)
        if m is None:
            pred_g[k] = corpus_mean[g]
        else:
            feat = np.delete(xin, g).reshape(1, -1)
            pred_g[k] = m.predict(feat)[0]
    genie_cos.append(cosine(pred_g, true))
    mean_cos.append(cosine(corpus_mean[midx], true))
    print(f"    {bam:<25} GENIE3 cos={genie_cos[-1]:+.4f}  meanfloor={mean_cos[-1]:+.4f}  n={len(midx)}", flush=True)

import statistics as st
def ms(x): return (st.mean(x), st.pstdev(x))
gm, gs = ms(genie_cos); mm, msd = ms(mean_cos)
print(f"\n=== Table 1 baseline (held-out YB1, mRNA cosine, n={len(genie_cos)} BAMs) ===")
print(f"  GENIE3 (ExtraTrees per-gene): {gm:+.4f} ± {gs:.4f}")
print(f"  mean-expression floor       : {mm:+.4f} ± {msd:.4f}")
print(f"  (this work / v5            : +0.944 ± 0.007)")
json.dump({"genie3_cos": genie_cos, "mean_cos": mean_cos,
           "genie3_mean": gm, "mean_floor": mm, "n_est": N_EST},
          open(f"{V5}/genie3_baseline_eval.json", "w"), indent=2)
print("  saved genie3_baseline_eval.json")
print("DONE_GENIE3")
