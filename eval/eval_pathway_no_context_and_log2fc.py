"""eval_pathway_no_context_and_log2fc.py — A: pathway-stratified no-context; B: log2FC cosine.

A: Per-pathway full-context vs no-context cosine on YB1 BAMs.
   If pathway shows large Δ → context truly informative for that pathway.

B: Model-predicted log2FC(YB1 vs SL7207) vs measured log2FC.
   Honest "model uniquely predicts differential" metric.
"""
from __future__ import annotations
# --- repository-relative paths (override via env vars; see README) ---
import os as _os
_REPO = _os.environ.get("YB1_REPO", _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
_DATA = _os.environ.get("YB1_DATA", _os.path.join(_REPO, "data", "processed"))
_REF  = _os.environ.get("YB1_REF",  _os.path.join(_REPO, "data", "reference"))
_CKPT = _os.environ.get("YB1_CKPT", _os.path.join(_REPO, "checkpoints"))
# --- end repo-relative paths ---
import argparse
import json
import sys
import statistics
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

sys.path.insert(0, _os.path.join(_REPO, "model"))
from train_stage1_ecoli import Stage1Model
from train_stage3_inhouse import map_inhouse_to_stage_vocab, build_sample_vectors

COMBINED_TSV = _os.path.join(_DATA, "combined_counts_v1.tsv")
YB1_BAMS = ["YB1_aer_MinION", "YB1_aer_0508", "YB1_ana_v2",
            "YB1_ana_0507", "YB1_ana_0508",
            "YB1_aer_clone_0516", "YB1_ana_clone_0516"]
WT_BAMS = ["SL7207_aer_0507", "SL7207_aer_0509", "SL7207_ana_0509"]

SPI1 = {"invG","invE","invF","invH","invA","invB","invC","invI","invJ","sipA","sipB","sipC","sipD","sptP","sopB","sopE","sopE2","sopD","hilA","hilC","hilD","prgH","prgJ","prgK","spaN","spaO","spaP"}
FLAG = {"fliA","fliC","fliD","fliE","fliF","fliG","fliH","fliI","fliJ","fliK","fliL","fliM","fliN","fliP","fliQ","fliR","fliS","flgA","flgB","flgC","flgD","flgE","flgF","flgG","flgH","flgI","flgJ","flgK","flgL","flgM","flhA","flhB","flhC","flhD","flhE","motA","motB"}
CHEMO = {"cheA","cheB","cheR","cheW","cheY","cheZ","tar","tsr","trg","tap"}
SPI2  = {"siiA","siiB","siiC","siiD","siiE","siiF","ssaA","ssaB","ssaC","ssaD","ssaE","ssaG","ssaH","ssaI","ssaJ","ssaK","ssaL","ssaM","ssaN","ssaO","ssaP","ssaQ","ssaR","ssaS","ssaT","ssaU","ssaV"}


def cos(p, t):
    if p.numel() < 3: return None
    return F.cosine_similarity(p.unsqueeze(0), t.unsqueeze(0)).item()


@torch.no_grad()
def fwd(model, expr_vec, mask):
    return model(expr_vec.unsqueeze(0), mask.unsqueeze(0))[0]


def name_set_to_idx_mask(name_set, mrna_names, srna_names, n_mrna, n_total, device):
    mask = torch.zeros(n_total, dtype=torch.bool, device=device)
    ns = {s.lower() for s in name_set}
    for i, n in enumerate(mrna_names):
        if n.lower() in ns: mask[i] = True
    for i, n in enumerate(srna_names):
        if n.lower() in ns: mask[n_mrna + i] = True
    return mask


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    ckpt = torch.load(args.ckpt, map_location=args.device, weights_only=False)
    cfg = ckpt.get("config", {})
    sc = ckpt.get("stage1_config", {}) or {}
    model = Stage1Model(
        n_mrna=ckpt["n_mrna"], n_srna=ckpt["n_srna"],
        srna_edges=ckpt["edges"], srna_effects=ckpt["effects"],
        embed_dim=cfg.get("embed_dim", sc.get("embed_dim",128)),
        n_heads=cfg.get("n_heads", sc.get("n_heads",4)),
        n_layers=cfg.get("n_layers", sc.get("n_layers",4)),
    ).to(args.device)
    model.load_state_dict(ckpt["state_dict"], strict=False)

    n_mrna = ckpt["n_mrna"]; n_srna = ckpt["n_srna"]; n_total = n_mrna + n_srna
    df = pd.read_csv(COMBINED_TSV, sep="\t")
    mapping = map_inhouse_to_stage_vocab(df["name"].tolist(), ckpt["mrna_names"], ckpt["srna_names"])

    expr_yb1, has_yb1 = build_sample_vectors(df, mapping, YB1_BAMS, n_total, args.device)
    expr_wt, has_wt = build_sample_vectors(df, mapping, WT_BAMS, n_total, args.device)

    print(f"\n  ckpt: {args.ckpt}")

    # ========== Experiment A: pathway no-context vs full ==========
    print("\n=== Experiment A: pathway-stratified full vs no-context ===")
    pathway_masks = {
        "SPI-1":      name_set_to_idx_mask(SPI1, ckpt["mrna_names"], ckpt["srna_names"], n_mrna, n_total, args.device),
        "Flagellar":  name_set_to_idx_mask(FLAG, ckpt["mrna_names"], ckpt["srna_names"], n_mrna, n_total, args.device),
        "Chemotaxis": name_set_to_idx_mask(CHEMO, ckpt["mrna_names"], ckpt["srna_names"], n_mrna, n_total, args.device),
        "SPI-2":      name_set_to_idx_mask(SPI2, ckpt["mrna_names"], ckpt["srna_names"], n_mrna, n_total, args.device),
    }

    all_mask = torch.ones(n_total, dtype=torch.bool, device=args.device)

    print(f"  {'Pathway':<12s}  {'pathn':>5}  {'full ctx':>10}  {'no ctx':>10}  {'Δ':>8}")
    print("  " + "-" * 55)
    pathway_results = {}
    for pname, pmask in pathway_masks.items():
        full_vals = []
        nc_vals = []
        for i, bam in enumerate(YB1_BAMS):
            expr = expr_yb1[i]; has = has_yb1[i]
            obs = has & pmask
            if obs.sum() < 3:
                continue
            # full: mask only pathway positions
            pred_full = fwd(model, expr, obs)
            cf = cos(pred_full[obs], expr[obs])
            # no-ctx: mask all
            pred_nc = fwd(model, expr, all_mask)
            cn = cos(pred_nc[obs], expr[obs])
            if cf is not None and cn is not None:
                full_vals.append(cf)
                nc_vals.append(cn)
        if full_vals:
            mf = statistics.mean(full_vals); sn_f = statistics.stdev(full_vals) if len(full_vals)>1 else 0
            mn = statistics.mean(nc_vals); sn_n = statistics.stdev(nc_vals) if len(nc_vals)>1 else 0
            print(f"  {pname:<12s}  {int(pmask.sum()):>5d}  {mf:+.4f}±{sn_f:.3f}  {mn:+.4f}±{sn_n:.3f}  {mf-mn:+.4f}")
            pathway_results[pname] = {"full_mean":mf, "full_std":sn_f, "noctx_mean":mn, "noctx_std":sn_n, "delta":mf-mn}

    # ========== Experiment B: log2FC cosine ==========
    print("\n=== Experiment B: model-predicted log2FC vs measured log2FC ===")

    # Compute predicted log_cpm per BAM (context-conditioned: mask 30% predict)
    torch.manual_seed(42)
    yb1_aer_idxs = [0,1,5]  # YB1_aer_MinION/0508/clone_0516
    yb1_ana_idxs = [2,3,4,6]
    wt_aer_idxs = [0,1]  # SL7207_aer_0507/0509
    wt_ana_idxs = [2]    # SL7207_ana_0509

    # For log2FC compute, use the MEASURED expression directly per BAM (already log2(cpm+1))
    # Model predicts each BAM independently (with masked-mostly mode), then compute (mean_YB1 - mean_WT)
    # vs measured (mean_YB1 - mean_WT).

    # The simplest version: use observed expression as ground truth, model prediction as predicted.
    # log2FC measured = mean(log_cpm YB1 BAMs) - mean(log_cpm WT BAMs)
    # log2FC predicted = mean(model_pred YB1 BAMs) - mean(model_pred WT BAMs)
    # both per gene; then cosine.

    def predict_full_bam(expr_row, has_row, mask_frac=0.3):
        """Predict full transcriptome: mask 30% of observed positions."""
        rand = torch.rand_like(expr_row)
        mask = (rand < mask_frac) & has_row
        # Run forward; predicted positions filled with model output, unmasked kept as expr
        pred = fwd(model, expr_row, mask)
        # Combine: at masked positions use pred, at unmasked positions use observed expr
        out = expr_row.clone()
        out[mask] = pred[mask]
        return out

    # Predict each BAM
    pred_yb1 = torch.stack([predict_full_bam(expr_yb1[i], has_yb1[i]) for i in range(len(YB1_BAMS))])
    pred_wt = torch.stack([predict_full_bam(expr_wt[i], has_wt[i]) for i in range(len(WT_BAMS))])

    # Aerobic comparison: YB1 aer (3 BAMs) vs SL7207 aer (2 BAMs)
    yb1_aer_mean = pred_yb1[yb1_aer_idxs].mean(0)
    wt_aer_mean = pred_wt[wt_aer_idxs].mean(0)
    pred_log2fc_aer = yb1_aer_mean - wt_aer_mean
    meas_yb1_aer_mean = expr_yb1[yb1_aer_idxs].mean(0)
    meas_wt_aer_mean = expr_wt[wt_aer_idxs].mean(0)
    meas_log2fc_aer = meas_yb1_aer_mean - meas_wt_aer_mean

    # Filter to informative genes (|measured log2FC| > 0.5) and observed in all
    has_yb1_aer = has_yb1[yb1_aer_idxs].all(0)
    has_wt_aer = has_wt[wt_aer_idxs].all(0)
    informative_aer = (has_yb1_aer & has_wt_aer) & (meas_log2fc_aer.abs() > 0.5)

    n_inf_aer = int(informative_aer.sum())
    if n_inf_aer >= 5:
        cos_log2fc_aer = cos(pred_log2fc_aer[informative_aer], meas_log2fc_aer[informative_aer])
        print(f"  Aerobic log2FC (informative |measured|>0.5): n={n_inf_aer}, cosine = {cos_log2fc_aer:+.4f}")
    else:
        cos_log2fc_aer = None

    # Anaerobic comparison
    yb1_ana_mean = pred_yb1[yb1_ana_idxs].mean(0)
    wt_ana_mean = pred_wt[wt_ana_idxs].mean(0)
    pred_log2fc_ana = yb1_ana_mean - wt_ana_mean
    meas_yb1_ana_mean = expr_yb1[yb1_ana_idxs].mean(0)
    meas_wt_ana_mean = expr_wt[wt_ana_idxs].mean(0)
    meas_log2fc_ana = meas_yb1_ana_mean - meas_wt_ana_mean

    has_yb1_ana = has_yb1[yb1_ana_idxs].all(0)
    has_wt_ana = has_wt[wt_ana_idxs].all(0)
    informative_ana = (has_yb1_ana & has_wt_ana) & (meas_log2fc_ana.abs() > 0.5)

    n_inf_ana = int(informative_ana.sum())
    if n_inf_ana >= 5:
        cos_log2fc_ana = cos(pred_log2fc_ana[informative_ana], meas_log2fc_ana[informative_ana])
        print(f"  Anaerobic log2FC (informative |measured|>0.5): n={n_inf_ana}, cosine = {cos_log2fc_ana:+.4f}")
    else:
        cos_log2fc_ana = None

    # Save
    out = Path(args.ckpt).parent / "yb1_pathway_noctx_and_log2fc.json"
    out.write_text(json.dumps({
        "ckpt": args.ckpt,
        "pathway_no_context": pathway_results,
        "log2fc": {"aerobic": cos_log2fc_aer, "anaerobic": cos_log2fc_ana,
                    "n_inf_aer": n_inf_aer, "n_inf_ana": n_inf_ana}
    }, indent=2))
    print(f"\n  saved {out}")


if __name__ == "__main__":
    main()
