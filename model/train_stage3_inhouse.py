"""train_stage3_inhouse.py — v5 Stage 3 fine-tune on in-house BAMs.

Stage 3: warm-start from Stage 2 ckpt and fine-tune on in-house Salmonella
nanopore RNA-seq from SL7207 (3 BAMs) + Δasd (2 BAMs). YB1 BAMs are NEVER
used during training (hold-out, red line).

Data: srna_counts_v2_with_0514.tsv (448 SL1344 sRNAs × 10 BAMs).
After cross-vocab mapping to Stage 2 vocabulary (case-insensitive gene name),
~33 features overlap. Stage 3 trains on these conserved features only.
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
import os
import random
import time

import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from train_stage2_kroger import load_stage1_ckpt

INHOUSE_TSV = _os.path.join(_DATA, "srna_counts_v2_with_0514.tsv")
# Server fallback
if not os.path.exists(INHOUSE_TSV):
    INHOUSE_TSV = _os.path.join(_DATA, "srna_counts_v2_with_0514.tsv")

# Stage 3 train set: SL7207 × 3 + Δasd × 2 (YB1 is hold-out)
STAGE3_TRAIN_BAMS = [
    "SL7207_aer_0507", "SL7207_aer_0509", "SL7207_ana_0509",
    "dAsd_aer_0511", "dAsd_ana_0512",
    "dAsd_aer_0514", "dAsd_ana_0514",
]


def map_inhouse_to_stage_vocab(inhouse_names, mrna_names, srna_names):
    """Returns {row_idx_in_tsv: stage_vocab_idx}."""
    lookup = {}
    for i, n in enumerate(mrna_names):
        lookup[n.lower()] = i
    for i, n in enumerate(srna_names):
        lookup[n.lower()] = len(mrna_names) + i
    out = {}
    for k, name in enumerate(inhouse_names):
        v_idx = lookup.get(str(name).lower())
        if v_idx is not None:
            out[k] = v_idx
    return out


def build_sample_vectors(df, mapping, bam_labels, n_genes, device):
    """Build [n_samples, n_genes] log2(CPM+1) tensor + observed mask."""
    n_s = len(bam_labels)
    expr = torch.zeros(n_s, n_genes, device=device)
    has = torch.zeros(n_s, n_genes, dtype=torch.bool, device=device)
    for s_idx, bam in enumerate(bam_labels):
        col = f"{bam}_cpm"
        if col not in df.columns:
            print(f"  WARN: column {col} missing")
            continue
        for row_idx, v_idx in mapping.items():
            cpm = df.iloc[row_idx][col]
            if pd.isna(cpm) or cpm <= 0:
                continue
            expr[s_idx, v_idx] = float(torch.log2(torch.tensor(cpm + 1.0)))
            has[s_idx, v_idx] = True
    return expr, has


def train_one_epoch(model, expr_train, has_train, mask_frac, opt):
    model.train()
    rand_mask = (torch.rand_like(expr_train) < mask_frac) & has_train
    if rand_mask.sum() == 0:
        return float("nan")
    pred = model(expr_train, rand_mask)
    loss = F.mse_loss(pred[rand_mask], expr_train[rand_mask])
    opt.zero_grad(set_to_none=True)
    loss.backward()
    nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step()
    return loss.item()


@torch.no_grad()
def eval_on(model, expr, has, mask_frac):
    model.eval()
    mask = (torch.rand_like(expr) < mask_frac) & has
    if mask.sum() == 0:
        return 0.0, 0.0
    pred = model(expr, mask)
    cos_vals, mse_vals = [], []
    for i in range(expr.shape[0]):
        m = mask[i]
        if m.sum() < 3:
            continue
        p, t = pred[i, m], expr[i, m]
        cos = F.cosine_similarity(p.unsqueeze(0), t.unsqueeze(0)).item()
        mse = F.mse_loss(p, t).item()
        cos_vals.append(cos)
        mse_vals.append(mse)
    return (sum(cos_vals) / max(1, len(cos_vals)),
            sum(mse_vals) / max(1, len(mse_vals)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage2-ckpt", required=True)
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--mask-frac", type=float, default=0.3)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--ckpt-dir",
                    default=_os.path.join(_CKPT, "checkpoints_stage3_inhouse"))
    args = ap.parse_args()
    torch.manual_seed(args.seed)
    random.seed(args.seed)

    print("=" * 68)
    print(f" v5 Stage 3 trainer (in-house BAMs, warm-start from Stage 2)")
    print(f"   stage2_ckpt: {args.stage2_ckpt}")
    print(f"   epochs: {args.epochs}  lr: {args.lr}  device: {args.device}")
    print(f"   train BAMs: {STAGE3_TRAIN_BAMS}")
    print("=" * 68)

    print("\n[1/4] loading Stage 2 ckpt + model ...")
    model, ckpt = load_stage1_ckpt(args.stage2_ckpt, args.device)
    print(f"      n_mrna={ckpt['n_mrna']} n_srna={ckpt['n_srna']} "
          f"n_genes={model.n_genes}")

    print("\n[2/4] loading in-house BAM CPM TSV ...")
    df = pd.read_csv(INHOUSE_TSV, sep="\t")
    print(f"      n_inhouse_features={len(df)}  n_columns={len(df.columns)}")

    print("\n[3/4] cross-vocab mapping ...")
    mapping = map_inhouse_to_stage_vocab(
        df["name"].tolist(), ckpt["mrna_names"], ckpt["srna_names"])
    print(f"      mapped {len(mapping)}/{len(df)} in-house features to Stage 2 vocab")

    print("\n[4/4] fine-tuning ...")
    expr_train, has_train = build_sample_vectors(
        df, mapping, STAGE3_TRAIN_BAMS, model.n_genes, args.device)
    print(f"      train set: {expr_train.shape[0]} BAMs x "
          f"{has_train.sum(1).float().mean():.1f} observed features each")

    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    os.makedirs(args.ckpt_dir, exist_ok=True)
    best_cos = -2.0
    history = []
    for epoch in range(args.epochs):
        t0 = time.time()
        loss = train_one_epoch(model, expr_train, has_train,
                                args.mask_frac, opt)
        # eval on same set (5 BAMs, no holdout — Stage 3 是 fine-tune,
        # 真 holdout 是 Stage 3 后的 YB1 evaluator)
        cos, mse = eval_on(model, expr_train, has_train, args.mask_frac)
        dt = time.time() - t0
        line = (f"  epoch {epoch+1:>3}/{args.epochs}  "
                f"train_mse={loss:.3f}  "
                f"val_cos={cos:+.4f}  val_mse={mse:.3f}  t={dt:.2f}s")
        print(line, flush=True)
        history.append({"epoch": epoch+1, "loss": loss,
                        "val_cos": cos, "val_mse": mse})
        if cos > best_cos:
            best_cos = cos
            torch.save({
                "epoch": epoch+1,
                "state_dict": model.state_dict(),
                "val_cos": cos,
                "config": vars(args),
                "stage1_config": ckpt.get("stage1_config") or ckpt.get("config"),
                "stage2_config": ckpt.get("config"),
                "edges": ckpt["edges"],
                "effects": ckpt["effects"],
                "n_mrna": ckpt["n_mrna"],
                "n_srna": ckpt["n_srna"],
                "mrna_names": ckpt["mrna_names"],
                "srna_names": ckpt["srna_names"],
                "inhouse_mapping": mapping,
                "train_bams": STAGE3_TRAIN_BAMS,
                "stage2_ckpt_path": args.stage2_ckpt,
                "history": history,
            }, f"{args.ckpt_dir}/best.pt")
    print(f"\nDONE.  best val_cos = {best_cos:+.4f}  ->  {args.ckpt_dir}/best.pt")


if __name__ == "__main__":
    main()
