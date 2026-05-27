"""eval_yb1_mrna_n7.py — YB1 mRNA-level hold-out eval (n=7).

Reads mrna_counts_v1.tsv (4731 mRNA × 16 BAMs) and evaluates ckpt prediction
quality at mRNA positions (the complementary set to the sRNA eval).
"""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

import pandas as pd
import torch
import torch.nn.functional as F

sys.path.insert(0, "/home/razer/v5_pathD")
from train_stage1_ecoli import Stage1Model
from train_stage3_inhouse import map_inhouse_to_stage_vocab, build_sample_vectors

NEW_INHOUSE_TSV = "/home/razer/v5_pathD/wetlab_data/mrna_counts_v1.tsv"

ORIG_5 = ["YB1_aer_MinION", "YB1_aer_0508",
          "YB1_ana_v2", "YB1_ana_0507", "YB1_ana_0508"]
NEW_2 = ["YB1_aer_clone_0516", "YB1_ana_clone_0516"]
ALL_7 = ORIG_5 + NEW_2


@torch.no_grad()
def eval_bam(model, expr_row, has_row, mask_frac=0.3, seed=42):
    torch.manual_seed(seed)
    expr = expr_row.unsqueeze(0)
    has = has_row.unsqueeze(0)
    mask = (torch.rand_like(expr) < mask_frac) & has
    if mask.sum() < 3:
        return None
    model.eval()
    pred = model(expr, mask)
    p = pred[0, mask[0]]
    t = expr[0, mask[0]]
    cos = F.cosine_similarity(p.unsqueeze(0), t.unsqueeze(0)).item()
    mse = F.mse_loss(p, t).item()
    return cos, mse, int(mask[0].sum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--mask-frac", type=float, default=0.3)
    args = ap.parse_args()

    print(f"\n  ckpt: {args.ckpt}")
    ckpt = torch.load(args.ckpt, map_location=args.device, weights_only=False)
    cfg = ckpt.get("config", {})
    sc = ckpt.get("stage1_config", {}) or ckpt.get("stage2_config", {}) or {}
    emb = cfg.get("embed_dim", sc.get("embed_dim", 128))
    nh = cfg.get("n_heads", sc.get("n_heads", 4))
    nl = cfg.get("n_layers", sc.get("n_layers", 4))
    model = Stage1Model(
        n_mrna=ckpt["n_mrna"], n_srna=ckpt["n_srna"],
        srna_edges=ckpt["edges"], srna_effects=ckpt["effects"],
        embed_dim=emb, n_heads=nh, n_layers=nl,
    ).to(args.device)
    model.load_state_dict(ckpt["state_dict"], strict=False)

    df = pd.read_csv(NEW_INHOUSE_TSV, sep="\t")
    mapping = map_inhouse_to_stage_vocab(
        df["name"].tolist(), ckpt["mrna_names"], ckpt["srna_names"])
    n_genes = ckpt["n_mrna"] + ckpt["n_srna"]
    expr, has = build_sample_vectors(df, mapping, ALL_7, n_genes, args.device)

    rows = []
    for i, bam in enumerate(ALL_7):
        r = eval_bam(model, expr[i], has[i], args.mask_frac)
        if r is None:
            continue
        cos, mse, n = r
        rows.append({"bam": bam, "cos": cos, "mse": mse, "n": n})

    print(f"  {'BAM':<25}  {'cos':>8}  {'mse':>7}  {'n':>5}")
    print("  " + "-" * 55)
    for r in rows:
        print(f"  {r['bam']:<25}  {r['cos']:+.4f}  {r['mse']:7.3f}  {r['n']:>5}")

    import statistics
    orig_cos = [r['cos'] for r in rows if r['bam'] in ORIG_5]
    new_cos = [r['cos'] for r in rows if r['bam'] in NEW_2]
    all_cos = [r['cos'] for r in rows]
    print()
    if orig_cos:
        m = statistics.mean(orig_cos); s = statistics.stdev(orig_cos) if len(orig_cos) > 1 else 0
        print(f"  Original 5 BAMs mean = {m:+.4f} ± {s:.4f}  (n=5)")
    if new_cos:
        m = statistics.mean(new_cos); s = statistics.stdev(new_cos) if len(new_cos) > 1 else 0
        print(f"  0516 plate-clone (n=2) = {m:+.4f} ± {s:.4f}")
    if all_cos:
        m = statistics.mean(all_cos); s = statistics.stdev(all_cos) if len(all_cos) > 1 else 0
        print(f"  Combined (n=7)         = {m:+.4f} ± {s:.4f}")

    out = Path(args.ckpt).parent / "yb1_mrna_n7_eval.json"
    out.write_text(json.dumps({
        "ckpt": args.ckpt,
        "tsv": NEW_INHOUSE_TSV,
        "eval_layer": "mRNA",
        "per_bam": rows,
        "orig_5_mean_cos": statistics.mean(orig_cos) if orig_cos else None,
        "new_2_mean_cos": statistics.mean(new_cos) if new_cos else None,
        "n7_mean_cos": statistics.mean(all_cos) if all_cos else None,
    }, indent=2))
    print(f"  saved {out}")


if __name__ == "__main__":
    main()
