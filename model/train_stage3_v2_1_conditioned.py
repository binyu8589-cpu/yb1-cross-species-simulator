"""train_stage3_v2_1_conditioned.py — pilot: add condition (aerobic/anaerobic) token.

Architectural change: existing model + new self.cond_emb (2 categories × embed_dim).
Adds cond_emb to gene_emb + type_marker before transformer layers.

Trains only Stage 3 (fine-tune from existing Stage 2 v2.1 ckpt).
Single seed pilot. ~10 min on RTX 5090.

Eval mode: "condition-only forward" — at inference, mask all 5,098 expression positions
but provide the condition token. Tests whether categorical condition information adds
predictive signal beyond pure gene_emb baseline.
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
import sys
import time
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, _os.path.join(_REPO, "model"))
from train_stage1_ecoli import Stage1Model
from train_stage3_inhouse import map_inhouse_to_stage_vocab, build_sample_vectors

# Use v3 TSV which includes 0516 plate-clone columns
INHOUSE_TSV = _os.path.join(_DATA, "srna_counts_v3_with_0516.tsv")

# In-house BAM → (strain, condition_idx) labels
BAM_LABELS = {
    # Training (no YB1)
    "SL7207_aer_0507":    ("SL7207", 0),  # aerobic = 0
    "SL7207_aer_0509":    ("SL7207", 0),
    "SL7207_ana_0509":    ("SL7207", 1),  # anaerobic = 1
    "dAsd_aer_0511":      ("Δasd", 0),
    "dAsd_aer_0514":      ("Δasd", 0),
    "dAsd_ana_0512":      ("Δasd", 1),
    "dAsd_ana_0514":      ("Δasd", 1),
    # YB1 hold-out (eval only)
    "YB1_aer_MinION":     ("YB1", 0),
    "YB1_aer_0508":       ("YB1", 0),
    "YB1_ana_v2":         ("YB1", 1),
    "YB1_ana_0507":       ("YB1", 1),
    "YB1_ana_0508":       ("YB1", 1),
    "YB1_aer_clone_0516": ("YB1", 0),
    "YB1_ana_clone_0516": ("YB1", 1),
}

TRAIN_BAMS = ["SL7207_aer_0507", "SL7207_aer_0509", "SL7207_ana_0509",
              "dAsd_aer_0511", "dAsd_aer_0514", "dAsd_ana_0512", "dAsd_ana_0514"]
YB1_BAMS = ["YB1_aer_MinION", "YB1_aer_0508", "YB1_ana_v2",
            "YB1_ana_0507", "YB1_ana_0508",
            "YB1_aer_clone_0516", "YB1_ana_clone_0516"]


class ConditionedStage1Model(Stage1Model):
    """Stage1Model + condition embedding (aerobic=0, anaerobic=1)."""
    def __init__(self, *args, n_conditions: int = 2, **kwargs):
        super().__init__(*args, **kwargs)
        self.cond_emb = nn.Embedding(n_conditions, self.embed_dim)
        nn.init.normal_(self.cond_emb.weight, std=0.02)

    def forward(self, expr, mask, cond_id=None):
        """expr [B, G] log_tpm; mask [B, G] bool; cond_id [B] long."""
        B, G = expr.shape
        ids = torch.arange(G, device=expr.device)
        gene_e = self.gene_emb(ids).unsqueeze(0).expand(B, -1, -1)
        type_e = self.type_in(self.type_marker).unsqueeze(0).expand(B, -1, -1)
        expr_e = self.expr_in(expr.unsqueeze(-1))
        x = gene_e + type_e
        m = mask.unsqueeze(-1).float()
        x = x + (1.0 - m) * expr_e + m * self.mask_token
        # Add condition embedding broadcast across all genes
        if cond_id is not None:
            cond_e = self.cond_emb(cond_id).unsqueeze(1).expand(-1, G, -1)
            x = x + cond_e
        for layer in self.layers:
            x = layer(x)
        pred = self.head(x).squeeze(-1)
        pred = self.srna_gate(pred)
        return pred


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage2-ckpt", required=True)
    ap.add_argument("--ckpt-dir", required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--mask-frac", type=float, default=0.3)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    print("=" * 70)
    print(" Stage 3 v2.1 + condition-token PILOT")
    print(f"  stage2_ckpt: {args.stage2_ckpt}")
    print(f"  ckpt_dir: {args.ckpt_dir}")
    print(f"  seed: {args.seed}  lr: {args.lr}  epochs: {args.epochs}")
    print("=" * 70)

    # Load stage 2 v2.1 ckpt
    print("\n[1] loading Stage 2 v2.1 ckpt ...")
    ckpt = torch.load(args.stage2_ckpt, map_location=args.device, weights_only=False)
    cfg = ckpt.get("config", {})
    sc = ckpt.get("stage1_config", {}) or {}
    embed_dim = cfg.get("embed_dim", sc.get("embed_dim", 128))
    n_heads = cfg.get("n_heads", sc.get("n_heads", 4))
    n_layers = cfg.get("n_layers", sc.get("n_layers", 4))

    model = ConditionedStage1Model(
        n_mrna=ckpt["n_mrna"], n_srna=ckpt["n_srna"],
        srna_edges=ckpt["edges"], srna_effects=ckpt["effects"],
        embed_dim=embed_dim, n_heads=n_heads, n_layers=n_layers,
        n_conditions=2,
    ).to(args.device)
    # Load existing weights (strict=False because cond_emb is new)
    missing, unexpected = model.load_state_dict(ckpt["state_dict"], strict=False)
    print(f"  loaded; missing {len(missing)} (new cond_emb), unexpected {len(unexpected)}")

    # Build training expr vectors with condition labels
    df = pd.read_csv(INHOUSE_TSV, sep="\t")
    n_mrna, n_srna = ckpt["n_mrna"], ckpt["n_srna"]
    n_total = n_mrna + n_srna
    mapping = map_inhouse_to_stage_vocab(
        df["name"].tolist(), ckpt["mrna_names"], ckpt["srna_names"])
    expr_train, has_train = build_sample_vectors(df, mapping, TRAIN_BAMS, n_total, args.device)
    cond_train = torch.tensor([BAM_LABELS[b][1] for b in TRAIN_BAMS], device=args.device)
    print(f"  train BAMs: {len(TRAIN_BAMS)}  conditions: {cond_train.tolist()}")

    # Fine-tune
    print(f"\n[2] fine-tuning with condition tokens (seed {args.seed}, {args.epochs} ep, lr {args.lr}) ...")
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    n_samples = len(TRAIN_BAMS)
    n_steps_per_epoch = 4
    best_loss = float("inf")
    history = []
    t0 = time.time()
    for ep in range(args.epochs):
        model.train()
        total_loss = 0.0
        for step in range(n_steps_per_epoch):
            # Random masking per sample
            rand = torch.rand_like(expr_train)
            mask = (rand < args.mask_frac) & has_train
            if mask.sum() < 5:
                continue
            pred = model(expr_train, mask, cond_train)
            loss = F.mse_loss(pred[mask], expr_train[mask])
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total_loss += loss.item()
        avg_loss = total_loss / n_steps_per_epoch
        history.append({"epoch": ep + 1, "loss": avg_loss})
        if ep % 10 == 0 or ep == args.epochs - 1:
            print(f"  ep {ep+1:>3}: loss = {avg_loss:.3f}")
        if avg_loss < best_loss:
            best_loss = avg_loss
    print(f"  done in {time.time() - t0:.0f}s")

    # Save ckpt
    os.makedirs(args.ckpt_dir, exist_ok=True)
    torch.save({
        "state_dict": model.state_dict(),
        "config": vars(args),
        "stage2_config": cfg,
        "stage1_config": sc,
        "edges": ckpt["edges"],
        "effects": ckpt["effects"],
        "n_mrna": ckpt["n_mrna"],
        "n_srna": ckpt["n_srna"],
        "mrna_names": ckpt["mrna_names"],
        "srna_names": ckpt["srna_names"],
        "vocab_names": ckpt.get("vocab_names"),
        "vocab_locus": ckpt.get("vocab_locus"),
        "history": history,
        "best_loss": best_loss,
        "n_conditions": 2,
    }, f"{args.ckpt_dir}/best.pt")
    print(f"  saved {args.ckpt_dir}/best.pt")

    # === Eval ===
    print("\n[3] YB1 hold-out eval (4 modes) ...")
    expr_yb1, has_yb1 = build_sample_vectors(df, mapping, YB1_BAMS, n_total, args.device)
    cond_yb1 = torch.tensor([BAM_LABELS[b][1] for b in YB1_BAMS], device=args.device)

    def cos(p, t):
        if p.numel() < 3: return None
        return F.cosine_similarity(p.unsqueeze(0), t.unsqueeze(0)).item()

    @torch.no_grad()
    def fwd(model, expr, mask, cond_id):
        return model(expr.unsqueeze(0), mask.unsqueeze(0), cond_id.unsqueeze(0))[0]

    all_mask = torch.ones(n_total, dtype=torch.bool, device=args.device)
    torch.manual_seed(42)

    print(f"  {'BAM':<22}  {'full':>9}  {'no_ctx':>9}  {'cond_only':>10}  {'NO_cond_NO_ctx':>14}")
    print("  " + "-" * 78)
    results = []
    for i, bam in enumerate(YB1_BAMS):
        expr = expr_yb1[i]; has = has_yb1[i]; cond = cond_yb1[i]
        # Full context with condition
        rand = torch.rand_like(expr)
        mask_full = (rand < args.mask_frac) & has
        pred_full = fwd(model, expr, mask_full, cond)
        cos_full = cos(pred_full[mask_full], expr[mask_full])
        # No expr context, with condition
        pred_nctx_cond = fwd(model, expr, all_mask, cond)
        cos_nctx_cond = cos(pred_nctx_cond[has], expr[has])
        # Condition-only (same as above, but separately named for clarity)
        cos_cond_only = cos_nctx_cond
        # NO condition, NO context (pass cond_id as None equivalent - use zeros)
        # Use zero embedding via setting cond_id to a special idx? We have 2 categories.
        # Trick: zero out cond_emb effect by manually subtracting cond_emb's mean
        # Simpler: just don't add cond_emb. Call forward with cond_id=None
        pred_no_all = model(expr.unsqueeze(0), all_mask.unsqueeze(0), None)[0]
        cos_no_all = cos(pred_no_all[has], expr[has])

        def fmt(v):
            return f"{v:+.4f}" if v is not None else "  N/A "
        print(f"  {bam:<22}  {fmt(cos_full)}  {fmt(cos_nctx_cond)}  {fmt(cos_cond_only)}  {fmt(cos_no_all)}")
        results.append({"bam": bam, "full": cos_full, "cond_only": cos_cond_only, "no_cond_no_ctx": cos_no_all})

    import statistics
    print()
    for k in ["full", "cond_only", "no_cond_no_ctx"]:
        vals = [r[k] for r in results if r[k] is not None]
        if vals:
            print(f"  {k:<20s}  mean = {statistics.mean(vals):+.4f} ± {statistics.stdev(vals):.4f}")

    Path(f"{args.ckpt_dir}/condition_token_pilot_eval.json").write_text(
        json.dumps({"per_bam": results, "history": history}, indent=2))
    print(f"\n  saved {args.ckpt_dir}/condition_token_pilot_eval.json")


if __name__ == "__main__":
    main()
