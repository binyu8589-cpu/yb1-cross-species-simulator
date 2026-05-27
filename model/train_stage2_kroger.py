"""train_stage2_kroger.py — v5 Stage 2 trainer (Salmonella transition).

Stage 2 task: fine-tune Stage 1 (E. coli) ckpt on Salmonella sRNA expression
from Kröger et al. 2013 (Dataset S2; 260 SL1344 sRNAs × ~22 conditions).
This is the curriculum's cross-species transition step:

  Stage 1 (Lim E. coli, 98 sRNAs, 3161 samples)
    ↓ warm-start ckpt (state_dict reused; vocab + edges unchanged)
  Stage 2 (Kröger Salmonella, 260 sRNAs, 23 conditions)
    ↓ Stage 2 ckpt
  Stage 3 (in-house Δasd + SL7207, 5 BAMs)
    ↓
  HOLD-OUT YB1

Design decisions:
  - Reuse Stage 1 Model architecture exactly (G_v5 = 4304 + 98 = 4402).
  - Map Kröger sRNAs to Stage 1 vocab by case-insensitive gene name; this is
    expected to recover ~30-60 of Kröger's 260 sRNAs (those conserved between
    E. coli and Salmonella). Salmonella-specific sRNAs (DapZ, PinT, InvR,
    IsrM, AmgR, ~215 STnc series) are NOT in Stage 1 vocab; for v1 of this
    trainer we skip them with a warning. Stage 2 v2 will extend vocab.
  - For mapped sRNAs only, supervise their expression on Kröger conditions
    via MLM: mask 15% of mapped sRNAs per condition, predict from the rest.

When NCBI Salmonella RNA-seq is processed, replace Kröger with the larger
mRNA + sRNA Salmonella compendium and re-train.
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

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from kroger_dataloader import KrogerExpressionDataset
from train_stage1_ecoli import Stage1Model, resolve_srna_edges


def map_kroger_to_stage1(kroger_srna_names: list[str],
                          stage1_mrna_names: list[str],
                          stage1_srna_names: list[str]) -> dict[int, int]:
    """Returns {kroger_idx: stage1_global_idx}. Case-insensitive name match."""
    lookup = {}
    for i, n in enumerate(stage1_mrna_names):
        lookup[n.lower()] = i
    for i, n in enumerate(stage1_srna_names):
        lookup[n.lower()] = len(stage1_mrna_names) + i
    out = {}
    for k_idx, k_name in enumerate(kroger_srna_names):
        s1_idx = lookup.get(k_name.lower())
        if s1_idx is not None:
            out[k_idx] = s1_idx
    return out


def load_stage1_ckpt(path: str, device: str):
    """Load a Stage 1/2/3 ckpt and rebuild its Stage1Model.

    Stage 1 ckpts have config = Stage 1 training args (includes embed_dim).
    Stage 2/3 ckpts have config = their own args + stage1_config = Stage 1 args.
    Falls back to stage1_config / stage2_config when needed.
    """
    ckpt = torch.load(path, map_location=device, weights_only=False)
    # Pick a config that has embed_dim
    for key in ("config", "stage1_config", "stage2_config"):
        cfg = ckpt.get(key, {}) or {}
        if "embed_dim" in cfg:
            break
    else:
        raise KeyError(f"No config with embed_dim in ckpt {path}")
    model = Stage1Model(
        n_mrna=ckpt["n_mrna"],
        n_srna=ckpt["n_srna"],
        srna_edges=ckpt["edges"],
        srna_effects=ckpt["effects"],
        embed_dim=cfg["embed_dim"],
        n_heads=cfg["n_heads"],
        n_layers=cfg["n_layers"],
    ).to(device)
    model.load_state_dict(ckpt["state_dict"])
    return model, ckpt


def train_one_epoch(model, kroger, mapping, mask_frac, device, opt):
    """One epoch over Kröger conditions. Per condition: build a sparse expression
    vector at G_v5 dims (zero where no data), mask 15% of mapped sRNAs, predict.
    """
    model.train()
    G = model.n_genes
    cond_indices = list(range(kroger.n_cond))
    random.shuffle(cond_indices)
    losses = []
    for cond_idx in cond_indices:
        # Sparse expr at G_v5 dims
        expr = torch.zeros(1, G, device=device)
        has = torch.zeros(1, G, dtype=torch.bool, device=device)
        for k_idx, s1_idx in mapping.items():
            expr[0, s1_idx] = kroger.log_tpm[k_idx, cond_idx]
            has[0, s1_idx] = True
        # Mask 15% of mapped positions
        rand_mask = (torch.rand_like(expr) < mask_frac) & has
        if rand_mask.sum() == 0:
            continue
        pred = model(expr, rand_mask)
        loss = F.mse_loss(pred[rand_mask], expr[rand_mask])
        opt.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        losses.append(loss.item())
    return sum(losses) / max(1, len(losses))


@torch.no_grad()
def eval_one(model, kroger, mapping, mask_frac, device):
    model.eval()
    G = model.n_genes
    cos_vals, mse_vals = [], []
    for cond_idx in range(kroger.n_cond):
        expr = torch.zeros(1, G, device=device)
        has = torch.zeros(1, G, dtype=torch.bool, device=device)
        for k_idx, s1_idx in mapping.items():
            expr[0, s1_idx] = kroger.log_tpm[k_idx, cond_idx]
            has[0, s1_idx] = True
        rand_mask = (torch.rand_like(expr) < mask_frac) & has
        if rand_mask.sum() < 3:
            continue
        pred = model(expr, rand_mask)
        p = pred[rand_mask]
        t = expr[rand_mask]
        cos = F.cosine_similarity(p.unsqueeze(0), t.unsqueeze(0)).item()
        mse = F.mse_loss(p, t).item()
        cos_vals.append(cos)
        mse_vals.append(mse)
    return (sum(cos_vals) / len(cos_vals) if cos_vals else 0.0,
            sum(mse_vals) / len(mse_vals) if mse_vals else 0.0,
            len(cos_vals))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage1-ckpt", required=True,
                    help="Path to Stage 1 ckpt to warm-start from")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--lr", type=float, default=1e-4,
                    help="Fine-tune lr (smaller than Stage 1's 3e-4)")
    ap.add_argument("--mask-frac", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--ckpt-dir",
                    default=_os.path.join(_CKPT, "checkpoints_stage2_kroger"))
    ap.add_argument("--freeze-base", action="store_true",
                    help="Freeze v3 transformer + heads, only train sRNA gate.")
    args = ap.parse_args()
    torch.manual_seed(args.seed)
    random.seed(args.seed)

    print("=" * 68)
    print(f" v5 Stage 2 trainer (Kröger Salmonella sRNA, warm-start from Stage 1)")
    print(f"   stage1_ckpt: {args.stage1_ckpt}")
    print(f"   epochs: {args.epochs}  lr: {args.lr}  seed: {args.seed}")
    print(f"   device: {args.device}  freeze_base: {args.freeze_base}")
    print("=" * 68)

    print("\n[1/4] loading Stage 1 ckpt + model ...")
    model, ckpt = load_stage1_ckpt(args.stage1_ckpt, args.device)
    n_p_total = sum(p.numel() for p in model.parameters())
    print(f"      n_mrna={ckpt['n_mrna']} n_srna={ckpt['n_srna']} "
          f"n_genes={model.n_genes}  total_params={n_p_total/1e6:.2f} M")
    print(f"      Stage 1 ckpt val_cos: {ckpt['val_cos']:+.4f}")

    print("\n[2/4] loading Kröger Salmonella ...")
    kroger = KrogerExpressionDataset()
    print(f"      n_kroger_srna={kroger.n_srna}  n_conditions={kroger.n_cond}")

    print("\n[3/4] cross-species ortholog mapping ...")
    mapping = map_kroger_to_stage1(
        kroger.srna_names, ckpt["mrna_names"], ckpt["srna_names"])
    print(f"      mapped {len(mapping)}/{kroger.n_srna} Kröger sRNAs to Stage 1 vocab")
    unmapped_examples = [n for i, n in enumerate(kroger.srna_names)
                         if i not in mapping][:8]
    print(f"      unmapped examples: {unmapped_examples}")
    if len(mapping) < 10:
        print(f"      ⚠ very few mapped — Stage 2 will have weak signal")

    print("\n[4/4] fine-tuning ...")
    # Move kroger.log_tpm to device
    kroger.log_tpm = kroger.log_tpm.to(args.device)
    # Optimizer
    if args.freeze_base:
        # Only train sRNA gate + bias
        for n, p in model.named_parameters():
            p.requires_grad = "srna" in n
        n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"      frozen base; trainable: {n_trainable} params (srna only)")
    opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad],
                            lr=args.lr)

    os.makedirs(args.ckpt_dir, exist_ok=True)
    best_cos = -2.0
    history = []
    for epoch in range(args.epochs):
        t0 = time.time()
        loss = train_one_epoch(model, kroger, mapping, args.mask_frac,
                                args.device, opt)
        val_cos, val_mse, val_n = eval_one(model, kroger, mapping,
                                             args.mask_frac, args.device)
        dt = time.time() - t0
        print(f"  epoch {epoch+1:>3}/{args.epochs}  train_mse={loss:.3f}  "
              f"val_cos={val_cos:+.4f}  val_mse={val_mse:.3f}  "
              f"({val_n} conds)  t={dt:.1f}s",
              flush=True)
        history.append({"epoch": epoch+1, "loss": loss,
                        "val_cos": val_cos, "val_mse": val_mse})
        if val_cos > best_cos:
            best_cos = val_cos
            torch.save({
                "epoch": epoch+1,
                "state_dict": model.state_dict(),
                "val_cos": val_cos,
                "config": vars(args),
                "stage1_config": ckpt["config"],
                "edges": ckpt["edges"],
                "effects": ckpt["effects"],
                "n_mrna": ckpt["n_mrna"],
                "n_srna": ckpt["n_srna"],
                "mrna_names": ckpt["mrna_names"],
                "srna_names": ckpt["srna_names"],
                "kroger_mapping": mapping,
                "stage1_ckpt_path": args.stage1_ckpt,
                "history": history,
            }, f"{args.ckpt_dir}/best.pt")
    print(f"\nDONE.  best val_cos = {best_cos:+.4f}  ->  {args.ckpt_dir}/best.pt")


if __name__ == "__main__":
    main()
