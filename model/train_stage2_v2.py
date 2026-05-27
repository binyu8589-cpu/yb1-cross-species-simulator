"""train_stage2_v2.py — v5 Stage 2 v2 trainer (full Salmonella SL1344 compendium).

Replaces train_stage2_kroger.py. Differences:
  - Extended vocab: 4731 SL1344 mRNA + 367 sRNA = 5098 genes (vs Stage 1's 4402)
  - Real expression matrix from 3328 NCBI Salmonella RNA-seq runs
  - Warm-start by ortholog transfer: copy Lim embeddings → SL1344 matched positions
    using ortholog_lim_to_sl1344.tsv (mRNA 50%, sRNA 28%); rest gets random init

Train task: MLM on full mRNA + sRNA log2(CPM+1) — mask 15% positions, predict
from rest. Same architecture as Stage 1 (AttnLayer × n_layers).

Inputs:
  - /home/razer/v5_pathD/master_expression_matrix.parquet (genes × samples)
  - /home/razer/v5_pathD/sl1344_vocab.tsv (idx, name, locus_tag, type, ...)
  - /home/razer/v5_pathD/ortholog_lim_to_sl1344.tsv (lim_idx, sl1344_idx, ...)
  - Stage 1 ckpt: /home/razer/v5_pathD/checkpoints_razer_stage1_S<seed>/best.pt

Output:
  - {ckpt_dir}/best.pt — Stage 2 v2 ckpt (state_dict + vocab + warm-start metadata)
"""
from __future__ import annotations
import argparse
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, random_split

sys.path.insert(0, "/home/razer/v5_pathD")
from train_stage1_ecoli import Stage1Model, resolve_srna_edges  # noqa: E402

V5_PATH = Path("/home/razer/v5_pathD")
VOCAB_TSV = V5_PATH / "sl1344_vocab.tsv"
ORTHO_TSV = V5_PATH / "ortholog_lim_to_sl1344.tsv"
MATRIX_PARQUET = V5_PATH / "master_expression_matrix.parquet"


class SL1344ExpressionDataset(Dataset):
    """Salmonella samples (3328 NCBI) projected to 5098 SL1344 vocab."""

    def __init__(self, parquet_path: Path, vocab_df: pd.DataFrame,
                 min_total_reads: int = 1_000_000):
        mat = pd.read_parquet(parquet_path)
        print(f"  raw SL1344 matrix: {mat.shape}", flush=True)
        vocab_locus = vocab_df["locus_tag"].tolist()
        mat = mat.reindex(vocab_locus).fillna(0)
        sample_totals = mat.sum(axis=0)
        keep = sample_totals >= min_total_reads
        mat = mat.loc[:, keep]
        print(f"  SL1344 after QC ({min_total_reads:,} reads): "
              f"{mat.shape[1]}/{len(keep)} samples kept", flush=True)
        lib_size = mat.sum(axis=0).clip(lower=1)
        cpm = mat.div(lib_size, axis=1) * 1e6
        log_cpm = np.log2(cpm + 1.0).astype(np.float32)
        self.sample_ids = list(log_cpm.columns)
        self.expr = torch.from_numpy(log_cpm.values.T.copy())  # [N, G_sl=5098]
        self.has_mask = torch.from_numpy(
            (mat.values.T > 0).astype(np.bool_).copy()
        )
        self.species = "salmonella"

    def __len__(self) -> int:
        return len(self.sample_ids)

    def __getitem__(self, idx: int) -> dict:
        return {
            "sample_id": self.sample_ids[idx],
            "log_cpm": self.expr[idx],
            "has_mask": self.has_mask[idx],
            "species": self.species,
        }


class EcoliLimReplayDataset(Dataset):
    """Lim 2023 E. coli compendium projected to 5098 SL1344 vocab via ortholog.

    For each Lim sample:
      - log_cpm: 5098-dim vector; position v[i_sl] = Lim_log2TPM[i_lim] if
        ortholog mapping exists (Lim → SL1344), else 0
      - has_mask: True only at positions where (ortholog exists) AND
        (Lim has_mask was True at the corresponding Lim index)

    Positions not covered by any Lim ortholog are treated as missing — the
    trainer masks loss at those positions for E. coli samples.

    This lets us replay Lim during Stage 2 v2 training so the model retains
    cross-species transferable features, supervised by E. coli signal only
    on the ~2182 SL1344 positions that have a Lim ortholog.
    """

    def __init__(self, lim_parquet: Path, vocab_df: pd.DataFrame,
                 ortho_df: pd.DataFrame, min_nonnan_frac: float = 0.10):
        # Read raw Lim parquet — genes x (meta_cols + sample_cols)
        df = pd.read_parquet(lim_parquet)
        meta_cols = ["B#", "Name", "Type", "Start", "Stop", "Strand", "Product"]
        sample_cols = [c for c in df.columns if c not in meta_cols]

        # Build Lim gene name → Lim global idx (mRNA first, then sRNA)
        # match the original ecoli_lim_dataloader convention
        is_mrna = df["Type"] == "CDS"
        is_srna = df["Type"] == "ncRNA"
        lim_mrna = df[is_mrna].reset_index(drop=True)
        lim_srna = df[is_srna].reset_index(drop=True)
        n_lim_mrna = len(lim_mrna)
        n_lim_srna = len(lim_srna)

        # log2(TPM+1), NaN→0; QC by per-sample non-NaN fraction
        mrna_expr = lim_mrna[sample_cols].to_numpy(dtype=np.float32)
        srna_expr = lim_srna[sample_cols].to_numpy(dtype=np.float32)
        mrna_has = ~np.isnan(mrna_expr)
        srna_has = ~np.isnan(srna_expr)
        mrna_expr = np.nan_to_num(mrna_expr, nan=0.0)
        srna_expr = np.nan_to_num(srna_expr, nan=0.0)
        mrna_log = np.log2(mrna_expr + 1.0)
        srna_log = np.log2(srna_expr + 1.0)
        nonnan_frac = (mrna_has.sum(0) + srna_has.sum(0)) / \
                      (mrna_has.shape[0] + srna_has.shape[0])
        keep_idx = np.where(nonnan_frac >= min_nonnan_frac)[0]
        self.sample_ids = [sample_cols[i] for i in keep_idx]
        n_kept = len(self.sample_ids)
        print(f"  Lim: {n_kept}/{len(sample_cols)} samples passed nonnan QC "
              f"({min_nonnan_frac:.0%})", flush=True)

        # Build ortholog map Lim global idx → SL1344 global idx
        # Lim global indexing: 0..n_lim_mrna-1 = mRNA, n_lim_mrna..n_lim_mrna+n_lim_srna-1 = sRNA
        lim_to_sl = {}
        for _, row in ortho_df.iterrows():
            if row["sl1344_idx"] < 0:
                continue
            lim_to_sl[int(row["lim_idx"])] = int(row["sl1344_idx"])
        print(f"  Lim→SL1344 ortholog edges: {len(lim_to_sl)}", flush=True)

        # Project Lim N samples × (n_lim_mrna + n_lim_srna) → N × 5098 vocab
        G_sl = len(vocab_df)
        proj_expr = np.zeros((n_kept, G_sl), dtype=np.float32)
        proj_has = np.zeros((n_kept, G_sl), dtype=np.bool_)
        n_mrna_covered = 0
        n_srna_covered = 0
        for lim_i, sl_i in lim_to_sl.items():
            if lim_i < n_lim_mrna:
                # mRNA
                proj_expr[:, sl_i] = mrna_log[lim_i, keep_idx]
                proj_has[:, sl_i] = mrna_has[lim_i, keep_idx]
                n_mrna_covered += 1
            elif lim_i < n_lim_mrna + n_lim_srna:
                # sRNA
                srna_local = lim_i - n_lim_mrna
                if srna_local < n_lim_srna:
                    proj_expr[:, sl_i] = srna_log[srna_local, keep_idx]
                    proj_has[:, sl_i] = srna_has[srna_local, keep_idx]
                    n_srna_covered += 1

        print(f"  Lim coverage on SL1344 5098 vocab: "
              f"{n_mrna_covered} mRNA + {n_srna_covered} sRNA "
              f"({n_mrna_covered + n_srna_covered}/{G_sl} = "
              f"{100*(n_mrna_covered+n_srna_covered)/G_sl:.1f}%)", flush=True)

        self.expr = torch.from_numpy(proj_expr)
        self.has_mask = torch.from_numpy(proj_has)
        self.species = "ecoli_lim"

    def __len__(self) -> int:
        return len(self.sample_ids)

    def __getitem__(self, idx: int) -> dict:
        return {
            "sample_id": self.sample_ids[idx],
            "log_cpm": self.expr[idx],
            "has_mask": self.has_mask[idx],
            "species": self.species,
        }


def load_stage1_ckpt(path: str, device: str):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    return ckpt


def build_warm_start_state_dict(stage1_ckpt: dict,
                                 sl1344_vocab: pd.DataFrame,
                                 ortholog: pd.DataFrame,
                                 embed_dim: int,
                                 n_mrna_sl: int, n_srna_sl: int) -> dict:
    """Build a state_dict for Stage 2 v2 model with Lim embeddings transferred.

    Strategy:
      - gene_emb[i_sl1344]: copy from stage1.gene_emb[i_lim] if ortholog exists,
        else random init from N(0, 0.02²).
      - All other weights: copy from Stage 1 (architecturally compatible since
        n_layers, n_heads, embed_dim are reused).
    """
    print("  Building warm-start state_dict ...", flush=True)
    sd_stage1 = stage1_ckpt["state_dict"]
    n_genes_sl = n_mrna_sl + n_srna_sl
    n_mrna_lim = stage1_ckpt["n_mrna"]
    n_srna_lim = stage1_ckpt["n_srna"]
    n_genes_lim = n_mrna_lim + n_srna_lim

    # These tensors depend on edge count (51 in Stage 2 v2 vs 98 in Stage 1) —
    # leave them at constructor-time init for the new edge set.
    EDGE_DEPENDENT_KEYS = {
        "srna_bias.biases", "srna_bias.i_idx", "srna_bias.j_idx",
        "srna_gate.weights", "srna_gate.signs",
        "srna_gate.srna_idx", "srna_gate.tgt_idx",
    }
    sd_new = {}
    for k, v in sd_stage1.items():
        if k in EDGE_DEPENDENT_KEYS:
            continue  # skip — let new model's constructor-init values remain
        if k == "gene_emb.weight":
            new_emb = torch.randn(n_genes_sl, embed_dim) * 0.02
            mapping = {}
            for _, row in ortholog.iterrows():
                if row["sl1344_idx"] < 0:
                    continue
                lim_i = int(row["lim_idx"])
                sl_i = int(row["sl1344_idx"])
                if lim_i < n_genes_lim and sl_i < n_genes_sl:
                    new_emb[sl_i] = v[lim_i].clone()
                    mapping[sl_i] = lim_i
            sd_new[k] = new_emb
            print(f"    gene_emb: warm-started {len(mapping)}/{n_genes_sl} "
                  f"({100*len(mapping)/n_genes_sl:.1f}%)", flush=True)
        elif k == "type_marker":
            # Recompute for new vocab
            sd_new[k] = torch.cat([
                torch.ones(n_mrna_sl), torch.zeros(n_srna_sl)
            ]).unsqueeze(-1)
        else:
            # Architecturally identical components — copy directly
            sd_new[k] = v.clone() if isinstance(v, torch.Tensor) else v
    return sd_new


def train_one_epoch(model, loader, mask_frac, device, opt):
    model.train()
    tot_loss, n_batches = 0.0, 0
    for batch in loader:
        expr = batch["log_cpm"].to(device)  # [B, G]
        has = batch["has_mask"].to(device)
        rand = torch.rand_like(expr)
        mask = (rand < mask_frac) & has  # [B, G] bool (True = predict)
        if mask.sum() == 0:
            continue
        pred = model(expr, mask)  # Stage1Model.forward(expr, mask)
        loss = F.mse_loss(pred[mask], expr[mask])
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        tot_loss += loss.item()
        n_batches += 1
    return tot_loss / max(1, n_batches)


def eval_one(model, loader, mask_frac, device):
    model.eval()
    all_pred, all_true = [], []
    with torch.no_grad():
        for batch in loader:
            expr = batch["log_cpm"].to(device)
            has = batch["has_mask"].to(device)
            rand = torch.rand_like(expr)
            mask = (rand < mask_frac) & has
            if mask.sum() == 0:
                continue
            pred = model(expr, mask)
            all_pred.append(pred[mask].flatten().cpu())
            all_true.append(expr[mask].flatten().cpu())
    if not all_pred:
        return 0.0, 0.0
    pv = torch.cat(all_pred)
    tv = torch.cat(all_true)
    cos = F.cosine_similarity(pv.unsqueeze(0), tv.unsqueeze(0)).item()
    mse = F.mse_loss(pv, tv).item()
    return cos, mse


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage1-ckpt", required=True)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--mask-frac", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--val-frac", type=float, default=0.1)
    ap.add_argument("--ckpt-dir", required=True)
    ap.add_argument("--replay-frac", type=float, default=0.20,
                    help="Fraction of each batch drawn from Lim E. coli replay set")
    ap.add_argument("--lim-parquet",
                    default="/home/razer/v5_pathD/ecoli_data/lim2023_annotated.parquet")
    args = ap.parse_args()
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    print("=" * 70)
    print(" v5 Stage 2 v2 trainer — SL1344 full compendium")
    print(f"  stage1_ckpt: {args.stage1_ckpt}")
    print(f"  matrix: {MATRIX_PARQUET}")
    print(f"  epochs={args.epochs} bs={args.batch_size} lr={args.lr}")
    print(f"  device={args.device} seed={args.seed}")
    print("=" * 70)

    # [1] Vocab + ortholog
    print("\n[1/5] loading vocab + ortholog ...")
    vocab = pd.read_csv(VOCAB_TSV, sep="\t")
    n_mrna = int((vocab["type"] == "mrna").sum())
    n_srna = int((vocab["type"] == "srna").sum())
    n_genes = n_mrna + n_srna
    ortho = pd.read_csv(ORTHO_TSV, sep="\t")
    print(f"  SL1344 vocab: {n_genes} ({n_mrna} mRNA + {n_srna} sRNA)")
    print(f"  ortholog: {len(ortho)} Lim entries, "
          f"{(ortho['sl1344_idx'] >= 0).sum()} matched")

    # [2] Load Stage 1 ckpt
    print("\n[2/5] loading Stage 1 ckpt ...")
    stage1 = load_stage1_ckpt(args.stage1_ckpt, args.device)
    stage1_cfg = stage1.get("config", {})
    embed_dim = stage1_cfg.get("embed_dim", 128)
    n_heads = stage1_cfg.get("n_heads", 4)
    n_layers = stage1_cfg.get("n_layers", 4)
    print(f"  embed_dim={embed_dim} n_heads={n_heads} n_layers={n_layers}")
    print(f"  Stage 1: n_mrna={stage1['n_mrna']} n_srna={stage1['n_srna']} "
          f"val_cos={stage1.get('val_cos', float('nan')):+.4f}")

    # [3] Datasets: Salmonella SL1344 (primary) + Lim E. coli (replay)
    print("\n[3/5] loading expression matrices ...")
    if not MATRIX_PARQUET.exists():
        print(f"FATAL: {MATRIX_PARQUET} not found — run build_master_expression_matrix.py first",
              file=sys.stderr)
        sys.exit(1)
    sal_ds = SL1344ExpressionDataset(MATRIX_PARQUET, vocab)

    use_replay = args.replay_frac > 0
    if use_replay:
        ecoli_ds = EcoliLimReplayDataset(Path(args.lim_parquet), vocab, ortho)
    else:
        ecoli_ds = None
        print("  replay-frac = 0 → Lim replay disabled")

    # Split Salmonella into train/val (val is Salmonella only, so val cos
    # measures the metric users care about: cross-species held-out fit)
    n_val = int(args.val_frac * len(sal_ds))
    n_train_sal = len(sal_ds) - n_val
    train_sal, val_sal = random_split(
        sal_ds, [n_train_sal, n_val],
        generator=torch.Generator().manual_seed(args.seed),
    )

    if use_replay:
        from torch.utils.data import ConcatDataset, WeightedRandomSampler
        full_train = ConcatDataset([train_sal, ecoli_ds])
        # Weight: each Lim sample gets weight w_e, each Salmonella sample w_s
        # so that effective per-sample probability matches replay_frac.
        n_sal = len(train_sal)
        n_eco = len(ecoli_ds)
        # P(salmonella sample drawn) = (1 - replay_frac), P(ecoli) = replay_frac
        w_sal = (1.0 - args.replay_frac) / max(1, n_sal)
        w_eco = args.replay_frac / max(1, n_eco)
        weights = [w_sal] * n_sal + [w_eco] * n_eco
        # Total draws per epoch = n_sal (one epoch = one pass over salmonella)
        sampler = WeightedRandomSampler(weights, num_samples=n_sal, replacement=True,
                                         generator=torch.Generator().manual_seed(args.seed))
        train_loader = DataLoader(full_train, batch_size=args.batch_size,
                                   sampler=sampler)
        print(f"  train: {n_sal} Salmonella + {n_eco} Lim (replay frac {args.replay_frac:.0%}, "
              f"epoch draws {n_sal})")
    else:
        train_loader = DataLoader(train_sal, batch_size=args.batch_size, shuffle=True)
        print(f"  train: {n_train_sal} Salmonella only (no replay)")

    val_loader = DataLoader(val_sal, batch_size=args.batch_size, shuffle=False)
    print(f"  val: {n_val} Salmonella")

    # [4] Build Stage 2 v2 model — keep Stage 1 sRNA edges (mostly stable orthologs)
    print("\n[4/5] building model ...")
    edges = stage1.get("edges", [])
    effects = stage1.get("effects", [])
    # Remap edges by ortholog (Lim idx -> SL1344 idx)
    lim_to_sl1344 = dict(zip(ortho["lim_idx"], ortho["sl1344_idx"]))
    new_edges, new_effects = [], []
    for (src, dst), eff in zip(edges, effects):
        s = lim_to_sl1344.get(src, -1)
        d = lim_to_sl1344.get(dst, -1)
        if s >= 0 and d >= 0:
            new_edges.append((int(s), int(d)))
            new_effects.append(eff)
    print(f"  edges remapped: {len(new_edges)}/{len(edges)}")

    model = Stage1Model(
        n_mrna=n_mrna, n_srna=n_srna,
        srna_edges=new_edges, srna_effects=new_effects,
        embed_dim=embed_dim, n_heads=n_heads, n_layers=n_layers,
    ).to(args.device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  total params: {n_params/1e6:.2f} M")

    # Warm-start state_dict
    warm_sd = build_warm_start_state_dict(stage1, vocab, ortho, embed_dim,
                                            n_mrna, n_srna)
    missing, unexpected = model.load_state_dict(warm_sd, strict=False)
    print(f"  load_state_dict: missing {len(missing)} keys, unexpected {len(unexpected)}")

    # [5] Train
    print("\n[5/5] training ...")
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    os.makedirs(args.ckpt_dir, exist_ok=True)
    best_cos = -2.0
    history = []
    for epoch in range(args.epochs):
        t0 = time.time()
        train_loss = train_one_epoch(model, train_loader, args.mask_frac,
                                      args.device, opt)
        val_cos, val_mse = eval_one(model, val_loader, args.mask_frac, args.device)
        dt = time.time() - t0
        print(f"  epoch {epoch+1:>3}/{args.epochs}  train_mse={train_loss:.3f}  "
              f"val_cos={val_cos:+.4f}  val_mse={val_mse:.3f}  t={dt:.1f}s",
              flush=True)
        history.append({"epoch": epoch+1, "train_mse": train_loss,
                        "val_cos": val_cos, "val_mse": val_mse})
        if val_cos > best_cos:
            best_cos = val_cos
            torch.save({
                "epoch": epoch+1,
                "state_dict": model.state_dict(),
                "val_cos": val_cos,
                "config": vars(args),
                "stage1_config": stage1_cfg,
                "stage1_ckpt_path": args.stage1_ckpt,
                "edges": new_edges,
                "effects": new_effects,
                "n_mrna": n_mrna,
                "n_srna": n_srna,
                "vocab_names": vocab["name"].tolist(),
                "vocab_locus": vocab["locus_tag"].tolist(),
                "history": history,
            }, f"{args.ckpt_dir}/best.pt")
    print(f"\nDONE. best val_cos = {best_cos:+.4f}  ->  {args.ckpt_dir}/best.pt")


if __name__ == "__main__":
    main()
