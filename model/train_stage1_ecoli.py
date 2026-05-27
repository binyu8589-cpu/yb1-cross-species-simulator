"""train_stage1_ecoli.py - v5 Stage 1 trainer (E. coli, Lim 2023 compendium).

Training task: masked expression modeling (BERT-style on biology).
  - Per sample: full expression vector [G] = mRNA + sRNA log2(TPM+1).
  - Mask 15 % of OBSERVED positions; model predicts those values from the
    remaining context.
  - Loss: MSE on masked positions only.
  - sRNA -> target literature edges injected as attention bias (init -0.5
    silencing / +0.5 activating), per-edge learnable.

Two modes:
  --mode smoke : 32-dim, 2 layers, 100 samples, 3 epochs (~3 min on Mac CPU)
  --mode full  : 128-dim, 4 layers, 3161 samples, 50 epochs (server GPU)

Output: best ckpt at {ckpt_dir}/best.pt with state_dict + val_cos metric.
"""
from __future__ import annotations
import argparse
import os
import random
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset, random_split

from ecoli_lim_dataloader import LimEcoliExpressionDataset
from srna_layer import SrnaEdgeBias, register_srna_hooks, SrnaGate

LIM_META_PARQUET = "/home/razer/v5_pathD/ecoli_data/lim2023_samples.parquet"

PERTURBATION_KEYWORDS = {
    "hfq":      ["hfq", "Hfq"],
    "KO":       ["knockout", "KO", "delta", "mutant", "Δ"],
    "anaerob":  ["anaerob", "oxygen"],
    "overexpr": ["overexpr", "pBAD", "IPTG"],
    "iron":     ["iron", "fur"],
}


def get_perturbation_sample_ids() -> set[str]:
    """Return Run IDs flagged as perturbation samples in Lim 2023 metadata."""
    import pandas as pd
    meta = pd.read_parquet(LIM_META_PARQUET)
    text_cols = ["SampleName", "LibraryName", "Experiment"]
    flag = pd.Series([False] * len(meta), index=meta.index)
    for kws in PERTURBATION_KEYWORDS.values():
        for col in text_cols:
            if col not in meta.columns:
                continue
            s = meta[col].fillna("").astype(str)
            for kw in kws:
                flag |= s.str.contains(kw, case=False, regex=False)
    return set(meta.loc[flag, "Run"].tolist())

SRNA_EDGES_TSV = "/home/razer/v5_pathD/wetlab_scripts/srna_target_edges_v1.tsv"


class AttnLayer(nn.Module):
    """Pre-norm transformer block whose .attn is MultiheadAttention (hookable)."""

    def __init__(self, embed_dim: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, n_heads, dropout=dropout,
                                          batch_first=True)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.ff = nn.Sequential(
            nn.Linear(embed_dim, 4 * embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(4 * embed_dim, embed_dim),
        )

    def forward(self, x, attn_mask=None):
        h = self.norm1(x)
        h, _ = self.attn(h, h, h, attn_mask=attn_mask, need_weights=False)
        x = x + h
        x = x + self.ff(self.norm2(x))
        return x


class Stage1Model(nn.Module):
    def __init__(self, n_mrna: int, n_srna: int,
                 srna_edges: list[tuple[int, int]],
                 srna_effects: list[str],
                 embed_dim: int = 128, n_heads: int = 4,
                 n_layers: int = 4, dropout: float = 0.1):
        super().__init__()
        self.n_mrna, self.n_srna = n_mrna, n_srna
        self.n_genes = n_mrna + n_srna
        self.embed_dim = embed_dim

        # Per-gene learnable token embedding (gene identity)
        self.gene_emb = nn.Embedding(self.n_genes, embed_dim)
        # Project scalar expression to embedding space (conditioning)
        self.expr_in = nn.Linear(1, embed_dim)
        # Learnable mask token replacing expr_in at masked positions
        self.mask_token = nn.Parameter(torch.randn(embed_dim) * 0.02)
        # Marker bit (1.0 if mRNA, 0.0 if sRNA) — small but useful for the head
        marker = torch.cat([
            torch.ones(n_mrna), torch.zeros(n_srna)
        ]).unsqueeze(-1)
        self.register_buffer("type_marker", marker)
        self.type_in = nn.Linear(1, embed_dim, bias=False)

        self.layers = nn.ModuleList([
            AttnLayer(embed_dim, n_heads, dropout) for _ in range(n_layers)
        ])

        self.srna_bias = SrnaEdgeBias(
            edges=srna_edges, effects=srna_effects,
            init_magnitude=0.5, n_genes=self.n_genes,
        )
        self.hook_handles = register_srna_hooks(self.layers, self.srna_bias)

        self.head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, 1),
        )

        self.srna_gate = SrnaGate(
            edges=srna_edges, effects=srna_effects,
            n_genes=self.n_genes, init_weight=0.5,
        )

    def forward(self, expr: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """expr [B, G] log_tpm; mask [B, G] bool (True = predict)."""
        B, G = expr.shape
        ids = torch.arange(G, device=expr.device)
        gene_e = self.gene_emb(ids).unsqueeze(0).expand(B, -1, -1)        # [B,G,D]
        type_e = self.type_in(self.type_marker).unsqueeze(0).expand(B, -1, -1)
        expr_e = self.expr_in(expr.unsqueeze(-1))                          # [B,G,D]
        x = gene_e + type_e
        # observed: add expr projection; masked: add mask token
        m = mask.unsqueeze(-1).float()
        x = x + (1.0 - m) * expr_e + m * self.mask_token
        for layer in self.layers:
            x = layer(x)
        pred = self.head(x).squeeze(-1)                                    # [B,G]
        pred = self.srna_gate(pred)                                        # sRNA modulation
        return pred


def resolve_srna_edges(mrna_names: list[str], srna_names: list[str]
                       ) -> tuple[list[tuple[int, int]], list[str], dict]:
    """Match literature edges to Lim vocabulary (case-insensitive)."""
    name_to_idx = {}
    for i, n in enumerate(mrna_names):
        name_to_idx[n.lower()] = i
    for i, n in enumerate(srna_names):
        name_to_idx[n.lower()] = len(mrna_names) + i

    edges, effects = [], []
    stats = {"kept": 0, "srna_miss": 0, "target_miss": 0, "total": 0}
    with open(SRNA_EDGES_TSV) as f:
        header = f.readline().rstrip().split("\t")
        col = {h: i for i, h in enumerate(header)}
        for line in f:
            parts = line.rstrip().split("\t")
            if not parts or not parts[0]:
                continue
            stats["total"] += 1
            s = parts[col["srna"]].lower()
            t = parts[col["target_gene"]].lower()
            si = name_to_idx.get(s)
            ti = name_to_idx.get(t)
            if si is None:
                stats["srna_miss"] += 1
                continue
            if ti is None:
                stats["target_miss"] += 1
                continue
            edges.append((si, ti))
            effects.append(parts[col["effect"]])
            stats["kept"] += 1
    return edges, effects, stats


def batch_to_expr(batch, device):
    mrna = batch["mrna_log_tpm"].to(device)
    srna = batch["srna_log_tpm"].to(device)
    has_m = batch["has_mrna_mask"].to(device)
    has_s = batch["has_srna_mask"].to(device)
    return (torch.cat([mrna, srna], dim=1),
            torch.cat([has_m, has_s], dim=1))


def make_mask(expr, has, mask_frac, structured_target_idx=None):
    """Returns the boolean mask of positions to predict.

    structured_target_idx: tensor of gene indices that should ALWAYS be masked.
        If provided, ignores mask_frac and just masks those positions.
    """
    if structured_target_idx is not None:
        m = torch.zeros_like(expr, dtype=torch.bool)
        m[:, structured_target_idx] = True
        return m & has
    return (torch.rand_like(expr) < mask_frac) & has


def train_epoch(model, loader, opt, mask_frac, device,
                structured_target_idx=None):
    model.train()
    total, n_batches = 0.0, 0
    for batch in loader:
        expr, has = batch_to_expr(batch, device)
        mask = make_mask(expr, has, mask_frac, structured_target_idx)
        if mask.sum() == 0:
            continue
        pred = model(expr, mask)
        loss = F.mse_loss(pred[mask], expr[mask])
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        total += loss.item()
        n_batches += 1
    return total / max(1, n_batches)


@torch.no_grad()
def evaluate(model, loader, mask_frac, device,
             structured_target_idx=None):
    """Returns (mean_cosine_on_masked, mean_mse_on_masked, n)."""
    model.eval()
    cos_sum, mse_sum, n = 0.0, 0.0, 0
    for batch in loader:
        expr, has = batch_to_expr(batch, device)
        mask = make_mask(expr, has, mask_frac, structured_target_idx)
        pred = model(expr, mask)
        for i in range(expr.shape[0]):
            m = mask[i]
            if m.sum() < 5:
                continue
            p, t = pred[i, m], expr[i, m]
            cos = F.cosine_similarity(p.unsqueeze(0), t.unsqueeze(0)).item()
            mse = F.mse_loss(p, t).item()
            cos_sum += cos
            mse_sum += mse
            n += 1
    return (cos_sum / max(1, n), mse_sum / max(1, n), n)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["smoke", "full"], default="smoke")
    ap.add_argument("--epochs", type=int)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--mask-frac", type=float, default=0.15)
    ap.add_argument("--embed-dim", type=int)
    ap.add_argument("--n-layers", type=int)
    ap.add_argument("--n-heads", type=int, default=4)
    ap.add_argument("--max-samples", type=int)
    ap.add_argument("--ckpt-dir", default="/Users/yubin/v2_data/v5_pathD/checkpoints_stage1")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--no-srna-edges", action="store_true",
                    help="Ablation: disable sRNA bias hook (toggle off).")
    ap.add_argument("--init-magnitude", type=float, default=0.5,
                    help="Initial |bias| for sRNA edges. Default 0.5.")
    ap.add_argument("--use-srna-gate", action="store_true",
                    help="Enable multiplicative sRNA gate after head.")
    ap.add_argument("--no-srna-gate", action="store_true",
                    help="Disable multiplicative sRNA gate (default off anyway).")
    ap.add_argument("--structured-mask", action="store_true",
                    help="Task v2: always mask sRNA TARGETS (never sRNAs). "
                    "Forces model to predict targets through sRNA context.")
    ap.add_argument("--test-perturbation", action="store_true",
                    help="Task v3 (cross-condition): test set = Lim "
                    "perturbation samples (hfq/KO/anaerob/overexpr/iron); "
                    "train set = vanilla samples.")
    args = ap.parse_args()
    torch.manual_seed(args.seed)
    random.seed(args.seed)

    if args.mode == "smoke":
        args.epochs = args.epochs or 3
        args.embed_dim = args.embed_dim or 32
        args.n_layers = args.n_layers or 2
        args.max_samples = args.max_samples or 100
    else:
        args.epochs = args.epochs or 50
        args.embed_dim = args.embed_dim or 128
        args.n_layers = args.n_layers or 4

    print("=" * 70)
    print(f" v5 Stage 1 trainer  mode={args.mode}")
    print(f"   epochs={args.epochs} batch={args.batch_size} lr={args.lr}")
    print(f"   embed_dim={args.embed_dim} n_layers={args.n_layers} "
          f"n_heads={args.n_heads}")
    print("=" * 70)

    # === Data ===
    print("\n[1/4] loading Lim 2023 compendium ...")
    full_ds = LimEcoliExpressionDataset()
    print(f"      {len(full_ds)} samples, "
          f"n_mrna={full_ds.n_mrna} n_srna={full_ds.n_srna}")

    if args.test_perturbation:
        # Cross-condition split: vanilla -> train, perturbation -> test
        pert_ids = get_perturbation_sample_ids()
        sample_ids = full_ds.sample_ids
        pert_idx = [i for i, sid in enumerate(sample_ids) if sid in pert_ids]
        vanilla_idx = [i for i, sid in enumerate(sample_ids) if sid not in pert_ids]
        random.seed(args.seed)
        random.shuffle(vanilla_idx)
        if args.max_samples and len(vanilla_idx) > args.max_samples:
            vanilla_idx = vanilla_idx[:args.max_samples]
        train_ds = Subset(full_ds, vanilla_idx)
        val_ds = Subset(full_ds, pert_idx)
        print(f"      [CROSS-CONDITION] train={len(train_ds)} vanilla, "
              f"test={len(val_ds)} perturbation")
    else:
        ds = full_ds
        if args.max_samples and len(ds) > args.max_samples:
            random.seed(42)
            indices = random.sample(range(len(full_ds)), args.max_samples)
            ds = Subset(full_ds, indices)
            print(f"      subsampled to {len(ds)} (max_samples={args.max_samples})")

        n_train = max(1, int(len(ds) * 0.9))
        n_val = max(1, len(ds) - n_train)
        train_ds, val_ds = random_split(
            ds, [n_train, n_val],
            generator=torch.Generator().manual_seed(42))
        print(f"      split: train={len(train_ds)} val={len(val_ds)}")
    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size,
                            shuffle=False, num_workers=0)

    # === Edges ===
    print("\n[2/4] resolving sRNA -> target literature edges ...")
    edges, effects, stats = resolve_srna_edges(
        full_ds.mrna_names(), full_ds.srna_names())
    print(f"      kept {stats['kept']}/{stats['total']} edges "
          f"(srna_miss={stats['srna_miss']} target_miss={stats['target_miss']})")
    print(f"      polarity: {effects.count('-')} silencing, "
          f"{effects.count('+')} activating")

    # === Model ===
    print("\n[3/4] building model ...")
    model = Stage1Model(
        n_mrna=full_ds.n_mrna,
        n_srna=full_ds.n_srna,
        srna_edges=edges,
        srna_effects=effects,
        embed_dim=args.embed_dim,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
    ).to(args.device)
    # Apply CLI init magnitude (overrides default 0.5 in Stage1Model)
    if args.init_magnitude != 0.5:
        with torch.no_grad():
            sign = model.srna_bias.biases.data.sign()
            model.srna_bias.biases.data = sign * args.init_magnitude
            print(f"      sRNA bias init_magnitude rescaled to "
                  f"±{args.init_magnitude}")
    n_p = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"      n_genes={model.n_genes}  trainable_params={n_p/1e6:.2f} M")

    if args.no_srna_edges:
        model.srna_bias.toggle(False)
        print(f"      [ABLATION] sRNA bias TOGGLED OFF — no attention bias")
    else:
        print(f"      sRNA bias active: {len(model.srna_bias.edges)} edges")
    if args.use_srna_gate and not args.no_srna_gate:
        model.srna_gate.toggle(True)
        print(f"      sRNA GATE active: {len(model.srna_gate.edges)} edges "
              f"(multiplicative)")
    else:
        model.srna_gate.toggle(False)
        print(f"      sRNA gate OFF (multiplicative)")

    # Structured mask: always mask sRNA targets
    structured_target_idx = None
    if args.structured_mask:
        target_set = sorted({ti for _, ti in edges})
        structured_target_idx = torch.tensor(target_set, dtype=torch.long,
                                              device=args.device)
        print(f"      [STRUCTURED MASK] always masking "
              f"{len(target_set)} sRNA targets (sRNAs themselves stay visible)")

    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    # === Train ===
    print("\n[4/4] training ...")
    os.makedirs(args.ckpt_dir, exist_ok=True)
    best_val = -2.0
    history = []
    for epoch in range(args.epochs):
        t0 = time.time()
        train_mse = train_epoch(model, train_loader, opt,
                                args.mask_frac, args.device,
                                structured_target_idx)
        val_cos, val_mse, val_n = evaluate(
            model, val_loader, args.mask_frac, args.device,
            structured_target_idx)
        dt = time.time() - t0
        line = (f"  epoch {epoch+1:>3}/{args.epochs}  "
                f"train_mse={train_mse:.3f}  "
                f"val_cos={val_cos:+.4f}  val_mse={val_mse:.3f}  "
                f"({val_n} val samples)  t={dt:.1f}s")
        print(line, flush=True)
        history.append({"epoch": epoch + 1, "train_mse": train_mse,
                        "val_cos": val_cos, "val_mse": val_mse, "dt_s": dt})
        if val_cos > best_val:
            best_val = val_cos
            ckpt = {
                "epoch": epoch + 1,
                "state_dict": model.state_dict(),
                "val_cos": val_cos,
                "val_mse": val_mse,
                "config": vars(args),
                "edges": ([] if args.no_srna_edges else edges),
                "effects": ([] if args.no_srna_edges else effects),
                "n_mrna": full_ds.n_mrna,
                "n_srna": full_ds.n_srna,
                "mrna_names": full_ds.mrna_names(),
                "srna_names": full_ds.srna_names(),
                "history": history,
            }
            torch.save(ckpt, f"{args.ckpt_dir}/best.pt")

    print(f"\nDONE.  best val_cos = {best_val:+.4f}  ->  {args.ckpt_dir}/best.pt")


if __name__ == "__main__":
    main()
