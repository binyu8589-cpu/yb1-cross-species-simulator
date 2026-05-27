"""ecoli_lim_dataloader.py — Stage 1 (E. coli) expression loader.

Source: Lim et al. 2023 RNA Biology, Harvard Dataverse doi:10.7910/DVN/QBMC9D
File:   <data>/ecoli_data/lim2023_annotated.parquet

Schema: 4,510 annotated transcripts (4,304 CDS + 98 ncRNA + 86 tRNA + 22 rRNA)
        x 3,376 SRA Run IDs (Illumina RNA-seq + ncRNA-seq).
Values: TPM (or similar abundance), with '-' replaced by NaN. ~18% NaN cells.

Returns per-sample tensors:
    mrna_log_tpm  : [n_mrna]   ; log2(TPM + 1), NaN -> 0
    srna_log_tpm  : [n_srna=98]; log2(TPM + 1), NaN -> 0
    sample_id     : SRR string
    has_mrna_mask : [n_mrna]   ; True where original value was not NaN
    has_srna_mask : [n_srna]   ; True where original value was not NaN
"""
from __future__ import annotations
# --- repository-relative paths (override via env vars; see README) ---
import os as _os
_REPO = _os.environ.get("YB1_REPO", _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
_DATA = _os.environ.get("YB1_DATA", _os.path.join(_REPO, "data", "processed"))
_REF  = _os.environ.get("YB1_REF",  _os.path.join(_REPO, "data", "reference"))
_CKPT = _os.environ.get("YB1_CKPT", _os.path.join(_REPO, "checkpoints"))
# --- end repo-relative paths ---
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


PARQUET = _os.environ.get("LIM_PARQUET", _os.path.join(_DATA, "lim2023_annotated.parquet"))
META_PARQUET = _os.path.join(_REF, "lim2023_samples.parquet")

META_COLS = ("B#", "Name", "Type", "Start", "Stop", "Strand", "Product")


class LimEcoliExpressionDataset(Dataset):
    """One Dataset item == one SRA Run.

    Args:
        parquet: path to annotated transcript parquet.
        keep_types: which transcript Types to include for mRNA slot.
                    Default ('CDS',). Anything outside this AND outside 'ncRNA'
                    is dropped (rRNA, tRNA).
        min_nonnan_frac: drop samples whose total non-NaN fraction is below
                         this threshold (default 0.10 — Lim has some samples
                         with very sparse coverage).
    """

    def __init__(self,
                 parquet: str = PARQUET,
                 keep_types: tuple[str, ...] = ("CDS",),
                 min_nonnan_frac: float = 0.10):
        df = pd.read_parquet(parquet)
        meta_cols = list(META_COLS)
        sample_cols = [c for c in df.columns if c not in meta_cols]
        meta = df[meta_cols].copy()

        mrna_mask = meta["Type"].isin(keep_types)
        srna_mask = meta["Type"] == "ncRNA"
        self.mrna_meta = meta[mrna_mask].reset_index(drop=True)
        self.srna_meta = meta[srna_mask].reset_index(drop=True)

        # Expression matrices (transcripts x samples)
        mrna_expr = df.loc[mrna_mask, sample_cols].to_numpy(dtype=np.float32)
        srna_expr = df.loc[srna_mask, sample_cols].to_numpy(dtype=np.float32)

        # NaN -> 0 after recording presence mask
        mrna_has = ~np.isnan(mrna_expr)
        srna_has = ~np.isnan(srna_expr)
        mrna_expr = np.nan_to_num(mrna_expr, nan=0.0)
        srna_expr = np.nan_to_num(srna_expr, nan=0.0)

        # log2(TPM + 1)
        mrna_log = np.log2(mrna_expr + 1.0)
        srna_log = np.log2(srna_expr + 1.0)

        # Per-sample QC: drop sparse samples
        nonnan_frac = (mrna_has.sum(0) + srna_has.sum(0)) / \
                      (mrna_has.shape[0] + srna_has.shape[0])
        keep_idx = np.where(nonnan_frac >= min_nonnan_frac)[0]
        self.sample_ids = [sample_cols[i] for i in keep_idx]
        self.mrna_log = torch.from_numpy(mrna_log[:, keep_idx].T.copy())  # [N, G_mrna]
        self.srna_log = torch.from_numpy(srna_log[:, keep_idx].T.copy())  # [N, G_srna]
        self.mrna_has = torch.from_numpy(mrna_has[:, keep_idx].T.copy())
        self.srna_has = torch.from_numpy(srna_has[:, keep_idx].T.copy())

        self.n_mrna = self.mrna_log.shape[1]
        self.n_srna = self.srna_log.shape[1]

    def __len__(self) -> int:
        return len(self.sample_ids)

    def __getitem__(self, idx: int) -> dict:
        return {
            "sample_id": self.sample_ids[idx],
            "mrna_log_tpm": self.mrna_log[idx],
            "srna_log_tpm": self.srna_log[idx],
            "has_mrna_mask": self.mrna_has[idx],
            "has_srna_mask": self.srna_has[idx],
        }

    def srna_names(self) -> list[str]:
        return self.srna_meta["Name"].tolist()

    def mrna_names(self) -> list[str]:
        return self.mrna_meta["Name"].tolist()


if __name__ == "__main__":
    ds = LimEcoliExpressionDataset()
    print(f"samples kept: {len(ds)} / 3376")
    print(f"  n_mrna={ds.n_mrna}  n_srna={ds.n_srna}")
    print(f"  first 8 sRNA names: {ds.srna_names()[:8]}")
    item = ds[0]
    print(f"  sample_id={item['sample_id']}")
    print(f"  mrna_log_tpm range: "
          f"[{item['mrna_log_tpm'].min():.2f}, {item['mrna_log_tpm'].max():.2f}], "
          f"has-frac {item['has_mrna_mask'].float().mean():.2f}")
    print(f"  srna_log_tpm range: "
          f"[{item['srna_log_tpm'].min():.2f}, {item['srna_log_tpm'].max():.2f}], "
          f"has-frac {item['has_srna_mask'].float().mean():.2f}")
    # sanity: pick MicC / GcvB / RyhB and see expression range across samples
    for name in ("micC", "gcvB", "ryhB", "oxyS", "rprA"):
        if name in [n.lower() for n in ds.srna_names()]:
            idx = [n.lower() for n in ds.srna_names()].index(name)
            vals = ds.srna_log[:, idx]
            obs = ds.srna_has[:, idx].sum().item()
            print(f"  {name:>5}: idx={idx}, observed in "
                  f"{obs}/{len(ds)} samples, "
                  f"mean log2(TPM+1)={vals[ds.srna_has[:, idx]].mean():.2f}")
