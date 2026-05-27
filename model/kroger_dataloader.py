"""kroger_dataloader.py — load Kröger 2013 mmc3 22-condition sRNA TPM matrix.

Source TSV: <data>/wetlab_scripts/kroger_srnas.tsv
Schema: SL identifier, common gene name, strand, start, end, then 22+ condition
columns of TPM values (ESP/LSP/InSPI2/Anaerobic/Bile/Cold shock/Peroxide/...).

Used as supervised expression target for sRNA nodes in v5 Path D training.
"""
from __future__ import annotations
# --- repository-relative paths (override via env vars; see README) ---
import os as _os
_REPO = _os.environ.get("YB1_REPO", _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
_DATA = _os.environ.get("YB1_DATA", _os.path.join(_REPO, "data", "processed"))
_REF  = _os.environ.get("YB1_REF",  _os.path.join(_REPO, "data", "reference"))
_CKPT = _os.environ.get("YB1_CKPT", _os.path.join(_REPO, "checkpoints"))
# --- end repo-relative paths ---
import csv
import math
from pathlib import Path

import torch
from torch.utils.data import Dataset


KROGER_TSV = _os.path.join(_REF, "kroger_srnas.tsv")

# Numeric condition columns (header text -> short condition id).
CONDITION_MAP = {
    "EEP": "EEP",
    "MEP": "MEP",
    "LEP": "LEP",
    "ESP average (Biol. Rep. 1 and 2)": "ESP",
    "LSP average (Biol. Rep. 1 and 2)": "LSP",
    "25\xc2\xb0C": "T25",
    "25°C": "T25",
    "Cold shock (15\xc2\xb0C)": "Cold",
    "Cold shock (15°C)": "Cold",
    "pH3 shock": "pH3",
    "pH5.8 shock": "pH58",
    "NaCl shock": "NaCl",
    "Bile shock": "Bile",
    "LowFe2+ shock": "LowFe",
    "Anaerobic shock": "AnaShock",
    "Anaerobic growth": "AnaGrowth",
    "Aerobic shock": "AerShock",
    "InSPI2 average (Biol. Rep. 1 and 2)": "InSPI2",
    "InSPI2 LowMg2+": "InSPI2_LowMg",
    "Peroxide shock (InSPI2)": "Peroxide",
    "Nitric oxide shock (InSPI2)": "NO",
    "NonSPI2": "NonSPI2",
    "Temp10": "Temp10",
    "Temp20": "Temp20",
    "RNA pool": "Pool",
}


def parse_float(s: str) -> float:
    try:
        return float(s)
    except (ValueError, TypeError):
        return float("nan")


def load_kroger_matrix(tsv_path: str = KROGER_TSV
                       ) -> tuple[list[str], list[str], torch.Tensor]:
    """Return (srna_names, condition_ids, tpm_matrix[n_srna, n_cond])."""
    with open(tsv_path) as f:
        reader = csv.reader(f, delimiter="\t")
        header = next(reader)
        cond_cols: list[tuple[int, str]] = []
        for i, h in enumerate(header):
            if h in CONDITION_MAP:
                cond_cols.append((i, CONDITION_MAP[h]))
        srna_names: list[str] = []
        rows: list[list[float]] = []
        for line in reader:
            if not line or not line[0]:
                continue
            name = line[1].strip() or line[0].strip()
            row = [parse_float(line[i]) for i, _ in cond_cols]
            srna_names.append(name)
            rows.append(row)
    cond_ids = [c for _, c in cond_cols]
    mat = torch.tensor(rows, dtype=torch.float32)
    # NaN -> 0 (Kröger uses blanks for missing)
    mat = torch.nan_to_num(mat, nan=0.0)
    return srna_names, cond_ids, mat


class KrogerExpressionDataset(Dataset):
    """One sample per condition: returns (cond_idx, sRNA log-TPM vector).

    log_tpm = log2(tpm + 1) — standard sRNA-quantification normalization.
    """

    def __init__(self, tsv_path: str = KROGER_TSV):
        self.srna_names, self.cond_ids, tpm = load_kroger_matrix(tsv_path)
        self.log_tpm = torch.log2(tpm + 1.0)  # [n_srna, n_cond]
        self.n_srna = self.log_tpm.shape[0]
        self.n_cond = self.log_tpm.shape[1]

    def __len__(self) -> int:
        return self.n_cond

    def __getitem__(self, idx: int):
        return idx, self.log_tpm[:, idx]


if __name__ == "__main__":
    ds = KrogerExpressionDataset()
    print(f"Kroger: n_srna={ds.n_srna} n_cond={ds.n_cond}")
    print(f"  conditions: {ds.cond_ids}")
    print(f"  log_tpm range: [{ds.log_tpm.min():.2f}, {ds.log_tpm.max():.2f}]")
    print(f"  mean per condition (first 5): "
          f"{[round(v, 2) for v in ds.log_tpm.mean(0)[:5].tolist()]}")
    print(f"  sample 0 (cond={ds.cond_ids[0]}): "
          f"first 5 sRNA log2(tpm+1) = "
          f"{[round(v, 2) for v in ds.log_tpm[:5, 0].tolist()]}")
