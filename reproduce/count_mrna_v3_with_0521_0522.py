"""count_mrna_v3_with_0521_0522.py

Count primary reads at SL1344 mRNA gene coordinates (4731 protein-coding genes)
on all 24 in-house BAMs (20 prior + 4 new from 2026-05-21 / 0522).

Adds (relative to v2):
  - PW_aer_0521 / PW_ana_0521          (PW bio-rep 3, all DAP)
  - SL7207_ana_0522                    (WT, NO DAP -- 2nd anaerobic DAP-free arm)
  - SL7207_dap_ana_0522                (WT + DAP -- pairs with 0520 => anaerobic
                                        DAP-matched WT control n=2, the gating
                                        replicate for updating paper §3.7)

Output: mrna_counts_v3_with_0521_0522.tsv (name, count_per_bam, cpm_per_bam).
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
import pysam

REF_GFF = _os.environ.get("SL1344_GFF", "/Users/yubin/v2_data/wetlab/references/sl1344.gff")
OUT_TSV = _os.path.join(_os.environ.get("WETLAB_FACTORIAL", "/Users/yubin/v2_data/wetlab/nanopore/v4_full_factorial"), "mrna_counts_v3_with_0521_0522.tsv")

ROOT = _os.environ.get("WETLAB_NANOPORE", "/Users/yubin/v2_data/wetlab/nanopore")
BAMS = {
    "YB1_aer_MinION":     f"{ROOT}/YB1_aerobic-MEP/YB1_aerobic-MEP.allreads_v3.sorted.bam",
    "YB1_aer_0508":       f"{ROOT}/RNA数据(高通量芯片)/20260508-YB1-MEP/YB1_aerobic_0508.sorted.bam",
    "YB1_ana_v2":         f"{ROOT}/YB1_anaerobic/YB1_anaerobic.allreads_v2.sorted.bam",
    "YB1_ana_0507":       f"{ROOT}/RNA数据(高通量芯片)/20260507-YB1-anaerobic/no_sample_id/20260507_1505_P2I-00303-A_PBK52705_b8ca0b72/YB1_anaerobic_rep2_0507.sorted.bam",
    "YB1_ana_0508":       f"{ROOT}/RNA数据(高通量芯片)/20260508-YB1-anaerobic/YB1_anaerobic_0508.sorted.bam",
    "YB1_aer_clone_0516": f"{ROOT}/RNA数据(高通量芯片)/20260516-YB1-平板克隆-MEP/YB1_aer_clone_0516.sorted.bam",
    "YB1_ana_clone_0516": f"{ROOT}/RNA数据(高通量芯片)/20260516-YB1-平板克隆-anaerobic/YB1_ana_clone_0516.sorted.bam",
    "SL7207_aer_0507":    f"{ROOT}/RNA数据(高通量芯片)/20260507-7207-MEP/no_sample_id/20260507_1515_P2I-00303-B_PBK59906_01e1fecf/SL7207_MEP_0507.sorted.bam",
    "SL7207_aer_0509":    f"{ROOT}/RNA数据(高通量芯片)/20260509-7207-MEP/SL7207_MEP_0509.sorted.bam",
    "SL7207_ana_0509":    f"{ROOT}/RNA数据(高通量芯片)/20260509-7207-anaerobic/SL7207_anaerobic_0509.sorted.bam",
    "dAsd_aer_0511":      f"{ROOT}/RNA数据(高通量芯片)/20260511-S014-MEP/delta_asd_aerobic_0511.sorted.bam",
    "dAsd_ana_0512":      f"{ROOT}/RNA数据(高通量芯片)/20260512-S014-anaerobic/delta_asd_anaerobic_0512.sorted.bam",
    "dAsd_aer_0514":      f"{ROOT}/RNA数据(高通量芯片)/20260514-S014-MEP/delta_asd_aerobic_0514.sorted.bam",
    "dAsd_ana_0514":      f"{ROOT}/RNA数据(高通量芯片)/20260514-S014-anaerobic/delta_asd_anaerobic_0514.sorted.bam",
    "PW_aer_0515":        f"{ROOT}/RNA数据(高通量芯片)/20260515-PW-DAP-MEP/pw_dap_aerobic_0515.sorted.bam",
    "PW_ana_0515":        f"{ROOT}/RNA数据(高通量芯片)/20260515-PW-DAP-anaerobic/pw_dap_anaerobic_0515.sorted.bam",
    "PW_aer_0518":        f"{ROOT}/RNA数据(高通量芯片)/20260518-PW-平板克隆-DAP-MEP/pw_dap_aerobic_0518.sorted.bam",
    "PW_ana_0518":        f"{ROOT}/RNA数据(高通量芯片)/20260518-PW-平板克隆-DAP-anaerobic/pw_dap_anaerobic_0518.sorted.bam",
    "SL7207_dap_aer_0520": f"{ROOT}/RNA数据(高通量芯片)/20260520-7207-dap-MEM/sl7207_dap_aerobic_0520.sorted.bam",
    "SL7207_dap_ana_0520": f"{ROOT}/RNA数据(高通量芯片)/20260520-7207-dap-anaerobic/sl7207_dap_anaerobic_0520.sorted.bam",
    # NEW 2026-05-21 PW bio-rep 3 (all DAP); MEM folder == aerobic
    "PW_aer_0521":        f"{ROOT}/RNA数据(高通量芯片)/20260521-PW-平板克隆-DAP-MEM/pw_dap_aerobic_0521.sorted.bam",
    "PW_ana_0521":        f"{ROOT}/RNA数据(高通量芯片)/20260521-PW-平板克隆-DAP-anaerobic/pw_dap_anaerobic_0521.sorted.bam",
    # NEW 2026-05-22 SL7207 anaerobic, DAP-matched bio-rep 2
    "SL7207_ana_0522":     f"{ROOT}/RNA数据(高通量芯片)/20260522-7207-平板克隆-anaerobic/sl7207_anaerobic_0522.sorted.bam",
    "SL7207_dap_ana_0522": f"{ROOT}/RNA数据(高通量芯片)/20260522-7207-平板克隆-DAP-anaerobic/sl7207_dap_anaerobic_0522.sorted.bam",
}
BAMS = {k: v.replace("RNA数据(高通量芯片)", "RNA数据（高通量芯片）") for k, v in BAMS.items()}


def parse_mrna_gff(gff_path):
    """Returns list of (name, chrom, start, end, strand) for protein-coding genes."""
    out = []
    seen_locus = set()
    with open(gff_path) as fh:
        for ln in fh:
            if ln.startswith("#") or not ln.strip():
                continue
            cols = ln.rstrip("\n").split("\t")
            if len(cols) < 9 or cols[2] != "gene":
                continue
            attrs = dict(kv.split("=", 1) for kv in cols[8].split(";") if "=" in kv)
            if attrs.get("gene_biotype") != "protein_coding":
                continue
            locus = attrs.get("locus_tag")
            name = attrs.get("Name") or attrs.get("gene") or locus
            if not locus or locus in seen_locus:
                continue
            seen_locus.add(locus)
            try:
                start = int(cols[3])
                end = int(cols[4])
            except ValueError:
                continue
            out.append((name, cols[0], start, end, cols[6]))
    return out


def count_bam(bam_path, mrna_list):
    """{name: count} for primary reads overlapping each mRNA."""
    counts = {n: 0 for n, *_ in mrna_list}
    total = 0
    if not Path(bam_path).exists():
        print(f"  WARN: BAM missing: {bam_path}")
        return counts, 0
    with pysam.AlignmentFile(bam_path, "rb") as f:
        for r in f.fetch(until_eof=True):
            if r.is_secondary or r.is_supplementary or r.is_unmapped:
                continue
            total += 1
        for name, chrom, start, end, strand in mrna_list:
            n = 0
            try:
                for r in f.fetch(chrom, start - 1, end):
                    if r.is_secondary or r.is_supplementary or r.is_unmapped:
                        continue
                    n += 1
            except (KeyError, ValueError):
                pass
            counts[name] = n
    return counts, total


def main():
    mrnas = parse_mrna_gff(REF_GFF)
    print(f"  parsed {len(mrnas)} mRNA gene records from {REF_GFF}")

    all_counts = {}
    totals = {}
    for label, bam in BAMS.items():
        print(f"  counting {label} ...", flush=True)
        counts, total = count_bam(bam, mrnas)
        all_counts[label] = counts
        totals[label] = total
        print(f"    {label}: {total:,} primary reads")

    Path(OUT_TSV).parent.mkdir(parents=True, exist_ok=True)
    labels = list(BAMS.keys())
    with open(OUT_TSV, "w", newline="") as fh:
        w = csv.writer(fh, delimiter="\t")
        header = ["name"] + [f"{l}_count" for l in labels] + [f"{l}_cpm" for l in labels]
        w.writerow(header)
        for name, *_ in mrnas:
            row = [name]
            for l in labels:
                row.append(all_counts[l][name])
            for l in labels:
                tot = totals[l]
                c = all_counts[l][name]
                cpm = (c / tot * 1e6) if tot > 0 else 0.0
                row.append(f"{cpm:.2f}")
            w.writerow(row)
    print(f"\n  wrote {OUT_TSV}")
    print(f"  {len(mrnas)} mRNA × {len(labels)} BAMs")


if __name__ == "__main__":
    main()
