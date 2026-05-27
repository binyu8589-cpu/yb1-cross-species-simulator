"""count_3new_merge_v4.py — count the 3 new 0525/0526 BAMs and merge into v3 CPM table → v4.

Only the 3 new samples are counted (fast); existing columns from
mrna_counts_v3_with_0521_0522.tsv are carried over unchanged.
New samples:
  SL7207_dap_aer_0525  (aerobic SL7207 + DAP — 2nd aerobic DAP-matched control; closes C5)
  SL7207_aer_0525      (aerobic SL7207, no DAP)
  dAsd_aer_0526        (Δasd aerobic)
"""
from __future__ import annotations
import csv
from pathlib import Path
import pysam

REF_GFF = "/Users/yubin/v2_data/wetlab/references/sl1344.gff"
FF = "/Users/yubin/v2_data/wetlab/nanopore/v4_full_factorial"
IN_TSV = f"{FF}/mrna_counts_v3_with_0521_0522.tsv"
OUT_TSV = f"{FF}/mrna_counts_v4_with_0525_0526.tsv"
R = "/Users/yubin/v2_data/wetlab/nanopore/RNA数据（高通量芯片）"
NEW = {
    "SL7207_dap_aer_0525": f"{R}/20260525-7207-平板克隆-DAP-MEP/sl7207_dap_aerobic_0525.sorted.bam",
    "SL7207_aer_0525":     f"{R}/20260525-7207-平板克隆-MEP/sl7207_aerobic_0525.sorted.bam",
    "dAsd_aer_0526":       f"{R}/20260526-S014-平板克隆-MEP/delta_asd_aerobic_0526.sorted.bam",
}


def parse_mrna_gff(gff):
    out, seen = [], set()
    for ln in open(gff):
        if ln.startswith("#") or not ln.strip():
            continue
        c = ln.rstrip("\n").split("\t")
        if len(c) < 9 or c[2] != "gene":
            continue
        a = dict(kv.split("=", 1) for kv in c[8].split(";") if "=" in kv)
        if a.get("gene_biotype") != "protein_coding":
            continue
        lt = a.get("locus_tag"); nm = a.get("Name") or a.get("gene") or lt
        if not lt or lt in seen:
            continue
        seen.add(lt)
        try:
            out.append((nm, c[0], int(c[3]), int(c[4])))
        except ValueError:
            pass
    return out


def count_bam(bam, mrnas):
    counts = {n: 0 for n, *_ in mrnas}; total = 0
    with pysam.AlignmentFile(bam, "rb") as f:
        for r in f.fetch(until_eof=True):
            if r.is_secondary or r.is_supplementary or r.is_unmapped:
                continue
            total += 1
        for name, chrom, s, e in mrnas:
            n = 0
            try:
                for r in f.fetch(chrom, s - 1, e):
                    if r.is_secondary or r.is_supplementary or r.is_unmapped:
                        continue
                    n += 1
            except (KeyError, ValueError):
                pass
            counts[name] = n
    return counts, total


def main():
    mrnas = parse_mrna_gff(REF_GFF)
    newc, newt = {}, {}
    for lab, bam in NEW.items():
        print(f"  counting {lab} ...", flush=True)
        if not Path(bam).exists():
            print(f"    MISSING {bam}"); continue
        c, t = count_bam(bam, mrnas)
        newc[lab], newt[lab] = c, t
        print(f"    {lab}: {t:,} primary reads")

    # read v3 table
    with open(IN_TSV) as fh:
        rd = list(csv.reader(fh, delimiter="\t"))
    header, body = rd[0], rd[1:]
    labels = list(newc.keys())
    new_header = header + [f"{l}_count" for l in labels] + [f"{l}_cpm" for l in labels]
    with open(OUT_TSV, "w", newline="") as fh:
        w = csv.writer(fh, delimiter="\t"); w.writerow(new_header)
        for row in body:
            name = row[0]
            row = row + [newc[l][name] for l in labels]
            for l in labels:
                t = newt[l]; c = newc[l][name]
                row.append(f"{(c/t*1e6) if t else 0:.2f}")
            w.writerow(row)
    print(f"\n  wrote {OUT_TSV}  (+{len(labels)} samples)")


if __name__ == "__main__":
    main()
