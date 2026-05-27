"""make_geo_sample_sheet.py

Generate a GEO/SRA sample sheet for the in-house in-vitro nanopore RNA-seq
(YB1 / SL7207 / Δasd / PW) underlying the NCS paper. For each paper-canonical
sample (the 24 BAMs used in analysis) it locates the raw fastq under the sample's
run directory, parses strain / oxygen / DAP / replicate metadata, and writes a
TSV ready to paste into the NCBI GEO/SRA submission template.

NOTE: this covers ONLY the in-vitro strain RNA-seq for this paper. It deliberately
excludes the in-vivo HeLa/mouse Illumina data in `wetlab/RNA-seq raw data/`
(reserved for the follow-on paper).

Output: wetlab/nanopore/GEO_sample_sheet.tsv
"""
from __future__ import annotations
import os, glob, csv
from pathlib import Path

ROOT = "/Users/yubin/v2_data/wetlab/nanopore"
OUT = f"{ROOT}/GEO_sample_sheet.tsv"

# Paper-canonical samples → BAM path (from count_mrna_v3_with_0521_0522.py).
# The run directory is the BAM's parent; raw fastq live under it (often fastq_pass/).
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
    "PW_aer_0521":        f"{ROOT}/RNA数据(高通量芯片)/20260521-PW-平板克隆-DAP-MEM/pw_dap_aerobic_0521.sorted.bam",
    "PW_ana_0521":        f"{ROOT}/RNA数据(高通量芯片)/20260521-PW-平板克隆-DAP-anaerobic/pw_dap_anaerobic_0521.sorted.bam",
    "SL7207_ana_0522":     f"{ROOT}/RNA数据(高通量芯片)/20260522-7207-平板克隆-anaerobic/sl7207_anaerobic_0522.sorted.bam",
    "SL7207_dap_ana_0522": f"{ROOT}/RNA数据(高通量芯片)/20260522-7207-平板克隆-DAP-anaerobic/sl7207_dap_anaerobic_0522.sorted.bam",
    "SL7207_dap_aer_0525": f"{ROOT}/RNA数据(高通量芯片)/20260525-7207-平板克隆-DAP-MEP/sl7207_dap_aerobic_0525.sorted.bam",
    "SL7207_aer_0525":     f"{ROOT}/RNA数据(高通量芯片)/20260525-7207-平板克隆-MEP/sl7207_aerobic_0525.sorted.bam",
    "dAsd_aer_0526":       f"{ROOT}/RNA数据(高通量芯片)/20260526-S014-平板克隆-MEP/delta_asd_aerobic_0526.sorted.bam",
}
# on-disk dirs use full-width parentheses
BAMS = {k: v.replace("RNA数据(高通量芯片)", "RNA数据（高通量芯片）") for k, v in BAMS.items()}

STRAIN = {
    "YB1":    ("Salmonella enterica Typhimurium YB1", "asd under hypoxia-conditional control (engineered; Yu et al. 2012)"),
    "SL7207": ("Salmonella enterica Typhimurium SL7207", "wild-type parent (aroA)"),
    "dAsd":   ("Salmonella enterica Typhimurium SL7207 Δasd", "asd single-gene knockout"),
    "PW":     ("Salmonella enterica Typhimurium PW", "engineered, reverse-control circuit absent"),
}

def parse(label):
    if label.startswith("SL7207"): s = "SL7207"
    elif label.startswith("YB1"):  s = "YB1"
    elif label.startswith("dAsd"): s = "dAsd"
    elif label.startswith("PW"):   s = "PW"
    else: s = "?"
    o2 = "aerobic" if "_aer" in label else ("anaerobic" if "_ana" in label else "?")
    dap = "+DAP" if "dap" in label.lower() else ("+DAP" if (s in ("YB1","dAsd","PW") and o2=="aerobic") else "none")
    # replicate / batch token = trailing identifier
    rep = label.split("_")[-1]
    instrument = "MinION" if "MinION" in label else "PromethION"
    name, geno = STRAIN[s]
    return name, geno, o2, dap, rep, instrument

def find_fastq(run_dir):
    """raw fastq under a run directory (fastq_pass preferred; else anywhere under it)."""
    pats = []
    for sub in ("fastq_pass", "fastq_pass/**", "**"):
        pats += glob.glob(os.path.join(run_dir, sub, "*.fastq.gz"), recursive=True)
        pats += glob.glob(os.path.join(run_dir, sub, "*.fq.gz"), recursive=True)
    files = sorted(set(p for p in pats if "fastq_fail" not in p))
    return files

rows = []
for label, bam in BAMS.items():
    run_dir = os.path.dirname(bam)
    name, geno, o2, dap, rep, instr = parse(label)
    fqs = find_fastq(run_dir)
    total_mb = round(sum(os.path.getsize(f) for f in fqs) / 1e6, 1) if fqs else 0.0
    rows.append({
        "sample_title": label,
        "organism": name,
        "genotype": geno,
        "oxygen": o2,
        "DAP_supplementation": dap,
        "replicate_batch": rep,
        "instrument_model": instr,
        "library_strategy": "RNA-Seq",
        "library_source": "TRANSCRIPTOMIC",
        "library_selection": "cDNA" if False else "RANDOM",
        "platform": "OXFORD_NANOPORE",
        "run_dir": os.path.relpath(run_dir, ROOT),
        "n_fastq": len(fqs),
        "total_fastq_MB": total_mb,
        "fastq_files": ";".join(os.path.basename(f) for f in fqs) if fqs else "FASTQ_NOT_FOUND_LOCALLY",
    })

with open(OUT, "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()), delimiter="\t")
    w.writeheader(); w.writerows(rows)

# console summary
print(f"  wrote {OUT}")
print(f"  {len(rows)} samples")
miss = [r["sample_title"] for r in rows if r["n_fastq"] == 0]
tot = round(sum(r["total_fastq_MB"] for r in rows) / 1000, 1)
print(f"  total fastq located: {tot} GB across {sum(r['n_fastq'] for r in rows)} files")
if miss:
    print(f"  ⚠ no local fastq found for {len(miss)} samples (fill manually): {miss}")
print("\n  per-strain × oxygen × DAP:")
from collections import Counter
c = Counter((r["organism"].split()[-1], r["oxygen"], r["DAP_supplementation"]) for r in rows)
for k, v in sorted(c.items()):
    print(f"    {k}: {v}")
