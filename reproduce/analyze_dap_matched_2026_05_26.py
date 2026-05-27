"""analyze_dap_matched_2026_05_22.py

(b) DAP-matched control analysis -- n=2 update with the 2026-05-22 SL7207 run.

The 2026-05-22 SL7207 anaerobic run adds a SECOND DAP-matched WT replicate
(SL7207_dap_ana_0522, pairs with 0520) and a second DAP-free anaerobic WT
(SL7207_ana_0522). This lets us re-ask the gating question at n=2:

  Do the YB1/PW engineering panel signatures (SPI-1 up, flagellar up,
  SPI-2 down) survive once the DAP nutritional confounder is removed --
  and does the n=1 (0520) conclusion replicate?

Reads : v4_full_factorial/mrna_counts_v4_with_0525_0526.tsv
Writes: v4_full_factorial/dap_matched_n2/  (panel_table.tsv, run.log via stdout)
"""
from __future__ import annotations
import csv
import math
from collections import OrderedDict
from pathlib import Path

FF = "/Users/yubin/v2_data/wetlab/nanopore/v4_full_factorial"
MRNA_TSV = f"{FF}/mrna_counts_v4_with_0525_0526.tsv"
OUT_DIR = f"{FF}/dap_matched_n2_aer"

# ---- pooled groups (averaged CPM across listed samples) --------------------
# n=2 upgrades (vs 0518/0520 analysis): WT_ana_DAP and WT_ana_noDAP now n=2;
# PW_aer / PW_ana now n=3. WT_aer_DAP stays n=1 (no new aerobic SL7207+DAP).
GROUPS = OrderedDict([
    ("YB1_aer",        ["YB1_aer_MinION", "YB1_aer_0508", "YB1_aer_clone_0516"]),
    ("YB1_ana",        ["YB1_ana_v2", "YB1_ana_0507", "YB1_ana_0508", "YB1_ana_clone_0516"]),
    ("PW_aer",         ["PW_aer_0515", "PW_aer_0518", "PW_aer_0521"]),
    ("PW_ana",         ["PW_ana_0515", "PW_ana_0518", "PW_ana_0521"]),
    ("WT_aer_noDAP",   ["SL7207_aer_0507", "SL7207_aer_0509", "SL7207_aer_0525"]),
    ("WT_ana_noDAP",   ["SL7207_ana_0509", "SL7207_ana_0522"]),
    ("WT_aer_DAP",     ["SL7207_dap_aer_0520", "SL7207_dap_aer_0525"]),
    ("WT_ana_DAP",     ["SL7207_dap_ana_0520", "SL7207_dap_ana_0522"]),
])

# test -> (noDAP control, DAP control)
CONTRASTS = OrderedDict([
    ("YB1_aer", ("WT_aer_noDAP", "WT_aer_DAP")),
    ("YB1_ana", ("WT_ana_noDAP", "WT_ana_DAP")),
    ("PW_aer",  ("WT_aer_noDAP", "WT_aer_DAP")),
    ("PW_ana",  ("WT_ana_noDAP", "WT_ana_DAP")),
])

PANELS = OrderedDict([
    ("DAP cascade", ["asd", "dapA", "dapB", "dapD", "dapE", "dapF"]),
    ("Peptidoglycan Mur cassette",
        ["murA", "murB", "murC", "murD", "murE", "murF", "mraY", "murG"]),
    ("SPI-1 (invasion T3SS)",
        ["invA", "invB", "invC", "invE", "invF", "invG", "invH", "invI", "invJ",
         "sipA", "sipB", "sipC", "sipD",
         "sopB", "sopD", "sopE", "sopE2", "sopA",
         "prgH", "prgI", "prgJ", "prgK",
         "hilA", "hilC", "hilD",
         "iagB", "sicA", "sicP", "sptP", "spaO", "spaP", "spaQ", "spaR", "spaS"]),
    ("SPI-2 (intracellular T3SS)",
        ["ssaB", "ssaC", "ssaD", "ssaE", "ssaG", "ssaH", "ssaJ", "ssaK", "ssaL",
         "ssaM", "ssaN", "ssaO", "ssaP", "ssaQ", "ssaR", "ssaS", "ssaT", "ssaU", "ssaV",
         "sseA", "sseB", "sseC", "sseD", "sseE", "sseF", "sseG", "sseI", "sseJ",
         "ssrA", "ssrB", "sopD2", "pipB", "pipB2", "sifA", "sifB"]),
    ("Flagellar / chemotaxis",
        ["flhD", "flhC", "fliA", "fliC", "fljB", "fliD", "fliS", "fliT",
         "flgB", "flgC", "flgD", "flgE", "flgF", "flgG", "flgH", "flgI", "flgK", "flgL",
         "flgM", "flgN", "fliF", "fliG", "fliM", "fliN", "flhA", "flhB",
         "motA", "motB", "cheA", "cheW", "cheY", "cheZ", "cheR", "cheB", "tar", "tsr"]),
])

PSEUDO = 1.0  # CPM pseudocount


def load_cpm(tsv):
    """Returns (cpm[sample][gene], gene_list, sample_set). Uses *_cpm columns."""
    with open(tsv) as fh:
        rd = csv.reader(fh, delimiter="\t")
        header = next(rd)
        cpm_cols = {h[:-4]: i for i, h in enumerate(header) if h.endswith("_cpm")}
        cpm = {s: {} for s in cpm_cols}
        genes = []
        for row in rd:
            g = row[0]
            genes.append(g)
            for s, i in cpm_cols.items():
                cpm[s][g] = float(row[i])
    return cpm, genes, set(cpm_cols)


def pooled_cpm(cpm, members):
    """Mean CPM across member samples -> {gene: cpm}."""
    genes = next(iter(cpm.values())).keys()
    out = {}
    for g in genes:
        out[g] = sum(cpm[m][g] for m in members) / len(members)
    return out


def binom_two_sided(k, n):
    """Exact two-sided binomial sign test p-value, p=0.5."""
    if n == 0:
        return float("nan")
    from math import comb
    def tail(x):
        return sum(comb(n, i) for i in range(x, n + 1)) / 2**n
    k_lo = min(k, n - k)
    k_hi = max(k, n - k)
    p = tail(k_hi) + (sum(math.comb(n, i) for i in range(0, k_lo + 1)) / 2**n)
    return min(1.0, p)


def log2fc(test_cpm, ctrl_cpm, gene):
    return math.log2((test_cpm[gene] + PSEUDO) / (ctrl_cpm[gene] + PSEUDO))


def panel_stats(test_cpm, ctrl_cpm, present_genes, panel_genes):
    fcs = [log2fc(test_cpm, ctrl_cpm, g) for g in panel_genes if g in present_genes]
    n = len(fcs)
    if n == 0:
        return None
    up = sum(1 for x in fcs if x > 0)
    down = sum(1 for x in fcs if x < 0)
    mean = sum(fcs) / n
    p = binom_two_sided(up, up + down) if (up + down) else float("nan")
    return n, up, down, mean, p


def cosine(test_cpm, ctrl_a, ctrl_b, genes):
    """Cosine between two log2FC vectors (same test, two controls) over all genes."""
    va = [log2fc(test_cpm, ctrl_a, g) for g in genes]
    vb = [log2fc(test_cpm, ctrl_b, g) for g in genes]
    dot = sum(a * b for a, b in zip(va, vb))
    na = math.sqrt(sum(a * a for a in va)); nb = math.sqrt(sum(b * b for b in vb))
    return dot / (na * nb) if na and nb else float("nan")


def main():
    cpm, genes, samples = load_cpm(MRNA_TSV)
    present = set(genes)
    print(f"  loaded {len(genes)} genes x {len(samples)} samples from {Path(MRNA_TSV).name}")
    missing = [m for grp in GROUPS.values() for m in grp if m not in samples]
    if missing:
        print(f"  WARN: samples missing from table: {sorted(set(missing))}")
    print("  replicate counts per group:")
    for g, members in GROUPS.items():
        ok = [m for m in members if m in samples]
        print(f"    {g:16s} n={len(ok)}  {ok}")

    pools = {g: pooled_cpm(cpm, [m for m in members if m in samples])
             for g, members in GROUPS.items()}

    Path(OUT_DIR).mkdir(parents=True, exist_ok=True)
    out_rows = [["contrast", "control", "panel", "n", "up", "down", "mean_log2FC", "sign_p"]]

    # ---- pure DAP effect (control-control) ---------------------------------
    print("\n================ PURE DAP EFFECT (SL7207+DAP vs SL7207 no-DAP) ===========")
    for cond, (cdap, cnod) in [("aerobic", ("WT_aer_DAP", "WT_aer_noDAP")),
                               ("anaerobic", ("WT_ana_DAP", "WT_ana_noDAP"))]:
        print(f"\n  [{cond}]  {cdap} vs {cnod}")
        for pname, pgenes in PANELS.items():
            st = panel_stats(pools[cdap], pools[cnod], present, pgenes)
            if st:
                n, up, down, mean, p = st
                print(f"    {pname:32s} n={n:3d} up={up:2d} down={down:2d} "
                      f"mean={mean:+.2f} p={p:.4f}")
                out_rows.append([f"DAPeffect_{cond}", cnod, pname, n, up, down,
                                 f"{mean:.4f}", f"{p:.4g}"])

    # ---- engineered strain vs no-DAP and +DAP controls ---------------------
    print("\n================ ENGINEERING PANELS: no-DAP vs DAP-matched control ========")
    for test, (c_nod, c_dap) in CONTRASTS.items():
        cos = cosine(pools[test], pools[c_nod], pools[c_dap], genes)
        print(f"\n  {test}   (cosine of log2FC vectors noDAP-ctrl vs DAP-ctrl = {cos:+.3f})")
        print(f"    {'panel':32s} {'  no-DAP ctrl':>22s}   {'  +DAP ctrl':>22s}")
        for pname, pgenes in PANELS.items():
            s_nod = panel_stats(pools[test], pools[c_nod], present, pgenes)
            s_dap = panel_stats(pools[test], pools[c_dap], present, pgenes)
            if not s_nod:
                continue
            n1, u1, d1, m1, p1 = s_nod
            n2, u2, d2, m2, p2 = s_dap
            print(f"    {pname:32s}  mean={m1:+.2f} p={p1:.3f} (u{u1}/d{d1})"
                  f"   mean={m2:+.2f} p={p2:.3f} (u{u2}/d{d2})")
            out_rows.append([test, c_nod, pname, n1, u1, d1, f"{m1:.4f}", f"{p1:.4g}"])
            out_rows.append([test, c_dap, pname, n2, u2, d2, f"{m2:.4f}", f"{p2:.4g}"])

    out_tsv = f"{OUT_DIR}/panel_table.tsv"
    with open(out_tsv, "w", newline="") as fh:
        csv.writer(fh, delimiter="\t").writerows(out_rows)
    print(f"\n  wrote {out_tsv}")


if __name__ == "__main__":
    main()
