"""make_figure5d_dap_matched.py

DAP-matched replacement for Figure 5 panel d (three-strain invasion-panel
decomposition, anaerobic). Each strain is compared to its OWN DAP-matched WT:
  YB1  (no DAP)  vs SL7207 no-DAP   (original B3; already DAP-symmetric)
  PW   (+DAP)    vs SL7207 + DAP    (F2_dap)
  Δasd (+DAP)    vs SL7207 + DAP    (E2_dap)

This corrects the earlier panel (all strains vs no-DAP WT), under which PW's
apparent super-activation was inflated by the DAP it received but the WT control
did not (exogenous DAP activates SPI-1 anaerobically). Ordering Δasd < YB1 < PW
is preserved; PW super-activation is ~1.5× YB1 (not 2-3×) once DAP-matched.

Values computed from this study's primary-norm per_gene_log2fc.csv files.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT = "/Users/yubin/v2_data/paper_figures/figure5d_dap_matched.png"
C_DASD, C_YB1, C_PW = "#8a8a8a", "#1b9e8a", "#e8743b"

PANEL = ["invG","sopB","sipB","sipD","hilA","hilC","fliI","flgJ","fliE","fliG","cheY","motA","motB","siiD","siiA"]
dasd = [-5.88,-3.45,-3.32,-2.71,-1.78,-5.57,-5.21,-4.59,-5.38,-5.03,-2.02,-3.73,-3.87,-3.93,-3.93]
pw   = [ 2.52, 2.87, 2.87, 1.73, 2.28, 0.98, 1.97, 1.81, 1.64, 0.72, 2.40, 1.98, 2.00, 2.31, 2.23]
yb1  = [ 2.44, 2.03, 1.24, 2.90, 1.54, 0.81, 1.18, 1.83,-2.36, 0.12, 1.88, 0.49, 0.17, 3.13, 2.76]
m_dasd, m_pw, m_yb1 = np.mean(dasd), np.mean(pw), np.mean(yb1)

fig, (axL, axR) = plt.subplots(1, 2, figsize=(13, 4.6), gridspec_kw={"width_ratios":[1, 2.3]})

# ---- left: panel-mean bars ----
strains = ["Δasd", "YB1", "PW"]
means   = [m_dasd, m_yb1, m_pw]
cols    = [C_DASD, C_YB1, C_PW]
xb = np.arange(3)
axL.bar(xb, means, 0.62, color=cols, edgecolor="k", linewidth=0.5)
axL.axhline(0, color="k", linewidth=0.6)
axL.set_xticks(xb); axL.set_xticklabels(strains, fontsize=11)
axL.set_ylabel("invasion panel mean log2FC", fontsize=10)
axL.set_title("(d) Three-strain decomposition, anaerobic\n(each strain vs its DAP-matched WT)",
              fontsize=10, loc="left")
for xi, v, lab in zip(xb, means, ["axis OFF","axis ON\n+ control","axis ON\nno control"]):
    axL.text(xi, v + (0.18 if v >= 0 else -0.45), f"{v:+.2f}", ha="center", fontsize=9, fontweight="bold")
    axL.text(xi, (0.55 if v < 0 else -0.7), lab, ha="center", fontsize=7.5, color="#444")
axL.annotate("", xy=(2, 2.02), xytext=(1, 1.34),
             arrowprops=dict(arrowstyle="->", color="#777", lw=1))
axL.text(1.5, 2.5, "~1.5×\n(was 2-3× vs\nunmatched WT)", ha="center", fontsize=7, color="#777")
axL.set_ylim(-4.8, 3.4)

# ---- right: per-gene grouped bars ----
x = np.arange(len(PANEL)); w = 0.27
axR.bar(x - w, dasd, w, label="Δasd vs WT+DAP", color=C_DASD, edgecolor="k", linewidth=0.3)
axR.bar(x,     yb1,  w, label="YB1 vs WT(no-DAP)", color=C_YB1, edgecolor="k", linewidth=0.3)
axR.bar(x + w, pw,   w, label="PW vs WT+DAP", color=C_PW, edgecolor="k", linewidth=0.3)
axR.axhline(0, color="k", linewidth=0.6)
axR.set_xticks(x); axR.set_xticklabels([f"$\\it{{{g}}}$" for g in PANEL], rotation=60, ha="right", fontsize=8)
axR.set_ylabel("log2FC vs DAP-matched WT (anaerobic)", fontsize=9.5)
axR.set_title("Three-strain invasion-primed panel (DAP-matched): PW > YB1 > Δasd",
              fontsize=10, loc="left")
axR.legend(fontsize=8, frameon=False, ncol=3, loc="lower center")
# section dividers / labels
for xpos, lab in [(2.5,"SPI-1 invasion"), (8.5,"Flagellar"), (10,"Chemo"), (13.5,"SPI-2")]:
    pass
axR.set_ylim(-6.5, 4.0)

fig.suptitle("Figure 5d (revised, DAP-matched) | Three-strain anaerobic decomposition — "
             "axis ordering preserved, PW super-activation ~1.5× once DAP-matched",
             fontsize=11, fontweight="bold", x=0.012, ha="left", y=1.02)
fig.tight_layout(rect=[0, 0, 1, 0.96])
fig.savefig(OUT, dpi=300, bbox_inches="tight")
print("wrote", OUT)
