"""Fig 1 — Circuit logic and experimental system (4 panels).
a genealogy: YB1 & PW share the sodA-antisense; differ in pepT forward-drive dose (1x vs 2x).
b shared pepT/sodA-antisense logic gate.  c O2/DAP design (PW anaerobic = -DAP, matched to YB1).
d DAP-matched control rationale."""
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
OUT = "/Users/dryu/claude_paper1_rebuild"
BLU, RED, GRY, GRN = "#1f6feb", "#b03a2e", "#8a8a8a", "#1a7a4a"
fig = plt.figure(figsize=(11, 8.4))
gs = fig.add_gridspec(2, 2, hspace=0.24, wspace=0.16, left=0.035, right=0.975, top=0.91, bottom=0.04)
fig.suptitle("Figure 1 | Circuit logic and experimental system", fontsize=13.5, fontweight="bold", x=0.035, ha="left")
def box(ax, x, y, w, h, t, fc="#fff", ec="#333", fs=9, fw="normal", tc="#000"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.01,rounding_size=0.02", fc=fc, ec=ec, lw=1.3))
    ax.text(x + w/2, y + h/2, t, ha="center", va="center", fontsize=fs, fontweight=fw, color=tc)
def arr(ax, x1, y1, x2, y2, color="#444", lw=1.5):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=12, color=color, lw=lw))

# a: genealogy (branch: Δasd -> YB1 1x / PW 2x, shared antisense)
ax = fig.add_subplot(gs[0, 0]); ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
ax.set_title("a  Strain genealogy", loc="left", fontsize=10.5, fontweight="bold")
box(ax, 0.30, 0.84, 0.40, 0.12, "SL7207  (WT parent)", ec=GRY, fs=8.6, fw="bold", tc=GRY)
box(ax, 0.30, 0.63, 0.40, 0.12, "Δ$asd$  (native $asd$ deletion)", ec=GRY, fs=8.4, fw="bold", tc=GRY)
arr(ax, 0.50, 0.84, 0.50, 0.75, color="#555"); ax.text(0.72, 0.80, "delete $asd$", fontsize=7, color="#333")
box(ax, 0.05, 0.34, 0.42, 0.16, "YB1\n1× $pepT$–$asd$\n+ $sodA$–antisense", ec=BLU, fs=8.0, fw="bold", tc=BLU)
box(ax, 0.54, 0.34, 0.42, 0.16, "PW\n2× $pepT$–$asd$\n+ $sodA$–antisense", ec=RED, fs=8.0, fw="bold", tc=RED)
arr(ax, 0.42, 0.63, 0.26, 0.50, color="#555"); arr(ax, 0.58, 0.63, 0.75, 0.50, color="#555")
ax.text(0.5, 0.18, "YB1 and PW share the $sodA$–antisense reverse-control;\nthey differ in $pepT$ forward-drive dose (1× vs 2×).",
        ha="center", fontsize=7.4, color="#333", fontstyle="italic")

# b: logic gate (shared by YB1 and PW)
ax = fig.add_subplot(gs[0, 1]); ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
ax.set_title("b  Shared logic gate ($pepT$ sense / $sodA$ antisense)", loc="left", fontsize=9.8, fontweight="bold")
box(ax, 0.04, 0.78, 0.27, 0.12, "$pepT$ promoter", ec=GRN, fs=8, tc=GRN)
box(ax, 0.04, 0.50, 0.27, 0.12, "$sodA$ promoter", ec=RED, fs=8, tc=RED)
box(ax, 0.56, 0.64, 0.20, 0.12, "$asd$", ec="#333", fs=10, fw="bold")
box(ax, 0.40, 0.50, 0.30, 0.10, "antisense $asd$", ec=RED, fs=7.5, tc=RED)
arr(ax, 0.31, 0.84, 0.56, 0.72, color=GRN); ax.text(0.40, 0.86, "hypoxia: ON", fontsize=6.8, color=GRN)
arr(ax, 0.31, 0.56, 0.40, 0.55, color=RED); ax.text(0.06, 0.45, "air: ON", fontsize=6.8, color=RED)
ax.annotate("", xy=(0.62, 0.64), xytext=(0.62, 0.60), arrowprops=dict(arrowstyle="-[", color=RED, lw=1.6))
ax.text(0.80, 0.70, "→ Asd\n→ growth", fontsize=7.6, va="center")
cells = [["condition", "$pepT$", "$sodA$→AS", "$asd$", "growth"],
         ["hypoxia", "ON", "off", "expressed", "permitted"], ["air", "off", "ON (repress)", "suppressed", "DAP-dep."]]
cw = [0.18, 0.13, 0.22, 0.19, 0.20]; rh = 0.105; tx, ty = 0.04, 0.05
for r, row in enumerate(cells):
    cx = tx
    for c, val in zip(cw, row):
        ax.add_patch(plt.Rectangle((cx, ty + (2 - r) * rh), c, rh, fc="#eef" if r == 0 else "#fff", ec="#bbb", lw=0.7))
        ax.text(cx + c/2, ty + (2 - r) * rh + rh/2, val, ha="center", va="center", fontsize=6.5, fontweight="bold" if r == 0 else "normal")
        cx += c
ax.text(0.04, 0.985, "shared by YB1 & PW (aerobic failsafe; not isolated here)", fontsize=6.4, color="#888", va="top")

# c: design matrix (PW anaerobic = -DAP, corrected)
ax = fig.add_subplot(gs[1, 0]); ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
ax.set_title("c  Oxygen / DAP experimental design  (n libraries; flagship = anaerobic)", loc="left", fontsize=9.5, fontweight="bold")
strains = [("SL7207 (WT)", "−DAP", GRY), ("Δ$asd$", "+DAP (auxotroph)", GRY), ("YB1", "aer +DAP / ana −DAP", BLU),
           ("PW", "aer +DAP / ana −DAP*", RED), ("SL7207+DAP ctrl", "+DAP", GRN)]
naer = [3, 3, 3, 3, 2]; nana = [2, 2, 4, 3, 2]
x0, y0, cw1, rh1 = 0.02, 0.84, [0.40, 0.18, 0.18, 0.22], 0.135
for j, (h, w) in enumerate(zip(["strain", "aerobic", "anaerobic", "DAP"], cw1)):
    cx = x0 + sum(cw1[:j]); ax.add_patch(plt.Rectangle((cx, y0), w, rh1, fc="#eef", ec="#bbb", lw=0.7))
    ax.text(cx + w/2, y0 + rh1/2, h, ha="center", va="center", fontsize=7.5, fontweight="bold")
for i, (s, dap, c) in enumerate(strains):
    yy = y0 - (i + 1) * rh1
    for j, (val, w) in enumerate(zip([s, f"n={naer[i]}", f"n={nana[i]}", dap], cw1)):
        cx = x0 + sum(cw1[:j]); ax.add_patch(plt.Rectangle((cx, yy), w, rh1, fc="#fff", ec="#ddd", lw=0.6))
        ax.text(cx + w/2, yy + rh1/2, val, ha="center", va="center", fontsize=6.8, color=c if j == 0 else "#333", fontweight="bold" if j == 0 else "normal")
ax.text(x0, 0.02, "*PW anaerobic run as the matched comparator to YB1 (−DAP). Flagship comparison = anaerobic YB1 vs PW.",
        fontsize=6.4, color="#555", fontstyle="italic")

# d: matched-control rationale
ax = fig.add_subplot(gs[1, 1]); ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
ax.set_title("d  Why DAP-matched controls are required", loc="left", fontsize=10.2, fontweight="bold")
box(ax, 0.06, 0.66, 0.34, 0.16, "strain\n(its DAP status)", ec=BLU, fs=8.2, tc=BLU)
box(ax, 0.60, 0.66, 0.34, 0.16, "SL7207 at\nmatched DAP", ec=GRN, fs=8.2, tc=GRN)
arr(ax, 0.40, 0.74, 0.60, 0.74, color="#333"); ax.text(0.50, 0.77, "matched", ha="center", fontsize=7.2, color=GRN)
box(ax, 0.06, 0.38, 0.34, 0.16, "strain\n(its DAP status)", ec=BLU, fs=8.2, tc=BLU)
box(ax, 0.60, 0.38, 0.34, 0.16, "SL7207 at\nWRONG DAP", ec=GRY, fs=8.2, tc=GRY)
ax.add_patch(FancyArrowPatch((0.40, 0.46), (0.60, 0.46), arrowstyle="-|>", mutation_scale=12, color=RED, lw=1.5))
ax.plot([0.46, 0.54], [0.42, 0.50], color=RED, lw=2.2); ax.plot([0.46, 0.54], [0.50, 0.42], color=RED, lw=2.2)
ax.text(0.50, 0.32, "confounded by DAP", ha="center", fontsize=7.0, color=RED)
ax.text(0.05, 0.15, "DAP alone shifts SPI-1 / SPI-2 / flagellar programs by up to ~1.8 log2FC\n(Fig. 4a). Each strain is compared to SL7207 at its own DAP status; the\nclean YB1 (1×) vs PW (2×) comparison is both −DAP.",
        fontsize=7.0, color="#333", va="top")
fig.savefig(f"{OUT}/fig1_system.png", dpi=200, bbox_inches="tight"); fig.savefig(f"{OUT}/fig1_system.pdf", bbox_inches="tight")
print("saved fig1")
