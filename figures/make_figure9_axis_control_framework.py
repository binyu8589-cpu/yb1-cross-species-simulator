"""Figure 9: (axis, control_circuit) design framework — three strains on the invasion-primed plane."""
# --- repository-relative paths (override via env vars; see README) ---
import os as _os
_REPO = _os.environ.get("YB1_REPO", _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
_DATA = _os.environ.get("YB1_DATA", _os.path.join(_REPO, "data", "processed"))
_REF  = _os.environ.get("YB1_REF",  _os.path.join(_REPO, "data", "reference"))
_CKPT = _os.environ.get("YB1_CKPT", _os.path.join(_REPO, "checkpoints"))
# --- end repo-relative paths ---
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, Circle, Rectangle


def main():
    fig, axes = plt.subplots(1, 2, figsize=(15, 6.5))

    # ========== Left panel: 2D axis-control plane ==========
    ax = axes[0]

    # Background regions
    ax.axhspan(0, 1, xmin=0.0, xmax=0.5, alpha=0.05, color="#E76F51")  # axis-off
    ax.axhspan(0, 1, xmin=0.5, xmax=1.0, alpha=0.05, color="#2A9D8F")  # axis-on
    ax.axhline(0.5, color="#999", linewidth=0.8, alpha=0.5)
    ax.axvline(0.5, color="#999", linewidth=0.8, alpha=0.5)

    # Three strain points
    strains = [
        # (x_axis, y_control, name, color, descrip)
        (0.18, 0.18, "Δasd", "#666666", "auxotrophy only\n(no engineering)\n→ invasion suppressed"),
        (0.85, 0.15, "PW",   "#E76F51", "engineering active\nbut no reverse control\n→ super-activation (~1.5× YB1)"),
        (0.85, 0.78, "YB1",  "#2A9D8F", "engineering + reverse control\n→ moderated working point"),
    ]
    for x, y, name, color, descrip in strains:
        ax.scatter([x], [y], s=550, c=color, edgecolor="black",
                    linewidth=2, zorder=5)
        ax.text(x, y + 0.08, name, ha="center", fontsize=15,
                fontweight="bold", color=color, zorder=6)
        ax.text(x, y - 0.13, descrip, ha="center", fontsize=8.5,
                color="#333", zorder=6)

    # YB1 invasion-primed log2FC zone
    invasion_band_top = 0.62
    invasion_band_bot = 0.42
    ax.axhspan(invasion_band_bot, invasion_band_top, xmin=0.55, xmax=0.95,
                alpha=0.18, color="#2A9D8F", zorder=1)
    ax.annotate("therapeutic\nworking-point band",
                xy=(0.85, 0.55), xytext=(0.55, 0.93),
                ha="center", fontsize=9.5, color="#1c6e62",
                arrowprops=dict(arrowstyle="->", color="#1c6e62", lw=1.2))

    # Quadrant labels
    ax.text(0.25, 0.95, "axis OFF\n(chassis fail)", ha="center", fontsize=10.5,
            color="#a64030", style="italic")
    ax.text(0.75, 0.95, "axis ON\n(candidate)", ha="center", fontsize=10.5,
            color="#1c6e62", style="italic")

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Regulatory axis activation\n(invasion-primed program: SPI-1 + Flagellar + Chemotaxis)",
                  fontsize=11)
    ax.set_ylabel("Reverse-control circuit\n(dampens working point)", fontsize=11)
    ax.set_title("(A) Three-strain decomposition: (axis, control_circuit) plane",
                 fontsize=12, loc="left", pad=10)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(True, alpha=0.2)

    # ========== Right panel: SPI-1/Flagellar/SPI-2 panel log2FC across 3 strains ==========
    ax2 = axes[1]
    GENES_BY_PANEL = [
        ("SPI-1 invasion", ["invG", "sopB", "sipB", "hilA", "hilC"]),
        ("Flagellar",      ["fliI", "fliE", "flgJ", "motA"]),
        ("Chemotaxis",     ["cheY"]),
        ("SPI-2",          ["siiD", "siiA"]),
    ]
    # DAP-matched anaerobic log2FC (consistent with Figure 5d / make_figure5d_dap_matched.py):
    #   Δasd (+DAP) vs WT+DAP, PW (+DAP) vs WT+DAP, YB1 (no-DAP) vs WT no-DAP (DAP-symmetric).
    # Replaces the earlier unmatched-WT values (PW was inflated ~2-3x by exogenous DAP
    # activating SPI-1 anaerobically). PW super-activation is ~1.5x YB1 once DAP-matched.
    PW = {"invG": 2.52, "sopB": 2.87, "sipB": 2.87, "hilA": 2.28, "hilC": 0.98,
          "fliI": 1.97, "fliE": 1.64, "flgJ": 1.81, "motA": 1.98,
          "cheY": 2.40, "siiD": 2.31, "siiA": 2.23}
    YB1 = {"invG": 2.44, "sopB": 2.03, "sipB": 1.24, "hilA": 1.54, "hilC": 0.81,
           "fliI": 1.18, "fliE": -2.36, "flgJ": 1.83, "motA": 0.49,
           "cheY": 1.88, "siiD": 3.13, "siiA": 2.76}
    DASD = {"invG": -5.88, "sopB": -3.45, "sipB": -3.32, "hilA": -1.78, "hilC": -5.57,
            "fliI": -5.21, "fliE": -5.38, "flgJ": -4.59, "motA": -3.73,
            "cheY": -2.02, "siiD": -3.93, "siiA": -3.93}

    all_genes = []
    panel_borders = []
    for panel_name, genes in GENES_BY_PANEL:
        all_genes.extend(genes)
        panel_borders.append(len(all_genes))

    x = np.arange(len(all_genes))
    w = 0.27
    pw_vals = [PW[g] for g in all_genes]
    yb1_vals = [YB1[g] for g in all_genes]
    dasd_vals = [DASD[g] for g in all_genes]

    ax2.bar(x - w, dasd_vals, w, color="#666666", edgecolor="black",
            linewidth=0.5, label="Δasd")
    ax2.bar(x,     yb1_vals,  w, color="#2A9D8F", edgecolor="black",
            linewidth=0.5, label="YB1")
    ax2.bar(x + w, pw_vals,   w, color="#E76F51", edgecolor="black",
            linewidth=0.5, label="PW")

    ax2.axhline(0, color="black", linewidth=0.5)
    # Panel separators
    for b in panel_borders[:-1]:
        ax2.axvline(b - 0.5, color="#999", linewidth=0.5, alpha=0.5, ls="--")
    ax2.set_xticks(x)
    ax2.set_xticklabels(all_genes, rotation=45, ha="right", fontsize=10)
    ax2.set_ylabel("log2FC vs SL7207 WT (anaerobic + DAP)", fontsize=11)
    ax2.set_title("(B) Three-strain invasion-primed panel: PW > YB1 > Δasd",
                  fontsize=12, loc="left", pad=10)
    ax2.legend(loc="lower right", fontsize=10, framealpha=0.95)
    ax2.spines[["top", "right"]].set_visible(False)
    ax2.grid(True, alpha=0.3, axis="y")
    # Panel labels at top
    boundaries = [0] + panel_borders
    for i, (pname, _) in enumerate(GENES_BY_PANEL):
        x_mid = (boundaries[i] + boundaries[i+1] - 1) / 2.0
        ax2.text(x_mid, 6.3, pname, ha="center", fontsize=9.5,
                 fontweight="bold", color="#333")
    ax2.set_ylim(-5.5, 7.0)

    fig.suptitle(
        "Engineered oncolytic Salmonella design = (regulatory axis, control circuit) tuple\n"
        "Reverse control is a dampener, not a trigger — engineering modifications install the axis",
        fontsize=12.5, y=1.02)
    fig.tight_layout()

    import os
    OUT = _os.path.join(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))), "figures", "figure9_axis_control_framework")
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    fig.savefig(OUT + ".png", dpi=300, bbox_inches="tight")
    fig.savefig(OUT + ".pdf", bbox_inches="tight")
    print(f"saved {OUT}.png + .pdf")
    plt.close(fig)


if __name__ == "__main__":
    main()
