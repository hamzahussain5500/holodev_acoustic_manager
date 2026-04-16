"""
Figure 2: Posterior Covariance Beacon Selection Policy
Publication-quality flowchart for the adaptive beacon subset selection.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch

# ── Colour palette (muted, publication-friendly) ──────────────────────────
BLUE    = "#4878A8"
GREEN   = "#5B9A6D"
ORANGE  = "#C07840"
RED     = "#B05050"
PURPLE  = "#7A6B90"
EDGE    = "#555555"
TXT     = "#333333"
WHITE   = "#FFFFFF"
LIGHT_BLUE  = "#D6DFE8"
LIGHT_GREEN = "#D5E5DA"
LIGHT_ORANGE = "#E8DDD0"
LIGHT_RED   = "#E0D0D0"

# ── Canvas ──────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 16))
ax.set_xlim(0, 12)
ax.set_ylim(0, 18.5)
ax.axis("off")
fig.patch.set_facecolor(WHITE)

# ── Helpers ─────────────────────────────────────────────────────────────────
def draw_box(ax, cx, cy, w, h, text, fc, ec=EDGE, fontsize=9,
             fontweight="normal", text_color=WHITE, style="round,pad=0.15",
             zorder=3):
    box = FancyBboxPatch(
        (cx - w / 2, cy - h / 2), w, h,
        boxstyle=style, facecolor=fc, edgecolor=ec,
        linewidth=1.3, zorder=zorder,
    )
    ax.add_patch(box)
    ax.text(cx, cy, text, ha="center", va="center",
            fontsize=fontsize, color=text_color, fontweight=fontweight,
            zorder=zorder + 1, linespacing=1.35)
    return box


def draw_diamond(ax, cx, cy, w, h, text, fc=ORANGE, ec=EDGE, fontsize=8.5,
                 text_color=WHITE):
    hw, hh = w / 2, h / 2
    verts = [(cx, cy + hh), (cx + hw, cy), (cx, cy - hh), (cx - hw, cy),
             (cx, cy + hh)]
    poly = plt.Polygon(verts, closed=True, facecolor=fc, edgecolor=ec,
                        linewidth=1.3, zorder=3)
    ax.add_patch(poly)
    ax.text(cx, cy, text, ha="center", va="center",
            fontsize=fontsize, color=text_color, fontweight="bold",
            zorder=4, linespacing=1.25)


def arrow(ax, x1, y1, x2, y2, label="", label_side="left", color=EDGE,
          fontsize=7.5, lw=1.2, label_offset=0.0, label_offset_y=0.0):
    ax.annotate(
        "", xy=(x2, y2), xytext=(x1, y1),
        arrowprops=dict(arrowstyle="-|>", color=color, lw=lw,
                        shrinkA=2, shrinkB=2),
        zorder=2,
    )
    if label:
        mx = (x1 + x2) / 2
        my = (y1 + y2) / 2
        ha = "center"
        if label_side == "left":
            offset_x = -0.3 + label_offset
            ha = "right"
        elif label_side == "right":
            offset_x = 0.3 + label_offset
            ha = "left"
        else:
            offset_x = label_offset
        ax.text(mx + offset_x, my + label_offset_y, label,
                fontsize=fontsize, color=TXT,
                ha=ha, va="center", fontstyle="italic", zorder=5)


def arrow_corner(ax, x1, y1, xm, ym, x2, y2, label="", color=EDGE,
                 fontsize=7.5, lw=1.2, label_offset_x=0.0, label_offset_y=0.0):
    """L-shaped arrow: (x1,y1) -> (xm,ym) -> (x2,y2)."""
    ax.plot([x1, xm], [y1, ym], color=color, lw=lw, zorder=2)
    ax.annotate(
        "", xy=(x2, y2), xytext=(xm, ym),
        arrowprops=dict(arrowstyle="-|>", color=color, lw=lw,
                        shrinkA=0, shrinkB=2),
        zorder=2,
    )
    if label:
        ax.text(xm + label_offset_x, ym + label_offset_y, label,
                fontsize=fontsize, color=TXT, ha="center", va="center",
                fontstyle="italic", zorder=5)


# ── Layout ──────────────────────────────────────────────────────────────────
CX = 6.0
BOX_W = 4.6
BOX_H = 0.72
DIA_W = 3.4
DIA_H = 1.15
STEP = 1.5

y = 17.8

# Title
ax.text(CX, y + 0.2, "Posterior Covariance Beacon Selection Policy",
        fontsize=13, ha="center", va="center", fontweight="bold", color=TXT)
y -= 0.7

# ── 1. Inputs ───────────────────────────────────────────────────────────────
draw_box(ax, CX, y, BOX_W + 0.6, BOX_H + 0.15,
         "Inputs: EKF covariance $P_k$,  AUV position $\\hat{\\mathbf{p}}_k$,\n"
         "beacon positions $\\{\\mathbf{s}_i\\}$, available beacon IDs",
         fc=BLUE, fontsize=8.5, fontweight="bold")
y_input = y
y -= STEP

# ── 2. Extract position covariance ─────────────────────────────────────────
draw_box(ax, CX, y, BOX_W, BOX_H,
         "Extract position uncertainty\n"
         r"$\sigma_{pos} = \sqrt{\mathrm{tr}(P_{pos})}$",
         fc=BLUE, fontsize=9)
arrow(ax, CX, y_input - (BOX_H + 0.15) / 2, CX, y + BOX_H / 2)
y_extract = y
y -= STEP + 0.1

# ── 3. Uncertainty gate decision ────────────────────────────────────────────
draw_diamond(ax, CX, y, DIA_W + 0.6, DIA_H + 0.05,
             r"$\sigma_{pos} < \tau_{off}$" "\n(uncertainty\nsufficiently low?)",
             fc=ORANGE, fontsize=8)
arrow(ax, CX, y_extract - BOX_H / 2, CX, y + (DIA_H + 0.05) / 2)
y_gate = y
y -= STEP + 0.3

# ── YES branch (left): acoustics OFF ───────────────────────────────────────
off_x = 2.2
off_y = y_gate - 0.1
draw_box(ax, off_x, off_y, 2.8, BOX_H + 0.1,
         "Acoustics OFF\n(0 beacons active)",
         fc=RED, fontsize=8.5, fontweight="bold")
# Arrow from diamond left to OFF box
arrow_corner(ax,
             CX - (DIA_W + 0.6) / 2, y_gate,
             off_x, y_gate,
             off_x, off_y + (BOX_H + 0.1) / 2,
             label="Yes", fontsize=8,
             label_offset_x=-0.5, label_offset_y=0.18)
y_off = off_y

# ── Re-enable check annotation ─────────────────────────────────────────────
reenable_y = off_y - 0.85
draw_box(ax, off_x, reenable_y, 2.8, BOX_H + 0.1,
         r"Re-enable when""\n"r"$\sigma_{pos} > \tau_{on}$"" (hysteresis)",
         fc="#888888", fontsize=7.5, text_color=WHITE)
arrow(ax, off_x, off_y - (BOX_H + 0.1) / 2, off_x, reenable_y + (BOX_H + 0.1) / 2)

# ── NO branch (down): proceed with selection ───────────────────────────────

# ── 4. Enumerate subsets ────────────────────────────────────────────────────
draw_box(ax, CX, y, BOX_W, BOX_H + 0.15,
         "Enumerate all beacon subsets\n"
         r"$\mathcal{S} = \{S \subseteq \mathrm{available} \mid 1 \leq |S| \leq N\}$",
         fc=BLUE, fontsize=8.5)
arrow(ax, CX, y_gate - (DIA_H + 0.05) / 2, CX, y + (BOX_H + 0.15) / 2,
      label="No", label_side="right")
y_enum = y
y -= STEP + 0.1

# ── 5. Score each subset ───────────────────────────────────────────────────
draw_box(ax, CX, y, BOX_W + 0.8, BOX_H + 0.45,
         "For each subset $S$, compute predicted\n"
         "posterior covariance trace:\n"
         r"$J(S) = \mathrm{tr}\!\left(P - PH_S^\top"
         r"(H_S P H_S^\top + R)^{-1} H_S P\right)"
         r" + \lambda\,|S|$",
         fc=BLUE, fontsize=8.5)
arrow(ax, CX, y_enum - (BOX_H + 0.15) / 2, CX, y + (BOX_H + 0.45) / 2)
y_score = y
y -= STEP + 0.25

# ── 6. Select best ─────────────────────────────────────────────────────────
draw_box(ax, CX, y, BOX_W, BOX_H,
         r"Select best subset:  $S^* = \arg\min_S\, J(S)$",
         fc=GREEN, fontsize=9, fontweight="bold")
arrow(ax, CX, y_score - (BOX_H + 0.45) / 2, CX, y + BOX_H / 2)
y_best = y
y -= STEP + 0.1

# ── 7. Hysteresis check ────────────────────────────────────────────────────
draw_diamond(ax, CX, y, DIA_W + 1.2, DIA_H + 0.1,
             "Hysteresis check:\n"
             r"$J(S^*) + \delta < J(S_{cur})$"
             "\n& dwell satisfied?",
             fc=ORANGE, fontsize=8)
arrow(ax, CX, y_best - BOX_H / 2, CX, y + (DIA_H + 0.1) / 2)
y_hyst = y
y -= STEP + 0.3

# ── YES branch: switch ─────────────────────────────────────────────────────
draw_box(ax, CX - 2.3, y, 2.6, BOX_H + 0.1,
         "Switch to $S^*$\n(reset dwell counter)",
         fc=GREEN, fontsize=8.5, fontweight="bold")
arrow_corner(ax,
             CX - (DIA_W + 1.2) / 2, y_hyst,
             CX - 2.3, y_hyst,
             CX - 2.3, y + (BOX_H + 0.1) / 2,
             label="Yes", fontsize=8,
             label_offset_x=-0.45, label_offset_y=0.18)
y_switch = y

# ── NO branch: hold ────────────────────────────────────────────────────────
draw_box(ax, CX + 2.3, y, 2.6, BOX_H + 0.1,
         "Keep current $S_{cur}$\n(increment dwell)",
         fc="#888888", fontsize=8.5, text_color=WHITE)
arrow_corner(ax,
             CX + (DIA_W + 1.2) / 2, y_hyst,
             CX + 2.3, y_hyst,
             CX + 2.3, y + (BOX_H + 0.1) / 2,
             label="No", fontsize=8,
             label_offset_x=0.45, label_offset_y=0.18)
y_hold = y
y -= STEP + 0.1

# ── 8. Output ──────────────────────────────────────────────────────────────
draw_box(ax, CX, y, BOX_W + 0.4, BOX_H + 0.1,
         "Output: active beacon IDs for EKF\nacoustic range updates",
         fc=PURPLE, fontsize=9, fontweight="bold")

# Arrows from switch and hold to output
arrow(ax, CX - 2.3, y_switch - (BOX_H + 0.1) / 2,
      CX - 0.5, y + (BOX_H + 0.1) / 2)
arrow(ax, CX + 2.3, y_hold - (BOX_H + 0.1) / 2,
      CX + 0.5, y + (BOX_H + 0.1) / 2)
y_out = y

# ── Legend ──────────────────────────────────────────────────────────────────
legend_items = [
    mpatches.Patch(facecolor=BLUE, edgecolor=EDGE, label="Process / computation"),
    mpatches.Patch(facecolor=ORANGE, edgecolor=EDGE, label="Decision"),
    mpatches.Patch(facecolor=GREEN, edgecolor=EDGE, label="Selection result"),
    mpatches.Patch(facecolor=RED, edgecolor=EDGE, label="Acoustics disabled"),
    mpatches.Patch(facecolor=PURPLE, edgecolor=EDGE, label="Output"),
]
ax.legend(handles=legend_items, loc="lower center", ncol=3,
          fontsize=7.5, framealpha=0.9, edgecolor="#d0d0d0",
          bbox_to_anchor=(0.5, -0.01))

# ── Key notation box (bottom right) ────────────────────────────────────────
notation = (
    r"$P_{pos}$: position covariance (3$\times$3 block of $P$)" "\n"
    r"$H_S$: range Jacobian for subset $S$" "\n"
    r"$R$: range measurement noise" "\n"
    r"$\lambda$: size penalty weight" "\n"
    r"$\delta$: switch margin" "\n"
    r"$\tau_{on},\,\tau_{off}$: uncertainty gate thresholds"
)
ax.text(10.8, y_out - 1.8, notation, fontsize=7, color=TXT,
        ha="right", va="top", linespacing=1.5,
        bbox=dict(boxstyle="round,pad=0.4", facecolor="#F9F9F9",
                  edgecolor="#CCCCCC", linewidth=0.8))

# ── Save ────────────────────────────────────────────────────────────────────
for ext in ("png", "svg", "pdf"):
    fig.savefig(
        f"/home/hamza/holodev_acoustic_manager/kalmaning/figures/"
        f"fig2_beacon_selection_policy.{ext}",
        dpi=300, bbox_inches="tight", facecolor=WHITE,
    )
plt.close(fig)
print("Figure 2 saved (png, svg, pdf).")
