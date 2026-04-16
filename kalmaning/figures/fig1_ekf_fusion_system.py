"""
Figure 1: EKF Fusion System Overview
Publication-quality flowchart for range-only acoustic localisation with
adaptive beacon management.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

# ── Colour palette (muted, publication-friendly) ──────────────────────────
BLUE    = "#4878A8"   # primary / process
GREEN   = "#5B9A6D"   # sensor updates
ORANGE  = "#C07840"   # decision / selection
RED     = "#B05050"   # highlights
PURPLE  = "#7A6B90"   # output
GREY_BG = "#F5F5F5"
EDGE    = "#555555"
TXT     = "#333333"
WHITE   = "#FFFFFF"

# ── Canvas ──────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8.5, 12))
ax.set_xlim(0, 10)
ax.set_ylim(0, 14)
ax.axis("off")
fig.patch.set_facecolor(WHITE)

# ── Helper functions ────────────────────────────────────────────────────────
def draw_box(ax, cx, cy, w, h, text, fc, ec=EDGE, fontsize=9,
             fontweight="normal", text_color=WHITE, style="round,pad=0.15",
             alpha=1.0, zorder=3):
    """Draw a rounded box centred at (cx, cy)."""
    box = FancyBboxPatch(
        (cx - w / 2, cy - h / 2), w, h,
        boxstyle=style, facecolor=fc, edgecolor=ec,
        linewidth=1.3, alpha=alpha, zorder=zorder,
    )
    ax.add_patch(box)
    ax.text(cx, cy, text, ha="center", va="center",
            fontsize=fontsize, color=text_color, fontweight=fontweight,
            zorder=zorder + 1, linespacing=1.35)
    return box


def draw_diamond(ax, cx, cy, w, h, text, fc=ORANGE, ec=EDGE, fontsize=8,
                 text_color=WHITE):
    """Draw a decision diamond centred at (cx, cy)."""
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
          fontsize=7.5, lw=1.2, label_dx=0.0, label_dy=0.0):
    """Draw an arrow with optional label."""
    ax.annotate(
        "", xy=(x2, y2), xytext=(x1, y1),
        arrowprops=dict(arrowstyle="-|>", color=color, lw=lw,
                        shrinkA=2, shrinkB=2),
        zorder=2,
    )
    if label:
        mx = (x1 + x2) / 2
        my = (y1 + y2) / 2
        offset_x = -0.45 if label_side == "left" else 0.45
        ha = "right" if label_side == "left" else "left"
        ax.text(mx + offset_x + label_dx, my + label_dy, label,
                fontsize=fontsize, color=TXT,
                ha=ha, va="center", fontstyle="italic", zorder=5)


# ── Layout constants ────────────────────────────────────────────────────────
CX = 5.0          # column centre
BOX_W = 4.2
BOX_H = 0.7
DIA_W = 3.0
DIA_H = 1.0
STEP = 1.35       # vertical spacing

# ── Vertical positions (top to bottom) ──────────────────────────────────────
y = 13.0

# Title
ax.text(CX, y + 0.3, "EKF Sensor Fusion System", fontsize=14,
        ha="center", va="center", fontweight="bold", color=TXT)
y -= 0.6

# ── 1. Initialisation ──────────────────────────────────────────────────────
draw_box(ax, CX, y, BOX_W, BOX_H,
         "Initialise EKF state\n"
         r"$\mathbf{x}_0 = [\mathbf{p},\,\mathbf{v}]^\top$,  $\mathbf{P}_0$",
         fc=BLUE, fontsize=9, fontweight="bold")
y_init = y
y -= STEP

# ── 2. IMU Predict ─────────────────────────────────────────────────────────
draw_box(ax, CX, y, BOX_W, BOX_H,
         "IMU Prediction Step\n"
         r"$\mathbf{x}^- = F\mathbf{x} + B\mathbf{a}_w$,"
         r"  $P^- = FPF^\top + Q$",
         fc=BLUE, fontsize=8.5)
arrow(ax, CX, y_init - BOX_H / 2, CX, y + BOX_H / 2)
y_imu = y
y -= STEP

# ── 3. DVL Update ──────────────────────────────────────────────────────────
draw_box(ax, CX, y, BOX_W, BOX_H,
         "DVL Velocity Update (Linear)\n"
         r"$\mathbf{z}_{dvl} = H_{dvl}\,\mathbf{x} + \mathbf{n}_{dvl}$",
         fc=GREEN, fontsize=8.5)
arrow(ax, CX, y_imu - BOX_H / 2, CX, y + BOX_H / 2)
y_dvl = y
y -= STEP

# ── 4. Depth Update ────────────────────────────────────────────────────────
draw_box(ax, CX, y, BOX_W, BOX_H,
         "Depth Update (Linear)\n"
         r"$z_{depth} = H_{depth}\,\mathbf{x} + n_{depth}$",
         fc=GREEN, fontsize=8.5)
arrow(ax, CX, y_dvl - BOX_H / 2, CX, y + BOX_H / 2)
y_depth = y
y -= STEP

# ── 5. Beacon Selection decision ───────────────────────────────────────────
draw_diamond(ax, CX, y, DIA_W + 0.4, DIA_H + 0.1,
             "Beacon Selection\nPolicy", fc=ORANGE, fontsize=8.5)
arrow(ax, CX, y_depth - BOX_H / 2, CX, y + (DIA_H + 0.1) / 2)
y_sel = y
y -= STEP + 0.15

# ── 6. Acoustic Range Update ───────────────────────────────────────────────
draw_box(ax, CX, y, BOX_W, BOX_H + 0.15,
         "Acoustic Range Update (Nonlinear)\n"
         r"$z_r = \|\mathbf{p} - \mathbf{s}_i\| + n_r$"
         "  for selected beacons",
         fc=GREEN, fontsize=8.5)
arrow(ax, CX, y_sel - (DIA_H + 0.1) / 2, CX, y + (BOX_H + 0.15) / 2,
      label="Active\nbeacons", label_side="left", label_dx=-0.15)
y_acous = y
y -= STEP

# ── 7. Output ──────────────────────────────────────────────────────────────
draw_box(ax, CX, y, BOX_W, BOX_H,
         r"Updated state  $\hat{\mathbf{x}}_k$  &  covariance  $P_k$",
         fc=PURPLE, fontsize=9, fontweight="bold")
arrow(ax, CX, y_acous - (BOX_H + 0.15) / 2, CX, y + BOX_H / 2)
y_out = y

# ── Loop-back arrow ────────────────────────────────────────────────────────
loop_x = CX + BOX_W / 2 + 0.6
ax.annotate(
    "", xy=(CX + BOX_W / 2, y_imu),
    xytext=(loop_x, y_out),
    arrowprops=dict(
        arrowstyle="-|>", color=EDGE, lw=1.2,
        connectionstyle="arc3,rad=-0.35",
        shrinkA=4, shrinkB=4,
    ),
    zorder=2,
)
ax.text(loop_x + 0.3, (y_imu + y_out) / 2, "Next\ntimestep",
        fontsize=7.5, color=TXT, ha="left", va="center",
        fontstyle="italic")

# ── Side annotation: "See Figure 2" near diamond ───────────────────────────
ax.annotate(
    "See Figure 2",
    xy=(CX - DIA_W / 2 - 0.2, y_sel),
    xytext=(CX - DIA_W / 2 - 1.6, y_sel),
    fontsize=8, color=ORANGE, fontweight="bold",
    ha="center", va="center",
    arrowprops=dict(arrowstyle="-|>", color=ORANGE, lw=1.0),
    zorder=5,
)

# ── Sensor-rate annotations (right side) ────────────────────────────────────
info_x = CX + BOX_W / 2 + 0.45
for yy, txt in [
    (y_imu,   "100 Hz"),
    (y_dvl,   "20 Hz"),
    (y_depth, "100 Hz"),
    (y_acous, "Async\n(round-robin)"),
]:
    ax.text(info_x, yy, txt, fontsize=7, color="#888888", ha="left",
            va="center", fontstyle="italic")

# ── Save ────────────────────────────────────────────────────────────────────
for ext in ("png", "svg", "pdf"):
    fig.savefig(
        f"/home/hamza/holodev_acoustic_manager/kalmaning/figures/fig1_ekf_fusion_system.{ext}",
        dpi=300, bbox_inches="tight", facecolor=WHITE,
    )
plt.close(fig)
print("Figure 1 saved (png, svg, pdf).")
