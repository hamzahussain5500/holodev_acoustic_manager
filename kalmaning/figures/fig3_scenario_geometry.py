"""
Figure 3: Scenario Geometry — SBL Beacon Array and AUV Spiral Trajectory

Single XY plan view showing the AUV spiral trajectory and the SBL beacon
cluster, with an inset zoom on the beacon array for readability.

Outputs saved to kalmaning/figures/results_scenario_geometry/
"""

import sys
import os

# Avoid mixed system/user Matplotlib toolkits (fixes KeyError: '3d')
sys.path = [p for p in sys.path if "/usr/lib/python3/dist-packages" not in p]
for _k in list(sys.modules.keys()):
    if "mpl_toolkits" in _k:
        del sys.modules[_k]

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 (registers '3d' projection)
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import matplotlib.patches as mpatches
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from trajectory import build_trajectory

# ── Colour palette (muted, matches fig1/fig2) ────────────────────────────
STEEL_BLUE  = "#4878A8"
SAGE_GREEN  = "#5B9A6D"
CLAY_ORANGE = "#C07840"
MID_GREY    = "#888888"
LIGHT_GREY  = "#E0E0E0"
TXT         = "#333333"
WHITE       = "#FFFFFF"

# ── Beacon positions (METHODOLOGY.md §2.6.2) ─────────────────────────────
BEACONS = {
    "USV 1": (np.array([0.0,  -660.0,   0.0]), "s"),
    "USV 2": (np.array([15.0, -660.0,   0.0]), "D"),
    "USV 3": (np.array([10.0, -650.0,   0.0]), "^"),
    "USV 4": (np.array([0.0,  -650.0, -10.0]), "v"),
}
BCN_NAMES = list(BEACONS.keys())

# ── Generate trajectory ──────────────────────────────────────────────────
traj = build_trajectory("spiral", {
    "spiral_center": (200.0, -200.0),
    "spiral_min_radius": 20.0, "spiral_max_radius": 50.0,
    "spiral_turns": 6, "spiral_points_per_rev": 250,
    "spiral_z_start": -5.0, "spiral_z_end": -150.0,
    "max_segment_length": 5.0,
})

# ── Derived ──────────────────────────────────────────────────────────────
bcn_pos = np.array([v[0] for v in BEACONS.values()])
bcn_ctr = bcn_pos.mean(axis=0)
traj_ctr = traj.mean(axis=0)
sep = np.linalg.norm(traj_ctr[:2] - bcn_ctr[:2])

# ══════════════════════════════════════════════════════════════════════════
# Main figure
# ══════════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(5.5, 7.5))
fig.patch.set_facecolor(WHITE)
ax.set_facecolor(WHITE)

# Trajectory
ax.plot(traj[:, 0], traj[:, 1], color=STEEL_BLUE, lw=1.2, zorder=3,
        label="AUV spiral trajectory")
ax.plot(traj[0, 0], traj[0, 1], marker="o", color=SAGE_GREEN, ms=7,
        zorder=5, markeredgecolor=WHITE, markeredgewidth=0.8, label="AUV spawn")

# Beacon cluster — single combined marker on main plot
ax.plot(bcn_ctr[0], bcn_ctr[1], marker="*", color=CLAY_ORANGE, ms=14,
        zorder=5, markeredgecolor=WHITE, markeredgewidth=0.6,
        label="SBL beacon array")

# Separation line
ax.plot([traj_ctr[0], bcn_ctr[0]], [traj_ctr[1], bcn_ctr[1]],
        ls="--", color=MID_GREY, lw=0.9, zorder=2)

# Distance label at midpoint
mid_x = (traj_ctr[0] + bcn_ctr[0]) / 2
mid_y = (traj_ctr[1] + bcn_ctr[1]) / 2
ax.text(mid_x + 20, mid_y, f"{sep:.0f} m",
        fontsize=10, color=TXT, ha="left", va="center", fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.2", fc=WHITE, ec=LIGHT_GREY, lw=0.6),
        zorder=6)

# Styling
ax.set_xlabel("X (m)", fontsize=10, color=TXT)
ax.set_ylabel("Y (m)", fontsize=10, color=TXT)
ax.set_aspect("equal")
ax.tick_params(labelsize=8, colors=TXT)
for sp in ["top", "right"]:
    ax.spines[sp].set_visible(False)
for sp in ax.spines.values():
    sp.set_color(LIGHT_GREY)
ax.grid(True, lw=0.3, color=LIGHT_GREY, zorder=0)

# Legend — clean, out of the way
ax.legend(fontsize=7.5, loc="center left", framealpha=0.95,
          edgecolor=LIGHT_GREY, handletextpad=0.5, borderpad=0.5,
          bbox_to_anchor=(0.0, 0.55))

# ══════════════════════════════════════════════════════════════════════════
# Inset: zoomed view of the SBL beacon cluster
# ══════════════════════════════════════════════════════════════════════════
# Position the inset in the lower-right area of the figure (empty space)
ax_inset = fig.add_axes([0.48, 0.08, 0.48, 0.30])  # [left, bottom, width, height]
ax_inset.set_facecolor("#FAFAFA")

# Draw each beacon with its own marker and a readable label
label_offsets = {
    "USV 1": (-3.5, -3.5, "right", "top"),
    "USV 2": (3.5,  -3.5, "left",  "top"),
    "USV 3": (3.5,   3.0, "left",  "bottom"),
    "USV 4": (-3.5,  3.0, "right", "bottom"),
}
for name, (pos, mkr) in BEACONS.items():
    ax_inset.plot(pos[0], pos[1], marker=mkr, color=CLAY_ORANGE, ms=10,
                  zorder=5, markeredgecolor=WHITE, markeredgewidth=0.8)
    dx, dy, ha, va = label_offsets[name]
    ax_inset.annotate(
        name, xy=(pos[0], pos[1]),
        xytext=(pos[0] + dx, pos[1] + dy),
        fontsize=8, color=TXT, ha=ha, va=va, fontweight="bold",
        zorder=6,
    )

# Dimension annotations
x_span = np.ptp(bcn_pos[:, 0])
y_span = np.ptp(bcn_pos[:, 1])

# Horizontal span
arr_y = bcn_pos[:, 1].min() - 6
ax_inset.annotate("", xy=(bcn_pos[:, 0].max(), arr_y),
                   xytext=(bcn_pos[:, 0].min(), arr_y),
                   arrowprops=dict(arrowstyle="<->", color=MID_GREY, lw=1.0))
ax_inset.text(bcn_pos[:, 0].mean(), arr_y - 2, f"{x_span:.0f} m",
              fontsize=7.5, color=MID_GREY, ha="center", va="top")

# Vertical span
arr_x = bcn_pos[:, 0].max() + 6
ax_inset.annotate("", xy=(arr_x, bcn_pos[:, 1].max()),
                   xytext=(arr_x, bcn_pos[:, 1].min()),
                   arrowprops=dict(arrowstyle="<->", color=MID_GREY, lw=1.0))
ax_inset.text(arr_x + 2, bcn_pos[:, 1].mean(), f"{y_span:.0f} m",
              fontsize=7.5, color=MID_GREY, ha="left", va="center")

# Inset title
ax_inset.set_title("SBL Beacon Array (zoomed)", fontsize=8.5,
                    fontweight="bold", color=TXT, pad=6)
ax_inset.set_xlabel("X (m)", fontsize=7.5, color=TXT)
ax_inset.set_ylabel("Y (m)", fontsize=7.5, color=TXT)
ax_inset.tick_params(labelsize=7, colors=TXT)

# Pad the inset view
pad = 12
ax_inset.set_xlim(bcn_pos[:, 0].min() - pad, bcn_pos[:, 0].max() + pad)
ax_inset.set_ylim(bcn_pos[:, 1].min() - pad, bcn_pos[:, 1].max() + pad)
ax_inset.set_aspect("equal")
ax_inset.grid(True, lw=0.3, color=LIGHT_GREY, zorder=0)
for sp in ax_inset.spines.values():
    sp.set_color(CLAY_ORANGE)
    sp.set_linewidth(1.0)

# ── Connector line from main beacon marker to inset ──────────────────────
# Draw in figure coordinates so it connects across axes
from matplotlib.patches import ConnectionPatch
con = ConnectionPatch(
    xyA=(bcn_ctr[0], bcn_ctr[1]), coordsA=ax.transData,
    xyB=(bcn_pos[:, 0].min() - pad, bcn_pos[:, 1].max() + pad), coordsB=ax_inset.transData,
    color=CLAY_ORANGE, lw=0.8, ls=":", alpha=0.7,
)
fig.add_artist(con)

# ── Save ─────────────────────────────────────────────────────────────────
out_dir = os.path.join(os.path.dirname(__file__), "results_scenario_geometry")
os.makedirs(out_dir, exist_ok=True)
for ext in ("png", "svg", "pdf"):
    fig.savefig(os.path.join(out_dir, f"fig3_scenario_geometry.{ext}"),
                dpi=300, bbox_inches="tight", facecolor=WHITE)
plt.close(fig)

# ══════════════════════════════════════════════════════════════════════════
# 3D figure: spiral trajectory
# ══════════════════════════════════════════════════════════════════════════
fig3d = plt.figure(figsize=(8, 5.5))
fig3d.patch.set_facecolor(WHITE)
ax3d = fig3d.add_subplot(111, projection="3d")

# Plot trajectory (x, y, z)
ax3d.plot(traj[:, 0], traj[:, 1], traj[:, 2],
          color=STEEL_BLUE, lw=1.2, label="AUV spiral trajectory")
ax3d.scatter(traj[0, 0], traj[0, 1], traj[0, 2],
             color=SAGE_GREEN, s=28, label="AUV spawn", zorder=5)

# 3D styling to match the paper aesthetic
ax3d.set_title("3D spiral trajectory (acoustic localization)", fontsize=10, color=TXT, pad=10)
ax3d.set_xlabel("X (m)", fontsize=9, color=TXT, labelpad=6)
ax3d.set_ylabel("Y (m)", fontsize=9, color=TXT, labelpad=6)
ax3d.set_zlabel("Z (m)", fontsize=9, color=TXT, labelpad=6)
ax3d.tick_params(labelsize=7, colors=TXT)
ax3d.grid(True, lw=0.3, color=LIGHT_GREY)
ax3d.view_init(elev=30, azim=-60)
ax3d.legend(fontsize=7.5, loc="upper right", framealpha=0.95, edgecolor=LIGHT_GREY)

for ext in ("png", "svg", "pdf"):
    fig3d.savefig(os.path.join(out_dir, f"fig3_spiral_trajectory_3d.{ext}"),
                  dpi=300, bbox_inches="tight", facecolor=WHITE)
plt.close(fig3d)

print(f"Figure 3 saved to {out_dir}/ (2D + 3D, png/svg/pdf).")
