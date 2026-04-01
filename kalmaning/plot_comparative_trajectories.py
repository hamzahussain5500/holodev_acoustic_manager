"""
Comparative trajectory plot with modem switch event markers.

Loads selector_meta.json from each policy's result directory, regenerates the
AUV trajectory by running a short passthrough EKF (all beacons always on, same
seed/traj), then plots:

  Panel A  — XY trajectory per policy, colored by active beacon count,
              with ★ markers at each switching event.
  Panel B  — Active beacon count vs time, all policies overlaid.
  Panel C  — Position error vs time, all policies overlaid, shaded by active count.

Usage:
    python3 plot_comparative_trajectories.py [--results-dir results_comparative]
                                              [--outdir .] [--show]
"""

import argparse
import json
import pathlib
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D

# ── helpers ──────────────────────────────────────────────────────────────────

COUNT_COLORS = {
    0: "#d62728",   # red    — acoustics off
    1: "#ff7f0e",   # orange — 1 beacon
    2: "#ff7f0e",   # orange — 2 beacons
    3: "#1f77b4",   # blue   — 3 beacons
    4: "#2ca02c",   # green  — 4 beacons
}
POLICY_COLORS = {
    "manual":   "#555555",
    "gdop":     "#9467bd",
    "weighted": "#d62728",
    "v2":       "#1f77b4",
}
POLICY_LABELS = {
    "manual":   "Manual (baseline)",
    "gdop":     "GDOP",
    "weighted": "Weighted",
    "v2":       "V2",
}
SWITCH_MARKER = "★"


def load_meta(path: pathlib.Path):
    """Return list of per-tick dicts from selector_meta.json."""
    return json.loads(path.read_text())


def find_switch_events(meta):
    """Return list of (t, from_n, to_n, reason, idx) for each subset change."""
    events = []
    prev_sel = None
    for i, r in enumerate(meta):
        sel = tuple(sorted(r.get("selected", [])))
        if prev_sel is not None and sel != prev_sel:
            events.append({
                "t": r["t"],
                "from_n": len(prev_sel),
                "to_n": len(sel),
                "reason": r.get("reason", ""),
                "idx": i,
            })
        prev_sel = sel
    return events


def colored_line_segments(x, y, c_vals, cmap_lut):
    """
    Return a LineCollection where each segment is colored by c_vals (int counts).
    c_vals must have same length as x/y.
    """
    points = np.array([x, y]).T.reshape(-1, 1, 2)
    segs = np.concatenate([points[:-1], points[1:]], axis=1)
    colors = [cmap_lut.get(int(c), "#888888") for c in c_vals[:-1]]
    lc = LineCollection(segs, colors=colors, linewidth=1.5, zorder=2)
    return lc


# ── trajectory reconstruction ─────────────────────────────────────────────────

def get_trajectory(results_root: pathlib.Path, args_override: dict):
    """
    Re-run the EKF with all beacons active (passthrough) to get true_pos / est_pos.
    Returns (times, true_pos, est_pos) arrays of shape (N, 3).

    Uses the config.json from any existing policy run to reconstruct CLI args,
    then calls run_single_trial with no selector (all beacons always on).
    """
    # Use values passed in directly (already resolved from the selected run dirs)
    duration = float(args_override.get("duration", 180.0))
    seed     = int(args_override.get("seed", 0))
    traj     = str(args_override.get("traj", "spiral"))
    targets  = args_override.get("targets", ["usv1", "usv2", "usv3", "usv4"])

    try:
        sys.path.insert(0, str(pathlib.Path(__file__).parent))
        from current_acoustic_EKF_patched import run_single_trial
    except ImportError as e:
        print(f"[ERROR] Cannot import run_single_trial: {e}")
        return None, None, None

    try:
        result = run_single_trial(
            seed=seed,
            config_overrides={
                "duration_sec": duration,
                "trajectory": traj,
                "use_currents": False,
            },
            return_timeseries=True,
            target_names=targets,
            make_plots=False,
        )
        ts = result.get("timeseries", {})
        times    = np.asarray(ts["t"])
        true_pos = np.asarray(ts["true_pos"])   # (N, 3)
        est_pos  = np.asarray(ts["est_pos"])     # (N, 3)
        return times, true_pos, est_pos
    except Exception as e:
        print(f"[WARN] Trajectory reconstruction failed: {e}")
        return None, None, None


# ── main plot ─────────────────────────────────────────────────────────────────

def plot_all(results_root: pathlib.Path, outdir: pathlib.Path, show: bool, args):
    """Build the three-panel comparative figure."""

    # Discover policy run directories
    policy_dirs = {}
    for policy in ("manual", "gdop", "weighted", "v2"):
        subdir = results_root / policy
        if not subdir.exists():
            continue
        # pick the most recently modified run directory
        runs = [d for d in sorted(subdir.iterdir()) if d.is_dir() and (d / "selector_meta.json").exists()]
        if runs:
            policy_dirs[policy] = max(runs, key=lambda d: d.stat().st_mtime)

    if not policy_dirs:
        print(f"[ERROR] No policy result directories found under {results_root}")
        sys.exit(1)

    print(f"Found {len(policy_dirs)} policies: {list(policy_dirs.keys())}")

    # Read scenario metadata from the selected run directories (not rglob —
    # avoids picking up stale config.json from a different-duration run).
    scenario_duration = args.duration
    scenario_seed     = args.seed
    scenario_traj     = args.traj
    for d in policy_dirs.values():
        cfg_path = d / "config.json"
        if cfg_path.exists():
            try:
                cli = json.loads(cfg_path.read_text()).get("cli_args", {})
                scenario_duration = float(cli.get("duration", scenario_duration))
                scenario_seed     = int(cli.get("seed", scenario_seed))
                scenario_traj     = str(cli.get("traj", scenario_traj))
                break
            except Exception:
                pass

    print(f"Scenario: traj={scenario_traj}  duration={scenario_duration:.0f}s  seed={scenario_seed}")

    # Reconstruct shared trajectory using the correct duration/seed/traj
    times_ref, true_pos, est_pos_ref = get_trajectory(
        results_root,
        {"duration": scenario_duration, "seed": scenario_seed, "traj": scenario_traj},
    )

    # Load meta for each policy
    policy_meta   = {}
    policy_events = {}
    policy_ts     = {}

    for policy, d in policy_dirs.items():
        meta = load_meta(d / "selector_meta.json")
        policy_meta[policy] = meta
        policy_events[policy] = find_switch_events(meta)

        # Build arrays aligned with the meta timestamps
        t_arr   = np.array([r["t"] for r in meta])
        n_arr   = np.array([r.get("n_active", r.get("n", 0)) for r in meta])
        err_arr = np.full(len(meta), np.nan)

        # Merge position error from timeseries CSV if available
        import csv
        csv_path = d / "timeseries.csv"
        if csv_path.exists():
            t_csv, err_csv = [], []
            with open(csv_path) as f:
                reader = csv.DictReader(f)
                for row in reader:
                    try:
                        t_csv.append(float(row["t"]))
                        err_csv.append(float(row.get("pos_err", row.get("err_norm", "nan"))))
                    except (ValueError, KeyError):
                        pass
            if t_csv:
                err_csv_arr = np.interp(t_arr, t_csv, err_csv)
                err_arr = err_csv_arr

        policy_ts[policy] = {"t": t_arr, "n": n_arr, "err": err_arr}

    # ── Figure layout ──────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(18, 14))
    fig.suptitle(
        f"Adaptive Modem Switching — Comparative Analysis\n"
        f"{scenario_traj.capitalize()} trajectory  |  {scenario_duration:.0f} s  |  seed {scenario_seed}"
        f"  |  18 m SBL cluster at ~450 m range",
        fontsize=13, fontweight="bold", y=0.98
    )

    # 4 trajectory subplots (top row) + 2 time-series subplots (bottom row)
    gs = fig.add_gridspec(2, 4, hspace=0.38, wspace=0.35,
                          top=0.92, bottom=0.07, left=0.06, right=0.97)

    ax_traj = {p: fig.add_subplot(gs[0, i]) for i, p in enumerate(["manual", "gdop", "weighted", "v2"])}
    ax_count = fig.add_subplot(gs[1, :2])
    ax_err   = fig.add_subplot(gs[1, 2:])

    # ── Beacon positions (approximate — from scenario) ────────────────────────
    # These are USV positions from the HoloOcean scenario; 18 m spread near origin
    beacon_xy = {
        "usv1": (9.0,  0.0),
        "usv2": (-9.0, 0.0),
        "usv3": (0.0,  9.0),
        "usv4": (0.0, -9.0),
    }

    # ── Panel A: Trajectories ──────────────────────────────────────────────────
    for policy, ax in ax_traj.items():
        meta   = policy_meta[policy]
        events = policy_events[policy]
        ts_p   = policy_ts[policy]

        t_meta = ts_p["t"]
        n_meta = ts_p["n"]

        # True trajectory (thin grey)
        if true_pos is not None and len(true_pos) > 0:
            # Interpolate n_active onto the true_pos time grid
            t_ref = times_ref
            n_interp = np.round(np.interp(t_ref, t_meta, n_meta)).astype(int)

            ax.add_collection(
                colored_line_segments(true_pos[:, 0], true_pos[:, 1], n_interp, COUNT_COLORS)
            )
            ax.autoscale_view()

            # Switch event markers on true trajectory
            for ev in events:
                # Find closest index in true_pos time array
                idx = np.argmin(np.abs(t_ref - ev["t"]))
                x_sw, y_sw = true_pos[idx, 0], true_pos[idx, 1]
                ax.plot(x_sw, y_sw, marker="*", markersize=12,
                        color="black", zorder=5, markeredgecolor="white", markeredgewidth=0.5)
                label = f"{ev['from_n']}→{ev['to_n']}"
                ax.annotate(
                    f"t={ev['t']:.0f}s\n{label}",
                    xy=(x_sw, y_sw),
                    xytext=(8, 8), textcoords="offset points",
                    fontsize=6.5, color="black",
                    bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="grey", alpha=0.8),
                    zorder=6,
                )
        else:
            # Fallback: just draw a placeholder
            ax.text(0.5, 0.5, "Trajectory\nnot available", transform=ax.transAxes,
                    ha="center", va="center", fontsize=9, color="grey")

        # Beacon positions
        for bname, (bx, by) in beacon_xy.items():
            ax.plot(bx, by, "^", markersize=7, color="#e377c2", zorder=4,
                    markeredgecolor="white", markeredgewidth=0.5)
        if list(beacon_xy.keys()):
            bxs = [v[0] for v in beacon_xy.values()]
            bys = [v[1] for v in beacon_xy.values()]
            ax.plot(bxs, bys, "^", markersize=7, color="#e377c2", zorder=4,
                    markeredgecolor="white", markeredgewidth=0.5, label="Beacons")

        ax.set_title(f"{POLICY_LABELS.get(policy, policy)}\n({len(events)} switches)",
                     fontsize=9, fontweight="bold")
        ax.set_xlabel("X (m)", fontsize=8)
        ax.set_ylabel("Y (m)", fontsize=8)
        ax.tick_params(labelsize=7)
        ax.set_aspect("equal", adjustable="datalim")
        ax.grid(True, alpha=0.25)

    # ── Shared legend for trajectory color coding ─────────────────────────────
    legend_patches = [
        mpatches.Patch(color=COUNT_COLORS[4], label="4 beacons"),
        mpatches.Patch(color=COUNT_COLORS[3], label="3 beacons"),
        mpatches.Patch(color=COUNT_COLORS[2], label="2 beacons"),
        mpatches.Patch(color=COUNT_COLORS[0], label="0 beacons (off)"),
        Line2D([0], [0], marker="*", color="black", markersize=9, linestyle="None",
               markeredgecolor="white", label="Switch event"),
        Line2D([0], [0], marker="^", color="#e377c2", markersize=7, linestyle="None",
               markeredgecolor="white", label="Beacon"),
    ]
    ax_traj["v2"].legend(handles=legend_patches, loc="upper right",
                         fontsize=6.5, framealpha=0.85, title="Trajectory color", title_fontsize=7)

    # ── Panel B: Active count vs time ─────────────────────────────────────────
    for policy in ("manual", "gdop", "weighted", "v2"):
        if policy not in policy_ts:
            continue
        ts_p = policy_ts[policy]
        t    = ts_p["t"]
        n    = ts_p["n"]
        color = POLICY_COLORS[policy]

        # Downsample for clarity (plot every 10th point)
        step = max(1, len(t) // 1800)
        ax_count.plot(t[::step], n[::step], color=color, linewidth=1.5,
                      label=POLICY_LABELS.get(policy, policy), zorder=3)

        # Mark switch events
        for ev in policy_events[policy]:
            ax_count.axvline(ev["t"], color=color, linewidth=0.7, linestyle="--", alpha=0.5, zorder=2)
            ax_count.annotate(
                f"  {ev['from_n']}→{ev['to_n']}",
                xy=(ev["t"], ev["to_n"]),
                xytext=(2, 3), textcoords="offset points",
                fontsize=6, color=color, zorder=4,
            )

    ax_count.set_xlabel("Time (s)", fontsize=9)
    ax_count.set_ylabel("Active beacons", fontsize=9)
    ax_count.set_title("Active beacon count vs time", fontsize=10, fontweight="bold")
    ax_count.set_ylim(-0.3, 4.6)
    ax_count.set_yticks([0, 1, 2, 3, 4])
    ax_count.legend(fontsize=8, loc="upper right")
    ax_count.grid(True, alpha=0.25)

    # ── Panel C: Position error vs time ───────────────────────────────────────
    for policy in ("manual", "gdop", "weighted", "v2"):
        if policy not in policy_ts:
            continue
        ts_p = policy_ts[policy]
        t    = ts_p["t"]
        err  = ts_p["err"]
        n    = ts_p["n"]
        color = POLICY_COLORS[policy]

        # Shade background by n=0 periods (acoustics off)
        off_mask = n == 0
        if off_mask.any():
            in_off = False
            t0_off = None
            for i, off in enumerate(off_mask):
                if off and not in_off:
                    in_off = True
                    t0_off = t[i]
                elif not off and in_off:
                    in_off = False
                    ax_err.axvspan(t0_off, t[i], alpha=0.08, color=color, zorder=1)
            if in_off:
                ax_err.axvspan(t0_off, t[-1], alpha=0.08, color=color, zorder=1)

        if np.isfinite(err).any():
            step = max(1, len(t) // 1800)
            ax_err.plot(t[::step], err[::step], color=color, linewidth=1.5,
                        label=POLICY_LABELS.get(policy, policy), zorder=3)

    ax_err.set_xlabel("Time (s)", fontsize=9)
    ax_err.set_ylabel("Position error (m)", fontsize=9)
    ax_err.set_title("Position error vs time\n(shaded = acoustics off)", fontsize=10, fontweight="bold")
    ax_err.legend(fontsize=8, loc="upper left")
    ax_err.grid(True, alpha=0.25)

    # ── Save ──────────────────────────────────────────────────────────────────
    outdir.mkdir(parents=True, exist_ok=True)
    out_fname = f"comparative_trajectories_T{int(scenario_duration)}_seed{scenario_seed}.png"
    out_path = outdir / out_fname
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {out_path}")

    if show:
        matplotlib.use("TkAgg")
        plt.show()
    plt.close(fig)

    # ── Per-policy switch summary table ───────────────────────────────────────
    print("\n" + "="*70)
    print(f"{'Policy':<12}  {'Switches':>8}  Switch events")
    print("-"*70)
    for policy in ("manual", "gdop", "weighted", "v2"):
        if policy not in policy_events:
            continue
        events = policy_events[policy]
        ev_str = "  |  ".join(
            f"t={ev['t']:.0f}s: {ev['from_n']}→{ev['to_n']} ({ev['reason'][:18]})"
            for ev in events
        )
        print(f"{POLICY_LABELS.get(policy, policy):<12}  {len(events):>8}  {ev_str}")
    print("="*70)


# ── entry point ───────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results-dir", default="results_comparative",
                    help="Root directory containing manual/, gdop/, weighted/, v2/ subdirs")
    ap.add_argument("--outdir", default=".",
                    help="Directory to write the output figure")
    ap.add_argument("--show", action="store_true",
                    help="Display the figure interactively (requires display)")
    ap.add_argument("--duration", type=float, default=180.0,
                    help="Simulation duration (used if config.json is unavailable)")
    ap.add_argument("--seed", type=int, default=0,
                    help="Random seed (used if config.json is unavailable)")
    ap.add_argument("--traj", default="spiral",
                    help="Trajectory type (used if config.json is unavailable)")
    args = ap.parse_args()

    results_root = pathlib.Path(args.results_dir)
    outdir       = pathlib.Path(args.outdir)

    if not results_root.exists():
        print(f"[ERROR] Results directory not found: {results_root}")
        sys.exit(1)

    plot_all(results_root, outdir, args.show, args)


if __name__ == "__main__":
    main()
