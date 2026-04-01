#!/usr/bin/env bash
# run_comparative_analysis.sh
#
# Comparative analysis of three modem-switching policies:
#   1. Manual (staged dropout baseline)
#   2. GDOP policy (geometry-only, baseline adaptive)
#   3. Weighted policy (multi-objective: geometry + energy + mission)
#   4. V2 policy (posterior covariance approximation, energy-aware)
#
# Usage:
#   bash run_comparative_analysis.sh [--duration 900] [--seed 0] [--traj spiral]
#                                    [--outdir results_comparative] [--bat-wh 10.0]
#
# All time-dependent parameters (phase schedules, dropout windows, battery drain)
# are automatically scaled to the requested duration.

set -euo pipefail
cd "$(dirname "$0")"

# ── Defaults ─────────────────────────────────────────────────────────────────
OUTDIR="results_comparative"
TRAJ="spiral"
DURATION=180
SEED=0
BAT_WH=10.0          # battery capacity (Wh)

# ── Parse CLI flags ───────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        --duration)  DURATION="$2";  shift 2 ;;
        --seed)       SEED="$2";      shift 2 ;;
        --traj)       TRAJ="$2";      shift 2 ;;
        --outdir)     OUTDIR="$2";    shift 2 ;;
        --bat-wh)     BAT_WH="$2";    shift 2 ;;
        *)
            echo "[ERROR] Unknown argument: $1"
            echo "Usage: $0 [--duration SEC] [--seed N] [--traj TYPE] [--outdir DIR] [--bat-wh WH]"
            exit 1 ;;
    esac
done

# ── Shared geometry parameters ────────────────────────────────────────────────
SIGMA_R=0.5          # range noise std (m) — matches EKF acoustic_measurement_std
GDOP_XY=8.0          # XY GDOP feasibility threshold
GDOP_3D=10.0

# ── Energy model ──────────────────────────────────────────────────────────────
# DRAIN_SCALE is calibrated so SOC drains 1.0→~0.05 over DURATION seconds
# with 2 active beacons.  Formula:
#   total_drain = DRAIN_SCALE × (BASE_W + 2×BEACON_W) × DURATION / (BAT_WH × 3600)
#   → DRAIN_SCALE = 0.95 × BAT_WH × 3600 / ((BASE_W + 2×BEACON_W) × DURATION)
BASE_W=4.0
BEACON_W=3.0
SOC_INIT=1.0
SOC_MIN=0.05
# Use python to compute DRAIN_SCALE once
DRAIN_SCALE=$(python3 -c "
bat=${BAT_WH}; base=${BASE_W}; bcn=${BEACON_W}; dur=${DURATION}
drain = 0.95 * bat * 3600.0 / ((base + 2*bcn) * dur)
print(f'{drain:.4f}')
")

# ── Time-scaled windows (proportional to DURATION) ───────────────────────────
# All absolute times are fractions of total duration:
#   Manual dropout:   usv4 off at 25%, usv3 off at 50%, usv2 off at 75%
#   Weighted phases:  survey 0-33%, transit 33-67%, low_power 67-100%
#   GDOP low_power:   SOC threshold 0.6 (unchanged — battery calibrated above)
T33=$(python3 -c "print(int(${DURATION}*1/3))")
T50=$(python3 -c "print(int(${DURATION}*1/2))")
T67=$(python3 -c "print(int(${DURATION}*2/3))")
T25=$(python3 -c "print(int(${DURATION}*1/4))")
T75=$(python3 -c "print(int(${DURATION}*3/4))")

DROPOUT_USV4="usv4:${T25}-${DURATION}"
DROPOUT_USV3="usv3:${T50}-${DURATION}"
DROPOUT_USV2="usv2:${T75}-${DURATION}"

PHASE_SCHEDULE="survey:0-${T33},transit:${T33}-${T67},low_power:${T67}-${DURATION}"

echo "============================================================"
echo " Comparative Modem Switching Analysis"
echo " Traj=$TRAJ  Duration=${DURATION}s  Seed=$SEED"
echo " Battery=${BAT_WH}Wh  DrainScale=${DRAIN_SCALE}"
echo " Manual dropouts: ${DROPOUT_USV4} ${DROPOUT_USV3} ${DROPOUT_USV2}"
echo " Weighted phases: ${PHASE_SCHEDULE}"
echo " Output: $OUTDIR/"
echo "============================================================"

# ─────────────────────────────────────────────────────────────────────────────
# 1. MANUAL BASELINE — staged dropouts 4→3→2→1
# ─────────────────────────────────────────────────────────────────────────────
echo ""
echo "--- [1/4] Manual baseline (staged dropout 4→3→2) ---"
python3 modem_switching_validation_fixed.py \
  --outdir "$OUTDIR/manual" \
  --mode manual \
  --traj "$TRAJ" --duration "$DURATION" --seed "$SEED" \
  --targets usv1 usv2 usv3 usv4 \
  --dropout "$DROPOUT_USV4" "$DROPOUT_USV3" "$DROPOUT_USV2" \
  --sigma-r "$SIGMA_R" \
  --make-plots

# ─────────────────────────────────────────────────────────────────────────────
# 2. GDOP POLICY — geometry + energy-aware size penalty
# ─────────────────────────────────────────────────────────────────────────────
echo ""
echo "--- [2/4] GDOP policy (geometry + energy-aware) ---"
python3 modem_switching_validation_fixed.py \
  --outdir "$OUTDIR/gdop" \
  --mode policy --policy-type gdop \
  --traj "$TRAJ" --duration "$DURATION" --seed "$SEED" \
  --targets usv1 usv2 usv3 usv4 \
  --sigma-r "$SIGMA_R" \
  --gdop-xy 30.0 --gdop-3d 40.0 \
  --min-beacons-xy 2 --min-beacons-3d 3 \
  --min-dwell-sec 15.0 \
  --switch-margin 2.0 \
  --size-penalty 0.5 \
  --battery-wh "$BAT_WH" --base-drain-w "$BASE_W" --beacon-drain-w "$BEACON_W" \
  --drain-scale "$DRAIN_SCALE" --soc-init "$SOC_INIT" --soc-min "$SOC_MIN" \
  --low-power-soc 0.6 \
  --make-plots

# ─────────────────────────────────────────────────────────────────────────────
# 3. WEIGHTED POLICY — multi-objective with scaled phase schedule
# Phases scale with duration: survey (0-33%) → transit (33-67%) → low_power (67-100%)
# Phase weights calibrated for near-degenerate SBL geometry:
#   survey:    (0.85, 0.08, 0.07) — w_obs/w_energy=10.6 > 4.2 threshold, n=4 wins
#   transit:   (0.25, 0.60, 0.15) — energy-dominant; n=2 wins
#   low_power: (0.15, 0.75, 0.10) — maximum energy conservation
# ─────────────────────────────────────────────────────────────────────────────
echo ""
echo "--- [3/4] Weighted policy (multi-objective + phase schedule) ---"
python3 modem_switching_validation_fixed.py \
  --outdir "$OUTDIR/weighted" \
  --mode policy --policy-type weighted \
  --traj "$TRAJ" --duration "$DURATION" --seed "$SEED" \
  --targets usv1 usv2 usv3 usv4 \
  --sigma-r "$SIGMA_R" \
  --gdop-xy "$GDOP_XY" --gdop-3d "$GDOP_3D" \
  --min-beacons-xy 2 --min-beacons-3d 3 \
  --min-dwell-sec 10.0 \
  --score-margin 0.03 \
  --power-save-tol 0.0 \
  --phase-schedule "$PHASE_SCHEDULE" \
  --target-unc-xy 0.5 --target-unc-3d 0.8 \
  --off-unc-mult 1.5 \
  --allow-zero-beacons \
  --battery-wh "$BAT_WH" --base-drain-w "$BASE_W" --beacon-drain-w "$BEACON_W" \
  --drain-scale "$DRAIN_SCALE" --soc-init "$SOC_INIT" --soc-min "$SOC_MIN" \
  --low-power-soc 0.3 \
  --energy-weight 0.3 \
  --make-plots

# ─────────────────────────────────────────────────────────────────────────────
# 4. V2 POLICY — posterior covariance approximation
# Four switching mechanisms (all time-independent — driven by EKF state):
#   1. Energy-driven subset reduction (4→2, energy_weight=0.195)
#   2. Uncertainty gate-off (2→0, off_thresh = 0.55×1.5 = 0.825m)
#   3. Uncertainty drift re-enable (0→2, on_thresh ≈ 1.031m)
#   4. Energy-based gate-off at low SOC (2→0, score_zero_below_soc=0.4)
# V2 parameters are largely duration-independent (EKF state driven) except
# min_dwell_sec which stays fixed at 15s regardless of total duration.
# ─────────────────────────────────────────────────────────────────────────────
echo ""
echo "--- [4/4] V2 policy (posterior covariance approximation) ---"
python3 modem_switching_validation_fixed.py \
  --outdir "$OUTDIR/v2" \
  --mode policy --policy-type v2 \
  --traj "$TRAJ" --duration "$DURATION" --seed "$SEED" \
  --targets usv1 usv2 usv3 usv4 \
  --sigma-r "$SIGMA_R" \
  --min-beacons-xy 0 --min-beacons-3d 0 \
  --allow-zero-beacons \
  --gdop-xy "$GDOP_XY" --gdop-3d "$GDOP_3D" \
  --min-dwell-sec 15.0 \
  --switch-margin 0.005 \
  --size-penalty 0.0 \
  --target-unc-xy 1.5 --target-unc-3d 2.0 \
  --off-unc-mult 0.55 \
  --battery-wh "$BAT_WH" --base-drain-w "$BASE_W" --beacon-drain-w "$BEACON_W" \
  --drain-scale "$DRAIN_SCALE" --soc-init "$SOC_INIT" --soc-min "$SOC_MIN" \
  --energy-weight 0.195 \
  --v2-low-power-soc 0.45 \
  --v2-energy-mult 5.0 \
  --v2-size-penalty-mult 1.0 \
  --v2-rank-deficit-penalty 5.0 \
  --v2-rank-deficit-mult 0.5 \
  --v2-score-zero-below-soc 0.4 \
  --make-plots

# ─────────────────────────────────────────────────────────────────────────────
# Summary table
# ─────────────────────────────────────────────────────────────────────────────
echo ""
echo "============================================================"
echo " Results Summary"
echo "============================================================"
python3 - "$OUTDIR" << 'PYEOF'
import json, sys, pathlib, os

outdir = pathlib.Path(sys.argv[1])
if not outdir.exists():
    print("  (output directory not found:", outdir, ")")
    sys.exit(0)

run_dirs = []
for subdir in sorted(outdir.iterdir()):
    if subdir.is_dir():
        for run_dir in sorted(subdir.iterdir(), key=os.path.getmtime):
            if run_dir.is_dir() and (run_dir / "summary.json").exists():
                run_dirs.append((subdir.name, run_dir))

print(f"{'Policy':<12} {'Run tag':<38} {'RMSE(m)':>8} {'Switches':>9} {'Active-set counts'}")
print("-" * 100)
for policy, d in run_dirs:
    sumf  = d / "summary.json"
    metaf = d / "selector_meta.json"
    try:
        rmse = json.loads(sumf.read_text()).get("rmse_pos", float("nan"))
    except Exception:
        rmse = float("nan")
    n_switches, n_active_vals, switch_times = 0, [], []
    if metaf.exists():
        try:
            meta = json.loads(metaf.read_text())
            prev = None
            for r in meta:
                sel = tuple(sorted(r.get("selected", [])))
                if prev is not None and sel != prev:
                    n_switches += 1
                    switch_times.append(f"t={r['t']:.0f}s:{len(list(prev))}→{len(sel)}")
                prev = sel
            n_active_vals = [len(r.get("selected", [])) for r in meta]
        except Exception:
            pass
    unique_counts = sorted(set(n_active_vals))
    print(f"{policy:<12} {d.name:<38} {rmse:>8.4f} {n_switches:>9d}   {unique_counts}")
    if switch_times:
        print(f"{'':>52} Switches: {', '.join(switch_times)}")
PYEOF

echo ""
echo "Figures saved under $OUTDIR/<run-tag>/"
echo ""
echo "To generate the trajectory + switch-event plot:"
echo "  python3 plot_comparative_trajectories.py --results-dir $OUTDIR --outdir $OUTDIR"
