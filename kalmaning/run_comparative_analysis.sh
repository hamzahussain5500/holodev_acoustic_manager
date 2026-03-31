#!/usr/bin/env bash
# run_comparative_analysis.sh
#
# Comparative analysis of three modem-switching policies:
#   1. Manual (staged dropout baseline)
#   2. GDOP policy (geometry-only, baseline adaptive)
#   3. Weighted policy (multi-objective: geometry + energy + mission)
#   4. V2 policy (posterior covariance approximation, energy-aware)
#
# All runs use identical scenario conditions for direct comparison.
# Outputs: results_comparative/<run-tag>/
#   - selector_meta.json  (per-tick decision log)
#   - timeseries.csv      (EKF state + selector overlaid)
#   - summary.json        (RMSE, final error)
#   - figures (active_count, pos_err, score_vs_time, soc_vs_time, gdop)

set -euo pipefail
cd "$(dirname "$0")"

OUTDIR="results_comparative"
TRAJ="spiral"
DURATION=600
SEED=1

# ── Shared geometry parameters ────────────────────────────────────────────────
SIGMA_R=0.5          # range noise std (m) — matches EKF acoustic_measurement_std
GDOP_XY=8.0          # XY GDOP feasibility threshold
GDOP_3D=10.0

# ── Energy model (shared across all policy runs) ─────────────────────────────
# Calibrated so SOC drains from 1.0 to ~0.15 over 180s with 2 active beacons.
BAT_WH=10.0
BASE_W=4.0           # baseline power draw (W)
BEACON_W=3.0         # per-beacon additional draw (W)
DRAIN_SCALE=20.0     # fractional SOC drain scale factor
SOC_INIT=1.0
SOC_MIN=0.05

echo "============================================================"
echo " Comparative Modem Switching Analysis"
echo " Traj=$TRAJ  Duration=${DURATION}s  Seed=$SEED"
echo " Output: $OUTDIR/"
echo "============================================================"

# ─────────────────────────────────────────────────────────────────────────────
# 1. MANUAL BASELINE — staged dropouts 4→3→2→1→0
# ─────────────────────────────────────────────────────────────────────────────
echo ""
echo "--- [1/4] Manual baseline (staged dropout 4→3→2) ---"
python3 modem_switching_validation_fixed.py \
  --outdir "$OUTDIR/manual" \
  --mode manual \
  --traj "$TRAJ" --duration "$DURATION" --seed "$SEED" \
  --targets usv1 usv2 usv3 usv4 \
  --dropout usv4:45-180 usv3:90-180 usv2:135-180 \
  --sigma-r "$SIGMA_R" \
  --make-plots

# ─────────────────────────────────────────────────────────────────────────────
# 2. GDOP POLICY — geometry + energy-aware size penalty
# When SOC drops below low_power_soc, the size_penalty is multiplied by 3×,
# incentivising the policy to switch to a smaller beacon subset.
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
# 3. WEIGHTED POLICY — multi-objective with phase schedule
# Phases: survey (0-45s) → cruise (45-90s) → transit (90-135s) → low_power (135-180s)
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
  --min-dwell-sec 8.0 \
  --score-margin 0.02 \
  --power-save-tol 0.03 \
  --phase-schedule "survey:0-45,cruise:45-90,transit:90-135,low_power:135-180" \
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
# Shows: 4→2 (energy-driven) + 2→0 (uncertainty gated off) + 0→2 (re-enabled)
# Parameters tuned for the near-degenerate SBL geometry (18m cluster at ~450m range).
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
  --min-dwell-sec 10.0 \
  --switch-margin 0.005 \
  --size-penalty 0.0 \
  --target-unc-xy 1.2 --target-unc-3d 1.5 \
  --off-unc-mult 0.9 \
  --battery-wh "$BAT_WH" --base-drain-w "$BASE_W" --beacon-drain-w "$BEACON_W" \
  --drain-scale "$DRAIN_SCALE" --soc-init "$SOC_INIT" --soc-min "$SOC_MIN" \
  --energy-weight 0.195 \
  --v2-low-power-soc 0.55 \
  --v2-energy-mult 4.0 \
  --v2-size-penalty-mult 1.0 \
  --v2-rank-deficit-penalty 5.0 \
  --v2-rank-deficit-mult 0.5 \
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

# Collect all run dirs two levels deep: outdir/policy_type/run_tag/
run_dirs = []
for subdir in sorted(outdir.iterdir()):
    if subdir.is_dir():
        for run_dir in sorted(subdir.iterdir(), key=os.path.getmtime):
            if run_dir.is_dir() and (run_dir / "summary.json").exists():
                run_dirs.append((subdir.name, run_dir))

print(f"{'Policy':<12} {'Run tag':<38} {'RMSE(m)':>8} {'Switches':>9} {'Active-set counts'}")
print("-" * 100)
for policy, d in run_dirs:
    sumf = d / "summary.json"
    metaf = d / "selector_meta.json"
    try:
        s = json.loads(sumf.read_text())
        rmse = s.get("rmse_pos", float("nan"))
    except Exception:
        rmse = float("nan")
    n_switches = 0
    n_active_vals = []
    switch_times = []
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
