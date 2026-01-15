#!/usr/bin/env python3
"""
modem_switching_validation.py

Test harness to validate acoustic "switching" behavior by toggling which beacon
ranges are fed into the EKF over time.

Two modes:
  1) manual  : user-defined dropout windows per beacon name (usv1..usv4)
  2) policy  : geometry-based selector (rank/GDOP), with dwell + margin (hysteresis)

Notes
-----
- This script assumes your EKF runner exposes `run_single_trial(...)` (preferred).
  If not available, it will fall back to `run_ekf_acoustics(...)` with fixed targets
  (no true switching), and will warn you.
- Beacons are referred to as names at the CLI: usv1/usv2/usv3/usv4.
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, TYPE_CHECKING

import numpy as np
import matplotlib.pyplot as plt

# --- EKF runner import (preferred: run_single_trial) --------------------------
RUNS_SUPPORT_SWITCHING = True
try:
    from current_acoustic_EKF_patched import run_single_trial  # type: ignore
except Exception:
    RUNS_SUPPORT_SWITCHING = False
    try:
        from current_acoustic_EKF_patched import run_ekf_acoustics  # type: ignore
    except Exception as e:
        raise ImportError(
            "Could not import EKF runner. Expected current_acoustic_EKF_patched.py "
            "to provide run_single_trial(...) or run_ekf_acoustics(...)."
        ) from e

# --- Optional: use external policy if you want (kept for compatibility) -------
try:
    from adaptive_modem_manager_v2 import AdaptiveModemManagerV2  # type: ignore
except Exception:
    AdaptiveModemManagerV2 = None  # type: ignore

if TYPE_CHECKING:
    from adaptive_modem_manager_v2 import AdaptiveModemManagerV2 as AdaptiveModemManagerV2Type
else:
    AdaptiveModemManagerV2Type = Any

try:
    from adaptive_modem_manager import AdaptiveModemManager as AdaptiveModemManagerLegacy  # type: ignore
except Exception:
    AdaptiveModemManagerLegacy = None  # type: ignore

# =============================================================================
# Dropout utilities (self-contained; replaces missing modem_dropout_test.py)
# =============================================================================

def parse_dropout_windows(specs: List[str]) -> Dict[str, List[Tuple[float, float]]]:
    """
    Parse dropout specs of the form:
      usv1:10-20
      usv2:30-45
    Multiple entries per beacon allowed.
    Returns { "usv1": [(10,20), ...], ... }
    """
    out: Dict[str, List[Tuple[float, float]]] = {}
    for s in specs:
        s = s.strip()
        if not s:
            continue
        try:
            name, rng = s.split(":", 1)
            a, b = rng.split("-", 1)
            t0, t1 = float(a), float(b)
            if t1 < t0:
                t0, t1 = t1, t0
            out.setdefault(name.strip(), []).append((t0, t1))
        except Exception as e:
            raise ValueError(f"Bad dropout spec '{s}'. Expected like usv1:10-20") from e
    # sort windows
    for k in out:
        out[k] = sorted(out[k], key=lambda x: x[0])
    return out


def in_dropout(t: float, windows: List[Tuple[float, float]]) -> bool:
    """Return True if time t is within any [t0,t1] window."""
    for t0, t1 in windows:
        if t0 <= t <= t1:
            return True
    return False


def filter_available_targets(
    t: float,
    target_info: List[Tuple[str, np.ndarray]],
    dropout: Dict[str, List[Tuple[float, float]]]
) -> List[Tuple[str, np.ndarray]]:
    """Remove targets that are in dropout at time t."""
    if not dropout:
        return target_info
    kept = []
    for name, pos in target_info:
        if name in dropout and in_dropout(t, dropout[name]):
            continue
        kept.append((name, pos))
    return kept


# =============================================================================
# Geometry utilities (rank/GDOP), including safe handling for 0/1 beacon cases
# =============================================================================

def _rank_svd(H: np.ndarray, tol: float = 1e-10) -> int:
    if H.size == 0:
        return 0
    _, s, _ = np.linalg.svd(H, full_matrices=False)
    return int(np.sum(s > tol))


def _unit_rows(vecs: np.ndarray) -> np.ndarray:
    """Normalize rows of vecs; safe for zeros."""
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    norms = np.where(norms <= 1e-12, 1.0, norms)
    return vecs / norms


def _gdop_metrics(H: np.ndarray, sigma_r: float, jitter: float = 1e-9) -> Dict[str, Any]:
    """Return rank, gdop, logdet, and FIM for a Jacobian block."""
    n, dim = H.shape
    rank = _rank_svd(H)
    Rinv = np.eye(n) / (sigma_r ** 2)
    F = H.T @ Rinv @ H if n else np.zeros((dim, dim))

    sign, logdet = np.linalg.slogdet(F + jitter * np.eye(dim))
    fim_logdet = float(logdet) if sign > 0 else float("-inf")

    gdop = float("nan")
    if rank >= dim:
        try:
            Finv = np.linalg.inv(F)
            gdop = float(np.sqrt(np.trace(Finv)))
        except np.linalg.LinAlgError:
            gdop = float("nan")

    return {"rank": rank, "gdop": gdop, "fim_logdet": fim_logdet, "fim": F}


def geom_metrics(
    auv_pos: np.ndarray,
    targets: List[Tuple[str, np.ndarray]],
    sigma_r: float = 0.5,
    mode: str = "xy",
    jitter: float = 1e-9
) -> Dict[str, Any]:
    """
    Compute Jacobian rank, FIM logdet, and GDOP for range-only geometry.

    mode="xy": uses horizontal components only (2D).
    mode="3d": uses full 3D (3D).

    Returns dict with: rank, gdop, fim_logdet, size
    """
    n = len(targets)
    if n == 0:
        return {
            "rank": 0,
            "gdop": float("inf"),
            "fim_logdet": float("-inf"),
            "size": 0,
            "rank_xy": 0,
            "gdop_xy": float("nan"),
            "fim_logdet_xy": float("-inf"),
            "rank_3d": 0,
            "gdop_3d": float("nan"),
            "fim_logdet_3d": float("-inf"),
        }

    pos = np.stack([p for _, p in targets], axis=0)  # (n,3)
    rel = pos - auv_pos.reshape(1, 3)                # (n,3)
    H3 = _unit_rows(rel)                             # (n,3)
    metrics_xy = _gdop_metrics(H3[:, :2], sigma_r, jitter)
    metrics_3d = _gdop_metrics(H3, sigma_r, jitter)

    mode_metrics = metrics_xy if mode == "xy" else metrics_3d

    return {
        "rank": mode_metrics["rank"],
        "gdop": mode_metrics["gdop"],
        "fim_logdet": mode_metrics["fim_logdet"],
        "fim": mode_metrics["fim"],
        "size": n,
        "rank_xy": metrics_xy["rank"],
        "gdop_xy": metrics_xy["gdop"],
        "fim_logdet_xy": metrics_xy["fim_logdet"],
        "rank_3d": metrics_3d["rank"],
        "gdop_3d": metrics_3d["gdop"],
        "fim_logdet_3d": metrics_3d["fim_logdet"],
    }


# =============================================================================
# Policy selector (flexible: supports 0..4 beacons if desired)
# =============================================================================

@dataclass
class PolicyParams:
    sigma_r: float = 0.5
    gdop_thresh_xy: float = 15.0
    gdop_thresh_3d: float = 15.0
    switch_margin: float = 1.0
    min_dwell_sec: float = 5.0
    size_penalty: float = 0.0          # encourages fewer beacons
    allow_unobservable_fallback: bool = True
    min_beacons_xy: int = 2            # typical for XY observability
    min_beacons_3d: int = 3            # typical for 3D observability
    max_beacons: int = 4


class GeometryPolicySelector:
    """
    Selects beacon subsets using rank + GDOP with dwell/margin hysteresis.

    Signature matches what run_single_trial(...) expects:
      selector_fn(t_sec, xhat6, depth_available, target_info) -> (selected_names, meta_dict)
    """
    def __init__(self, params: PolicyParams):
        self.p = params
        self.active_names: List[str] = []
        self.last_switch_t: float = -1e9
        self.last_obj: float = float("inf")

    def _score_subset(self, auv_pos: np.ndarray, subset: List[Tuple[str, np.ndarray]], mode: str) -> Tuple[float, Dict[str, Any]]:
        m = geom_metrics(auv_pos, subset, sigma_r=self.p.sigma_r, mode=mode)
        gdop = float(m["gdop"])
        if not math.isfinite(gdop):
            gdop = float("inf")
        # objective: geometry + size cost
        obj = gdop + self.p.size_penalty * float(m["size"])
        return obj, m

    def decide(self, t: float, xhat: np.ndarray, depth_available: bool, target_info: List[Tuple[str, np.ndarray]]) -> Tuple[List[str], Dict[str, Any]]:
        auv_pos = np.asarray(xhat[:3], dtype=float)
        mode = "xy" if depth_available else "3d"
        rank_req = 2 if depth_available else 3
        gdop_thresh = self.p.gdop_thresh_xy if depth_available else self.p.gdop_thresh_3d
        min_beacons = self.p.min_beacons_xy if depth_available else self.p.min_beacons_3d
        max_beacons = min(self.p.max_beacons, len(target_info))

        # enumerate subsets
        best_feasible = None     # (obj, names, metrics)
        best_any = None          # best overall including 0-beacon
        best_nonzero = None      # best objective with >=1 beacon (no feasibility check)

        # allow turning acoustics fully off if min_beacons == 0
        sizes = list(range(min_beacons, max_beacons + 1))
        if min_beacons == 0 and 0 not in sizes:
            sizes = [0] + sizes

        # Pre-build lookup for stable ordering
        name_to_pos = {n: p for n, p in target_info}
        names_all = [n for n, _ in target_info]

        for r in sizes:
            if r == 0:
                obj = 0.0  # "free" acoustics-off objective (pure energy)
                metrics = {"rank": 0, "gdop": float("inf"), "fim_logdet": float("-inf"), "size": 0}
                cand = (obj, [], metrics)
                if best_any is None or obj < best_any[0]:
                    best_any = cand
                continue

            for combo in itertools.combinations(names_all, r):
                subset = [(n, name_to_pos[n]) for n in combo]
                obj, metrics = self._score_subset(auv_pos, subset, mode=mode)

                # keep best-any (for fallback)
                if best_any is None or obj < best_any[0]:
                    best_any = (obj, list(combo), metrics)

                # track best nonzero irrespective of feasibility
                if best_nonzero is None or obj < best_nonzero[0]:
                    best_nonzero = (obj, list(combo), metrics)

                # feasibility check
                if metrics["rank"] >= rank_req and math.isfinite(metrics["gdop"]) and metrics["gdop"] <= gdop_thresh:
                    if best_feasible is None:
                        best_feasible = (obj, list(combo), metrics)
                    else:
                        # Prefer smaller size first, then lower objective
                        if len(combo) < len(best_feasible[1]) or (len(combo) == len(best_feasible[1]) and obj < best_feasible[0]):
                            best_feasible = (obj, list(combo), metrics)

        # choose candidate: feasible if exists else best_any
        chosen = best_feasible if best_feasible is not None else (best_nonzero if best_nonzero is not None else best_any)
        if chosen is None:
            # no targets at all
            empty_meta = {
                "mode": mode,
                "reason": "no_targets",
                "rank": 0,
                "gdop": float("inf"),
                "fim_logdet": float("-inf"),
                "size": 0,
                "rank_xy": 0,
                "gdop_xy": float("nan"),
                "rank_3d": 0,
                "gdop_3d": float("nan"),
            }
            return [], empty_meta

        cand_obj, cand_names, cand_metrics = chosen

        # hysteresis: dwell time + margin
        reason = "init"
        if self.active_names:
            if (t - self.last_switch_t) < self.p.min_dwell_sec:
                # keep current
                keep_metrics = geom_metrics(auv_pos, [(n, name_to_pos[n]) for n in self.active_names if n in name_to_pos],
                                           sigma_r=self.p.sigma_r, mode=mode)
                return self.active_names, {"mode": mode, "reason": "dwell_hold", **keep_metrics}

            # compute current objective
            cur_subset = [(n, name_to_pos[n]) for n in self.active_names if n in name_to_pos]
            cur_obj, cur_metrics = self._score_subset(auv_pos, cur_subset, mode=mode) if cur_subset else (0.0, {"rank": 0, "gdop": float("inf"), "fim_logdet": float("-inf"), "size": 0})

            improved = (cand_obj + 1e-12) < (cur_obj - self.p.switch_margin)
            power_saves = (len(cand_names) < len(self.active_names)) and ((cand_obj - cur_obj) <= self.p.switch_margin)

            if improved or power_saves:
                reason = "switch"
            else:
                # keep current
                return self.active_names, {"mode": mode, "reason": "no_switch", **cur_metrics}

        # apply switch / init
        self.active_names = cand_names
        self.last_switch_t = t
        self.last_obj = cand_obj
        meta = {
            "mode": mode,
            "reason": reason,
            "rank": cand_metrics.get("rank", 0),
            "gdop": cand_metrics.get("gdop", float("nan")),
            "fim_logdet": cand_metrics.get("fim_logdet", float("-inf")),
            "size": cand_metrics.get("size", 0),
            "rank_xy": cand_metrics.get("rank_xy", 0),
            "gdop_xy": cand_metrics.get("gdop_xy", float("nan")),
            "rank_3d": cand_metrics.get("rank_3d", 0),
            "gdop_3d": cand_metrics.get("gdop_3d", float("nan")),
        }
        return cand_names, meta


# =============================================================================
# Weighted policy selector (mission-aware multi-objective scoring)
# =============================================================================


def _clamp01(x: float) -> float:
    return max(0.0, min(1.0, x))


def posterior_uncertainty(P_prior: np.ndarray, F: np.ndarray, depth_available: bool) -> float:
    """Approximate posterior position uncertainty sqrt(trace(P_post))."""
    if P_prior is None or P_prior.size == 0:
        return float("inf")
    dim = 2 if depth_available else 3
    try:
        Pp_inv = np.linalg.pinv(P_prior[0:dim, 0:dim])
        M = Pp_inv + F[0:dim, 0:dim]
        P_post = np.linalg.pinv(M)
        return float(np.sqrt(np.trace(P_post)))
    except Exception:
        return float("inf")


class EnergyModel:
    def __init__(self, base_drain_w: float, beacon_drain_w: float, battery_wh: float, drain_scale: float, soc_init: float, soc_min: float):
        self.base = float(base_drain_w)
        self.beacon = float(beacon_drain_w)
        self.battery_wh = max(1e-6, float(battery_wh))
        self.drain_scale = float(drain_scale)
        self.soc = max(0.0, min(1.0, float(soc_init)))
        self.soc_min = max(0.0, min(1.0, float(soc_min)))

    def power(self, n_beacons: int) -> float:
        return self.base + self.beacon * float(max(0, n_beacons))

    def predict_soc(self, dt: float, n_beacons: int) -> float:
        power_w = self.power(n_beacons)
        dsoc = (power_w * max(0.0, dt)) / (self.battery_wh * 3600.0)  # Wh conversion: W * s / 3600
        soc_pred = self.soc - self.drain_scale * dsoc
        return max(self.soc_min, min(1.0, soc_pred))

    def step(self, dt: float, n_beacons: int) -> float:
        self.soc = self.predict_soc(dt, n_beacons)
        return self.soc


@dataclass
class WeightedParams:
    sigma_r: float = 0.5
    min_beacons_xy: int = 2
    min_beacons_3d: int = 3
    max_beacons: int = 4
    rank_req_xy: int = 2
    rank_req_3d: int = 3
    gdop_thresh_xy: float = 15.0
    gdop_thresh_3d: float = 15.0
    min_dwell_sec: float = 5.0
    score_margin: float = 0.05
    power_save_tol: float = 0.02
    soc_init: float = 1.0
    soc_min: float = 0.0
    low_power_soc: float = 0.1
    base_drain_w: float = 0.0
    beacon_drain_w: float = 1.0
    drain_scale: float = 1.0
    battery_wh: float = 100.0
    target_unc_xy: float = 5.0
    target_unc_3d: float = 8.0
    off_unc_mult: float = 1.5
    mission_phase: str = "cruise"
    phase_schedule: Optional[List[Tuple[str, float, float]]] = None
    allow_zero: bool = False
    churn_interval_sec: float = 0.0   # if >0, allow periodic tie-rotation
    churn_score_eps: float = 0.02     # score proximity to consider for churn
    weight_scale_obs: float = 1.0
    weight_scale_energy: float = 1.0
    weight_scale_mission: float = 1.0


DEFAULT_PHASE_WEIGHTS = {
    "survey": (0.6, 0.2, 0.2),
    "cruise": (0.5, 0.3, 0.2),
    "transit": (0.3, 0.5, 0.2),
    "low_power": (0.2, 0.7, 0.1),
}


def parse_phase_schedule(spec: Optional[str]) -> Optional[List[Tuple[str, float, float]]]:
    if not spec:
        return None
    spans: List[Tuple[str, float, float]] = []
    for part in spec.split(','):
        part = part.strip()
        if not part:
            continue
        try:
            name, rng = part.split(':', 1)
            a, b = rng.split('-', 1)
            t0, t1 = float(a), float(b)
            spans.append((name.strip(), min(t0, t1), max(t0, t1)))
        except Exception as e:
            raise ValueError(f"Bad phase schedule chunk '{part}', expected phase:t0-t1") from e
    return spans if spans else None


class WeightedPolicySelector:
    def __init__(self, params: WeightedParams, phase_weights: Dict[str, Tuple[float, float, float]]):
        self.p = params
        self.phase_weights = phase_weights
        self.active_names: List[str] = []
        self.last_switch_t: float = -1e9
        self.last_score: float = float("-inf")
        self.energy = EnergyModel(params.base_drain_w, params.beacon_drain_w, params.battery_wh, params.drain_scale, params.soc_init, params.soc_min)
        self.last_t: Optional[float] = None
        self.last_churn_t: float = -1e9

    def _phase(self, t: float) -> str:
        if self.energy.soc <= self.p.low_power_soc:
            return "low_power"
        if self.p.phase_schedule:
            for name, t0, t1 in self.p.phase_schedule:
                if t0 <= t < t1:
                    return name
        return self.p.mission_phase

    def _subset_key(self, names: List[str]) -> Tuple[str, ...]:
        return tuple(sorted(names))

    def _feasible(self, metrics: Dict[str, Any], depth_available: bool) -> bool:
        rank_req = self.p.rank_req_xy if depth_available else self.p.rank_req_3d
        gdop_thresh = self.p.gdop_thresh_xy if depth_available else self.p.gdop_thresh_3d
        if metrics.get("rank", 0) < rank_req:
            return False
        gdop = float(metrics.get("gdop", float("inf")))
        if not math.isfinite(gdop):
            return False
        if gdop_thresh is not None and math.isfinite(gdop_thresh) and gdop > gdop_thresh:
            return False
        return True

    def decide(
        self,
        t: float,
        xhat: np.ndarray,
        depth_available: bool,
        target_info: List[Tuple[str, np.ndarray]],
        covariance: Optional[np.ndarray] = None,
    ) -> Tuple[List[str], Dict[str, Any]]:
        mode = "xy" if depth_available else "3d"
        if self.energy.soc <= self.p.soc_min + 1e-9:
            empty_meta = {
                "mode": mode,
                "reason": "battery_depleted",
                "rank": 0,
                "gdop": float("inf"),
                "fim_logdet": float("-inf"),
                "size": 0,
                "rank_xy": 0,
                "gdop_xy": float("nan"),
                "rank_3d": 0,
                "gdop_3d": float("nan"),
                "score": 0.0,
                "soc": self.energy.soc,
                "phase": self._phase(t),
                "f_obs": 0.0,
                "f_energy": 0.0,
                "f_mission": 0.0,
            }
            return [], empty_meta
        if not target_info:
            empty_meta = {
                "mode": mode,
                "reason": "no_targets",
                "rank": 0,
                "gdop": float("inf"),
                "fim_logdet": float("-inf"),
                "size": 0,
                "rank_xy": 0,
                "gdop_xy": float("nan"),
                "rank_3d": 0,
                "gdop_3d": float("nan"),
                "score": 0.0,
                "soc": self.energy.soc,
                "phase": self._phase(t),
                "f_obs": 0.0,
                "f_energy": 0.0,
                "f_mission": 0.0,
            }
            return [], empty_meta

        dt = 0.0 if self.last_t is None else max(0.0, float(t) - float(self.last_t))
        name_to_pos = {n: p for n, p in target_info}
        names_all = [n for n, _ in target_info]
        min_beacons = self.p.min_beacons_xy if depth_available else self.p.min_beacons_3d
        max_beacons = min(self.p.max_beacons, len(target_info))

        sizes = list(range(min_beacons, max_beacons + 1))
        if self.p.allow_zero and 0 not in sizes:
            sizes = [0] + sizes

        candidates: List[Dict[str, Any]] = []
        logdets: List[float] = []

        for r in sizes:
            if r == 0:
                metrics = {"rank": 0, "gdop": float("inf"), "fim_logdet": float("-inf"), "size": 0, "fim": np.zeros((3, 3))}
                # Zero-beacon feasibility gate based on uncertainty allowance
                target = self.p.target_unc_xy if depth_available else self.p.target_unc_3d
                allow_zero = self.p.allow_zero
                if allow_zero and target > 0:
                    unc0 = float(np.sqrt(np.trace(covariance[0:2, 0:2] if (covariance is not None and covariance.size >= 9 and depth_available) else (covariance[0:3, 0:3] if covariance is not None and covariance.size >= 9 else np.eye(3) * (target ** 2)))))
                    if unc0 <= self.p.off_unc_mult * target:
                        candidates.append({"names": [], "metrics": metrics})
                        logdets.append(metrics["fim_logdet"])
                continue

            for combo in itertools.combinations(names_all, r):
                subset = [(n, name_to_pos[n]) for n in combo]
                metrics = geom_metrics(xhat[:3], subset, sigma_r=self.p.sigma_r, mode=mode)
                candidates.append({"names": list(combo), "metrics": metrics})
                logdets.append(metrics["fim_logdet"])

        # Normalize observability term using logdet across candidates
        finite_logdets = [ld for ld in logdets if math.isfinite(ld)]
        if finite_logdets:
            logdet_min = min(finite_logdets)
            logdet_max = max(finite_logdets)
        else:
            logdet_min = logdet_max = 0.0
        denom = (logdet_max - logdet_min) + 1e-9

        phase = self._phase(t)
        w_obs, w_energy, w_mission = self.phase_weights.get(phase, self.phase_weights.get("cruise", (0.5, 0.3, 0.2)))
        w_obs *= self.p.weight_scale_obs
        w_energy *= self.p.weight_scale_energy
        w_mission *= self.p.weight_scale_mission
        w_sum = w_obs + w_energy + w_mission
        if w_sum > 0:
            w_obs /= w_sum
            w_energy /= w_sum
            w_mission /= w_sum

        def mission_term(metrics: Dict[str, Any]) -> float:
            target = self.p.target_unc_xy if depth_available else self.p.target_unc_3d
            if target <= 0:
                return 0.0
            F = metrics.get("fim")
            if covariance is not None and covariance.size >= 9 and isinstance(F, np.ndarray):
                u = posterior_uncertainty(covariance, F, depth_available)
            else:
                gdop = float(metrics.get("gdop", float("inf")))
                if not math.isfinite(gdop):
                    return 0.0
                u = gdop * self.p.sigma_r
            return _clamp01(max(0.0, 1.0 - abs(u - target) / target))

        # Precompute current subset for hysteresis comparison
        current_key = self._subset_key(self.active_names)
        current_candidate: Optional[Dict[str, Any]] = None

        best_feasible = None
        best_nonzero = None
        best_any = None
        candidate_list: List[Dict[str, Any]] = []

        for cand in candidates:
            names = cand["names"]
            metrics = cand["metrics"]
            ld = metrics.get("fim_logdet", float("-inf"))
            f_obs = 0.0
            if math.isfinite(ld) and math.isfinite(logdet_min) and math.isfinite(logdet_max):
                f_obs = (ld - logdet_min) / denom
                f_obs = _clamp01(f_obs)

            size = metrics.get("size", 0)
            power_cfg = self.energy.power(size)
            power_max = self.energy.power(self.p.max_beacons)
            energy_margin = 0.0
            if power_max > 1e-9:
                energy_margin = _clamp01((power_max - power_cfg) / power_max)
            soc_pred = self.energy.predict_soc(dt, size)
            f_energy = _clamp01(soc_pred * energy_margin)

            f_mission = mission_term(metrics)
            score = (w_obs * f_obs) + (w_energy * f_energy) + (w_mission * f_mission)

            cand.update({
                "f_obs": f_obs,
                "f_energy": f_energy,
                "f_mission": f_mission,
                "score": score,
                "phase": phase,
                "weights": (w_obs, w_energy, w_mission),
                "soc_after": soc_pred,
                "feasible": self._feasible(metrics, depth_available),
            })

            key = self._subset_key(names)
            if key == current_key:
                current_candidate = cand

            if size > 0:
                cand["feasible"] = cand["feasible"] and metrics.get("rank", 0) >= (self.p.rank_req_xy if depth_available else self.p.rank_req_3d)
                if cand["feasible"]:
                    if (best_feasible is None) or (cand["score"] > best_feasible["score"]):
                        best_feasible = cand
                if (best_nonzero is None) or (cand["score"] > best_nonzero["score"]):
                    best_nonzero = cand
            else:
                # size==0 only if allowed and gated above
                pass
            if (best_any is None) or (cand["score"] > best_any["score"]):
                best_any = cand

            candidate_list.append(cand)

        chosen = best_feasible or best_nonzero or best_any
        if chosen is None:
            return [], {"mode": mode, "reason": "no_targets", "rank": 0, "gdop": float("inf"), "fim_logdet": float("-inf"), "size": 0, "score": 0.0, "soc": self.energy.soc, "phase": phase, "f_obs": 0.0, "f_energy": 0.0, "f_mission": 0.0}

        reason = "init"
        final_cand = chosen

        # Optional churn: if enabled, periodically rotate among near-best candidates to force switching exploration
        churn_enabled = self.p.churn_interval_sec > 0.0
        if churn_enabled and (t - self.last_churn_t) >= self.p.churn_interval_sec and candidate_list:
            # Force rotation among candidates to encourage switching
            sorted_cands = sorted(candidate_list, key=lambda c: c.get("score", float("-inf")), reverse=True)
            keys = [self._subset_key(c.get("names", [])) for c in sorted_cands]
            cur_idx = keys.index(current_key) if current_key in keys else -1
            next_idx = (cur_idx + 1) % len(sorted_cands)
            cand = sorted_cands[next_idx]
            if self._subset_key(cand.get("names", [])) != current_key:
                final_cand = cand
                reason = "churn_force"
                self.last_churn_t = t

        # Hysteresis
        if self.active_names:
            if (t - self.last_switch_t) < self.p.min_dwell_sec and current_candidate is not None:
                final_cand = current_candidate
                reason = "dwell_hold"
            else:
                cur_score = current_candidate["score"] if current_candidate is not None else float("-inf")
                new_score = chosen["score"]
                improved = new_score >= (cur_score + self.p.score_margin)
                power_saves = (len(chosen["names"]) < len(self.active_names)) and (new_score >= cur_score - self.p.power_save_tol)
                if improved or power_saves or not current_candidate:
                    reason = "switch"
                    final_cand = chosen
                else:
                    reason = "no_switch"
                    if current_candidate is not None:
                        final_cand = current_candidate

        self.active_names = list(final_cand["names"])
        self.last_switch_t = t
        self.last_score = final_cand["score"]
        # Commit SOC step using chosen set
        self.energy.step(dt, len(self.active_names))
        self.last_t = t

        metrics = final_cand.get("metrics", {})
        meta = {
            "mode": mode,
            "reason": reason,
            "rank": metrics.get("rank", 0),
            "gdop": metrics.get("gdop", float("inf")),
            "fim_logdet": metrics.get("fim_logdet", float("-inf")),
            "size": metrics.get("size", 0),
            "rank_xy": metrics.get("rank_xy", 0),
            "gdop_xy": metrics.get("gdop_xy", float("nan")),
            "rank_3d": metrics.get("rank_3d", 0),
            "gdop_3d": metrics.get("gdop_3d", float("nan")),
            "score": final_cand.get("score", 0.0),
            "f_obs": final_cand.get("f_obs", 0.0),
            "f_energy": final_cand.get("f_energy", 0.0),
            "f_mission": final_cand.get("f_mission", 0.0),
            "soc": self.energy.soc,
            "phase": final_cand.get("phase", phase),
            "w_obs": w_obs,
            "w_energy": w_energy,
            "w_mission": w_mission,
        }
        return self.active_names, meta


# =============================================================================
# Plot helpers (kept light; we rely on runner for core outputs)
# =============================================================================

saved_figs: List[Path] = []
SHOW_PLOTS = False


def save_fig(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=300)
    saved_figs.append(path)
    if not SHOW_PLOTS:
        plt.close()



# =============================================================================
# Main
# =============================================================================

def build_argparser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Validate modem switching by dropout or policy selection.")
    ap.add_argument("--outdir", type=str, default="results_modem_switching", help="Output directory root.")
    ap.add_argument("--duration", type=float, default=180.0, help="Simulation duration (s).")
    ap.add_argument("--seed", type=int, default=3, help="Random seed.")
    ap.add_argument("--traj", type=str, default="spiral", choices=["spiral", "figure8", "lawnmower", "concentric"], help="Trajectory type.")
    ap.add_argument("--currents", action="store_true", help="Enable currents (if runner supports).")
    ap.add_argument("--make-plots", action="store_true", help="Request EKF runner to emit full diagnostic plots.")
    ap.add_argument("--show-plots", action="store_true", help="Display plots after saving (matplotlib blocking show).")

    # Mode
    ap.add_argument("--mode", type=str, default="policy", choices=["policy", "manual"], help="Selector mode.")
    ap.add_argument("--policy-type", type=str, default="gdop", choices=["gdop", "weighted", "v2"], help="Policy flavor when mode=policy.")

    # Targets
    ap.add_argument("--targets", nargs="*", default=["usv1", "usv2", "usv3", "usv4"], help="Beacon names to consider (usv1..usv4).")

    # Dropouts (manual or policy-aware)
    ap.add_argument("--dropout", nargs="*", default=[], help="Dropout windows, e.g. usv2:30-45 usv3:80-120")

    # Policy parameters (shared)
    ap.add_argument("--sigma-r", type=float, default=0.5, help="Range noise std for geometry scoring (m).")
    ap.add_argument("--gdop-xy", type=float, default=15.0, help="GDOP threshold in XY mode (depth available).")
    ap.add_argument("--gdop-3d", type=float, default=15.0, help="GDOP threshold in 3D mode (no depth).")
    ap.add_argument("--switch-margin", type=float, default=1.0, help="Switch only if objective improves by this margin (GDOP policy).")
    ap.add_argument("--min-dwell-sec", type=float, default=5.0, help="Minimum seconds between switches.")
    ap.add_argument("--size-penalty", type=float, default=0.0, help="Add penalty per active beacon to favor smaller sets (GDOP policy).")

    # Weighted policy parameters
    ap.add_argument("--score-margin", type=float, default=0.05, help="Score margin required to switch (weighted policy).")
    ap.add_argument("--power-save-tol", type=float, default=0.02, help="Allow switching to fewer beacons if score no worse than tol (weighted policy).")
    ap.add_argument("--rank-req-xy", type=int, default=2, help="Rank requirement in XY mode for feasibility (weighted policy).")
    ap.add_argument("--rank-req-3d", type=int, default=3, help="Rank requirement in 3D mode for feasibility (weighted policy).")

    # Energy model
    ap.add_argument("--soc-init", type=float, default=1.0, help="Initial state of charge (0-1).")
    ap.add_argument("--soc-min", type=float, default=0.0, help="Minimum SOC clamp (0-1).")
    ap.add_argument("--battery-wh", type=float, default=100.0, help="Battery capacity in Wh (used for SOC drain).")
    ap.add_argument("--base-drain-w", type=float, default=0.0, help="Base power drain (W).")
    ap.add_argument("--beacon-drain-w", type=float, default=1.0, help="Additional drain per active beacon (W).")
    ap.add_argument("--drain-scale", type=float, default=1.0, help="Scale factor mapping drain to SOC decrement per second.")
    ap.add_argument("--low-power-soc", type=float, default=0.1, help="SOC threshold to enter low_power phase.")

    # Mission term / uncertainty targets
    ap.add_argument("--target-unc-xy", type=float, default=5.0, help="Target sqrt(trace(P_xy)) (m) for mission scoring when depth available.")
    ap.add_argument("--target-unc-3d", type=float, default=8.0, help="Target sqrt(trace(P_xyz)) (m) for mission scoring when depth unavailable.")
    ap.add_argument("--off-unc-mult", type=float, default=1.5, help="Allow zero-beacon only if unc <= off_unc_mult * target_unc.")
    ap.add_argument("--target-unc", type=float, default=None, help="(compat) Set both target-unc-xy and target-unc-3d to this value.")

    # Mission phase control
    ap.add_argument("--mission-phase", type=str, default="cruise", choices=["survey", "cruise", "transit", "low_power"], help="Fixed mission phase (if no schedule).")
    ap.add_argument("--phase-schedule", type=str, default=None, help="Optional time schedule: phase:t0-t1 comma-separated, e.g. 'survey:0-60,cruise:60-140,transit:140-9999'.")
    ap.add_argument("--churn-interval-sec", type=float, default=0.0, help="If >0, periodically rotate among near-best subsets to force switching (seconds).")
    ap.add_argument("--churn-score-eps", type=float, default=0.02, help="Score proximity for churn candidate selection.")
    ap.add_argument("--energy-weight", type=float, default=None, help="(compat) Scale the energy weight in the weighted policy (values >1 emphasize energy).")
    ap.add_argument("--v2-low-power-soc", type=float, default=0.2, help="(v2) SOC threshold for low-power behavior.")
    ap.add_argument("--v2-energy-mult", type=float, default=2.0, help="(v2) Energy weight multiplier when SOC is low.")
    ap.add_argument("--v2-size-penalty-mult", type=float, default=2.0, help="(v2) Size penalty multiplier when SOC is low.")
    ap.add_argument("--v2-rank-deficit-mult", type=float, default=1.0, help="(v2) Rank deficit penalty multiplier when SOC is low (use <1 to allow 1-beacon).")
    ap.add_argument("--v2-rank-deficit-penalty", type=float, default=10.0, help="(v2) Base rank deficit penalty (lower to allow 1-beacon).")

    # Allow 0..4 / 1-beacon behavior
    ap.add_argument("--min-beacons-xy", type=int, default=2, help="Minimum beacons when depth is available (XY geometry).")
    ap.add_argument("--min-beacons-3d", type=int, default=3, help="Minimum beacons when depth not available (3D geometry).")
    ap.add_argument("--allow-one-beacon", action="store_true", help="Shortcut: set min-beacons-xy=1.")
    ap.add_argument("--allow-zero-beacons", action="store_true", help="Shortcut: set min-beacons-xy=0 (permits acoustics off).")

    return ap


def main():
    args = build_argparser().parse_args()

    global SHOW_PLOTS
    SHOW_PLOTS = bool(args.show_plots)

    # Reset saved_figs per invocation in case the module is reused.
    saved_figs.clear()

    if args.allow_one_beacon:
        args.min_beacons_xy = min(args.min_beacons_xy, 1)
    if args.allow_zero_beacons:
        args.min_beacons_xy = 0
    if args.target_unc is not None:
        args.target_unc_xy = float(args.target_unc)
        args.target_unc_3d = float(args.target_unc)

    out_root = Path(args.outdir)
    out_root.mkdir(parents=True, exist_ok=True)

    dropout = parse_dropout_windows(args.dropout) if args.dropout else {}

    # The EKF runner expects target_names and (optionally) selector_fn + dropout.
    # We create selector_fn with the signature used in your patched runner:
    #   selector_fn(t, xhat, depth_available, target_info) -> selected_names
    #
    # target_info is a list of (name, position) tuples, name-based (usv1..usv4).

    selector_meta_log: List[Dict[str, Any]] = []
    phase_schedule = parse_phase_schedule(args.phase_schedule)

    def _jsonable_meta(val: Any):
        if isinstance(val, np.ndarray):
            return val.tolist()
        if isinstance(val, np.generic):
            return val.item()
        if isinstance(val, dict):
            return {k: _jsonable_meta(v) for k, v in val.items()}
        if isinstance(val, (list, tuple)):
            return [_jsonable_meta(v) for v in val]
        return val

    def make_decision_payload(chosen: List[str], meta: Dict[str, Any]) -> Dict[str, Any]:
        """Adapt selector output to the dict shape expected by run_ekf_acoustics."""
        mode_str = str(meta.get("mode", "xy"))
        reason = meta.get("reason", "")
        mode_field = f"{mode_str}:{reason}" if reason else mode_str

        def _maybe_float(val: Any) -> float:
            try:
                return float(val)
            except Exception:
                return float("nan")

        rank_xy = _maybe_float(meta.get("rank_xy", meta.get("rank")))
        gdop_xy = _maybe_float(meta.get("gdop_xy", meta.get("gdop")))
        rank_3d = _maybe_float(meta.get("rank_3d", meta.get("rank")))
        gdop_3d = _maybe_float(meta.get("gdop_3d", meta.get("gdop")))

        return {
            "active_names": list(chosen),
            "mode": mode_field,
            "rank_xy": rank_xy,
            "gdop_xy": gdop_xy,
            "rank_3d": rank_3d,
            "gdop_3d": gdop_3d,
            "n_active": len(chosen),
            "active_set": list(chosen),
            "soc": meta.get("soc", None),
        }

    if args.mode == "manual":
        def selector_fn(t: float, xhat: np.ndarray, depth_available: bool, target_info: List[Tuple[str, np.ndarray]], covariance: Optional[np.ndarray] = None):
            avail = filter_available_targets(t, target_info, dropout)
            selected = [n for n, _ in avail if n in set(args.targets)]
            selector_meta_log.append({"t": t, "mode": "manual", "selected": selected, "n": len(selected), "n_active": len(selected)})
            metrics = geom_metrics(np.asarray(xhat[:3], dtype=float), [(n, p) for n, p in avail if n in selected], sigma_r=args.sigma_r, mode="xy" if depth_available else "3d")
            meta = {
                "mode": "xy" if depth_available else "3d",
                "reason": "manual",
                "rank": metrics.get("rank", 0),
                "gdop": metrics.get("gdop", float("nan")),
                "rank_xy": metrics.get("rank_xy", 0),
                "gdop_xy": metrics.get("gdop_xy", float("nan")),
                "rank_3d": metrics.get("rank_3d", 0),
                "gdop_3d": metrics.get("gdop_3d", float("nan")),
            }
            return make_decision_payload(selected, meta)
    else:
        # Policy mode
        if args.policy_type == "weighted":
            weight_scale_energy = 1.0 if args.energy_weight is None else float(args.energy_weight)
            w_params = WeightedParams(
                sigma_r=args.sigma_r,
                min_beacons_xy=args.min_beacons_xy,
                min_beacons_3d=args.min_beacons_3d,
                max_beacons=4,
                rank_req_xy=args.rank_req_xy,
                rank_req_3d=args.rank_req_3d,
                gdop_thresh_xy=args.gdop_xy,
                gdop_thresh_3d=args.gdop_3d,
                min_dwell_sec=args.min_dwell_sec,
                score_margin=args.score_margin,
                power_save_tol=args.power_save_tol,
                soc_init=args.soc_init,
                soc_min=args.soc_min,
                low_power_soc=args.low_power_soc,
                base_drain_w=args.base_drain_w,
                beacon_drain_w=args.beacon_drain_w,
                drain_scale=args.drain_scale,
                battery_wh=args.battery_wh,
                target_unc_xy=args.target_unc_xy,
                target_unc_3d=args.target_unc_3d,
                off_unc_mult=args.off_unc_mult,
                mission_phase=args.mission_phase,
                phase_schedule=phase_schedule,
                allow_zero=bool(args.allow_zero_beacons),
                churn_interval_sec=float(args.churn_interval_sec),
                churn_score_eps=float(args.churn_score_eps),
                weight_scale_energy=weight_scale_energy,
            )
            sel = WeightedPolicySelector(w_params, DEFAULT_PHASE_WEIGHTS)

            def selector_fn(t: float, xhat: np.ndarray, depth_available: bool, target_info: List[Tuple[str, np.ndarray]], covariance: Optional[np.ndarray] = None):
                avail = filter_available_targets(t, target_info, dropout)
                avail = [(n, p) for n, p in avail if n in set(args.targets)]
                chosen, meta = sel.decide(t, xhat, depth_available, avail, covariance=covariance)
                selector_meta_log.append({"t": t, "mode": "policy_weighted", "selected": chosen, "n_active": len(chosen), **meta})
                return make_decision_payload(chosen, meta)
        else:
            if args.policy_type == "v2":
                if AdaptiveModemManagerV2 is None:
                    raise ImportError("adaptive_modem_manager_v2.AdaptiveModemManagerV2 is not available")

                mgr_v2: Optional[AdaptiveModemManagerV2Type] = None
                ticks_per_sec = 100.0
                dwell_steps = max(1, int(round(args.min_dwell_sec * ticks_per_sec))) if args.min_dwell_sec > 0 else 1

                def selector_fn(t: float, xhat: np.ndarray, depth_available: bool, target_info: List[Tuple[str, np.ndarray]], covariance: Optional[np.ndarray] = None):
                    nonlocal mgr_v2
                    import numpy as np  # ensure local scope access
                    avail = filter_available_targets(t, target_info, dropout)
                    avail = [(n, p) for n, p in avail if n in set(args.targets)]

                    if mgr_v2 is None:
                        beacon_map = {n: p for n, p in target_info if n in set(args.targets)}
                        mgr_v2 = AdaptiveModemManagerV2(
                            beacon_positions=beacon_map,
                            sigma_r=args.sigma_r,
                            size_penalty=args.size_penalty,
                            rank_deficit_penalty=float(args.v2_rank_deficit_penalty),
                            min_dwell_steps=dwell_steps,
                            switch_margin=args.switch_margin,
                            target_unc_xy=args.target_unc_xy,
                            target_unc_3d=args.target_unc_3d,
                            off_unc_mult=args.off_unc_mult,
                            gate_min_dwell_steps=dwell_steps,
                            min_subset_size=0,
                            max_subset_size=4,
                            prefer_smaller=True,
                            battery_wh=args.battery_wh,
                            base_drain_w=args.base_drain_w,
                            beacon_drain_w=args.beacon_drain_w,
                            drain_scale=args.drain_scale,
                            soc_init=args.soc_init,
                            soc_min=args.soc_min,
                            energy_weight=float(args.energy_weight) if args.energy_weight is not None else 0.0,
                            low_power_soc=float(args.v2_low_power_soc),
                            energy_weight_low_power_mult=float(args.v2_energy_mult),
                            size_penalty_low_power_mult=float(args.v2_size_penalty_mult),
                            rank_deficit_penalty_low_power_mult=float(args.v2_rank_deficit_mult),
                        )

                    P_use = covariance
                    if P_use is None or P_use.size < 9:
                        target_unc = args.target_unc_xy if depth_available else args.target_unc_3d
                        P_use = np.eye(3) * (target_unc ** 2)

                    chosen, metrics = mgr_v2.select(
                        t=t,
                        a_pos=np.asarray(xhat[:3], dtype=float),
                        P=np.asarray(P_use, dtype=float),
                        available_ids=[n for n, _ in avail],
                        depth_available=depth_available,
                    )

                    md = metrics.__dict__ if hasattr(metrics, "__dict__") else dict(metrics)
                    meta = {
                        "mode": "xy" if depth_available else "3d",
                        "reason": md.get("reason", ""),
                        "rank_xy": md.get("rank_xy", 0),
                        "gdop_xy": md.get("gdop_xy", float("nan")),
                        "rank_3d": md.get("rank_3d", 0),
                        "gdop_3d": md.get("gdop_3d", float("nan")),
                        "score": md.get("score", 0.0),
                        "soc": md.get("soc", None),
                    }
                    selector_meta_log.append({"t": t, "mode": "policy_v2", "selected": chosen, "n_active": len(chosen), **meta})
                    return make_decision_payload([str(n) for n in chosen], meta)

            else:
                # If the user did not ask for 0/1-beacon behavior, and the legacy manager exists,
                # we can use it. Otherwise, use our flexible selector (supports 0..4).
                use_flex = args.min_beacons_xy < 2 or args.min_beacons_3d < 3

                if AdaptiveModemManagerLegacy is not None and not use_flex:
                    try:
                        mgr = AdaptiveModemManagerLegacy(
                            sigma_r=args.sigma_r,
                            gdop_xy_thresh=args.gdop_xy,
                            gdop_3d_thresh=args.gdop_3d,
                            min_dwell_steps=1,
                            switch_margin=args.switch_margin,
                            size_penalty=args.size_penalty,
                        )

                        def selector_fn(t: float, xhat: np.ndarray, depth_available: bool, target_info: List[Tuple[str, np.ndarray]], covariance: Optional[np.ndarray] = None):
                            avail = filter_available_targets(t, target_info, dropout)
                            chosen, meta = mgr.decide(np.asarray(xhat[:3], dtype=float), depth_available=depth_available)
                            selector_meta_log.append({"t": t, "mode": "policy_adm", "selected": chosen, "n_active": len(chosen), **meta})
                            return make_decision_payload(chosen, meta)
                    except TypeError:
                        use_flex = True

                if AdaptiveModemManagerLegacy is None or use_flex:
                    params = PolicyParams(
                        sigma_r=args.sigma_r,
                        gdop_thresh_xy=args.gdop_xy,
                        gdop_thresh_3d=args.gdop_3d,
                        switch_margin=args.switch_margin,
                        min_dwell_sec=args.min_dwell_sec,
                        size_penalty=args.size_penalty,
                        min_beacons_xy=args.min_beacons_xy,
                        min_beacons_3d=args.min_beacons_3d,
                        max_beacons=4,
                    )
                    sel = GeometryPolicySelector(params)

                    def selector_fn(t: float, xhat: np.ndarray, depth_available: bool, target_info: List[Tuple[str, np.ndarray]], covariance: Optional[np.ndarray] = None):
                        avail = filter_available_targets(t, target_info, dropout)
                        avail = [(n, p) for n, p in avail if n in set(args.targets)]
                        chosen, meta = sel.decide(t, xhat, depth_available, avail)
                        selector_meta_log.append({"t": t, "mode": "policy_flex", "selected": chosen, "n_active": len(chosen), **meta})
                        return make_decision_payload(chosen, meta)

    # Wrapper to match run_ekf_acoustics selector signature and print switching events
    last_active_state = {"names": None}

    def selector_fn_wrapped(t_current: float, ekf_state: np.ndarray, depth_available: bool, target_info: List[Tuple[str, np.ndarray]], covariance: Optional[np.ndarray] = None):
        payload = selector_fn(t_current, ekf_state, depth_available, target_info, covariance=covariance)
        new_active = tuple(sorted(payload.get("active_names", payload.get("active_set", []))))
        prev_active = last_active_state["names"]

        if prev_active is None:
            print(f"[selector] t={float(t_current):.2f}s active={list(new_active)} (initial)")
        elif new_active != prev_active:
            print(f"[selector] t={float(t_current):.2f}s active {list(prev_active)} -> {list(new_active)}")

        last_active_state["names"] = new_active
        return payload

    # ---------------- Run trial ----------------
    trial_cfg = dict(
        seed=int(args.seed),
        config_overrides={
            "duration_sec": float(args.duration),
            "trajectory": str(args.traj),
            "use_currents": bool(args.currents),
            "modem_selector_fn": selector_fn_wrapped,
        },
        return_timeseries=True,
        target_names=list(args.targets),
        make_plots=bool(args.make_plots or args.show_plots),
    )

    # output directory for this run
    tag = f"{args.mode}_{args.traj}_seed{args.seed}_T{int(args.duration)}"
    out_dir = out_root / tag
    out_dir.mkdir(parents=True, exist_ok=True)

    # Save config
    def _jsonable(val):
        if callable(val):
            return getattr(val, "__name__", "callable")
        if isinstance(val, dict):
            return {kk: _jsonable(vv) for kk, vv in val.items()}
        if isinstance(val, (list, tuple)):
            return [ _jsonable(vv) for vv in val ]
        return val

    config_payload = {k: _jsonable(v) for k, v in trial_cfg.items()}
    config_payload["cli_args"] = _jsonable(vars(args))
    with open(out_dir / "config.json", "w") as f:
        json.dump(config_payload, f, indent=2)

    if not RUNS_SUPPORT_SWITCHING:
        print("[WARNING] EKF runner does not expose run_single_trial(...).")
        print("          Falling back to run_ekf_acoustics(...): this will NOT truly switch mid-run.")
        print("          Update current_acoustic_EKF_patched.py to include run_single_trial with selector_fn support.")

        # Best-effort run with fixed targets
        run_ekf_acoustics(target_names=trial_cfg["target_names"], verbose=False)  # type: ignore
        return

    # Preferred runner path
    results = run_single_trial(**trial_cfg)  # type: ignore

    # ---------------- Save selector meta log ----------------
    with open(out_dir / "selector_meta.json", "w") as f:
        json.dump(_jsonable_meta(selector_meta_log), f, indent=2)

    # ---------------- Plot quick overlays (if runner returned timeseries) ----------------
    # Expect runner to return dict containing `timeseries` as a pandas DF or dict-like; handle both.
    ts = results.get("timeseries", None)
    if ts is not None:
        try:
            import pandas as pd  # type: ignore
            if isinstance(ts, pd.DataFrame):
                df = ts.copy()
            else:
                # Only include 1D series to avoid object-array issues
                def _is_1d_series(v: Any) -> bool:
                    try:
                        return np.asarray(v).ndim == 1
                    except Exception:
                        return False

                flat_ts = {k: v for k, v in ts.items() if _is_1d_series(v)} if isinstance(ts, dict) else ts
                df = pd.DataFrame(flat_ts)

            meta_df = pd.DataFrame(selector_meta_log)
            if not meta_df.empty and "t" in meta_df.columns:
                # Normalize selected -> active_set_str for plotting
                if "selected" in meta_df.columns:
                    meta_df["active_set_str"] = meta_df["selected"].apply(
                        lambda v: ",".join(v) if isinstance(v, (list, tuple)) else ("" if v is None else str(v))
                    )
                df = pd.merge_asof(df.sort_values("t"), meta_df.sort_values("t"), on="t", direction="nearest")

            # Ensure required columns exist for downstream plots/exports.
            for col in ["rank_xy", "gdop_xy", "rank_3d", "gdop_3d", "n_active", "active_set"]:
                if col not in df.columns:
                    df[col] = np.nan

            if "active_set" in df.columns and df["active_set"].isna().all() and "active_names" in df.columns:
                df["active_set"] = df["active_names"]
            if "n_active" in df.columns and df["n_active"].isna().all():
                if "active_names" in df.columns:
                    df["n_active"] = df["active_names"].apply(lambda x: len(x) if hasattr(x, "__len__") else np.nan)
                elif "active_count" in df.columns:
                    df["n_active"] = df["active_count"]

            df.to_csv(out_dir / "timeseries.csv", index=False)

            # plots
            if "t" in df.columns and "active_count" in df.columns:
                plt.figure(figsize=(8, 3))
                plt.plot(df["t"], df["active_count"])
                plt.xlabel("time (s)")
                plt.ylabel("# active beacons")
                plt.grid(True, alpha=0.3)

                # Annotate switches with active set + key config
                if not meta_df.empty and "active_set_str" in meta_df.columns:
                    meta_df_sorted = meta_df.sort_values("t")
                    prev = None
                    for _, row in meta_df_sorted.iterrows():
                        cur = row.get("active_set_str", "")
                        if prev is None or cur != prev:
                            reason = row.get("reason", "")
                            phase = row.get("phase", "")
                            w_obs = row.get("w_obs", None)
                            w_energy = row.get("w_energy", None)
                            w_mission = row.get("w_mission", None)
                            soc = row.get("soc", None)
                            parts = [f"set=[{cur}]" if cur else "set=[]"]
                            if reason:
                                parts.append(f"reason={reason}")
                            if phase:
                                parts.append(f"phase={phase}")
                            if w_obs is not None and w_energy is not None and w_mission is not None:
                                parts.append(f"w=({w_obs:.2f},{w_energy:.2f},{w_mission:.2f})")
                            if soc is not None and isinstance(soc, (int, float, np.floating)):
                                parts.append(f"soc={float(soc):.2f}")
                            label = " | ".join(parts)
                            plt.axvline(float(row["t"]), color="gray", linestyle="--", alpha=0.3)
                            plt.text(float(row["t"]), df["active_count"].max() + 0.05, label, rotation=90, va="bottom", fontsize=7)
                            prev = cur
                save_fig(out_dir / "fig_active_count.png")

            if "t" in df.columns and "active_set_str" in df.columns:
                plt.figure(figsize=(8, 3))
                y = np.arange(len(df)) * 0.0 + 1.0
                plt.plot(df["t"], y, alpha=0.0)
                for i, (tval, aset) in enumerate(zip(df["t"], df["active_set_str"])):
                    if i == 0 or aset != df["active_set_str"].iloc[i - 1]:
                        plt.text(float(tval), 1.0, aset, rotation=90, va="bottom", fontsize=7)
                plt.yticks([])
                plt.xlabel("time (s)")
                plt.title("Active beacon set over time")
                plt.grid(True, axis="x", alpha=0.3)
                save_fig(out_dir / "fig_active_set.png")

            if "t" in df.columns and "pos_err" in df.columns:
                plt.figure(figsize=(8, 3))
                plt.plot(df["t"], df["pos_err"])
                plt.xlabel("time (s)")
                plt.ylabel("position error (m)")
                plt.grid(True, alpha=0.3)
                save_fig(out_dir / "fig_pos_err.png")

            if "t" in df.columns:
                gdop_xy_series = df["gdop_xy"] if "gdop_xy" in df.columns else None
                gdop_3d_series = df["gdop_3d"] if "gdop_3d" in df.columns else None
                has_xy = gdop_xy_series is not None and np.isfinite(gdop_xy_series.to_numpy()).any()
                has_3d = gdop_3d_series is not None and np.isfinite(gdop_3d_series.to_numpy()).any()
                if has_xy or has_3d:
                    plt.figure(figsize=(8, 3))
                    if has_xy:
                        plt.plot(df["t"], gdop_xy_series, label="GDOP (xy)", color="C0")
                    if has_3d:
                        plt.plot(df["t"], gdop_3d_series, label="GDOP (3d)", color="C1")
                    # Policy may select based on XY when depth is available; gdop_3d is still logged for interpretability.
                    plt.xlabel("time (s)")
                    plt.ylabel("GDOP")
                    plt.legend()
                    plt.grid(True, alpha=0.3)
                    save_fig(out_dir / "fig_gdop.png")

            if "t" in df.columns and "score" in df.columns:
                score_vals = df["score"]
                if np.isfinite(score_vals.to_numpy()).any():
                    plt.figure(figsize=(8, 3))
                    plt.plot(df["t"], df["score"], label="score")
                    for comp in ["f_obs", "f_energy", "f_mission"]:
                        if comp in df.columns and np.isfinite(df[comp].to_numpy()).any():
                            plt.plot(df["t"], df[comp], label=comp)
                    plt.xlabel("time (s)")
                    plt.ylabel("score")
                    plt.legend()
                    plt.grid(True, alpha=0.3)
                    save_fig(out_dir / "fig_score_vs_time.png")

            if "t" in df.columns and "soc" in df.columns:
                soc_vals = df["soc"]
                if np.isfinite(soc_vals.to_numpy()).any():
                    plt.figure(figsize=(8, 3))
                    plt.plot(df["t"], df["soc"], label="soc")
                    plt.xlabel("time (s)")
                    plt.ylabel("SOC")
                    plt.grid(True, alpha=0.3)
                    save_fig(out_dir / "fig_soc_vs_time.png")

        except Exception:
            # Fallback without pandas
            try:
                import csv

                def _to_arr(key):
                    val = ts.get(key, []) if isinstance(ts, dict) else []
                    return np.asarray(val)

                t = _to_arr("t")
                active_count = _to_arr("active_count")
                active_set = _to_arr("active_set")
                selector_mode = _to_arr("mode")
                gdop_xy = _to_arr("gdop_xy")
                gdop_3d = _to_arr("gdop_3d")
                rank_xy_ts = _to_arr("rank_xy")
                rank_3d_ts = _to_arr("rank_3d")
                pos_err = _to_arr("err_norm") if "err_norm" in ts else _to_arr("pos_err")

                meta = selector_meta_log if selector_meta_log else []
                meta_times = np.asarray([m.get("t", "") for m in meta]) if meta else np.asarray([])
                active_set_str = np.asarray([
                    ",".join(m.get("selected", [])) if isinstance(m.get("selected", []), (list, tuple)) else ("" if m.get("selected", None) is None else str(m.get("selected", None)))
                    for m in meta
                ]) if meta else np.asarray([])
                soc_arr = np.asarray([m.get("soc", "") for m in meta]) if meta else np.asarray([])
                phase_arr = np.asarray([m.get("phase", "") for m in meta]) if meta else np.asarray([])
                score_arr = np.asarray([m.get("score", "") for m in meta]) if meta else np.asarray([])
                f_obs_arr = np.asarray([m.get("f_obs", "") for m in meta]) if meta else np.asarray([])
                f_energy_arr = np.asarray([m.get("f_energy", "") for m in meta]) if meta else np.asarray([])
                f_mission_arr = np.asarray([m.get("f_mission", "") for m in meta]) if meta else np.asarray([])
                rank_xy_meta = np.asarray([m.get("rank_xy", m.get("rank", "")) for m in meta]) if meta else np.asarray([])
                rank_3d_meta = np.asarray([m.get("rank_3d", m.get("rank", "")) for m in meta]) if meta else np.asarray([])
                gdop_xy_meta = np.asarray([m.get("gdop_xy", m.get("gdop", "")) for m in meta]) if meta else np.asarray([])
                gdop_3d_meta = np.asarray([m.get("gdop_3d", m.get("gdop", "")) for m in meta]) if meta else np.asarray([])
                n_active_meta = np.asarray([len(m.get("selected", [])) if isinstance(m.get("selected", []), (list, tuple)) else m.get("n_active", "") for m in meta]) if meta else np.asarray([])
                active_set_meta = np.asarray([m.get("selected", "") for m in meta]) if meta else np.asarray([])

                def _safe_float(val):
                    try:
                        if val is None:
                            return ""
                        if isinstance(val, str) and val.strip() == "":
                            return ""
                        return float(val)
                    except Exception:
                        return ""

                fieldnames = ["t", "active_count", "active_set", "selector_mode", "gdop_xy", "gdop_3d", "rank_xy", "rank_3d", "n_active", "pos_err", "soc", "mission_phase", "score", "f_obs", "f_energy", "f_mission"]
                with open(out_dir / "timeseries.csv", "w", newline="") as f:
                    w = csv.writer(f)
                    w.writerow(fieldnames)
                    n = len(t)
                    for i in range(n):
                        w.writerow([
                            _safe_float(t[i]) if t.size else "",
                            _safe_float(active_count[i]) if active_count.size else "",
                            str(active_set[i]) if active_set.size else (str(active_set_meta[i]) if active_set_meta.size and i < len(active_set_meta) else ""),
                            str(selector_mode[i]) if selector_mode.size else "",
                            _safe_float(gdop_xy[i]) if gdop_xy.size else (_safe_float(gdop_xy_meta[i]) if gdop_xy_meta.size and i < len(gdop_xy_meta) else ""),
                            _safe_float(gdop_3d[i]) if gdop_3d.size else (_safe_float(gdop_3d_meta[i]) if gdop_3d_meta.size and i < len(gdop_3d_meta) else ""),
                            _safe_float(rank_xy_ts[i]) if rank_xy_ts.size else (_safe_float(rank_xy_meta[i]) if rank_xy_meta.size and i < len(rank_xy_meta) else ""),
                            _safe_float(rank_3d_ts[i]) if rank_3d_ts.size else (_safe_float(rank_3d_meta[i]) if rank_3d_meta.size and i < len(rank_3d_meta) else ""),
                            _safe_float(n_active_meta[i]) if n_active_meta.size and i < len(n_active_meta) else (len(active_set[i]) if active_set.size and hasattr(active_set[i], "__len__") else ""),
                            _safe_float(pos_err[i]) if pos_err.size else "",
                            _safe_float(soc_arr[i]) if soc_arr.size and i < len(soc_arr) else "",
                            str(phase_arr[i]) if phase_arr.size and i < len(phase_arr) else "",
                            _safe_float(score_arr[i]) if score_arr.size and i < len(score_arr) else "",
                            _safe_float(f_obs_arr[i]) if f_obs_arr.size and i < len(f_obs_arr) else "",
                            _safe_float(f_energy_arr[i]) if f_energy_arr.size and i < len(f_energy_arr) else "",
                            _safe_float(f_mission_arr[i]) if f_mission_arr.size and i < len(f_mission_arr) else "",
                        ])

                if t.size and active_count.size:
                    plt.figure(figsize=(8, 3))
                    plt.plot(t, active_count)
                    plt.xlabel("time (s)")
                    plt.ylabel("# active beacons")
                    plt.grid(True, alpha=0.3)
                    if meta and active_set_str.size and meta_times.size:
                        prev = None
                        for tt, aset in zip(meta_times, active_set_str):
                            if prev is None or aset != prev:
                                plt.axvline(float(tt), color="gray", linestyle="--", alpha=0.3)
                                plt.text(float(tt), float(np.nanmax(active_count)) + 0.05, f"set=[{aset}]", rotation=90, va="bottom", fontsize=7)
                                prev = aset
                    save_fig(out_dir / "fig_active_count.png")

                if meta and active_set_str.size and meta_times.size:
                    plt.figure(figsize=(8, 3))
                    plt.plot(t, np.zeros_like(t), alpha=0.0)
                    prev = None
                    for tt, aset in zip(meta_times, active_set_str):
                        if prev is None or aset != prev:
                            plt.text(float(tt), 0.0, aset, rotation=90, va="bottom", fontsize=7)
                            prev = aset
                    plt.yticks([])
                    plt.xlabel("time (s)")
                    plt.title("Active beacon set over time")
                    plt.grid(True, axis="x", alpha=0.3)
                    save_fig(out_dir / "fig_active_set.png")

                if t.size and pos_err.size:
                    plt.figure(figsize=(8, 3))
                    plt.plot(t, pos_err)
                    plt.xlabel("time (s)")
                    plt.ylabel("position error (m)")
                    plt.grid(True, alpha=0.3)
                    save_fig(out_dir / "fig_pos_err.png")

                def _finite_any(arr):
                    try:
                        return np.isfinite(np.asarray(arr, dtype=float)).any()
                    except Exception:
                        return False

                if t.size:
                    xy_vals = gdop_xy if gdop_xy.size else gdop_xy_meta
                    d3_vals = gdop_3d if gdop_3d.size else gdop_3d_meta
                    has_xy = _finite_any(xy_vals)
                    has_3d = _finite_any(d3_vals)
                    if has_xy or has_3d:
                        plt.figure(figsize=(8, 3))
                        if has_xy:
                            n_xy = min(len(t), len(xy_vals))
                            plt.plot(t[:n_xy], xy_vals[:n_xy], label="GDOP (xy)")
                        if has_3d:
                            n_3d = min(len(t), len(d3_vals))
                            plt.plot(t[:n_3d], d3_vals[:n_3d], label="GDOP (3d)")
                        # Policy may select based on XY when depth is available; gdop_3d is still logged for interpretability.
                        plt.xlabel("time (s)")
                        plt.ylabel("GDOP")
                        plt.legend()
                        plt.grid(True, alpha=0.3)
                        save_fig(out_dir / "fig_gdop.png")

                if t.size and score_arr.size:
                    if _finite_any(score_arr):
                        plt.figure(figsize=(8, 3))
                        plt.plot(t[:len(score_arr)], score_arr, label="score")
                        if f_obs_arr.size and _finite_any(f_obs_arr):
                            plt.plot(t[:len(f_obs_arr)], f_obs_arr, label="f_obs")
                        if f_energy_arr.size and _finite_any(f_energy_arr):
                            plt.plot(t[:len(f_energy_arr)], f_energy_arr, label="f_energy")
                        if f_mission_arr.size and _finite_any(f_mission_arr):
                            plt.plot(t[:len(f_mission_arr)], f_mission_arr, label="f_mission")
                        plt.xlabel("time (s)")
                        plt.ylabel("score")
                        plt.legend()
                        plt.grid(True, alpha=0.3)
                        save_fig(out_dir / "fig_score_vs_time.png")

                if t.size and soc_arr.size:
                    if _finite_any(soc_arr):
                        plt.figure(figsize=(8, 3))
                        plt.plot(t[:len(soc_arr)], soc_arr, label="soc")
                        plt.xlabel("time (s)")
                        plt.ylabel("SOC")
                        plt.grid(True, alpha=0.3)
                        save_fig(out_dir / "fig_soc_vs_time.png")

            except Exception as e2:
                print(f"[WARN] Could not save/plot timeseries: {e2}")

        # If nothing was plotted from timeseries, try selector-only quick plot to avoid empty outputs.
        if not saved_figs and selector_meta_log:
            try:
                import pandas as pd  # type: ignore
                meta_df = pd.DataFrame(selector_meta_log)
                if not meta_df.empty and "t" in meta_df.columns:
                    if "n_active" in meta_df.columns:
                        plt.figure(figsize=(8, 3))
                        plt.plot(meta_df["t"], meta_df["n_active"], label="n_active")
                        plt.xlabel("time (s)")
                        plt.ylabel("# active beacons")
                        plt.grid(True, alpha=0.3)
                        save_fig(out_dir / "fig_active_count.png")
                    if "score" in meta_df.columns:
                        plt.figure(figsize=(8, 3))
                        plt.plot(meta_df["t"], meta_df["score"], label="score")
                        plt.xlabel("time (s)")
                        plt.ylabel("score")
                        plt.grid(True, alpha=0.3)
                        save_fig(out_dir / "fig_score_vs_time.png")
            except Exception as e3:
                print(f"[WARN] Selector-only quick plot failed: {e3}")
    else:
        # If runner did not return timeseries, still write selector meta for inspection.
        if selector_meta_log:
            try:
                import pandas as pd  # type: ignore

                meta_df = pd.DataFrame(selector_meta_log)
                meta_df.to_csv(out_dir / "timeseries.csv", index=False)

                if "t" in meta_df.columns and "n_active" in meta_df.columns:
                    plt.figure(figsize=(8, 3))
                    plt.plot(meta_df["t"], meta_df["n_active"], label="n_active")
                    plt.xlabel("time (s)")
                    plt.ylabel("# active beacons")
                    plt.grid(True, alpha=0.3)
                    save_fig(out_dir / "fig_active_count.png")

                if "t" in meta_df.columns and "score" in meta_df.columns:
                    plt.figure(figsize=(8, 3))
                    plt.plot(meta_df["t"], meta_df["score"], label="score")
                    for comp in ["f_obs", "f_energy", "f_mission"]:
                        if comp in meta_df:
                            plt.plot(meta_df["t"], meta_df[comp], label=comp)
                    plt.xlabel("time (s)")
                    plt.ylabel("score")
                    plt.legend()
                    plt.grid(True, alpha=0.3)
                    save_fig(out_dir / "fig_score_vs_time.png")

                if "t" in meta_df.columns and "soc" in meta_df.columns:
                    plt.figure(figsize=(8, 3))
                    plt.plot(meta_df["t"], meta_df["soc"], label="soc")
                    plt.xlabel("time (s)")
                    plt.ylabel("SOC")
                    plt.grid(True, alpha=0.3)
                    save_fig(out_dir / "fig_soc_vs_time.png")

            except Exception as e:
                print(f"[WARN] Could not write selector-only timeseries (pandas path): {e}")
                # Fallback: write CSV and basic plots without pandas dependency
                try:
                    import csv

                    fieldnames = sorted({k for row in selector_meta_log for k in row.keys()})
                    timeseries_csv = out_dir / "timeseries.csv"
                    with open(timeseries_csv, "w", newline="") as f:
                        w = csv.DictWriter(f, fieldnames=fieldnames)
                        w.writeheader()
                        for row in selector_meta_log:
                            w.writerow(row)

                    # Basic plots: score and soc if available
                    t_vals = [row.get("t") for row in selector_meta_log if "t" in row]
                    if t_vals:
                        n_active_vals = []
                        for row in selector_meta_log:
                            if "n_active" in row:
                                n_active_vals.append(row.get("n_active"))
                            elif "selected" in row and isinstance(row.get("selected"), (list, tuple)):
                                n_active_vals.append(len(row.get("selected")))
                            else:
                                n_active_vals.append(None)
                        if any(v is not None for v in n_active_vals):
                            plt.figure(figsize=(8, 3))
                            plt.plot(t_vals, n_active_vals, label="n_active")
                            plt.xlabel("time (s)")
                            plt.ylabel("# active beacons")
                            plt.grid(True, alpha=0.3)
                            save_fig(out_dir / "fig_active_count.png")

                        score_vals = [row.get("score") for row in selector_meta_log]
                        if any(s is not None for s in score_vals):
                            plt.figure(figsize=(8, 3))
                            plt.plot(t_vals, score_vals, label="score")
                            plt.xlabel("time (s)")
                            plt.ylabel("score")
                            plt.grid(True, alpha=0.3)
                            save_fig(out_dir / "fig_score_vs_time.png")

                        soc_vals = [row.get("soc") for row in selector_meta_log]
                        if any(s is not None for s in soc_vals):
                            plt.figure(figsize=(8, 3))
                            plt.plot(t_vals, soc_vals, label="soc")
                            plt.xlabel("time (s)")
                            plt.ylabel("SOC")
                            plt.grid(True, alpha=0.3)
                            save_fig(out_dir / "fig_soc_vs_time.png")
                except Exception as e2:
                    print(f"[WARN] Could not write selector-only timeseries (csv fallback): {e2}")

        # Final guard: if still no figures, attempt minimal plot from selector_meta_log
        if not saved_figs and selector_meta_log:
            try:
                t_vals = [row.get("t") for row in selector_meta_log if row.get("t") is not None]
                if t_vals:
                    n_active_vals = [row.get("n_active") if "n_active" in row else (len(row.get("selected")) if isinstance(row.get("selected"), (list, tuple)) else None) for row in selector_meta_log]
                    if any(v is not None for v in n_active_vals):
                        plt.figure(figsize=(8, 3))
                        plt.plot(t_vals, n_active_vals, label="n_active")
                        plt.xlabel("time (s)")
                        plt.ylabel("# active beacons")
                        plt.grid(True, alpha=0.3)
                        save_fig(out_dir / "fig_active_count.png")
            except Exception as e3:
                print(f"[WARN] Minimal selector plot failed: {e3}")

    # ---------------- Report saved plots ----------------
    if saved_figs:
        fig_paths_txt = out_dir / "fig_paths.txt"
        with open(fig_paths_txt, "w") as f:
            for p in saved_figs:
                f.write(str(p) + "\n")
        print(f"[info] saved figures: {len(saved_figs)} (listed in {fig_paths_txt})")
        if args.show_plots:
            try:
                plt.show(block=True)
            except Exception as e:
                print(f"[WARN] Could not display plots: {e}")
    else:
        print("[info] no figures were generated (saved_figs empty)")

    # If user wants to always see plots and none were generated, try showing current pyplot state to aid debugging.
    if args.show_plots and not saved_figs:
        try:
            plt.show(block=True)
        except Exception as e:
            print(f"[WARN] Could not display empty plot state: {e}")

    # ---------------- Summary ----------------
    summary = {
        "rmse_pos": results.get("rmse_pos", results.get("pos_rmse_total", None)),
        "final_pos_err": results.get("final_pos_err", results.get("final_position_error", None)),
        "out_dir": str(out_dir),
    }
    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("Done.")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
