"""
adaptive_modem_manager_v2.py

Adaptive modem (beacon) subset selection for range-only acoustic updates.

- Supports selecting 0..N beacons (including 1 beacon).
- Multi-objective selection: minimize predicted posterior position uncertainty + size penalty.
- Robust to beacon dropouts via `available_ids` (IDs, not indices).
- Provides both 3D and XY (2D) geometry metrics (rank/GDOP/FIM logdet/CRLB std).
- Uncertainty gating with hysteresis and switching hysteresis (dwell + margin).

This module is geometry/math-only and does not run HoloOcean.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from itertools import combinations
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

ArrayLike = Union[np.ndarray, Sequence[float]]


def _safe_rank(A: np.ndarray, tol: float = 1e-10) -> int:
    if A.size == 0:
        return 0
    try:
        s = np.linalg.svd(A, compute_uv=False)
        return int(np.sum(s > tol))
    except np.linalg.LinAlgError:
        return 0


def _fim_from_H(H: np.ndarray, sigma_r: float) -> np.ndarray:
    if H.size == 0:
        return np.zeros((H.shape[1], H.shape[1]))
    w = 1.0 / (sigma_r**2)
    return (H.T @ H) * w


def _gdop_from_fim(F: np.ndarray) -> float:
    if F.size == 0:
        return float("inf")
    try:
        Finv = np.linalg.inv(F)
    except np.linalg.LinAlgError:
        Finv = np.linalg.pinv(F, rcond=1e-12)
    tr = float(np.trace(Finv))
    if (not np.isfinite(tr)) or tr < 0:
        return float("inf")
    return float(np.sqrt(tr))


def _logdet_psd(F: np.ndarray, eps: float = 1e-12) -> float:
    if F.size == 0:
        return -float("inf")
    d = F.shape[0]
    F_reg = F + eps * np.eye(d)
    sign, ld = np.linalg.slogdet(F_reg)
    if sign <= 0 or (not np.isfinite(ld)):
        return -float("inf")
    return float(ld)


def _crlb_std_from_fim(F: np.ndarray) -> np.ndarray:
    if F.size == 0:
        return np.full((F.shape[0],), np.inf, dtype=float)
    try:
        Finv = np.linalg.inv(F)
    except np.linalg.LinAlgError:
        Finv = np.linalg.pinv(F, rcond=1e-12)
    diag = np.maximum(np.diag(Finv), 0.0)
    return np.sqrt(diag)


def _range_jacobian_pos(a_pos: np.ndarray, beacon_pos: np.ndarray) -> np.ndarray:
    # Range: ||p - s||, Jacobian wrt p is (p - s)/||p - s||
    rel = a_pos[None, :] - beacon_pos
    r = np.linalg.norm(rel, axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        H = rel / r[:, None]
    return np.nan_to_num(H, nan=0.0, posinf=0.0, neginf=0.0)


def _range_jacobian_xy(a_pos: np.ndarray, beacon_pos: np.ndarray) -> np.ndarray:
    # Jacobian wrt (x,y): (dx/r, dy/r)
    rel_xy = a_pos[None, :2] - beacon_pos[:, :2]
    r = np.linalg.norm(a_pos[None, :] - beacon_pos, axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        Hxy = rel_xy / r[:, None]
    return np.nan_to_num(Hxy, nan=0.0, posinf=0.0, neginf=0.0)


def _posterior_cov_trace(P: np.ndarray, H: np.ndarray, sigma_r: float) -> float:
    """
    Predict posterior covariance trace for a linearized update:
      P+ = P - P H^T (H P H^T + R)^{-1} H P
    """
    if H.size == 0:
        return float(np.trace(P))

    R = (sigma_r**2) * np.eye(H.shape[0])
    S = H @ P @ H.T + R
    try:
        Sinv = np.linalg.inv(S)
    except np.linalg.LinAlgError:
        Sinv = np.linalg.pinv(S, rcond=1e-12)

    K = P @ H.T @ Sinv
    Pplus = P - K @ (H @ P)
    Pplus = 0.5 * (Pplus + Pplus.T)  # symmetrize
    tr = float(np.trace(Pplus))
    return tr if np.isfinite(tr) else float(np.trace(P))


@dataclass
class SelectionMetrics:
    t: Optional[float]
    chosen_ids: List[Union[str, int]]
    n_chosen: int

    # 3D geometry
    rank_3d: int
    gdop_3d: float
    fim_logdet_3d: float
    crlb_std_3d: List[float]

    # XY geometry
    rank_xy: int
    gdop_xy: float
    fim_logdet_xy: float
    crlb_std_xy: List[float]

    # decision telemetry
    acoustics_enabled: bool
    reason: str
    score: float
    soc: Optional[float]


class EnergyModel:
    def __init__(
        self,
        base_drain_w: float,
        beacon_drain_w: float,
        battery_wh: float,
        drain_scale: float,
        soc_init: float,
        soc_min: float,
    ) -> None:
        self.base = float(base_drain_w)
        self.beacon = float(beacon_drain_w)
        self.battery_wh = max(1e-6, float(battery_wh))
        self.drain_scale = float(drain_scale)
        self.soc_init = max(0.0, min(1.0, float(soc_init)))
        self.soc = self.soc_init
        self.soc_min = max(0.0, min(1.0, float(soc_min)))

    def power(self, n_beacons: int) -> float:
        return self.base + self.beacon * float(max(0, n_beacons))

    def predict_soc(self, dt: float, n_beacons: int) -> float:
        power_w = self.power(n_beacons)
        dsoc = (power_w * max(0.0, dt)) / (self.battery_wh * 3600.0)
        soc_pred = self.soc - self.drain_scale * dsoc
        return max(self.soc_min, min(1.0, soc_pred))

    def step(self, dt: float, n_beacons: int) -> float:
        self.soc = self.predict_soc(dt, n_beacons)
        return self.soc


class AdaptiveModemManagerV2:
    """
    Adaptive subset selector with uncertainty gating + switching hysteresis.

    Important: This selects *IDs* (strings/ints), not indices.
    """

    def __init__(
        self,
        beacon_positions: Dict[Union[str, int], ArrayLike],
        sigma_r: float = 0.5,
        size_penalty: float = 0.0,
        rank_deficit_penalty: float = 10.0,
        min_dwell_steps: int = 10,
        switch_margin: float = 1e-3,
        enable_uncertainty_gate: bool = True,
        gate_on_threshold: float = 1.5,   # turn ON if trace(Ppos) > this
        gate_off_threshold: float = 1.0,  # turn OFF if trace(Ppos) < this
        target_unc_xy: Optional[float] = None,
        target_unc_3d: Optional[float] = None,
        off_unc_mult: float = 1.0,
        on_unc_mult: Optional[float] = None,
        gate_min_dwell_steps: int = 0,
        min_subset_size: int = 0,         # allows 0..N
        max_subset_size: Optional[int] = None,
        prefer_smaller: bool = True,
        # Optional energy model (disabled if battery_wh <= 0 or energy_weight <= 0)
        battery_wh: float = 0.0,
        base_drain_w: float = 0.0,
        beacon_drain_w: float = 0.0,
        drain_scale: float = 1.0,
        soc_init: float = 1.0,
        soc_min: float = 0.0,
        energy_weight: float = 0.0,
        low_power_soc: float = 0.2,
        energy_weight_low_power_mult: float = 2.0,
        size_penalty_low_power_mult: float = 2.0,
        rank_deficit_penalty_low_power_mult: float = 1.0,
    ) -> None:
        self.beacon_positions = {k: np.asarray(v, dtype=float).reshape(3,) for k, v in beacon_positions.items()}
        self.sigma_r = float(sigma_r)

        self.size_penalty = float(size_penalty)
        self.rank_deficit_penalty = float(rank_deficit_penalty)
        self.min_dwell_steps = int(min_dwell_steps)
        self.switch_margin = float(switch_margin)

        self.enable_uncertainty_gate = bool(enable_uncertainty_gate)
        self.gate_on_threshold = float(gate_on_threshold)
        self.gate_off_threshold = float(gate_off_threshold)
        self.target_unc_xy = float(target_unc_xy) if target_unc_xy is not None else None
        self.target_unc_3d = float(target_unc_3d) if target_unc_3d is not None else None
        self.off_unc_mult = float(off_unc_mult)
        self.on_unc_mult = float(on_unc_mult) if on_unc_mult is not None else None
        self.gate_min_dwell_steps = int(gate_min_dwell_steps)

        self.min_subset_size = int(min_subset_size)
        self.max_subset_size = max_subset_size
        self.prefer_smaller = bool(prefer_smaller)

        self.energy_weight = float(energy_weight)
        self.low_power_soc = float(low_power_soc)
        self.energy_weight_low_power_mult = float(energy_weight_low_power_mult)
        self.size_penalty_low_power_mult = float(size_penalty_low_power_mult)
        self.rank_deficit_penalty_low_power_mult = float(rank_deficit_penalty_low_power_mult)
        if battery_wh > 0.0 and self.energy_weight > 0.0:
            self.energy = EnergyModel(
                base_drain_w=base_drain_w,
                beacon_drain_w=beacon_drain_w,
                battery_wh=battery_wh,
                drain_scale=drain_scale,
                soc_init=soc_init,
                soc_min=soc_min,
            )
        else:
            self.energy = None
        self._soc_init = float(soc_init)
        self._last_t: Optional[float] = None

        self._acoustics_enabled = True
        # Pre-seed with all known beacons so the first evaluation starts from
        # the full set and can only switch to a subset if it genuinely improves.
        self._current_ids: List[Union[str, int]] = list(self.beacon_positions.keys())
        self._dwell = 0
        self._last_available: Optional[Tuple[Union[str, int], ...]] = None
        self._gate_dwell = 0

    def reset(self) -> None:
        self._acoustics_enabled = True
        self._current_ids = list(self.beacon_positions.keys())
        self._dwell = 0
        self._last_available = None
        self._last_t = None
        self._gate_dwell = 0
        if self.energy is not None:
            self.energy.soc = self.energy.soc_init

    def _update_uncert_gate(self, value: float, on_threshold: float, off_threshold: float) -> None:
        if not self.enable_uncertainty_gate:
            self._acoustics_enabled = True
            return
        desired = self._acoustics_enabled
        if self._acoustics_enabled:
            if value < off_threshold:
                desired = False
        else:
            if value > on_threshold:
                desired = True

        if desired != self._acoustics_enabled:
            if self.gate_min_dwell_steps > 0 and self._gate_dwell < self.gate_min_dwell_steps:
                self._gate_dwell += 1
                return
            self._acoustics_enabled = desired
            self._gate_dwell = 0
        else:
            self._gate_dwell = 0

    def _uncertainty_thresholds(self, depth_available: bool) -> Tuple[Optional[float], Optional[float], bool]:
        """Return (off_thresh, on_thresh, use_sqrt_trace)."""
        target_unc = self.target_unc_xy if depth_available else self.target_unc_3d
        if target_unc is not None and target_unc > 0.0 and self.off_unc_mult > 0.0:
            off = self.off_unc_mult * target_unc
            on_mult = self.on_unc_mult if self.on_unc_mult is not None else max(self.off_unc_mult * 1.25, self.off_unc_mult + 1e-6)
            on = on_mult * target_unc
            return off, on, True
        return self.gate_off_threshold, self.gate_on_threshold, False

    def _enumerate_subsets(self, ids: List[Union[str, int]]) -> List[List[Union[str, int]]]:
        n = len(ids)
        max_k = self.max_subset_size if self.max_subset_size is not None else n
        max_k = min(max_k, n)
        min_k = max(0, self.min_subset_size)

        out: List[List[Union[str, int]]] = []
        for k in range(min_k, max_k + 1):
            for combo in combinations(ids, k):
                out.append(list(combo))
        return out

    def _subset_geometry(self, a_pos: np.ndarray, subset: List[Union[str, int]]) -> dict:
        if not subset:
            return dict(
                rank_3d=0, gdop_3d=float("inf"), fim_logdet_3d=-float("inf"), crlb_std_3d=[float("inf")] * 3,
                rank_xy=0, gdop_xy=float("inf"), fim_logdet_xy=-float("inf"), crlb_std_xy=[float("inf")] * 2,
            )

        B = np.array([self.beacon_positions[i] for i in subset], dtype=float)

        H3 = _range_jacobian_pos(a_pos, B)
        F3 = _fim_from_H(H3, self.sigma_r)
        rank3 = _safe_rank(H3)
        gdop3 = _gdop_from_fim(F3)
        logdet3 = _logdet_psd(F3)
        crlb3 = _crlb_std_from_fim(F3).tolist()

        Hxy = _range_jacobian_xy(a_pos, B)
        Fxy = _fim_from_H(Hxy, self.sigma_r)
        rankxy = _safe_rank(Hxy)
        gdopxy = _gdop_from_fim(Fxy)
        logdetxy = _logdet_psd(Fxy)
        crlbxy = _crlb_std_from_fim(Fxy).tolist()

        return dict(
            rank_3d=rank3, gdop_3d=gdop3, fim_logdet_3d=logdet3, crlb_std_3d=crlb3,
            rank_xy=rankxy, gdop_xy=gdopxy, fim_logdet_xy=logdetxy, crlb_std_xy=crlbxy,
        )

    def metrics_to_dict(self, metrics: SelectionMetrics) -> dict:
        d = asdict(metrics)
        d["chosen_ids"] = [str(x) for x in d["chosen_ids"]]
        return d

    def select(
        self,
        t: Optional[float],
        a_pos: ArrayLike,
        P: np.ndarray,
        available_ids: Sequence[Union[str, int]],
        depth_available: bool = True,
    ) -> Tuple[List[Union[str, int]], SelectionMetrics]:
        """
        Decide beacons at time t.

        P can be full EKF covariance (position is first 3 states) OR a 3x3 Ppos directly.
        """
        a_pos = np.asarray(a_pos, dtype=float).reshape(3,)
        avail = [i for i in available_ids if i in self.beacon_positions]
        avail_key = tuple(sorted(avail, key=lambda x: str(x)))

        # availability change -> reset dwell and clamp current set
        if self._last_available is None or avail_key != self._last_available:
            self._dwell = 0
            self._current_ids = [i for i in self._current_ids if i in avail]
            self._last_available = avail_key

        # extract Ppos
        Ppos = P if P.shape == (3, 3) else P[:3, :3]
        trace_p = float(np.trace(Ppos))

        dt = 0.0 if (self._last_t is None or t is None) else max(0.0, float(t) - float(self._last_t))

        if self.energy is not None and self.energy.soc <= self.energy.soc_min + 1e-9:
            geo = self._subset_geometry(a_pos, [])
            metrics = SelectionMetrics(
                t=t, chosen_ids=[], n_chosen=0,
                rank_3d=geo["rank_3d"], gdop_3d=geo["gdop_3d"], fim_logdet_3d=geo["fim_logdet_3d"], crlb_std_3d=geo["crlb_std_3d"],
                rank_xy=geo["rank_xy"], gdop_xy=geo["gdop_xy"], fim_logdet_xy=geo["fim_logdet_xy"], crlb_std_xy=geo["crlb_std_xy"],
                acoustics_enabled=False, reason="battery_depleted", score=float("inf"),
                soc=self.energy.soc,
            )
            self._current_ids = []
            self._acoustics_enabled = False
            self.energy.step(dt, 0)
            self._last_t = t
            return [], metrics

        off_thresh, on_thresh, use_sqrt = self._uncertainty_thresholds(depth_available)
        if depth_available:
            Psub = Ppos[:2, :2]
        else:
            Psub = Ppos
        unc_value = float(np.sqrt(max(0.0, np.trace(Psub)))) if use_sqrt else trace_p
        self._update_uncert_gate(unc_value, on_thresh, off_thresh)

        # uncertainty gating -> 0 beacons
        if not self._acoustics_enabled:
            geo = self._subset_geometry(a_pos, [])
            metrics = SelectionMetrics(
                t=t, chosen_ids=[], n_chosen=0,
                rank_3d=geo["rank_3d"], gdop_3d=geo["gdop_3d"], fim_logdet_3d=geo["fim_logdet_3d"], crlb_std_3d=geo["crlb_std_3d"],
                rank_xy=geo["rank_xy"], gdop_xy=geo["gdop_xy"], fim_logdet_xy=geo["fim_logdet_xy"], crlb_std_xy=geo["crlb_std_xy"],
                acoustics_enabled=False, reason="uncertainty_gated_off", score=unc_value,
                soc=self.energy.soc if self.energy is not None else None,
            )
            self._current_ids = []
            if self.energy is not None:
                self.energy.step(dt, 0)
            self._last_t = t
            return [], metrics

        # score on XY if depth available, else score on 3D
        if depth_available:
            target_rank = 2
            use_xy = True
        else:
            target_rank = 3
            use_xy = False

        # dwell hold
        if self._dwell < self.min_dwell_steps and all(i in avail for i in self._current_ids):
            self._dwell += 1
            subset = list(self._current_ids)
            geo = self._subset_geometry(a_pos, subset)
            B = np.array([self.beacon_positions[i] for i in subset], dtype=float) if subset else np.zeros((0, 3))

            if use_xy:
                H = _range_jacobian_xy(a_pos, B) if subset else np.zeros((0, 2))
                rank_now = geo["rank_xy"]
            else:
                H = _range_jacobian_pos(a_pos, B) if subset else np.zeros((0, 3))
                rank_now = geo["rank_3d"]

            tr_post = _posterior_cov_trace(Psub, H, self.sigma_r)
            deficit = max(0, target_rank - int(rank_now))
            energy_penalty = 0.0
            size_pen = self.size_penalty
            energy_w = self.energy_weight
            rank_pen = self.rank_deficit_penalty
            if self.energy is not None and self.energy_weight > 0.0:
                if self.energy.soc <= self.low_power_soc:
                    energy_w *= self.energy_weight_low_power_mult
                    size_pen *= self.size_penalty_low_power_mult
                    rank_pen *= self.rank_deficit_penalty_low_power_mult
                max_power = self.energy.power(self.max_subset_size if self.max_subset_size is not None else len(avail))
                power_norm = (self.energy.power(len(subset)) / max_power) if max_power > 0 else 0.0
                energy_penalty = energy_w * power_norm
            score = tr_post + size_pen * len(subset) + rank_pen * deficit + energy_penalty

            metrics = SelectionMetrics(
                t=t, chosen_ids=subset, n_chosen=len(subset),
                rank_3d=geo["rank_3d"], gdop_3d=geo["gdop_3d"], fim_logdet_3d=geo["fim_logdet_3d"], crlb_std_3d=geo["crlb_std_3d"],
                rank_xy=geo["rank_xy"], gdop_xy=geo["gdop_xy"], fim_logdet_xy=geo["fim_logdet_xy"], crlb_std_xy=geo["crlb_std_xy"],
                acoustics_enabled=True, reason="dwell_hold", score=float(score),
                soc=self.energy.soc if self.energy is not None else None,
            )
            if self.energy is not None:
                self.energy.step(dt, len(subset))
            self._last_t = t
            return subset, metrics

        # evaluate all candidates (0..N)
        candidates = self._enumerate_subsets(avail)

        best_subset: List[Union[str, int]] = []
        best_score = float("inf")
        best_geo = None

        for subset in candidates:
            geo = self._subset_geometry(a_pos, subset)
            B = np.array([self.beacon_positions[i] for i in subset], dtype=float) if subset else np.zeros((0, 3))

            if use_xy:
                H = _range_jacobian_xy(a_pos, B) if subset else np.zeros((0, 2))
                rank_now = geo["rank_xy"]
            else:
                H = _range_jacobian_pos(a_pos, B) if subset else np.zeros((0, 3))
                rank_now = geo["rank_3d"]

            tr_post = _posterior_cov_trace(Psub, H, self.sigma_r)
            deficit = max(0, target_rank - int(rank_now))
            energy_penalty = 0.0
            size_pen = self.size_penalty
            energy_w = self.energy_weight
            rank_pen = self.rank_deficit_penalty
            if self.energy is not None and self.energy_weight > 0.0:
                if self.energy.soc <= self.low_power_soc:
                    energy_w *= self.energy_weight_low_power_mult
                    size_pen *= self.size_penalty_low_power_mult
                    rank_pen *= self.rank_deficit_penalty_low_power_mult
                max_power = self.energy.power(self.max_subset_size if self.max_subset_size is not None else len(avail))
                power_norm = (self.energy.power(len(subset)) / max_power) if max_power > 0 else 0.0
                energy_penalty = energy_w * power_norm
            score = tr_post + size_pen * len(subset) + rank_pen * deficit + energy_penalty

            if score < best_score - 1e-12:
                best_score = score
                best_subset = subset
                best_geo = geo
            elif abs(score - best_score) <= 1e-12 and self.prefer_smaller:
                if len(subset) < len(best_subset):
                    best_score = score
                    best_subset = subset
                    best_geo = geo

        if best_geo is None:
            best_geo = self._subset_geometry(a_pos, best_subset)

        # compare with current under switch margin
        cur_subset = [i for i in self._current_ids if i in avail]
        cur_geo = self._subset_geometry(a_pos, cur_subset)
        Bc = np.array([self.beacon_positions[i] for i in cur_subset], dtype=float) if cur_subset else np.zeros((0, 3))

        if use_xy:
            Hc = _range_jacobian_xy(a_pos, Bc) if cur_subset else np.zeros((0, 2))
            rank_c = cur_geo["rank_xy"]
        else:
            Hc = _range_jacobian_pos(a_pos, Bc) if cur_subset else np.zeros((0, 3))
            rank_c = cur_geo["rank_3d"]

        tr_c = _posterior_cov_trace(Psub, Hc, self.sigma_r)
        deficit_c = max(0, target_rank - int(rank_c))
        energy_penalty_c = 0.0
        size_pen_c = self.size_penalty
        energy_w_c = self.energy_weight
        rank_pen_c = self.rank_deficit_penalty
        if self.energy is not None and self.energy_weight > 0.0:
            if self.energy.soc <= self.low_power_soc:
                energy_w_c *= self.energy_weight_low_power_mult
                size_pen_c *= self.size_penalty_low_power_mult
                rank_pen_c *= self.rank_deficit_penalty_low_power_mult
            max_power = self.energy.power(self.max_subset_size if self.max_subset_size is not None else len(avail))
            power_norm = (self.energy.power(len(cur_subset)) / max_power) if max_power > 0 else 0.0
            energy_penalty_c = energy_w_c * power_norm
        cur_score = tr_c + size_pen_c * len(cur_subset) + rank_pen_c * deficit_c + energy_penalty_c

        if (best_score + self.switch_margin) < cur_score:
            self._current_ids = list(best_subset)
            self._dwell = 0
            reason = "switched"
            out_geo = best_geo
            out_score = best_score
        else:
            self._current_ids = list(cur_subset)
            # Keep dwell at min_dwell_steps so the next call re-evaluates
            # immediately rather than re-entering the dwell_hold path.
            self._dwell = self.min_dwell_steps
            reason = "held_margin"
            out_geo = cur_geo
            out_score = cur_score

        metrics = SelectionMetrics(
            t=t, chosen_ids=list(self._current_ids), n_chosen=len(self._current_ids),
            rank_3d=out_geo["rank_3d"], gdop_3d=out_geo["gdop_3d"], fim_logdet_3d=out_geo["fim_logdet_3d"], crlb_std_3d=out_geo["crlb_std_3d"],
            rank_xy=out_geo["rank_xy"], gdop_xy=out_geo["gdop_xy"], fim_logdet_xy=out_geo["fim_logdet_xy"], crlb_std_xy=out_geo["crlb_std_xy"],
            acoustics_enabled=True, reason=reason, score=float(out_score),
            soc=self.energy.soc if self.energy is not None else None,
        )
        if self.energy is not None:
            self.energy.step(dt, len(self._current_ids))
        self._last_t = t
        return list(self._current_ids), metrics
