"""
Adaptive modem (beacon) subset selection for range-only acoustic updates.

- Supports selecting 0..N beacons (including 1 beacon)
- Multi-objective: minimize predicted posterior position uncertainty + size and rank penalties
- Robust to dropout via available_ids (IDs, not indices)
- Provides both XY (2D) and 3D geometry metrics (rank, GDOP, logdet)
- Uncertainty gating with hysteresis (dwell + switch margin)

This module is geometry/math-only and does not run HoloOcean.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from itertools import combinations
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

ArrayLike = Union[np.ndarray, Sequence[float]]


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

def _unit_rows(mat: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms = np.where(norms <= 1e-12, 1.0, norms)
    return mat / norms


def _geom_metrics(a_pos: np.ndarray, beacon_pos: np.ndarray, sigma_r: float) -> Dict[str, float]:
    """Compute geometry metrics for XY and 3D.

    Returns: dict with FIMs, logdets, ranks, GDOPs (inf when not observable).
    """
    if beacon_pos.size == 0:
        return {
            "H3": np.zeros((0, 3)),
            "Hxy": np.zeros((0, 2)),
            "F3": np.zeros((3, 3)),
            "Fxy": np.zeros((2, 2)),
            "logdet3": -float("inf"),
            "logdetxy": -float("inf"),
            "rank3": 0,
            "rankxy": 0,
            "gdop3": float("inf"),
            "gdopxy": float("inf"),
        }

    rel = beacon_pos - a_pos.reshape(1, 3)
    H3 = _unit_rows(rel)  # (n,3)
    Hxy = H3[:, :2]

    def _fim(H: np.ndarray) -> Tuple[np.ndarray, float, int, float]:
        if H.size == 0:
            return np.zeros((H.shape[1], H.shape[1])), -float("inf"), 0, float("inf")
        Rinv = np.eye(H.shape[0]) / (sigma_r ** 2)
        F = H.T @ Rinv @ H
        sign, logdet = np.linalg.slogdet(F + 1e-12 * np.eye(F.shape[0]))
        logdet_val = float(logdet) if sign > 0 else -float("inf")
        # Rank and GDOP
        try:
            s = np.linalg.svd(F, compute_uv=False)
            rank = int(np.sum(s > 1e-10))
        except np.linalg.LinAlgError:
            rank = 0
        if rank < F.shape[0]:
            gdop = float("inf")
        else:
            try:
                Finv = np.linalg.inv(F)
            except np.linalg.LinAlgError:
                Finv = np.linalg.pinv(F, rcond=1e-12)
            tr = float(np.trace(Finv))
            gdop = float(np.sqrt(tr)) if np.isfinite(tr) and tr >= 0 else float("inf")
        return F, logdet_val, rank, gdop

    F3, logdet3, rank3, gdop3 = _fim(H3)
    Fxy, logdetxy, rankxy, gdopxy = _fim(Hxy)

    return {
        "H3": H3,
        "Hxy": Hxy,
        "F3": F3,
        "Fxy": Fxy,
        "logdet3": logdet3,
        "logdetxy": logdetxy,
        "rank3": rank3,
        "rankxy": rankxy,
        "gdop3": gdop3,
        "gdopxy": gdopxy,
    }


def _posterior_cov_trace(P: np.ndarray, H: np.ndarray, sigma_r: float) -> float:
    """Predict posterior covariance trace for linearized range update."""
    if H.size == 0:
        return float(np.trace(P))
    R = (sigma_r ** 2) * np.eye(H.shape[0])
    S = H @ P @ H.T + R
    try:
        Sinv = np.linalg.inv(S)
    except np.linalg.LinAlgError:
        Sinv = np.linalg.pinv(S, rcond=1e-12)
    K = P @ H.T @ Sinv
    Pp = P - K @ (H @ P)
    Pp = 0.5 * (Pp + Pp.T)
    tr = float(np.trace(Pp))
    return tr if np.isfinite(tr) else float(np.trace(P))


@dataclass
class SelectionMetrics:
    t: Optional[float]
    chosen_ids: List[Union[str, int]]
    n_chosen: int
    rank_3d: int
    gdop_3d: float
    fim_logdet_3d: float
    rank_xy: int
    gdop_xy: float
    fim_logdet_xy: float
    acoustics_enabled: bool
    reason: str
    score: float


class AdaptiveModemManagerV2:
    """Adaptive subset selector with uncertainty gating + dwell hysteresis.

    Inputs are beacon IDs (strings or ints), not indices.
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
        gate_on_threshold: float = 1.5,
        gate_off_threshold: float = 1.0,
        min_subset_size: int = 0,
        max_subset_size: Optional[int] = None,
        prefer_smaller: bool = True,
    ) -> None:
        self.beacon_positions = {k: np.asarray(v, dtype=float).reshape(3,) for k, v in beacon_positions.items()}
        self.sigma_r = float(sigma_r)
        self.size_penalty = float(size_penalty)
        self.rank_deficit_penalty = float(rank_deficit_penalty)
        self.min_dwell_steps = max(1, int(min_dwell_steps))
        self.switch_margin = float(switch_margin)
        self.enable_uncertainty_gate = bool(enable_uncertainty_gate)
        self.gate_on_threshold = float(gate_on_threshold)
        self.gate_off_threshold = float(gate_off_threshold)
        self.min_subset_size = max(0, int(min_subset_size))
        self.max_subset_size = max_subset_size
        self.prefer_smaller = bool(prefer_smaller)
        self._acoustics_enabled = True
        self._current_ids: List[Union[str, int]] = []
        self._dwell = 0
        self._last_available: Optional[Tuple[Union[str, int], ...]] = None

    # ------------------------------------------------------------------
    def reset(self) -> None:
        self._acoustics_enabled = True
        self._current_ids = []
        self._dwell = 0
        self._last_available = None

    # ------------------------------------------------------------------
    def _update_uncert_gate(self, trace_p: float) -> None:
        if not self.enable_uncertainty_gate:
            self._acoustics_enabled = True
            return
        if self._acoustics_enabled and trace_p < self.gate_off_threshold:
            self._acoustics_enabled = False
        elif (not self._acoustics_enabled) and trace_p > self.gate_on_threshold:
            self._acoustics_enabled = True

    # ------------------------------------------------------------------
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

    # ------------------------------------------------------------------
    def select(
        self,
        t: Optional[float],
        a_pos: ArrayLike,
        P: np.ndarray,
        available_ids: Sequence[Union[str, int]],
        depth_available: bool = True,
    ) -> Tuple[List[Union[str, int]], SelectionMetrics]:
        a_pos = np.asarray(a_pos, dtype=float).reshape(3,)
        avail = [i for i in available_ids if i in self.beacon_positions]
        avail_key = tuple(sorted(avail, key=lambda x: str(x)))

        # availability changes reset dwell
        if self._last_available is None or avail_key != self._last_available:
            self._dwell = 0
            self._current_ids = [i for i in self._current_ids if i in avail]
            self._last_available = avail_key

        # extract Ppos
        Ppos = P if P.shape == (3, 3) else P[:3, :3]
        trace_p = float(np.trace(Ppos))
        self._update_uncert_gate(trace_p)

        # Uncertainty gate -> acoustics off
        if not self._acoustics_enabled:
            metrics = SelectionMetrics(
                t=t,
                chosen_ids=[],
                n_chosen=0,
                rank_3d=0,
                gdop_3d=float("inf"),
                fim_logdet_3d=-float("inf"),
                rank_xy=0,
                gdop_xy=float("inf"),
                fim_logdet_xy=-float("inf"),
                acoustics_enabled=False,
                reason="uncertainty_gated_off",
                score=trace_p,
            )
            self._current_ids = []
            return [], metrics

        use_xy = bool(depth_available)
        Psub = Ppos[:2, :2] if use_xy else Ppos
        target_rank = 2 if use_xy else 3

        # dwell hold
        if self._dwell < self.min_dwell_steps and all(i in avail for i in self._current_ids):
            self._dwell += 1
            subset = list(self._current_ids)
            pos = np.array([self.beacon_positions[i] for i in subset]) if subset else np.zeros((0, 3))
            g = _geom_metrics(a_pos, pos, self.sigma_r)
            H = g["Hxy"] if use_xy else g["H3"]
            rank_now = g["rankxy"] if use_xy else g["rank3"]
            tr_post = _posterior_cov_trace(Psub, H, self.sigma_r)
            deficit = max(0, target_rank - rank_now)
            score = tr_post + self.size_penalty * len(subset) + self.rank_deficit_penalty * deficit
            metrics = SelectionMetrics(
                t=t,
                chosen_ids=subset,
                n_chosen=len(subset),
                rank_3d=g["rank3"],
                gdop_3d=g["gdop3"],
                fim_logdet_3d=g["logdet3"],
                rank_xy=g["rankxy"],
                gdop_xy=g["gdopxy"],
                fim_logdet_xy=g["logdetxy"],
                acoustics_enabled=True,
                reason="dwell_hold",
                score=float(score),
            )
            return subset, metrics

        candidates = self._enumerate_subsets(avail)
        best_subset: List[Union[str, int]] = []
        best_score = float("inf")
        best_geo: Optional[Dict[str, float]] = None

        for subset in candidates:
            pos = np.array([self.beacon_positions[i] for i in subset]) if subset else np.zeros((0, 3))
            g = _geom_metrics(a_pos, pos, self.sigma_r)
            H = g["Hxy"] if use_xy else g["H3"]
            rank_now = g["rankxy"] if use_xy else g["rank3"]
            tr_post = _posterior_cov_trace(Psub, H, self.sigma_r)
            deficit = max(0, target_rank - rank_now)
            score = tr_post + self.size_penalty * len(subset) + self.rank_deficit_penalty * deficit

            better = score < best_score - 1e-12
            tie = abs(score - best_score) <= 1e-12 and self.prefer_smaller and len(subset) < len(best_subset)
            if better or tie:
                best_score = score
                best_subset = subset
                best_geo = g

        if best_geo is None:
            best_geo = _geom_metrics(a_pos, np.zeros((0, 3)), self.sigma_r)

        # Compare against current under switch margin
        cur_subset = [i for i in self._current_ids if i in avail]
        pos_cur = np.array([self.beacon_positions[i] for i in cur_subset]) if cur_subset else np.zeros((0, 3))
        gcur = _geom_metrics(a_pos, pos_cur, self.sigma_r)
        Hc = gcur["Hxy"] if use_xy else gcur["H3"]
        rank_c = gcur["rankxy"] if use_xy else gcur["rank3"]
        tr_c = _posterior_cov_trace(Psub, Hc, self.sigma_r)
        deficit_c = max(0, target_rank - rank_c)
        cur_score = tr_c + self.size_penalty * len(cur_subset) + self.rank_deficit_penalty * deficit_c

        if (best_score + self.switch_margin) < cur_score:
            self._current_ids = list(best_subset)
            self._dwell = 0
            reason = "switched"
            out_geo = best_geo
            out_score = best_score
        else:
            self._current_ids = list(cur_subset)
            self._dwell = 1
            reason = "held_margin"
            out_geo = gcur
            out_score = cur_score

        metrics = SelectionMetrics(
            t=t,
            chosen_ids=list(self._current_ids),
            n_chosen=len(self._current_ids),
            rank_3d=out_geo["rank3"],
            gdop_3d=out_geo["gdop3"],
            fim_logdet_3d=out_geo["logdet3"],
            rank_xy=out_geo["rankxy"],
            gdop_xy=out_geo["gdopxy"],
            fim_logdet_xy=out_geo["logdetxy"],
            acoustics_enabled=True,
            reason=reason,
            score=float(out_score),
        )
        return list(self._current_ids), metrics


def metrics_to_dict(metrics: SelectionMetrics) -> dict:
    d = asdict(metrics)
    d["chosen_ids"] = [str(x) for x in d.get("chosen_ids", [])]
    return d
