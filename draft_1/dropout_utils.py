"""
dropout_utils.py

Beacon availability + dropout scheduling helpers.

Use this in your validation harness to filter available beacons before
calling the selector.

Supported schedule formats:
1) dict: {"usv2": [(30, 50), (100, 120)], "usv4": [(60, 90)]}
2) string: "usv2:30-50;100-120 | usv4:60-90"
3) CSV: columns beacon_id,t_start,t_end
"""

from __future__ import annotations

from typing import Dict, List, Sequence, Tuple, Union, Optional
import csv
import re

BeaconId = Union[str, int]
Schedule = Dict[BeaconId, List[Tuple[float, float]]]


def in_interval(t: float, t0: float, t1: float) -> bool:
    return (t >= t0) and (t <= t1)


def in_dropout(t: float, intervals: List[Tuple[float, float]]) -> bool:
    for a, b in intervals:
        if in_interval(t, a, b):
            return True
    return False


def filter_available(t: float, ids: Sequence[BeaconId], schedule: Optional[Schedule]) -> List[BeaconId]:
    if not schedule:
        return list(ids)
    out: List[BeaconId] = []
    for i in ids:
        if not in_dropout(t, schedule.get(i, [])):
            out.append(i)
    return out


def parse_schedule(schedule_like) -> Schedule:
    if schedule_like is None:
        return {}

    if isinstance(schedule_like, dict):
        out: Schedule = {}
        for k, v in schedule_like.items():
            out[k] = [(float(a), float(b)) for a, b in v]
        return out

    if isinstance(schedule_like, str):
        s = schedule_like.strip()
        if not s:
            return {}
        chunks = [c.strip() for c in re.split(r"[|\n]+", s) if c.strip()]
        out: Schedule = {}
        for chunk in chunks:
            if ":" not in chunk:
                continue
            bid, rest = chunk.split(":", 1)
            bid = bid.strip()
            parts = [p.strip() for p in re.split(r"[;,]+", rest) if p.strip()]
            intervals: List[Tuple[float, float]] = []
            for p in parts:
                m = re.match(r"^\s*([0-9]*\.?[0-9]+)\s*-\s*([0-9]*\.?[0-9]+)\s*$", p)
                if not m:
                    continue
                a = float(m.group(1)); b = float(m.group(2))
                if b < a:
                    a, b = b, a
                intervals.append((a, b))
            out[bid] = intervals
        return out

    raise TypeError(f"Unsupported schedule type: {type(schedule_like)}")


def load_schedule_csv(
    csv_path: str,
    beacon_id_col: str = "beacon_id",
    t0_col: str = "t_start",
    t1_col: str = "t_end",
) -> Schedule:
    out: Schedule = {}
    with open(csv_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            bid = row[beacon_id_col]
            t0 = float(row[t0_col]); t1 = float(row[t1_col])
            if t1 < t0:
                t0, t1 = t1, t0
            out.setdefault(bid, []).append((t0, t1))
    return out
