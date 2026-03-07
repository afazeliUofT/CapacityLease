from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Iterable

import math
import numpy as np
from scipy.optimize import brentq


@dataclass
class ScalarSearchResult:
    x: float
    value: float
    payload: Any = None



def safe_brentq(
    func: Callable[[float], float],
    a: float,
    b: float,
    *,
    xtol: float,
    rtol: float,
    maxiter: int,
) -> float:
    """Robust scalar root solver with bisection fallback."""
    fa = float(func(a))
    fb = float(func(b))
    if abs(fa) <= xtol:
        return float(a)
    if abs(fb) <= xtol:
        return float(b)
    if fa == 0.0:
        return float(a)
    if fb == 0.0:
        return float(b)
    if fa * fb > 0:
        raise ValueError(f"Root is not bracketed on [{a}, {b}] because f(a)={fa}, f(b)={fb}.")
    try:
        return float(brentq(func, a, b, xtol=xtol, rtol=rtol, maxiter=maxiter))
    except Exception:
        lo, hi = float(a), float(b)
        flo, fhi = fa, fb
        for _ in range(maxiter * 4):
            mid = 0.5 * (lo + hi)
            fmid = float(func(mid))
            if abs(fmid) <= xtol or abs(hi - lo) <= max(xtol, rtol * max(1.0, abs(mid))):
                return float(mid)
            if flo * fmid <= 0:
                hi, fhi = mid, fmid
            else:
                lo, flo = mid, fmid
        return float(0.5 * (lo + hi))



def expand_bracket(
    func: Callable[[float], float],
    start: float,
    *,
    step: float = 1.0,
    growth: float = 2.0,
    max_value: float = 1e6,
    target_sign: int = 1,
) -> tuple[float, float] | None:
    """Find [start, hi] such that func(start) and func(hi) have opposite sign.

    target_sign = +1 means search until func(hi) >= 0 starting from func(start) <= 0.
    target_sign = -1 means search until func(hi) <= 0 starting from func(start) >= 0.
    """
    lo = float(start)
    flo = float(func(lo))
    hi = float(max(step, lo + step))
    fhi = float(func(hi))
    if target_sign > 0:
        while fhi < 0 and hi < max_value:
            hi = hi * growth if hi > 0 else step
            fhi = float(func(hi))
        if flo <= 0 <= fhi:
            return lo, hi
    else:
        while fhi > 0 and hi < max_value:
            hi = hi * growth if hi > 0 else step
            fhi = float(func(hi))
        if flo >= 0 >= fhi:
            return lo, hi
    return None



def _local_maxima_indices(values: np.ndarray) -> list[int]:
    if values.size == 0:
        return []
    if values.size <= 2:
        return list(range(values.size))
    out: list[int] = []
    for idx in range(values.size):
        left = values[idx - 1] if idx > 0 else -math.inf
        right = values[idx + 1] if idx < values.size - 1 else -math.inf
        if values[idx] >= left and values[idx] >= right:
            out.append(idx)
    return out



def grid_refine_maximize_scalar(
    objective: Callable[[float], tuple[float, Any]],
    lo: float,
    hi: float,
    *,
    points: int,
    refine_levels: int,
    top_k: int,
) -> ScalarSearchResult | None:
    """Robust coarse-to-fine maximization for possibly non-smooth scalar objectives.

    objective(x) returns (value, payload). Invalid points should return (-inf, payload).
    """
    if hi < lo:
        return None
    if abs(hi - lo) <= 1e-15:
        value, payload = objective(lo)
        if not np.isfinite(value):
            return None
        return ScalarSearchResult(x=float(lo), value=float(value), payload=payload)

    intervals: list[tuple[float, float]] = [(float(lo), float(hi))]
    best: ScalarSearchResult | None = None

    for level in range(refine_levels + 1):
        candidates: list[tuple[tuple[float, float], list[tuple[float, float, Any]]]] = []
        for left, right in intervals:
            xs = np.linspace(left, right, points)
            samples: list[tuple[float, float, Any]] = []
            for x in xs:
                value, payload = objective(float(x))
                samples.append((float(x), float(value), payload))
                if np.isfinite(value) and (best is None or value > best.value):
                    best = ScalarSearchResult(x=float(x), value=float(value), payload=payload)
            candidates.append(((left, right), samples))

        if level == refine_levels:
            break

        new_intervals: list[tuple[float, float]] = []
        ranked: list[tuple[float, float, float]] = []
        for (left, right), samples in candidates:
            values = np.asarray([sample[1] for sample in samples], dtype=float)
            xs = np.asarray([sample[0] for sample in samples], dtype=float)
            for idx in _local_maxima_indices(values):
                if np.isfinite(values[idx]):
                    ranked.append((float(values[idx]), float(xs[idx]), float(idx)))
        if not ranked:
            break
        ranked.sort(reverse=True, key=lambda item: item[0])

        used: list[tuple[float, float]] = []
        for _, x_star, _ in ranked[: max(top_k, 1)]:
            for left, right in intervals:
                if left - 1e-12 <= x_star <= right + 1e-12:
                    xs = np.linspace(left, right, points)
                    idx = int(np.argmin(np.abs(xs - x_star)))
                    left_idx = max(0, idx - 1)
                    right_idx = min(points - 1, idx + 1)
                    interval = (float(xs[left_idx]), float(xs[right_idx]))
                    if all(abs(interval[0] - prev[0]) > 1e-12 or abs(interval[1] - prev[1]) > 1e-12 for prev in used):
                        used.append(interval)
                    break
        if not used:
            break
        intervals = used

    return best
