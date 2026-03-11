cat > tools/apply_scalar_search_rescue_patch.py <<'PY'
#!/usr/bin/env python3
from __future__ import annotations

import shutil
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
TARGET = REPO_ROOT / "capacity_lease" / "numerics.py"
BACKUP = REPO_ROOT / "capacity_lease" / "numerics.py.pre_scalar_search_rescue.bak"

BEGIN_ANCHOR = "def grid_refine_maximize_scalar("
END_ANCHOR = "    return best"


REPLACEMENT = '''def grid_refine_maximize_scalar(
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

    This implementation adds staggered-grid rescue sampling when the valid region
    is sparse or is missed entirely by the primary lattice.
    """
    if hi < lo:
        return None

    if abs(hi - lo) <= 1e-15:
        value, payload = objective(lo)
        if not np.isfinite(value):
            return None
        return ScalarSearchResult(x=float(lo), value=float(value), payload=payload)

    point_count = max(int(points), 3)
    level_count = max(int(refine_levels), 0)
    top_count = max(int(top_k), 1)

    eval_cache: dict[float, tuple[float, float, Any]] = {}

    def eval_point(x: float) -> tuple[float, float, Any]:
        x = float(x)
        key = round(x, 14)
        cached = eval_cache.get(key)
        if cached is None:
            value, payload = objective(x)
            cached = (x, float(value), payload)
            eval_cache[key] = cached
        return cached

    def base_grid(left: float, right: float) -> np.ndarray:
        if abs(right - left) <= 1e-15:
            return np.asarray([float(left)], dtype=float)
        return np.linspace(float(left), float(right), point_count, dtype=float)

    def staggered_grid(xs: np.ndarray) -> np.ndarray:
        xs = np.asarray(xs, dtype=float)
        if xs.size <= 1:
            return xs
        mids = 0.5 * (xs[:-1] + xs[1:])
        return np.unique(np.concatenate([xs, mids]))

    def quarter_refined_grid(xs: np.ndarray) -> np.ndarray:
        xs = np.asarray(xs, dtype=float)
        if xs.size <= 1:
            return xs
        mids = 0.5 * (xs[:-1] + xs[1:])
        left_quarters = 0.5 * (xs[:-1] + mids)
        right_quarters = 0.5 * (mids + xs[1:])
        return np.unique(np.concatenate([xs, mids, left_quarters, right_quarters]))

    def sample_interval(left: float, right: float) -> tuple[np.ndarray, list[tuple[float, float, Any]], int]:
        xs = base_grid(left, right)
        sample_xs = xs
        samples = [eval_point(x) for x in sample_xs]
        values = np.asarray([sample[1] for sample in samples], dtype=float)
        finite_count = int(np.isfinite(values).sum())

        if xs.size > 1 and finite_count <= 2:
            sample_xs = staggered_grid(xs)
            samples = [eval_point(x) for x in sample_xs]
            values = np.asarray([sample[1] for sample in samples], dtype=float)
            finite_count = int(np.isfinite(values).sum())

        if xs.size > 1 and finite_count == 0:
            sample_xs = quarter_refined_grid(xs)
            samples = [eval_point(x) for x in sample_xs]
            values = np.asarray([sample[1] for sample in samples], dtype=float)
            finite_count = int(np.isfinite(values).sum())

        sample_xs = np.asarray([sample[0] for sample in samples], dtype=float)
        return sample_xs, samples, finite_count

    intervals: list[tuple[float, float]] = [(float(lo), float(hi))]
    best: ScalarSearchResult | None = None

    for level in range(level_count + 1):
        candidates: list[tuple[float, float, np.ndarray, list[tuple[float, float, Any]], int]] = []

        for left, right in intervals:
            sample_xs, samples, finite_count = sample_interval(left, right)

            for x, value, payload in samples:
                if np.isfinite(value) and (best is None or value > best.value):
                    best = ScalarSearchResult(x=float(x), value=float(value), payload=payload)

            candidates.append((float(left), float(right), sample_xs, samples, int(finite_count)))

        if level == level_count:
            break

        ranked: list[tuple[float, float, np.ndarray, int]] = []
        for _left, _right, sample_xs, samples, finite_count in candidates:
            values = np.asarray([sample[1] for sample in samples], dtype=float)
            xs = np.asarray([sample[0] for sample in samples], dtype=float)

            if not np.isfinite(values).any():
                continue

            for idx in _local_maxima_indices(values):
                value = float(values[idx])
                if np.isfinite(value):
                    ranked.append((value, float(xs[idx]), xs, int(finite_count)))

        if not ranked:
            break

        ranked.sort(reverse=True, key=lambda item: item[0])

        used: list[tuple[float, float]] = []
        new_intervals: list[tuple[float, float]] = []

        for _value, x_star, xs, finite_count in ranked[:top_count]:
            idx = int(np.argmin(np.abs(xs - x_star)))
            span = 2 if finite_count <= 2 else 1

            left_idx = max(0, idx - span)
            right_idx = min(xs.size - 1, idx + span)
            interval = (float(xs[left_idx]), float(xs[right_idx]))

            if interval[0] == interval[1] and xs.size > 1:
                left_idx = max(0, idx - 1)
                right_idx = min(xs.size - 1, idx + 1)
                interval = (float(xs[left_idx]), float(xs[right_idx]))

            if all(
                abs(interval[0] - prev[0]) > 1e-12 or abs(interval[1] - prev[1]) > 1e-12
                for prev in used
            ):
                used.append(interval)
                new_intervals.append(interval)

        if not new_intervals:
            break

        intervals = new_intervals

    return best
'''


def main() -> None:
    if not TARGET.exists():
        raise SystemExit(f"Target file not found: {TARGET}")

    text = TARGET.read_text()

    start = text.find(BEGIN_ANCHOR)
    if start == -1:
        raise RuntimeError(f"Begin anchor not found in {TARGET}:\n{BEGIN_ANCHOR}")

    end = text.find(END_ANCHOR, start)
    if end == -1:
        raise RuntimeError(f"End anchor not found in {TARGET}:\n{END_ANCHOR}")
    end += len(END_ANCHOR)

    shutil.copy2(TARGET, BACKUP)
    patched = text[:start] + REPLACEMENT + text[end:]
    TARGET.write_text(patched)

    print(f"Patched: {TARGET}")
    print(f"Backup written to: {BACKUP}")


if __name__ == "__main__":
    main()
PY
chmod +x tools/apply_scalar_search_rescue_patch.py
